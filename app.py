import os, sys, pickle, uuid, threading, io, json, time, webbrowser
import tempfile
from flask import Flask, request, jsonify, render_template, Response

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split, ParameterSampler
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              ExtraTreesClassifier, AdaBoostClassifier, HistGradientBoostingClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score)
from sklearn.preprocessing import LabelEncoder
from collections import Counter

# exe로 패키징된 경우 리소스 경로를 _MEIPASS(압축 해제 임시폴더)로 재지정
BASE_DIR = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__,
            template_folder=os.path.join(BASE_DIR, 'templates'),
            static_folder=os.path.join(BASE_DIR, 'static'))

# 병렬 작업 수: 저사양 배포 환경(Render 등)에서는 1로 제한해 메모리 초과 방지
# 필요 시 환경변수 N_JOBS로 재정의 가능
N_JOBS = int(os.environ.get('N_JOBS', 1 if os.environ.get('RENDER') else -1))

# 고정 경로 사용: gunicorn 다중 워커에서도 세션/잡 상태를 공유하기 위함
# (tempfile.mkdtemp()는 프로세스마다 다른 폴더를 만들어 워커 간 '세션 없음' 오류 발생)
SESSIONS_DIR = os.path.join(tempfile.gettempdir(), 'ensemble-ml-sessions')
JOBS_DIR     = os.path.join(tempfile.gettempdir(), 'ensemble-ml-jobs')
os.makedirs(SESSIONS_DIR, exist_ok=True)
os.makedirs(JOBS_DIR, exist_ok=True)

STALE_SECONDS = 24 * 3600  # 24시간 지난 세션/잡 파일은 정리

def _cleanup_dir(dirpath):
    """오래된 상태 파일 제거 (디스크 누적 방지)."""
    now = time.time()
    try:
        for fn in os.listdir(dirpath):
            fp = os.path.join(dirpath, fn)
            try:
                if now - os.path.getmtime(fp) > STALE_SECONDS:
                    os.remove(fp)
            except OSError:
                pass
    except OSError:
        pass

# ─────────────────────────────────────────────
# Job state helpers (파일 기반 → 다중 워커 안전)
# ─────────────────────────────────────────────
def _job_path(job_id):
    return os.path.join(JOBS_DIR, f'{job_id}.json')

def save_job(job_id, data):
    tmp = _job_path(job_id) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f)
    os.replace(tmp, _job_path(job_id))  # 원자적 교체 (폴링 중 부분 읽기 방지)

def load_job(job_id):
    path = _job_path(job_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

# ─────────────────────────────────────────────
# Optional imports
# ─────────────────────────────────────────────
try:
    from xgboost import XGBClassifier as _XGB
    _xgb_ok = True
except Exception:
    _xgb_ok = False

try:
    from catboost import CatBoostClassifier as _CBC
    _catboost_ok = True
except Exception:
    _catboost_ok = False

# ─────────────────────────────────────────────
# Session state helpers
# ─────────────────────────────────────────────
def save_state(sid, state):
    path = os.path.join(SESSIONS_DIR, f'{sid}.pkl')
    with open(path, 'wb') as f:
        pickle.dump(state, f)

def load_state(sid):
    path = os.path.join(SESSIONS_DIR, f'{sid}.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

# ─────────────────────────────────────────────
# ML helpers
# ─────────────────────────────────────────────
def _auto_preprocess(df, feature_cols):
    issues = []
    df = df.copy()
    for col in feature_cols:
        if col not in df.columns:
            continue
        if df[col].dtype == object:
            converted = pd.to_numeric(df[col], errors='coerce')
            if converted.notna().sum() > 0.5 * len(df):
                bad = int(df[col].notna().sum() - converted.notna().sum())
                df[col] = converted
                if bad > 0:
                    issues.append(f'[{col}] {bad}개 비숫자 값 → NaN 변환')
        if pd.api.types.is_numeric_dtype(df[col]):
            inf_cnt = int(np.isinf(df[col].replace([None], np.nan).astype(float)).sum()) if df[col].notna().any() else 0
            if inf_cnt > 0:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                issues.append(f'[{col}] {inf_cnt}개 무한대 → NaN 변환')
        na_cnt = int(df[col].isna().sum())
        if na_cnt > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                fv = df[col].median()
                df[col] = df[col].fillna(fv)
                issues.append(f'[{col}] {na_cnt}개 결측값 → 중앙값({fv:.3g}) 보충')
            else:
                fv = df[col].mode().iloc[0] if len(df[col].dropna()) > 0 else 'unknown'
                df[col] = df[col].fillna(fv)
                issues.append(f'[{col}] {na_cnt}개 결측값 → 최빈값("{fv}") 보충')
    return df, issues

def _make_model(name, params=None):
    p = params or {}
    if name == 'Random Forest':
        kw = {k: v for k, v in p.items() if k in ['n_estimators','max_depth','min_samples_leaf','max_leaf_nodes','max_features','criterion','class_weight']}
        return RandomForestClassifier(**kw, random_state=42, n_jobs=N_JOBS)
    elif name == 'Gradient Boosting':
        kw = {k: v for k, v in p.items() if k in ['n_estimators','max_depth','learning_rate','min_samples_leaf','max_leaf_nodes']}
        return GradientBoostingClassifier(**kw, random_state=42)
    elif name == 'Extra Trees':
        kw = {k: v for k, v in p.items() if k in ['n_estimators','max_depth','min_samples_leaf','max_leaf_nodes','max_features','class_weight']}
        return ExtraTreesClassifier(**kw, random_state=42, n_jobs=N_JOBS)
    elif name == 'AdaBoost':
        p2 = dict(p)
        depth = int(p2.pop('base_max_depth', 1))
        kw = {k: v for k, v in p2.items() if k in ['n_estimators','learning_rate']}
        base = DecisionTreeClassifier(max_depth=depth, random_state=42)
        return AdaBoostClassifier(estimator=base, **kw, random_state=42)
    elif name == 'XGBoost':
        if not _xgb_ok:
            raise ValueError('XGBoost 미지원')
        kw = {k: v for k, v in p.items() if k in ['n_estimators','max_depth','learning_rate','min_child_weight','max_leaves']}
        return _XGB(**kw, random_state=42, verbosity=0, eval_metric='logloss', n_jobs=N_JOBS)
    elif name == 'HistGBM (LightGBM계열)':
        kw = {k: v for k, v in p.items() if k in ['max_iter','max_depth','learning_rate','min_samples_leaf','max_leaf_nodes','class_weight']}
        return HistGradientBoostingClassifier(**kw, random_state=42)
    elif name == 'CatBoost':
        if not _catboost_ok:
            raise ValueError('CatBoost 미지원')
        kw = {k: v for k, v in p.items() if k in ['iterations','depth','learning_rate','l2_leaf_reg','auto_class_weights']}
        kw.setdefault('iterations', 100)
        return _CBC(**kw, random_state=42, verbose=0, thread_count=1, task_type='CPU')
    raise ValueError(f'Unknown model: {name}')

CW_MODELS      = {'Random Forest', 'Extra Trees', 'HistGBM (LightGBM계열)', 'CatBoost'}
CW_UNSUPPORTED = {'Gradient Boosting', 'AdaBoost', 'XGBoost'}

def _cross_val_f1(make_model_fn, X, y, cv, sample_weight=None, resample_strategy=None):
    """수동 CV — sklearn fit_params/params API 변경 영향 없음.
    resample_strategy: 'oversample'/'undersample'이면 train fold에만 리샘플링 적용."""
    scores = []
    for tr, val in cv.split(X, y):
        m = make_model_fn()
        X_tr, y_tr_orig = X[tr], y[tr]
        sw_tr = sample_weight[tr] if sample_weight is not None else None

        # 폴드별 리샘플링 — val에는 적용하지 않아 leakage 방지
        if resample_strategy in ('oversample', 'undersample'):
            X_tr, y_tr_orig = _resample(X_tr, y_tr_orig, resample_strategy)
            sw_tr = None  # 리샘플 후 sample_weight 불필요

        # 희소 클래스가 폴드 학습셋에서 빠지면 XGBoost가 라벨 불연속으로 실패하므로
        # 폴드마다 0..k-1로 재인코딩 후 예측을 원래 라벨로 복원
        classes = np.unique(y_tr_orig)
        y_tr    = np.searchsorted(classes, y_tr_orig)

        if sw_tr is not None:
            m.fit(X_tr, y_tr, sample_weight=sw_tr)
        else:
            m.fit(X_tr, y_tr)
        pred_enc = np.asarray(m.predict(X[val])).ravel().astype(int)
        pred     = classes[pred_enc]
        scores.append(f1_score(y[val], pred, average='weighted', zero_division=0))
    return np.array(scores)

def _compute_sample_weight(y):
    counts = Counter(y)
    total  = len(y)
    n_cls  = len(counts)
    w = {c: total / (n_cls * cnt) for c, cnt in counts.items()}
    return np.array([w[yi] for yi in y])

def _resample(X, y, strategy):
    rng = np.random.default_rng(42)
    counts = Counter(y)
    if strategy == 'oversample':
        target_n = max(counts.values())
        Xp, yp = [X.copy()], [y.copy()]
        for cls, cnt in counts.items():
            if cnt < target_n:
                idx   = np.where(y == cls)[0]
                extra = rng.choice(idx, target_n - cnt, replace=True)
                Xp.append(X[extra]); yp.append(y[extra])
        Xr, yr = np.vstack(Xp), np.concatenate(yp)
        shuf = rng.permutation(len(yr))
        return Xr[shuf], yr[shuf]
    elif strategy == 'undersample':
        target_n = min(counts.values())
        Xp, yp  = [], []
        for cls in counts:
            idx    = np.where(y == cls)[0]
            chosen = rng.choice(idx, target_n, replace=False)
            Xp.append(X[chosen]); yp.append(y[chosen])
        Xr, yr = np.vstack(Xp), np.concatenate(yp)
        shuf = rng.permutation(len(yr))
        return Xr[shuf], yr[shuf]
    return X, y

def _make_model_balanced(name, params, strategy):
    p = dict(params)
    if strategy == 'balanced' and name in CW_MODELS:
        if name == 'CatBoost':
            p['auto_class_weights'] = 'Balanced'
        else:
            p['class_weight'] = 'balanced'
    return _make_model(name, p)

class _RelabelWrapper:
    """학습셋 라벨이 0..k-1로 연속되지 않으면(희소 클래스 분할 누락) XGBoost가 실패하므로
    내부적으로 재인코딩해 학습하고 예측 시 원래 라벨로 복원하는 래퍼."""
    def __init__(self, model, n_classes):
        self.model = model
        self.n_classes = int(n_classes)
    def fit(self, X, y, **kw):
        self.classes_ = np.unique(y)
        self.model.fit(X, np.searchsorted(self.classes_, y), **kw)
        return self
    def predict(self, X):
        p = np.asarray(self.model.predict(X)).ravel().astype(int)
        return self.classes_[p]
    def predict_proba(self, X):
        # 재인코딩된 컬럼을 원래 클래스 인덱스 위치로 복원
        # (누락 클래스 확률은 0) — 컬럼-클래스 불일치 방지
        p   = np.asarray(self.model.predict_proba(X))
        out = np.zeros((p.shape[0], self.n_classes), dtype=float)
        out[:, self.classes_.astype(int)] = p
        return out
    @property
    def feature_importances_(self):
        return self.model.feature_importances_

def _fit_model(model, name, X_tr, y_tr, strategy, n_classes=None):
    n_all = int(n_classes) if n_classes is not None else int(np.max(y_tr)) + 1
    if len(np.unique(y_tr)) != n_all:
        model = _RelabelWrapper(model, n_all)
    if strategy in ('oversample', 'undersample'):
        X_tr, y_tr = _resample(X_tr, y_tr, strategy)
        model.fit(X_tr, y_tr)
    elif strategy == 'balanced':
        if name in CW_UNSUPPORTED:
            model.fit(X_tr, y_tr, sample_weight=_compute_sample_weight(y_tr))
        else:
            model.fit(X_tr, y_tr)
    else:
        model.fit(X_tr, y_tr)
    return model

# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route('/')
def index():
    # send_from_directory('templates', ...)는 실행 위치(cwd)에 의존 → render_template 사용
    return render_template('index.html')

@app.route('/api/load_data', methods=['POST'])
def api_load_data():
    body = request.get_json(silent=True)
    if not body:
        return jsonify({'error': '요청 본문이 없습니다'}), 400
    sid              = body.get('session_id') or str(uuid.uuid4())
    csv_str          = body.get('csv', '')
    target_col       = body.get('target', '')
    if not csv_str or not target_col:
        return jsonify({'error': 'csv 또는 target 누락'}), 400
    selected_features = body.get('features', [])
    balance_strategy  = body.get('balance', 'none')
    test_size         = float(body.get('test_size', 0.2))

    try:
        df = pd.read_csv(io.StringIO(csv_str))
        df.columns = [c.strip() for c in df.columns]
        feature_names = [c for c in (selected_features or df.columns.tolist()) if c != target_col and c in df.columns]
        df, prep_issues = _auto_preprocess(df, feature_names)

        encoders = {}
        for col in feature_names:
            if col not in df.columns:
                continue
            try:
                df[col].astype(float)
            except (ValueError, TypeError):
                enc = LabelEncoder()
                df[col] = enc.fit_transform(df[col].fillna('missing').astype(str))
                encoders[col] = enc

        X_safe = df[feature_names].copy()
        for col in X_safe.columns:
            try:
                X_safe[col] = X_safe[col].astype(float)
            except (ValueError, TypeError):
                X_safe[col] = LabelEncoder().fit_transform(
                    X_safe[col].fillna('missing').astype(str)
                ).astype(float)

        X  = X_safe.fillna(0).values.astype(float)
        le = LabelEncoder()
        y  = le.fit_transform(df[target_col].astype(str))
        is_binary = len(le.classes_) == 2
        dist = {str(le.inverse_transform([k])[0]): int(v) for k, v in Counter(y).items()}

        state = {
            'X': X, 'y': y, 'le': le, 'encoders': encoders,
            'feature_names': feature_names, 'target_name': target_col,
            'is_binary': is_binary, 'balance_strategy': balance_strategy,
            'test_size': test_size, 'model': None
        }
        save_state(sid, state)

        return jsonify({
            'n_samples': int(len(df)), 'n_features': int(len(feature_names)),
            'classes': le.classes_.tolist(), 'is_binary': bool(is_binary),
            'features': feature_names, 'class_dist': dist,
            'balance_strategy': balance_strategy, 'prep_issues': prep_issues,
            'session_id': sid
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

def _evaluate_one_model(state, model_name, cv_folds):
    """단일 모델 교차검증 평가. evaluate_single(동기)과 evaluate/start(백그라운드)가 공유."""
    X, y = state['X'], state['y']
    balance_strategy = state['balance_strategy']
    cv   = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    resample    = balance_strategy if balance_strategy in ('oversample', 'undersample') else None
    strat_model = 'balanced' if balance_strategy == 'balanced' else 'none'
    use_sw = strat_model == 'balanced' and model_name in CW_UNSUPPORTED
    sw     = _compute_sample_weight(y) if use_sw else None

    try:
        scores = _cross_val_f1(
            lambda: _make_model_balanced(model_name, {}, strat_model),
            X, y, cv, sample_weight=sw, resample_strategy=resample
        )
        return {'name': model_name, 'f1': float(scores.mean()), 'std': float(scores.std()), 'ok': True, 'cv_folds': cv_folds}
    except Exception as e:
        if sw is None:
            return {'name': model_name, 'f1': 0.0, 'std': 0.0, 'ok': False, 'err': str(e)}
        try:
            scores2 = _cross_val_f1(
                lambda: _make_model_balanced(model_name, {}, strat_model),
                X, y, cv, resample_strategy=resample
            )
            return {'name': model_name, 'f1': float(scores2.mean()), 'std': float(scores2.std()), 'ok': True, 'cv_folds': cv_folds}
        except Exception as e2:
            return {'name': model_name, 'f1': 0.0, 'std': 0.0, 'ok': False, 'err': str(e2)}

@app.route('/api/evaluate_single', methods=['POST'])
def api_evaluate_single():
    body = request.get_json(silent=True)
    if not body:
        return jsonify({'ok': False, 'err': '요청 본문 없음'}), 400
    sid        = body.get('session_id')
    model_name = body.get('model')
    if not model_name:
        return jsonify({'ok': False, 'err': 'model 누락'}), 400
    cv_folds   = int(body.get('cv_folds', 5))

    state = load_state(sid)
    if not state:
        return jsonify({'ok': False, 'err': '세션 없음'}), 400

    return jsonify(_evaluate_one_model(state, model_name, cv_folds))

MODEL_LIST = ['Random Forest', 'Gradient Boosting', 'Extra Trees', 'AdaBoost',
              'XGBoost', 'HistGBM (LightGBM계열)', 'CatBoost']

def _run_evaluate_all(job_id, sid, cv_folds):
    """전 모델 순차 평가를 백그라운드 스레드에서 실행.
    HTTP 요청-응답 안에서 느린 모델(RF/GB/ET/AdaBoost)까지 다 기다리면
    프록시/게이트웨이(Render 등)의 타임아웃(보통 30~60초)에 걸려 500이 나므로,
    튜닝(/api/tune/*)과 동일하게 job 방식으로 분리해 타임아웃 설정과 무관하게 만든다."""
    state = load_state(sid)
    if not state:
        save_job(job_id, {'progress': 0, 'done': True, 'error': '세션 없음', 'results': []})
        return

    job = {'progress': 0, 'done': False, 'error': None, 'results': []}
    save_job(job_id, job)
    try:
        for i, model_name in enumerate(MODEL_LIST):
            job['current'] = model_name
            save_job(job_id, job)
            r = _evaluate_one_model(state, model_name, cv_folds)
            job['results'].append(r)
            job['progress'] = int((i + 1) / len(MODEL_LIST) * 100)
            save_job(job_id, job)
        job['done'] = True
        job['current'] = None
        save_job(job_id, job)
    except Exception as e:
        job['done'] = True
        job['error'] = str(e)
        save_job(job_id, job)
    finally:
        _cleanup_dir(JOBS_DIR)

@app.route('/api/evaluate/start', methods=['POST'])
def api_evaluate_start():
    body = request.get_json(silent=True)
    if not body:
        return jsonify({'error': '요청 본문 없음'}), 400
    sid      = body.get('session_id')
    cv_folds = int(body.get('cv_folds', 5))
    if not load_state(sid):
        return jsonify({'error': '세션 없음'}), 400

    job_id = str(uuid.uuid4())
    save_job(job_id, {'progress': 0, 'done': False, 'error': None, 'results': []})
    t = threading.Thread(target=_run_evaluate_all, args=(job_id, sid, cv_folds), daemon=True)
    t.start()
    return jsonify({'job_id': job_id})

@app.route('/api/evaluate/status', methods=['GET'])
def api_evaluate_status():
    job_id = request.args.get('job_id')
    job    = load_job(job_id) if job_id else None
    if not job:
        return jsonify({'error': '작업 없음'}), 404
    return jsonify(job)

@app.route('/api/train_and_eval', methods=['POST'])
def api_train_and_eval():
    body = request.get_json(silent=True)
    if not body:
        return jsonify({'error': '요청 본문 없음'}), 400
    sid        = body.get('session_id')
    model_name = body.get('model')
    if not model_name:
        return jsonify({'error': 'model 누락'}), 400
    params     = body.get('params', {})
    threshold  = float(body.get('threshold', 0.5))

    state = load_state(sid)
    if not state:
        return jsonify({'error': '세션 없음'}), 400

    X, y             = state['X'], state['y']
    le               = state['le']
    feature_names    = state['feature_names']
    is_binary        = state['is_binary']
    balance_strategy = state['balance_strategy']
    test_size        = state['test_size']

    try:
        # 표본 1개짜리 클래스가 있으면 층화 분할이 불가능하므로 일반 분할로 대체
        stratify_arg = y if min(Counter(y).values()) >= 2 else None
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=42, stratify=stratify_arg)
        strat = balance_strategy
        m = _make_model_balanced(model_name, params, 'balanced' if strat == 'balanced' else 'none')
        m = _fit_model(m, model_name, X_tr, y_tr, strat, n_classes=len(le.classes_))

        if is_binary and hasattr(m, 'predict_proba'):
            y_pred = (m.predict_proba(X_te)[:, 1] >= threshold).astype(int)
        else:
            y_pred = m.predict(X_te)

        avg = 'binary' if is_binary else 'weighted'
        cm  = confusion_matrix(y_te, y_pred, labels=list(range(len(le.classes_))))
        res = {
            'accuracy':  float(accuracy_score(y_te, y_pred)),
            'precision': float(precision_score(y_te, y_pred, average=avg, zero_division=0)),
            'recall':    float(recall_score(y_te, y_pred, average=avg, zero_division=0)),
            'f1':        float(f1_score(y_te, y_pred, average=avg, zero_division=0)),
            'confusion_matrix': cm.tolist(),
            'classes':   le.classes_.tolist(),
            'balance_strategy': strat
        }
        if is_binary and hasattr(m, 'predict_proba'):
            try:
                res['auc'] = float(roc_auc_score(y_te, m.predict_proba(X_te)[:, 1]))
            except Exception:
                pass
        if is_binary:
            res['per_class'] = {
                str(le.classes_[i]): {
                    'precision': float(precision_score(y_te, y_pred, pos_label=i, average='binary', zero_division=0)),
                    'recall':    float(recall_score(y_te, y_pred, pos_label=i, average='binary', zero_division=0)),
                    'f1':        float(f1_score(y_te, y_pred, pos_label=i, average='binary', zero_division=0))
                } for i in range(len(le.classes_))
            }
        if hasattr(m, 'feature_importances_'):
            fi = sorted(zip(feature_names, m.feature_importances_.tolist()), key=lambda x: -x[1])[:10]
            res['feature_importance'] = fi

        state['model']     = m
        state['threshold'] = threshold   # 예측 시 재사용
        save_state(sid, state)
        return jsonify(res)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

def _run_tune(job_id, sid, model_name, balance_strategy, cv_folds):
    """Background tuning job."""
    job = {'progress': 0, 'done': False, 'result': None, 'error': None}
    state = load_state(sid)
    if not state:
        save_job(job_id, {'progress': 0, 'done': True, 'error': '세션 없음', 'result': None})
        return

    X, y      = state['X'], state['y']
    is_binary = state['is_binary']

    _GRIDS = {
        'Random Forest':        {'n_estimators':[50,100,200,300],'max_depth':[5,8,12,18,None],'min_samples_leaf':[1,3,5,10],'max_leaf_nodes':[20,50,100,None],'max_features':[0.3,0.5,0.7,0.8],'criterion':['gini','entropy']},
        'Gradient Boosting':    {'n_estimators':[50,100,200],'max_depth':[3,4,5,7],'learning_rate':[0.03,0.05,0.1,0.2],'min_samples_leaf':[1,3,5,10],'max_leaf_nodes':[10,31,50,None]},
        'Extra Trees':          {'n_estimators':[50,100,200,300],'max_depth':[5,10,15,None],'min_samples_leaf':[1,3,5,10],'max_leaf_nodes':[20,50,100,None],'max_features':[0.3,0.5,0.7,0.8]},
        'AdaBoost':             {'n_estimators':[50,100,200],'learning_rate':[0.5,1.0,1.5,2.0],'base_max_depth':[1,2,3]},
        'XGBoost':              {'n_estimators':[50,100,200],'max_depth':[3,4,5,6,8],'learning_rate':[0.03,0.05,0.1,0.2],'min_child_weight':[1,3,5,10],'max_leaves':[0,16,31,63]},
        'HistGBM (LightGBM계열)': {'max_iter':[50,100,200],'max_depth':[3,5,7,None],'learning_rate':[0.03,0.05,0.1,0.2],'min_samples_leaf':[10,20,30,50],'max_leaf_nodes':[15,31,63,127]},
        'CatBoost':             {'iterations':[50,100,150],'depth':[4,5,6],'learning_rate':[0.03,0.05,0.1,0.15],'l2_leaf_reg':[1,3,5]},
    }

    grid = _GRIDS.get(model_name)
    if not grid:
        save_job(job_id, {'progress': 0, 'done': True, 'error': f'{model_name}: 지원하지 않는 모델', 'result': None})
        return

    try:
        # oversample/undersample은 폴드별 리샘플링으로 처리 (evaluate_single과 동일 방식)
        resample    = balance_strategy if balance_strategy in ('oversample', 'undersample') else None
        strat_model = 'balanced' if balance_strategy == 'balanced' else 'none'
        cv         = StratifiedKFold(n_splits=int(cv_folds), shuffle=True, random_state=42)
        n_iter1, n_iter2 = 10, 8
        param_list = list(ParameterSampler(grid, n_iter=n_iter1, random_state=42))
        best_cv    = -1
        best_params = None
        best_estimator = None

        use_sw = strat_model == 'balanced' and model_name in CW_UNSUPPORTED
        sw     = _compute_sample_weight(y) if use_sw else None

        # Phase 1: 10 trials (0→75%)
        for i, params in enumerate(param_list):
            p = dict(params)
            try:
                sc = _cross_val_f1(
                    lambda _p=p: _make_model_balanced(model_name, _p, strat_model),
                    X, y, cv, sample_weight=sw, resample_strategy=resample
                )
                s = float(sc.mean())
            except Exception:
                s = 0.0
            if s > best_cv:
                best_cv     = s
                best_params = p
            job['progress'] = int((i + 1) / n_iter1 * 75)
            save_job(job_id, job)

        # Phase 2: anti-overfit refinement (75→95%)
        if best_params is None:
            save_job(job_id, {'progress': 0, 'done': True, 'result': None, 'error': '모든 파라미터 탐색 실패'})
            return

        m_best = _make_model_balanced(model_name, dict(best_params), strat_model)
        m_best = _fit_model(m_best, model_name, X, y, balance_strategy)
        train_score    = float(f1_score(y, m_best.predict(X), average='weighted', zero_division=0))
        gap            = train_score - best_cv
        best_estimator = m_best

        if gap > 0.15:
            tight = {k: v for k, v in grid.items()}
            if 'max_depth' in tight:
                tight['max_depth'] = [d for d in tight['max_depth'] if d is None or d <= 7] or [5]
            if 'min_samples_leaf' in tight:
                tight['min_samples_leaf'] = [v for v in tight['min_samples_leaf'] if v >= 3] or [5]
            tight_list = list(ParameterSampler(tight, n_iter=n_iter2, random_state=0))
            best_s2, best_params2 = -1, None
            for j, params2 in enumerate(tight_list):
                p2 = dict(params2)
                try:
                    sc2 = _cross_val_f1(
                        lambda _p=p2: _make_model_balanced(model_name, _p, strat_model),
                        X, y, cv, sample_weight=sw, resample_strategy=resample
                    )
                    s2 = float(sc2.mean())
                except Exception:
                    s2 = 0.0
                if s2 >= best_cv * 0.95 and s2 > best_cv - 0.05 and s2 > best_s2:
                    best_s2     = s2
                    best_params2 = dict(params2)
                job['progress'] = 75 + int((j + 1) / n_iter2 * 20)
                save_job(job_id, job)
            if best_params2 is not None:
                best_cv     = best_s2
                best_params = best_params2

        job['progress'] = 98
        save_job(job_id, job)

        # Clean params
        clean = {}
        for k, v in best_params.items():
            if v is None:                  clean[k] = None
            elif isinstance(v, np.integer): clean[k] = int(v)
            elif isinstance(v, np.floating): clean[k] = float(v)
            else:                           clean[k] = v

        # Optimal threshold (binary) — holdout split으로 data leakage 방지
        best_thr = None
        if is_binary and hasattr(best_estimator, 'predict_proba'):
            try:
                stratify_thr = y if min(Counter(y).values()) >= 2 else None
                X_thr_tr, X_thr_val, y_thr_tr, y_thr_val = train_test_split(
                    X, y, test_size=0.2, random_state=99, stratify=stratify_thr
                )
                # (버그 수정) 기존 코드는 미정의 변수 strat 참조 → NameError가
                # except에 삼켜져 최적 threshold가 절대 반환되지 않았음
                m_thr = _make_model_balanced(model_name, dict(best_params), strat_model)
                m_thr = _fit_model(m_thr, model_name, X_thr_tr, y_thr_tr,
                                   balance_strategy, n_classes=len(np.unique(y)))
                proba   = m_thr.predict_proba(X_thr_val)[:, 1]
                best_f1 = 0.0
                for thr in np.arange(0.1, 0.91, 0.01):
                    preds = (proba >= thr).astype(int)
                    f     = float(f1_score(y_thr_val, preds, zero_division=0))
                    if f > best_f1:
                        best_f1  = f
                        best_thr = round(float(thr), 2)
            except Exception:
                pass  # threshold 탐색 실패 시 기본값(0.5) 사용

        result = {'params': clean, 'cv_score': best_cv, 'train_score': train_score, 'gap': gap}
        if best_thr is not None:
            result['threshold'] = best_thr

        save_job(job_id, {'progress': 100, 'done': True, 'result': result, 'error': None})
    except Exception as e:
        save_job(job_id, {'progress': 0, 'done': True, 'result': None, 'error': str(e)})
    finally:
        _cleanup_dir(JOBS_DIR)
        _cleanup_dir(SESSIONS_DIR)

@app.route('/api/tune/start', methods=['POST'])
def api_tune_start():
    body = request.get_json(silent=True)
    if not body:
        return jsonify({'error': '요청 본문 없음'}), 400
    sid              = body.get('session_id')
    model_name       = body.get('model')
    if not model_name:
        return jsonify({'error': 'model 누락'}), 400
    balance_strategy = body.get('balance', 'none')
    cv_folds         = int(body.get('cv_folds', 5))

    job_id = str(uuid.uuid4())
    save_job(job_id, {'progress': 0, 'done': False, 'result': None, 'error': None})

    t = threading.Thread(target=_run_tune, args=(job_id, sid, model_name, balance_strategy, cv_folds), daemon=True)
    t.start()
    return jsonify({'job_id': job_id})

@app.route('/api/tune/status', methods=['GET'])
def api_tune_status():
    job_id = request.args.get('job_id')
    job    = load_job(job_id) if job_id else None
    if not job:
        return jsonify({'error': '작업 없음'}), 404
    return jsonify(job)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    body = request.get_json(silent=True)
    if not body:
        return jsonify({'error': '요청 본문 없음'}), 400
    sid     = body.get('session_id')
    csv_str = body.get('csv')
    if not csv_str:
        return jsonify({'error': 'csv 누락'}), 400

    state = load_state(sid)
    if not state:
        return jsonify({'error': '세션 없음'}), 400

    model = state.get('model')
    if model is None:
        return jsonify({'error': '먼저 모델을 학습해주세요'}), 400

    feature_names = state['feature_names']
    le            = state['le']
    encoders      = state['encoders']
    is_binary     = state['is_binary']
    threshold     = state.get('threshold', 0.5)  # 학습 시 저장된 threshold 사용

    try:
        df_n = pd.read_csv(io.StringIO(csv_str))
        df_n.columns = [c.strip() for c in df_n.columns]
        df_p, _ = _auto_preprocess(df_n, feature_names)

        for col, enc in encoders.items():
            if col in df_p.columns:
                df_p[col] = df_p[col].apply(
                    lambda x: int(enc.transform([str(x)])[0]) if str(x) in enc.classes_ else 0
                )

        missing = [f for f in feature_names if f not in df_p.columns]
        if missing:
            return jsonify({'error': f'누락 피처: {missing}'}), 400

        X_df = pd.DataFrame(index=df_p.index)
        for f in feature_names:
            col = df_p[f].copy()
            if col.dtype == object:
                col = pd.to_numeric(col, errors='coerce')
            if col.isna().all():
                col = pd.Series(0, index=df_p.index)
            X_df[f] = col
        X_n = X_df.fillna(0).values.astype(float)

        if is_binary and hasattr(model, 'predict_proba'):
            raw_preds = (model.predict_proba(X_n)[:, 1] >= threshold).astype(int)
        else:
            raw_preds = model.predict(X_n)
        preds = le.inverse_transform(raw_preds).tolist()
        r         = df_n.copy()
        r['예측결과'] = preds

        if hasattr(model, 'predict_proba'):
            proba   = model.predict_proba(X_n)
            classes = le.classes_.tolist()
            for i, cls in enumerate(classes):
                r[f'P({cls})'] = np.round(proba[:, i], 4)
            if is_binary:
                r['확률(양성)'] = [f'{v:.2f}%' for v in proba[:, 1] * 100]

        return Response(r.to_csv(index=False), mimetype='text/csv')
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    # 로컬 실행 시 1.5초 후 브라우저 자동 오픈 (Render 등 서버 환경 제외)
    if not os.environ.get('RENDER'):
        threading.Timer(1.5, lambda: webbrowser.open(f'http://localhost:{port}')).start()
    app.run(host='0.0.0.0', port=port, debug=False)
