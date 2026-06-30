import os, pickle, uuid, threading, io
import tempfile
from flask import Flask, request, jsonify, send_from_directory, Response

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split, ParameterSampler
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, AdaBoostClassifier, HistGradientBoostingClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score)
from sklearn.preprocessing import LabelEncoder
from collections import Counter

app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = os.environ.get('SECRET_KEY', 'ensemble-ml-2024-secret')

SESSIONS_DIR = tempfile.mkdtemp()
JOBS = {}  # job_id -> {'progress': int, 'done': bool, 'result': dict, 'error': str}

def _cleanup_jobs():
    """ìë£ë jobì´ 200ê° ì´ê³¼ ì ì¤ëë ê² ì ê±°."""
    done_ids = [k for k, v in JOBS.items() if v.get('done')]
    if len(done_ids) > 200:
        for k in done_ids[:100]:
            JOBS.pop(k, None)

# âââââââââââââââââââââââââââââââââââââââââââââ
# Optional imports
# âââââââââââââââââââââââââââââââââââââââââââââ
try:
    from xgboost import XGBClassifier as _XGB
    _xgb_ok = True
except:
    _xgb_ok = False

try:
    from catboost import CatBoostClassifier as _CBC
    _catboost_ok = True
except:
    _catboost_ok = False

# âââââââââââââââââââââââââââââââââââââââââââââ
# Session state helpers
# âââââââââââââââââââââââââââââââââââââââââââââ
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

# âââââââââââââââââââââââââââââââââââââââââââââ
# ML helpers
# âââââââââââââââââââââââââââââââââââââââââââââ
def _auto_preprocess(df, target_col, feature_cols):
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
                    issues.append(f'[{col}] {bad}ê° ë¹ì«ì ê° â NaN ë³í')
        if pd.api.types.is_numeric_dtype(df[col]):
            inf_cnt = int(np.isinf(df[col].replace([None], np.nan).astype(float)).sum()) if df[col].notna().any() else 0
            if inf_cnt > 0:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                issues.append(f'[{col}] {inf_cnt}ê° ë¬´íë â NaN ë³í')
        na_cnt = int(df[col].isna().sum())
        if na_cnt > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                fv = df[col].median()
                df[col] = df[col].fillna(fv)
                issues.append(f'[{col}] {na_cnt}ê° ê²°ì¸¡ê° â ì¤ìê°({fv:.3g}) ë³´ì¶©')
            else:
                fv = df[col].mode().iloc[0] if len(df[col].dropna()) > 0 else 'unknown'
                df[col] = df[col].fillna(fv)
                issues.append(f'[{col}] {na_cnt}ê° ê²°ì¸¡ê° â ìµë¹ê°("{fv}") ë³´ì¶©')
    return df, issues

def _make_model(name, params=None):
    p = params or {}
    if name == 'Random Forest':
        kw = {k: v for k, v in p.items() if k in ['n_estimators','max_depth','min_samples_leaf','max_leaf_nodes','max_features','criterion']}
        return RandomForestClassifier(**kw, random_state=42, n_jobs=-1)
    elif name == 'Gradient Boosting':
        kw = {k: v for k, v in p.items() if k in ['n_estimators','max_depth','learning_rate','min_samples_leaf','max_leaf_nodes']}
        return GradientBoostingClassifier(**kw, random_state=42)
    elif name == 'Extra Trees':
        kw = {k: v for k, v in p.items() if k in ['n_estimators','max_depth','min_samples_leaf','max_leaf_nodes','max_features']}
        return ExtraTreesClassifier(**kw, random_state=42, n_jobs=-1)
    elif name == 'AdaBoost':
        p2 = dict(p)
        depth = int(p2.pop('base_max_depth', 1))
        kw = {k: v for k, v in p2.items() if k in ['n_estimators','learning_rate']}
        base = DecisionTreeClassifier(max_depth=depth, random_state=42)
        return AdaBoostClassifier(estimator=base, **kw, random_state=42)
    elif name == 'XGBoost':
        if not _xgb_ok:
            raise ValueError('XGBoost ë¯¸ì§ì')
        kw = {k: v for k, v in p.items() if k in ['n_estimators','max_depth','learning_rate','min_child_weight','max_leaves']}
        return _XGB(**kw, random_state=42, verbosity=0, eval_metric='logloss', n_jobs=-1)
    elif name == 'HistGBM (LightGBMê³ì´)':
        kw = {k: v for k, v in p.items() if k in ['max_iter','max_depth','learning_rate','min_samples_leaf','max_leaf_nodes']}
        return HistGradientBoostingClassifier(**kw, random_state=42)
    elif name == 'CatBoost':
        if not _catboost_ok:
            raise ValueError('CatBoost ë¯¸ì§ì')
        kw = {k: v for k, v in p.items() if k in ['iterations','depth','learning_rate','l2_leaf_reg']}
        kw.setdefault('iterations', 100)
        return _CBC(**kw, random_state=42, verbose=0, thread_count=1, task_type='CPU')
    raise ValueError f'Unknown model: {name}')

CW_MODELS = {'Random Forest', 'Extra Trees', 'HistGBM (LightGBMê³ì´)', 'CatBoost'}
CW_UNSUPPORTED = {'Gradient Boosting', 'AdaBoost', 'XGBoost'}

def _compute_sample_weight(y):
    counts = Counter(y)
    total = len(y)
    n_cls = len(counts)
    w = {c: total / (n_cls * cnt) for c, cnt in counts.items()}
    return np.array([w[yi] for yi in y])

def _resample(X, y, strategy):
    np.random.seed(42)
    counts = Counter(y)
    if strategy == 'oversample':
        target_n = max(counts.values())
        Xp, yp = [X.copy()], [y.copy()]
        for cls, cnt in counts.items():
            if cnt < target_n:
                idx = np.where(y == cls)[0]
                extra = np.random.choice(idx, target_n - cnt, replace=True)
                Xp.append(X[extra]); yp.append(y[extra])
        Xr, yr = np.vstack(Xp), np.concatenate(yp)
        shuf = np.random.permutation(len(yr))
        return Xr[shuf], yr[shuf]
    elif strategy == 'undersample':
        target_n = min(counts.values())
        Xp, yp = [], []
        for cls in counts:
            idx = np.where(y == cls)[0]
            chosen = np.random.choice(idx, target_n, replace=False)
            Xp.append(X[chosen]); yp.append(y[chosen])
        Xr, yr = np.vstack(Xp), np.concatenate(yp)
        shuf = np.random.permutation(len(yr))
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

def _fit_model(model, name, X_tr, y_tr, strategy):
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

# âââââââââââââââââââââââââââââââââââââââââââââ
# Routes
# âââââââââââââââââââââââââââââââââââââââââââââ
@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

@app.route('/api/load_data', methods=['POST'])
def api_load_data():
    body = request.get_json(silent=True)
    if not body:
        return jsonify({'error': 'ìì²­ ë³¸ë¬¸ì´ ììµëë¤'}), 400
    sid = body.get('session_id', str(uuid.uuid4()))
    csv_str = body.get('csv', '')
    target_col = body.get('target', '')
    if not csv_str or not target_col:
        return jsonify({'error': 'csv ëë target ëë½'}), 400
    selected_features = body.get('features', [])
    balance_strategy = body.get('balance', 'none')
    test_size = float(body.get('test_size', 0.2))

    try:
        df = pd.read_csv(io.StringIO(csv_str))
        df.columns = [c.strip() for c in df.columns]  # ì»¬ë¼ëª ê³µë°± ì ê±°
        feature_names = [c for c in (selected_features or df.columns.tolist()) if c != target_col and c in df.columns]
        df, prep_issues = _auto_preprocess(df, target_col, feature_names)

        encoders = {}
        for col in feature_names:
            if col not in df.columns:
                continue
            # float ë³í ìë â ì¤í¨íë©´ LabelEncoder ì ì©
            try:
                df[col].astype(float)
            except (ValueError, TypeError):
                enc = LabelEncoder()
                df[col] = enc.fit_transform(df[col].fillna('missing').astype(str))
                encoders[col] = enc

        # ìµì¢ ìì ì¥ì¹: ì»¬ë¼ë³ë¡ float ë³í ê°ë¥íê² ê°ì  ì²ë¦¬
        X_safe = df[feature_names].copy()
        for col in X_safe.columns:
            try:
                X_safe[col] = X_safe[col].astype(float)
            except (ValueError, TypeError):
                X_safe[col] = LabelEncoder().fit_transform(
                    X_safe[col].fillna('missing').astype(str)
                ).astype(float)

        X = X_safe.fillna(0).values.astype(float)
        le = LabelEncoder()
        y = le.fit_transform(df[target_col].astype(str))
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


@app.route('/api/evaluate_single', methods=['POST'])
def api_evaluate_single():
    body = request.get_json(silent=True)
    if not body:
        return jsonify({'ok': False, 'err': 'ìì²­ ë³¸ë¬¸ ìì'}), 400
    sid = body.get('session_id')
    model_name = body['model']
    cv_folds = int(body.get('cv_folds', 5))

    state = load_state(sid)
    if not state:
        return jsonify({'ok': False, 'err': 'ì¸ì ìì'}), 400

    X, y = state['X'], state['y']
    balance_strategy = state['balance_strategy']
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    strat = 'balanced' if balance_strategy in ('balanced','oversample','undersample') else 'none'

    try:
        m = _make_model_balanced(model_name, {}, strat)
        fit_p = {}
        if strat == 'balanced' and model_name in CW_UNSUPPORTED:
            fit_p = {'sample_weight': _compute_sample_weight(y)}
        try:
            scores = cross_val_score(m, X, y, cv=cv, scoring='f1_weighted', error_score=0,
                                     **({"fit_params": fit_p} if fit_p else {}))
        except TypeError:
            scores = cross_val_score(m, X, y, cv=cv, scoring='f1_weighted', error_score=0)
        return jsonify({'name': model_name, 'f1': float(scores.mean()), 'std': float(scores.std()), 'ok': True, 'cv_folds': cv_folds})
    except Exception as e:
        try:
            m2 = _make_model_balanced(model_name, {}, strat)
            scores2 = cross_val_score(m2, X, y, cv=cv, scoring='f1_weighted', error_score=0)
            return jsonify({'name': model_name, 'f1': float(scores2.mean()), 'std': float(scores2.std()), 'ok': True, 'cv_folds': cv_folds})
        except Exception as e2:
            return jsonify({'name': model_name, 'f1': 0.0, 'std': 0.0, 'ok': False, 'err': str(e2)})


@app.route('/api/train_and_eval', methods=['POST'])
def api_train_and_eval():
    body = request.get_json(silent=True)
    if not body:
        return jsonify({'error': 'ìì²­ ë³¸ë¬¸ ìì'}), 400
    sid = body.get('session_id')
    model_name = body['model']
    params = body.get('params', {})
    threshold = float(body.get('threshold', 0.5))

    state = load_state(sid)
    if not state:
        return jsonify({'error': 'ì¸ì ìì'}), 400

    X, y = state['X'], state['y']
    le = state['le']
    feature_names = state['feature_names']
    is_binary = state['is_binary']
    balance_strategy = state['balance_strategy']
    test_size = state['test_size']

    try:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        strat = balance_strategy
        m = _make_model_balanced(model_name, params, 'balanced' if strat == 'balanced' else 'none')
        _fit_model(m, model_name, X_tr, y_tr, strat)

        if is_binary and hasattr(m, 'predict_proba'):
            y_pred = (m.predict_proba(X_te)[:, 1] >= threshold).astype(int)
        else:
            y_pred = m.predict(X_te)

        avg = 'binary' if is_binary else 'weighted'
        cm = confusion_matrix(y_te, y_pred)
        res = {
            'accuracy': float(accuracy_score(y_te, y_pred)),
            'precision': float(precision_score(y_te, y_pred, average=avg, zero_division=0)),
            'recall': float(recall_score(y_te, y_pred, average=avg, zero_division=0)),
            'f1': float(f1_score(y_te, y_pred, average=avg, zero_division=0)),
            'confusion_matrix': cm.tolist(),
            'classes': le.classes_.tolist(),
            'balance_strategy': strat
        }
        if is_binary and hasattr(m, 'predict_proba'):
            try:
                res['auc'] = float(roc_auc_score(y_te, m.predict_proba(X_te)[:, 1]))
            except:
                pass
        if is_binary:
            res['per_class'] = {
                str(le.classes_[i]): {
                    'precision': float(precision_score(y_te, y_pred, pos_label=i, average='binary', zero_division=0)),
                    'recall': float(recall_score(y_te, y_pred, pos_label=i, average='binary', zero_division=0)),
                    'f1': float(f1_score(y_te, y_pred, pos_label=i, average='binary', zero_division=0))
                } for i in range(len(le.classes_))
            }
        if hasattr(m, 'feature_importances_'):
            fi = sorted(zip(feature_names, m.feature_importances_.tolist()), key=lambda x: -x[1])[:10]
            res['feature_importance'] = fi

        # Save trained model to state
        state['model'] = m
        save_state(sid, state)

        return jsonify(res)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


def _run_tune(job_id, sid, model_name, balance_strategy, cv_folds):
    """Background tuning job."""
    state = load_state(sid)
    if not state:
        JOBS[job_id] = {'progress': 0, 'done': True, 'error': 'ì¸ì ìì', 'result': None}
        return

    X, y = state['X'], state['y']
    is_binary = state['is_binary']

    _GRIDS = {
        'Random Forest': {'n_estimators':[50,100,200,300],'max_depth':[5,8,12,18,None],'min_samples_leaf':[1,3,5,10],'max_leaf_nodes':[20,50,100,None],'max_features':[0.3,0.5,0.7,0.8],'criterion':['gini','entropy']},
        'Gradient Boosting': {'n_estimators':[50,100,200],'max_depth':[3,4,5,7],'learning_rate':[0.03,0.05,0.1,0.2],'min_samples_leaf':[1,3,5,10],'max_leaf_nodes':[10,31,50,None]},
        'Extra Trees': {'n_estimators':[50,100,200,300],'max_depth':[5,10,15,None],'min_samples_leaf':[1,3,5,10],'max_leaf_nodes':[20,50,100,None],'max_features':[0.3,0.5,0.7,0.8]},
        'AdaBoost': {'n_estimators':[50,100,200],'learning_rate':[0.5,1.0,1.5,2.0],'base_max_depth':[1,2,3]},
        'XGBoost': {'n_estimators':[50,100,200],'max_depth':[3,4,5,6,8],'learning_rate':[0.03,0.05,0.1,0.2],'min_child_weight':[1,3,5,10],'max_leaves':[0,16,31,63]},
        'HistGBM (LightGBMê³ì´)': {'max_iter':[50,100,200],'max_depth':[3,5,7,None],'learning_rate':[0.03,0.05,0.1,0.2],'min_samples_leaf':[10,20,30,50],'max_leaf_nodes':[15,31,63,127]},
        'CatBoost': {'iterations':[50,100,150],'depth':[4,5,6],'learning_rate':[0.03,0.05,0.1,0.15],'l2_leaf_reg':[1,3,5]},
    }

    grid = _GRIDS.get(model_name)
    if not grid:
        JOBS[job_id] = {'progress': 0, 'done': True, 'error': f'{model_name}: ì§ìíì§ ìë ëª¨ë¸', 'result': None}
        return

    try:
        strat = 'balanced' if balance_strategy in ('balanced','oversample','undersample') else 'none'
        cv = StratifiedKFold(n_splits=int(cv_folds), shuffle=True, random_state=42)
        n_iter1 = 10; n_iter2 = 8
        param_list = list(ParameterSampler(grid, n_iter=n_iter1, random_state=42))
        best_cv = -1; best_params = None; best_estimator = None

        # Phase 1: 10 trials (0â75%)
        for i, params in enumerate(param_list):
            m = _make_model_balanced(model_name, dict(params), strat)
            try:
                if strat == 'balanced' and model_name in CW_UNSUPPORTED:
                    sc = cross_val_score(m, X, y, cv=cv, scoring='f1_weighted', error_score=0,
                                         fit_params={'sample_weight': _compute_sample_weight(y)})
                else:
                    sc = cross_val_score(m, X, y, cv=cv, scoring='f1_weighted', error_score=0)
                s = float(sc.mean())
            except:
                s = 0.0
            if s > best_cv:
                best_cv = s; best_params = dict(params)
            JOBS[job_id]['progress'] = int((i + 1) / n_iter1 * 75)

        # Phase 2: anti-overfit refinement (75â95%)
        if best_params is None:
            JOBS[job_id] = {'progress': 0, 'done': True, 'result': None, 'error': 'ëª¨ë  íë¼ë¯¸í° íì ì¤í¨'}
            return
        m_best = _make_model_balanced(model_name, dict(best_params), strat)
        if strat == 'balanced' and model_name in CW_UNSUPPORTED:
            m_best.fit(X, y, sample_weight=_compute_sample_weight(y))
        else:
            m_best.fit(X, y)
        train_score = float(f1_score(y, m_best.predict(X), average='weighted', zero_division=0))
        gap = train_score - best_cv; best_estimator = m_best

        if gap > 0.15:
            tight = {k: v for k, v in grid.items()}
            if 'max_depth' in tight:
                tight['max_depth'] = [d for d in tight['max_depth'] if d is None or d <= 7] or [5]
            if 'min_samples_leaf' in tight:
                tight['min_samples_leaf'] = [v for v in tight['min_samples_leaf'] if v >= 3] or [5]
            tight_list = list(ParameterSampler(tight, n_iter=n_iter2, random_state=42))
            for j, params2 in enumerate(tight_list):
                m2 = _make_model_balanced(model_name, dict(params2), strat)
                try:
                    if strat == 'balanced' and model_name in CW_UNSUPPORTED:
                        sc2 = cross_val_score(m2, X, y, cv=cv, scoring='f1_weighted', error_score=0,
                                              fit_params={'sample_weight': _compute_sample_weight(y)})
                    else:
                        sc2 = cross_val_score(m2, X, y, cv=cv, scoring='f1_weighted', error_score=0)
                    s2 = float(sc2.mean())
                except:
                    s2 = 0.0
                if s2 >= best_cv * 0.95 and s2 > best_cv - 0.05:
                    best_cv = max(best_cv, s2); best_params = dict(params2)
                JOBS[job_id]['progress'] = 75 + int((j + 1) / n_iter2 * 20)

        JOBS[job_id]['progress'] = 98

        # Clean params
        clean = {}
        for k, v in best_params.items():
            if v is None: clean[k] = None
            elif isinstance(v, np.integer): clean[k] = int(v)
            elif isinstance(v, np.floating): clean[k] = float(v)
            else: clean[k] = v

        # Optimal threshold (binary)
        best_thr = None
        if is_binary and hasattr(best_estimator, 'predict_proba'):
            proba = best_estimator.predict_proba(X)[:, 1]
            best_f1 = 0.0
            for thr in np.arange(0.1, 0.91, 0.01):
                preds = (proba >= thr).astype(int)
                f = float(f1_score(y, preds, zero_division=0))
                if f > best_f1:
                    best_f1 = f; best_thr = round(float(thr), 2)

        result = {'params': clean, 'cv_score': best_cv, 'train_score': train_score, 'gap': gap}
        if best_thr is not None:
            result['threshold'] = best_thr

        JOBS[job_id] = {'progress': 100, 'done': True, 'result': result, 'error': None}
    except Exception as e:
        JOBS[job_id] = {'progress': 0, 'done': True, 'result': None, 'error': str(e)}
    finally:
        _cleanup_jobs()


@app.route('/api/tune/start', methods=['POST'])
def api_tune_start():
    body = request.get_json(silent=True)
    if not body:
        return jsonify({'error': 'ìì²­ ë³¸ë¬¸ ìì'}), 400
    sid = body.get('session_id')
    model_name = body['model']
    balance_strategy = body.get('balance', 'none')
    cv_folds = int(body.get('cv_folds', 5))

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {'progress': 0, 'done': False, 'result': None, 'error': None}

    t = threading.Thread(target=_run_tune, args=(job_id, sid, model_name, balance_strategy, cv_folds), daemon=True)
    t.start()
    return jsonify({'job_id': job_id})


@app.route('/api/tune/status', methods=['GET'])
def api_tune_status():
    job_id = request.args.get('job_id')
    job = JOBS.get(job_id)
    if not job:
        return jsonify({'error': 'ìì ìì'}), 404
    return jsonify(job)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    body = request.get_json(silent=True)
    if not body:
        return jsonify({'error': 'ìì²­ ë³¸ë¬¸ ìì'}), 400
    sid = body.get('session_id')
    csv_str = body['csv']
    threshold = float(body.get('threshold', 0.5))

    state = load_state(sid)
    if not state:
        return jsonify({'error': 'ì¸ì ìì'}), 400

    model = state.get('model')
    if model is None:
        return jsonify({'error': 'ë¨¼ì  ëª¨ë¸ì íìµí´ì£¼ì¸ì'}), 400

    feature_names = state['feature_names']
    le = state['le']
    encoders = state['encoders']
    is_binary = state['is_binary']

    try:
        df_n = pd.read_csv(io.StringIO(csv_str))
        df_n.columns = [c.strip() for c in df_n.columns]
        df_p, _ = _auto_preprocess(df_n, None, feature_names)

        for col, enc in encoders.items():
            if col in df_p.columns:
                df_p[col] = df_p[col].apply(
                    lambda x: int(enc.transform([str(x)])[0]) if str(x) in enc.classes_ else 0
                )

        missing = [f for f in feature_names if f not in df_p.columns]
        if missing:
            return jsonify({'error': f'ëë½ í¼ì²: {missing}'}), 400

        X_df = pd.DataFrame(index=df_p.index)
        for f in feature_names:
            col = df_p[f].copy()
            if col.dtype == object:
                col = pd.to_numeric(col, errors='coerce')
                if col.isna().all():  # ìì  ë¬¸ìì´ ì»¬ë¼ â 0ì¼ë¡ ì±ì
                    col = pd.Series(0, index=df_p.index)
            X_df[f] = col
        X_n = X_df.fillna(0).values.astype(float)

        if is_binary and hasattr(model, 'predict_proba'):
            proba_all = model.predict_proba(X_n)
            raw_preds = (proba_all[:, 1] >= threshold).astype(int)
        else:
            proba_all = None
            raw_preds = model.predict(X_n)

        preds = le.inverse_transform(raw_preds).tolist()
        r = df_n.copy()
        r['ìì¸¡ê²°ê³¼'] = preds

        if hasattr(model, 'predict_proba'):
            proba = proba_all if proba_all is not None else model.predict_proba(X_n)
            classes = le.classes_.tolist()
            for i, cls in enumerate(classes):
                r[f'P({cls})'] = np.round(proba[:, i], 4)
            if is_binary:
                r['íë¥ (ìì±)']
