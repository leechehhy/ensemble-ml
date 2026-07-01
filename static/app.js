/* ═══════════════════════════════════════════════════════════
   EnsembleML Studio — app.js
   Flask API 연동 프론트엔드 로직
   ═══════════════════════════════════════════════════════════ */

// ── 전역 상태 ───────────────────────────────────────────────
let SESSION_ID   = null;
let csvData      = null;   // 학습용 CSV 문자열
let predCsvData  = null;   // 예측용 CSV 문자열
let allColumns   = [];
let selectedFeatures = [];
let targetCol    = null;
let classInfo    = null;   // { classes, is_binary, class_dist }
let cvFolds      = 5;
let selectedModel = null;
let bestParams   = {};
let threshold    = 0.5;
let predResultCsv = null;  // 다운로드용 예측 결과
let previewRows  = [];     // 미리보기용 원본 행 (target 변경 시 재렌더링용)

// ── 모델 정의 ───────────────────────────────────────────────
const MODELS = [
  'Random Forest',
  'Extra Trees',
  'Gradient Boosting',
  'AdaBoost',
  'XGBoost',
  'HistGBM (LightGBM계열)',
  'CatBoost',
];

const MODEL_DESC = {
  'Random Forest':        { emoji:'🌲', desc:'다수의 결정 트리를 병렬 학습. 과적합에 강하고 안정적.', pros:'빠른 학습, 해석 용이', cons:'메모리 사용량 높음' },
  'Extra Trees':          { emoji:'🌳', desc:'Random Forest보다 더 무작위적인 트리 구성으로 분산 감소.', pros:'빠름, 노이즈에 강함', cons:'Random Forest 대비 편향 증가 가능' },
  'Gradient Boosting':    { emoji:'📈', desc:'트리를 순차적으로 쌓아 이전 오류를 보정.', pros:'높은 정확도', cons:'학습 느림, 하이퍼파라미터 민감' },
  'AdaBoost':             { emoji:'🔁', desc:'오분류 샘플에 가중치를 높여 반복 학습.', pros:'간단한 구조', cons:'노이즈·이상값에 민감' },
  'XGBoost':              { emoji:'⚡', desc:'최적화된 Gradient Boosting. 속도와 성능 모두 우수.', pros:'빠름, 정규화 내장', cons:'파라미터 튜닝 필요' },
  'HistGBM (LightGBM계열)': { emoji:'💡', desc:'히스토그램 기반 GBM. 대용량 데이터에 강함.', pros:'매우 빠름, 메모리 효율', cons:'소규모 데이터 과적합 주의' },
  'CatBoost':             { emoji:'🐱', desc:'범주형 변수 처리에 특화된 Gradient Boosting.', pros:'범주형 자동 처리', cons:'학습 느릴 수 있음' },
};

const PARAM_CONFIG = {
  'Random Forest': [
    { key:'n_estimators',    label:'트리 수 (n_estimators)',     min:50,  max:500, step:50,  def:100 },
    { key:'max_depth',       label:'최대 깊이 (max_depth)',      min:2,   max:30,  step:1,   def:10, nullable:true },
    { key:'min_samples_leaf',label:'리프 최소 샘플',             min:1,   max:20,  step:1,   def:1  },
    { key:'max_features',    label:'피처 비율 (max_features)',   min:0.1, max:1.0, step:0.05,def:0.5, float:true },
  ],
  'Extra Trees': [
    { key:'n_estimators',    label:'트리 수 (n_estimators)',     min:50,  max:500, step:50,  def:100 },
    { key:'max_depth',       label:'최대 깊이 (max_depth)',      min:2,   max:30,  step:1,   def:10, nullable:true },
    { key:'min_samples_leaf',label:'리프 최소 샘플',             min:1,   max:20,  step:1,   def:1  },
    { key:'max_features',    label:'피처 비율 (max_features)',   min:0.1, max:1.0, step:0.05,def:0.5, float:true },
  ],
  'Gradient Boosting': [
    { key:'n_estimators',    label:'트리 수 (n_estimators)',     min:50,  max:300, step:50,  def:100 },
    { key:'max_depth',       label:'최대 깊이 (max_depth)',      min:2,   max:10,  step:1,   def:3  },
    { key:'learning_rate',   label:'학습률 (learning_rate)',     min:0.01,max:0.3, step:0.01,def:0.1, float:true },
    { key:'min_samples_leaf',label:'리프 최소 샘플',             min:1,   max:20,  step:1,   def:1  },
  ],
  'AdaBoost': [
    { key:'n_estimators',    label:'트리 수 (n_estimators)',     min:50,  max:300, step:50,  def:100 },
    { key:'learning_rate',   label:'학습률 (learning_rate)',     min:0.1, max:3.0, step:0.1, def:1.0, float:true },
    { key:'base_max_depth',  label:'기본 트리 깊이',             min:1,   max:5,   step:1,   def:1  },
  ],
  'XGBoost': [
    { key:'n_estimators',    label:'트리 수 (n_estimators)',     min:50,  max:300, step:50,  def:100 },
    { key:'max_depth',       label:'최대 깊이 (max_depth)',      min:2,   max:10,  step:1,   def:6  },
    { key:'learning_rate',   label:'학습률 (learning_rate)',     min:0.01,max:0.3, step:0.01,def:0.1, float:true },
    { key:'min_child_weight',label:'자식 최소 가중치',           min:1,   max:20,  step:1,   def:1  },
  ],
  'HistGBM (LightGBM계열)': [
    { key:'max_iter',        label:'반복 수 (max_iter)',         min:50,  max:300, step:50,  def:100 },
    { key:'max_depth',       label:'최대 깊이 (max_depth)',      min:2,   max:15,  step:1,   def:5, nullable:true },
    { key:'learning_rate',   label:'학습률 (learning_rate)',     min:0.01,max:0.3, step:0.01,def:0.1, float:true },
    { key:'min_samples_leaf',label:'리프 최소 샘플',             min:5,   max:100, step:5,   def:20 },
  ],
  'CatBoost': [
    { key:'iterations',      label:'반복 수 (iterations)',       min:50,  max:300, step:50,  def:100 },
    { key:'depth',           label:'트리 깊이 (depth)',          min:2,   max:10,  step:1,   def:6  },
    { key:'learning_rate',   label:'학습률 (learning_rate)',     min:0.01,max:0.3, step:0.01,def:0.1, float:true },
    { key:'l2_leaf_reg',     label:'L2 정규화 (l2_leaf_reg)',   min:1,   max:10,  step:1,   def:3  },
  ],
};

// ── 초기화 ──────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
  initTheme();
  initFileUpload();
  initPredUpload();
  initDragDrop();

  // 테마
  document.getElementById('themeToggle').addEventListener('click', toggleTheme);

  // 업로드 존 클릭 → 파일 선택
  document.getElementById('uploadZone').addEventListener('click', () =>
    document.getElementById('fileInput').click());

  // 피처 선택 버튼 (전체선택/해제/반전 — 테이블 체크박스 조작)
  document.getElementById('btnFeatAll').addEventListener('click', featSelectAll);
  document.getElementById('btnFeatNone').addEventListener('click', featDeselectAll);
  document.getElementById('btnFeatInvert').addEventListener('click', featInvert);

  // 분할 슬라이더
  document.getElementById('splitSlider').addEventListener('input', e =>
    updateSplitLabel(e.target.value));

  // CV 폴드 버튼
  document.getElementById('cv3btn').addEventListener('click', function() { setCvFolds(3, this); });
  document.getElementById('cv5btn').addEventListener('click', function() { setCvFolds(5, this); });
  document.getElementById('cv10btn').addEventListener('click', function() { setCvFolds(10, this); });

  // 스텝 이동
  document.getElementById('step1NextBtn').addEventListener('click', goStep2);
  document.getElementById('step2BackBtn').addEventListener('click', goStep1);
  document.getElementById('step2NextBtn').addEventListener('click', goStep3);
  document.getElementById('step3BackBtn').addEventListener('click', () => setStep(2));
  document.getElementById('step3NextBtn').addEventListener('click', goStep4);
  document.getElementById('step4BackBtn').addEventListener('click', () => setStep(3));

  // 평가 / 튜닝
  document.getElementById('evalBtn').addEventListener('click', runEval);
  document.getElementById('tuneBtn').addEventListener('click', autoTuneParams);

  // 임계값 슬라이더
  document.getElementById('thresholdSlider').addEventListener('input', e => {
    threshold = parseFloat(e.target.value);
    document.getElementById('thresholdVal').textContent = threshold.toFixed(2);
    debounceEval();
  });

  // 데이터 탐색 패널
  document.getElementById('corrToggleBtn').addEventListener('click', openCorrPanel);
  document.getElementById('corrCloseBtn').addEventListener('click', closeCorrPanel);
  document.getElementById('etab-corr').addEventListener('click', () => showExploreTab('corr'));
  document.getElementById('etab-dist').addEventListener('click', () => showExploreTab('dist'));
  document.getElementById('etab-target').addEventListener('click', () => showExploreTab('target'));

  // 예측
  document.getElementById('predZone').addEventListener('click', () =>
    document.getElementById('predInput').click());
  document.getElementById('predClearBtn').addEventListener('click', clearPredFile);
  document.getElementById('predictBtn').addEventListener('click', runPredict);
  document.getElementById('downloadBtn').addEventListener('click', downloadResults);
  document.getElementById('resetPredBtn').addEventListener('click', resetPredict);

  // 로딩 화면 제거
  setTimeout(() => {
    document.getElementById('loader').style.opacity = '0';
    document.getElementById('loader').style.transition = 'opacity .4s';
    setTimeout(() => {
      document.getElementById('loader').style.display = 'none';
      document.getElementById('app').classList.remove('hidden');
    }, 400);
  }, 700);
});

// ── 테마 ────────────────────────────────────────────────────
function initTheme() {
  const saved = localStorage.getItem('theme');
  if (saved === 'dark') {
    document.documentElement.classList.add('dark');
    document.getElementById('themeToggle').textContent = '☀️ 라이트';
  }
}

function toggleTheme() {
  const isDark = document.documentElement.classList.toggle('dark');
  document.getElementById('themeToggle').textContent = isDark ? '☀️ 라이트' : '🌙 다크';
  localStorage.setItem('theme', isDark ? 'dark' : 'light');
}

// ── 스텝 이동 ────────────────────────────────────────────────
function setStep(n) {
  [1,2,3,4].forEach(i => {
    document.getElementById(`step${i}`)?.classList.toggle('hidden', i !== n);
    const sc = document.getElementById(`sc${i}`);
    const sl = document.getElementById(`sl${i}`);
    if (!sc || !sl) return;
    sc.classList.remove('active','done');
    sl.classList.remove('active','done');
    if (i < n)  { sc.classList.add('done');   sl.classList.add('done'); }
    if (i === n){ sc.classList.add('active');  sl.classList.add('active'); }
    const ln = document.getElementById(`ln${i}`);
    if (ln) ln.classList.toggle('done', i < n);
  });
}

function goStep1() { setStep(1); }
function goStep2() {
  if (!validateStep1()) return;
  setStep(2);
  loadDataAndEval();
}
function goStep3() { setStep(3); buildParamControls(selectedModel); }
function goStep4() { setStep(4); }

// ── STEP 1 유효성 검사 ───────────────────────────────────────
function validateStep1() {
  if (!csvData) { alert('파일을 먼저 업로드하세요.'); return false; }
  const sel = getSelectedFeatures();
  if (sel.length === 0) {
    document.getElementById('featWarn').classList.remove('hidden');
    return false;
  }
  document.getElementById('featWarn').classList.add('hidden');
  return true;
}

// ── 파일 업로드 ──────────────────────────────────────────────
function initFileUpload() {
  const input = document.getElementById('fileInput');
  input.addEventListener('change', e => handleFile(e.target.files[0]));
}

function initDragDrop() {
  const zone = document.getElementById('uploadZone');
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('dragover');
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  });
}

function handleFile(file) {
  if (!file) return;
  const ext = file.name.split('.').pop().toLowerCase();
  if (!['csv','xlsx','xls'].includes(ext)) {
    alert('CSV 또는 Excel 파일만 지원합니다.');
    return;
  }
  const reader = new FileReader();
  reader.onload = e => {
    try {
      let csv;
      if (ext === 'csv') {
        csv = e.target.result;
      } else {
        const wb = XLSX.read(e.target.result, { type: 'binary' });
        const ws = wb.Sheets[wb.SheetNames[0]];
        csv = XLSX.utils.sheet_to_csv(ws);
      }
      csvData = csv;
      parseAndPreview(csv, file.name);
    } catch(err) {
      alert('파일 파싱 오류: ' + err.message);
    }
  };
  if (ext === 'csv') reader.readAsText(file, 'UTF-8');
  else               reader.readAsBinaryString(file);
}

function parseAndPreview(csv, filename) {
  const lines = csv.trim().split('\n');
  const headers = lines[0].split(',').map(h => h.trim().replace(/^"|"$/g,''));
  allColumns = headers;
  previewRows = lines.slice(1, 6);

  // 파일 정보 표시
  document.getElementById('fileInfoMsg').textContent =
    `✅ ${filename} 로드 완료 — ${lines.length - 1}행 × ${headers.length}열`;
  document.getElementById('fileInfo').classList.remove('hidden');

  // Target 선택 드롭다운
  const sel = document.getElementById('targetSelect');
  sel.innerHTML = headers.map(h => `<option value="${h}">${h}</option>`).join('');
  sel.value = headers[headers.length - 1];
  sel.addEventListener('change', () => refreshFeatureSelection(sel.value));

  // 행/열 정보
  document.getElementById('infoRows').textContent = `${lines.length - 1}행`;
  document.getElementById('infoCols').textContent = `${headers.length}열`;

  document.getElementById('previewSection').classList.remove('hidden');

  // target 설정 후 테이블 렌더링 (체크박스 포함)
  targetCol = sel.value;
  renderPreviewTable(headers, previewRows, targetCol);

  updateFeatCount();
  updateDistribution();
}

// ── 데이터 미리보기 테이블 (헤더에 체크박스 포함) ───────────
function renderPreviewTable(headers, rows, target) {
  const wrap = document.getElementById('previewTable');

  // 헤더: target 열은 배지, 나머지는 체크박스
  const thHtml = headers.map(h => {
    if (h === target) {
      return `<th>${h}<span class="target-badge"> (target)</span></th>`;
    }
    return `<th><label class="feat-th-label"><input type="checkbox" class="feat-col-cb" data-col="${h}" checked> ${h}</label></th>`;
  }).join('');

  let html = `<table><thead><tr>${thHtml}</tr></thead><tbody>`;
  rows.forEach(row => {
    const cells = row.split(',').map(c => c.trim().replace(/^"|"$/g,''));
    html += '<tr>' + cells.map(c => `<td>${c}</td>`).join('') + '</tr>';
  });
  html += '</tbody></table>';
  wrap.innerHTML = html;

  // 체크박스 변경 시 카운트 업데이트
  wrap.querySelectorAll('.feat-col-cb').forEach(cb => {
    cb.addEventListener('change', updateFeatCount);
  });
}

// ── 피처 선택 갱신 (target 변경 시 호출) ────────────────────
function refreshFeatureSelection(target) {
  targetCol = target;
  document.getElementById('infoTarget').textContent = `Target: ${target}`;

  // 현재 체크 상태 기억
  const prevChecked = new Set(
    [...document.querySelectorAll('.feat-col-cb:checked')].map(cb => cb.dataset.col)
  );

  // 테이블 재렌더링
  renderPreviewTable(allColumns, previewRows, target);

  // 이전 체크 상태 복원
  document.querySelectorAll('.feat-col-cb').forEach(cb => {
    cb.checked = prevChecked.size === 0 || prevChecked.has(cb.dataset.col);
  });

  updateFeatCount();
  updateDistribution();
}

// ── 피처 선택 읽기 ──────────────────────────────────────────
function getSelectedFeatures() {
  return [...document.querySelectorAll('.feat-col-cb:checked')]
    .map(cb => cb.dataset.col);
}

function updateFeatCount() {
  const n = getSelectedFeatures().length;
  document.getElementById('featCount').textContent = `선택된 피처: ${n}개`;
  const warn = document.getElementById('featWarn');
  if (n === 0) warn.classList.remove('hidden');
  else         warn.classList.add('hidden');
}

// 전체선택 / 전체해제 / 반전
function featSelectAll() {
  document.querySelectorAll('.feat-col-cb').forEach(cb => cb.checked = true);
  updateFeatCount();
}
function featDeselectAll() {
  document.querySelectorAll('.feat-col-cb').forEach(cb => cb.checked = false);
  updateFeatCount();
}
function featInvert() {
  document.querySelectorAll('.feat-col-cb').forEach(cb => cb.checked = !cb.checked);
  updateFeatCount();
}

// ── 훈련/테스트 분할 ─────────────────────────────────────────
function updateSplitLabel(val) {
  document.getElementById('splitLabel').textContent = `훈련 ${val}% / 테스트 ${100-val}%`;
}

// ── 교차검증 ─────────────────────────────────────────────────
function setCvFolds(n, btn) {
  cvFolds = n;
  document.querySelectorAll('#cv3btn,#cv5btn,#cv10btn').forEach(b => {
    b.classList.remove('btn-primary');
    b.classList.add('btn-outline');
  });
  btn.classList.add('btn-primary');
  btn.classList.remove('btn-outline');
}

// ── 클래스 분포 ──────────────────────────────────────────────
function updateDistribution() {
  if (!csvData || !targetCol) return;
  const lines   = csvData.trim().split('\n');
  const headers = lines[0].split(',').map(h => h.trim().replace(/^"|"$/g, ''));
  const tIdx    = headers.indexOf(targetCol);
  if (tIdx < 0) return;

  const counts = {};
  lines.slice(1).forEach(line => {
    if (!line.trim()) return;
    const cells = line.split(',');
    const val   = (cells[tIdx] || '').trim().replace(/^"|"$/g, '');
    if (val) counts[val] = (counts[val] || 0) + 1;
  });

  if (Object.keys(counts).length === 0) return;
  renderDistribution(counts, Object.keys(counts).length === 2);
}

function renderDistribution(classDist, isBinary) {
  const section = document.getElementById('distSection');
  const chart   = document.getElementById('distChart');
  section.classList.remove('hidden');

  const total = Object.values(classDist).reduce((a,b) => a+b, 0);
  const counts = Object.values(classDist);
  const maxCnt = Math.max(...counts);
  const minCnt = Math.min(...counts);
  const ratio  = maxCnt / (minCnt || 1);
  const isImbalanced = ratio > 3;

  const tag = document.getElementById('distImbalanceTag');
  if (isImbalanced) {
    tag.innerHTML = `<span style="background:var(--danger);color:#fff;font-size:10px;padding:2px 8px;border-radius:20px;">⚠ 불균형 (${ratio.toFixed(1)}:1)</span>`;
  } else {
    tag.innerHTML = `<span style="background:var(--success);color:#fff;font-size:10px;padding:2px 8px;border-radius:20px;">✅ 균형</span>`;
  }

  const colors = ['#C86000','#16A34A','#7C3AED','#D97706','#0284C7','#DB2777'];
  chart.innerHTML = Object.entries(classDist).map(([cls, cnt], i) => {
    const pct = (cnt / total * 100).toFixed(1);
    const barW = (cnt / maxCnt * 100).toFixed(1);
    const color = colors[i % colors.length];
    return `<div class="dist-row">
      <div class="dist-label" title="${cls}">${cls}</div>
      <div class="dist-bar-bg">
        <div class="dist-bar-fill" style="width:${barW}%;background:${color};"></div>
        <div class="dist-bar-text">${pct}%</div>
      </div>
      <div class="dist-cnt">${cnt.toLocaleString()}건</div>
    </div>`;
  }).join('');

  const imbalBlock = document.getElementById('imbalanceBlock');
  if (isImbalanced) {
    imbalBlock.classList.remove('hidden');
    document.getElementById('imbalanceWarnMsg').innerHTML =
      `<strong>⚠ 클래스 불균형 감지!</strong> 최대 ${ratio.toFixed(1)}배 차이. 처리 전략을 선택하세요.`;
    renderBalanceOptions();
  } else {
    imbalBlock.classList.add('hidden');
  }

  classInfo = { isBinary, classDist, isImbalanced };
}

function renderBalanceOptions() {
  const opts = [
    { val:'none',        title:'처리 안 함',     desc:'원본 데이터 그대로 학습' },
    { val:'balanced',    title:'클래스 가중치',  desc:'소수 클래스에 자동 가중치 부여 (권장)' },
    { val:'oversample',  title:'오버샘플링',     desc:'소수 클래스 복제 증가' },
    { val:'undersample', title:'언더샘플링',     desc:'다수 클래스 축소' },
  ];
  const wrap = document.getElementById('balanceOpts');
  wrap.innerHTML = opts.map(o => `
    <div class="balance-opt${o.val==='balanced'?' selected':''}" data-val="${o.val}" onclick="selectBalance(this)">
      <div>
        <div style="font-size:12px;font-weight:700;">${o.title}</div>
        <div style="font-size:11px;color:var(--muted);margin-top:2px;">${o.desc}</div>
      </div>
    </div>`).join('');
}

function selectBalance(el) {
  document.querySelectorAll('.balance-opt').forEach(o => o.classList.remove('selected'));
  el.classList.add('selected');
}

function getBalanceStrategy() {
  const sel = document.querySelector('.balance-opt.selected');
  return sel ? sel.dataset.val : 'none';
}

// ── 데이터 탐색 패널 ─────────────────────────────────────────
function openCorrPanel()  { document.getElementById('corrPanel').classList.remove('hidden'); drawCorrHeatmap(); }
function closeCorrPanel() { document.getElementById('corrPanel').classList.add('hidden'); }
function showExploreTab(tab, btn) {
  ['corr','dist','target'].forEach(t => {
    document.getElementById(`exploreTab${t.charAt(0).toUpperCase()+t.slice(1)}`).classList.toggle('hidden', t !== tab);
    document.getElementById(`etab-${t}`).classList.toggle('active', t === tab);
  });
  if (tab === 'corr')   drawCorrHeatmap();
  if (tab === 'dist')   drawDistCharts();
  if (tab === 'target') drawTargetChart();
}

function parseCsvMatrix() {
  if (!csvData) return null;
  const lines   = csvData.trim().split('\n');
  const headers = lines[0].split(',').map(h => h.trim().replace(/^"|"$/g,''));
  const rows    = lines.slice(1).map(l => l.split(',').map(v => parseFloat(v.trim().replace(/^"|"$/g,''))));
  return { headers, rows };
}

function drawCorrHeatmap() {
  const canvas = document.getElementById('corrCanvas');
  if (!canvas || !csvData) return;
  const { headers, rows } = parseCsvMatrix();
  const feats = getSelectedFeatures().filter(f => {
    const idx = headers.indexOf(f);
    return idx >= 0 && rows.some(r => !isNaN(r[idx]));
  }).slice(0, 12);

  if (feats.length < 2) { canvas.style.display='none'; document.getElementById('corrLegend').textContent='수치형 피처가 2개 이상 필요합니다.'; return; }
  canvas.style.display = 'block';

  const idxMap = feats.map(f => headers.indexOf(f));
  const n      = feats.length;
  const cell   = Math.min(52, Math.floor(480 / n));
  const pad    = 100;
  const W      = cell * n + pad;
  const H      = cell * n + pad;
  canvas.width  = W;
  canvas.height = H;
  const ctx     = canvas.getContext('2d');
  ctx.clearRect(0, 0, W, H);

  const vals = idxMap.map(idx => rows.map(r => r[idx]).filter(v => !isNaN(v)));
  const mean = v => v.reduce((a,b)=>a+b,0)/v.length;
  const std  = (v,m) => Math.sqrt(v.reduce((a,b)=>a+(b-m)**2,0)/v.length) || 1;
  const corr = (a,b) => {
    const ma=mean(a), mb=mean(b), sa=std(a,ma), sb=std(b,mb);
    const pairs = a.map((x,i)=>[x,b[i]]).filter(([x,y])=>!isNaN(x)&&!isNaN(y));
    return pairs.reduce((s,[x,y])=>s+(x-ma)*(y-mb),0)/(pairs.length*sa*sb);
  };

  const isDark = document.documentElement.classList.contains('dark');
  ctx.font      = `${Math.max(9, cell/5)}px Segoe UI, sans-serif`;
  ctx.textAlign = 'center';

  for (let i=0;i<n;i++) for (let j=0;j<n;j++) {
    const r = corr(vals[i], vals[j]);
    const t = isNaN(r) ? 0 : r;
    const red = t > 0 ? Math.round(200*t) : 0;
    const blu = t < 0 ? Math.round(200*-t) : 0;
    ctx.fillStyle = `rgb(${red+50},50,${blu+50})`;
    ctx.fillRect(pad + j*cell, pad + i*cell, cell-1, cell-1);
    ctx.fillStyle = '#fff';
    ctx.fillText(isNaN(r)?'—':r.toFixed(2), pad+j*cell+cell/2, pad+i*cell+cell/2+4);
  }

  ctx.fillStyle  = isDark ? '#FFF0D8' : '#1C0800';
  ctx.textAlign  = 'right';
  ctx.font       = `${Math.max(9,cell/5)}px Segoe UI, sans-serif`;
  feats.forEach((f,i) => ctx.fillText(f.length>12?f.slice(0,11)+'…':f, pad-4, pad+i*cell+cell/2+4));
  ctx.textAlign  = 'center';
  feats.forEach((f,j) => {
    ctx.save();
    ctx.translate(pad+j*cell+cell/2, pad-4);
    ctx.rotate(-Math.PI/4);
    ctx.fillText(f.length>10?f.slice(0,9)+'…':f, 0, 0);
    ctx.restore();
  });
  document.getElementById('corrLegend').textContent = '🔴 양의 상관 → 높을수록 함께 증가  |  🔵 음의 상관 → 반대 방향';
}

function drawDistCharts() {
  if (!csvData) return;
  const area = document.getElementById('distChartArea');
  area.innerHTML = '';
  const { headers, rows } = parseCsvMatrix();
  const feats = getSelectedFeatures().slice(0, 16);
  feats.forEach(f => {
    const idx  = headers.indexOf(f);
    if (idx < 0) return;
    const vals = rows.map(r=>r[idx]).filter(v=>!isNaN(v));
    const canvas = document.createElement('canvas');
    canvas.width  = 260;
    canvas.height = 140;
    const label   = document.createElement('div');
    label.style.cssText = 'font-size:11px;color:var(--muted);margin-bottom:4px;font-weight:600;';
    label.textContent   = f;
    const wrap = document.createElement('div');
    wrap.appendChild(label);
    wrap.appendChild(canvas);
    area.appendChild(wrap);
    drawHistogram(canvas, vals, f);
  });
}

function drawHistogram(canvas, vals, label) {
  const ctx = canvas.getContext('2d');
  const bins = 12;
  const mn = Math.min(...vals), mx = Math.max(...vals);
  const bw = (mx - mn) / bins || 1;
  const counts = new Array(bins).fill(0);
  vals.forEach(v => { const b = Math.min(Math.floor((v-mn)/bw), bins-1); counts[b]++; });
  const maxC = Math.max(...counts);
  const W=canvas.width, H=canvas.height, pad=10;
  ctx.clearRect(0,0,W,H);
  counts.forEach((c,i) => {
    const x   = pad + i*(W-pad*2)/bins;
    const bW  = (W-pad*2)/bins - 1;
    const bH  = ((H-pad*2) * c/maxC);
    ctx.fillStyle = '#C86000';
    ctx.fillRect(x, H-pad-bH, bW, bH);
  });
}

function drawTargetChart() {
  const canvas = document.getElementById('targetCanvas');
  if (!canvas || !csvData || !targetCol) return;
  const { headers, rows } = parseCsvMatrix();
  const tIdx  = headers.indexOf(targetCol);
  if (tIdx < 0) return;
  const feats = getSelectedFeatures().filter(f => {
    const i = headers.indexOf(f);
    return i>=0 && rows.some(r=>!isNaN(r[i]));
  }).slice(0, 8);
  if (feats.length === 0) return;

  canvas.width  = 600;
  canvas.height = 200;
  const ctx     = canvas.getContext('2d');
  ctx.clearRect(0,0,600,200);
  ctx.fillStyle = '#888';
  ctx.font      = '12px Segoe UI';
  ctx.textAlign = 'center';
  ctx.fillText('타깃별 비교는 데이터 로드 후 사용 가능합니다.', 300, 100);
}

// ── STEP 2: 데이터 로드 & 모델 평가 ───────────────────────────
async function loadDataAndEval() {
  const split    = parseInt(document.getElementById('splitSlider').value);
  const testSize = (100 - split) / 100;
  const balance  = getBalanceStrategy();
  const features = getSelectedFeatures();
  const target   = document.getElementById('targetSelect').value;

  let loadResp;
  try {
    const r = await fetch('/api/load_data', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({
        csv: csvData,
        target,
        features,
        balance,
        test_size: testSize,
        session_id: SESSION_ID,
      })
    });
    loadResp = await r.json();
  } catch(e) {
    alert('데이터 로드 오류: ' + e.message);
    goStep1();
    return;
  }

  if (loadResp.error) { alert('오류: ' + loadResp.error); goStep1(); return; }

  SESSION_ID = loadResp.session_id;
  classInfo  = { isBinary: loadResp.is_binary, classDist: loadResp.class_dist };

  if (loadResp.prep_issues?.length > 0) {
    const rep = document.getElementById('prepReport');
    const iss = document.getElementById('prepIssues');
    iss.innerHTML = loadResp.prep_issues.map(i=>`<div class="prep-issue">• ${i}</div>`).join('');
    rep.classList.remove('hidden');
  }

  renderDistribution(loadResp.class_dist, loadResp.is_binary);

  const results  = document.getElementById('step2Results');
  const progress = document.getElementById('evalProgress');
  const grid     = document.getElementById('modelGrid');
  results.classList.add('hidden');
  progress.style.display = 'block';
  grid.innerHTML = '';

  // 이전 로그 초기화
  logItems = {};
  document.getElementById('progLog').innerHTML = '';

  const scores = [];
  for (let i = 0; i < MODELS.length; i++) {
    const model = MODELS[i];
    const pct   = Math.round((i / MODELS.length) * 100);
    document.getElementById('progTitle').textContent = `평가 중: ${model}`;
    document.getElementById('progPct').textContent   = `${pct}%`;
    document.getElementById('progBar').style.width   = `${pct}%`;
    addProgLog(model, 'running', i);

    try {
      const r = await fetch('/api/evaluate_single', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ session_id:SESSION_ID, model, cv_folds:cvFolds })
      });
      const d = await r.json();
      scores.push({ model, f1: d.ok ? d.f1 : 0, std: d.ok ? d.std : 0, ok: d.ok, err: d.err });
      updateProgLog(model, d.ok ? d.f1 : 0, d.ok);
    } catch(e) {
      scores.push({ model, f1:0, std:0, ok:false, err:e.message });
      updateProgLog(model, 0, false);
    }
  }

  document.getElementById('progTitle').textContent = '✅ 평가 완료';
  document.getElementById('progPct').textContent   = '100%';
  document.getElementById('progBar').style.width   = '100%';

  const best = scores.reduce((a,b) => b.f1 > a.f1 ? b : a, scores[0]);
  selectedModel = best.model;

  scores.sort((a,b)=>b.f1-a.f1).forEach(s => {
    const isRec  = s.model === best.model;
    const iseSel = s.model === selectedModel;
    const desc   = MODEL_DESC[s.model] || {};
    const color  = s.f1 >= 0.8 ? 'var(--success)' : s.f1 >= 0.6 ? 'var(--warning)' : 'var(--danger)';
    const card   = document.createElement('div');
    card.className = `model-card${isRec?' recommended':''}${iseSel?' selected':''}`;
    card.dataset.model = s.model;
    card.innerHTML = `
      <div class="model-name">${desc.emoji||'🤖'} ${s.model}</div>
      <div class="model-score" style="color:${color}">${s.ok?(s.f1*100).toFixed(1)+'%':'—'}</div>
      <div class="model-lbl">F1-weighted (${cvFolds}-Fold CV)</div>
      ${s.ok?`<div class="model-std">±${(s.std*100).toFixed(1)}%</div>`:''}
      <div class="score-bar"><div class="score-bar-fill" style="width:${(s.f1*100).toFixed(1)}%"></div></div>
    `;
    card.addEventListener('click', () => {
      document.querySelectorAll('.model-card').forEach(c=>c.classList.remove('selected'));
      card.classList.add('selected');
      selectedModel = s.model;
      showModelDesc(s.model);
    });
    grid.appendChild(card);
  });

  showModelDesc(selectedModel);
  document.getElementById('step2CvInfo').innerHTML =
    `📊 ${cvFolds}-Fold 교차검증 F1 결과. <strong>${best.model}</strong>이 자동 추천됩니다.`;

  setTimeout(() => {
    progress.style.display = 'none';
    results.classList.remove('hidden');
  }, 300);
}

let logItems = {};
function addProgLog(model, state, idx) {
  const log  = document.getElementById('progLog');
  const item = document.createElement('div');
  item.className  = `log-item ${state}`;
  item.id         = `log-${model.replace(/\s/g,'_')}`;
  item.innerHTML  = `<span>${MODEL_DESC[model]?.emoji||'🤖'} ${model}</span><span class="log-score"></span>`;
  log.appendChild(item);
  logItems[model] = item;
}

function updateProgLog(model, score, ok) {
  const item = logItems[model];
  if (!item) return;
  item.classList.remove('running');
  item.classList.add('done');
  const sc = item.querySelector('.log-score');
  sc.textContent = ok ? `${(score*100).toFixed(1)}%` : '오류';
  sc.style.color = ok ? 'var(--success)' : 'var(--danger)';
}

function showModelDesc(model) {
  const panel = document.getElementById('modelDescPanel');
  const d     = MODEL_DESC[model];
  if (!d) { panel.style.display='none'; return; }
  panel.style.display = 'block';
  panel.innerHTML = `
    <div style="font-size:14px;font-weight:700;margin-bottom:8px;">${d.emoji} ${model}</div>
    <div style="font-size:12px;color:var(--muted);margin-bottom:10px;">${d.desc}</div>
    <div style="display:flex;gap:8px;flex-wrap:wrap;">
      <div style="flex:1;padding:8px 12px;background:rgba(22,163,74,.08);border:1px solid rgba(22,163,74,.25);border-radius:8px;font-size:12px;color:#15803d;">
        ✅ 장점<br><span style="color:var(--text)">${d.pros}</span>
      </div>
      <div style="flex:1;padding:8px 12px;background:rgba(245,158,11,.08);border:1px solid rgba(245,158,11,.25);border-radius:8px;font-size:12px;color:#92400e;">
        ⚠ 단점<br><span style="color:var(--text)">${d.cons}</span>
      </div>
    </div>`;
}

// ── STEP 3: 파라미터 컨트롤 ──────────────────────────────────
function buildParamControls(model) {
  if (!model) return;
  selectedModel = model;
  const configs  = PARAM_CONFIG[model] || [];
  const grid     = document.getElementById('paramControls');
  bestParams     = {};
  grid.innerHTML = '';

  const tabs = document.getElementById('modelTabs');
  tabs.innerHTML = MODELS.map(m => `
    <button class="tab-btn${m===model?' active':''}" onclick="switchModel('${m}',this)">${MODEL_DESC[m]?.emoji||''} ${m}</button>
  `).join('');

  configs.forEach(cfg => {
    bestParams[cfg.key] = cfg.def;
    const wrap = document.createElement('div');
    wrap.innerHTML = `
      <div class="param-lbl">${cfg.label}: <span id="pval-${cfg.key}">${cfg.def}</span></div>
      <input type="range" class="slider" id="pslider-${cfg.key}"
        min="${cfg.min}" max="${cfg.max}" step="${cfg.step}" value="${cfg.def}"
        oninput="updateParam('${cfg.key}',this.value,${!!cfg.float})">
    `;
    grid.appendChild(wrap);
  });

  const thrCtrl = document.getElementById('thresholdControl');
  if (classInfo?.isBinary) {
    thrCtrl.classList.remove('hidden');
    document.getElementById('thresholdSlider').value = '0.5';
    document.getElementById('thresholdVal').textContent = '0.50';
  } else {
    thrCtrl.classList.add('hidden');
  }

  showModelDescStep3(model);
  resetMetrics();
  document.getElementById('step3NextBtn').disabled = true;
}

function updateParam(key, val, isFloat) {
  const v = isFloat ? parseFloat(val) : parseInt(val);
  bestParams[key] = v;
  document.getElementById(`pval-${key}`).textContent = isFloat ? v.toFixed(2) : v;
  debounceEval();
}

function switchModel(model, btn) {
  document.querySelectorAll('#modelTabs .tab-btn').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  buildParamControls(model);
}

function showModelDescStep3(model) {
  const panel = document.getElementById('step3ModelDesc');
  const d     = MODEL_DESC[model];
  if (!d) { panel.style.display='none'; return; }
  panel.style.display = 'block';
  panel.innerHTML = `<strong>${d.emoji} ${model}</strong> — ${d.desc}`;
}

function resetMetrics() {
  document.getElementById('metricsGrid').innerHTML = `
    <div class="metric-card"><div class="metric-val" style="color:var(--muted)">—</div><div class="metric-lbl">Accuracy</div></div>
    <div class="metric-card"><div class="metric-val" style="color:var(--muted)">—</div><div class="metric-lbl">Precision</div></div>
    <div class="metric-card"><div class="metric-val" style="color:var(--muted)">—</div><div class="metric-lbl">Recall</div></div>
    <div class="metric-card"><div class="metric-val" style="color:var(--muted)">—</div><div class="metric-lbl">F1 Score</div></div>
  `;
  document.getElementById('cmSection').classList.add('hidden');
  document.getElementById('fiSection').classList.add('hidden');
  document.getElementById('perClassSection').classList.add('hidden');
}

// ── 디바운스 평가 ────────────────────────────────────────────
let debTimer = null;
function debounceEval() {
  clearTimeout(debTimer);
  debTimer = setTimeout(runEval, 600);
}

// ── 평가 실행 ────────────────────────────────────────────────
async function runEval() {
  if (!SESSION_ID || !selectedModel) return;
  const btn = document.getElementById('evalBtn');
  btn.disabled    = true;
  btn.textContent = '평가 중...';

  threshold = parseFloat(document.getElementById('thresholdSlider')?.value || 0.5);

  try {
    const r = await fetch('/api/train_and_eval', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({
        session_id: SESSION_ID,
        model:      selectedModel,
        params:     bestParams,
        threshold,
      })
    });
    const d = await r.json();
    if (d.error) { alert('평가 오류: ' + d.error); return; }
    renderMetrics(d);
    document.getElementById('step3NextBtn').disabled = false;
  } catch(e) {
    alert('오류: ' + e.message);
  } finally {
    btn.disabled    = false;
    btn.textContent = '📈 성능 평가 실행';
  }
}

function renderMetrics(d) {
  const color = v => v >= 0.8 ? 'c-good' : v >= 0.6 ? 'c-mid' : 'c-bad';
  document.getElementById('metricsGrid').innerHTML = `
    <div class="metric-card"><div class="metric-val ${color(d.accuracy)}">${(d.accuracy*100).toFixed(1)}%</div><div class="metric-lbl">Accuracy</div></div>
    <div class="metric-card"><div class="metric-val ${color(d.precision)}">${(d.precision*100).toFixed(1)}%</div><div class="metric-lbl">Precision</div></div>
    <div class="metric-card"><div class="metric-val ${color(d.recall)}">${(d.recall*100).toFixed(1)}%</div><div class="metric-lbl">Recall</div></div>
    <div class="metric-card"><div class="metric-val ${color(d.f1)}">${(d.f1*100).toFixed(1)}%</div><div class="metric-lbl">F1 Score</div></div>
    ${d.auc!=null?`<div class="metric-card"><div class="metric-val ${color(d.auc)}">${(d.auc*100).toFixed(1)}%</div><div class="metric-lbl">AUC-ROC</div></div>`:''}
  `;

  if (d.confusion_matrix) {
    const cm     = d.confusion_matrix;
    const cls    = d.classes || [];
    const cmSec  = document.getElementById('cmSection');
    const cmWrap = document.getElementById('cmWrap');
    cmSec.classList.remove('hidden');
    let html = '<table class="cm-table"><thead><tr><th></th>' +
      cls.map(c=>`<th>예측: ${c}</th>`).join('') + '</tr></thead><tbody>';
    cm.forEach((row, i) => {
      html += `<tr><th>실제: ${cls[i]}</th>` +
        row.map((v,j) => {
          const bg = i===j ? 'rgba(22,163,74,.15)' : v>0 ? 'rgba(220,38,38,.08)' : '';
          return `<td style="background:${bg}">${v}</td>`;
        }).join('') + '</tr>';
    });
    html += '</tbody></table>';
    cmWrap.innerHTML = html;
  }

  if (d.per_class) {
    const sec = document.getElementById('perClassSection');
    sec.classList.remove('hidden');
    sec.innerHTML = '<div style="font-size:13px;font-weight:600;margin:16px 0 10px;">클래스별 성능</div>' +
      Object.entries(d.per_class).map(([cls, m]) => `
        <div style="margin-bottom:8px;padding:10px;background:var(--surface2);border-radius:8px;font-size:12px;">
          <strong>${cls}</strong> —
          Precision: <span style="color:var(--primary)">${(m.precision*100).toFixed(1)}%</span>
          &nbsp;Recall: <span style="color:var(--primary)">${(m.recall*100).toFixed(1)}%</span>
          &nbsp;F1: <span style="color:var(--primary)">${(m.f1*100).toFixed(1)}%</span>
        </div>`).join('');
  }

  if (d.feature_importance?.length > 0) {
    const sec   = document.getElementById('fiSection');
    const chart = document.getElementById('fiChart');
    sec.classList.remove('hidden');
    const max = d.feature_importance[0][1];
    chart.innerHTML = d.feature_importance.map(([name, val]) => `
      <div class="fi-row">
        <div class="fi-name" title="${name}">${name}</div>
        <div class="fi-bar-bg"><div class="fi-bar-fill" style="width:${(val/max*100).toFixed(1)}%"></div></div>
        <div class="fi-val">${(val*100).toFixed(1)}%</div>
      </div>`).join('');
  }
}

// ── 파라미터 자동 튜닝 ───────────────────────────────────────
async function autoTuneParams() {
  if (!SESSION_ID || !selectedModel) return;
  const btn = document.getElementById('tuneBtn');
  btn.disabled = true;

  const overlay = document.getElementById('tuneOverlay');
  overlay.classList.add('show');
  document.getElementById('tuneOvPct').textContent = '0%';
  document.getElementById('tuneOvBar').style.width  = '0%';
  document.getElementById('tuneOvEta').textContent  = '계산 중...';

  let startTime = Date.now();
  let jobId;

  try {
    const r = await fetch('/api/tune/start', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({
        session_id: SESSION_ID,
        model:      selectedModel,
        balance:    getBalanceStrategy(),
        cv_folds:   cvFolds,
      })
    });
    const d = await r.json();
    if (d.error) throw new Error(d.error);
    jobId = d.job_id;
  } catch(e) {
    overlay.classList.remove('show');
    btn.disabled = false;
    alert('튜닝 시작 오류: ' + e.message);
    return;
  }

  const poll = setInterval(async () => {
    try {
      const r  = await fetch(`/api/tune/status?job_id=${jobId}`);
      const st = await r.json();
      const p  = st.progress || 0;
      document.getElementById('tuneOvPct').textContent = `${p}%`;
      document.getElementById('tuneOvBar').style.width  = `${p}%`;

      const elapsed = (Date.now() - startTime) / 1000;
      if (p > 5) {
        const est = (elapsed / p * 100) - elapsed;
        document.getElementById('tuneOvEta').textContent = `예상 잔여: ${est.toFixed(0)}초`;
      }

      if (st.done) {
        clearInterval(poll);
        overlay.classList.remove('show');
        btn.disabled = false;
        if (st.error) { alert('튜닝 오류: ' + st.error); return; }
        applyTuneResult(st.result);
      }
    } catch(e) {
      clearInterval(poll);
      overlay.classList.remove('show');
      btn.disabled = false;
      alert('폴링 오류: ' + e.message);
    }
  }, 800);
}

function applyTuneResult(res) {
  if (!res?.params) return;

  Object.entries(res.params).forEach(([k, v]) => {
    if (v == null) return;
    const slider = document.getElementById(`pslider-${k}`);
    const label  = document.getElementById(`pval-${k}`);
    if (slider) {
      const cfg = (PARAM_CONFIG[selectedModel]||[]).find(c=>c.key===k);
      slider.value  = v;
      if (label) label.textContent = cfg?.float ? parseFloat(v).toFixed(2) : v;
      bestParams[k] = v;
    }
  });

  if (res.threshold != null) {
    threshold = res.threshold;
    const tSlider = document.getElementById('thresholdSlider');
    const tVal    = document.getElementById('thresholdVal');
    if (tSlider) { tSlider.value = threshold; tVal.textContent = threshold.toFixed(2); }
  }

  const panel = document.getElementById('tuneResult');
  panel.classList.remove('hidden');
  panel.innerHTML = `
    <div class="tune-result-title">🎯 추천 파라미터 적용 완료</div>
    <div style="margin-bottom:8px;">
      <span class="tune-stat">CV F1: <span>${(res.cv_score*100).toFixed(1)}%</span></span>
      <span class="tune-stat">Train F1: <span>${(res.train_score*100).toFixed(1)}%</span></span>
      <span class="tune-stat">Gap: <span>${(res.gap*100).toFixed(1)}%</span></span>
      ${res.threshold!=null?`<span class="tune-stat">최적 Threshold: <span>${res.threshold}</span></span>`:''}
    </div>
    <div style="font-size:11px;color:var(--muted);">
      ${Object.entries(res.params).map(([k,v])=>`<span class="tune-stat">${k}: <span>${v}</span></span>`).join('')}
    </div>
  `;

  runEval();
}

// ── STEP 4: 예측 ─────────────────────────────────────────────
function initPredUpload() {
  document.getElementById('predInput').addEventListener('change', e => handlePredFile(e.target.files[0]));
  const zone = document.getElementById('predZone');
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
  zone.addEventListener('drop', e => {
    e.preventDefault(); zone.classList.remove('dragover');
    if (e.dataTransfer.files[0]) handlePredFile(e.dataTransfer.files[0]);
  });
}

function handlePredFile(file) {
  if (!file) return;
  const ext = file.name.split('.').pop().toLowerCase();
  const reader = new FileReader();
  reader.onload = e => {
    try {
      if (ext === 'csv') {
        predCsvData = e.target.result;
      } else {
        const wb = XLSX.read(e.target.result, { type:'binary' });
        predCsvData = XLSX.utils.sheet_to_csv(wb.Sheets[wb.SheetNames[0]]);
      }
      const rows = predCsvData.trim().split('\n').length - 1;
      document.getElementById('predFileMsg').textContent = `✅ ${file.name} — ${rows}행`;
      document.getElementById('predFileInfo').classList.remove('hidden');
      document.getElementById('predictBtn').disabled = false;
    } catch(err) {
      alert('파일 파싱 오류: ' + err.message);
    }
  };
  if (ext === 'csv') reader.readAsText(file, 'UTF-8');
  else               reader.readAsBinaryString(file);
}

function clearPredFile() {
  predCsvData = null;
  predResultCsv = null;
  document.getElementById('predInput').value = '';
  document.getElementById('predFileInfo').classList.add('hidden');
  document.getElementById('predResults').classList.add('hidden');
  document.getElementById('predSummary').classList.add('hidden');
  document.getElementById('predictBtn').disabled = true;
}

async function runPredict() {
  if (!SESSION_ID || !predCsvData) return;
  const btn     = document.getElementById('predictBtn');
  const spinner = document.getElementById('predictSpinner');
  btn.disabled  = true;
  spinner.classList.remove('hidden');

  try {
    const r = await fetch('/api/predict', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ session_id:SESSION_ID, csv:predCsvData })
    });

    if (!r.ok) {
      const err = await r.json();
      throw new Error(err.error || '서버 오류');
    }

    predResultCsv = await r.text();
    renderPredResults(predResultCsv);
  } catch(e) {
    alert('예측 오류: ' + e.message);
  } finally {
    btn.disabled = false;
    spinner.classList.add('hidden');
  }
}

function renderPredResults(csv) {
  const lines   = csv.trim().split('\n');
  const headers = lines[0].split(',').map(h=>h.trim().replace(/^"|"$/g,''));
  const rows    = lines.slice(1, 6);

  let html = '<table><thead><tr>' + headers.map(h=>`<th>${h}</th>`).join('') + '</tr></thead><tbody>';
  rows.forEach(row => {
    const cells = row.split(',').map(c=>c.trim().replace(/^"|"$/g,''));
    html += '<tr>' + cells.map((c,i) => {
      const h = headers[i];
      const cls = h==='예측결과' ? 'result-pred' : h.startsWith('확률') ? 'result-prob-hi' : '';
      return `<td class="${cls}">${c}</td>`;
    }).join('') + '</tr>';
  });
  html += '</tbody></table>';
  document.getElementById('predTable').innerHTML = html;

  const predCol = lines.slice(1).map(l => {
    const cells = l.split(',');
    const idx   = headers.indexOf('예측결과');
    return idx >= 0 ? cells[idx]?.trim().replace(/^"|"$/g,'') : null;
  }).filter(Boolean);

  const summary = predCol.reduce((acc,v) => { acc[v]=(acc[v]||0)+1; return acc; }, {});
  const sumHtml = Object.entries(summary).map(([cls,cnt]) =>
    `<div class="dist-row">
       <div class="dist-label">${cls}</div>
       <div class="dist-cnt" style="font-weight:700;color:var(--primary)">${cnt}건</div>
     </div>`).join('');

  document.getElementById('predSummaryContent').innerHTML = sumHtml + `<div style="font-size:11px;color:var(--muted);margin-top:6px;">전체 ${predCol.length}건 예측 완료</div>`;
  document.getElementById('predSummary').classList.remove('hidden');
  document.getElementById('predResults').classList.remove('hidden');
}

function downloadResults() {
  if (!predResultCsv) return;
  const blob = new Blob(['﻿' + predResultCsv], { type:'text/csv;charset=utf-8;' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href     = url;
  a.download = `예측결과_${new Date().toISOString().slice(0,10)}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

function resetPredict() {
  clearPredFile();
}
