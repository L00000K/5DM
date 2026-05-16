import { initApiKeyModal } from './api-key.js';
import { initUploader } from './data-parser.js';
import { initTextInput } from './text-input.js';
import { runAIAnalysis, interpretGeology, inferStratOrderFromData, inferUnitParameters, generateSemanticModel, oracleRefinement, generateReportNarrative } from './claude-client.js';
import { exportConfig, importConfig } from './project-config.js';
import { buildVoxelGrid } from './interpolator.js';
import { initScene } from './scene.js';
import { initLayerControls } from './layer-controls.js';
import { initExporter } from './exporter.js';
import { parseConstraints, applyConstraints, constraintSummary } from './constraints.js';
import { parseGeoMap } from './geo-map.js';
import { FenceSection } from './fence-section.js';
import { IsopachMap  } from './isopach.js';
import { ModelReport } from './report.js';
import { PlanView } from './plan-view.js';
import { renderPropertiesTable, applyBS5930Colors } from './properties.js';
import { saveSession, loadSession, hasSavedSession } from './session.js';
import { calculateSettlement, renderSettlementResults } from './settlement.js';
import { calculateBearingCapacity, renderBearingResults } from './bearing.js';
import { calculatePileCapacity, renderPileResults } from './pile.js';
import { parseLabCSV } from './lab-import.js';
import { assessRisk, renderRiskReport } from './risk-engine.js';
import { BHLogView } from './bh-log-view.js';
import { CPTLogView } from './cpt-log-view.js';
import { parseCPT } from './data-parser.js';

// ── Global application state ──────────────────────────────────────────────────
export const AppState = {
  apiKey: null,
  demoMode: false,
  cellSizeH: 1,
  cellSizeZ: 0.25,
  kNeighbors: 5,
  idwPower: 2,
  interpMethod: 'idw',
  rawBoreholes: [],
  geoUnits: [],
  classifiedBH: [],
  voxelGrid: null,
  scene: null,
  hiddenUnits: new Set(),
  certaintyThreshold: 0,
  parsedConstraints: [],
  topoPoints: null,
  fenceSection: null,
  isopachMap: null,
  planView: null,
  bhLogView: null,
  cptLogView: null,
  cptLogs: [],
  report: null,
  stratOrder: [],
  anisoAzimuth: 0,
  anisoRatio: 1,
  trendOrder: 1,
  semanticModel: null,
  semanticWeight: 0.3,
  niEpochs: 400,
  oracleEnabled: false,
  varRange: null,
  varSill: null,
  varNugget: null,
};

// ── Logging utility ────────────────────────────────────────────────────────────
export function log(msg, type = 'info') {
  const el  = document.createElement('div');
  el.className = `log-entry log-${type}`;
  const now = new Date();
  const ts  = now.toTimeString().slice(0, 8);
  el.innerHTML = `<span class="log-ts">${ts}</span>${escHtml(msg)}`;
  const logEl = document.getElementById('status-log');
  logEl.appendChild(el);
  logEl.scrollTop = logEl.scrollHeight;
}

export function analysisLog(header, body, type = 'ai') {
  const entry = document.createElement('div');
  entry.className = `analysis-entry entry-${type}`;
  entry.innerHTML = `
    <div class="entry-header">${escHtml(header)}</div>
    <div class="entry-body">${escHtml(body)}</div>`;
  const al = document.getElementById('analysis-log');
  al.appendChild(entry);
  al.scrollTop = al.scrollHeight;
}

function escHtml(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// ── Welcome overlay ────────────────────────────────────────────────────────────
function showWelcome() {
  document.getElementById('welcome-overlay')?.classList.remove('hidden');
}

function hideWelcome() {
  document.getElementById('welcome-overlay')?.classList.add('hidden');
}

// ── Tab switching ──────────────────────────────────────────────────────────────
function initTabs() {
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => switchTab(btn.dataset.tab));
  });
}

function switchTab(tabName) {
  document.querySelectorAll('.tab-btn').forEach(b =>
    b.classList.toggle('active', b.dataset.tab === tabName));
  document.querySelectorAll('.tab-content').forEach(c =>
    c.classList.toggle('active', c.id === `tab-${tabName}`));
}

// ── Cell size inputs ───────────────────────────────────────────────────────────
function initCellSizeInputs() {
  const inX = document.getElementById('cell-size-x');
  const inY = document.getElementById('cell-size-y');
  const inZ = document.getElementById('cell-size-z');

  inX?.addEventListener('input', () => {
    AppState.cellSizeH = parseFloat(inX.value) || 1;
    if (inY) inY.value = inX.value;
  });
  inY?.addEventListener('input', () => {
    AppState.cellSizeH = parseFloat(inY.value) || 1;
    if (inX) inX.value = inY.value;
  });
  inZ?.addEventListener('input', () => {
    AppState.cellSizeZ = parseFloat(inZ.value) || 0.25;
  });
}

function initInterpolationSettings() {
  const kSlider = document.getElementById('k-neighbors');
  const kVal    = document.getElementById('k-neighbors-val');
  const pSlider = document.getElementById('idw-power');
  const pVal    = document.getElementById('idw-power-val');

  kSlider?.addEventListener('input', () => {
    AppState.kNeighbors = parseInt(kSlider.value);
    if (kVal) kVal.textContent = kSlider.value;
  });
  pSlider?.addEventListener('input', () => {
    AppState.idwPower = parseFloat(pSlider.value);
    if (pVal) pVal.textContent = parseFloat(pSlider.value).toFixed(1);
  });

  document.querySelectorAll('input[name="interp-method"]').forEach(radio => {
    radio.addEventListener('change', () => {
      AppState.interpMethod = radio.value;
      const vPanel  = document.getElementById('variogram-panel');
      const ukPanel = document.getElementById('uk-trend-panel');
      const niPanel = document.getElementById('ni-panel');
      if (vPanel)  vPanel.style.display  = ['kriging', 'uk'].includes(radio.value) ? 'block' : 'none';
      if (ukPanel) ukPanel.style.display = radio.value === 'uk' ? 'block' : 'none';
      if (niPanel) niPanel.style.display = radio.value === 'neural-implicit' ? 'block' : 'none';
      if (['kriging', 'uk'].includes(radio.value) && AppState.classifiedBH.length) {
        _renderVariogram(AppState.classifiedBH);
      }
    });
  });

  const trendSel = document.getElementById('uk-trend-order');
  trendSel?.addEventListener('change', () => {
    AppState.trendOrder = parseInt(trendSel.value);
  });

  // Neural implicit controls
  document.getElementById('ni-epochs')?.addEventListener('change', e => {
    AppState.niEpochs = parseInt(e.target.value) || 400;
  });
  document.getElementById('ni-oracle-toggle')?.addEventListener('change', e => {
    AppState.oracleEnabled = e.target.checked;
  });

  const azInput   = document.getElementById('aniso-azimuth');
  const ratSlider = document.getElementById('aniso-ratio');
  const ratVal    = document.getElementById('aniso-ratio-val');
  azInput?.addEventListener('input', () => {
    AppState.anisoAzimuth = parseFloat(azInput.value) || 0;
  });
  ratSlider?.addEventListener('input', () => {
    AppState.anisoRatio = parseFloat(ratSlider.value) || 1;
    if (ratVal) ratVal.textContent = parseFloat(ratSlider.value).toFixed(1);
  });
}

// ── Spherical variogram auto-fitting ─────────────────────────────────────────
// Grid search over (nugget, partial_sill, range) minimising SSR to empirical γ(h)
function _fitSphericalVariogram(empirical, binCentres) {
  // empirical: array of gamma values, binCentres: corresponding h values
  const n = empirical.length;
  if (n < 3) return null;
  const sillEst = Math.max(...empirical);
  const maxH    = binCentres[n - 1];

  let bestSSR = Infinity, bestParams = null;
  // Grid: nugget ∈ [0, 0.4×sill], sill ∈ [0.5×sillEst, 1.2×sillEst], range ∈ [0.1×maxH, maxH]
  for (let nc = 0; nc <= 4; nc++) {
    const C0 = nc * 0.1 * sillEst;
    for (let sc = 5; sc <= 12; sc++) {
      const C1 = sc * 0.1 * sillEst - C0;
      if (C1 <= 0) continue;
      for (let rc = 1; rc <= 10; rc++) {
        const a = rc * 0.1 * maxH;
        let ssr = 0;
        for (let i = 0; i < n; i++) {
          const h = binCentres[i];
          const model = h >= a
            ? C0 + C1
            : C0 + C1 * (1.5 * h / a - 0.5 * Math.pow(h / a, 3));
          ssr += Math.pow(empirical[i] - model, 2);
        }
        if (ssr < bestSSR) { bestSSR = ssr; bestParams = { nugget: C0, partialSill: C1, range: a }; }
      }
    }
  }
  return bestParams;
}

// ── Empirical variogram computation and rendering ──────────────────────────────
function _renderVariogram(boreholes) {
  const canvas = document.getElementById('variogram-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.clientWidth || 240;
  const H = canvas.clientHeight || 100;
  canvas.width  = W;
  canvas.height = H;

  // Compute empirical variogram: γ(h) = 0.5 × mean[(u(x+h) - u(x))²]
  // Treat unit code as a numeric indicator (1 if same unit, 0 if different)
  const pts = [];
  boreholes.forEach(bh => {
    bh.layers?.forEach(l => {
      if (!l.unitCode) return;
      const mid = (l.top + l.base) / 2;
      pts.push({ x: bh.x, y: bh.y, z: bh.groundLevel - mid, code: l.unitCode });
    });
  });

  if (pts.length < 4) {
    ctx.fillStyle = '#556677';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Not enough data for variogram', W * 0.5, H * 0.5);
    return;
  }

  // Pair up all points, compute separation h and indicator variance
  const BINS   = 10;
  const maxH   = Math.max(...boreholes.map(b =>
    Math.hypot(b.x - boreholes[0].x, b.y - boreholes[0].y))) || 100;
  const binW   = maxH / BINS;
  const gammas = new Float64Array(BINS);
  const counts = new Float64Array(BINS);

  for (let i = 0; i < pts.length; i++) {
    for (let j = i + 1; j < pts.length; j++) {
      const h   = Math.hypot(pts[i].x - pts[j].x, pts[i].y - pts[j].y);
      const bin = Math.min(BINS - 1, Math.floor(h / binW));
      const diff = pts[i].code === pts[j].code ? 0 : 1;
      gammas[bin] += diff * diff;
      counts[bin]++;
    }
  }

  const binData = Array.from({ length: BINS }, (_, i) => ({
    h:     (i + 0.5) * binW,
    gamma: counts[i] > 0 ? gammas[i] / (2 * counts[i]) : null,
  })).filter(d => d.gamma !== null);

  if (!binData.length) return;
  const vals     = binData.map(d => d.gamma);
  const centres  = binData.map(d => d.h);
  const maxV     = Math.max(...vals);

  // Auto-fit spherical model and store in AppState
  const fitted = _fitSphericalVariogram(vals, centres);
  if (fitted) {
    AppState.varRange  = fitted.range;
    AppState.varSill   = fitted.nugget + fitted.partialSill;
    AppState.varNugget = fitted.nugget;
  }

  // Draw
  ctx.clearRect(0, 0, W, H);
  const PAD = { l: 28, r: 8, t: 8, b: 20 };
  const cW  = W - PAD.l - PAD.r;
  const cH  = H - PAD.t - PAD.b;
  const maxVPlot = Math.max(maxV, fitted ? fitted.nugget + fitted.partialSill : maxV) * 1.1;

  // Axes
  ctx.strokeStyle = '#3a4d62'; ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(PAD.l, PAD.t);
  ctx.lineTo(PAD.l, PAD.t + cH);
  ctx.lineTo(PAD.l + cW, PAD.t + cH);
  ctx.stroke();

  // Labels
  ctx.fillStyle = '#8a9bb0'; ctx.font = '8px monospace';
  ctx.textAlign = 'right';
  ctx.fillText((maxVPlot).toFixed(2), PAD.l - 2, PAD.t + 8);
  ctx.fillText('0', PAD.l - 2, PAD.t + cH);
  ctx.textAlign = 'center';
  ctx.fillText(`0`, PAD.l, PAD.t + cH + 12);
  ctx.fillText(`${maxH.toFixed(0)}m`, PAD.l + cW, PAD.t + cH + 12);
  ctx.fillText('γ(h)', PAD.l - 20, PAD.t + cH * 0.5);

  // Fitted spherical model (orange dashed)
  if (fitted) {
    const { nugget: C0, partialSill: C1, range: a } = fitted;
    ctx.strokeStyle = '#e8924a'; ctx.setLineDash([4, 2]); ctx.lineWidth = 1.2;
    ctx.beginPath();
    for (let px = 0; px <= cW; px++) {
      const h   = (px / cW) * maxH;
      const gam = h >= a ? C0 + C1 : C0 + C1 * (1.5 * h / a - 0.5 * Math.pow(h / a, 3));
      const py  = PAD.t + cH * (1 - gam / maxVPlot);
      if (px === 0) ctx.moveTo(PAD.l + px, py); else ctx.lineTo(PAD.l + px, py);
    }
    ctx.stroke();
    ctx.setLineDash([]);
    // Range marker
    const rangeX = PAD.l + (a / maxH) * cW;
    ctx.strokeStyle = '#e8924a'; ctx.lineWidth = 0.7; ctx.setLineDash([2, 2]);
    ctx.beginPath(); ctx.moveTo(rangeX, PAD.t); ctx.lineTo(rangeX, PAD.t + cH); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#e8924a'; ctx.font = '7px monospace'; ctx.textAlign = 'left';
    ctx.fillText(`r=${a.toFixed(0)}m`, rangeX + 2, PAD.t + 9);
  }

  // Empirical points
  ctx.fillStyle = '#5ab8e0';
  ctx.strokeStyle = '#5ab8e0'; ctx.lineWidth = 1.5;
  ctx.beginPath();
  binData.forEach(({ h, gamma }, i) => {
    const px = PAD.l + (h / maxH) * cW;
    const py = PAD.t + cH * (1 - gamma / maxVPlot);
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  });
  ctx.stroke();
  binData.forEach(({ h, gamma }) => {
    const px = PAD.l + (h / maxH) * cW;
    const py = PAD.t + cH * (1 - gamma / maxVPlot);
    ctx.beginPath(); ctx.arc(px, py, 2.5, 0, Math.PI * 2); ctx.fill();
  });

  // Summary text
  if (fitted) {
    ctx.fillStyle = '#8a9bb0'; ctx.font = '7px monospace'; ctx.textAlign = 'right';
    ctx.fillText(`C0=${fitted.nugget.toFixed(2)} C1=${fitted.partialSill.toFixed(2)}`, W - PAD.r, H - 4);
  }
}

// ── Collapsible sections ───────────────────────────────────────────────────────
function initCollapsibles() {
  [['topo-toggle', 'topo-section'], ['geomap-toggle', 'geomap-section'], ['cpt-toggle', 'cpt-section']].forEach(([tid, sid]) => {
    const toggle  = document.getElementById(tid);
    const section = document.getElementById(sid);
    if (!toggle || !section) return;
    toggle.addEventListener('click', () => {
      section.hidden = !section.hidden;
      const arrow = toggle.querySelector('.collapse-arrow');
      if (arrow) arrow.textContent = section.hidden ? '›' : '⌄';
    });
  });
}

// ── Topography import ──────────────────────────────────────────────────────────
function initTopoUpload() {
  const dropZone  = document.getElementById('drop-topo');
  const fileInput = document.getElementById('file-topo');
  const fileInfo  = document.getElementById('topo-file-info');
  if (!dropZone || !fileInput) return;

  dropZone.addEventListener('click', () => fileInput.click());
  dropZone.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') fileInput.click();
  });
  dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    const file = e.dataTransfer?.files[0];
    if (file) parseTopoFile(file, fileInfo);
  });
  fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) parseTopoFile(fileInput.files[0], fileInfo);
  });

  document.getElementById('show-topo')?.addEventListener('change', e => {
    if (AppState.scene) AppState.scene.toggleTopography(e.target.checked);
  });
}

function parseTopoFile(file, infoEl) {
  const reader = new FileReader();
  reader.onload = e => {
    const lines = e.target.result.split('\n').map(l => l.trim()).filter(Boolean);
    const points = [];
    for (const line of lines) {
      const parts = line.split(/[,\t ]+/);
      const x = parseFloat(parts[0]);
      const y = parseFloat(parts[1]);
      const z = parseFloat(parts[2]);
      if (!isNaN(x) && !isNaN(y) && !isNaN(z)) points.push({ x, y, z });
    }
    if (points.length < 3) {
      log('Topo file needs at least 3 valid X,Y,Z rows', 'warn');
      return;
    }
    AppState.topoPoints = points;
    infoEl.innerHTML = `<div class="file-item">
      <span class="file-name">${escHtml(file.name)}</span>
      <span class="file-size">${points.length} pts</span></div>`;
    if (AppState.scene) {
      AppState.scene.showTopography(points);
      log(`Topography loaded — ${points.length} points`, 'ok');
    }
  };
  reader.readAsText(file);
}

// ── Reset ──────────────────────────────────────────────────────────────────────
function initReset() {
  document.getElementById('btn-reset')?.addEventListener('click', () => {
    if (!confirm('Reset all data and model?')) return;
    AppState.rawBoreholes = [];
    AppState.geoUnits = [];
    AppState.classifiedBH = [];
    AppState.voxelGrid = null;
    AppState.hiddenUnits = new Set();
    AppState.certaintyThreshold = 0;
    AppState.parsedConstraints = [];
    AppState.topoPoints = null;

    document.getElementById('analysis-log').innerHTML =
      '<div class="log-entry log-info">Run AI Analysis to see results here.</div>';
    document.getElementById('unit-legend').innerHTML =
      '<p class="hint">Units appear after model is built</p>';
    document.getElementById('status-log').innerHTML =
      '<div class="log-entry log-info">Reset. Drop files or load a sample site.</div>';
    const cs = document.getElementById('constraints-summary');
    if (cs) cs.innerHTML = '';
    const tf = document.getElementById('topo-file-info');
    if (tf) tf.innerHTML = '';

    updateInfoPanel();
    setEnabled('btn-run-ai', false);
    setEnabled('btn-interpret-geology', false);
    setEnabled('btn-build-model', false);
    setEnabled('btn-apply-constraints', false);
    setEnabled('btn-export-gltf', false);
    setEnabled('btn-export-obj', false);
    setEnabled('btn-export-json', false);
    setEnabled('btn-export-vtk', false);
    setEnabled('btn-export-pointcloud', false);
    setEnabled('btn-export-stats', false);
    setEnabled('btn-export-bh-csv', false);
    setEnabled('btn-export-ags', false);
    setEnabled('btn-export-props', false);
    setEnabled('btn-auto-params', false);
    setEnabled('btn-isopach', false);
    setEnabled('btn-model-report', false);
    setEnabled('btn-ai-narrative', false);
    setEnabled('btn-validate-model', false);
    setEnabled('btn-assess-risk', false);
    setEnabled('btn-plan-view', false);
    setEnabled('btn-export-contacts', false);
    setEnabled('btn-export-surfaces', false);
    setEnabled('btn-param-apply', false);
    setEnabled('btn-param-reset', false);
    setEnabled('btn-build-isosurfaces', false);
    setEnabled('btn-semantic-model', false);
    updateStratColumn();
    if (AppState.scene) AppState.scene.clear();
    showWelcome();
    switchTab('data');
  });
}

// ── Helper: enable/disable buttons ────────────────────────────────────────────
export function setEnabled(id, enabled) {
  const el = document.getElementById(id);
  if (el) el.disabled = !enabled;
}

// ── Sample site loading ────────────────────────────────────────────────────────
async function loadDemoSite(demoName) {
  hideWelcome();
  log(`Loading ${demoName}…`, 'info');
  try {
    const res = await fetch(`./assets/${demoName}.json`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    AppState.rawBoreholes = data.boreholes.map(bh => ({
      id: bh.id,
      x: bh.x,
      y: bh.y,
      groundLevel: bh.ground_level,
      depth: bh.depth,
      layers: bh.layers.map(l => ({
        top: l.top,
        base: l.base,
        description: l.description,
        unitCode: l.unit_code,
        certainty: l.certainty ?? 0.9,
        sptN: l.spt_n ?? null,
      })),
      classified: true,
    }));

    AppState.geoUnits = data.geological_units.map(u => ({
      id: u.id,
      code: u.code,
      name: u.name,
      color: u.color,
      description: u.description,
    }));
    AppState.classifiedBH = AppState.rawBoreholes;
    AppState.demoMode = true;

    if (data.site?.constraints) {
      document.getElementById('constraints-text').value = data.site.constraints;
    }

    if (data.topography?.length) {
      AppState.topoPoints = data.topography;
      document.getElementById('show-topo').checked = true;
      log(`Topography loaded — ${data.topography.length} points.`, 'ok');
    }

    updateLegend();
    updateInfoPanel();
    updateBHTable();
    updateBHChart();
    updateStratColumn();
    updateBHUnitStats();
    setEnabled('btn-run-ai', true);
    setEnabled('btn-build-model', true);
    setEnabled('btn-export-bh-csv', true);
      setEnabled('btn-export-ags', true);
    setEnabled('btn-export-props', true);
    setEnabled('btn-auto-params', true);
    setEnabled('btn-interpret-geology', true);
    setEnabled('btn-semantic-model', true);
    log(`${data.site?.name ?? demoName} — ${AppState.rawBoreholes.length} boreholes loaded.`, 'ok');

    setTimeout(() => document.getElementById('btn-build-model').click(), 200);
  } catch (err) {
    log(`Demo load failed: ${err.message}`, 'error');
    showWelcome();
  }
}

// ── Constraints ────────────────────────────────────────────────────────────────
function initConstraints() {
  document.getElementById('btn-apply-constraints')?.addEventListener('click', () => {
    if (!AppState.voxelGrid) { log('Build the 3D model first.', 'warn'); return; }
    const text = document.getElementById('constraints-text').value;
    AppState.parsedConstraints = parseConstraints(text, AppState.geoUnits);
    const count = applyConstraints(AppState.voxelGrid, AppState.parsedConstraints, AppState.geoUnits);
    AppState.scene.buildVoxels(AppState.voxelGrid, AppState.geoUnits, AppState.classifiedBH);
    updateVolumeStats();
    updateUnitStats();
    refreshLegendVolumes();
    renderConstraintSummary(constraintSummary(AppState.parsedConstraints));
    log(`Constraints applied — ${count} voxels reassigned.`, 'ok');
  });

  document.querySelectorAll('.example-rule').forEach(el => {
    el.addEventListener('click', () => {
      const ta = document.getElementById('constraints-text');
      const rule = el.dataset.rule;
      const existing = ta.value.trim();
      ta.value = existing ? existing + '\n' + rule : rule;
    });
  });
}

function renderConstraintSummary(items) {
  const el = document.getElementById('constraints-summary');
  if (!el) return;
  el.innerHTML = '';
  if (!items.length) return;
  items.forEach(item => {
    const div = document.createElement('div');
    div.className = `constraint-item ${item.active ? 'constraint-active' : 'constraint-note'}`;
    div.innerHTML = `<span class="constraint-label">${escHtml(item.label)}</span>` +
                    `<span class="constraint-text">${escHtml(item.text)}</span>`;
    el.appendChild(div);
  });
}

// ── Run AI Analysis ────────────────────────────────────────────────────────────
function initRunAI() {
  document.getElementById('btn-run-ai')?.addEventListener('click', async () => {
    if (!AppState.rawBoreholes.length) { log('No borehole data loaded.', 'warn'); return; }
    switchTab('analysis');
    setEnabled('btn-run-ai', false);
    document.getElementById('analysis-log').innerHTML = '';
    try {
      const { units, classified } = await runAIAnalysis(
        AppState.rawBoreholes, AppState.apiKey, AppState.demoMode
      );
      AppState.geoUnits = units;
      AppState.classifiedBH = classified;
      setEnabled('btn-export-bh-csv', true);
      setEnabled('btn-export-ags', true);
      setEnabled('btn-export-props', true);
      setEnabled('btn-auto-params', true);
      updateLegend();
      updateBHTable();
      updateBHChart();
      updateBHUnitStats();
      AppState.bhLogView?.draw(classified.filter(b => !b.synthetic), units);
      log(`Analysis complete — ${units.length} units classified.`, 'ok');
      // Auto-fit variogram so kriging has sensible initial params
      _renderVariogram(classified);
      setEnabled('btn-build-model', true);
      setEnabled('btn-run-ai', true);
      setEnabled('btn-interpret-geology', true);
      setEnabled('btn-semantic-model', true);
      saveSession(AppState);
    } catch (err) {
      log(`AI analysis failed: ${err.message}`, 'error');
      analysisLog('Error', err.message, 'error');
      setEnabled('btn-run-ai', true);
    }
  });
}

// ── Build 3D Model ─────────────────────────────────────────────────────────────
function initBuildModel() {
  document.getElementById('btn-build-model')?.addEventListener('click', async () => {
    if (!AppState.classifiedBH.length) { log('Run AI analysis first.', 'warn'); return; }
    setEnabled('btn-build-model', false);
    log('Building voxel grid…', 'info');
    try {
      showBuildProgress(true);
      await new Promise(r => setTimeout(r, 0));
      const { order: _stratOrder } = AppState.classifiedBH.length
        ? inferStratOrderFromData(AppState.classifiedBH, AppState.geoUnits)
        : { order: [] };
      AppState.stratOrder = _stratOrder;
      if (_stratOrder.length) {
        log(`Stratigraphic order: ${_stratOrder.join(' → ')}`, 'info');
      }
      const siteHistory = document.getElementById('input-site-history')?.value ?? '';
      const unitDescs   = Array.from(document.querySelectorAll('#desc-list .desc-item'))
        .map(el => el.querySelector('.desc-text')?.textContent?.trim() ?? el.textContent.trim())
        .filter(Boolean);
      const apiKey = sessionStorage.getItem('anthropic_api_key') ?? '';

      AppState.voxelGrid = await buildVoxelGrid(
        AppState.classifiedBH, AppState.geoUnits, AppState.cellSizeH,
        { kNeighbors: AppState.kNeighbors, idwPower: AppState.idwPower,
          method: AppState.interpMethod, cellSizeZ: AppState.cellSizeZ,
          stratOrder: _stratOrder,
          anisoAzimuth:  AppState.anisoAzimuth,
          anisoRatio:    AppState.anisoRatio,
          trendOrder:    AppState.trendOrder,
          semanticModel: AppState.semanticModel,
          semanticWeight: AppState.semanticWeight ?? 0.3,
          siteHistory, unitDescriptions: unitDescs,
          niEpochs: AppState.niEpochs ?? 400,
          oracleApiKey: AppState.oracleEnabled && apiKey ? apiKey : null,
          oracleRefineFn: oracleRefinement,
          demoMode: !apiKey,
          varRange:  AppState.varRange,
          varSill:   AppState.varSill,
          varNugget: AppState.varNugget,
          onProgress: p => setBuildProgress(p) }
      );
      showBuildProgress(false);
      updateInfoPanel();
      AppState.scene.buildVoxels(AppState.voxelGrid, AppState.geoUnits, AppState.classifiedBH);
      updateVolumeStats();
      updateUnitStats();
      refreshLegendVolumes();
      log(`Model ready — ${AppState.voxelGrid.nx}×${AppState.voxelGrid.ny}×${AppState.voxelGrid.nz} grid.`, 'ok');
      setEnabled('btn-export-gltf', true);
      setEnabled('btn-export-obj', true);
      setEnabled('btn-export-json', true);
      setEnabled('btn-export-vtk', true);
      setEnabled('btn-export-pointcloud', true);
      setEnabled('btn-export-stats', true);
      setEnabled('btn-build-model', true);
      setEnabled('btn-apply-constraints', true);
      setEnabled('btn-isopach', true);
      setEnabled('btn-model-report', true);
      setEnabled('btn-ai-narrative', true);
      setEnabled('btn-validate-model', true);
      setEnabled('btn-assess-risk', true);
      setEnabled('btn-plan-view', true);
      setEnabled('btn-export-contacts', true);
      setEnabled('btn-export-surfaces', true);
      setEnabled('btn-param-apply', true);
      setEnabled('btn-param-reset', true);
      setEnabled('btn-build-isosurfaces', true);
      const fdPanel = document.getElementById('foundation-panel');
      if (fdPanel) fdPanel.style.display = 'block';
      // GWT interpolate button: only enable if BHs have gwtDepth data
      const hasGWT = AppState.classifiedBH.some(b => b.gwtDepth != null && !b.synthetic);
      setEnabled('btn-gwt-interpolate', hasGWT);
      AppState.report?.compute(AppState.voxelGrid, AppState.classifiedBH, AppState.geoUnits);
      AppState._origUnitIds = null; // invalidate topo clip cache after rebuild
      AppState._onTopoClipUpdate?.();
      updateStratColumn();

      // Auto-apply pre-set constraints
      const constraintText = document.getElementById('constraints-text').value.trim();
      if (constraintText && AppState.geoUnits.length) {
        AppState.parsedConstraints = parseConstraints(constraintText, AppState.geoUnits);
        const actionable = AppState.parsedConstraints.filter(r => r.type !== 'note');
        if (actionable.length) {
          const count = applyConstraints(AppState.voxelGrid, AppState.parsedConstraints, AppState.geoUnits);
          AppState.scene.buildVoxels(AppState.voxelGrid, AppState.geoUnits, AppState.classifiedBH);
          updateVolumeStats();
          renderConstraintSummary(constraintSummary(AppState.parsedConstraints));
          if (count > 0) log(`Auto-applied ${actionable.length} constraints — ${count} voxels adjusted.`, 'ok');
        }
      }

      if (AppState.topoPoints) AppState.scene.showTopography(AppState.topoPoints);

      // Auto-save session after successful model build
      saveSession(AppState);
    } catch (err) {
      showBuildProgress(false);
      log(`Build failed: ${err.message}`, 'error');
      console.error(err);
      setEnabled('btn-build-model', true);
    }
  });
}

// ── Geological interpretation (AI semantic layer) ─────────────────────────────
function initInterpretGeology() {
  const btn = document.getElementById('btn-interpret-geology');
  if (!btn) return;

  btn.addEventListener('click', async () => {
    const siteHistory  = document.getElementById('input-site-history')?.value ?? '';
    const unitDescs    = Array.from(document.querySelectorAll('#desc-list .desc-item'))
      .map(el => el.textContent.trim()).join('\n');

    if (!AppState.geoUnits.length) { log('Load borehole data and run AI Analysis first.', 'warn'); return; }

    btn.disabled = true;
    log('Requesting geological interpretation from Claude…', 'info');
    switchTab('analysis');

    try {
      const result = await interpretGeology(
        siteHistory, unitDescs, AppState.geoUnits,
        AppState.apiKey, AppState.demoMode
      );

      // Show interpretation in analysis log
      analysisLog('Geological Interpretation',
        result.interpretation_summary + '\n\n' +
        `Stratigraphic order (top to bottom): ${(result.stratigraphic_order ?? []).join(' › ')}\n` +
        (result.interpolation_advice ? `\nAdvice: ${result.interpolation_advice}` : ''), 'ai');

      // Show hazards
      if (result.hazards?.length) {
        analysisLog('Geohazards',
          result.hazards.map(h => `⚠ ${h.type.toUpperCase()}: ${h.description}`).join('\n'), 'warn');
      }

      // Auto-populate constraints textarea
      if (result.constraints?.length) {
        const ct = document.getElementById('constraints-text');
        if (ct) {
          const existing = ct.value.trim();
          const newRules = result.constraints.join('\n');
          ct.value = existing ? existing + '\n' + newRules : newRules;
        }
        analysisLog('Generated Constraints',
          result.constraints.map(c => `• ${c}`).join('\n') + '\n\nAdded to Rules tab.', 'ok');
      }

      // Apply colour suggestions if provided
      if (result.colour_suggestions) {
        let coloured = 0;
        for (const [code, hex] of Object.entries(result.colour_suggestions)) {
          const unit = AppState.geoUnits.find(u => u.code === code);
          if (unit && /^#[0-9a-f]{6}$/i.test(hex)) { unit.color = hex; coloured++; }
        }
        if (coloured) { updateLegend(); updateStratColumn(); }
      }

      // Infer stratigraphic order from data and show
      if (AppState.classifiedBH.length) {
        const { order, pairs } = inferStratOrderFromData(AppState.classifiedBH, AppState.geoUnits);
        const topPairs = pairs.slice(0, 6).map(p => `${p.above} → ${p.below} (${p.count}×)`).join(', ');
        if (topPairs) {
          analysisLog('Data-Derived Stratigraphy',
            `Observed sequence (top→base): ${order.join(' › ')}\nTop observed pairs: ${topPairs}`, 'ai');
        }
      }

      log('Geological interpretation complete.', 'ok');
    } catch (err) {
      log(`Interpretation failed: ${err.message}`, 'error');
      analysisLog('Error', err.message, 'error');
    } finally {
      btn.disabled = false;
    }
  });
}

// ── Project config file export / import ───────────────────────────────────────
function initProjectConfig() {
  // Export
  document.getElementById('btn-export-config')?.addEventListener('click', () => {
    if (!AppState.geoUnits.length && !AppState.classifiedBH.length) {
      log('Nothing to export — load data first.', 'warn'); return;
    }
    const config = exportConfig(AppState);
    const blob   = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' });
    const url    = URL.createObjectURL(blob);
    const a      = document.createElement('a');
    a.href = url;
    a.download = `geomodel-project-${new Date().toISOString().slice(0, 10)}.geomodel`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 5000);
    log('Project config exported.', 'ok');
  });

  // Import
  const fileInput = document.getElementById('file-import-config');
  document.getElementById('btn-import-config')?.addEventListener('click', () => fileInput?.click());

  fileInput?.addEventListener('change', async () => {
    const file = fileInput.files[0];
    if (!file) return;
    try {
      const text = await file.text();
      const data = JSON.parse(text);
      const cfg  = importConfig(data);

      AppState.geoUnits     = cfg.geoUnits;
      AppState.classifiedBH = cfg.classifiedBH;
      AppState.rawBoreholes = cfg.classifiedBH;
      AppState.cellSizeH    = cfg.cellSizeH;
      AppState.cellSizeZ    = cfg.cellSizeZ;
      AppState.kNeighbors   = cfg.kNeighbors;
      AppState.idwPower     = cfg.idwPower;
      AppState.interpMethod  = cfg.interpMethod;
      AppState.anisoAzimuth  = cfg.anisoAzimuth  ?? 0;
      AppState.anisoRatio    = cfg.anisoRatio    ?? 1;
      const azEl = document.getElementById('aniso-azimuth');
      if (azEl) azEl.value = AppState.anisoAzimuth;
      const arEl = document.getElementById('aniso-ratio');
      if (arEl) { arEl.value = AppState.anisoRatio; const arv = document.getElementById('aniso-ratio-val'); if (arv) arv.textContent = AppState.anisoRatio.toFixed(1); }

      const ct = document.getElementById('constraints-text');
      if (ct && cfg.constraints) ct.value = cfg.constraints;

      const sh = document.getElementById('input-site-history');
      if (sh && cfg.siteHistory) sh.value = cfg.siteHistory;

      const gwt = document.getElementById('gwt-elevation');
      if (gwt && cfg.gwtElevation != null) {
        gwt.value = cfg.gwtElevation;
        AppState.scene?.setGroundwaterTable(cfg.gwtElevation);
      }

      updateLegend();
      updateInfoPanel();
      updateBHTable();
      updateBHChart();
      updateStratColumn();
      updateBHUnitStats();
      renderPropertiesTable(AppState.geoUnits, () => updateLegend());
      AppState.bhLogView?.draw(AppState.classifiedBH.filter(b => !b.synthetic), AppState.geoUnits);
      setEnabled('btn-run-ai', true);
      setEnabled('btn-build-model', AppState.classifiedBH.length > 0);
      setEnabled('btn-export-bh-csv', AppState.classifiedBH.length > 0);
      setEnabled('btn-export-props', AppState.geoUnits.length > 0);
      setEnabled('btn-auto-params', AppState.geoUnits.length > 0);
      hideWelcome();
      log(`Project loaded: ${AppState.geoUnits.length} units, ${AppState.classifiedBH.length} BH. Click "Build 3D Model".`, 'ok');
      switchTab('data');
      fileInput.value = '';
      setTimeout(() => {
        if (AppState.classifiedBH.length) document.getElementById('btn-build-model')?.click();
      }, 100);
    } catch (err) {
      log(`Config import failed: ${err.message}`, 'error');
    }
  });
}

// ── Settlement estimator ──────────────────────────────────────────────────────
function initSettlement() {
  const btn  = document.getElementById('btn-calc-settlement');
  const res  = document.getElementById('settlement-results');
  if (!btn || !res) return;

  btn.addEventListener('click', () => {
    const grid = AppState.voxelGrid;
    if (!grid) { log('Build model first.', 'warn'); return; }
    const foundElev = parseFloat(document.getElementById('sett-found-level')?.value ?? '0');
    const loadKPa   = parseFloat(document.getElementById('sett-load')?.value ?? '50');
    if (isNaN(foundElev) || isNaN(loadKPa)) { log('Enter valid foundation level and load.', 'warn'); return; }
    const result = calculateSettlement(grid, AppState.geoUnits, foundElev, loadKPa);
    renderSettlementResults(result, res);
    if (result) {
      const hasCalc = result.layers.some(l => l.settlement !== null);
      log(`Settlement: ${hasCalc ? result.total.toFixed(1) + ' mm total' : 'set Cc and e0 in Props tab'}.`, hasCalc ? 'ok' : 'warn');
    }
  });
}

// ── Bearing capacity calculator ───────────────────────────────────────────────
function initBearingCapacity() {
  const btn = document.getElementById('btn-calc-bearing');
  const res = document.getElementById('bearing-results');
  if (!btn || !res) return;

  btn.addEventListener('click', () => {
    const grid = AppState.voxelGrid;
    if (!grid) { log('Build model first.', 'warn'); return; }
    const foundElev = parseFloat(document.getElementById('bear-found-level')?.value ?? '0');
    const B  = parseFloat(document.getElementById('bear-width')?.value ?? '2');
    const L  = parseFloat(document.getElementById('bear-length')?.value ?? '2');
    const z  = parseFloat(document.getElementById('bear-depth')?.value ?? '1');
    const FS = parseFloat(document.getElementById('bear-fs')?.value ?? '3');
    if ([foundElev, B, L, z, FS].some(isNaN)) {
      log('Enter valid foundation parameters.', 'warn'); return;
    }
    const result = calculateBearingCapacity(grid, AppState.geoUnits, foundElev, B, L, z, FS);
    renderBearingResults(result, res);
    if (result && !result.error) {
      const qa = result.undrained?.qa ?? result.drained?.qa;
      log(`Bearing capacity: q_allow = ${qa} kPa (FS ${FS}) — unit: ${result.unit.code}`, 'ok');
    }
  });
}

// ── Pile capacity calculator ──────────────────────────────────────────────────
function initPileCapacity() {
  const btn = document.getElementById('btn-calc-pile');
  const res = document.getElementById('pile-results');
  if (!btn || !res) return;

  btn.addEventListener('click', () => {
    const grid = AppState.voxelGrid;
    if (!grid) { log('Build model first.', 'warn'); return; }
    const headElev = parseFloat(document.getElementById('pile-head-level')?.value ?? '');
    const toeElev  = parseFloat(document.getElementById('pile-toe-level')?.value ?? '');
    const D        = parseFloat(document.getElementById('pile-diameter')?.value ?? '0.45');
    const FS       = parseFloat(document.getElementById('pile-fs')?.value ?? '2.5');
    if ([headElev, toeElev, D, FS].some(isNaN)) {
      log('Enter valid pile parameters.', 'warn'); return;
    }
    const result = calculatePileCapacity(grid, AppState.geoUnits, headElev, toeElev, D, FS);
    renderPileResults(result, res);
    if (result && !result.error) {
      log(`Pile capacity: Q_ult=${result.Qult} kN · Q_a=${result.Qa} kN (FS ${FS})`, 'ok');
    }
  });
}

// ── BS5930 colour presets ──────────────────────────────────────────────────────
function initColorPresets() {
  document.getElementById('btn-bs5930-colors')?.addEventListener('click', () => {
    if (!AppState.geoUnits.length) { log('Load data first.', 'warn'); return; }
    const n = applyBS5930Colors(AppState.geoUnits);
    updateLegend();
    renderPropertiesTable(AppState.geoUnits, () => updateLegend());
    if (AppState.scene && AppState.voxelGrid) {
      AppState.scene.buildVoxels(AppState.voxelGrid, AppState.geoUnits, AppState.classifiedBH);
      updateVolumeStats();
      updateUnitStats();
      updateStratColumn();
    }
    log(`BS5930 colours applied — ${n} unit${n !== 1 ? 's' : ''} matched.`, n > 0 ? 'ok' : 'warn');
  });
}

// ── Auto-infer geotechnical parameters via Claude ─────────────────────────────
function initAutoParams() {
  document.getElementById('btn-auto-params')?.addEventListener('click', async () => {
    if (!AppState.geoUnits.length) { log('Load data first.', 'warn'); return; }
    const btn = document.getElementById('btn-auto-params');
    btn.disabled = true;
    btn.textContent = '⏳ Inferring…';
    const demoMode = !AppState.apiKey;
    log(`Auto-inferring parameters for ${AppState.geoUnits.length} unit(s)${demoMode ? ' (demo)' : ' via Claude'}…`, 'info');

    let updated = 0;
    for (const unit of AppState.geoUnits) {
      try {
        const p = await inferUnitParameters(unit, AppState.apiKey, demoMode);
        if (!unit.params) unit.params = {};
        // Map inferred fields → unit.params, only filling gaps (don't overwrite lab data)
        const MAP = {
          gamma_kNm3: 'gamma', cu_kPa: 'cu', phi_deg: 'phi',
          cprime_kPa: 'cprime', E_MPa: 'E', Cc: 'Cc', e0: 'e0', N_spt: 'N_spt',
        };
        for (const [src, dst] of Object.entries(MAP)) {
          if (p[src] != null && unit.params[dst] == null) {
            unit.params[dst] = p[src];
          }
        }
        if (p.notes) unit._autoParamNotes = p.notes;
        updated++;
      } catch (err) {
        log(`Auto-params failed for ${unit.code}: ${err.message}`, 'warn');
      }
    }

    renderPropertiesTable(AppState.geoUnits, () => updateLegend());
    btn.disabled = false;
    btn.textContent = '🧠 Auto Params';
    log(`Auto-params: ${updated}/${AppState.geoUnits.length} unit(s) updated.`, 'ok');
  });
}

// ── Topography clipping ───────────────────────────────────────────────────────
function initTopoClip() {
  const chk = document.getElementById('topo-clip');
  const row = document.getElementById('topo-clip-row');
  if (!chk || !row) return;

  const apply = () => {
    const grid = AppState.voxelGrid;
    const topo = AppState.topoPoints;
    if (!grid) return;

    if (chk.checked && topo?.length) {
      if (!AppState._origUnitIds) {
        AppState._origUnitIds = new Uint8Array(grid.unitIds);
      }
      const { nx, ny, nz, cellSize: cs, cellHeight: ch, origin: O } = grid;
      const newIds = new Uint8Array(AppState._origUnitIds);
      for (let iy = 0; iy < ny; iy++) {
        for (let ix = 0; ix < nx; ix++) {
          const wx = O.x + (ix + 0.5) * cs;
          const wz = O.z + (iy + 0.5) * cs;
          let minD = Infinity, groundY = O.y + nz * ch;
          for (const p of topo) {
            const d = (p.x - wx) ** 2 + (p.y - wz) ** 2;
            if (d < minD) { minD = d; groundY = p.z; }
          }
          for (let iz = 0; iz < nz; iz++) {
            const voxMid = O.y + (iz + 0.5) * ch;
            if (voxMid > groundY) newIds[ix + iy * nx + iz * nx * ny] = 0;
          }
        }
      }
      grid.unitIds = newIds;
    } else if (!chk.checked && AppState._origUnitIds) {
      grid.unitIds = new Uint8Array(AppState._origUnitIds);
    }

    if (AppState.scene && AppState.voxelGrid) {
      AppState.scene.buildVoxels(grid, AppState.geoUnits, AppState.classifiedBH);
      updateVolumeStats();
    }
  };

  chk.addEventListener('change', apply);

  // Disable if no topo loaded
  const updateTopoClipState = () => {
    const enabled = !!(AppState.topoPoints?.length && AppState.voxelGrid);
    if (row) row.style.opacity = enabled ? '1' : '0.45';
    if (chk) chk.disabled = !enabled;
  };
  window.addEventListener('geomodel:data-loaded', updateTopoClipState);
  // Call after model build too
  AppState._onTopoClipUpdate = updateTopoClipState;
}

// ── Cursor world coordinates ──────────────────────────────────────────────────
function initCursorCoords() {
  const el = document.getElementById('cursor-coords');
  if (!el) return;

  const canvas = document.getElementById('three-canvas');
  if (!canvas) return;

  canvas.addEventListener('mousemove', e => {
    const scene = AppState.scene;
    if (!scene?._modelBounds?.grid) { el.hidden = true; return; }
    const rect = canvas.getBoundingClientRect();
    const pt   = scene._canvasToWorld(e.clientX - rect.left, e.clientY - rect.top);
    if (pt) {
      el.hidden = false;
      el.textContent = `X ${pt.x.toFixed(1)} · Y ${pt.y.toFixed(1)} · Z ${pt.z.toFixed(1)} m`;
    } else {
      el.hidden = true;
    }
  });
  canvas.addEventListener('mouseleave', () => { el.hidden = true; });
}

// ── Keyboard shortcuts modal ──────────────────────────────────────────────────
function initShortcutsModal() {
  const btn   = document.getElementById('btn-shortcuts');
  const modal = document.getElementById('modal-shortcuts');
  const close = document.getElementById('btn-shortcuts-close');
  if (!btn || !modal) return;
  const show = () => { modal.hidden = false; close?.focus(); };
  const hide = () => { modal.hidden = true; };
  btn.addEventListener('click', show);
  close?.addEventListener('click', hide);
  modal.querySelector('.modal-backdrop')?.addEventListener('click', hide);
  window.addEventListener('keydown', e => {
    if (e.key === '?' && !modal.hidden) { hide(); return; }
    if (e.key === '?' && !e.ctrlKey && !e.metaKey &&
        document.activeElement?.tagName !== 'INPUT' &&
        document.activeElement?.tagName !== 'TEXTAREA') {
      show();
    }
  });
}

// ── Unit Editor Modal ─────────────────────────────────────────────────────────
function initUnitEditor() {
  const modal   = document.getElementById('modal-unit-editor');
  const list    = document.getElementById('unit-editor-list');
  const btnOpen = document.getElementById('btn-edit-units');
  const btnAdd  = document.getElementById('btn-add-unit');
  const btnApply= document.getElementById('btn-unit-editor-apply');
  const btnClose= document.getElementById('btn-unit-editor-close');
  if (!modal) return;

  // Track draft units (copy so edits are cancelable)
  let draft = [];

  function _nextId() {
    return (Math.max(0, ...draft.map(u => u.id)) + 1);
  }

  function _randomColor() {
    const hue = Math.floor(Math.random() * 360);
    return `hsl(${hue},45%,45%)`;
  }

  function _hslToHex(hsl) {
    // Accepts #xxx or hsl(...) or any CSS color
    if (hsl.startsWith('#')) return hsl;
    const el = document.createElement('canvas');
    el.width = el.height = 1;
    const ctx = el.getContext('2d');
    ctx.fillStyle = hsl;
    ctx.fillRect(0, 0, 1, 1);
    const [r, g, b] = ctx.getImageData(0, 0, 1, 1).data;
    return `#${[r, g, b].map(x => x.toString(16).padStart(2, '0')).join('')}`;
  }

  function buildRow(unit) {
    const row = document.createElement('div');
    row.className = 'unit-editor-row';
    row.dataset.uid = unit.id;
    row.innerHTML = `
      <input type="color" class="ue-color" value="${_hslToHex(unit.color ?? '#888888')}" title="Unit colour">
      <input type="text"  class="ue-code input-sm" value="${unit.code ?? ''}" placeholder="Code" maxlength="8" title="Short code">
      <input type="text"  class="ue-name input-sm" value="${unit.name ?? ''}" placeholder="Name" style="flex:1" title="Full name">
      <input type="text"  class="ue-desc input-sm" value="${unit.description ?? ''}" placeholder="Description (optional)" style="flex:2" title="Description">
      <button class="btn-ghost btn-sm ue-del" title="Delete unit">✕</button>
    `;
    row.querySelector('.ue-del').addEventListener('click', () => {
      draft = draft.filter(u => u.id !== unit.id);
      row.remove();
    });
    return row;
  }

  function open() {
    // Deep copy so cancel doesn't corrupt state
    draft = AppState.geoUnits.map(u => ({ ...u, params: { ...(u.params ?? {}) } }));
    list.innerHTML = '';
    if (!draft.length) {
      // Seed with default units if project is empty
      draft = [
        { id: 1, code: 'MG',  name: 'Made Ground',            color: '#8B6914', description: 'Variable fill material' },
        { id: 2, code: 'RTD', name: 'River Terrace Deposits',  color: '#D4A843', description: 'Gravel with sand' },
        { id: 3, code: 'LC',  name: 'London Clay',             color: '#4A6080', description: 'Stiff fissured clay' },
        { id: 4, code: 'CH',  name: 'Chalk',                   color: '#EDE8D8', description: 'Soft to medium hard chalk' },
      ];
    }
    draft.forEach(u => list.appendChild(buildRow(u)));
    modal.hidden = false;
  }

  function apply() {
    // Read current form values back into draft
    list.querySelectorAll('.unit-editor-row').forEach(row => {
      const uid  = parseInt(row.dataset.uid, 10);
      const unit = draft.find(u => u.id === uid);
      if (!unit) return;
      unit.color       = row.querySelector('.ue-color').value;
      unit.code        = row.querySelector('.ue-code').value.trim().toUpperCase();
      unit.name        = row.querySelector('.ue-name').value.trim();
      unit.description = row.querySelector('.ue-desc').value.trim();
    });

    // Remove blanks and duplicates
    const seen = new Set();
    draft = draft.filter(u => {
      if (!u.code || seen.has(u.code)) return false;
      seen.add(u.code);
      return true;
    });

    AppState.geoUnits = draft;

    // Refresh all dependant UI
    updateLegend();
    renderPropertiesTable(AppState.geoUnits, () => updateLegend());
    setEnabled('btn-run-ai', AppState.geoUnits.length > 0 && AppState.classifiedBH.length === 0);
    setEnabled('btn-auto-params', AppState.geoUnits.length > 0);
    setEnabled('btn-export-props', AppState.geoUnits.length > 0);
    log(`Unit definitions updated — ${AppState.geoUnits.length} unit(s).`, 'ok');
    modal.hidden = true;
  }

  btnOpen?.addEventListener('click', open);
  btnAdd?.addEventListener('click', () => {
    const unit = { id: _nextId(), code: '', name: '', color: _randomColor(), description: '' };
    draft.push(unit);
    const row = buildRow(unit);
    list.appendChild(row);
    row.querySelector('.ue-code')?.focus();
  });
  btnApply?.addEventListener('click', apply);
  btnClose?.addEventListener('click', () => { modal.hidden = true; });
  modal.querySelector('.modal-backdrop')?.addEventListener('click', () => { modal.hidden = true; });

  // Reclassify: replace all BH layer unitCodes from → to
  document.getElementById('btn-reclassify')?.addEventListener('click', () => {
    const fromCode = document.getElementById('ue-from-code')?.value.trim().toUpperCase();
    const toCode   = document.getElementById('ue-to-code')?.value.trim().toUpperCase();
    const resultEl = document.getElementById('reclassify-result');
    if (!fromCode || !toCode) { if (resultEl) resultEl.textContent = 'Enter both codes.'; return; }
    if (fromCode === toCode)  { if (resultEl) resultEl.textContent = 'Codes are the same.'; return; }

    let count = 0;
    for (const bh of AppState.classifiedBH) {
      for (const layer of bh.layers) {
        if (layer.unitCode === fromCode) { layer.unitCode = toCode; count++; }
      }
    }

    // Remove the 'from' unit if no layers remain using it and it's not the 'to' unit
    const stillUsed = AppState.classifiedBH.some(bh => bh.layers.some(l => l.unitCode === fromCode));
    if (!stillUsed) {
      AppState.geoUnits = AppState.geoUnits.filter(u => u.code !== fromCode);
    }

    updateLegend();
    updateBHTable();
    renderPropertiesTable(AppState.geoUnits, () => updateLegend());
    if (resultEl) resultEl.textContent = `${count} layer(s) reclassified`;
    log(`Reclassified ${count} layer(s): ${fromCode} → ${toCode}`, count > 0 ? 'ok' : 'warn');
  });
}

// ── Log sub-tab switcher (BH / CPT) ───────────────────────────────────────────
function initLogSubTabs() {
  document.querySelectorAll('.log-sub-tab').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.log-sub-tab').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const key = btn.dataset.logTab;
      document.getElementById('log-sub-bh').hidden  = (key !== 'bh');
      document.getElementById('log-sub-cpt').hidden = (key !== 'cpt');
    });
  });
}

// ── CPT data import ───────────────────────────────────────────────────────────
function initCPTImport() {
  AppState.cptLogView = new CPTLogView();
  const drop = document.getElementById('drop-cpt');
  const file = document.getElementById('file-cpt');
  const info = document.getElementById('cpt-file-info');

  const process = async files => {
    let all = [];
    for (const f of files) {
      try {
        const text = await new Promise((res, rej) => {
          const r = new FileReader();
          r.onload = e => res(e.target.result);
          r.onerror = () => rej(new Error(`Cannot read ${f.name}`));
          r.readAsText(f);
        });
        const logs = parseCPT(text);
        if (logs.length) {
          all = all.concat(logs);
          log(`${f.name}: ${logs.length} CPT log(s) parsed`, 'ok');
        } else {
          log(`${f.name}: no CPT data found`, 'warn');
        }
      } catch (err) {
        log(`CPT parse error: ${err.message}`, 'error');
      }
    }
    if (all.length) {
      AppState.cptLogs = all;
      if (info) info.textContent = `${all.length} CPT log(s) loaded`;
      AppState.cptLogView.draw(all);
    }
  };

  drop?.addEventListener('click', () => file?.click());
  drop?.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('drag-over'); });
  drop?.addEventListener('dragleave', () => drop.classList.remove('drag-over'));
  drop?.addEventListener('drop', e => {
    e.preventDefault(); drop.classList.remove('drag-over');
    process(Array.from(e.dataTransfer?.files ?? []));
  });
  file?.addEventListener('change', () => { if (file.files.length) process(Array.from(file.files)); });
}

// ── Borehole Log Strip View ────────────────────────────────────────────────────
function initBHLogView() {
  AppState.bhLogView = new BHLogView();

  // Redraw when user switches to the Logs tab and data is available
  document.querySelectorAll('.tab-btn[data-tab="logs"]').forEach(btn => {
    btn.addEventListener('click', () => {
      const bhs = AppState.classifiedBH.filter(b => !b.synthetic);
      if (bhs.length && AppState.geoUnits.length) {
        AppState.bhLogView.draw(bhs, AppState.geoUnits);
      }
    });
  });
}

// ── Groundwater table ─────────────────────────────────────────────────────────
function initGWT() {
  const input = document.getElementById('gwt-elevation');
  const clear = document.getElementById('btn-gwt-clear');

  input?.addEventListener('change', () => {
    if (!AppState.scene) return;
    const val = parseFloat(input.value);
    AppState.scene.setGroundwaterTable(isNaN(val) ? null : val);
    if (!isNaN(val)) log(`GWT set at ${val.toFixed(1)} mAOD`, 'ok');
  });
  clear?.addEventListener('click', () => {
    if (input) input.value = '';
    AppState.scene?.setGroundwaterTable(null);
  });

  let gwtSurfaceShown = false;
  const interpBtn = document.getElementById('btn-gwt-interpolate');
  interpBtn?.addEventListener('click', () => {
    if (!AppState.scene || !AppState.voxelGrid) {
      log('Build the 3D model first.', 'warn'); return;
    }
    const bhsWithGWT = AppState.classifiedBH.filter(b => b.gwtDepth != null && !b.synthetic);
    if (!bhsWithGWT.length) {
      log('No per-borehole GWT depths found. Add a "gwt_depth" column to your CSV or include an AGS WSTB group.', 'warn');
      return;
    }
    if (gwtSurfaceShown) {
      AppState.scene.toggleInterpolatedGWT(false);
      AppState.scene._clearInterpGWT?.();
      gwtSurfaceShown = false;
      interpBtn.textContent = '≈ Interpolate GWT Surface from BH Data';
      log('GWT surface removed.', 'info');
    } else {
      AppState.scene.showInterpolatedGWT(bhsWithGWT, AppState.voxelGrid);
      gwtSurfaceShown = true;
      interpBtn.textContent = '✕ Remove GWT Surface';
      const meanElev = (bhsWithGWT.reduce((s,b)=>(s+(b.groundLevel??0)-b.gwtDepth),0)/bhsWithGWT.length).toFixed(1);
      log(`GWT surface interpolated from ${bhsWithGWT.length} BHs · mean elevation ≈ ${meanElev} mAOD.`, 'ok');
    }
  });

  // Update GWT BH count badge whenever data changes
  window.addEventListener('geomodel:data-ready', _updateGWTCount);
  function _updateGWTCount() {
    const el = document.getElementById('gwt-bh-count');
    if (!el) return;
    const n = AppState.classifiedBH.filter(b => b.gwtDepth != null && !b.synthetic).length;
    el.textContent = n > 0 ? `${n} BH${n>1?'s':''} with GWT data` : '';
    if (interpBtn) interpBtn.disabled = !n || !AppState.voxelGrid;
  }
}

// ── Camera preset views ────────────────────────────────────────────────────────
function initCameraPresets() {
  document.querySelectorAll('.cam-preset-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      if (!AppState.scene) return;
      AppState.scene.setCameraView(btn.dataset.preset);
    });
  });
}

// ── Session save / load ────────────────────────────────────────────────────────
function initSession() {
  const saveBtn = document.getElementById('btn-save-session');
  const loadBtn = document.getElementById('btn-load-session');

  saveBtn?.addEventListener('click', () => {
    if (!AppState.geoUnits.length && !AppState.classifiedBH.length) {
      log('Nothing to save.', 'warn'); return;
    }
    const ok = saveSession(AppState);
    log(ok ? 'Session saved (browser storage).' : 'Save failed — storage may be full.', ok ? 'ok' : 'error');
  });

  loadBtn?.addEventListener('click', async () => {
    if (!hasSavedSession()) { log('No saved session found.', 'warn'); return; }
    const data = loadSession();
    if (!data) { log('Could not read saved session.', 'error'); return; }

    AppState.geoUnits    = data.geoUnits ?? [];
    AppState.classifiedBH = data.classifiedBH ?? [];
    AppState.rawBoreholes = data.classifiedBH ?? [];
    AppState.cellSizeH   = data.cellSizeH ?? 1;
    AppState.cellSizeZ   = data.cellSizeZ ?? 0.25;
    AppState.kNeighbors  = data.kNeighbors ?? 5;
    AppState.idwPower    = data.idwPower ?? 2;
    AppState.interpMethod  = data.interpMethod ?? 'idw';
    AppState.anisoAzimuth  = data.anisoAzimuth  ?? 0;
    AppState.anisoRatio    = data.anisoRatio    ?? 1;
    const azEl2 = document.getElementById('aniso-azimuth');
    if (azEl2) azEl2.value = AppState.anisoAzimuth;
    const arEl2 = document.getElementById('aniso-ratio');
    if (arEl2) { arEl2.value = AppState.anisoRatio; const arv2 = document.getElementById('aniso-ratio-val'); if (arv2) arv2.textContent = AppState.anisoRatio.toFixed(1); }

    const ct = document.getElementById('constraints-text');
    if (ct && data.constraintsText) ct.value = data.constraintsText;

    updateLegend();
    updateInfoPanel();
    updateBHTable();
    updateBHChart();
    updateStratColumn();
    setEnabled('btn-run-ai', AppState.classifiedBH.length > 0);
    setEnabled('btn-build-model', AppState.classifiedBH.length > 0);
    setEnabled('btn-export-bh-csv', true);
      setEnabled('btn-export-ags', true);
    setEnabled('btn-export-props', true);
    setEnabled('btn-auto-params', true);
    hideWelcome();
    log(`Session restored — ${AppState.classifiedBH.length} BH, ${AppState.geoUnits.length} units. Click "Build 3D Model" to regenerate.`, 'ok');
    switchTab('data');

    setTimeout(() => {
      if (AppState.classifiedBH.length) document.getElementById('btn-build-model')?.click();
    }, 100);
  });

  if (hasSavedSession()) {
    log('A saved session is available. Click "📂 Load" to restore it.', 'info');
  }
}

// ── Plan view (horizontal slice) ──────────────────────────────────────────────
function initPlanView() {
  AppState.planView = new PlanView();

  document.getElementById('btn-plan-view')?.addEventListener('click', () => {
    if (!AppState.voxelGrid) { log('Build the 3D model first.', 'warn'); return; }
    AppState.planView.draw(AppState.voxelGrid, AppState.geoUnits, AppState.classifiedBH);
  });

  // Show/hide unit probability selector when mode changes
  document.getElementById('plan-view-mode')?.addEventListener('change', e => {
    const wrap = document.getElementById('plan-view-prob-unit-wrap');
    if (wrap) wrap.style.display = e.target.value === 'probability' ? 'flex' : 'none';
    if (AppState.voxelGrid && AppState.planView?.visible) {
      AppState.planView.draw(AppState.voxelGrid, AppState.geoUnits, AppState.classifiedBH);
    }
  });

  document.getElementById('plan-view-prob-unit')?.addEventListener('change', () => {
    if (AppState.voxelGrid && AppState.planView?.visible) {
      AppState.planView.draw(AppState.voxelGrid, AppState.geoUnits, AppState.classifiedBH);
    }
  });
}

// ── Unit properties tab ───────────────────────────────────────────────────────
function initPropertiesTab() {
  const refresh = () => renderPropertiesTable(AppState.geoUnits, () => updateLegend());

  document.querySelectorAll('.tab-btn[data-tab="properties"]').forEach(btn => {
    btn.addEventListener('click', refresh);
  });

  document.getElementById('btn-export-props')?.addEventListener('click', () => {
    // handled in exporter.js
  });
}

// ── Rename legend units on double-click ───────────────────────────────────────
function initLegendRename() {
  // Delegated listener on the legend container
  const legend = document.getElementById('unit-legend');
  legend?.addEventListener('dblclick', e => {
    const nameEl = e.target.closest('.legend-name');
    if (!nameEl) return;
    const item = nameEl.closest('.legend-item');
    if (!item) return;
    const code = item.dataset.code;
    const unit = AppState.geoUnits.find(u => u.code === code);
    if (!unit) return;
    nameEl.contentEditable = 'true';
    nameEl.focus();
    const range = document.createRange();
    range.selectNodeContents(nameEl);
    window.getSelection().removeAllRanges();
    window.getSelection().addRange(range);

    const commit = () => {
      nameEl.contentEditable = 'false';
      const newName = nameEl.textContent.trim();
      if (newName) unit.name = newName;
      else nameEl.textContent = unit.name;
      // Refresh stats that show unit name
      updateUnitStats();
    };
    nameEl.addEventListener('blur',    commit, { once: true });
    nameEl.addEventListener('keydown', e => {
      if (e.key === 'Enter') { e.preventDefault(); nameEl.blur(); }
      if (e.key === 'Escape') {
        nameEl.textContent = unit.name;
        nameEl.contentEditable = 'false';
      }
    }, { once: true });
    e.stopPropagation();
  });
}

// ── Model quality / coverage report ──────────────────────────────────────────
function initModelReport() {
  AppState.report = new ModelReport();

  document.getElementById('btn-model-report')?.addEventListener('click', () => {
    if (!AppState.voxelGrid) { log('Build the 3D model first.', 'warn'); return; }
    const gwt       = parseFloat(document.getElementById('gwt-elevation')?.value ?? '') || null;
    const riskRpt   = assessRisk(AppState.voxelGrid, AppState.geoUnits, AppState.classifiedBH, gwt);
    AppState.report.exportHTML(AppState.voxelGrid, AppState.classifiedBH, AppState.geoUnits, riskRpt);
    log('Report HTML downloaded.', 'ok');
  });

  document.getElementById('btn-ai-narrative')?.addEventListener('click', async () => {
    if (!AppState.voxelGrid) { log('Build the 3D model first.', 'warn'); return; }
    const btn = document.getElementById('btn-ai-narrative');
    if (btn) btn.disabled = true;
    log('Generating AI narrative…', 'info');
    const siteCtx = document.getElementById('input-site-history')?.value ?? '';
    const apiKey  = sessionStorage.getItem('anthropic_api_key') ?? '';
    try {
      const result = await generateReportNarrative(
        AppState.geoUnits, AppState.classifiedBH, AppState.voxelGrid,
        siteCtx, apiKey, !apiKey,
      );
      if (result) {
        const el = document.getElementById('ai-narrative-output');
        if (el) {
          el.style.display = 'block';
          el.innerHTML = `
            <div style="margin-bottom:8px;font-size:12px;color:var(--text-mid);font-weight:600">AI-Generated Geotechnical Interpretation</div>
            <div style="font-size:11px;line-height:1.6;color:var(--text-main);margin-bottom:8px">${escHtml(result.narrative)}</div>
            ${result.key_findings?.length ? `<div style="font-size:11px;font-weight:600;color:var(--accent-cyan);margin-bottom:3px">Key Findings</div><ul style="font-size:11px;margin:0 0 6px;padding-left:16px;color:var(--text-mid)">${result.key_findings.map(f=>`<li>${escHtml(f)}</li>`).join('')}</ul>` : ''}
            ${result.geotechnical_risks?.length ? `<div style="font-size:11px;font-weight:600;color:#d04040;margin-bottom:3px">Geotechnical Risks</div><ul style="font-size:11px;margin:0 0 6px;padding-left:16px;color:var(--text-mid)">${result.geotechnical_risks.map(r=>`<li>${escHtml(r)}</li>`).join('')}</ul>` : ''}
            ${result.recommendations?.length ? `<div style="font-size:11px;font-weight:600;color:var(--green);margin-bottom:3px">Recommendations</div><ul style="font-size:11px;margin:0;padding-left:16px;color:var(--text-mid)">${result.recommendations.map(r=>`<li>${escHtml(r)}</li>`).join('')}</ul>` : ''}
          `;
        }
        log('AI narrative generated.', 'ok');
      }
    } catch (err) {
      log(`Narrative generation failed: ${err.message}`, 'error');
    } finally {
      if (btn) btn.disabled = false;
    }
  });

  document.getElementById('btn-validate-model')?.addEventListener('click', () => {
    if (!AppState.voxelGrid) { log('Build the 3D model first.', 'warn'); return; }
    const result = AppState.report.validateModel(
      AppState.voxelGrid, AppState.classifiedBH, AppState.geoUnits
    );
    if (result) {
      log(`Model validation: ${result.accuracy}% accuracy across ${result.total} BH layer samples.`,
        +result.accuracy >= 70 ? 'ok' : 'warn');
    }
  });
}

// ── 3D Annotation labels ──────────────────────────────────────────────────────
function initAnnotations() {
  const btn = document.getElementById('btn-annotate');
  const toggle = () => {
    if (!AppState.scene) return;
    AppState.scene.setAnnotationMode(!AppState.scene._annotationMode);
    if (AppState.scene._annotationMode) {
      if (AppState.scene._vbhMode) { AppState.scene.setVBHMode(false); btn?.classList.remove('active'); }
      if (AppState.scene._measureMode) AppState.scene.setMeasureMode(false);
    }
  };
  btn?.addEventListener('click', toggle);
  window.addEventListener('geomodel:toggle-annotate', toggle);
}

// ── Stratigraphic column ──────────────────────────────────────────────────────
function updateStratColumn() {
  const canvas = document.getElementById('strat-canvas');
  const hint   = document.getElementById('strat-hint');
  if (!canvas) return;
  const geoUnits = AppState.geoUnits;
  const grid     = AppState.voxelGrid;
  if (!geoUnits.length) {
    canvas.hidden = true;
    if (hint) hint.hidden = false;
    return;
  }
  if (hint) hint.hidden = true;
  canvas.hidden = false;

  // Compute mean thickness per unit from grid (or equal height if no grid)
  const thickByUnit = {};
  if (grid) {
    const { nx, ny, nz, cellHeight: ch, unitIds } = grid;
    const counts = {};
    geoUnits.forEach(u => { counts[u.id] = 0; });
    for (let iz = 0; iz < nz; iz++) {
      for (let iy = 0; iy < ny; iy++) {
        for (let ix = 0; ix < nx; ix++) {
          const uid = unitIds[ix + iy * nx + iz * nx * ny];
          if (uid && counts[uid] !== undefined) counts[uid]++;
        }
      }
    }
    // Convert column-voxel counts to mean thickness per unit
    geoUnits.forEach(u => {
      thickByUnit[u.id] = counts[u.id] > 0 ? (counts[u.id] / (nx * ny)) * ch : 0.5;
    });
  } else {
    geoUnits.forEach(u => { thickByUnit[u.id] = 1; });
  }

  const totalThick = Object.values(thickByUnit).reduce((a, b) => a + b, 0);
  const W = canvas.parentElement?.clientWidth ?? 200;
  const H = Math.max(geoUnits.length * 16, 80);
  canvas.width  = W;
  canvas.height = H;

  const ctx     = canvas.getContext('2d');
  const BARW    = 28;
  const LBLX    = BARW + 8;
  const PAD     = 4;
  ctx.clearRect(0, 0, W, H);

  let y = PAD;
  geoUnits.forEach(unit => {
    const thick = thickByUnit[unit.id] ?? 1;
    const frac  = totalThick > 0 ? thick / totalThick : 1 / geoUnits.length;
    const barH  = Math.max(14, frac * (H - PAD * 2));

    ctx.fillStyle = unit.color;
    ctx.fillRect(PAD, y, BARW, barH);
    ctx.strokeStyle = 'rgba(0,0,0,0.15)'; ctx.lineWidth = 0.5;
    ctx.strokeRect(PAD, y, BARW, barH);

    ctx.fillStyle = '#1c2a38';
    ctx.font = 'bold 10px Inter, sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(unit.code, LBLX + PAD, y + barH * 0.5);

    ctx.fillStyle = '#8898a8';
    ctx.font = '9px Inter, sans-serif';
    const nameX = LBLX + PAD + 30;
    if (nameX < W - 4) ctx.fillText(unit.name.slice(0, 18), nameX, y + barH * 0.5);

    if (grid) {
      ctx.fillStyle = '#c8cdd6';
      ctx.font = '8px Inter, sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText(`${thick.toFixed(1)}m`, W - 4, y + barH * 0.5);
    }

    y += barH;
  });
}

// ── Isopach map ──────────────────────────────────────────────────────────────
function initIsopachMap() {
  AppState.isopachMap = new IsopachMap();

  document.getElementById('btn-isopach')?.addEventListener('click', () => {
    if (!AppState.voxelGrid) { log('Build the 3D model first.', 'warn'); return; }
    AppState.isopachMap.draw(AppState.voxelGrid, AppState.geoUnits, AppState.classifiedBH);
  });
}

// ── Fence section (2D cross-section) ─────────────────────────────────────────
function initFenceSection() {
  AppState.fenceSection = new FenceSection();

  document.getElementById('slicer-section-btn')?.addEventListener('click', () => {
    if (!AppState.voxelGrid) { log('Build the 3D model first.', 'warn'); return; }
    const slicer = AppState.scene?.slicer;
    if (!slicer?._hasSlice) { log('Draw a section line first.', 'warn'); return; }
    AppState.fenceSection.draw(
      AppState.voxelGrid,
      AppState.geoUnits,
      slicer._normal,
      slicer._centerD,
      slicer._thickness,
      AppState.classifiedBH
    );
  });
}

// ── Screenshot ────────────────────────────────────────────────────────────────
function initScreenshot() {
  document.getElementById('btn-screenshot')?.addEventListener('click', () => {
    if (!AppState.scene) return;
    const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    AppState.scene.takeScreenshot(`geomodel-${ts}.png`);
    log('Screenshot saved.', 'ok');
  });
}

// ── Measure tool ──────────────────────────────────────────────────────────────
function initMeasureTool() {
  const btn = document.getElementById('btn-measure');
  const toggle = () => {
    if (!AppState.scene) return;
    AppState.scene.setMeasureMode(!AppState.scene._measureMode);
    if (AppState.scene._measureMode && AppState.scene._vbhMode) {
      AppState.scene.setVBHMode(false);
      document.getElementById('btn-vbh')?.classList.remove('active');
    }
  };
  btn?.addEventListener('click', toggle);
  window.addEventListener('geomodel:toggle-measure', toggle);
}

// ── Background toggle ─────────────────────────────────────────────────────────
function initBackgroundToggle() {
  document.getElementById('dark-background')?.addEventListener('change', e => {
    if (AppState.scene) AppState.scene.setBackground(e.target.checked);
  });
}

// ── Borehole column chart ─────────────────────────────────────────────────────
export function updateBHChart() {
  const wrap   = document.getElementById('bh-chart-wrap');
  const canvas = document.getElementById('bh-chart-canvas');
  if (!wrap || !canvas) return;

  const bhs = AppState.classifiedBH.filter(b => !b.synthetic && b.layers?.length);
  if (!bhs.length || !AppState.geoUnits.length) {
    wrap.hidden = true;
    return;
  }

  const ctx   = canvas.getContext('2d');
  const PAD   = 8;
  const BARW  = 22;
  const GAP   = 5;
  const H     = 130;
  const lblH  = 22;
  const drawH = H - PAD - lblH;
  const totalW = bhs.length * (BARW + GAP) + PAD * 2;

  canvas.width  = totalW;
  canvas.height = H;
  wrap.hidden = false;

  ctx.clearRect(0, 0, totalW, H);

  const unitByCode = {};
  AppState.geoUnits.forEach(u => { unitByCode[u.code] = u; });

  bhs.forEach((bh, bi) => {
    const maxDepth = Math.max(...bh.layers.map(l => l.base));
    if (!maxDepth) return;
    const x = PAD + bi * (BARW + GAP);

    bh.layers.forEach(layer => {
      const u = unitByCode[layer.unitCode];
      if (!u) return;
      const y0 = PAD + (layer.top  / maxDepth) * drawH;
      const y1 = PAD + (layer.base / maxDepth) * drawH;
      ctx.fillStyle = u.color;
      ctx.fillRect(x, y0, BARW, Math.max(1, y1 - y0));
    });

    // Frame
    ctx.strokeStyle = '#c8cdd6';
    ctx.lineWidth = 0.5;
    ctx.strokeRect(x, PAD, BARW, drawH);

    // Label
    ctx.fillStyle = '#4a6275';
    ctx.font = '8px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.save();
    ctx.translate(x + BARW / 2, H - 4);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(bh.id, 0, 0);
    ctx.restore();
  });
}

// ── Borehole data table with inline layer editor ──────────────────────────────
export function updateBHTable() {
  const wrap = document.getElementById('bh-data-table-wrap');
  if (!wrap) return;
  const bhs = AppState.classifiedBH.filter(b => !b.synthetic);
  if (!bhs.length) {
    wrap.innerHTML = '<p class="hint" style="padding:8px">No borehole data loaded.</p>';
    return;
  }
  const unitByCode = {};
  AppState.geoUnits.forEach(u => { unitByCode[u.code] = u; });

  let html = `<table class="bh-table"><thead><tr>
    <th>ID</th><th>X</th><th>Y</th><th>GL</th><th>D(m)</th><th>Units</th><th></th>
  </tr></thead><tbody>`;

  for (const bh of bhs) {
    const maxBase = bh.layers.length ? Math.max(...bh.layers.map(l => l.base)) : (bh.depth ?? 0);
    const chips = bh.layers.map(l => {
      const u = unitByCode[l.unitCode];
      const bg = u?.color ?? '#888';
      return `<span class="unit-chip" style="background:${bg};font-size:9px;padding:1px 4px">${escHtml(l.unitCode)}</span>`;
    }).join('');
    const safeId = escHtml(bh.id);
    html += `<tr class="bh-summary-row" data-bhid="${safeId}">
      <td class="bh-id">${safeId}</td>
      <td>${bh.x?.toFixed(1) ?? '—'}</td>
      <td>${bh.y?.toFixed(1) ?? '—'}</td>
      <td>${bh.groundLevel?.toFixed(1) ?? '—'}</td>
      <td>${maxBase.toFixed(1)}</td>
      <td class="bh-chips">${chips}</td>
      <td><button class="bh-edit-btn btn-ghost btn-sm" data-bhid="${safeId}" title="Edit layers">✎</button></td>
    </tr>
    <tr class="bh-edit-row" data-bhid="${safeId}" hidden><td colspan="7"></td></tr>`;
  }
  html += '</tbody></table>';
  wrap.innerHTML = html;

  wrap.querySelectorAll('.bh-edit-btn').forEach(btn => {
    btn.addEventListener('click', () => _openBHEditor(btn.dataset.bhid, wrap));
  });
}

function _openBHEditor(bhid, wrap) {
  const editRow = wrap.querySelector(`.bh-edit-row[data-bhid="${_cssEsc(bhid)}"]`);
  if (!editRow) return;
  if (!editRow.hidden) { editRow.hidden = true; return; }
  wrap.querySelectorAll('.bh-edit-row').forEach(r => { r.hidden = true; });

  const bh = AppState.classifiedBH.find(b => b.id === bhid);
  if (!bh) return;

  const td = editRow.querySelector('td');
  td.innerHTML = '';

  const buildTable = () => {
    const selOpts = AppState.geoUnits.map(u =>
      `<option value="${escHtml(u.code)}">${escHtml(u.code)} — ${escHtml(u.name)}</option>`
    ).join('');
    let t = `<table class="bh-layer-edit-table">
      <thead><tr><th>Top(m)</th><th>Base(m)</th><th>Unit</th><th>Cert.</th><th></th></tr></thead>
      <tbody>`;
    bh.layers.forEach((l, i) => {
      const opts = AppState.geoUnits.map(u =>
        `<option value="${escHtml(u.code)}"${u.code === l.unitCode ? ' selected' : ''}>${escHtml(u.code)}</option>`
      ).join('');
      t += `<tr data-layeridx="${i}">
        <td><input type="number" class="bhe-top cell-size-input" value="${l.top ?? ''}" step="0.1" style="width:48px"></td>
        <td><input type="number" class="bhe-base cell-size-input" value="${l.base ?? ''}" step="0.1" style="width:48px"></td>
        <td><select class="bhe-code isopach-select" style="max-width:68px">${opts}</select></td>
        <td><input type="number" class="bhe-cert cell-size-input" value="${(l.certainty ?? 0.9).toFixed(2)}" min="0" max="1" step="0.05" style="width:40px"></td>
        <td><button class="bhe-del btn-ghost btn-sm">✕</button></td>
      </tr>`;
    });
    t += '</tbody></table>';
    return t;
  };

  const panel = document.createElement('div');
  panel.className = 'bh-edit-panel';

  const layerWrap = document.createElement('div');
  layerWrap.className = 'bh-edit-layers';
  layerWrap.innerHTML = buildTable();

  const actions = document.createElement('div');
  actions.className = 'bh-edit-actions';
  actions.innerHTML = `
    <button class="btn-ghost btn-sm bhe-add">+ Layer</button>
    <button class="btn-primary btn-sm bhe-apply">✓ Apply</button>`;

  panel.appendChild(layerWrap);
  panel.appendChild(actions);
  td.appendChild(panel);
  editRow.hidden = false;

  layerWrap.addEventListener('click', e => {
    if (!e.target.classList.contains('bhe-del')) return;
    const idx = parseInt(e.target.closest('tr').dataset.layeridx);
    bh.layers.splice(idx, 1);
    layerWrap.innerHTML = buildTable();
  });

  actions.querySelector('.bhe-add')?.addEventListener('click', () => {
    const lastBase = bh.layers.length ? Math.max(...bh.layers.map(l => l.base ?? 0)) : 0;
    bh.layers.push({ top: lastBase, base: lastBase + 1,
                     unitCode: AppState.geoUnits[0]?.code ?? '',
                     certainty: 0.7, description: '' });
    layerWrap.innerHTML = buildTable();
  });

  actions.querySelector('.bhe-apply')?.addEventListener('click', () => {
    const rows = layerWrap.querySelectorAll('tbody tr');
    const newLayers = [];
    rows.forEach(row => {
      const idx  = parseInt(row.dataset.layeridx);
      const top  = parseFloat(row.querySelector('.bhe-top')?.value ?? '');
      const base = parseFloat(row.querySelector('.bhe-base')?.value ?? '');
      const code = row.querySelector('.bhe-code')?.value ?? '';
      const cert = parseFloat(row.querySelector('.bhe-cert')?.value ?? '0.9');
      if (!isNaN(top) && !isNaN(base) && code) {
        newLayers.push({ top, base, unitCode: code,
                         certainty: isNaN(cert) ? 0.9 : Math.min(1, Math.max(0, cert)),
                         description: bh.layers[idx]?.description ?? '' });
      }
    });
    bh.layers = newLayers.sort((a, b) => a.top - b.top);
    editRow.hidden = true;
    updateBHTable();
    updateBHChart();
    log(`BH ${bhid} updated — ${newLayers.length} layer${newLayers.length !== 1 ? 's' : ''}. Rebuild model to apply.`, 'ok');
    if (AppState.classifiedBH.length) setEnabled('btn-build-model', true);
  });
}

function _cssEsc(s) {
  return String(s).replace(/[!"#$%&'()*+,./:;<=>?@[\\\]^`{|}~]/g, c => `\\${c}`);
}

// ── Unit statistics ───────────────────────────────────────────────────────────
// ── Geotechnical risk assessment ─────────────────────────────────────────────
// ── Geological Scenarios ──────────────────────────────────────────────────────
function initScenarioManager() {
  const STORAGE_KEY = 'geomodel_scenarios';
  const MAX_SCENARIOS = 5;

  function loadScenarios() {
    try { return JSON.parse(sessionStorage.getItem(STORAGE_KEY) || '[]'); } catch { return []; }
  }
  function saveScenarios(list) {
    try { sessionStorage.setItem(STORAGE_KEY, JSON.stringify(list)); } catch (e) {
      log('Scenario save failed — storage full.', 'warn');
    }
  }

  function captureState(name) {
    return {
      name,
      createdAt: new Date().toISOString(),
      geoUnits: JSON.parse(JSON.stringify(AppState.geoUnits)),
      classifiedBH: JSON.parse(JSON.stringify(AppState.classifiedBH)),
      cellSizeH:    AppState.cellSizeH,
      cellSizeZ:    AppState.cellSizeZ,
      kNeighbors:   AppState.kNeighbors,
      idwPower:     AppState.idwPower,
      interpMethod: AppState.interpMethod,
      anisoAzimuth: AppState.anisoAzimuth,
      anisoRatio:   AppState.anisoRatio,
    };
  }

  async function restoreState(sc) {
    AppState.geoUnits     = sc.geoUnits;
    AppState.classifiedBH = sc.classifiedBH;
    AppState.cellSizeH    = sc.cellSizeH;
    AppState.cellSizeZ    = sc.cellSizeZ ?? 0.25;
    AppState.kNeighbors   = sc.kNeighbors ?? 5;
    AppState.idwPower     = sc.idwPower   ?? 2;
    AppState.interpMethod = sc.interpMethod ?? 'idw';
    AppState.anisoAzimuth = sc.anisoAzimuth ?? 0;
    AppState.anisoRatio   = sc.anisoRatio   ?? 1;
    updateLegend();
    updateInfoPanel();
    updateBHTable();
    updateBHChart();
    updateStratColumn();
    setEnabled('btn-run-ai', sc.classifiedBH.length > 0);
    setEnabled('btn-build-model', sc.classifiedBH.length > 0);
    log(`Switched to scenario "${sc.name}". Click Build 3D Model to regenerate.`, 'ok');
  }

  function renderList() {
    const listEl = document.getElementById('scenario-list');
    if (!listEl) return;
    const scenarios = loadScenarios();
    if (!scenarios.length) {
      listEl.innerHTML = '<p class="hint" style="padding:8px">No saved scenarios yet.</p>';
      return;
    }
    listEl.innerHTML = scenarios.map((sc, i) => {
      const date = new Date(sc.createdAt).toLocaleString('en-GB', { dateStyle:'short', timeStyle:'short' });
      const unitCodes = sc.geoUnits.map(u => `<span style="display:inline-block;width:8px;height:8px;border-radius:2px;background:${u.color};margin-right:2px;vertical-align:middle"></span>${u.code}`).join(' ');
      return `<div style="display:flex;align-items:center;gap:6px;padding:7px 8px;border:1px solid var(--border);border-radius:var(--radius);margin-bottom:5px;background:var(--bg-surface)">
        <div style="flex:1;min-width:0">
          <div style="font-weight:600;font-size:12px;margin-bottom:2px">${sc.name}</div>
          <div style="font-size:10px;color:var(--text-mid)">${date} · ${sc.classifiedBH?.length ?? 0} BHs · ${unitCodes}</div>
        </div>
        <button data-idx="${i}" class="btn-scenario-load btn-secondary btn-sm" style="white-space:nowrap">Switch</button>
        <button data-idx="${i}" class="btn-scenario-del btn-ghost btn-sm" title="Delete" style="color:#e06040">✕</button>
      </div>`;
    }).join('');

    listEl.querySelectorAll('.btn-scenario-load').forEach(btn => {
      btn.addEventListener('click', async () => {
        const sc = loadScenarios()[+btn.dataset.idx];
        if (sc) { await restoreState(sc); }
      });
    });
    listEl.querySelectorAll('.btn-scenario-del').forEach(btn => {
      btn.addEventListener('click', () => {
        const list = loadScenarios();
        list.splice(+btn.dataset.idx, 1);
        saveScenarios(list);
        renderList();
        log('Scenario deleted.', 'info');
      });
    });
  }

  // Modal open/close
  const modal  = document.getElementById('modal-scenarios');
  const openBtn = document.getElementById('btn-scenario-manager');
  const closeBtn = document.getElementById('modal-scenarios-close');
  openBtn?.addEventListener('click', () => { renderList(); modal?.removeAttribute('hidden'); });
  closeBtn?.addEventListener('click', () => modal?.setAttribute('hidden', ''));
  modal?.addEventListener('click', e => { if (e.target === modal) modal.setAttribute('hidden', ''); });

  // Save current state as a new scenario
  document.getElementById('btn-scenario-save')?.addEventListener('click', () => {
    if (!AppState.geoUnits.length) { log('Nothing to save — load data first.', 'warn'); return; }
    const nameInput = document.getElementById('scenario-name-input');
    const name = (nameInput?.value.trim()) || `Scenario ${new Date().toLocaleTimeString('en-GB')}`;
    const list = loadScenarios();
    if (list.length >= MAX_SCENARIOS) {
      log(`Max ${MAX_SCENARIOS} scenarios. Delete one first.`, 'warn'); return;
    }
    list.push(captureState(name));
    saveScenarios(list);
    if (nameInput) nameInput.value = '';
    renderList();
    log(`Scenario "${name}" saved.`, 'ok');
  });
}

// ── Marching-cubes isosurfaces ────────────────────────────────────────────────
function initIsosurfaces() {
  const btn = document.getElementById('btn-build-isosurfaces');
  if (!btn) return;
  let built = false;
  let visible = false;

  btn.addEventListener('click', async () => {
    if (!AppState.voxelGrid || !AppState.scene) { log('Build the 3D model first.', 'warn'); return; }

    if (built) {
      // Toggle visibility
      visible = !visible;
      AppState.scene.setIsosurfacesVisible(visible);
      btn.classList.toggle('active', visible);
      log(`Isosurfaces ${visible ? 'shown' : 'hidden'}.`, 'info');
      return;
    }

    btn.disabled = true;
    btn.textContent = '⬡ Building…';
    log('Building marching-cubes isosurfaces…', 'info');
    await new Promise(r => setTimeout(r, 0));

    const op = parseFloat(document.getElementById('surface-opacity')?.value ?? 55) / 100;
    AppState.scene.buildIsosurfaces(
      AppState.voxelGrid, AppState.geoUnits, op,
      p => { btn.textContent = `⬡ ${(p * 100).toFixed(0)}%`; },
    );

    built = true;
    visible = true;
    AppState.scene.setIsosurfacesVisible(true);
    btn.disabled = false;
    btn.textContent = '⬡ Isosurfaces';
    btn.classList.add('active');
    log(`Isosurfaces built for ${AppState.geoUnits.length} unit(s).`, 'ok');
  });
}

// ── Semantic Knowledge Model ───────────────────────────────────────────────────
function initSemanticModel() {
  const weightSlider = document.getElementById('semantic-weight');
  const weightVal    = document.getElementById('semantic-weight-val');
  weightSlider?.addEventListener('input', () => {
    AppState.semanticWeight = parseFloat(weightSlider.value) / 100;
    if (weightVal) weightVal.textContent = weightSlider.value + '%';
  });

  document.getElementById('btn-semantic-model')?.addEventListener('click', async () => {
    if (!AppState.geoUnits.length) { log('Run AI analysis first to define geological units.', 'warn'); return; }
    setEnabled('btn-semantic-model', false);
    log('Generating semantic knowledge model…', 'info');
    try {
      const siteCtx = document.getElementById('input-site-history')?.value ?? '';
      const result  = await generateSemanticModel(
        AppState.geoUnits, AppState.classifiedBH, siteCtx,
        AppState.apiKey, AppState.demoMode
      );
      AppState.semanticModel = result;
      analysisLog('Semantic Knowledge Model', result.model_narrative ?? 'Model generated.', 'ok');
      if (result.synthetic_anchors?.length) {
        log(`Semantic model: ${result.synthetic_anchors.length} synthetic anchor(s) ready.`, 'ok');
      }
      log('Semantic model ready — rebuild the 3D model to apply.', 'ok');
    } catch (err) {
      log(`Semantic model error: ${err.message}`, 'error');
    } finally {
      setEnabled('btn-semantic-model', true);
    }
  });
}

// ── Parameter View (color voxels by engineering parameter) ────────────────────
function initParameterView() {
  const PARAM_LABELS = {
    cu: 'Cu — Undrained Shear Strength (kPa)',
    phi: "φ′ — Friction Angle (°)",
    Cc: 'Cc — Compression Index',
    E: 'E — Stiffness Modulus (MPa)',
    gamma: 'γ — Unit Weight (kN/m³)',
    N_spt: 'SPT N — Blow Count',
    boundary: 'Boundary uncertainty (blend ratio 0–1)',
  };

  document.getElementById('btn-param-apply')?.addEventListener('click', () => {
    if (!AppState.voxelGrid || !AppState.scene) { log('Build the 3D model first.', 'warn'); return; }
    const paramName = document.getElementById('param-select')?.value;
    if (!paramName) { log('Select a parameter to display.', 'warn'); return; }

    if (paramName === 'boundary') {
      AppState.scene.colorByBoundaryUncertainty();
      document.getElementById('param-scale-min').textContent = '0.0';
      document.getElementById('param-scale-mid').textContent = '0.5';
      document.getElementById('param-scale-max').textContent = '1.0';
      document.getElementById('param-scale-label').textContent = 'Boundary uncertainty — blue=certain, red=near contact';
      document.getElementById('param-colorscale').style.display = 'block';
      log('Parameter view: boundary uncertainty (blend ratio)', 'ok');
      return;
    }

    const range = AppState.scene.colorByParameter(paramName, AppState.geoUnits);
    if (!range) {
      log(`No unit has the parameter "${paramName}" defined. Use Auto-Fill Parameters first.`, 'warn');
      return;
    }
    const { min, max } = range;
    const mid = ((min + max) / 2).toFixed(1);
    document.getElementById('param-scale-min').textContent = min.toFixed(1);
    document.getElementById('param-scale-mid').textContent = mid;
    document.getElementById('param-scale-max').textContent = max.toFixed(1);
    document.getElementById('param-scale-label').textContent = PARAM_LABELS[paramName] ?? paramName;
    document.getElementById('param-colorscale').style.display = 'block';
    log(`Parameter view: ${PARAM_LABELS[paramName] ?? paramName} (${min.toFixed(1)} – ${max.toFixed(1)})`, 'ok');
  });

  document.getElementById('btn-param-reset')?.addEventListener('click', () => {
    if (!AppState.scene) return;
    AppState.scene.resetUnitColors();
    document.getElementById('param-colorscale').style.display = 'none';
    log('Restored unit colours.', 'info');
  });
}

// ── Structural orientation import ─────────────────────────────────────────────
// Parses strike/dip measurements and computes circular-mean strike azimuth.
// Accepted formats: "045 15" (azimuth dip), "strike 060, dip 20 NW", "N45E 15"
function initOrientationImport() {
  document.getElementById('btn-orientation-parse')?.addEventListener('click', () => {
    const text = document.getElementById('orientation-text')?.value ?? '';
    const lines = text.split(/\r?\n/).map(l => l.trim()).filter(Boolean);
    const azimuths = [];
    for (const line of lines) {
      // Try "strike NNN, dip DD" or "strike NNN dip DD"
      let m = line.match(/strike\s+(\d+)/i) ?? line.match(/^(\d{1,3})[,\s]+\d/);
      if (m) { azimuths.push(parseFloat(m[1])); continue; }
      // Try plain "azimuth dip" pair
      const parts = line.split(/[\s,]+/);
      if (parts.length >= 2 && isFinite(parts[0]) && isFinite(parts[1])) {
        azimuths.push(parseFloat(parts[0]));
      }
    }
    if (!azimuths.length) {
      document.getElementById('orientation-result').textContent = 'No valid measurements found.';
      return;
    }
    // Circular mean of azimuths
    const sinSum = azimuths.reduce((s, a) => s + Math.sin(a * Math.PI / 180), 0);
    const cosSum = azimuths.reduce((s, a) => s + Math.cos(a * Math.PI / 180), 0);
    let meanAz = Math.atan2(sinSum, cosSum) * 180 / Math.PI;
    if (meanAz < 0) meanAz += 360;
    meanAz = Math.round(meanAz);
    AppState.anisoAzimuth = meanAz;
    const azEl = document.getElementById('aniso-azimuth');
    if (azEl) azEl.value = meanAz;
    document.getElementById('orientation-result').textContent =
      `${azimuths.length} measurements → mean strike ${meanAz}°N applied.`;
    log(`Structural orientation: ${azimuths.length} measurements, mean strike ${meanAz}°N`, 'ok');
  });
}

// ── Foundation design grid export ────────────────────────────────────────────
function initFoundationExport() {
  document.getElementById('btn-foundation-export')?.addEventListener('click', () => {
    const grid = AppState.voxelGrid;
    if (!grid) { log('Build the 3D model first.', 'warn'); return; }
    const fdDepth = parseFloat(document.getElementById('foundation-depth')?.value ?? '1.5');
    const dSigma  = parseFloat(document.getElementById('foundation-load')?.value  ?? '100');
    if (!isFinite(fdDepth) || !isFinite(dSigma)) {
      log('Enter valid foundation parameters.', 'warn'); return;
    }

    const { nx, ny, nz, cellSize: cs, cellHeight: ch, origin: O, unitIds } = grid;
    const unitMap = {};
    AppState.geoUnits.forEach(u => { unitMap[u.id] = u; });

    const rows = ['x,y,elevation_mAOD,unit,bearing_kPa,settlement_mm'];
    for (let iy = 0; iy < ny; iy++) {
      for (let ix = 0; ix < nx; ix++) {
        // Find surface voxel for this column
        let surfIz = -1;
        for (let jz = nz - 1; jz >= 0; jz--) {
          if (unitIds[ix + iy * nx + jz * nx * ny]) { surfIz = jz; break; }
        }
        if (surfIz < 0) continue;
        const surfElev = O.y + surfIz * ch + ch;
        const fdElev   = surfElev - fdDepth;
        const fdIz     = Math.max(0, Math.min(nz - 1, Math.floor((fdElev - O.y) / ch)));
        const uid = unitIds[ix + iy * nx + fdIz * nx * ny];
        const unit = unitMap[uid];
        if (!unit) continue;

        const p   = unit.params ?? {};
        const cu  = p.cu   ?? 0;
        const Cc  = p.Cc   ?? 0;
        const E   = p.E    ?? 1;
        const gam = p.gamma ?? 20;
        const e0  = 0.7;    // typical initial void ratio

        // Undrained bearing capacity: qu = Nc*Cu, Nc=5.14 (Skempton)
        const bearing = (cu > 0) ? 5.14 * cu : (p.N_spt ?? 0) * 5; // rough SPT correlation

        // Consolidation settlement: δ = Cc/(1+e0) * H * log10((σv0+Δσ)/σv0)
        const sigV0 = gam * fdDepth;
        const hLayer = Math.min(fdDepth * 3, 10); // effective influence depth
        const settlement = Cc > 0 && sigV0 > 0
          ? (Cc / (1 + e0)) * hLayer * 1000 * Math.log10((sigV0 + dSigma) / sigV0)
          : E > 0 ? (dSigma * hLayer * 1000) / E : 0;

        const wx = O.x + (ix + 0.5) * cs;
        const wy = O.z + (iy + 0.5) * cs;
        rows.push(`${wx.toFixed(1)},${wy.toFixed(1)},${fdElev.toFixed(2)},${unit.code},${bearing.toFixed(1)},${settlement.toFixed(1)}`);
      }
    }

    const blob = new Blob([rows.join('\n')], { type: 'text/csv' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `foundation-grid-${fdDepth}m.csv`;
    a.click();
    log(`Foundation grid exported: ${rows.length - 1} cells @ ${fdDepth} m bgl, Δσ = ${dSigma} kPa`, 'ok');
  });
}

function initRiskAssessment() {
  const container = document.getElementById('risk-results');
  renderRiskReport(null, container);

  document.getElementById('btn-assess-risk')?.addEventListener('click', () => {
    if (!AppState.voxelGrid) { log('Build the 3D model first.', 'warn'); return; }
    const gwt = parseFloat(document.getElementById('gwt-elevation')?.value ?? '') || null;
    const report = assessRisk(
      AppState.voxelGrid, AppState.geoUnits, AppState.classifiedBH, gwt
    );
    renderRiskReport(report, container);
    if (report) {
      const hc = report.zones.filter(z => z.level === 'high').length;
      log(`Risk assessment: ${report.zones.length} hazard zone(s) · Overall ${report.overallLevel}.`,
        hc > 0 ? 'warn' : 'ok');
    }
  });
}

// ── BH-derived unit parameter statistics ─────────────────────────────────────
function updateBHUnitStats() {
  const el = document.getElementById('bh-unit-stats');
  if (!el) return;
  const bhs = AppState.classifiedBH.filter(b => !b.synthetic);
  const geoUnits = AppState.geoUnits;
  if (!bhs.length || !geoUnits.length) {
    el.innerHTML = '<p class="hint">Load borehole data to see statistics.</p>';
    return;
  }

  // Accumulate per unit: thickness, sptN, certainty
  const acc = {};
  geoUnits.forEach(u => { acc[u.code] = { thickArr: [], sptArr: [], certArr: [] }; });

  for (const bh of bhs) {
    for (const l of bh.layers) {
      const code = l.unitCode;
      if (!code || !acc[code]) continue;
      const thick = (l.base - l.top);
      if (thick > 0) acc[code].thickArr.push(thick);
      if (l.sptN != null && l.sptN > 0) acc[code].sptArr.push(l.sptN);
      if (l.certainty != null) acc[code].certArr.push(l.certainty);
    }
  }

  const mean = arr => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : null;
  const std  = (arr, m) => arr.length > 1
    ? Math.sqrt(arr.reduce((s, v) => s + (v - m) ** 2, 0) / arr.length) : 0;
  const fmt  = (v, dp = 1) => v != null ? v.toFixed(dp) : '—';

  el.innerHTML = '';
  for (const unit of geoUnits) {
    const s = acc[unit.code];
    if (!s || !s.thickArr.length) continue;
    const mThick = mean(s.thickArr), sdThick = std(s.thickArr, mThick);
    const mSPT   = mean(s.sptArr),   sdSPT   = std(s.sptArr, mSPT);
    const mCert  = mean(s.certArr);

    const div = document.createElement('div');
    div.className = 'stat-row';
    div.innerHTML = `
      <div class="stat-hdr">
        <span class="stat-swatch" style="background:${unit.color}"></span>
        <span class="stat-code">${escHtml(unit.code)}</span>
        <span class="stat-name">${escHtml(unit.name)}</span>
        <span class="stat-val" style="margin-left:auto;font-size:10px;color:var(--text-dim)">${s.thickArr.length} int.</span>
      </div>
      <div class="stat-grid">
        <span class="stat-lbl">Thickness</span>
        <span class="stat-val">${fmt(mThick)} ± ${fmt(sdThick)} m</span>
        ${mSPT != null ? `<span class="stat-lbl">SPT N</span>
        <span class="stat-val">${fmt(mSPT, 0)} ± ${fmt(sdSPT, 0)}</span>` : ''}
        ${mCert != null ? `<span class="stat-lbl">Certainty</span>
        <span class="stat-val">${(mCert * 100).toFixed(0)}%</span>` : ''}
      </div>`;
    el.appendChild(div);
  }
}

function updateUnitStats() {
  const el = document.getElementById('unit-stats');
  if (!el) return;
  const grid = AppState.voxelGrid;
  const geoUnits = AppState.geoUnits;
  if (!grid || !geoUnits.length) {
    el.innerHTML = '<p class="hint">Build model to see statistics</p>';
    return;
  }
  const { nx, ny, nz, cellHeight: ch, origin, unitIds, certainty } = grid;

  // Per-unit stats + depth histogram (count per iz level)
  const stats = {};
  geoUnits.forEach(u => {
    stats[u.id] = { count: 0, certSum: 0, minElev: Infinity, maxElev: -Infinity, depthCounts: new Float32Array(nz) };
  });

  for (let iz = 0; iz < nz; iz++) {
    const elev = origin.y + iz * ch + ch * 0.5;
    for (let iy = 0; iy < ny; iy++) {
      for (let ix = 0; ix < nx; ix++) {
        const flat = ix + iy * nx + iz * nx * ny;
        const uid  = unitIds[flat];
        if (!uid || !stats[uid]) continue;
        const s = stats[uid];
        s.count++;
        s.certSum += certainty[flat];
        s.depthCounts[iz]++;
        s.minElev  = Math.min(s.minElev, elev);
        s.maxElev  = Math.max(s.maxElev, elev);
      }
    }
  }

  // Volume per voxel
  const { cellSize: cs } = grid;
  const voxelVol = cs * cs * ch;

  el.innerHTML = '';
  geoUnits.forEach(unit => {
    const s = stats[unit.id];
    if (!s || !s.count) return;
    const avgCert  = (s.certSum / s.count * 100).toFixed(0);
    const thick    = (s.maxElev - s.minElev).toFixed(1);
    const volume   = (s.count * voxelVol).toFixed(0);
    const maxDC    = Math.max(...s.depthCounts);

    // Build inline depth histogram SVG (horizontal bars, each iz = one row)
    const svgH  = Math.min(nz * 4, 120);
    const svgW  = 80;
    const barMaxW = svgW - 2;
    const barH  = Math.max(1, svgH / nz);
    const bars  = Array.from(s.depthCounts)
      .reverse() // top of model at top
      .map((cnt, i) => {
        const w = maxDC > 0 ? (cnt / maxDC) * barMaxW : 0;
        const y = i * barH;
        return `<rect x="0" y="${y.toFixed(1)}" width="${w.toFixed(1)}" height="${(barH - 0.5).toFixed(1)}" fill="${unit.color}" opacity="0.8"/>`;
      }).join('');

    const div = document.createElement('div');
    div.className = 'stat-row';
    div.innerHTML = `
      <div class="stat-hdr">
        <span class="stat-swatch" style="background:${unit.color}"></span>
        <span class="stat-code">${escHtml(unit.code)}</span>
        <span class="stat-name">${escHtml(unit.name)}</span>
      </div>
      <div style="display:flex;gap:8px;align-items:flex-start;margin-top:4px">
        <svg width="${svgW}" height="${svgH}" style="flex-shrink:0;background:var(--bg-surface);border-radius:3px" title="Depth distribution">
          ${bars}
        </svg>
        <div class="stat-grid" style="flex:1">
          <span class="stat-lbl">Span</span><span class="stat-val">${thick} m</span>
          <span class="stat-lbl">Cert.</span><span class="stat-val">${avgCert}%</span>
          <span class="stat-lbl">Top</span><span class="stat-val">${s.maxElev.toFixed(1)} m</span>
          <span class="stat-lbl">Base</span><span class="stat-val">${s.minElev.toFixed(1)} m</span>
          <span class="stat-lbl">Vol</span><span class="stat-val">${Number(volume).toLocaleString()} m³</span>
        </div>
      </div>`;
    el.appendChild(div);
  });
}

// ── Build progress bar helpers ─────────────────────────────────────────────────
function showBuildProgress(visible) {
  const el = document.getElementById('build-progress-wrap');
  if (el) el.hidden = !visible;
}

function setBuildProgress(fraction) {
  const fill = document.getElementById('build-progress-fill');
  const pct  = document.getElementById('build-progress-pct');
  if (fill) fill.style.width = `${(fraction * 100).toFixed(0)}%`;
  if (pct)  pct.textContent  = `${(fraction * 100).toFixed(0)}%`;
}

// ── Update right-panel info ────────────────────────────────────────────────────
export function updateInfoPanel() {
  const g = AppState.voxelGrid;
  document.getElementById('info-bh-count').textContent = AppState.rawBoreholes.length || '—';
  if (g) {
    document.getElementById('info-voxel-count').textContent = (g.nx * g.ny * g.nz).toLocaleString();
    document.getElementById('info-grid-size').textContent   = `${g.nx}×${g.ny}×${g.nz}`;
    document.getElementById('info-cell-size').textContent   = `${g.cellSize} × ${g.cellHeight.toFixed(2)} m`;
  } else {
    document.getElementById('info-voxel-count').textContent = '—';
    document.getElementById('info-grid-size').textContent   = '—';
    document.getElementById('info-cell-size').textContent   = '—';
  }
}

// ── Update unit legend ─────────────────────────────────────────────────────────
export function updateLegend() {
  const container = document.getElementById('unit-legend');
  container.innerHTML = '';

  // Compute volumes if grid available
  const g   = AppState.voxelGrid;
  const pcts = {};
  if (g) {
    const counts = {};
    AppState.geoUnits.forEach(u => { counts[u.id] = 0; });
    g.unitIds.forEach(id => { if (id && counts[id] !== undefined) counts[id]++; });
    const total = Object.values(counts).reduce((a, b) => a + b, 0);
    AppState.geoUnits.forEach(u => { pcts[u.code] = total > 0 ? (counts[u.id] / total * 100) : 0; });
  }

  AppState.geoUnits.forEach(unit => {
    const item = document.createElement('div');
    item.className = 'legend-item';
    item.dataset.code = unit.code;
    const pct = pcts[unit.code];
    const pctHtml = pct !== undefined
      ? `<span class="legend-pct">${pct.toFixed(0)}%</span>`
      : '';
    item.innerHTML = `
      <div class="legend-swatch" style="background:${unit.color}"></div>
      <span class="legend-code">${escHtml(unit.code)}</span>
      <span class="legend-name">${escHtml(unit.name)}</span>
      ${pctHtml}
      <span class="legend-eye">👁</span>`;
    item.title = unit.description || unit.name;

    // Swatch click → open colour picker
    const swatch = item.querySelector('.legend-swatch');
    swatch?.addEventListener('click', e => {
      e.stopPropagation();
      const picker = document.createElement('input');
      picker.type  = 'color';
      picker.value = unit.color;
      picker.style.position = 'fixed';
      picker.style.opacity  = '0';
      picker.style.pointerEvents = 'none';
      document.body.appendChild(picker);
      picker.click();
      picker.addEventListener('input', () => {
        unit.color = picker.value;
        swatch.style.background = picker.value;
        if (AppState.scene && AppState.voxelGrid) {
          AppState.scene.buildVoxels(AppState.voxelGrid, AppState.geoUnits, AppState.classifiedBH);
          updateVolumeStats();
          updateUnitStats();
        }
      });
      picker.addEventListener('change', () => document.body.removeChild(picker));
    });

    item.addEventListener('click', () => {
      const hidden = AppState.hiddenUnits.has(unit.code);
      if (hidden) AppState.hiddenUnits.delete(unit.code);
      else AppState.hiddenUnits.add(unit.code);
      item.classList.toggle('hidden-unit', !hidden);
      if (AppState.scene) AppState.scene.setUnitVisibility(unit.code, hidden);
    });
    container.appendChild(item);
  });
}

// ── Refresh legend with updated volumes after build ───────────────────────────
function refreshLegendVolumes() {
  if (AppState.geoUnits.length) updateLegend();
}

// ── View mode buttons ──────────────────────────────────────────────────────────
function initViewModeButtons() {
  document.querySelectorAll('.view-mode-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      if (AppState.scene) AppState.scene.setViewMode(btn.dataset.mode);
    });
  });
}

// ── Virtual BH button ──────────────────────────────────────────────────────────
function initVBHButton() {
  const btn = document.getElementById('btn-vbh');
  const toggle = () => {
    if (!AppState.scene) return;
    AppState.scene.setVBHMode(!AppState.scene._vbhMode);
    btn?.classList.toggle('active', AppState.scene._vbhMode);
  };
  btn?.addEventListener('click', toggle);
  window.addEventListener('geomodel:toggle-vbh', toggle);
  document.getElementById('log-popup-close')?.addEventListener('click', () => {
    AppState.scene?._hideLogPopup();
    AppState.scene?.setVBHMode(false);
    btn?.classList.remove('active');
  });
}

// ── Lab data import ────────────────────────────────────────────────────────────
function initLabImport() {
  const dropZone  = document.getElementById('drop-lab');
  const fileInput = document.getElementById('file-lab');
  const fileInfo  = document.getElementById('lab-file-info');
  const labToggle = document.getElementById('lab-toggle');
  const labSection = document.getElementById('lab-section');
  if (!dropZone || !fileInput) return;

  labToggle?.addEventListener('click', () => {
    if (!labSection) return;
    labSection.hidden = !labSection.hidden;
    const arrow = labToggle.querySelector('.collapse-arrow');
    if (arrow) arrow.textContent = labSection.hidden ? '›' : '⌄';
  });

  const parseFile = file => {
    if (!AppState.geoUnits.length) { log('Load borehole data first so unit codes can be resolved.', 'warn'); return; }
    const reader = new FileReader();
    reader.onload = e => {
      const { parsed, skipped, updated } = parseLabCSV(e.target.result, AppState.geoUnits);
      if (!parsed) { log(`Lab import: no rows parsed (${skipped} skipped).`, 'warn'); return; }
      fileInfo.innerHTML = `<div class="file-item">
        <span class="file-name">${escHtml(file.name)}</span>
        <span class="file-size">${parsed} values</span></div>`;
      renderPropertiesTable(AppState.geoUnits, () => updateLegend());
      log(`Lab data: ${parsed} values imported · ${updated.join(' · ') || 'no units matched'}`, 'ok');
    };
    reader.readAsText(file);
  };

  dropZone.addEventListener('click', () => fileInput.click());
  dropZone.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') fileInput.click(); });
  dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault(); dropZone.classList.remove('drag-over');
    const file = e.dataTransfer?.files[0];
    if (file) parseFile(file);
  });
  fileInput.addEventListener('change', () => { if (fileInput.files[0]) parseFile(fileInput.files[0]); });
}

// ── Geological map import ──────────────────────────────────────────────────────
function initGeoMapImport() {
  const dropZone  = document.getElementById('drop-geomap');
  const fileInput = document.getElementById('file-geomap');
  const fileInfo  = document.getElementById('geomap-file-info');
  if (!dropZone || !fileInput) return;

  dropZone.addEventListener('click', () => fileInput.click());
  dropZone.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') fileInput.click();
  });
  dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    const file = e.dataTransfer?.files[0];
    if (file) parseGeoMapFile(file, fileInfo);
  });
  fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) parseGeoMapFile(fileInput.files[0], fileInfo);
  });
}

function parseGeoMapFile(file, infoEl) {
  if (!AppState.geoUnits.length) {
    log('Load borehole data first so unit codes can be resolved.', 'warn');
    return;
  }
  const reader = new FileReader();
  reader.onload = e => {
    const { boreholes, count, skipped } = parseGeoMap(
      e.target.result, AppState.geoUnits, AppState.rawBoreholes
    );
    if (!count) {
      log(`Geological map: no valid rows parsed (${skipped} skipped).`, 'warn');
      return;
    }
    const real = AppState.rawBoreholes.filter(b => !b.synthetic);
    AppState.classifiedBH = [...real, ...boreholes];
    infoEl.innerHTML = `<div class="file-item">
      <span class="file-name">${escHtml(file.name)}</span>
      <span class="file-size">${count} pts</span></div>`;
    setEnabled('btn-export-bh-csv', true);
      setEnabled('btn-export-ags', true);
    log(`Geological map: ${count} constraint points added (${skipped} skipped).`, 'ok');
  };
  reader.readAsText(file);
}

// ── Surface opacity ────────────────────────────────────────────────────────────
function initSurfaceOpacity() {
  const slider = document.getElementById('surface-opacity');
  const val    = document.getElementById('surface-opacity-val');
  slider?.addEventListener('input', () => {
    const op = parseInt(slider.value) / 100;
    if (val) val.textContent = `${slider.value}%`;
    if (AppState.scene) AppState.scene.setSurfaceOpacity(op);
  });
}

// ── Volume statistics ──────────────────────────────────────────────────────────
function updateVolumeStats() {
  const el = document.getElementById('unit-volumes');
  if (!el) return;
  const grid = AppState.voxelGrid;
  const geoUnits = AppState.geoUnits;
  if (!grid || !geoUnits.length) {
    el.innerHTML = '<p class="hint">Build model to see volumes</p>';
    return;
  }
  const cellVol = grid.cellSize * grid.cellSize * grid.cellHeight;
  const counts = {};
  geoUnits.forEach(u => { counts[u.id] = 0; });
  grid.unitIds.forEach(id => { if (id && counts[id] !== undefined) counts[id]++; });
  const total = Object.values(counts).reduce((a, b) => a + b, 0);

  el.innerHTML = '';
  geoUnits.forEach(unit => {
    const n   = counts[unit.id] ?? 0;
    const vol = (n * cellVol);
    const pct = total > 0 ? (n / total * 100).toFixed(0) : 0;
    const row = document.createElement('div');
    row.className = 'vol-row';
    row.innerHTML = `
      <div class="vol-row-hdr">
        <span class="vol-code" style="color:${unit.color}">${escHtml(unit.code)}</span>
        <span class="vol-num">${vol < 1000 ? vol.toFixed(0) : (vol/1000).toFixed(1)+'k'} m³</span>
        <span class="vol-pct">${pct}%</span>
      </div>
      <div class="vol-bar-wrap">
        <div class="vol-bar-fill" style="width:${pct}%;background:${unit.color}"></div>
      </div>`;
    el.appendChild(row);
  });
}

// ── Transparency controls ──────────────────────────────────────────────────────
function initTransparencyControls() {
  const alphaSlider = document.getElementById('global-alpha');
  const alphaVal    = document.getElementById('global-alpha-val');
  alphaSlider?.addEventListener('input', () => {
    const alpha = parseInt(alphaSlider.value) / 100;
    if (alphaVal) alphaVal.textContent = `${alphaSlider.value}%`;
    if (AppState.scene) AppState.scene.setGlobalAlpha(alpha);
  });

  const chk    = document.getElementById('transp-enable');
  const slider = document.getElementById('transp-amount');
  const val    = document.getElementById('transp-val');
  const row    = document.getElementById('transp-slider-row');
  const update = () => {
    const enabled = chk.checked;
    const amount  = parseInt(slider.value) / 100;
    if (val) val.textContent = `${slider.value}%`;
    if (row) row.style.opacity = enabled ? '1' : '0.4';
    if (AppState.scene) AppState.scene.setTransparencyMode(enabled, amount);
  };
  chk?.addEventListener('change', update);
  slider?.addEventListener('input', update);
  if (row) row.style.opacity = '0.4';

  const fadeChk    = document.getElementById('color-fade-enable');
  const fadeSlider = document.getElementById('color-fade-amount');
  const fadeVal    = document.getElementById('color-fade-val');
  const fadeRow    = document.getElementById('color-fade-row');
  const updateFade = () => {
    const enabled = fadeChk.checked;
    const amount  = parseInt(fadeSlider.value) / 100;
    if (fadeVal) fadeVal.textContent = `${fadeSlider.value}%`;
    if (fadeRow) fadeRow.style.opacity = enabled ? '1' : '0.4';
    if (AppState.scene) AppState.scene.setColorFadeMode(enabled, amount);
  };
  fadeChk?.addEventListener('change', updateFade);
  fadeSlider?.addEventListener('input', updateFade);
  if (fadeRow) fadeRow.style.opacity = '0.4';
}

// ── Vertical exaggeration ──────────────────────────────────────────────────────
function initVerticalExaggeration() {
  document.getElementById('vert-exag')?.addEventListener('input', e => {
    const ve = Math.max(0.1, parseFloat(e.target.value) || 1);
    if (AppState.scene) AppState.scene.setVerticalExaggeration(ve);
  });
}

// ── BH sticks toggle ──────────────────────────────────────────────────────────
function initBHSticksToggle() {
  document.getElementById('show-bh-sticks')?.addEventListener('change', e => {
    if (AppState.scene) AppState.scene.toggleBoreholeSticks(e.target.checked);
  });
}

// ── Certainty threshold ────────────────────────────────────────────────────────
function initCertaintySlider() {
  const slider = document.getElementById('certainty-threshold');
  const val    = document.getElementById('certainty-val');
  slider?.addEventListener('input', () => {
    AppState.certaintyThreshold = parseInt(slider.value) / 100;
    if (val) val.textContent = `${slider.value}%`;
    if (AppState.scene) AppState.scene.setCertaintyThreshold(AppState.certaintyThreshold);
  });
}

// ── Welcome overlay interactions ───────────────────────────────────────────────
function initWelcomeOverlay() {
  const welcomeDrop = document.getElementById('welcome-drop');
  if (welcomeDrop) {
    welcomeDrop.addEventListener('dragover', e => { e.preventDefault(); welcomeDrop.classList.add('drag-over'); });
    welcomeDrop.addEventListener('dragleave', () => welcomeDrop.classList.remove('drag-over'));
    welcomeDrop.addEventListener('drop', e => {
      e.preventDefault();
      welcomeDrop.classList.remove('drag-over');
      const files = Array.from(e.dataTransfer?.files ?? []);
      if (files.length) {
        hideWelcome();
        const dt = new DataTransfer();
        files.forEach(f => dt.items.add(f));
        const fileInput = document.getElementById('file-bh');
        if (fileInput) {
          try {
            Object.defineProperty(fileInput, 'files', { value: dt.files, configurable: true });
          } catch (_) {}
          fileInput.dispatchEvent(new Event('change', { bubbles: true }));
        }
      }
    });
    welcomeDrop.addEventListener('click', () => document.getElementById('file-bh')?.click());
    welcomeDrop.addEventListener('keydown', e => {
      if (e.key === 'Enter' || e.key === ' ') document.getElementById('file-bh')?.click();
    });
  }

  document.querySelectorAll('.welcome-sample-btn').forEach(btn => {
    btn.addEventListener('click', () => loadDemoSite(btn.dataset.demo));
  });
}

// ── Main init ──────────────────────────────────────────────────────────────────
async function init() {
  initTabs();
  initCellSizeInputs();
  initInterpolationSettings();
  initCollapsibles();
  initTopoUpload();
  initLabImport();
  initGeoMapImport();
  initReset();
  initApiKeyModal();
  initCertaintySlider();
  initTransparencyControls();
  initSurfaceOpacity();
  initVerticalExaggeration();
  initBHSticksToggle();
  initConstraints();
  initRunAI();
  initBuildModel();
  initViewModeButtons();
  initVBHButton();
  initIsopachMap();
  initFenceSection();
  initScreenshot();
  initBackgroundToggle();
  initMeasureTool();
  initModelReport();
  initAnnotations();
  initPlanView();
  initPropertiesTab();
  initLegendRename();
  initTopoClip();
  initCursorCoords();
  initShortcutsModal();
  initGWT();
  initCameraPresets();
  initSession();
  initProjectConfig();
  initInterpretGeology();
  initSettlement();
  initBearingCapacity();
  initPileCapacity();
  initColorPresets();
  initAutoParams();
  initUnitEditor();
  initRiskAssessment();
  initSemanticModel();
  initParameterView();
  initIsosurfaces();
  initScenarioManager();
  initOrientationImport();
  initFoundationExport();
  initBHLogView();
  initLogSubTabs();
  initCPTImport();
  initWelcomeOverlay();

  // Sample tile buttons (left panel)
  document.querySelectorAll('.sample-tile').forEach(btn => {
    btn.addEventListener('click', () => loadDemoSite(btn.dataset.demo));
  });

  window.addEventListener('geomodel:api-key-set', e => {
    AppState.apiKey = e.detail.key;
    AppState.demoMode = !e.detail.key;
    log(e.detail.key ? '✓ API key configured' : 'Demo mode active', 'ok');
  });

  const scene = await initScene('three-canvas');
  AppState.scene = scene;

  initUploader({
    onParsed(boreholes) {
      AppState.rawBoreholes = boreholes;
      AppState.classifiedBH = boreholes.filter(b => b.classified);
      updateInfoPanel();
      updateBHTable();
      setEnabled('btn-run-ai', boreholes.length > 0);
      if (boreholes.some(b => b.classified)) setEnabled('btn-build-model', true);
      hideWelcome();
      log(`Parsed ${boreholes.length} boreholes.`, 'ok');
    },
  });

  initTextInput();

  window.addEventListener('geomodel:data-loaded', e => {
    if (e.detail?.boreholes?.length) hideWelcome();
  });
}

initLayerControls();
initExporter();
init();
