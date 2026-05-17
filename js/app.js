import { initApiKeyModal } from './api-key.js';
import { initUploader } from './data-parser.js';
import { initTextInput } from './text-input.js';
import { runAIAnalysis, interpretGeology, inferStratOrderFromData, inferUnitParameters, generateSemanticModel, oracleRefinement, generateReportNarrative, parseGeologicalFeatures } from './claude-client.js';
import { parseShapesFromClaude, generateShapeBoreholes } from './geo-shapes.js';
import { exportConfig, importConfig } from './project-config.js';
import { buildVoxelGrid, buildVoxelGridMonteCarlo } from './interpolator.js';
import { initScene } from './scene.js';
import { initLayerControls } from './layer-controls.js';
import { initExporter } from './exporter.js';
import { parseConstraints, applyConstraints, constraintSummary } from './constraints.js';
import { compositeBH } from './semantic-engine.js';
import { parseGeoMap } from './geo-map.js';
import { FenceSection } from './fence-section.js';
import { IsopachMap  } from './isopach.js';
import { ModelReport } from './report.js';
import { PlanView } from './plan-view.js';
import { renderPropertiesTable, applyBS5930Colors } from './properties.js';
import { saveSession, loadSession, hasSavedSession } from './session.js';
import { calculateSettlement, renderSettlementResults,
         consolidationTimeCurve, renderConsolidationCurve } from './settlement.js';
import { calculateBearingCapacity, renderBearingResults } from './bearing.js';
import { calculatePileCapacity, renderPileResults } from './pile.js';
import { parseLabCSV } from './lab-import.js';
import { assessRisk, renderRiskReport } from './risk-engine.js';
import { BHLogView } from './bh-log-view.js';
import { CPTLogView } from './cpt-log-view.js';
import { parseCPT } from './data-parser.js';
import { computeOrientations, orientationStats, renderStereonet, renderRoseDiagram } from './stereonet.js';
import { bishopAnalysis, renderSlopeSection } from './slope-stability.js';
import { assessLiquefaction, renderLiquefactionProfile, summarizeCPTLiquefaction } from './liquefaction.js';
import { computeMohrCircle, renderMohrCircle } from './mohr-circle.js';
import { parseSectionFromText, sectionToVirtualBoreholes,
         sketchToVirtualBoreholes, fenceLength } from './section-interpreter.js';
import { SectionSketch } from './section-sketch.js';
import { FourierEncoder } from './geo-implicit.js';
import { ConceptStore, CONCEPT_AXES } from './concept-store.js';
import { encodeGeologicalConcept } from './claude-client.js';

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
  compositingEnabled: false,
  compositingInterval: 1.0,
  monteCarloEnabled: false,
  mcRealisations: 20,
  faultPlanes: [],
  shapeBoreholes: [],
  sectionBoreholes: [],   // virtual BHs from section descriptions/sketches
  conceptStore: null,     // ConceptStore — geological concept embeddings for neural field
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

  // Monte Carlo toggle
  document.getElementById('mc-enabled')?.addEventListener('change', e => {
    AppState.monteCarloEnabled = e.target.checked;
  });
  document.getElementById('mc-n')?.addEventListener('change', e => {
    AppState.mcRealisations = parseInt(e.target.value) || 20;
  });

  // Compositing toggle
  document.getElementById('composite-enabled')?.addEventListener('change', e => {
    AppState.compositingEnabled = e.target.checked;
    const wrap = document.getElementById('composite-interval-wrap');
    if (wrap) wrap.style.display = e.target.checked ? 'block' : 'none';
  });
  document.getElementById('composite-interval')?.addEventListener('change', e => {
    AppState.compositingInterval = parseFloat(e.target.value) || 1.0;
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
    setEnabled('btn-export-blockmodel', false);
    setEnabled('btn-export-bh-csv', false);
    setEnabled('btn-export-ags', false);
    setEnabled('btn-export-props', false);
    setEnabled('btn-auto-params', false);
    setEnabled('btn-isopach', false);
    setEnabled('btn-model-report', false);
    setEnabled('btn-ai-narrative', false);
    setEnabled('btn-validate-model', false);
    setEnabled('btn-compare-methods', false);
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
    setEnabled('btn-parse-features', true);
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
      setEnabled('btn-parse-features', true);
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

      // Apply compositing if enabled
      let bhForModel = AppState.classifiedBH;
      if (AppState.compositingEnabled) {
        const interval = AppState.compositingInterval;
        bhForModel = compositeBH(AppState.classifiedBH, interval);
        log(`Compositing BH data at ${interval}m intervals → ${bhForModel.reduce((s,b)=>s+b.layers.length,0)} intervals`, 'info');
      }

      // Inject shape boreholes (geological feature primitives)
      if (AppState.shapeBoreholes?.length) {
        bhForModel = [...bhForModel, ...AppState.shapeBoreholes];
        log(`Injecting ${AppState.shapeBoreholes.length} geological feature virtual boreholes`, 'info');
      }

      // Inject section boreholes (described or sketched cross-sections)
      if (AppState.sectionBoreholes?.length) {
        bhForModel = [...bhForModel, ...AppState.sectionBoreholes];
        log(`Injecting ${AppState.sectionBoreholes.length} section virtual boreholes`, 'info');
      }

      // Log concept store status
      if (AppState.conceptStore && !AppState.conceptStore.isEmpty) {
        log(`Conceptual model: ${AppState.conceptStore.concepts.length} concept(s) will shape neural field geometry`, 'info');
      }

      // Extract fault planes from constraint text before building
      const constraintText = document.getElementById('constraints-text')?.value?.trim() ?? '';
      if (constraintText && AppState.geoUnits.length) {
        AppState.parsedConstraints = parseConstraints(constraintText, AppState.geoUnits);
      }
      AppState.faultPlanes = (AppState.parsedConstraints ?? [])
        .filter(r => r.type === 'fault')
        .map(r => r.axis === 'x'
          ? { px: r.coord, py: 0, fnx: 1, fny: 0 }
          : { px: 0, py: r.coord, fnx: 0, fny: 1 });
      if (AppState.faultPlanes.length) {
        log(`Fault planes: ${AppState.faultPlanes.length} active — restricting BH search across faults`, 'info');
      }

      const gridOptions = {
        kNeighbors: AppState.kNeighbors, idwPower: AppState.idwPower,
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
        faultPlanes: AppState.faultPlanes,
        conceptStore: AppState.conceptStore ?? null,
        onProgress: (p, loss, meta) => setBuildProgress(p, loss, meta),
      };

      if (AppState.monteCarloEnabled) {
        AppState.voxelGrid = await buildVoxelGridMonteCarlo(
          bhForModel, AppState.geoUnits, AppState.cellSizeH,
          { ...gridOptions, nRealisations: AppState.mcRealisations ?? 20, perturbSigmaM: 0.5 }
        );
      } else {
        AppState.voxelGrid = await buildVoxelGrid(
          bhForModel, AppState.geoUnits, AppState.cellSizeH, gridOptions
        );
      }
      showBuildProgress(false);
      updateInfoPanel();
      AppState.scene.buildVoxels(AppState.voxelGrid, AppState.geoUnits, AppState.classifiedBH);
      AppState.scene.drawConceptDomains?.(AppState.conceptStore);
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
      setEnabled('btn-export-blockmodel', true);
      setEnabled('btn-build-model', true);
      setEnabled('btn-apply-constraints', true);
      setEnabled('btn-isopach', true);
      setEnabled('btn-model-report', true);
      setEnabled('btn-ai-narrative', true);
      setEnabled('btn-validate-model', true);
      setEnabled('btn-compare-methods', true);
      setEnabled('btn-assess-risk', true);
      setEnabled('btn-plan-view', true);
      setEnabled('btn-export-contacts', true);
      setEnabled('btn-export-surfaces', true);
      setEnabled('btn-export-stl', true);
      setEnabled('btn-stereonet', true);
      setEnabled('btn-slope-stability', true);
      window.dispatchEvent(new CustomEvent('geomodel:model-built'));
      // Refresh fence section with new grid data if it's currently visible
      if (AppState.fenceSection?.visible && AppState.fenceSection._lastArgs) {
        const fa = AppState.fenceSection._lastArgs;
        AppState.fenceSection.draw(
          AppState.voxelGrid, AppState.geoUnits,
          fa.normal, fa.centerD, fa.thickness, AppState.classifiedBH, AppState.conceptStore
        );
      }
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

    // Append consolidation time curve if cv data available
    const curve = consolidationTimeCurve(result);
    if (curve) {
      const t50s = curve.t50 < 1 ? `${(curve.t50*12).toFixed(0)} months` : `${curve.t50.toFixed(1)} yrs`;
      const t90s = curve.t90 ? (curve.t90 < 1 ? `${(curve.t90*12).toFixed(0)} months` : `${curve.t90.toFixed(1)} yrs`) : '—';
      const div = document.createElement('div');
      div.style.cssText = 'margin-top:8px;border-top:1px solid var(--border);padding-top:6px';
      div.innerHTML = `<p style="font-size:10px;color:var(--text-mid);margin:0 0 4px">Consolidation rate (cv=${curve.cvEff.toFixed(2)} m²/yr · Hdr=${curve.Hdr.toFixed(1)}m) — t₅₀=${t50s} · t₉₀=${t90s}</p>
        ${renderConsolidationCurve(curve, res.clientWidth || 260, 130)}`;
      res.appendChild(div);
      log(`Consolidation: t₅₀≈${t50s} · t₉₀≈${t90s} · cv=${curve.cvEff.toFixed(2)} m²/yr`, 'ok');
    } else if (result) {
      const hasCalc = result.layers.some(l => l.settlement !== null);
      log(`Settlement: ${hasCalc ? result.total.toFixed(1) + ' mm total (set cv for time curve)' : 'set Cc and e0 in Props tab'}.`, hasCalc ? 'ok' : 'warn');
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
      window.dispatchEvent(new CustomEvent('geomodel:cpt-loaded'));
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
    AppState.planView.draw(AppState.voxelGrid, AppState.geoUnits, AppState.classifiedBH, AppState.conceptStore);
  });

  // Show/hide unit probability selector when mode changes
  document.getElementById('plan-view-mode')?.addEventListener('change', e => {
    const probWrap  = document.getElementById('plan-view-prob-unit-wrap');
    const depthWrap = document.getElementById('plan-view-depth-unit-wrap');
    if (probWrap)  probWrap.style.display  = e.target.value === 'probability' ? 'flex' : 'none';
    if (depthWrap) depthWrap.style.display = e.target.value === 'depth'       ? 'flex' : 'none';
    if (AppState.voxelGrid && AppState.planView?.visible) {
      AppState.planView.draw(AppState.voxelGrid, AppState.geoUnits, AppState.classifiedBH, AppState.conceptStore);
    }
  });

  document.getElementById('plan-view-prob-unit')?.addEventListener('change', () => {
    if (AppState.voxelGrid && AppState.planView?.visible) {
      AppState.planView.draw(AppState.voxelGrid, AppState.geoUnits, AppState.classifiedBH, AppState.conceptStore);
    }
  });

  document.getElementById('plan-view-depth-unit')?.addEventListener('change', () => {
    if (AppState.voxelGrid && AppState.planView?.visible) {
      AppState.planView.draw(AppState.voxelGrid, AppState.geoUnits, AppState.classifiedBH, AppState.conceptStore);
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

  document.getElementById('btn-compare-methods')?.addEventListener('click', async () => {
    if (!AppState.classifiedBH.length) { log('Run AI analysis first.', 'warn'); return; }
    const btn = document.getElementById('btn-compare-methods');
    const out  = document.getElementById('method-comparison-results');
    if (btn) btn.disabled = true;
    if (out) { out.style.display = 'block'; out.innerHTML = '<p class="hint" style="font-size:10px">Running comparison…</p>'; }

    const METHODS = ['idw', 'kriging', 'gp', 'nn', 'rbf'];
    const results = [];
    const siteHistory = document.getElementById('input-site-history')?.value ?? '';

    for (const method of METHODS) {
      try {
        log(`Comparing method: ${method.toUpperCase()}…`, 'info');
        const grid = await buildVoxelGrid(
          AppState.classifiedBH, AppState.geoUnits, AppState.cellSizeH,
          {
            kNeighbors: AppState.kNeighbors, idwPower: AppState.idwPower,
            method, cellSizeZ: AppState.cellSizeZ,
            anisoAzimuth: AppState.anisoAzimuth, anisoRatio: AppState.anisoRatio,
            trendOrder: AppState.trendOrder,
            varRange: AppState.varRange, varSill: AppState.varSill, varNugget: AppState.varNugget,
          }
        );
        const { nx, ny, nz, origin: O, cellSize: cs, cellHeight: ch, unitIds, certainty } = grid;
        const bhs = AppState.classifiedBH.filter(b => !b.synthetic && b.layers?.length);
        const unitById = {};
        AppState.geoUnits.forEach(u => { unitById[u.id] = u; });
        let correct = 0, total = 0, certSum = 0;
        for (const bh of bhs) {
          const ix = Math.max(0, Math.min(nx-1, Math.round((bh.x - O.x) / cs - 0.5)));
          const iy = Math.max(0, Math.min(ny-1, Math.round((bh.y - O.z) / cs - 0.5)));
          for (const layer of bh.layers) {
            if (!layer.unitCode) continue;
            const elev = (bh.groundLevel ?? 0) - (layer.top + layer.base) / 2;
            const iz   = Math.max(0, Math.min(nz-1, Math.round((elev - O.y) / ch - 0.5)));
            const flat = ix + iy * nx + iz * nx * ny;
            const pred = unitById[unitIds[flat]];
            total++;
            certSum += certainty[flat];
            if (pred?.code === layer.unitCode) correct++;
          }
        }
        const accuracy = total > 0 ? (correct / total * 100).toFixed(1) : '0';
        const avgCert  = total > 0 ? (certSum / total * 100).toFixed(1) : '0';
        results.push({ method, accuracy: +accuracy, avgCert: +avgCert });
      } catch (err) {
        results.push({ method, accuracy: null, avgCert: null, err: err.message });
      }
    }

    results.sort((a, b) => (b.accuracy ?? 0) - (a.accuracy ?? 0));

    const LABELS = { idw: 'IDW', kriging: 'Kriging', gp: 'GP', nn: 'NN', rbf: 'RBF' };
    const rows = results.map((r, i) => {
      const acc   = r.accuracy != null ? `${r.accuracy}%` : '—';
      const cert  = r.avgCert  != null ? `${r.avgCert}%`  : '—';
      const accColor = r.accuracy >= 80 ? 'var(--green)' : r.accuracy >= 60 ? '#c8a855' : '#d04040';
      const best  = i === 0 ? ' ★' : '';
      return `<tr><td>${LABELS[r.method] ?? r.method}${best}</td><td style="color:${accColor};font-weight:600">${acc}</td><td>${cert}</td></tr>`;
    }).join('');

    if (out) out.innerHTML = `<table style="width:100%;font-size:10px;border-collapse:collapse">
      <thead><tr style="color:var(--text-muted)"><th style="text-align:left;padding-bottom:3px">Method</th><th>Accuracy</th><th>Certainty</th></tr></thead>
      <tbody>${rows}</tbody>
    </table><p class="hint" style="font-size:9px;margin-top:4px">★ Best accuracy — consider switching to this method</p>`;

    log(`Method comparison complete. Best: ${results[0]?.method?.toUpperCase() ?? '—'} (${results[0]?.accuracy ?? 0}%)`, 'ok');
    if (btn) btn.disabled = false;
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
      AppState.classifiedBH,
      AppState.conceptStore
    );
    window.dispatchEvent(new CustomEvent('geomodel:fence-updated'));
  });

  document.getElementById('fence-export-dxf')?.addEventListener('click', () => {
    if (!AppState.fenceSection?._lastArgs) { log('Show a cross-section first.', 'warn'); return; }
    AppState.fenceSection.exportDXF();
    log('Cross-section exported as DXF.', 'ok');
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

function setBuildProgress(fraction, loss, meta) {
  const fill = document.getElementById('build-progress-fill');
  const pct  = document.getElementById('build-progress-pct');
  if (fill) fill.style.width = `${(fraction * 100).toFixed(0)}%`;
  if (pct)  pct.textContent  = `${(fraction * 100).toFixed(0)}%`;
  // On first call (fraction=0, meta present) report training set composition
  if (fraction === 0 && meta?.nSamples !== undefined) {
    const conceptInfo = meta.nVirtual > 0
      ? ` · ${meta.nVirtual} concept-virtual`
      : '';
    log(`Training: ${meta.nReal} BH samples${conceptInfo} · ${meta.nSamples} total`, 'info');
  }
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
  // Populate unit affinity multiselect in concept panel
  const affinitySel = document.getElementById('concept-unit-affinity');
  if (affinitySel && AppState.geoUnits.length) {
    const prev = new Set(Array.from(affinitySel.selectedOptions).map(o => o.value));
    affinitySel.innerHTML = AppState.geoUnits.map(u =>
      `<option value="${u.code}"${prev.has(u.code) ? ' selected' : ''}>${u.code} — ${u.name}</option>`
    ).join('');
  }

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
  initStereonet();
  initSlopeStability();
  initGeoFeatures();
  initLiquefaction();
  initMohrCircle();
  initSectionInterpreter();
  initConceptPanel();

  // Sample tile buttons (left panel)
  document.querySelectorAll('.sample-tile').forEach(btn => {
    btn.addEventListener('click', () => loadDemoSite(btn.dataset.demo));
  });

  window.addEventListener('geomodel:api-key-set', e => {
    AppState.apiKey = e.detail.key;
    AppState.demoMode = !e.detail.key;
    log(e.detail.key ? '✓ API key configured' : 'Demo mode active', 'ok');
  });

  // Traceability: show concept + BH attribution when hovering a voxel
  window.addEventListener('geomodel:voxel-hover', e => {
    const panel   = document.getElementById('traceability-panel');
    const content = document.getElementById('trace-content');
    if (!panel || !content) return;
    const d = e.detail;
    if (!d) { panel.classList.add('hidden'); return; }
    if (!AppState.conceptStore || AppState.conceptStore.isEmpty) return;

    const attr = _computeAttribution(d.worldX, d.worldY, d.worldZ);
    panel.classList.remove('hidden');
    content.innerHTML = _renderAttribution(attr, d.unitCode);
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

// ── Geological feature shape injection ───────────────────────────────────────
function initGeoFeatures() {
  const btn    = document.getElementById('btn-parse-features');
  const result = document.getElementById('feature-parse-result');
  if (!btn) return;

  btn.addEventListener('click', async () => {
    const text = document.getElementById('input-geo-features')?.value?.trim();
    if (!text) { log('Enter feature descriptions first.', 'warn'); return; }
    if (!AppState.geoUnits.length) { log('Run AI Analysis first to classify units.', 'warn'); return; }

    btn.disabled = true; btn.textContent = '⏳ Parsing features…';
    if (result) result.textContent = 'Parsing…';

    try {
      const bhs    = AppState.classifiedBH.filter(b => !b.synthetic);
      const xs     = bhs.map(b => b.x), ys = bhs.map(b => b.y);
      const bbox   = {
        minX: xs.length ? Math.min(...xs) : 0,
        maxX: xs.length ? Math.max(...xs) : 100,
        minY: ys.length ? Math.min(...ys) : 0,
        maxY: ys.length ? Math.max(...ys) : 100,
        maxGL: Math.max(...bhs.map(b => b.groundLevel ?? 0), 0),
      };
      const apiKey  = sessionStorage.getItem('anthropic_api_key') ?? '';
      const demoMode = !apiKey;

      log(`Parsing geological feature descriptions${demoMode ? ' (demo)' : ' via Claude'}…`, 'info');
      const shapes = await parseGeologicalFeatures(text, AppState.geoUnits, bbox, apiKey, demoMode);

      // Resolve unit codes in shape objects
      const resolved = parseShapesFromClaude(shapes, AppState.geoUnits);

      // Generate virtual boreholes from shape primitives
      const shapeBHs = generateShapeBoreholes(resolved, bbox, AppState.semanticWeight ?? 0.3);
      AppState.shapeBoreholes = shapeBHs;

      const nShapes = resolved.length, nBHs = shapeBHs.length;
      log(`Feature injection: ${nShapes} shape(s) → ${nBHs} virtual observation point(s)`, 'ok');

      if (result) {
        const items = resolved.map(s =>
          `• ${s.feature_type ?? '?'} — ${s.unit?.code ?? s.unit_code ?? 'unknown'} (${((s.confidence ?? 0.5) * 100).toFixed(0)}% confidence)`
        ).join('\n');
        result.textContent = items || 'No features parsed';
      }

      // Show rebuild hint
      setEnabled('btn-build-model', AppState.classifiedBH.length > 0);
      log('Rebuild the 3D model to apply injected shape features.', 'info');
    } catch (err) {
      log(`Feature parsing error: ${err.message}`, 'error');
      if (result) result.textContent = `Error: ${err.message}`;
    } finally {
      btn.disabled = false;
      btn.textContent = '✦ Parse & Inject Features →';
    }
  });
}

// ── Stereonet surface orientation panel ───────────────────────────────────────
function initStereonet() {
  const btn     = document.getElementById('btn-stereonet');
  const sel     = document.getElementById('stereonet-unit');
  const sCanvas = document.getElementById('stereonet-canvas');
  const rCanvas = document.getElementById('rose-canvas');
  const statsEl = document.getElementById('stereonet-stats');
  if (!btn || !sCanvas || !rCanvas) return;

  // Populate unit selector when model changes
  window.addEventListener('geomodel:model-built', () => {
    if (!sel) return;
    sel.innerHTML = '<option value="">All units</option>';
    for (const u of AppState.geoUnits) {
      const opt = document.createElement('option');
      opt.value = u.code; opt.textContent = `${u.code} — ${u.name ?? ''}`;
      sel.appendChild(opt);
    }
  });

  btn.addEventListener('click', () => {
    const grid = AppState.voxelGrid;
    if (!grid) { log('Build the 3D model first.', 'warn'); return; }

    const allOrientations = computeOrientations(grid, AppState.geoUnits);
    const filterCode = sel?.value ?? '';
    const stats = orientationStats(allOrientations);

    // Build orientation set for display
    let displayOrientations = [];
    let displayColor = '#5ab8e0';
    if (filterCode) {
      displayOrientations = allOrientations[filterCode] ?? [];
      const u = AppState.geoUnits.find(u => u.code === filterCode);
      displayColor = u?.color ?? '#5ab8e0';
    } else {
      for (const orients of Object.values(allOrientations)) displayOrientations.push(...orients);
    }

    renderStereonet(sCanvas, displayOrientations, displayColor);
    renderRoseDiagram(rCanvas, displayOrientations, displayColor);

    if (statsEl) {
      if (filterCode && stats[filterCode]) {
        const s = stats[filterCode];
        statsEl.innerHTML = `${filterCode}: mean dip ${s.meanDip}° → ${s.meanDipDir}° · σ=${s.stdDip}° · n=${s.n}`;
      } else {
        const lines = Object.entries(stats).map(([code, s]) =>
          `${code}: ${s.meanDip}°/${s.meanDipDir}°`).join(' | ');
        statsEl.innerHTML = lines || 'No orientation data';
      }
    }
    log(`Stereonet: ${displayOrientations.length} orientation measurements computed`, 'ok');
  });
}

// ── Bishop slope stability panel ──────────────────────────────────────────────
function initSlopeStability() {
  const btn     = document.getElementById('btn-slope-stability');
  const results = document.getElementById('slope-results');
  if (!btn || !results) return;

  btn.addEventListener('click', () => {
    const grid = AppState.voxelGrid;
    if (!grid) { log('Build the 3D model first.', 'warn'); return; }

    const cPrime  = parseFloat(document.getElementById('slope-cprime')?.value) || null;
    const phiDeg  = parseFloat(document.getElementById('slope-phi')?.value) || null;
    const gwtDepth = parseFloat(document.getElementById('slope-gwt')?.value) || null;

    btn.disabled = true; btn.textContent = '⏳ Analysing…';
    log('Running Bishop simplified slope stability analysis…', 'info');

    setTimeout(() => {
      try {
        const result = bishopAnalysis(grid, AppState.geoUnits, {
          cPrime:   cPrime ?? undefined,
          phiDeg:   phiDeg ?? undefined,
          gwtDepth: gwtDepth ?? undefined,
        });
        if (!result) {
          results.innerHTML = '<p class="hint">Could not find a valid slip circle. Check that the model has sufficient extent.</p>';
        } else {
          const svg = renderSlopeSection(result, results.clientWidth || 360, 180);
          results.innerHTML = svg;
          const color = result.Fs < 1.2 ? '#e84040' : result.Fs < 1.5 ? '#e8924a' : '#4ae87a';
          results.innerHTML += `<p style="margin:4px 0 0;font-size:10px;font-family:monospace;color:${color}">Fs = ${result.Fs.toFixed(2)} · ${result.unitName} · c′=${result.params.cPrime.toFixed(0)}kPa · φ=${result.params.phiDeg}° · γ=${result.params.gamma}kN/m³</p>`;
          log(`Bishop: Fs = ${result.Fs.toFixed(2)} (${result.Fs < 1.2 ? 'UNSAFE' : result.Fs < 1.5 ? 'MARGINAL' : 'STABLE'})`, result.Fs < 1.2 ? 'error' : result.Fs < 1.5 ? 'warn' : 'ok');
        }
      } catch (err) {
        results.innerHTML = `<p class="hint" style="color:#e84040">Error: ${err.message}</p>`;
        log(`Slope stability error: ${err.message}`, 'error');
      } finally {
        btn.disabled = false; btn.textContent = '⚖ Run Bishop Analysis';
      }
    }, 10);
  });
}

function initLiquefaction() {
  const btn      = document.getElementById('btn-liquefaction');
  const summary  = document.getElementById('liq-summary');
  const profiles = document.getElementById('liq-profiles');
  if (!btn || !summary || !profiles) return;

  // Enable button once CPT logs exist
  window.addEventListener('geomodel:cpt-loaded', () => setEnabled('btn-liquefaction', true));
  // Also enable if CPT data already present when panel first shown
  if (AppState.cptLogs?.length) setEnabled('btn-liquefaction', true);

  btn.addEventListener('click', () => {
    const logs = AppState.cptLogs;
    if (!logs?.length) { log('No CPT data loaded — upload CPT file first.', 'warn'); return; }

    const opts = {
      amax:  parseFloat(document.getElementById('liq-amax')?.value)  || 0.15,
      Mw:    parseFloat(document.getElementById('liq-mw')?.value)    || 7.0,
      gwt:   parseFloat(document.getElementById('liq-gwt')?.value)   || 2.0,
      gamma: parseFloat(document.getElementById('liq-gamma')?.value) || 18.0,
    };

    btn.disabled = true; btn.textContent = '⏳ Computing…';
    log(`Liquefaction assessment: PGA=${opts.amax}g, Mw=${opts.Mw}, GWT=${opts.gwt}m`, 'info');

    setTimeout(() => {
      try {
        const results = summarizeCPTLiquefaction(logs, opts);
        const maxLPI = Math.max(...results.map(r => r.result.lpi));
        const nLiquefy = results.filter(r => r.result.lpi >= 2).length;

        const lpiCol = maxLPI < 2 ? '#4a7c59' : maxLPI < 5 ? '#f1c40f' : maxLPI < 15 ? '#e67e22' : '#c0392b';
        summary.innerHTML = `<div style="background:${lpiCol}22;border:1px solid ${lpiCol};border-radius:4px;padding:5px 8px;margin-bottom:4px">
          <b style="color:${lpiCol}">${nLiquefy}/${results.length} CPT logs at risk</b>
          <span style="color:var(--text-mid)"> · Max LPI = ${maxLPI.toFixed(1)} (${results.find(r=>r.result.lpi===maxLPI)?.result.lpiRating})</span>
        </div>
        <table style="width:100%;font-size:10px;border-collapse:collapse">
          <tr><th style="text-align:left;color:var(--text-mid);padding:2px">CPT</th>
              <th style="color:var(--text-mid);padding:2px">LPI</th>
              <th style="color:var(--text-mid);padding:2px">Rating</th>
              <th style="color:var(--text-mid);padding:2px">Min FS</th></tr>
          ${results.map(r => `<tr>
            <td style="padding:2px;color:var(--text-primary)">${r.log.id}</td>
            <td style="padding:2px;text-align:right;color:${r.lpiCol};font-weight:bold">${r.result.lpi.toFixed(1)}</td>
            <td style="padding:2px;text-align:center;color:${r.lpiCol}">${r.result.lpiRating}</td>
            <td style="padding:2px;text-align:right;color:var(--text-mid)">${isFinite(r.minFS) ? r.minFS.toFixed(2) : '—'}</td>
          </tr>`).join('')}
        </table>`;

        profiles.innerHTML = results.map(r =>
          `<div style="text-align:center">
            <div style="font-size:9px;color:var(--text-mid);margin-bottom:2px">${r.log.id}</div>
            ${renderLiquefactionProfile(r.result, 180, 260)}
          </div>`
        ).join('');

        log(`Liquefaction complete — max LPI ${maxLPI.toFixed(1)}, ${nLiquefy} at-risk CPT logs`, nLiquefy ? 'warn' : 'ok');
      } catch (err) {
        summary.innerHTML = `<p class="hint" style="color:#e84040">Error: ${err.message}</p>`;
        log(`Liquefaction error: ${err.message}`, 'error');
      } finally {
        btn.disabled = false; btn.textContent = '⚡ Run Liquefaction Assessment';
      }
    }, 10);
  });
}

// ── Section Interpreter (text description + sketch) ───────────────────────────
function initSectionInterpreter() {
  // ── Sub-tab switching ──────────────────────────────────────────────────────
  document.querySelectorAll('.section-tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.section-tab-btn').forEach(b => {
        b.style.borderBottomColor = 'transparent';
        b.style.color = 'var(--text-mid)';
      });
      btn.style.borderBottomColor = 'var(--accent)';
      btn.style.color = 'var(--accent)';
      const tab = btn.dataset.stab;
      document.getElementById('section-tab-describe').hidden = tab !== 'describe';
      document.getElementById('section-tab-sketch').hidden   = tab !== 'sketch';
    });
  });

  // ── Fence selector ─────────────────────────────────────────────────────────
  const fenceSel = document.getElementById('section-fence-select');
  const _updateFenceSel = () => {
    if (!fenceSel) return;
    const fence = _getCurrentFence();
    if (fence) {
      const len = fenceLength(fence).toFixed(0);
      fenceSel.innerHTML = `<option value="current">Current fence (${len} m)</option>`;
    } else {
      fenceSel.innerHTML = '<option value="">— draw a fence section first —</option>';
    }
  };
  window.addEventListener('geomodel:fence-updated', _updateFenceSel);

  const _getCurrentFence = () => {
    const args = AppState.fenceSection?._lastArgs;
    if (!args) return null;
    const { grid, normal, centerD } = args;
    if (!grid || !normal) return null;
    const { nx, ny, cellSize: cs, origin: O } = grid;
    // Along-section direction (perpendicular to normal in XZ plane, i.e. Easting/Northing)
    const along = { x: normal.z, z: -normal.x };
    // Project model centre onto section plane
    const cx0  = O.x + nx * cs * 0.5;
    const cz0  = O.z + ny * cs * 0.5;
    const proj  = centerD - (normal.x * cx0 + normal.z * cz0);
    const sx0   = cx0 + normal.x * proj;
    const sz0   = cz0 + normal.z * proj;
    const halfW = Math.max(nx * cs, ny * cs) * 0.6;
    // Fence in geological coords: X=Easting (wx), Y=Northing (wz)
    return {
      startX: sx0 + along.x * (-halfW),  startY: sz0 + along.z * (-halfW),
      endX:   sx0 + along.x *   halfW,   endY:   sz0 + along.z *   halfW,
    };
  };

  // ── Text description path ──────────────────────────────────────────────────
  const parseBtn    = document.getElementById('btn-parse-section');
  const parseResult = document.getElementById('section-parse-result');
  const descArea    = document.getElementById('input-section-desc');

  // Enable parse button once we have units
  window.addEventListener('geomodel:data-ready',  () => setEnabled('btn-parse-section', AppState.geoUnits.length > 0));
  window.addEventListener('geomodel:model-built', () => setEnabled('btn-parse-section', AppState.geoUnits.length > 0));

  parseBtn?.addEventListener('click', async () => {
    const text = descArea?.value?.trim();
    if (!text) { log('Enter a section description first.', 'warn'); return; }
    if (!AppState.geoUnits.length) { log('Run AI classification first.', 'warn'); return; }

    const fence = _getCurrentFence();
    if (!fence) {
      log('Draw a fence section in the 3D view first (right-click → Fence section), then come back.', 'warn');
      parseResult.textContent = '⚠ No fence line — draw one in the 3D view first.';
      return;
    }

    parseBtn.disabled = true;
    parseBtn.textContent = '⏳ Parsing with Claude…';
    parseResult.textContent = '';
    log('Parsing geological section description…', 'info');

    try {
      const parsed = await parseSectionFromText(
        text, AppState.geoUnits, fence,
        AppState.apiKey, AppState.demoMode,
      );

      const gl   = Math.max(...AppState.classifiedBH.map(b => b.groundLevel ?? 0), 0);
      const vbhs = sectionToVirtualBoreholes(parsed, fence, AppState.geoUnits, gl);
      AppState.sectionBoreholes = [...(AppState.sectionBoreholes ?? []), ...vbhs];

      // Extract conceptual statements and encode them into the ConceptStore.
      // This is the core semantic pathway: descriptions → dense embeddings that
      // warp the neural field's coordinate space, shaping the output geometry.
      if (!AppState.conceptStore) AppState.conceptStore = new ConceptStore();
      const statements = parsed.conceptual_statements ?? parsed.semantic_keywords ?? [];
      const conceptIds = [];
      for (const stmt of statements) {
        if (!stmt?.trim()) continue;
        try {
          const emb = await encodeGeologicalConcept(stmt, AppState.apiKey, AppState.demoMode);
          const id  = AppState.conceptStore.add({
            description: stmt,
            embedding:   emb,
            confidence:  parsed.confidence ?? 0.72,
            domain:      { type: 'global' },
          });
          conceptIds.push(id);
        } catch (e) { log(`Concept encode warning: ${e.message}`, 'warn'); }
      }
      _renderConceptList();

      const kws = statements.join(', ') || '—';
      parseResult.innerHTML =
        `<span style="color:#4ae87a">✓ ${vbhs.length} virtual boreholes · ${conceptIds.length} concepts encoded</span><br>
         <span style="color:var(--text-mid)">Concepts: ${kws.slice(0,120)}</span><br>
         <span style="color:var(--text-mid)">Rebuild the 3D model to apply.</span>`;
      log(`Section parsed: ${vbhs.length} virtual BHs · ${conceptIds.length} concepts added to ConceptStore`, 'ok');
      setEnabled('btn-build-model', true);

    } catch (err) {
      parseResult.innerHTML = `<span style="color:#e84040">Error: ${err.message}</span>`;
      log(`Section parse error: ${err.message}`, 'error');
    } finally {
      parseBtn.disabled = false;
      parseBtn.textContent = '✦ Parse Section & Inject →';
    }
  });

  // ── Sketch path ────────────────────────────────────────────────────────────
  const sketchCanvas = document.getElementById('sketch-canvas');
  const sketchInfo   = document.getElementById('sketch-info');
  const unitSel      = document.getElementById('sketch-unit-select');
  const undoBtn      = document.getElementById('btn-sketch-undo');
  const clearBtn     = document.getElementById('btn-sketch-clear');
  const injectBtn    = document.getElementById('btn-sketch-inject');
  let sketch = null;

  const _initSketch = () => {
    if (!sketchCanvas) return;
    if (sketch) sketch.destroy();
    sketch = new SectionSketch(sketchCanvas, sketchInfo);
    const fence = _getCurrentFence();
    if (fence && AppState.geoUnits.length) {
      const gl = Math.max(...AppState.classifiedBH.map(b => b.groundLevel ?? 0), 0);
      sketch.setContext(fence, AppState.geoUnits, 30, AppState.voxelGrid);
      const firstUnit = AppState.geoUnits.find(u => u.code !== 'UNKN');
      if (firstUnit) { sketch.setActiveUnit(firstUnit.code); unitSel.value = firstUnit.code; }
    }
  };

  // Populate unit selector when model is built
  const _populateSketchUnits = () => {
    if (!unitSel || !AppState.geoUnits.length) return;
    unitSel.innerHTML = AppState.geoUnits
      .filter(u => u.code !== 'UNKN')
      .map(u => `<option value="${u.code}" style="color:${u.color}">${u.code} — ${u.name}</option>`)
      .join('');
    if (sketch) sketch.setActiveUnit(AppState.geoUnits.find(u => u.code !== 'UNKN')?.code);
  };
  window.addEventListener('geomodel:model-built', () => { _populateSketchUnits(); _initSketch(); });
  window.addEventListener('geomodel:data-ready',  _populateSketchUnits);

  // Switch to sketch tab → init sketch
  document.querySelector('.section-tab-btn[data-stab="sketch"]')?.addEventListener('click', _initSketch);

  unitSel?.addEventListener('change', () => sketch?.setActiveUnit(unitSel.value));

  undoBtn?.addEventListener('click',  () => {
    sketch?.undoLast();
    if (undoBtn) undoBtn.disabled  = !sketch?.hasStrokes();
    if (clearBtn) clearBtn.disabled = !sketch?.hasStrokes();
    if (injectBtn) injectBtn.disabled = !sketch?.hasStrokes();
  });
  clearBtn?.addEventListener('click', () => {
    sketch?.clearAll();
    if (undoBtn) undoBtn.disabled  = true;
    if (clearBtn) clearBtn.disabled = true;
    if (injectBtn) injectBtn.disabled = true;
  });

  // Enable buttons when sketch has strokes
  if (sketchCanvas) {
    sketchCanvas.addEventListener('mouseup',   _sketchBtnUpdate);
    sketchCanvas.addEventListener('touchend',  _sketchBtnUpdate);
  }
  function _sketchBtnUpdate() {
    const has = sketch?.hasStrokes() ?? false;
    if (undoBtn)  undoBtn.disabled  = !has;
    if (clearBtn) clearBtn.disabled = !has;
    if (injectBtn) injectBtn.disabled = !has;
  }

  injectBtn?.addEventListener('click', () => {
    if (!sketch?.hasStrokes()) return;
    const fence = _getCurrentFence();
    if (!fence) { log('No fence line — draw one first.', 'warn'); return; }
    const gl   = Math.max(...AppState.classifiedBH.map(b => b.groundLevel ?? 0), 0);
    const vbhs = sketch.toVirtualBoreholes(gl);
    if (!vbhs.length) { log('No valid strokes to inject.', 'warn'); return; }

    AppState.sectionBoreholes = [...(AppState.sectionBoreholes ?? []), ...vbhs];
    log(`Sketch: ${vbhs.length} virtual boreholes injected from section sketch. Rebuild model to apply.`, 'ok');
    setEnabled('btn-build-model', true);
    sketch.clearAll();
    _sketchBtnUpdate();
  });
}

function initMohrCircle() {
  const btn     = document.getElementById('btn-mohr');
  const result  = document.getElementById('mohr-result');
  const unitSel = document.getElementById('mohr-unit');
  if (!btn || !result || !unitSel) return;

  // Populate unit selector whenever units are available
  const _populateUnits = () => {
    if (!AppState.geoUnits.length) return;
    unitSel.innerHTML = '<option value="">— select unit —</option>' +
      AppState.geoUnits
        .filter(u => u.code !== 'UNKN')
        .map(u => `<option value="${u.id}">${u.code} — ${u.name}</option>`)
        .join('');
  };
  window.addEventListener('geomodel:model-built', _populateUnits);
  window.addEventListener('geomodel:data-ready',  _populateUnits);

  btn.addEventListener('click', () => {
    const uid      = parseInt(unitSel.value);
    const unit     = AppState.geoUnits.find(u => u.id === uid);
    if (!unit) { log('Select a geological unit first.', 'warn'); return; }

    const depth     = parseFloat(document.getElementById('mohr-depth')?.value)  || 5;
    const gwt       = parseFloat(document.getElementById('mohr-gwt')?.value)    || 2;
    const dSigma    = parseFloat(document.getElementById('mohr-dsigma')?.value) || 0;
    const undrained = document.getElementById('mohr-undrained')?.checked ?? false;

    try {
      const mc  = computeMohrCircle(unit, depth, gwt, dSigma, undrained);
      const svg = renderMohrCircle(mc, result.clientWidth || 280, 210);
      const fsCol = mc.Fs < 1.0 ? '#e84040' : mc.Fs < 1.5 ? '#e8924a' : '#4ae87a';
      result.innerHTML = svg +
        `<p style="margin:4px 0 0;font-size:9px;font-family:monospace;color:var(--text-mid)">
          σ₁'=${mc.sigma1.toFixed(0)} · σ₃'=${mc.sigma3.toFixed(0)} · u=${mc.u.toFixed(0)} kPa
          <span style="color:${fsCol};float:right">FS=${isFinite(mc.Fs)?mc.Fs.toFixed(2):'∞'}</span>
        </p>`;
      log(`Mohr circle: ${unit.code} at ${depth}m — FS = ${isFinite(mc.Fs)?mc.Fs.toFixed(2):'∞'} (${mc.mode})`,
          mc.Fs < 1.0 ? 'error' : mc.Fs < 1.5 ? 'warn' : 'ok');
    } catch (err) {
      result.innerHTML = `<p class="hint" style="color:#e84040">Error: ${err.message}</p>`;
    }
  });

  // Live redraw when inputs change
  ['mohr-depth', 'mohr-gwt', 'mohr-dsigma'].forEach(id => {
    document.getElementById(id)?.addEventListener('change', () => {
      if (unitSel.value) btn.click();
    });
  });
  document.getElementById('mohr-undrained')?.addEventListener('change', () => {
    if (unitSel.value) btn.click();
  });
}

initLayerControls();
initExporter();
init();

// ── Conceptual Model Panel ────────────────────────────────────────────────────
// Users enter free-text geological concepts (palaeochannel E-W, stepped rockhead,
// River Terrace morphology, fault NE-SW). Claude encodes each as a 32-dim embedding
// on CONCEPT_AXES. The embeddings are stored in AppState.conceptStore and warp the
// neural implicit field's coordinate space, making output geometry reflect conceptual input.

const CONCEPT_LIBRARY = [
  { label: 'Palaeochannel E-W', axes: { east_west_elongation: 0.9, channel_morphology: 1.0, erosional_contact: 0.9, gravel_basal_lag: 0.8, incision_depth_ratio: 0.8 } },
  { label: 'Palaeochannel N-S', axes: { north_south_elongation: 0.9, channel_morphology: 1.0, erosional_contact: 0.9, gravel_basal_lag: 0.8, incision_depth_ratio: 0.8 } },
  { label: 'River Terrace',     axes: { horizontal_layering: 0.7, lateral_continuity: 0.8, gravel_basal_lag: 0.7, fining_upward: 0.4, erosional_contact: 0.6 } },
  { label: 'Fault E-W (stepped)', axes: { fault_controlled: 1.0, stepped_boundary: 0.9, structural_complexity: 0.7 } },
  { label: 'Fault N-S (stepped)', axes: { fault_controlled: 1.0, stepped_boundary: 0.9, structural_complexity: 0.7, deepens_east: 0.5 } },
  { label: 'Deepening NE',      axes: { deepens_north: 0.6, deepens_east: 0.6, inclined_bedding: 0.5, dip_magnitude: 0.5 } },
  { label: 'Karst / Dissolution', axes: { dissolution_features: 1.0, irregular_base: 0.9, structural_complexity: 0.5 } },
  { label: 'Bedded / Tabular',  axes: { horizontal_layering: 0.9, lateral_continuity: 0.9, vertical_anisotropy: 0.7 } },
  { label: 'Dome / Anticline',  axes: { dome_anticline: 0.9, lateral_continuity: 0.6 } },
  { label: 'Sand Lens / Pod',   axes: { channel_morphology: 0.5, lateral_thinning_north: 0.6, lateral_thinning_south: 0.6, lateral_continuity: -0.5 } },
];

function _libraryEmbedding(axes) {
  const emb = new Float32Array(32);
  emb[26] = 0.7; // data_confidence default
  for (const [name, val] of Object.entries(axes)) {
    const i = CONCEPT_AXES.indexOf(name);
    if (i >= 0) emb[i] = val;
  }
  return emb;
}

function _initConceptLibrary() {
  // Toggle header
  const toggle = document.getElementById('concept-lib-toggle');
  const body   = document.getElementById('concept-lib');
  if (toggle && body) {
    toggle.addEventListener('click', () => {
      const hidden = body.hasAttribute('hidden');
      if (hidden) body.removeAttribute('hidden'); else body.setAttribute('hidden', '');
      const arrow = toggle.querySelector('.collapse-arrow');
      if (arrow) arrow.textContent = hidden ? '⌄' : '›';
    });
  }

  const grid = document.getElementById('concept-lib-grid');
  if (!grid) return;

  grid.innerHTML = CONCEPT_LIBRARY.map((tmpl, idx) =>
    `<button class="concept-lib-chip" data-lib-idx="${idx}" title="${Object.entries(tmpl.axes).map(([k,v]) => k+': '+v.toFixed(1)).join(', ')}">${tmpl.label}</button>`
  ).join('');

  grid.addEventListener('click', e => {
    const btn = e.target.closest('[data-lib-idx]');
    if (!btn) return;
    const tmpl = CONCEPT_LIBRARY[parseInt(btn.dataset.libIdx)];
    if (!tmpl) return;
    const emb  = _libraryEmbedding(tmpl.axes);
    AppState.conceptStore.add({ description: tmpl.label, embedding: emb, confidence: 0.75, domain: { type: 'global' } });
    _renderConceptList();
    _updateConceptInfluenceBar();
    _saveConceptStore();
    log(`Library concept added: "${tmpl.label}"`, 'ok');
  });
}

function _updateConceptInfluenceBar() {
  const el = document.getElementById('concept-global-tensor');
  if (!el || !AppState.conceptStore) return;
  const t = AppState.conceptStore.globalTensor();
  el.textContent = `Global warp: E-W ×${t.Ax.toFixed(1)} · N-S ×${t.Ay.toFixed(1)} · Z ×${t.Az.toFixed(1)}`;
  el.style.display = AppState.conceptStore.isEmpty ? 'none' : 'block';
}

function _saveConceptStore() {
  try {
    sessionStorage.setItem('geomodel:concepts', AppState.conceptStore.serialize());
  } catch { /* quota or private-mode — silently ignore */ }
}

// ── Concept store export/import ──────────────────────────────────────────────
window._exportConceptStore = function() {
  if (!AppState.conceptStore || AppState.conceptStore.isEmpty) {
    log('No concepts to export', 'warn'); return;
  }
  const json = AppState.conceptStore.serialize();
  const blob = new Blob([json], { type: 'application/json' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href = url; a.download = 'geomodel-concepts.json'; a.click();
  URL.revokeObjectURL(url);
  log(`Exported ${AppState.conceptStore.concepts.length} concepts`, 'ok');
};

window._importConceptStore = function(inputEl) {
  const file = inputEl?.files?.[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = e => {
    try {
      const imported = ConceptStore.deserialize(e.target.result);
      if (!AppState.conceptStore) AppState.conceptStore = new ConceptStore();
      // Merge imported concepts into the current store
      for (const c of imported.concepts) {
        AppState.conceptStore.add({
          description:  c.description,
          embedding:    c.embedding,
          confidence:   c.confidence,
          domain:       c.domain,
          unitAffinity: c.unitAffinity,
        });
      }
      _renderConceptList();
      _saveConceptStore();
      log(`Imported ${imported.concepts.length} concepts from ${file.name}`, 'ok');
    } catch (err) { log(`Import failed: ${err.message}`, 'error'); }
    inputEl.value = ''; // reset file input
  };
  reader.readAsText(file);
};

// Update concept confidence inline (called from sparkline row button)
window._updateConceptConf = function(id, val) {
  const c = AppState.conceptStore?.concepts.find(c => c.id === id);
  if (!c) return;
  c.confidence = parseFloat(val);
  _renderConceptList();
  _saveConceptStore();
  _updateConceptInfluenceBar();
};

function initConceptPanel() {
  if (!AppState.conceptStore) AppState.conceptStore = new ConceptStore();

  // Restore persisted concepts from sessionStorage (overrides demo defaults if present)
  const saved = sessionStorage.getItem('geomodel:concepts');
  if (saved) {
    AppState.conceptStore = ConceptStore.deserialize(saved);
  } else {
    _loadDemoConceptsIfEmpty();
  }

  const textarea     = document.getElementById('concept-description');
  const confidence   = document.getElementById('concept-confidence');
  const confLabel    = document.getElementById('concept-confidence-val');
  const domainSel    = document.getElementById('concept-domain');
  const drawBboxRow  = document.getElementById('concept-draw-bbox-row');
  const drawBboxBtn  = document.getElementById('btn-draw-bbox');
  const bboxPreview  = document.getElementById('concept-bbox-preview');
  const encodeBtn    = document.getElementById('btn-encode-concept');
  const clearBtn     = document.getElementById('btn-clear-concepts');
  const listEl       = document.getElementById('concept-list');

  // Drawn bbox domain stored when user completes a plan-view drag
  let _drawnBboxDomain = null;

  // Render concept library chips and scenario list
  _initConceptLibrary();

  // Wire scenario section toggle
  const scToggle = document.getElementById('concept-scenario-toggle');
  const scBody   = document.getElementById('concept-scenario-body');
  if (scToggle && scBody) {
    scToggle.addEventListener('click', () => {
      const hidden = scBody.hasAttribute('hidden');
      if (hidden) { scBody.removeAttribute('hidden'); _renderScenarioList(); }
      else scBody.setAttribute('hidden', '');
      const arrow = scToggle.querySelector('.collapse-arrow');
      if (arrow) arrow.textContent = hidden ? '⌄' : '›';
    });
  }

  confidence?.addEventListener('input', () => {
    if (confLabel) confLabel.textContent = parseFloat(confidence.value).toFixed(2);
  });

  // Show/hide draw bbox row based on domain selection
  domainSel?.addEventListener('change', () => {
    const isDraw = domainSel.value === 'draw';
    if (drawBboxRow) drawBboxRow.style.display = isDraw ? 'flex' : 'none';
    if (!isDraw) { _drawnBboxDomain = null; if (bboxPreview) bboxPreview.textContent = ''; }
  });

  // Trigger plan view bbox drawing
  drawBboxBtn?.addEventListener('click', () => {
    if (!AppState.planView) {
      log('Open the Plan View first (press P or use toolbar)', 'warn'); return;
    }
    if (!AppState.planView.visible) {
      AppState.planView.show();
      AppState.planView.draw(AppState.voxelGrid, AppState.geoUnits, AppState.classifiedBH, AppState.conceptStore);
    }
    AppState.planView.startBboxDraw(domain => {
      _drawnBboxDomain = { type: 'bbox', ...domain, sigma: 30 };
      if (bboxPreview) {
        const w = (domain.maxX - domain.minX).toFixed(0);
        const d = (domain.maxY - domain.minY).toFixed(0);
        bboxPreview.textContent = `${w}×${d}m`;
      }
      log(`Bbox domain set: ${(domain.maxX - domain.minX).toFixed(0)}m × ${(domain.maxY - domain.minY).toFixed(0)}m`, 'ok');
    });
    log('Drag on the Plan View to draw the concept spatial domain', 'info');
  });

  encodeBtn?.addEventListener('click', async () => {
    const text = textarea?.value?.trim();
    if (!text) return;

    encodeBtn.disabled   = true;
    encodeBtn.textContent = '⟳ Encoding…';
    try {
      const emb  = await encodeGeologicalConcept(text, AppState.apiKey, AppState.demoMode);
      const conf = parseFloat(confidence?.value ?? 0.7);
      // Use drawn bbox if user completed a plan-view draw, else fallback to selector
      const domainType = domainSel?.value ?? 'global';
      const domain = domainType === 'draw'
        ? (_drawnBboxDomain ?? { type: 'global' })
        : _buildDomain(domainType);
      // Collect selected unit affinity codes (multi-select)
      const unitAffinitySel = document.getElementById('concept-unit-affinity');
      const unitAffinity = unitAffinitySel
        ? Array.from(unitAffinitySel.selectedOptions).map(o => o.value)
        : [];
      AppState.conceptStore.add({ description: text, embedding: emb, confidence: conf, domain, unitAffinity });
      _renderConceptList();
      _saveConceptStore();
      if (textarea) textarea.value = '';
      // Clear selection after encoding
      if (unitAffinitySel) Array.from(unitAffinitySel.options).forEach(o => o.selected = false);
      // Reset drawn bbox after encoding so it's not accidentally reused
      _drawnBboxDomain = null;
      if (bboxPreview) bboxPreview.textContent = '';
      log(`Concept encoded: "${text.slice(0, 60)}" — ${AppState.conceptStore.concepts.length} total`, 'ok');
    } catch (err) {
      log(`Concept encode error: ${err.message}`, 'error');
    } finally {
      encodeBtn.disabled   = false;
      encodeBtn.textContent = '✦ Encode Concept →';
    }
  });

  clearBtn?.addEventListener('click', () => {
    AppState.conceptStore.clear();
    _renderConceptList();
    _saveConceptStore();
    log('Conceptual model cleared', 'info');
  });

  _renderConceptList();
}

function _buildDomain(type) {
  if (type === 'global') return { type: 'global' };
  // BBox from current plan view bounds if available
  if (type === 'site' && AppState.classifiedBH.length) {
    const xs = AppState.classifiedBH.map(b => b.x);
    const ys = AppState.classifiedBH.map(b => b.y);
    return {
      type: 'bbox',
      minX: Math.min(...xs), maxX: Math.max(...xs),
      minY: Math.min(...ys), maxY: Math.max(...ys),
      sigma: 30,
    };
  }
  return { type: 'global' };
}

// ── Concept scenario management ───────────────────────────────────────────────
// Scenarios are saved to sessionStorage as a list of named ConceptStore snapshots.
// Allows A/B comparison between interpretations without losing work.

const SCENARIOS_KEY = 'geomodel:concept-scenarios';

function _loadScenarios() {
  try { return JSON.parse(sessionStorage.getItem(SCENARIOS_KEY) ?? '[]'); } catch { return []; }
}

function _saveScenarios(scenarios) {
  try { sessionStorage.setItem(SCENARIOS_KEY, JSON.stringify(scenarios)); } catch {}
}

window._saveConceptScenario = function() {
  if (!AppState.conceptStore || AppState.conceptStore.isEmpty) {
    log('No concepts to save as scenario', 'warn'); return;
  }
  const name = prompt('Scenario name:', `Scenario ${_loadScenarios().length + 1}`);
  if (!name?.trim()) return;
  const scenarios = _loadScenarios();
  scenarios.push({ name: name.trim(), json: AppState.conceptStore.serialize(), savedAt: Date.now() });
  _saveScenarios(scenarios);
  _renderScenarioList();
  log(`Scenario "${name.trim()}" saved (${AppState.conceptStore.concepts.length} concepts)`, 'ok');
};

window._loadConceptScenario = function(idx) {
  const scenarios = _loadScenarios();
  const sc = scenarios[parseInt(idx)];
  if (!sc) return;
  AppState.conceptStore = ConceptStore.deserialize(sc.json);
  _renderConceptList();
  _saveConceptStore();
  log(`Scenario "${sc.name}" loaded — ${AppState.conceptStore.concepts.length} concepts`, 'ok');
};

window._deleteConceptScenario = function(idx) {
  const scenarios = _loadScenarios();
  const sc = scenarios.splice(parseInt(idx), 1)[0];
  _saveScenarios(scenarios);
  _renderScenarioList();
  if (sc) log(`Scenario "${sc.name}" deleted`, 'info');
};

function _renderScenarioList() {
  const el = document.getElementById('concept-scenario-list');
  if (!el) return;
  const scenarios = _loadScenarios();
  if (!scenarios.length) {
    el.innerHTML = '<div class="concept-empty" style="font-size:10px">No saved scenarios.</div>';
    return;
  }
  el.innerHTML = scenarios.map((sc, i) => {
    const dt = new Date(sc.savedAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const store = ConceptStore.deserialize(sc.json);
    return `<div class="scenario-entry">
      <button class="scenario-load" onclick="_loadConceptScenario(${i})" title="Load this scenario">${sc.name}</button>
      <span class="scenario-meta">${store.concepts.length}c · ${dt}</span>
      <button class="scenario-del" onclick="_deleteConceptScenario(${i})" title="Delete">×</button>
    </div>`;
  }).join('');
}

// ── Concept coherence scoring ─────────────────────────────────────────────────
// Computes how well the built voxel model's spatial geometry agrees with each
// concept's stated axes. Checks E-W elongation, N-S elongation, and layering.
// Returns [{conceptId, score, details}] — score 0–1 (1 = perfect agreement).
export function computeConceptCoherence() {
  const grid = AppState.voxelGrid;
  const store = AppState.conceptStore;
  if (!grid || !store || store.isEmpty) return [];

  const { nx, ny, nz, cellSize: cs, cellHeight: ch, origin: O, unitIds } = grid;
  const geoUnits = AppState.geoUnits;
  const unitById = {};
  geoUnits.forEach(u => { unitById[u.id] = u; });

  // For each unit: compute centroid and bounding box of its occupied voxels
  const unitBounds = {};
  for (let iz = 0; iz < nz; iz++) {
    for (let iy = 0; iy < ny; iy++) {
      for (let ix = 0; ix < nx; ix++) {
        const uid = unitIds[ix + iy * nx + iz * nx * ny];
        if (!uid) continue;
        if (!unitBounds[uid]) unitBounds[uid] = { minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity, minZ: Infinity, maxZ: -Infinity, count: 0 };
        const b = unitBounds[uid];
        const wx = O.x + (ix + 0.5) * cs;
        const wy = O.z + (iy + 0.5) * cs;
        const wz = O.y + (iz + 0.5) * ch;
        if (wx < b.minX) b.minX = wx; if (wx > b.maxX) b.maxX = wx;
        if (wy < b.minY) b.minY = wy; if (wy > b.maxY) b.maxY = wy;
        if (wz < b.minZ) b.minZ = wz; if (wz > b.maxZ) b.maxZ = wz;
        b.count++;
      }
    }
  }

  return store.concepts.map(c => {
    const emb = c.embedding;
    let score = 0.5; // neutral start
    const details = [];

    // Check all units or only affinity-specified ones
    const targetUnitIds = c.unitAffinity?.length
      ? geoUnits.filter(u => c.unitAffinity.includes(u.code)).map(u => u.id)
      : Object.keys(unitBounds).map(Number);

    for (const uid of targetUnitIds) {
      const b = unitBounds[uid];
      if (!b || b.count < 3) continue;
      const spanX = b.maxX - b.minX;   // E-W extent
      const spanY = b.maxY - b.minY;   // N-S extent
      const spanZ = b.maxZ - b.minZ;   // vertical extent

      // E-W elongation: concept says spanX >> spanY
      const ewAxis = emb[3]; // east_west_elongation
      const nsAxis = emb[4]; // north_south_elongation
      if (Math.abs(ewAxis) > 0.3) {
        const actualRatio = spanX / Math.max(spanY, 0.1);
        const expectedRatio = Math.exp(ewAxis * 1.4) / Math.exp(nsAxis * 1.4);
        const match = 1 - Math.min(1, Math.abs(Math.log(actualRatio / expectedRatio)) / 1.5);
        details.push({ axis: 'E-W elongation', expected: ewAxis.toFixed(2), actual: actualRatio.toFixed(1), match });
        score += (match - 0.5) * 0.3 * Math.abs(ewAxis);
      }

      // Depth incision: concept says spanZ >> horiz span
      const inciAxis = emb[29]; // incision_depth_ratio
      if (Math.abs(inciAxis) > 0.3) {
        const actualDepth = spanZ / Math.max(Math.min(spanX, spanY), 0.1);
        const expectedDepth = Math.exp(inciAxis * 0.8);
        const match = 1 - Math.min(1, Math.abs(Math.log(actualDepth / Math.max(expectedDepth, 0.1))) / 1.5);
        details.push({ axis: 'incision depth', expected: inciAxis.toFixed(2), actual: actualDepth.toFixed(2), match });
        score += (match - 0.5) * 0.2 * Math.abs(inciAxis);
      }

      // Layering: concept says spanX/spanY >> spanZ (horizontal flat body)
      const layerAxis = emb[0]; // horizontal_layering
      if (Math.abs(layerAxis) > 0.3) {
        const actualFlat = Math.max(spanX, spanY) / Math.max(spanZ, 0.01);
        const flat = layerAxis > 0 ? (actualFlat > 3 ? 1 : actualFlat / 3) : (actualFlat < 3 ? 1 : 3 / actualFlat);
        details.push({ axis: 'horizontal layering', expected: layerAxis.toFixed(2), actual: actualFlat.toFixed(1), match: flat });
        score += (flat - 0.5) * 0.2 * Math.abs(layerAxis);
      }
    }

    score = Math.max(0, Math.min(1, score));
    return { conceptId: c.id, description: c.description, score, details };
  });
}

// Show coherence scores when triggered (button in concept panel)
window._showConceptCoherence = function() {
  const results = computeConceptCoherence();
  if (!results.length) { log('Build the 3D model first, then check coherence.', 'warn'); return; }
  const el = document.getElementById('concept-coherence-output');
  if (!el) return;
  el.innerHTML = results.map(r => {
    const pct = (r.score * 100).toFixed(0);
    const col = r.score > 0.65 ? 'var(--accent)' : r.score > 0.4 ? '#f0b429' : 'var(--red)';
    return `<div class="coherence-row">
      <span class="coherence-name" title="${r.description}">${r.description.slice(0, 35)}…</span>
      <div class="coherence-bar-wrap"><div class="coherence-bar" style="width:${pct}%;background:${col}"></div></div>
      <span class="coherence-pct" style="color:${col}">${pct}%</span>
    </div>` +
    (r.details.length
      ? `<div class="coherence-details">${r.details.map(d => `<span>${d.axis}: exp ${d.expected} · got ${d.actual} · ${(d.match*100).toFixed(0)}%</span>`).join(' | ')}</div>`
      : '');
  }).join('');
  el.style.display = 'block';
};

export function _renderConceptList() {
  const listEl = document.getElementById('concept-list');
  if (!listEl || !AppState.conceptStore) return;
  const concepts = AppState.conceptStore.concepts;
  if (!concepts.length) {
    listEl.innerHTML = '<div class="concept-empty">No concepts encoded yet.</div>';
    _updateConceptInfluenceBar();
    return;
  }
  listEl.innerHTML = concepts.map(c => {
    const bars = Array.from(c.embedding).map((v, i) => {
      const pct  = Math.round(Math.abs(v) * 100);
      const col  = v >= 0 ? 'var(--accent)' : 'var(--red)';
      return `<div class="concept-bar-wrap" title="${CONCEPT_AXES[i]}: ${v.toFixed(2)}">
        <div class="concept-bar" style="width:${pct}%;background:${col}"></div>
      </div>`;
    }).join('');
    const dom     = c.domain?.type === 'bbox' ? '⬛ bbox' : '🌐 global';
    const affText = c.unitAffinity?.length ? ` · ${c.unitAffinity.join(',')}` : '';
    const confPct = (c.confidence * 100).toFixed(0);
    return `<div class="concept-entry" data-id="${c.id}">
      <div class="concept-header">
        <span class="concept-desc" title="${c.description}">${c.description.slice(0, 55)}${c.description.length > 55 ? '…' : ''}</span>
        <button class="concept-remove" title="Remove concept" onclick="_removeConcept('${c.id}')">×</button>
      </div>
      <div class="concept-meta">
        <span class="concept-dom-tag">${dom}${affText}</span>
        <label class="concept-conf-row">
          conf <input type="range" class="concept-conf-slider" min="0" max="100" value="${confPct}"
            oninput="this.nextElementSibling.textContent=this.value+'%'; _updateConceptConf('${c.id}', this.value/100)"
          ><span class="concept-conf-val">${confPct}%</span>
        </label>
      </div>
      <div class="concept-axes">${bars}</div>
    </div>`;
  }).join('');
  _updateConceptInfluenceBar();
  // Update 3D scene concept domain boxes (only bbox concepts show a 3D marker)
  AppState.scene?.drawConceptDomains?.(AppState.conceptStore);
}

window._removeConcept = function(id) {
  AppState.conceptStore?.remove(id);
  _renderConceptList();
  _saveConceptStore();
  log(`Concept removed`, 'info');
};

function _loadDemoConceptsIfEmpty() {
  if (!AppState.conceptStore.isEmpty) return;
  // Pre-encode two demo concepts so the app works in demo mode immediately
  const { _demoEmb } = _demoConcepts();
  for (const c of _demoEmb) {
    AppState.conceptStore.add(c);
  }
  setTimeout(_renderConceptList, 0);
}

function _demoConcepts() {
  const palaeochannel = new Float32Array(32);
  palaeochannel[0]=-0.8; palaeochannel[3]=0.9; palaeochannel[4]=-0.7;
  palaeochannel[5]=1.0;  palaeochannel[8]=0.9; palaeochannel[19]=0.6;
  palaeochannel[22]=0.5; palaeochannel[23]=0.8; palaeochannel[26]=0.7;
  palaeochannel[27]=0.9; palaeochannel[29]=0.8;

  const terrace = new Float32Array(32);
  terrace[0]=0.7; terrace[8]=0.6; terrace[9]=0.8;
  terrace[22]=0.4; terrace[23]=0.7; terrace[26]=0.8;

  return { _demoEmb: [
    { description: 'Palaeochannel trending E-W, gravel-filled, with erosional base', embedding: palaeochannel, confidence: 0.75, domain: { type: 'global' } },
    { description: 'River Terrace Deposits — laterally continuous gravel and sand', embedding: terrace, confidence: 0.80, domain: { type: 'global' } },
  ]};
}

// ── Traceability: compute attribution for a world position ────────────────────
// Called from scene.js on voxel hover. Returns object for tooltip rendering.
export function getVoxelAttribution(worldX, worldY, worldZ, unitCode = null) {
  const concepts = AppState.conceptStore
    ? AppState.conceptStore.computeAt(worldX, worldY, worldZ, unitCode)
    : { weights: [], tensor: { Ax: 1, Ay: 1, Az: 1 }, totalWeight: 0, activeAxes: [] };

  // Nearest boreholes at this depth (IDW attribution)
  const depth = (AppState.voxelGrid ? AppState.voxelGrid.origin?.y ?? 0 : 0) - worldZ;
  const bhWeights = _nearestBHWeights(worldX, worldY, worldZ, depth, 3);

  const tensor = concepts.tensor;
  return {
    conceptWeights:    concepts.weights.slice(0, 4),
    bhWeights,
    tensor:            { Ax: tensor.Ax.toFixed(2), Ay: tensor.Ay.toFixed(2), Az: tensor.Az.toFixed(2) },
    semanticDominance: concepts.totalWeight > 0 ? Math.min(1, concepts.totalWeight) : 0,
    activeAxes:        concepts.activeAxes ?? [],
  };
}

function _nearestBHWeights(wx, wy, wz, depth, k) {
  const bhs = AppState.classifiedBH ?? [];
  const candidates = bhs
    .map(bh => {
      const dist = Math.hypot(bh.x - wx, bh.y - wy);
      const layer = bh.layers?.find(l => {
        const d = (bh.groundLevel ?? 0) - wz;
        return d >= l.top && d <= l.base;
      });
      return { id: bh.id, dist, hasLayer: !!layer };
    })
    .filter(c => c.dist < 500)
    .sort((a, b) => a.dist - b.dist)
    .slice(0, k);

  if (!candidates.length) return [];
  const wSum = candidates.reduce((s, c) => s + 1 / (c.dist * c.dist + 0.01), 0);
  return candidates.map(c => ({
    id:     c.id,
    weight: (1 / (c.dist * c.dist + 0.01)) / wSum,
  }));
}

// Internal alias for traceability (same as exported getVoxelAttribution)
function _computeAttribution(worldX, worldY, worldZ) {
  return getVoxelAttribution(worldX, worldY, worldZ);
}

function _renderAttribution(attr, unitCode) {
  if (!attr) return '';
  const { conceptWeights, bhWeights, tensor, semanticDominance, activeAxes } = attr;

  const semPct = (semanticDominance * 100).toFixed(0);
  const datPct = Math.max(0, 100 - semPct).toFixed(0);

  const conceptRows = conceptWeights.length
    ? conceptWeights.map(c =>
        `<div class="trace-row">
          <div class="trace-bar-wrap">
            <div class="trace-bar-fill" style="width:${(c.weight * 100).toFixed(0)}%"></div>
          </div>
          <span class="trace-label" title="${c.description}">${c.description.slice(0, 35)}${c.description.length > 35 ? '…' : ''}</span>
          <span class="trace-weight">${(c.weight * 100).toFixed(0)}%</span>
        </div>`).join('')
    : '<div class="trace-row"><span class="trace-label" style="color:var(--text-dim)">No active concepts</span></div>';

  const bhRows = bhWeights.length
    ? bhWeights.map(b =>
        `<div class="trace-row">
          <div class="trace-bar-wrap">
            <div class="trace-bar-fill" style="width:${(b.weight * 100).toFixed(0)}%;background:var(--text-mid)"></div>
          </div>
          <span class="trace-label">${b.id}</span>
          <span class="trace-weight">${(b.weight * 100).toFixed(0)}%</span>
        </div>`).join('')
    : '<div class="trace-row"><span class="trace-label" style="color:var(--text-dim)">No boreholes nearby</span></div>';

  const axesRows = activeAxes?.length
    ? activeAxes.map(a => {
        const pct  = (Math.abs(a.value) * 100).toFixed(0);
        const col  = a.value >= 0 ? 'var(--accent)' : 'var(--red)';
        const sign = a.value >= 0 ? '+' : '';
        return `<div class="trace-row">
          <div class="trace-bar-wrap">
            <div class="trace-bar-fill" style="width:${pct}%;background:${col}"></div>
          </div>
          <span class="trace-label" style="font-family:var(--font-mono);font-size:10px">${a.name}</span>
          <span class="trace-weight" style="color:${col}">${sign}${a.value.toFixed(2)}</span>
        </div>`;
      }).join('')
    : '<div class="trace-row"><span class="trace-label" style="color:var(--text-dim)">No significant axes</span></div>';

  return `
    <div class="trace-section">
      <div class="trace-section-hdr">Semantic influence: ${semPct}% · Data: ${datPct}%</div>
    </div>
    <div class="trace-section">
      <div class="trace-section-hdr">Concepts</div>
      ${conceptRows}
    </div>
    <div class="trace-section">
      <div class="trace-section-hdr">Nearest Boreholes</div>
      ${bhRows}
    </div>
    <div class="trace-section">
      <div class="trace-section-hdr">Active Geometry Axes</div>
      ${axesRows}
    </div>
    <div class="trace-section">
      <div class="trace-section-hdr">Coordinate Warp</div>
      <div class="trace-warp">E-W ×${tensor.Ax} · N-S ×${tensor.Ay} · Z ×${tensor.Az}</div>
    </div>`;
}
