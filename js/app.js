import { initApiKeyModal } from './api-key.js';
import { initUploader } from './data-parser.js';
import { initTextInput } from './text-input.js';
import { runAIAnalysis } from './claude-client.js';
import { buildVoxelGrid } from './interpolator.js';
import { initScene } from './scene.js';
import { initLayerControls } from './layer-controls.js';
import { initExporter } from './exporter.js';

// ── Global application state ──────────────────────────────────────────────────
export const AppState = {
  apiKey: null,
  step: 1,
  demoMode: false,
  cellSize: 5,        // metres
  kNeighbors: 5,
  idwPower: 2,
  rawBoreholes: [],   // BHLog[] from parser
  geoUnits: [],       // GeoUnit[] after AI classification
  classifiedBH: [],   // classified BHLog[]
  voxelGrid: null,    // { nx, ny, nz, unitIds, certainty, origin, cellSize }
  scene: null,        // Scene manager instance
  hiddenUnits: new Set(),
  certaintyThreshold: 0,
};

// ── Logging utility ────────────────────────────────────────────────────────────
export function log(msg, type = 'info') {
  const el = document.createElement('div');
  el.className = `log-entry log-${type}`;
  el.textContent = msg;
  const log = document.getElementById('status-log');
  log.appendChild(el);
  log.scrollTop = log.scrollHeight;
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
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ── Step navigation ────────────────────────────────────────────────────────────
export function goToStep(n) {
  AppState.step = n;
  document.querySelectorAll('.step').forEach(el => {
    const s = parseInt(el.dataset.step);
    el.classList.toggle('active', s === n);
  });
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  const views = ['view-data', 'view-analysis', 'view-model', 'view-model'];
  const viewId = views[n - 1];
  if (viewId) document.getElementById(viewId)?.classList.add('active');
}

// ── Tab switching ──────────────────────────────────────────────────────────────
function initTabs() {
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const tab = btn.dataset.tab;
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b === btn));
      document.querySelectorAll('.tab-content').forEach(c => {
        c.classList.toggle('active', c.id === `tab-${tab}`);
      });
    });
  });
}

// ── Cell size control ──────────────────────────────────────────────────────────
function initCellSize() {
  const slider = document.getElementById('cell-size');
  const readout = document.getElementById('cell-size-val');
  slider.addEventListener('input', () => {
    AppState.cellSize = parseInt(slider.value);
    readout.textContent = `${slider.value} m`;
  });
}

function initInterpolationSettings() {
  const kSlider  = document.getElementById('k-neighbors');
  const kVal     = document.getElementById('k-neighbors-val');
  const pSlider  = document.getElementById('idw-power');
  const pVal     = document.getElementById('idw-power-val');

  kSlider?.addEventListener('input', () => {
    AppState.kNeighbors = parseInt(kSlider.value);
    kVal.textContent = kSlider.value;
  });
  pSlider?.addEventListener('input', () => {
    AppState.idwPower = parseFloat(pSlider.value);
    pVal.textContent = parseFloat(pSlider.value).toFixed(1);
  });
}

// ── Reset ──────────────────────────────────────────────────────────────────────
function initReset() {
  document.getElementById('btn-reset').addEventListener('click', () => {
    if (!confirm('Reset all data and model?')) return;
    AppState.rawBoreholes = [];
    AppState.geoUnits = [];
    AppState.classifiedBH = [];
    AppState.voxelGrid = null;
    AppState.hiddenUnits = new Set();
    AppState.certaintyThreshold = 0;
    document.getElementById('data-table-body').innerHTML =
      '<tr><td colspan="6" class="table-empty">No data loaded</td></tr>';
    document.getElementById('plan-canvas').getContext('2d')
      ?.clearRect(0, 0, 9999, 9999);
    document.getElementById('analysis-log').innerHTML =
      '<div class="log-entry log-info">Waiting for analysis to start…</div>';
    document.getElementById('unit-legend').innerHTML =
      '<p class="hint">Units appear after analysis</p>';
    document.getElementById('status-log').innerHTML =
      '<div class="log-entry log-info">Reset. Load data or use Demo mode.</div>';
    updateInfoPanel();
    goToStep(1);
    setEnabled('btn-parse', false);
    setEnabled('btn-run-ai', false);
    setEnabled('btn-build-model', false);
    setEnabled('btn-export-gltf', false);
    setEnabled('btn-export-obj', false);
    setEnabled('btn-export-json', false);
    if (AppState.scene) AppState.scene.clear();
  });
}

// ── Helper: enable/disable buttons ────────────────────────────────────────────
export function setEnabled(id, enabled) {
  const el = document.getElementById(id);
  if (el) el.disabled = !enabled;
}

// ── Run AI Analysis button ─────────────────────────────────────────────────────
function initRunAI() {
  document.getElementById('btn-run-ai').addEventListener('click', async () => {
    if (!AppState.rawBoreholes.length) {
      log('No borehole data loaded.', 'warn');
      return;
    }
    goToStep(2);
    setEnabled('btn-run-ai', false);
    document.getElementById('analysis-log').innerHTML = '';
    try {
      const { units, classified } = await runAIAnalysis(
        AppState.rawBoreholes,
        AppState.apiKey,
        AppState.demoMode
      );
      AppState.geoUnits = units;
      AppState.classifiedBH = classified;
      updateLegend();
      log(`Analysis complete — ${units.length} units classified.`, 'ok');
      setEnabled('btn-build-model', true);
      setEnabled('btn-run-ai', true);
    } catch (err) {
      log(`AI analysis failed: ${err.message}`, 'error');
      analysisLog('Error', err.message, 'error');
      setEnabled('btn-run-ai', true);
    }
  });
}

// ── Build 3D Model button ──────────────────────────────────────────────────────
function initBuildModel() {
  document.getElementById('btn-build-model').addEventListener('click', async () => {
    if (!AppState.classifiedBH.length) {
      log('Run AI analysis first.', 'warn');
      return;
    }
    setEnabled('btn-build-model', false);
    log('Building voxel grid…', 'info');
    try {
      await new Promise(r => setTimeout(r, 0)); // allow UI repaint
      AppState.voxelGrid = buildVoxelGrid(
        AppState.classifiedBH, AppState.geoUnits, AppState.cellSize,
        { kNeighbors: AppState.kNeighbors, idwPower: AppState.idwPower }
      );
      updateInfoPanel();
      goToStep(4);
      AppState.scene.buildVoxels(AppState.voxelGrid, AppState.geoUnits, AppState.classifiedBH);
      log(`Voxel model ready — ${AppState.voxelGrid.nx}×${AppState.voxelGrid.ny}×${AppState.voxelGrid.nz} grid.`, 'ok');
      setEnabled('btn-export-gltf', true);
      setEnabled('btn-export-obj', true);
      setEnabled('btn-export-json', true);
      setEnabled('btn-build-model', true);
    } catch (err) {
      log(`Build failed: ${err.message}`, 'error');
      console.error(err);
      setEnabled('btn-build-model', true);
    }
  });
}

// ── Update right-panel info ────────────────────────────────────────────────────
export function updateInfoPanel() {
  const g = AppState.voxelGrid;
  const bh = AppState.rawBoreholes.length;
  document.getElementById('info-bh-count').textContent = bh || '—';
  if (g) {
    document.getElementById('info-voxel-count').textContent =
      (g.nx * g.ny * g.nz).toLocaleString();
    document.getElementById('info-grid-size').textContent =
      `${g.nx}×${g.ny}×${g.nz}`;
    document.getElementById('info-cell-size').textContent =
      `${g.cellSize} m`;
  } else {
    document.getElementById('info-voxel-count').textContent = '—';
    document.getElementById('info-grid-size').textContent = '—';
    document.getElementById('info-cell-size').textContent = '—';
  }
}

// ── Update unit legend ─────────────────────────────────────────────────────────
export function updateLegend() {
  const container = document.getElementById('unit-legend');
  container.innerHTML = '';
  AppState.geoUnits.forEach(unit => {
    const item = document.createElement('div');
    item.className = 'legend-item';
    item.dataset.code = unit.code;
    item.innerHTML = `
      <div class="legend-swatch" style="background:${unit.color}"></div>
      <span class="legend-code">${escHtml(unit.code)}</span>
      <span class="legend-name">${escHtml(unit.name)}</span>
      <span class="legend-eye">👁</span>`;
    item.title = unit.description || unit.name;
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

// ── Transparency controls ──────────────────────────────────────────────────────
function initTransparencyControls() {
  const chk    = document.getElementById('transp-enable');
  const slider = document.getElementById('transp-amount');
  const val    = document.getElementById('transp-val');
  const row    = document.getElementById('transp-slider-row');

  const update = () => {
    const enabled = chk.checked;
    const amount  = parseInt(slider.value) / 100;
    val.textContent = `${slider.value}%`;
    if (row) row.style.opacity = enabled ? '1' : '0.4';
    if (AppState.scene) AppState.scene.setTransparencyMode(enabled, amount);
  };
  chk?.addEventListener('change', update);
  slider?.addEventListener('input', update);
  if (row) row.style.opacity = '0.4'; // starts disabled
}

// ── BH sticks toggle ──────────────────────────────────────────────────────────
function initBHSticksToggle() {
  const chk = document.getElementById('show-bh-sticks');
  chk?.addEventListener('change', () => {
    if (AppState.scene) AppState.scene.toggleBoreholeSticks(chk.checked);
  });
}

// ── Certainty threshold ────────────────────────────────────────────────────────
function initCertaintySlider() {
  const slider = document.getElementById('certainty-threshold');
  const val    = document.getElementById('certainty-val');
  slider.addEventListener('input', () => {
    AppState.certaintyThreshold = parseInt(slider.value) / 100;
    val.textContent = `${slider.value}%`;
    if (AppState.scene) AppState.scene.setCertaintyThreshold(AppState.certaintyThreshold);
  });
}

// ── Step nav clicks ────────────────────────────────────────────────────────────
function initStepNav() {
  document.querySelectorAll('.step').forEach(btn => {
    btn.addEventListener('click', () => {
      const n = parseInt(btn.dataset.step);
      if (n <= AppState.step) goToStep(n);
    });
  });
}

// ── Main init ──────────────────────────────────────────────────────────────────
async function init() {
  initTabs();
  initCellSize();
  initInterpolationSettings();
  initReset();
  initApiKeyModal();
  initStepNav();
  initCertaintySlider();
  initTransparencyControls();
  initBHSticksToggle();
  initRunAI();
  initBuildModel();

  // Listen for API key updates
  window.addEventListener('geomodel:api-key-set', e => {
    AppState.apiKey = e.detail.key;
    AppState.demoMode = !e.detail.key;
    log(e.detail.key ? '✓ API key configured' : 'Demo mode active', 'ok');
  });

  // Init scene (Three.js)
  const scene = await initScene('three-canvas');
  AppState.scene = scene;

  // Init uploader (wires drop zone + parse button)
  initUploader({
    onParsed(boreholes) {
      AppState.rawBoreholes = boreholes;
      document.getElementById('info-bh-count').textContent = boreholes.length;
      setEnabled('btn-run-ai', boreholes.length > 0);
      setEnabled('btn-parse', boreholes.length > 0);
      goToStep(1);
    }
  });

  // Init text input
  initTextInput();

  // Demo button
  document.getElementById('btn-load-demo').addEventListener('click', async () => {
    try {
      log('Loading demo site…', 'info');
      const res  = await fetch('./assets/demo-site.json');
      const data = await res.json();

      // Convert demo format to BHLog format
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
          certainty: l.certainty,
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

      populateDataTable(AppState.rawBoreholes);
      drawPlanView(AppState.rawBoreholes);
      updateLegend();
      updateInfoPanel();
      setEnabled('btn-run-ai', true);
      setEnabled('btn-build-model', true);
      log(`Demo loaded — ${AppState.rawBoreholes.length} boreholes.`, 'ok');
      goToStep(1);

      // Auto-build model
      setTimeout(() => document.getElementById('btn-build-model').click(), 300);
    } catch (err) {
      log(`Demo load failed: ${err.message}`, 'error');
    }
  });

  // Listen for data events from uploader
  window.addEventListener('geomodel:data-loaded', e => {
    populateDataTable(e.detail.boreholes);
    drawPlanView(e.detail.boreholes);
  });
}

// ── Data table population ──────────────────────────────────────────────────────
export function populateDataTable(boreholes) {
  const tbody = document.getElementById('data-table-body');
  tbody.innerHTML = '';
  boreholes.forEach(bh => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${escHtml(bh.id)}</td>
      <td>${bh.x.toFixed(1)}</td>
      <td>${bh.y.toFixed(1)}</td>
      <td>${bh.groundLevel?.toFixed(2) ?? '—'}</td>
      <td>${bh.depth?.toFixed(1) ?? '—'}</td>
      <td>${bh.layers.length}</td>`;
    tbody.appendChild(tr);
  });
}

// ── 2D Plan View ───────────────────────────────────────────────────────────────
export function drawPlanView(boreholes) {
  const canvas = document.getElementById('plan-canvas');
  const wrap   = document.getElementById('plan-view-wrap');
  const pad    = 32;
  canvas.width  = wrap.clientWidth  || 400;
  canvas.height = wrap.clientHeight || 300;

  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (!boreholes.length) return;

  const xs = boreholes.map(b => b.x);
  const ys = boreholes.map(b => b.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const rangeX = maxX - minX || 1;
  const rangeY = maxY - minY || 1;

  const cw = canvas.width  - pad * 2;
  const ch = canvas.height - pad * 2;
  const scale = Math.min(cw / rangeX, ch / rangeY);

  function tx(x) { return pad + (x - minX) * scale; }
  function ty(y) { return pad + (maxY - y) * scale; }

  // Grid
  ctx.strokeStyle = 'rgba(255,255,255,0.06)';
  ctx.lineWidth = 1;
  for (let gx = minX; gx <= maxX + 1; gx += 50) {
    ctx.beginPath();
    ctx.moveTo(tx(gx), pad);
    ctx.lineTo(tx(gx), pad + ch);
    ctx.stroke();
  }
  for (let gy = minY; gy <= maxY + 1; gy += 50) {
    ctx.beginPath();
    ctx.moveTo(pad, ty(gy));
    ctx.lineTo(pad + cw, ty(gy));
    ctx.stroke();
  }

  // Borehole symbols
  const tooltip = document.getElementById('plan-tooltip');

  boreholes.forEach(bh => {
    const cx = tx(bh.x);
    const cy = ty(bh.y);

    ctx.beginPath();
    ctx.arc(cx, cy, 5, 0, Math.PI * 2);
    ctx.fillStyle = '#e8a030';
    ctx.fill();

    ctx.beginPath();
    ctx.moveTo(cx - 8, cy);
    ctx.lineTo(cx + 8, cy);
    ctx.moveTo(cx, cy - 8);
    ctx.lineTo(cx, cy + 8);
    ctx.strokeStyle = 'rgba(232,160,48,0.4)';
    ctx.lineWidth = 1;
    ctx.stroke();

    ctx.fillStyle = '#d8e4f0';
    ctx.font = '10px Inter, sans-serif';
    ctx.fillText(bh.id, cx + 7, cy - 5);
  });

  canvas.onmousemove = evt => {
    const rect = canvas.getBoundingClientRect();
    const mx = evt.clientX - rect.left;
    const my = evt.clientY - rect.top;
    const hit = boreholes.find(bh => {
      const dx = tx(bh.x) - mx, dy = ty(bh.y) - my;
      return Math.hypot(dx, dy) < 10;
    });
    if (hit) {
      tooltip.hidden = false;
      tooltip.style.left = `${mx + 12}px`;
      tooltip.style.top  = `${my - 8}px`;
      tooltip.innerHTML  = `
        <div class="tooltip-title">${escHtml(hit.id)}</div>
        <div class="tooltip-row"><span>X</span><span class="tooltip-val">${hit.x.toFixed(1)} m</span></div>
        <div class="tooltip-row"><span>Y</span><span class="tooltip-val">${hit.y.toFixed(1)} m</span></div>
        <div class="tooltip-row"><span>GL</span><span class="tooltip-val">${hit.groundLevel?.toFixed(2) ?? '—'} mAOD</span></div>
        <div class="tooltip-row"><span>Depth</span><span class="tooltip-val">${hit.depth?.toFixed(1) ?? '—'} m</span></div>`;
    } else {
      tooltip.hidden = true;
    }
  };
  canvas.onmouseleave = () => { tooltip.hidden = true; };
}

// ── Layer controls init (right panel certainty + legend wiring already done) ──
initLayerControls();
initExporter();
init();
