import { initApiKeyModal } from './api-key.js';
import { initUploader } from './data-parser.js';
import { initTextInput } from './text-input.js';
import { runAIAnalysis } from './claude-client.js';
import { buildVoxelGrid } from './interpolator.js';
import { initScene } from './scene.js';
import { initLayerControls } from './layer-controls.js';
import { initExporter } from './exporter.js';
import { parseConstraints, applyConstraints, constraintSummary } from './constraints.js';
import { parseGeoMap } from './geo-map.js';

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
};

// ── Logging utility ────────────────────────────────────────────────────────────
export function log(msg, type = 'info') {
  const el = document.createElement('div');
  el.className = `log-entry log-${type}`;
  el.textContent = msg;
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
    radio.addEventListener('change', () => { AppState.interpMethod = radio.value; });
  });
}

// ── Collapsible sections ───────────────────────────────────────────────────────
function initCollapsibles() {
  [['topo-toggle', 'topo-section'], ['geomap-toggle', 'geomap-section']].forEach(([tid, sid]) => {
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
    setEnabled('btn-build-model', false);
    setEnabled('btn-apply-constraints', false);
    setEnabled('btn-export-gltf', false);
    setEnabled('btn-export-obj', false);
    setEnabled('btn-export-json', false);
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
    setEnabled('btn-run-ai', true);
    setEnabled('btn-build-model', true);
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

// ── Build 3D Model ─────────────────────────────────────────────────────────────
function initBuildModel() {
  document.getElementById('btn-build-model')?.addEventListener('click', async () => {
    if (!AppState.classifiedBH.length) { log('Run AI analysis first.', 'warn'); return; }
    setEnabled('btn-build-model', false);
    log('Building voxel grid…', 'info');
    try {
      await new Promise(r => setTimeout(r, 0));
      AppState.voxelGrid = buildVoxelGrid(
        AppState.classifiedBH, AppState.geoUnits, AppState.cellSizeH,
        { kNeighbors: AppState.kNeighbors, idwPower: AppState.idwPower,
          method: AppState.interpMethod, cellSizeZ: AppState.cellSizeZ }
      );
      updateInfoPanel();
      AppState.scene.buildVoxels(AppState.voxelGrid, AppState.geoUnits, AppState.classifiedBH);
      updateVolumeStats();
      log(`Model ready — ${AppState.voxelGrid.nx}×${AppState.voxelGrid.ny}×${AppState.voxelGrid.nz} grid.`, 'ok');
      setEnabled('btn-export-gltf', true);
      setEnabled('btn-export-obj', true);
      setEnabled('btn-export-json', true);
      setEnabled('btn-build-model', true);
      setEnabled('btn-apply-constraints', true);

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
      updateInfoPanel();
      setEnabled('btn-run-ai', boreholes.length > 0);
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
