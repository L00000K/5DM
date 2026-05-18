import { initApiKeyModal } from './api-key.js';
import { initUploader } from './data-parser.js';
import { initTextInput } from './text-input.js';
import { runAIAnalysis, interpretGeology, inferStratOrderFromData, inferUnitParameters, generateSemanticModel, oracleRefinement, generateReportNarrative, parseGeologicalFeatures, suggestConceptsFromBoreholes, scoreConceptCoherence } from './claude-client.js';
import { parseShapesFromClaude, generateShapeBoreholes } from './geo-shapes.js';
import { exportConfig, importConfig } from './project-config.js';
import { buildVoxelGrid, buildVoxelGridMonteCarlo, buildIndicatorKriging, detectAndCorrectInversions, buildParamVolumes, detectPinchouts, identifySequenceSurfaces, predictBoreholeLog, generateCrossSection } from './interpolator.js';
import { inferGeoImplicit } from './geo-implicit.js';
import { initScene } from './scene.js';
import { initLayerControls } from './layer-controls.js';
import { initExporter } from './exporter.js';
import { parseConstraints, applyConstraints, constraintSummary, inferStratigraphicOrder, applyStratigraphicOrder } from './constraints.js';
import { compositeBH } from './semantic-engine.js';
import { parseGeoMap } from './geo-map.js';
import { FenceSection } from './fence-section.js';
import { StratCorrelation } from './strat-correlation.js';
import { IsopachMap  } from './isopach.js';
import { ModelReport } from './report.js';
import { PlanView } from './plan-view.js';
import { renderPropertiesTable, applyBS5930Colors, GEO_PERIODS } from './properties.js';
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
import { FourierEncoder, measureConceptGeometry, analyzeBoreholeGeometry } from './geo-implicit.js';
import { ConceptStore, CONCEPT_AXES, warpPoint, computeWarpedBounds } from './concept-store.js';
import { encodeGeologicalConcept, refineConceptsWithClaude, extractConceptsFromText, analyseBoreholeGaps, setupConceptsFromSiteDescription, analyseUnitSimilarity, compileGeologicalRules, recommendDrillingLocations } from './claude-client.js';

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
  geoEvents: [],          // Geological event timeline (oldest first)
  stratCorr: null,        // StratCorrelation panel
  trainedModel: null,     // cached neural-implicit trained model for fast re-inference
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

    // Populate manual override fields (only if not already manually set)
    const nugEl = document.getElementById('var-nugget');
    const silEl = document.getElementById('var-sill');
    const ranEl = document.getElementById('var-range');
    const infEl = document.getElementById('var-model-info');
    if (nugEl && !nugEl._manualSet) nugEl.value = fitted.nugget.toFixed(3);
    if (silEl && !silEl._manualSet) silEl.value = (fitted.nugget + fitted.partialSill).toFixed(3);
    if (ranEl && !ranEl._manualSet) ranEl.value  = fitted.range.toFixed(1);
    if (infEl) infEl.textContent = `Spherical model  C₀=${fitted.nugget.toFixed(3)}  C₁=${fitted.partialSill.toFixed(3)}  a=${fitted.range.toFixed(1)}m`;
  }
}

// ── Coordinate reference system conversion ────────────────────────────────────
function initCRSTools() {
  const bngBtn  = document.getElementById('btn-bng-to-wgs84');
  const wgsBtn  = document.getElementById('btn-wgs84-to-bng');
  const result  = document.getElementById('crs-result');
  const gridRef = document.getElementById('crs-gridref');

  bngBtn?.addEventListener('click', async () => {
    const E = parseFloat(document.getElementById('crs-easting')?.value ?? 'NaN');
    const N = parseFloat(document.getElementById('crs-northing')?.value ?? 'NaN');
    if (!isFinite(E) || !isFinite(N)) { if (result) result.textContent = 'Enter easting and northing.'; return; }
    const { bngToWGS84, toOSGridRef } = await import('./crs.js');
    const { lat, lon } = bngToWGS84(E, N);
    if (result) result.textContent = `Lat: ${lat.toFixed(6)}°N  Lon: ${lon.toFixed(6)}°E`;
    const latEl = document.getElementById('crs-lat');
    const lonEl = document.getElementById('crs-lon');
    if (latEl) latEl.value = lat.toFixed(6);
    if (lonEl) lonEl.value = lon.toFixed(6);
    if (gridRef) gridRef.value = toOSGridRef(E, N);
  });

  wgsBtn?.addEventListener('click', async () => {
    const lat = parseFloat(document.getElementById('crs-lat')?.value ?? 'NaN');
    const lon = parseFloat(document.getElementById('crs-lon')?.value ?? 'NaN');
    if (!isFinite(lat) || !isFinite(lon)) { if (result) result.textContent = 'Enter latitude and longitude.'; return; }
    const { wgs84ToBNG, toOSGridRef } = await import('./crs.js');
    const { E, N } = wgs84ToBNG(lat, lon);
    if (result) result.textContent = `E: ${E.toFixed(1)} m  N: ${N.toFixed(1)} m`;
    const eEl = document.getElementById('crs-easting');
    const nEl = document.getElementById('crs-northing');
    if (eEl) eEl.value = E.toFixed(0);
    if (nEl) nEl.value = N.toFixed(0);
    if (gridRef) gridRef.value = toOSGridRef(E, N);
  });
}

// ── Variogram manual controls ────────────────────────────────────────────────
function initVariogramControls() {
  document.getElementById('btn-var-apply')?.addEventListener('click', () => {
    const nugget = parseFloat(document.getElementById('var-nugget')?.value ?? 'NaN');
    const sill   = parseFloat(document.getElementById('var-sill')?.value ?? 'NaN');
    const range  = parseFloat(document.getElementById('var-range')?.value ?? 'NaN');
    if (!isFinite(nugget) || !isFinite(sill) || !isFinite(range)) {
      log('Enter valid nugget, sill, and range values.', 'warn'); return;
    }
    if (sill <= nugget) { log('Sill must be greater than nugget.', 'warn'); return; }
    if (range <= 0)     { log('Range must be positive.', 'warn'); return; }
    AppState.varNugget = nugget;
    AppState.varSill   = sill;
    AppState.varRange  = range;
    // Mark as manually set so auto-fit won't overwrite
    ['var-nugget','var-sill','var-range'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el._manualSet = true;
    });
    const infEl = document.getElementById('var-model-info');
    if (infEl) infEl.textContent = `Manual: C₀=${nugget.toFixed(3)}  C₁=${(sill - nugget).toFixed(3)}  a=${range.toFixed(1)}m  ✓`;
    log(`Variogram: nugget=${nugget.toFixed(3)}, sill=${sill.toFixed(3)}, range=${range.toFixed(1)}m applied.`, 'ok');
  });

  document.getElementById('btn-var-refit')?.addEventListener('click', () => {
    ['var-nugget','var-sill','var-range'].forEach(id => {
      const el = document.getElementById(id);
      if (el) { el._manualSet = false; el.value = ''; }
    });
    _renderVariogram(AppState.classifiedBH);
    log('Variogram auto-refit from data.', 'info');
  });
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
    const text = e.target.result;
    const isAsc = file.name.toLowerCase().endsWith('.asc')
      || text.slice(0, 200).toLowerCase().includes('ncols');

    const points = isAsc ? _parseAscGrid(text) : _parseXYZCSV(text);

    if (points.length < 3) {
      log('Topo file needs at least 3 valid data points. Supports XYZ CSV and Esri ASCII Grid (.asc).', 'warn');
      return;
    }
    AppState.topoPoints = points;
    infoEl.innerHTML = `<div class="file-item">
      <span class="file-name">${escHtml(file.name)}</span>
      <span class="file-size">${points.length} pts</span></div>`;
    if (AppState.scene) {
      AppState.scene.showTopography(points);
      log(`Topography loaded — ${points.length} points${isAsc ? ' (Esri ASCII Grid)' : ''}`, 'ok');
    }
  };
  reader.readAsText(file);
}

function _parseXYZCSV(text) {
  const lines = text.split('\n').map(l => l.trim()).filter(Boolean);
  const points = [];
  for (const line of lines) {
    const parts = line.split(/[,\t ]+/);
    const x = parseFloat(parts[0]), y = parseFloat(parts[1]), z = parseFloat(parts[2]);
    if (!isNaN(x) && !isNaN(y) && !isNaN(z)) points.push({ x, y, z });
  }
  return points;
}

function _parseAscGrid(text) {
  const lines = text.split('\n').map(l => l.trim()).filter(Boolean);
  const hdr = {};
  let dataStart = 0;
  // Parse header key-value pairs
  for (let i = 0; i < lines.length; i++) {
    const m = lines[i].match(/^(\w+)\s+([-\d.eE+]+)/);
    if (m) {
      hdr[m[1].toLowerCase()] = parseFloat(m[2]);
      dataStart = i + 1;
    } else if (lines[i].match(/^[-\d.eE+]/)) {
      dataStart = i;
      break;
    }
  }

  const ncols    = hdr.ncols    ?? hdr.nCols   ?? 0;
  const nrows    = hdr.nrows    ?? hdr.nRows   ?? 0;
  const xllcorner = hdr.xllcorner ?? hdr.xllcenter ?? 0;
  const yllcorner = hdr.yllcorner ?? hdr.yllcenter ?? 0;
  const cellsize  = hdr.cellsize ?? hdr.cellSize ?? 1;
  const nodata    = hdr.nodata_value ?? hdr['nodata_value'] ?? -9999;

  if (!ncols || !nrows) return [];

  const points = [];
  let row = 0;
  for (let li = dataStart; li < lines.length && row < nrows; li++) {
    const vals = lines[li].split(/\s+/).filter(Boolean).map(Number);
    for (let col = 0; col < vals.length && col < ncols; col++) {
      const z = vals[col];
      if (z !== nodata && isFinite(z)) {
        const x = xllcorner + (col + 0.5) * cellsize;
        // ASC rows go top-to-bottom, so row 0 = northernmost
        const y = yllcorner + (nrows - row - 0.5) * cellsize;
        points.push({ x, y, z });
      }
    }
    if (vals.length > 0) row++;
  }

  // Sub-sample if too large (> 40k points → every Nth)
  if (points.length > 40000) {
    const step = Math.ceil(points.length / 40000);
    return points.filter((_, i) => i % step === 0);
  }
  return points;
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
    setEnabled('btn-export-qty-report', false);
    setEnabled('btn-export-bh-csv', false); setEnabled('btn-export-las', false);
    setEnabled('btn-export-ags', false);
    setEnabled('btn-export-props', false);
    setEnabled('btn-auto-params', false);
    setEnabled('btn-isopach', false);
    setEnabled('btn-model-report', false);
    setEnabled('btn-ai-narrative', false);
    setEnabled('btn-validate-model', false);
    setEnabled('btn-loocv', false);
    setEnabled('btn-compare-methods', false);
    setEnabled('btn-assess-risk', false);
    setEnabled('btn-drill-plan', false);
    setEnabled('btn-recommend-drilling', false);
    setEnabled('btn-seq-surfaces', false);
    setEnabled('btn-strat-corr', false);
    setEnabled('btn-plan-view', false);
    setEnabled('btn-export-contacts', false);
    setEnabled('btn-export-surfaces', false);
    setEnabled('btn-param-apply', false);
    setEnabled('btn-param-reset', false);
    setEnabled('btn-build-isosurfaces', false);
    setEnabled('btn-uncertainty-surface', false);
    setEnabled('btn-grade-apply', false);
    setEnabled('btn-crossplot-draw', false); setEnabled('btn-depthplot-draw', false);
    setEnabled('btn-formation-stats', false);
    setEnabled('btn-semantic-model', false);
    setEnabled('btn-depth-vol-compute', false);
    setEnabled('btn-show-concept-influence', false);
    setEnabled('btn-show-dominant-concept', false);
    const _pvp = document.getElementById('prob-vol-panel');
    if (_pvp) _pvp.style.display = 'none';
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

    // Load pre-encoded geological concepts from demo JSON
    if (data.concepts?.length) {
      if (!AppState.conceptStore) AppState.conceptStore = new ConceptStore();
      AppState.conceptStore.clear();
      for (const c of data.concepts) {
        AppState.conceptStore.add({
          description: c.description,
          embedding:   new Float32Array(c.embedding),
          confidence:  c.confidence ?? 0.75,
          domain:      c.domain ?? { type: 'global' },
          unitAffinity: c.unitAffinity ?? [],
        });
      }
      log(`Conceptual model: ${data.concepts.length} geological concept(s) loaded.`, 'ok');
      // Refresh concept panel UI if it's already initialised
      setTimeout(() => { _renderConceptList?.(); _updateConceptInfluenceBar?.(); }, 50);
    }

    updateLegend();
    updateInfoPanel();
    updateBHTable();
    updateBHChart();
    updateStratColumn();
    updateBHUnitStats();
    setEnabled('btn-run-ai', true);
    setEnabled('btn-build-model', true);
    setEnabled('btn-export-bh-csv', true); setEnabled('btn-export-las', true);
      setEnabled('btn-export-ags', true); setEnabled('btn-export-geojson-tops', true);
    setEnabled('btn-export-props', true); setEnabled('btn-formation-stats', true); setEnabled('btn-crossplot-draw', true); setEnabled('btn-depthplot-draw', true);
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
      setEnabled('btn-export-bh-csv', true); setEnabled('btn-export-las', true);
      setEnabled('btn-export-ags', true); setEnabled('btn-export-geojson-tops', true);
      setEnabled('btn-export-props', true); setEnabled('btn-formation-stats', true); setEnabled('btn-crossplot-draw', true); setEnabled('btn-depthplot-draw', true);
      setEnabled('btn-auto-params', true);
      setEnabled('btn-strat-corr', classified.filter(b => !b.synthetic).length >= 2);
      updateLegend();
      updateBHTable();
      updateBHChart();
      updateBHUnitStats();
      AppState.bhLogView?.draw(classified.filter(b => !b.synthetic), units, AppState.voxelGrid);
      if (!document.getElementById('log-sub-spt')?.hidden) drawSPTProfile();
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
      const stratLocked = document.getElementById('strat-manual-lock')?.checked;
      let _stratOrder;
      if (stratLocked && AppState.stratOrder?.length) {
        _stratOrder = AppState.stratOrder;
        log(`Stratigraphic order (manual): ${_stratOrder.join(' → ')}`, 'info');
      } else {
        const inferred = AppState.classifiedBH.length
          ? inferStratOrderFromData(AppState.classifiedBH, AppState.geoUnits)
          : { order: [] };
        _stratOrder = inferred.order;

        // If inference produced no order, try to derive one from concept temporal ranks.
        // Concepts with unitAffinity + temporalOrder suggest which units are older/younger.
        if (!_stratOrder.length && AppState.conceptStore && !AppState.conceptStore.isEmpty) {
          const rankedUnits = [];
          for (const concept of AppState.conceptStore.concepts) {
            if (concept.temporalOrder == null || !concept.unitAffinity?.length) continue;
            for (const code of concept.unitAffinity) {
              if (!AppState.geoUnits.find(u => u.code === code)) continue;
              const existing = rankedUnits.find(r => r.code === code);
              if (!existing) rankedUnits.push({ code, rank: concept.temporalOrder });
              else existing.rank = Math.min(existing.rank, concept.temporalOrder); // take oldest rank
            }
          }
          if (rankedUnits.length >= 2) {
            rankedUnits.sort((a, b) => b.rank - a.rank); // highest rank = youngest (top)
            _stratOrder = rankedUnits.map(r => r.code);
            // Append any remaining units not covered by concepts
            for (const u of AppState.geoUnits) {
              if (!_stratOrder.includes(u.code)) _stratOrder.push(u.code);
            }
            log(`Stratigraphic order (concept temporal ranks): ${_stratOrder.join(' → ')}`, 'info');
          }
        }

        AppState.stratOrder = _stratOrder;
        if (_stratOrder.length) {
          log(`Stratigraphic order (inferred): ${_stratOrder.join(' → ')}`, 'info');
        }
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
        niEpochs: AppState.niEpochs ?? 600,
        oracleApiKey: AppState.oracleEnabled && apiKey ? apiKey : null,
        oracleRefineFn: oracleRefinement,
        demoMode: !apiKey,
        varRange:  AppState.varRange,
        varSill:   AppState.varSill,
        varNugget: AppState.varNugget,
        faultPlanes: AppState.faultPlanes,
        topoPoints:  AppState.topoPoints ?? null,
        conceptStore: AppState.conceptStore ?? null,
        nMCPasses:   AppState.nMCPasses ?? 8,
        // Concept-driven iterative refinement: enabled by default when concepts are active.
        // After first inference, fine-tune on virtual samples in uncertain concept-driven zones.
        conceptRefinement: document.getElementById('ni-concept-refinement')?.checked ?? true,
        onProgress: (p, loss, meta) => setBuildProgress(p, loss, meta),
      };

      if (AppState.monteCarloEnabled) {
        AppState.voxelGrid = await buildVoxelGridMonteCarlo(
          bhForModel, AppState.geoUnits, AppState.cellSizeH,
          { ...gridOptions, nRealisations: AppState.mcRealisations ?? 20, perturbSigmaM: 0.5 }
        );
      } else if (AppState.interpMethod === 'indicator-kriging') {
        AppState.voxelGrid = await buildIndicatorKriging(
          bhForModel, AppState.geoUnits, AppState.cellSizeH,
          { ...gridOptions, range: AppState.varRange, sill: AppState.varSill, nugget: AppState.varNugget }
        );
      } else {
        AppState.voxelGrid = await buildVoxelGrid(
          bhForModel, AppState.geoUnits, AppState.cellSizeH, gridOptions
        );
      }
      // Cache trained neural model for fast concept-ensemble re-inference
      AppState.trainedModel = AppState.voxelGrid?.trainedModel ?? null;
      // Store borehole refs on grid for drilling recommendation BH coverage penalty
      if (AppState.voxelGrid) AppState.voxelGrid._boreholes = bhForModel;

      // Build 3D parameter volumes (SPT, cu, phi, gamma) from borehole test data
      if (AppState.voxelGrid) {
        try {
          const pvols = buildParamVolumes(bhForModel, AppState.geoUnits, AppState.voxelGrid);
          AppState.voxelGrid.paramVolumes = pvols;
          if (pvols.size) log(`Parameter volumes built: ${[...pvols.keys()].join(', ')}`, 'info');
        } catch (e) {
          console.warn('buildParamVolumes error:', e.message);
        }
      }

      // Stratigraphic inversion correction
      if (AppState.voxelGrid && document.getElementById('correct-inversions')?.checked && _stratOrder?.length) {
        try {
          const inv = detectAndCorrectInversions(AppState.voxelGrid, AppState.geoUnits, _stratOrder);
          if (inv.invertedCount > 0) {
            log(`Inversion correction: ${inv.invertedCount} inverted voxels → ${inv.corrections} corrected (${(inv.invertedFraction * 100).toFixed(1)}% of grid)`, 'info');
          }
        } catch (e) {
          console.warn('detectAndCorrectInversions error:', e.message);
        }
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
      setEnabled('btn-export-qty-report', true);
      setEnabled('btn-build-model', true);
      setEnabled('btn-apply-constraints', true);
      setEnabled('btn-isopach', true);
      setEnabled('btn-model-report', true);
      setEnabled('btn-ai-narrative', true);
      setEnabled('btn-validate-model', true);
      setEnabled('btn-loocv', AppState.classifiedBH.filter(b => !b.synthetic).length >= 3);
      setEnabled('btn-compare-methods', true);
      setEnabled('btn-assess-risk', true);
      setEnabled('btn-drill-plan', true);
      setEnabled('btn-recommend-drilling', true);
      setEnabled('btn-seq-surfaces', true);
      setEnabled('btn-strat-corr', AppState.classifiedBH.filter(b => !b.synthetic).length >= 2);
      setEnabled('btn-plan-view', true);
      setEnabled('btn-export-contacts', true);
      setEnabled('btn-export-surfaces', true);
      setEnabled('btn-export-stl', true);
      setEnabled('btn-export-geojson', true);
      setEnabled('btn-stereonet', true);
      setEnabled('btn-slope-stability', true);
      window.dispatchEvent(new CustomEvent('geomodel:model-built'));
      _initCertaintyHistUnit();
      if (!document.getElementById('tab-analysis')?.hidden) drawCertaintyHistogram();
      // Refresh BH log view predictions now that model is available
      {
        const bhs = AppState.classifiedBH.filter(b => !b.synthetic);
        if (bhs.length && AppState.geoUnits.length) {
          AppState.bhLogView?.draw(bhs, AppState.geoUnits, AppState.voxelGrid);
        }
      }
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
      setEnabled('btn-uncertainty-surface', true);
      setEnabled('btn-grade-apply', true);
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
      const postBuildConstraintText = document.getElementById('constraints-text')?.value?.trim() ?? '';
      if (postBuildConstraintText && AppState.geoUnits.length) {
        AppState.parsedConstraints = parseConstraints(postBuildConstraintText, AppState.geoUnits);
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

      // Auto-run concept coherence + geometry verification after neural implicit build
      if (AppState.interpMethod === 'neural-implicit' && AppState.conceptStore && !AppState.conceptStore.isEmpty) {
        const coherence = computeConceptCoherence();
        if (coherence.length) {
          const poor = coherence.filter(r => r.score < 0.5);
          const good = coherence.filter(r => r.score >= 0.65);
          const msg = poor.length
            ? `Concept coherence: ${good.length}/${coherence.length} concepts well-represented. ${poor.length} concept(s) below 50% — check the Concepts tab for details.`
            : `Concept coherence: all ${coherence.length} concept(s) well-represented in the 3D model.`;
          log(msg, poor.length ? 'warn' : 'ok');
          const el = document.getElementById('concept-coherence-output');
          if (el) { window._showConceptCoherence?.(); }
        }

        // Geometry verification: measure E-W vs N-S elongation ratios per unit
        // and compare against concept predictions. Shows user that the concept shaped the geometry.
        const geoCheck = measureConceptGeometry(AppState.voxelGrid, AppState.geoUnits, AppState.conceptStore);
        AppState._lastGeoCheck = geoCheck;
        if (geoCheck.length) {
          _showConceptGeometryReport(geoCheck);
        }
      }

      // Model QC dashboard — auto-runs after every build
      _renderModelQC();
      // Show borehole gap analysis panel (reset ready state)
      const gapPanel = document.getElementById('borehole-gap-panel');
      if (gapPanel) gapPanel.removeAttribute('hidden');
      const gapResults = document.getElementById('borehole-gap-results');
      if (gapResults) { gapResults.style.display = 'none'; gapResults.innerHTML = ''; }

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

      // Restore concept store from project config
      if (cfg.concepts) {
        try {
          const cs = ConceptStore.deserialize(JSON.stringify(cfg.concepts));
          AppState.conceptStore = cs;
          _renderConceptList();
          log(`Restored ${cs.concepts.length} geological concept(s) from project.`, 'ok');
        } catch { /* ignore malformed concept data */ }
      }

      updateLegend();
      updateInfoPanel();
      updateBHTable();
      updateBHChart();
      updateStratColumn();
      updateBHUnitStats();
      renderPropertiesTable(AppState.geoUnits, () => updateLegend());
      AppState.bhLogView?.draw(AppState.classifiedBH.filter(b => !b.synthetic), AppState.geoUnits, AppState.voxelGrid);
      setEnabled('btn-run-ai', true);
      setEnabled('btn-build-model', AppState.classifiedBH.length > 0);
      setEnabled('btn-export-bh-csv', AppState.classifiedBH.length > 0); setEnabled('btn-export-las', AppState.classifiedBH.length > 0);
      setEnabled('btn-export-props', AppState.geoUnits.length > 0); setEnabled('btn-formation-stats', AppState.geoUnits.length > 0); setEnabled('btn-crossplot-draw', AppState.geoUnits.length > 0);
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
    setEnabled('btn-export-props', AppState.geoUnits.length > 0); setEnabled('btn-formation-stats', AppState.geoUnits.length > 0); setEnabled('btn-crossplot-draw', AppState.geoUnits.length > 0);
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

  // Auto-correlate unit descriptions via AI
  document.getElementById('btn-auto-correlate')?.addEventListener('click', async () => {
    const btn = document.getElementById('btn-auto-correlate');
    const out  = document.getElementById('auto-correlate-results');
    if (!out) return;
    if (!AppState.classifiedBH.length) {
      out.style.display = 'block';
      out.innerHTML = '<p class="hint" style="color:var(--red);padding:4px">Load classified borehole data first.</p>';
      return;
    }
    btn.disabled = true; btn.textContent = '✦ Analysing…';
    try {
      const { autoCorrelateUnits } = await import('./claude-client.js');
      const corrections = await autoCorrelateUnits(
        AppState.classifiedBH, AppState.geoUnits,
        AppState.apiKey, AppState.demoMode
      );
      out.style.display = 'block';
      if (!corrections.length) {
        out.innerHTML = '<p style="font-size:10px;color:var(--green);padding:4px">✓ All descriptions correctly classified.</p>';
      } else {
        out.innerHTML = corrections.map(c => `
          <div class="corr-row" style="font-size:10px;border:1px solid var(--border);border-radius:4px;padding:5px 7px;margin-bottom:4px;background:var(--bg-deep)">
            <div style="color:var(--text-mid);margin-bottom:2px;line-height:1.3">"${escHtml(c.description.slice(0, 70))}"</div>
            <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap">
              <span style="color:var(--red)">${escHtml(c.currentCode)}</span>
              <span>→</span>
              <span style="color:var(--green);font-weight:600">${escHtml(c.recommendedCode)}</span>
              <span style="color:var(--text-dim);font-size:9px">${escHtml(c.reason ?? '')} (${((c.confidence ?? 0)*100).toFixed(0)}%)</span>
              <button class="btn-ghost btn-sm" style="font-size:9px;padding:1px 5px;margin-left:auto"
                onclick="window._applyCorr('${escHtml(c.currentCode)}','${escHtml(c.recommendedCode)}', this)">Apply</button>
            </div>
          </div>`).join('');
      }
    } catch (e) {
      out.style.display = 'block';
      out.innerHTML = `<p style="font-size:10px;color:var(--red);padding:4px">Error: ${escHtml(e.message)}</p>`;
    } finally {
      btn.disabled = false; btn.textContent = '✦ Auto-correlate unit descriptions (AI)';
    }
  });
}

window._applyCorr = function(fromCode, toCode, btn) {
  let count = 0;
  for (const bh of AppState.classifiedBH) {
    for (const layer of bh.layers) {
      if (layer.unitCode === fromCode) { layer.unitCode = toCode; count++; }
    }
  }
  updateLegend(); updateBHTable();
  renderPropertiesTable(AppState.geoUnits, () => updateLegend());
  log(`Auto-corr applied: ${fromCode} → ${toCode} (${count} layer(s))`, 'ok');
  btn.textContent = `✓ Applied (${count})`;
  btn.disabled = true;
};

// ── Log sub-tab switcher (BH / CPT / SPT) ────────────────────────────────────
function initLogSubTabs() {
  document.querySelectorAll('.log-sub-tab').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.log-sub-tab').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const key = btn.dataset.logTab;
      document.getElementById('log-sub-bh').hidden        = (key !== 'bh');
      document.getElementById('log-sub-cpt').hidden       = (key !== 'cpt');
      document.getElementById('log-sub-spt').hidden       = (key !== 'spt');
      document.getElementById('log-sub-depthplot').hidden = (key !== 'depthplot');
      if (key === 'spt') drawSPTProfile();
    });
  });

  document.getElementById('btn-depthplot-draw')?.addEventListener('click', drawDepthPlot);
  document.getElementById('depthplot-param')?.addEventListener('change', () => {
    if (!document.getElementById('log-sub-depthplot')?.hidden) drawDepthPlot();
  });
}

function drawDepthPlot() {
  const canvas   = document.getElementById('depthplot-canvas');
  const hintEl   = document.getElementById('depthplot-hint');
  const legendEl = document.getElementById('depthplot-legend');
  if (!canvas) return;

  const paramKey = document.getElementById('depthplot-param')?.value ?? 'N_spt';
  const PARAM_LABELS = {
    N_spt: 'SPT N', cu: 'Cu (kPa)', phi: 'φ′ (°)',
    E: 'E (MPa)', gamma: 'γ (kN/m³)',
  };

  const bhs = AppState.classifiedBH.filter(b => !b.synthetic && b.layers?.length);
  if (!bhs.length) { if (hintEl) hintEl.hidden = false; return; }
  if (hintEl) hintEl.hidden = true;

  // Collect data: for each BH, array of {depth, value}
  const bhData = [];
  let maxDepth = 0, maxVal = 0;

  // Assign a distinct colour to each BH
  const COLORS = [
    '#e05050','#3080e8','#30b860','#e89020','#9848e0',
    '#20b8c8','#e84080','#40c840','#8060e0','#e86020',
  ];

  for (let bi = 0; bi < bhs.length; bi++) {
    const bh = bhs[bi];
    const pts = [];
    for (const layer of bh.layers) {
      const depth = (layer.top + layer.base) / 2;
      let v = null;
      if (paramKey === 'N_spt') v = layer.sptN;
      else {
        const unit = AppState.geoUnits.find(u => u.code === layer.unitCode);
        v = unit?.params?.[paramKey] ?? null;
      }
      if (v != null && isFinite(v)) {
        pts.push({ depth, value: v });
        maxDepth = Math.max(maxDepth, layer.base);
        maxVal   = Math.max(maxVal, v);
      }
    }
    if (pts.length) {
      bhData.push({ id: bh.id, color: COLORS[bi % COLORS.length], pts });
    }
  }

  if (!bhData.length) {
    log(`No "${PARAM_LABELS[paramKey] ?? paramKey}" data in borehole layers.`, 'warn');
    return;
  }

  const W = 260, H = 340, PAD_L = 44, PAD_R = 10, PAD_T = 18, PAD_B = 28;
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, W, H);

  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;

  const toX = (v)  => PAD_L + (v / (maxVal || 1)) * plotW;
  const toY = (d)  => PAD_T + (d / (maxDepth || 1)) * plotH;

  // Unit colour bands (optional)
  const showBands = document.getElementById('depthplot-unit-bands')?.checked ?? true;
  if (showBands) {
    const unitDepths = [];
    for (const bh of bhs.slice(0, 1)) { // use first BH for band extents
      for (const layer of bh.layers) {
        const unit = AppState.geoUnits.find(u => u.code === layer.unitCode);
        if (unit) unitDepths.push({ top: layer.top, base: layer.base, color: unit.color });
      }
    }
    for (const ud of unitDepths) {
      ctx.fillStyle = ud.color + '22';
      ctx.fillRect(PAD_L, toY(ud.top), plotW, toY(ud.base) - toY(ud.top));
    }
  }

  // Grid lines
  ctx.strokeStyle = '#e0e0e0'; ctx.lineWidth = 0.5;
  const nXTicks = 5;
  for (let i = 0; i <= nXTicks; i++) {
    const x = PAD_L + i / nXTicks * plotW;
    ctx.beginPath(); ctx.moveTo(x, PAD_T); ctx.lineTo(x, PAD_T + plotH); ctx.stroke();
  }
  const nYTicks = 6;
  for (let i = 0; i <= nYTicks; i++) {
    const y = PAD_T + i / nYTicks * plotH;
    ctx.beginPath(); ctx.moveTo(PAD_L, y); ctx.lineTo(PAD_L + plotW, y); ctx.stroke();
  }

  // Axes
  ctx.strokeStyle = '#888'; ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(PAD_L, PAD_T); ctx.lineTo(PAD_L, PAD_T + plotH);
  ctx.moveTo(PAD_L, PAD_T + plotH); ctx.lineTo(PAD_L + plotW, PAD_T + plotH);
  ctx.stroke();

  // Axis labels
  ctx.fillStyle = '#555'; ctx.font = '8px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText(PARAM_LABELS[paramKey] ?? paramKey, PAD_L + plotW / 2, H - 4);
  ctx.save(); ctx.translate(10, PAD_T + plotH / 2);
  ctx.rotate(-Math.PI / 2); ctx.fillText('Depth (m bgl)', 0, 0); ctx.restore();

  // Tick values
  ctx.fillStyle = '#888'; ctx.font = '7px sans-serif';
  for (let i = 0; i <= nXTicks; i++) {
    ctx.textAlign = 'center';
    ctx.fillText((i / nXTicks * maxVal).toFixed(0), PAD_L + i / nXTicks * plotW, PAD_T + plotH + 10);
  }
  for (let i = 0; i <= nYTicks; i++) {
    ctx.textAlign = 'right';
    ctx.fillText((i / nYTicks * maxDepth).toFixed(0), PAD_L - 3, PAD_T + i / nYTicks * plotH + 3);
  }

  // Draw each BH as a step/line plot
  for (const bh of bhData) {
    ctx.strokeStyle = bh.color; ctx.lineWidth = 1.5;
    ctx.beginPath();
    let first = true;
    for (const pt of bh.pts) {
      const x = toX(pt.value);
      const y = toY(pt.depth);
      if (first) { ctx.moveTo(x, y); first = false; } else ctx.lineTo(x, y);
    }
    ctx.stroke();
    // Dots at data points
    ctx.fillStyle = bh.color;
    for (const pt of bh.pts) {
      ctx.beginPath();
      ctx.arc(toX(pt.value), toY(pt.depth), 2, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // Legend
  if (legendEl) {
    legendEl.innerHTML = bhData.map(bh =>
      `<span style="display:flex;align-items:center;gap:3px">
        <span style="width:14px;height:2px;background:${bh.color};display:inline-block"></span>
        <span style="font-size:9px">${escHtml(bh.id)}</span>
      </span>`
    ).join('');
  }

  log(`Depth plot: ${PARAM_LABELS[paramKey] ?? paramKey} for ${bhData.length} boreholes`, 'ok');
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

// ── SPT Depth Envelope Profile ────────────────────────────────────────────────
function drawSPTProfile() {
  const canvas  = document.getElementById('spt-profile-canvas');
  const hint    = document.getElementById('spt-profile-hint');
  const expBtn  = document.getElementById('btn-spt-export');
  if (!canvas) return;

  const bhs = AppState.classifiedBH.filter(b => !b.synthetic && b.layers?.length);
  if (!bhs.length) { hint && (hint.hidden = false); canvas.hidden = true; return; }

  const hasSPT = bhs.some(b => b.layers.some(l => l.sptN != null));
  if (!hasSPT) { hint && (hint.hidden = false); canvas.hidden = true; return; }

  hint && (hint.hidden = true);
  canvas.hidden = false;
  if (expBtn) expBtn.disabled = false;

  const showBHs   = document.getElementById('spt-show-bhs')?.checked ?? true;
  const showUnits = document.getElementById('spt-show-units')?.checked ?? false;
  const binSize   = parseFloat(document.getElementById('spt-bin-size')?.value ?? '1');

  // Find total depth range
  let maxDepth = 0;
  bhs.forEach(b => {
    b.layers.forEach(l => { if (l.base > maxDepth) maxDepth = l.base; });
  });
  if (maxDepth < 1) maxDepth = 20;

  // Build depth bins: 0 → maxDepth
  const nBins = Math.ceil(maxDepth / binSize);
  const bins  = Array.from({ length: nBins }, (_, i) => ({
    depth: (i + 0.5) * binSize,
    vals: [],
  }));

  // Collect SPT values into bins from layer midpoint depth
  bhs.forEach(b => {
    b.layers.forEach(l => {
      if (l.sptN == null || l.sptN <= 0) return;
      const midDepth = ((l.top ?? 0) + (l.base ?? 0)) / 2;
      const bi = Math.floor(midDepth / binSize);
      if (bi >= 0 && bi < nBins) bins[bi].vals.push(l.sptN);
    });
  });

  // Statistical summary per bin
  const binStats = bins.map(b => {
    if (!b.vals.length) return { depth: b.depth, min: null, mean: null, max: null, p25: null, p75: null };
    const sorted = [...b.vals].sort((a, c) => a - c);
    const n = sorted.length;
    const mean = sorted.reduce((s, v) => s + v, 0) / n;
    const p25  = sorted[Math.floor(n * 0.25)];
    const p75  = sorted[Math.floor(n * 0.75)];
    return { depth: b.depth, min: sorted[0], max: sorted[n - 1], mean, p25, p75 };
  });

  // Individual BH traces: depth vs sptN per BH
  const bhTraces = bhs.map(b => {
    const pts = [];
    b.layers.forEach(l => {
      if (l.sptN == null || l.sptN <= 0) return;
      const midDepth = ((l.top ?? 0) + (l.base ?? 0)) / 2;
      pts.push({ depth: midDepth, n: l.sptN });
    });
    pts.sort((a, c) => a.depth - c.depth);
    return { id: b.id, pts };
  });

  // Canvas sizing
  const PAD_L = 52, PAD_R = 20, PAD_T = 30, PAD_B = 24;
  const UNIT_BAND_W = showUnits ? 16 : 0;
  const W = 320;
  const H = Math.max(380, nBins * Math.max(6, 320 / nBins));
  canvas.width  = W;
  canvas.height = H;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#f8fafb';
  ctx.fillRect(0, 0, W, H);

  const maxN    = 60;
  const drawW   = W - PAD_L - PAD_R - UNIT_BAND_W;
  const drawH   = H - PAD_T - PAD_B;

  const toX = n => PAD_L + UNIT_BAND_W + (Math.min(n, maxN) / maxN) * drawW;
  const toY = d => PAD_T + (d / maxDepth) * drawH;

  // ── Unit color bands on left ──
  if (showUnits && AppState.geoUnits.length) {
    const unitByCode = {};
    AppState.geoUnits.forEach(u => { unitByCode[u.code] = u; });
    // Collect depth ranges per unit across all BHs
    const unitDepths = {};
    bhs.forEach(b => {
      b.layers.forEach(l => {
        const u = unitByCode[l.unitCode];
        if (!u) return;
        if (!unitDepths[u.code]) unitDepths[u.code] = { top: Infinity, base: -Infinity, color: u.color };
        unitDepths[u.code].top  = Math.min(unitDepths[u.code].top, l.top ?? 0);
        unitDepths[u.code].base = Math.max(unitDepths[u.code].base, l.base ?? 0);
      });
    });
    Object.values(unitDepths).forEach(ud => {
      const y1 = toY(ud.top), y2 = toY(ud.base);
      ctx.fillStyle = ud.color ?? '#ccc';
      ctx.globalAlpha = 0.55;
      ctx.fillRect(PAD_L, y1, UNIT_BAND_W - 1, Math.max(1, y2 - y1));
      ctx.globalAlpha = 1;
    });
    // Band border
    ctx.strokeStyle = '#ccc'; ctx.lineWidth = 0.5;
    ctx.strokeRect(PAD_L, PAD_T, UNIT_BAND_W - 1, drawH);
  }

  // ── Grid lines (vertical N-value) ──
  ctx.strokeStyle = '#dde3ea'; ctx.lineWidth = 0.5;
  ctx.fillStyle = '#7a8a9a'; ctx.font = '9px monospace'; ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (let n = 0; n <= 60; n += 10) {
    const x = toX(n);
    ctx.beginPath(); ctx.moveTo(x, PAD_T); ctx.lineTo(x, PAD_T + drawH); ctx.stroke();
    ctx.fillText(`${n}`, x, PAD_T + drawH + 3);
  }
  // Grid lines horizontal (depth ticks)
  ctx.fillStyle = '#7a8a9a'; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  const depthTick = maxDepth <= 10 ? 1 : maxDepth <= 30 ? 5 : maxDepth <= 80 ? 10 : 20;
  for (let d = 0; d <= maxDepth + 0.01; d += depthTick) {
    const y = toY(d);
    ctx.strokeStyle = '#dde3ea'; ctx.lineWidth = 0.5;
    ctx.beginPath(); ctx.moveTo(PAD_L, y); ctx.lineTo(PAD_L + UNIT_BAND_W + drawW, y); ctx.stroke();
    ctx.fillStyle = '#7a8a9a';
    ctx.fillText(`${d.toFixed(0)}m`, PAD_L - 4, y);
  }

  // ── Axis labels ──
  ctx.fillStyle = '#334455'; ctx.font = 'bold 9px Inter, sans-serif';
  ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  ctx.fillText('SPT N (blows/300mm)', PAD_L + UNIT_BAND_W + drawW / 2, PAD_T + drawH + 14);
  ctx.save();
  ctx.translate(11, PAD_T + drawH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText('Depth below GL (m)', 0, 0);
  ctx.restore();

  // ── Individual BH traces ──
  if (showBHs) {
    bhTraces.forEach((bh, i) => {
      if (!bh.pts.length) return;
      ctx.strokeStyle = `hsla(${(i * 47) % 360},60%,50%,0.45)`;
      ctx.lineWidth = 0.8;
      ctx.setLineDash([2, 2]);
      ctx.beginPath();
      bh.pts.forEach((p, j) => {
        const x = toX(p.n), y = toY(p.depth);
        if (j === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.setLineDash([]);
      // Dot per observation
      ctx.fillStyle = `hsla(${(i * 47) % 360},55%,45%,0.7)`;
      bh.pts.forEach(p => {
        ctx.beginPath();
        ctx.arc(toX(p.n), toY(p.depth), 2.5, 0, Math.PI * 2);
        ctx.fill();
      });
    });
  }

  // ── IQR band (p25–p75) ──
  const validBins = binStats.filter(b => b.mean != null);
  if (validBins.length >= 2) {
    ctx.fillStyle = 'rgba(66,114,196,0.18)';
    ctx.beginPath();
    validBins.forEach((b, i) => {
      const x = toX(b.p25), y = toY(b.depth);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    [...validBins].reverse().forEach(b => ctx.lineTo(toX(b.p75), toY(b.depth)));
    ctx.closePath(); ctx.fill();

    // Min–Max envelope (outer, faint)
    ctx.fillStyle = 'rgba(66,114,196,0.07)';
    ctx.beginPath();
    validBins.forEach((b, i) => {
      const x = toX(b.min), y = toY(b.depth);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    [...validBins].reverse().forEach(b => ctx.lineTo(toX(b.max), toY(b.depth)));
    ctx.closePath(); ctx.fill();

    // Mean line
    ctx.strokeStyle = '#2255aa';
    ctx.lineWidth = 2;
    ctx.beginPath();
    validBins.forEach((b, i) => {
      const x = toX(b.mean), y = toY(b.depth);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  // ── Legend ──
  const legX = PAD_L + UNIT_BAND_W + drawW - 110;
  const legY = PAD_T + 6;
  ctx.fillStyle = 'rgba(248,250,251,0.88)';
  ctx.fillRect(legX - 4, legY - 4, 115, showBHs ? 68 : 48);
  ctx.strokeStyle = '#cdd5dd'; ctx.lineWidth = 0.5;
  ctx.strokeRect(legX - 4, legY - 4, 115, showBHs ? 68 : 48);
  ctx.font = '8.5px Inter, sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';

  // Mean
  ctx.strokeStyle = '#2255aa'; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(legX, legY + 6); ctx.lineTo(legX + 22, legY + 6); ctx.stroke();
  ctx.fillStyle = '#334455'; ctx.fillText('Mean', legX + 26, legY + 6);

  // IQR band
  ctx.fillStyle = 'rgba(66,114,196,0.25)';
  ctx.fillRect(legX, legY + 18, 22, 10);
  ctx.fillStyle = '#334455'; ctx.fillText('IQR (25–75%)', legX + 26, legY + 23);

  // Min-Max
  ctx.fillStyle = 'rgba(66,114,196,0.10)';
  ctx.fillRect(legX, legY + 34, 22, 10);
  ctx.fillStyle = '#334455'; ctx.fillText('Min–Max range', legX + 26, legY + 39);

  if (showBHs) {
    ctx.strokeStyle = 'rgba(120,120,180,0.6)'; ctx.lineWidth = 1;
    ctx.setLineDash([2, 2]);
    ctx.beginPath(); ctx.moveTo(legX, legY + 54); ctx.lineTo(legX + 22, legY + 54); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#334455'; ctx.fillText('Individual BHs', legX + 26, legY + 54);
  }

  // ── Title ──
  ctx.fillStyle = '#223344'; ctx.font = 'bold 10px Inter, sans-serif';
  ctx.textAlign = 'left'; ctx.textBaseline = 'top';
  ctx.fillText(`SPT N Profile  (${bhs.length} BHs, bin=${binSize}m)`, PAD_L + UNIT_BAND_W, 8);

  // ── Chart border ──
  ctx.strokeStyle = 'rgba(40,60,80,0.15)'; ctx.lineWidth = 0.7;
  ctx.strokeRect(PAD_L + UNIT_BAND_W, PAD_T, drawW, drawH);
}

function initSPTProfile() {
  ['spt-show-bhs', 'spt-show-units'].forEach(id => {
    document.getElementById(id)?.addEventListener('change', () => {
      if (!document.getElementById('log-sub-spt')?.hidden) drawSPTProfile();
    });
  });
  document.getElementById('spt-bin-size')?.addEventListener('change', () => {
    if (!document.getElementById('log-sub-spt')?.hidden) drawSPTProfile();
  });
  document.getElementById('btn-spt-export')?.addEventListener('click', () => {
    const canvas = document.getElementById('spt-profile-canvas');
    if (!canvas) return;
    const a = document.createElement('a');
    a.href = canvas.toDataURL('image/png');
    a.download = 'spt-profile.png';
    a.click();
  });
}

// ── Borehole Log Strip View ────────────────────────────────────────────────────
function initBHLogView() {
  AppState.bhLogView = new BHLogView();

  // Redraw when user switches to the Logs tab and data is available
  document.querySelectorAll('.tab-btn[data-tab="logs"]').forEach(btn => {
    btn.addEventListener('click', () => {
      const bhs = AppState.classifiedBH.filter(b => !b.synthetic);
      if (bhs.length && AppState.geoUnits.length) {
        AppState.bhLogView.draw(bhs, AppState.geoUnits, AppState.voxelGrid);
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

// ── Named View Bookmarks ───────────────────────────────────────────────────────
function initViewBookmarks() {
  const BM_KEY = 'geo_view_bookmarks';
  AppState.cameraBookmarks = (() => {
    try { return JSON.parse(localStorage.getItem(BM_KEY) || '[]'); } catch { return []; }
  })();

  const toggleBtn  = document.getElementById('btn-save-bookmark');
  const panel      = document.getElementById('bookmark-panel');
  const nameInput  = document.getElementById('bookmark-name-input');
  const confirmBtn = document.getElementById('btn-bookmark-confirm');
  const listEl     = document.getElementById('bookmark-list');

  function persist() {
    try { localStorage.setItem(BM_KEY, JSON.stringify(AppState.cameraBookmarks)); } catch {}
  }

  function renderList() {
    if (!listEl) return;
    if (!AppState.cameraBookmarks.length) {
      listEl.innerHTML = '<div class="bm-empty">No saved views yet</div>';
      return;
    }
    listEl.innerHTML = AppState.cameraBookmarks.map(bm => `
      <div class="bm-entry">
        <span class="bm-icon">📷</span>
        <button class="bm-load" onclick="window._bmLoad(${bm.id})" title="Restore view">${escHtml(bm.name)}</button>
        <button class="bm-del"  onclick="window._bmDel(${bm.id})"  title="Delete">✕</button>
      </div>`).join('');
  }

  toggleBtn?.addEventListener('click', (e) => {
    e.stopPropagation();
    panel?.classList.toggle('hidden');
  });

  document.addEventListener('click', (e) => {
    if (!panel?.classList.contains('hidden') && !panel?.contains(e.target) && e.target !== toggleBtn) {
      panel?.classList.add('hidden');
    }
  });

  confirmBtn?.addEventListener('click', () => {
    if (!AppState.scene) { log('Load a model first.', 'warn'); return; }
    const rawName = nameInput?.value.trim();
    const name = rawName || `View ${AppState.cameraBookmarks.length + 1}`;
    const state = AppState.scene.getCameraState();
    AppState.cameraBookmarks.push({ id: Date.now(), name, state });
    persist();
    if (nameInput) nameInput.value = '';
    renderList();
    log(`View saved: "${name}"`, 'ok');
  });

  nameInput?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') confirmBtn?.click();
  });

  window._bmLoad = (id) => {
    const bm = AppState.cameraBookmarks.find(b => b.id === id);
    if (bm && AppState.scene) {
      AppState.scene.setCameraState(bm.state);
      panel?.classList.add('hidden');
      log(`Restored view: "${bm.name}"`, 'ok');
    }
  };

  window._bmDel = (id) => {
    const bm = AppState.cameraBookmarks.find(b => b.id === id);
    AppState.cameraBookmarks = AppState.cameraBookmarks.filter(b => b.id !== id);
    persist();
    renderList();
    if (bm) log(`Deleted view: "${bm.name}"`, 'info');
  };

  renderList();
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
    setEnabled('btn-export-bh-csv', true); setEnabled('btn-export-las', true);
      setEnabled('btn-export-ags', true); setEnabled('btn-export-geojson-tops', true);
    setEnabled('btn-export-props', true); setEnabled('btn-formation-stats', true); setEnabled('btn-crossplot-draw', true); setEnabled('btn-depthplot-draw', true);
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
    const mode = document.getElementById('plan-view-mode')?.value ?? 'unit';
    if (!AppState.voxelGrid && mode !== 'concept_territory') {
      log('Build the 3D model first.', 'warn'); return;
    }
    AppState.planView.draw(AppState.voxelGrid, AppState.geoUnits, AppState.classifiedBH, AppState.conceptStore);
  });

  // Show/hide unit probability selector when mode changes
  document.getElementById('plan-view-mode')?.addEventListener('change', e => {
    const probWrap  = document.getElementById('plan-view-prob-unit-wrap');
    const depthWrap = document.getElementById('plan-view-depth-unit-wrap');
    if (probWrap)  probWrap.style.display  = e.target.value === 'probability' ? 'flex' : 'none';
    if (depthWrap) depthWrap.style.display = e.target.value === 'depth'       ? 'flex' : 'none';
    const canDraw = AppState.voxelGrid || (e.target.value === 'concept_territory' && AppState.conceptStore && !AppState.conceptStore.isEmpty);
    if (canDraw && AppState.planView?.visible) {
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

// ── Stratigraphic correlation panel ──────────────────────────────────────────
function initStratCorrelation() {
  AppState.stratCorr = new StratCorrelation();

  document.getElementById('btn-strat-corr')?.addEventListener('click', () => {
    const bhs = AppState.classifiedBH.filter(b => !b.synthetic && b.layers?.length);
    if (!bhs.length) { log('No classified boreholes available.', 'warn'); return; }
    AppState.stratCorr.draw(bhs, AppState.geoUnits);
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

  document.getElementById('btn-formation-stats')?.addEventListener('click', () => {
    const panel = document.getElementById('formation-stats-panel');
    if (!panel) return;
    const wasHidden = panel.style.display === 'none';
    panel.style.display = wasHidden ? 'block' : 'none';
    if (wasHidden) drawFormationStats();
  });

  document.getElementById('fstat-metric')?.addEventListener('change', () => {
    if (document.getElementById('formation-stats-panel')?.style.display !== 'none') {
      drawFormationStats();
    }
  });

  document.getElementById('btn-fstat-export')?.addEventListener('click', () => {
    exportFormationStatsCSV();
  });
}

function drawFormationStats() {
  const canvas  = document.getElementById('fstat-canvas');
  if (!canvas || !AppState.geoUnits.length) return;

  const bhs     = AppState.classifiedBH.filter(b => !b.synthetic && b.layers?.length);
  const metric  = document.getElementById('fstat-metric')?.value ?? 'thickness';
  const units   = AppState.geoUnits;
  const grid    = AppState.voxelGrid;

  // Build per-unit data from boreholes
  const unitByCode = {};
  units.forEach(u => { unitByCode[u.code] = u; });

  const unitData = {}; // code → { thickArr, depthArr, volume }
  units.forEach(u => { unitData[u.code] = { thickArr: [], depthArr: [], volume: 0 }; });

  bhs.forEach(b => {
    b.layers.forEach(l => {
      const d = unitData[l.unitCode];
      if (!d) return;
      const thick = (l.base ?? 0) - (l.top ?? 0);
      if (thick > 0) d.thickArr.push(thick);
      d.depthArr.push(l.top ?? 0);
    });
  });

  // Volume from voxel grid
  if (grid) {
    const { nx, ny, nz, cellSize: cs, unitIds } = grid;
    const voxVol = cs * cs * (grid.cellH ?? cs);
    unitIds.forEach(uid => {
      if (uid > 0 && uid <= units.length) {
        unitData[units[uid - 1].code].volume += voxVol;
      }
    });
  }

  // Quantile helper
  const quantile = (arr, q) => {
    if (!arr.length) return null;
    const s = [...arr].sort((a, b) => a - b);
    const pos = (s.length - 1) * q;
    const lo  = Math.floor(pos);
    const hi  = Math.ceil(pos);
    return s[lo] + (s[hi] - s[lo]) * (pos - lo);
  };
  const stats = (arr) => ({
    n: arr.length,
    min: quantile(arr, 0),
    p25: quantile(arr, 0.25),
    med: quantile(arr, 0.5),
    p75: quantile(arr, 0.75),
    max: quantile(arr, 1),
    mean: arr.length ? arr.reduce((s, v) => s + v, 0) / arr.length : null,
  });

  // Canvas layout
  const ROW_H  = 28;
  const PAD_L  = 52;
  const PAD_R  = 80;
  const PAD_T  = 28;
  const PAD_B  = 20;
  const W      = 320;
  const nUnits = units.length;
  const H      = PAD_T + nUnits * ROW_H + PAD_B;

  canvas.width  = W;
  canvas.height = H;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#f8fafb';
  ctx.fillRect(0, 0, W, H);

  // Determine axis range
  const allVals = units.flatMap(u => {
    const d = unitData[u.code];
    if (metric === 'thickness') return d.thickArr;
    if (metric === 'depth') return d.depthArr;
    return [];
  });
  const maxVal = metric === 'volume'
    ? Math.max(...units.map(u => unitData[u.code].volume), 1)
    : (allVals.length ? Math.max(...allVals) * 1.05 : 20);
  const minVal = 0;
  const drawW  = W - PAD_L - PAD_R;

  const toX = v => PAD_L + ((v - minVal) / (maxVal - minVal)) * drawW;

  // Axis ticks
  ctx.strokeStyle = '#dde3ea'; ctx.lineWidth = 0.5;
  ctx.fillStyle = '#7a8a9a'; ctx.font = '8.5px monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  const tickCount = 5;
  for (let i = 0; i <= tickCount; i++) {
    const v = minVal + (maxVal - minVal) * (i / tickCount);
    const x = toX(v);
    ctx.beginPath(); ctx.moveTo(x, PAD_T); ctx.lineTo(x, PAD_T + nUnits * ROW_H); ctx.stroke();
    const lbl = metric === 'volume' ? (v >= 1000 ? `${(v/1000).toFixed(0)}k` : v.toFixed(0)) : v.toFixed(1);
    ctx.fillText(lbl, x, PAD_T + nUnits * ROW_H + 4);
  }

  // Title
  const titles = { thickness: 'Thickness (m)', depth: 'Depth to Top (m)', volume: 'Model Volume (m³)' };
  ctx.fillStyle = '#334455'; ctx.font = 'bold 9px Inter, sans-serif';
  ctx.textAlign = 'center'; ctx.fillText(titles[metric], PAD_L + drawW / 2, 8);

  // Per-unit rows
  units.forEach((u, i) => {
    const d  = unitData[u.code];
    const cy = PAD_T + i * ROW_H + ROW_H / 2;

    // Unit color swatch + label
    ctx.fillStyle = u.color ?? '#888';
    ctx.fillRect(2, cy - 6, 10, 12);
    ctx.fillStyle = '#334455'; ctx.font = 'bold 8.5px Inter, sans-serif';
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    ctx.fillText(u.code.slice(0, 6), PAD_L - 4, cy);

    if (metric === 'volume') {
      // Bar chart
      const vol = d.volume;
      if (vol > 0) {
        ctx.fillStyle = (u.color ?? '#4472c4') + 'bb';
        ctx.fillRect(toX(0), cy - 7, toX(vol) - toX(0), 14);
        ctx.fillStyle = '#334455'; ctx.font = '8px monospace'; ctx.textAlign = 'left';
        ctx.fillText(
          vol >= 1000 ? `${(vol / 1000).toFixed(1)}k m³` : `${vol.toFixed(0)} m³`,
          toX(vol) + 3, cy
        );
      } else {
        ctx.fillStyle = '#aaa'; ctx.font = '8px monospace'; ctx.textAlign = 'left';
        ctx.fillText('— no model', toX(0) + 3, cy);
      }
    } else {
      const arr  = metric === 'thickness' ? d.thickArr : d.depthArr;
      if (!arr.length) {
        ctx.fillStyle = '#aaa'; ctx.font = '8px Inter, sans-serif'; ctx.textAlign = 'left';
        ctx.fillText('no data', toX(0) + 3, cy); return;
      }
      const st = stats(arr);

      // Min–Max whisker line
      ctx.strokeStyle = (u.color ?? '#4472c4'); ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.moveTo(toX(st.min), cy); ctx.lineTo(toX(st.max), cy); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(toX(st.min), cy - 4); ctx.lineTo(toX(st.min), cy + 4); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(toX(st.max), cy - 4); ctx.lineTo(toX(st.max), cy + 4); ctx.stroke();

      // IQR box
      ctx.fillStyle = (u.color ?? '#4472c4') + '55';
      ctx.fillRect(toX(st.p25), cy - 6, toX(st.p75) - toX(st.p25), 12);
      ctx.strokeRect(toX(st.p25), cy - 6, toX(st.p75) - toX(st.p25), 12);

      // Median tick
      ctx.strokeStyle = u.color ?? '#4472c4'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(toX(st.med), cy - 6); ctx.lineTo(toX(st.med), cy + 6); ctx.stroke();

      // Right label: n, mean
      ctx.fillStyle = '#556677'; ctx.font = '8px monospace'; ctx.textAlign = 'left';
      ctx.fillText(`n=${st.n} μ=${st.mean?.toFixed(1)}`, W - PAD_R + 4, cy);
    }
  });

  // Legend for box-plot key (only for non-volume metrics)
  if (metric !== 'volume') {
    const lx = PAD_L, ly = PAD_T - 18;
    ctx.fillStyle = 'rgba(80,100,130,0.15)';
    ctx.fillRect(lx, ly + 2, 18, 8);
    ctx.strokeStyle = '#556677'; ctx.lineWidth = 0.8;
    ctx.strokeRect(lx, ly + 2, 18, 8);
    ctx.beginPath(); ctx.moveTo(lx + 9, ly + 2); ctx.lineTo(lx + 9, ly + 10); ctx.stroke();
    ctx.fillStyle = '#556677'; ctx.font = '7.5px Inter, sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    ctx.fillText('IQR box | median | whiskers = min/max', lx + 22, ly + 3);
  }
}

// ── Certainty Histogram ───────────────────────────────────────────────────────
function _initCertaintyHistUnit() {
  const sel = document.getElementById('cert-hist-unit');
  if (!sel) return;
  sel.innerHTML = '<option value="__all__">All units</option>'
    + AppState.geoUnits.map(u => `<option value="${u.id}">${u.code} — ${u.name}</option>`).join('');
  sel.onchange = () => drawCertaintyHistogram();

  const wrap = document.getElementById('certainty-hist-wrap');
  if (wrap) wrap.style.display = AppState.voxelGrid ? 'block' : 'none';
}

function drawCertaintyHistogram() {
  const canvas = document.getElementById('certainty-hist-canvas');
  const wrap   = document.getElementById('certainty-hist-wrap');
  const grid   = AppState.voxelGrid;
  if (!canvas || !grid) { if (wrap) wrap.style.display = 'none'; return; }
  if (wrap) wrap.style.display = 'block';

  const { unitIds, certainty, nx, ny, nz } = grid;
  const selVal = document.getElementById('cert-hist-unit')?.value ?? '__all__';
  const filterUid = selVal === '__all__' ? null : parseInt(selVal, 10);

  // Collect certainty values
  const vals = [];
  for (let i = 0; i < unitIds.length; i++) {
    if (!unitIds[i]) continue;
    if (filterUid !== null && unitIds[i] !== filterUid) continue;
    vals.push(certainty[i]);
  }
  if (!vals.length) return;

  // Build histogram: 20 bins 0→1
  const nBins = 20;
  const counts = new Int32Array(nBins);
  vals.forEach(v => {
    const b = Math.min(nBins - 1, Math.floor(v * nBins));
    counts[b]++;
  });
  const maxCount = Math.max(...counts, 1);
  const meanCert = vals.reduce((s, v) => s + v, 0) / vals.length;

  const W = 300, H = 140;
  const PAD_L = 38, PAD_R = 12, PAD_T = 20, PAD_B = 22;
  canvas.width  = W; canvas.height = H;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#f8fafb'; ctx.fillRect(0, 0, W, H);

  const drawW = W - PAD_L - PAD_R;
  const drawH = H - PAD_T - PAD_B;
  const barW  = drawW / nBins;
  const toY   = c => PAD_T + drawH - (c / maxCount) * drawH;

  // Grid lines
  ctx.strokeStyle = '#dde3ea'; ctx.lineWidth = 0.5;
  [0.25, 0.5, 0.75, 1.0].forEach(f => {
    const y = PAD_T + drawH - f * drawH;
    ctx.beginPath(); ctx.moveTo(PAD_L, y); ctx.lineTo(PAD_L + drawW, y); ctx.stroke();
    ctx.fillStyle = '#7a8a9a'; ctx.font = '8px monospace'; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    ctx.fillText(`${(f * maxCount).toFixed(0)}`, PAD_L - 2, y);
  });

  // Bars — coloured by certainty level
  counts.forEach((c, i) => {
    const certMid = (i + 0.5) / nBins;
    const t  = certMid;
    // Green at high certainty, red at low
    const r  = Math.round(255 * (1 - t));
    const g  = Math.round(200 * t);
    const bl = 60;
    const x  = PAD_L + i * barW;
    const y  = toY(c);
    const h  = PAD_T + drawH - y;
    ctx.fillStyle = `rgba(${r},${g},${bl},0.75)`;
    ctx.fillRect(x + 0.5, y, barW - 1, h);
  });

  // X axis labels
  ctx.fillStyle = '#7a8a9a'; ctx.font = '8px monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  [0, 0.25, 0.5, 0.75, 1.0].forEach(v => {
    const x = PAD_L + v * drawW;
    ctx.fillText(v.toFixed(2), x, PAD_T + drawH + 4);
  });

  // Mean line
  const meanX = PAD_L + meanCert * drawW;
  ctx.strokeStyle = '#2255aa'; ctx.lineWidth = 1.5; ctx.setLineDash([3, 2]);
  ctx.beginPath(); ctx.moveTo(meanX, PAD_T); ctx.lineTo(meanX, PAD_T + drawH); ctx.stroke();
  ctx.setLineDash([]);

  // Axis labels
  ctx.fillStyle = '#334455'; ctx.font = 'bold 8px Inter, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('Classification Certainty', PAD_L + drawW / 2, 6);
  ctx.fillStyle = '#2255aa'; ctx.font = '7.5px monospace';
  ctx.fillText(`μ=${(meanCert*100).toFixed(1)}%`, meanX, PAD_T + drawH + 13);

  // Count label
  ctx.fillStyle = '#7a8a9a'; ctx.font = '7.5px monospace'; ctx.textAlign = 'left';
  ctx.fillText(`n=${vals.length.toLocaleString()} voxels`, PAD_L + 2, 7);
}

// Show certainty histogram when switching to Analysis tab
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.tab-btn[data-tab="analysis"]').forEach(btn => {
    btn.addEventListener('click', () => {
      if (AppState.voxelGrid) { _initCertaintyHistUnit(); drawCertaintyHistogram(); }
    });
  });
});

function exportFormationStatsCSV() {
  const bhs   = AppState.classifiedBH.filter(b => !b.synthetic && b.layers?.length);
  const units  = AppState.geoUnits;
  const grid   = AppState.voxelGrid;
  if (!units.length) return;

  const unitData = {};
  units.forEach(u => { unitData[u.code] = { thickArr: [], depthArr: [], volume: 0 }; });
  bhs.forEach(b => {
    b.layers.forEach(l => {
      const d = unitData[l.unitCode];
      if (!d) return;
      const thick = (l.base ?? 0) - (l.top ?? 0);
      if (thick > 0) d.thickArr.push(thick);
      d.depthArr.push(l.top ?? 0);
    });
  });
  if (grid) {
    const { unitIds, cellSize: cs } = grid;
    const voxVol = cs * cs * (grid.cellH ?? cs);
    unitIds.forEach(uid => {
      if (uid > 0 && uid <= units.length) unitData[units[uid - 1].code].volume += voxVol;
    });
  }

  const q = (arr, p) => {
    if (!arr.length) return '';
    const s = [...arr].sort((a, b) => a - b);
    const i = (s.length - 1) * p;
    return (s[Math.floor(i)] + (s[Math.ceil(i)] - s[Math.floor(i)]) * (i - Math.floor(i))).toFixed(2);
  };

  let csv = 'Code,Name,N_thick,Thick_min,Thick_p25,Thick_med,Thick_p75,Thick_max,Thick_mean,'
           + 'N_depth,Depth_min,Depth_p25,Depth_med,Depth_p75,Depth_max,Depth_mean,Volume_m3\n';
  units.forEach(u => {
    const d  = unitData[u.code];
    const ta = d.thickArr, da = d.depthArr;
    const mn = a => a.length ? (a.reduce((s,v)=>s+v,0)/a.length).toFixed(2) : '';
    csv += `${u.code},${u.name},${ta.length},${q(ta,0)},${q(ta,.25)},${q(ta,.5)},${q(ta,.75)},${q(ta,1)},${mn(ta)},`
         + `${da.length},${q(da,0)},${q(da,.25)},${q(da,.5)},${q(da,.75)},${q(da,1)},${mn(da)},${d.volume.toFixed(0)}\n`;
  });

  const blob = new Blob([csv], { type: 'text/csv' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'formation-statistics.csv';
  a.click();
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
        siteCtx, apiKey, !apiKey, AppState.conceptStore,
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

  // ── Leave-one-out cross-validation ──────────────────────────────────────────
  document.getElementById('btn-loocv')?.addEventListener('click', async () => {
    const realBHs = AppState.classifiedBH.filter(b => !b.synthetic && b.layers?.length >= 1);
    if (realBHs.length < 3) { log('Need ≥ 3 real boreholes for LOO-CV.', 'warn'); return; }

    const btn = document.getElementById('btn-loocv');
    const out  = document.getElementById('loocv-results');
    if (btn) btn.disabled = true;
    if (out) { out.style.display = 'block'; out.innerHTML = '<p class="hint" style="font-size:10px">Running LOO cross-validation…</p>'; }

    const opts = {
      kNeighbors: AppState.kNeighbors, idwPower: AppState.idwPower,
      method: 'idw',  // IDW only — fast for LOO
      cellSizeZ: AppState.cellSizeZ,
      anisoAzimuth: AppState.anisoAzimuth, anisoRatio: AppState.anisoRatio,
    };
    const unitById = {};
    AppState.geoUnits.forEach(u => { unitById[u.id] = u; });

    const perBH = [];
    let totalCorrect = 0, totalLayers = 0;

    for (let i = 0; i < realBHs.length; i++) {
      const testBH = realBHs[i];
      // Training set: all boreholes except testBH (include synthetic)
      const trainBHs = AppState.classifiedBH.filter(b => b !== testBH);
      if (trainBHs.filter(b => !b.synthetic).length === 0) continue;

      await new Promise(r => setTimeout(r, 0)); // yield to UI
      try {
        const g = await buildVoxelGrid(trainBHs, AppState.geoUnits, AppState.cellSizeH, opts);
        const { nx, ny, nz, origin: O, cellSize: cs, cellHeight: ch, unitIds, certainty } = g;
        const ix = Math.max(0, Math.min(nx - 1, Math.round((testBH.x - O.x) / cs - 0.5)));
        const iy = Math.max(0, Math.min(ny - 1, Math.round((testBH.y - O.z) / cs - 0.5)));
        let bhCorrect = 0, bhTotal = 0;
        for (const layer of testBH.layers) {
          if (!layer.unitCode) continue;
          const elev = (testBH.groundLevel ?? 0) - (layer.top + layer.base) / 2;
          const iz   = Math.max(0, Math.min(nz - 1, Math.round((elev - O.y) / ch - 0.5)));
          const flat = ix + iy * nx + iz * nx * ny;
          const pred = unitById[unitIds[flat]];
          bhTotal++;
          if (pred?.code === layer.unitCode) bhCorrect++;
        }
        const acc = bhTotal > 0 ? (bhCorrect / bhTotal * 100).toFixed(0) : '—';
        perBH.push({ id: testBH.id, correct: bhCorrect, total: bhTotal, acc: +acc });
        totalCorrect += bhCorrect;
        totalLayers  += bhTotal;
        if (out) out.innerHTML = `<p class="hint" style="font-size:10px">Running… ${i + 1}/${realBHs.length} done</p>`;
      } catch (err) {
        perBH.push({ id: testBH.id, correct: 0, total: 0, acc: NaN, err: err.message });
      }
    }

    const loocvAcc = totalLayers > 0 ? (totalCorrect / totalLayers * 100).toFixed(1) : '0';
    const colOk  = +loocvAcc >= 70 ? 'var(--green)' : +loocvAcc >= 50 ? '#c8a855' : '#d04040';

    const rows = perBH.map(r => {
      const bar = isNaN(r.acc) ? '—' : `${'▓'.repeat(Math.round(r.acc/10))}${'░'.repeat(10-Math.round(r.acc/10))}`;
      const c   = r.acc >= 80 ? 'var(--green)' : r.acc >= 50 ? '#c8a855' : '#d04040';
      return `<tr style="border-top:1px solid var(--border)">
        <td style="padding:2px 4px;font-weight:600;color:var(--accent)">${r.id}</td>
        <td style="padding:2px 4px;font-family:var(--font-mono);font-size:9px">${bar}</td>
        <td style="padding:2px 4px;font-weight:600;color:${c}">${isNaN(r.acc) ? '—' : r.acc+'%'}</td>
        <td style="padding:2px 4px;color:var(--text-muted)">${r.correct}/${r.total}</td>
      </tr>`;
    }).join('');

    if (out) out.innerHTML = `
      <div style="font-size:11px;font-weight:700;color:${colOk};margin-bottom:4px">
        LOO accuracy: ${loocvAcc}% <span style="font-weight:400;color:var(--text-mid);font-size:10px">(${totalLayers} samples)</span>
      </div>
      <table style="width:100%;font-size:10px;border-collapse:collapse">
        <thead><tr style="color:var(--text-muted)">
          <th style="text-align:left;padding:2px 4px">BH</th>
          <th></th>
          <th style="padding:2px 4px">Acc</th>
          <th style="padding:2px 4px">n</th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>
      <p class="hint" style="font-size:9px;margin-top:4px">Each BH removed from training set in turn; IDW prediction compared to actual layers. High values = model generalises well.</p>`;

    log(`LOO cross-validation: ${loocvAcc}% overall (${realBHs.length} BHs, ${totalLayers} layer samples).`,
      +loocvAcc >= 70 ? 'ok' : 'warn');
    if (btn) btn.disabled = false;
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

// ── Stratigraphic column — drag-to-reorder ────────────────────────────────────
function updateStratColumn() {
  const list = document.getElementById('strat-order-list');
  const hint = document.getElementById('strat-hint');
  if (!list) return;
  const geoUnits = AppState.geoUnits;
  if (!geoUnits.length) {
    list.innerHTML = '';
    if (hint) hint.hidden = false;
    return;
  }
  if (hint) hint.hidden = true;

  // Compute mean thickness per unit from grid (if available)
  const thickByCode = {};
  const grid = AppState.voxelGrid;
  if (grid) {
    const { nx, ny, nz, cellHeight: ch, unitIds } = grid;
    const counts = {};
    geoUnits.forEach(u => { counts[u.id] = 0; });
    for (let iz = 0; iz < nz; iz++)
      for (let iy = 0; iy < ny; iy++)
        for (let ix = 0; ix < nx; ix++) {
          const uid = unitIds[ix + iy * nx + iz * nx * ny];
          if (uid && counts[uid] !== undefined) counts[uid]++;
        }
    geoUnits.forEach(u => {
      thickByCode[u.code] = counts[u.id] > 0 ? (counts[u.id] / (nx * ny)) * ch : 0;
    });
  }

  // Build ordered list: if user has a locked manual order use it; otherwise use
  // the auto-inferred stratOrder (if any), or the raw geoUnits order.
  const locked = document.getElementById('strat-manual-lock')?.checked;
  if (!locked) {
    // Rebuild from inferred / default order
    const order = AppState.stratOrder?.length
      ? AppState.stratOrder
      : geoUnits.map(u => u.code);
    // Units not mentioned in stratOrder come at the end
    const mentioned = new Set(order);
    const extra = geoUnits.filter(u => !mentioned.has(u.code)).map(u => u.code);
    AppState._stratDisplayOrder = [...order, ...extra];
  }
  const displayOrder = AppState._stratDisplayOrder ?? geoUnits.map(u => u.code);
  const unitByCode = {};
  geoUnits.forEach(u => { unitByCode[u.code] = u; });

  list.innerHTML = '';
  displayOrder.forEach((code, rank) => {
    const unit = unitByCode[code];
    if (!unit) return;
    const thick = thickByCode[code];
    const thickStr = thick != null && thick > 0 ? `${thick.toFixed(1)}m` : '';
    const li = document.createElement('li');
    li.className = 'strat-order-item';
    li.draggable = true;
    li.dataset.code = code;
    li.innerHTML = `
      <span class="strat-drag-handle" title="Drag to reorder">⠿</span>
      <span class="strat-swatch" style="background:${unit.color}"></span>
      <span class="strat-code">${unit.code}</span>
      <span class="strat-name">${unit.name.slice(0, 20)}</span>
      ${thickStr ? `<span class="strat-thick">${thickStr}</span>` : ''}
    `;
    list.appendChild(li);
  });

  // ── Drag-and-drop reorder logic ────────────────────────────────────────────
  let dragSrc = null;
  list.querySelectorAll('.strat-order-item').forEach(item => {
    item.addEventListener('dragstart', e => {
      dragSrc = item;
      e.dataTransfer.effectAllowed = 'move';
      item.classList.add('dragging');
    });
    item.addEventListener('dragend', () => {
      item.classList.remove('dragging');
      list.querySelectorAll('.strat-order-item').forEach(i => i.classList.remove('drag-over'));
    });
    item.addEventListener('dragover', e => {
      e.preventDefault();
      if (item !== dragSrc) {
        list.querySelectorAll('.strat-order-item').forEach(i => i.classList.remove('drag-over'));
        item.classList.add('drag-over');
      }
    });
    item.addEventListener('drop', e => {
      e.preventDefault();
      if (dragSrc && dragSrc !== item) {
        const allItems = [...list.querySelectorAll('.strat-order-item')];
        const srcIdx  = allItems.indexOf(dragSrc);
        const dstIdx  = allItems.indexOf(item);
        if (srcIdx < dstIdx) {
          item.after(dragSrc);
        } else {
          item.before(dragSrc);
        }
        // Update AppState
        const newOrder = [...list.querySelectorAll('.strat-order-item')]
          .map(el => el.dataset.code);
        AppState._stratDisplayOrder = newOrder;
        // Lock automatically on manual reorder so rebuild doesn't override
        const lockEl = document.getElementById('strat-manual-lock');
        if (lockEl) lockEl.checked = true;
        AppState.stratOrder = newOrder;
        log(`Stratigraphic order updated manually: ${newOrder.join(' → ')}`, 'info');
      }
      list.querySelectorAll('.strat-order-item').forEach(i => i.classList.remove('drag-over'));
    });
  });
}

function _initStratLockToggle() {
  document.getElementById('strat-manual-lock')?.addEventListener('change', e => {
    if (!e.target.checked) {
      // Re-infer order from data on next updateStratColumn call
      AppState._stratDisplayOrder = null;
      updateStratColumn();
    }
  });
}

function initStratOrderButtons() {
  const inferBtn  = document.getElementById('btn-infer-strat-order');
  const applyBtn  = document.getElementById('btn-apply-strat-order');
  const confDiv   = document.getElementById('strat-confidence');

  inferBtn?.addEventListener('click', () => {
    const bhs = AppState.classifiedBH.filter(b => !b.synthetic);
    if (!bhs.length || !AppState.geoUnits.length) {
      log('Run AI analysis first.', 'warn'); return;
    }
    const result = inferStratigraphicOrder(bhs, AppState.geoUnits);
    if (!result.length) return;

    // Update strat order from inference
    const orderedCodes = result.map(r => r.code);
    AppState.stratOrder        = orderedCodes;
    AppState._stratDisplayOrder = orderedCodes;
    updateStratColumn();

    // Show confidence
    if (confDiv) {
      confDiv.style.display = 'block';
      confDiv.innerHTML = result.map(r => {
        const pct  = Math.round(r.confidence * 100);
        const col  = pct >= 80 ? 'var(--green)' : pct >= 50 ? '#c8a855' : '#d04040';
        const bar  = '▓'.repeat(Math.round(pct / 10)) + '░'.repeat(10 - Math.round(pct / 10));
        return `<div style="display:flex;align-items:center;gap:4px;margin-bottom:1px">
          <span style="font-family:var(--font-mono);width:46px;color:var(--accent)">${r.code}</span>
          <span style="font-family:var(--font-mono);font-size:8px;letter-spacing:-0.5px;color:${col}">${bar}</span>
          <span style="color:${col};width:28px">${pct}%</span>
        </div>`;
      }).join('');
    }

    // Enable apply button
    if (applyBtn) applyBtn.disabled = !AppState.voxelGrid;
    log(`Stratigraphic order inferred: ${orderedCodes.join(' → ')} (youngest→oldest)`, 'ok');
  });

  applyBtn?.addEventListener('click', () => {
    if (!AppState.voxelGrid) { log('Build the 3D model first.', 'warn'); return; }
    const order = AppState.stratOrder?.length ? AppState.stratOrder
                  : (AppState._stratDisplayOrder ?? AppState.geoUnits.map(u => u.code));
    if (!order.length) { log('Set stratigraphic order first.', 'warn'); return; }

    const modified = applyStratigraphicOrder(AppState.voxelGrid, order, AppState.geoUnits);
    AppState.scene.buildVoxels(AppState.voxelGrid, AppState.geoUnits, AppState.classifiedBH);
    updateVolumeStats?.();
    updateUnitStats?.();
    log(`Stratigraphic order enforced: ${modified.toLocaleString()} voxel(s) corrected.`,
      modified > 0 ? 'ok' : 'info');
  });

  // Enable apply button after a model is built
  window.addEventListener('geomodel:model-built', () => {
    if (applyBtn && AppState.stratOrder?.length) applyBtn.disabled = false;
  });
}

// ── Isopach map ──────────────────────────────────────────────────────────────
function initIsopachMap() {
  AppState.isopachMap = new IsopachMap();

  document.getElementById('btn-isopach')?.addEventListener('click', () => {
    if (!AppState.voxelGrid) { log('Build the 3D model first.', 'warn'); return; }
    AppState.isopachMap.draw(
      AppState.voxelGrid, AppState.geoUnits, AppState.classifiedBH, AppState.conceptStore
    );
  });

  document.getElementById('isopach-export-csv')?.addEventListener('click', () => {
    if (!AppState.isopachMap?._lastArgs) { log('Open the isopach map first.', 'warn'); return; }
    AppState.isopachMap.exportCSV();
    log('Horizon data exported as CSV', 'ok');
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

  // Redraw when overlay toggles change
  ['fence-show-uncertainty', 'fence-show-coverage', 'fence-show-patterns', 'fence-show-ribbons', 'fence-show-concepts'].forEach(id => {
    document.getElementById(id)?.addEventListener('change', () => {
      if (AppState.fenceSection?.visible) AppState.fenceSection._redraw();
    });
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

// ── Formation Tops Matrix ─────────────────────────────────────────────────────
function renderFormationTopsTable() {
  const wrap    = document.getElementById('formation-tops-wrap');
  const expBtn  = document.getElementById('btn-tops-export');
  if (!wrap) return;

  const bhs   = AppState.classifiedBH.filter(b => !b.synthetic && b.layers?.length);
  const units = AppState.geoUnits;
  if (!bhs.length || !units.length) {
    wrap.innerHTML = '<p class="hint" style="padding:8px">Load classified borehole data.</p>';
    return;
  }

  const useElev = document.getElementById('tops-show-elev')?.checked ?? false;

  // Build lookup: bhId → unitCode → {top, base}
  const matrix = {};
  bhs.forEach(bh => {
    matrix[bh.id] = {};
    bh.layers.forEach(l => {
      if (!matrix[bh.id][l.unitCode]) {
        matrix[bh.id][l.unitCode] = { top: l.top, base: l.base };
      }
    });
  });

  const fmtVal = (bh, code) => {
    const entry = matrix[bh.id]?.[code];
    if (!entry) return '—';
    if (useElev) {
      const elev = (bh.groundLevel ?? 0) - entry.top;
      return elev.toFixed(1);
    }
    return entry.top.toFixed(1);
  };

  // Header row
  let html = `<table class="formation-tops-table"><thead><tr>
    <th style="position:sticky;left:0;background:var(--bg-surface);z-index:1">Unit</th>
    ${bhs.map(b => `<th>${escHtml(b.id.slice(0, 8))}</th>`).join('')}
  </tr></thead><tbody>`;

  units.forEach(u => {
    const hasSome = bhs.some(b => matrix[b.id]?.[u.code]);
    if (!hasSome) return;
    const swatch = `<span style="display:inline-block;width:9px;height:9px;border-radius:2px;background:${u.color};margin-right:4px;vertical-align:middle"></span>`;
    html += `<tr>
      <td style="position:sticky;left:0;background:var(--bg-surface);white-space:nowrap;font-size:10px;font-weight:600">
        ${swatch}${escHtml(u.code)}
      </td>
      ${bhs.map(b => {
        const entry = matrix[b.id]?.[u.code];
        const val   = fmtVal(b, u.code);
        const absent = !entry;
        return `<td style="font-size:9px;font-family:var(--font-mono);color:${absent ? '#aaa' : 'var(--text)'};text-align:right">${val}</td>`;
      }).join('')}
    </tr>`;
  });

  html += '</tbody></table>';
  wrap.innerHTML = html;
  if (expBtn) expBtn.disabled = false;
}

function exportFormationTopsCSV() {
  const bhs   = AppState.classifiedBH.filter(b => !b.synthetic && b.layers?.length);
  const units = AppState.geoUnits;
  if (!bhs.length) return;

  const useElev = document.getElementById('tops-show-elev')?.checked ?? false;
  const matrix = {};
  bhs.forEach(bh => {
    matrix[bh.id] = {};
    bh.layers.forEach(l => {
      if (!matrix[bh.id][l.unitCode]) {
        matrix[bh.id][l.unitCode] = { top: l.top, base: l.base };
      }
    });
  });

  const fmtVal = (bh, code) => {
    const entry = matrix[bh.id]?.[code];
    if (!entry) return '';
    if (useElev) return ((bh.groundLevel ?? 0) - entry.top).toFixed(2);
    return entry.top.toFixed(2);
  };

  const header = ['Unit', ...bhs.map(b => b.id)];
  const label  = useElev ? 'top_elevation_mAOD' : 'depth_to_top_m';
  let csv = `# Formation Top Matrix — ${label}\n${header.join(',')}\n`;
  units.forEach(u => {
    const vals = bhs.map(b => fmtVal(b, u.code));
    if (vals.every(v => !v)) return;
    csv += [u.code, ...vals].join(',') + '\n';
  });

  const blob = new Blob([csv], { type: 'text/csv' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'formation-tops.csv';
  a.click();
}

function initBHDataSubTabs() {
  document.querySelectorAll('.bhdata-sub-tab').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.bhdata-sub-tab').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const key = btn.dataset.bhdTab;
      document.getElementById('bhd-sub-raw').hidden       = (key !== 'raw');
      document.getElementById('bhd-sub-tops').hidden      = (key !== 'tops');
      document.getElementById('bhd-sub-deviation').hidden = (key !== 'deviation');
      if (key === 'tops') renderFormationTopsTable();
      if (key === 'deviation') _refreshDeviationUI();
    });
  });

  document.getElementById('tops-show-elev')?.addEventListener('change', () => {
    if (!document.getElementById('bhd-sub-tops')?.hidden) renderFormationTopsTable();
  });

  document.getElementById('btn-tops-export')?.addEventListener('click', () => {
    exportFormationTopsCSV();
  });

  // Deviation survey panel
  document.getElementById('dev-csv-input')?.addEventListener('change', async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const text = await file.text();
    const bhId = document.getElementById('dev-bh-select')?.value;
    if (!bhId) { log('Select a borehole first.', 'warn'); return; }
    _loadDeviationSurvey(bhId, text, file.name);
    e.target.value = '';
  });

  document.getElementById('btn-dev-clear')?.addEventListener('click', () => {
    const bhId = document.getElementById('dev-bh-select')?.value;
    if (!bhId) return;
    const bh = AppState.classifiedBH.find(b => b.id === bhId);
    if (bh) { delete bh.deviation; delete bh.deviationPath; }
    _refreshDeviationUI();
    log(`Deviation survey removed from ${bhId}.`, 'info');
  });

  document.getElementById('dev-bh-select')?.addEventListener('change', () => _refreshDeviationUI());
}

function _refreshDeviationUI() {
  const sel = document.getElementById('dev-bh-select');
  const wrap = document.getElementById('dev-survey-table-wrap');
  const summary = document.getElementById('dev-summary');
  if (!sel) return;

  // Repopulate BH select
  const bhs = AppState.classifiedBH.filter(b => !b.synthetic && b.layers?.length);
  const curVal = sel.value;
  sel.innerHTML = '<option value="">— Select borehole —</option>'
    + bhs.map(b => {
        const hasDev = b.deviation?.length >= 2;
        return `<option value="${escHtml(b.id)}"${b.id === curVal ? ' selected' : ''}>${escHtml(b.id)}${hasDev ? ' ✓' : ''}</option>`;
      }).join('');

  const bhId = sel.value;
  const bh = bhId ? AppState.classifiedBH.find(b => b.id === bhId) : null;
  if (!bh?.deviation?.length) {
    if (wrap) wrap.innerHTML = '<p class="hint" style="padding:6px;font-size:10px">No deviation survey loaded. Upload a CSV file.</p>';
    if (summary) summary.textContent = '';
    return;
  }

  const survey = bh.deviation;
  let html = `<table class="formation-tops-table"><thead><tr>
    <th>Depth (m)</th><th>Inclination (°)</th><th>Azimuth (°)</th></tr></thead><tbody>`;
  survey.forEach(s => {
    html += `<tr>
      <td style="font-family:var(--font-mono);font-size:9px;text-align:right">${s.depth.toFixed(2)}</td>
      <td style="font-family:var(--font-mono);font-size:9px;text-align:right">${s.incl.toFixed(2)}</td>
      <td style="font-family:var(--font-mono);font-size:9px;text-align:right">${s.azim.toFixed(2)}</td>
    </tr>`;
  });
  html += '</tbody></table>';
  if (wrap) wrap.innerHTML = html;

  // Summary stats
  const maxDepth = survey[survey.length - 1]?.depth ?? 0;
  const maxIncl  = Math.max(...survey.map(s => s.incl));
  if (summary) summary.textContent = `${survey.length} stations · TD ${maxDepth.toFixed(1)} m · Max inclination ${maxIncl.toFixed(1)}°`;
}

function _loadDeviationSurvey(bhId, csvText, filename) {
  // Dynamic import to keep bundle slim
  import('./deviation.js').then(({ parseDeviationCSV, buildDeviationPath }) => {
    const survey = parseDeviationCSV(csvText);
    if (!survey.length) { log(`No valid survey data in ${filename}.`, 'error'); return; }
    const bh = AppState.classifiedBH.find(b => b.id === bhId);
    if (!bh) return;
    bh.deviation = survey;
    bh.deviationPath = buildDeviationPath(bh.x, bh.y, bh.groundLevel ?? 0, survey);
    _refreshDeviationUI();
    log(`Deviation survey loaded for ${bhId}: ${survey.length} stations, max depth ${survey[survey.length - 1]?.depth?.toFixed(1)} m.`, 'ok');
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
  let isoBuilt = false;
  let isoVisible = false;
  let uncertBuilt = false;
  let uncertVisible = false;

  // Reset state when a new model is built
  window.addEventListener('geomodel:model-built', () => {
    isoBuilt = false;
    isoVisible = false;
    uncertBuilt = false;
    uncertVisible = false;
    btn.classList.remove('active');
    const ub = document.getElementById('btn-uncertainty-surface');
    if (ub) ub.classList.remove('active');
  });

  btn.addEventListener('click', async () => {
    if (!AppState.voxelGrid || !AppState.scene) { log('Build the 3D model first.', 'warn'); return; }

    if (isoBuilt) {
      isoVisible = !isoVisible;
      AppState.scene.setIsosurfacesVisible(isoVisible);
      btn.classList.toggle('active', isoVisible);
      log(`Isosurfaces ${isoVisible ? 'shown' : 'hidden'}.`, 'info');
      return;
    }

    btn.disabled = true;
    btn.textContent = '⬡ Building…';
    log(AppState.voxelGrid.probVolumes?.size > 0
      ? 'Building isosurfaces from MC probability volumes…'
      : 'Building marching-cubes isosurfaces…', 'info');
    await new Promise(r => setTimeout(r, 0));

    const op = parseFloat(document.getElementById('surface-opacity')?.value ?? 55) / 100;
    AppState.scene.buildIsosurfaces(
      AppState.voxelGrid, AppState.geoUnits, op,
      p => { btn.textContent = `⬡ ${(p * 100).toFixed(0)}%`; },
    );

    isoBuilt = true;
    isoVisible = true;
    AppState.scene.setIsosurfacesVisible(true);
    btn.disabled = false;
    btn.textContent = '⬡ Isosurfaces';
    btn.classList.add('active');
    const src = AppState.voxelGrid.probVolumes?.size > 0 ? ' (MC probability)' : '';
    log(`Isosurfaces built for ${AppState.geoUnits.length} unit(s)${src}.`, 'ok');
  });

  // ── Uncertainty isosurface ─────────────────────────────────────────────────
  const uncertBtn = document.getElementById('btn-uncertainty-surface');
  uncertBtn?.addEventListener('click', async () => {
    if (!AppState.voxelGrid || !AppState.scene) { log('Build the 3D model first.', 'warn'); return; }

    if (uncertBuilt) {
      uncertVisible = !uncertVisible;
      AppState.scene.setUncertaintySurfaceVisible(uncertVisible);
      uncertBtn.classList.toggle('active', uncertVisible);
      log(`Uncertainty surface ${uncertVisible ? 'shown' : 'hidden'}.`, 'info');
      return;
    }

    uncertBtn.disabled = true;
    uncertBtn.textContent = '◈ Building…';
    log('Building uncertainty isosurface…', 'info');
    await new Promise(r => setTimeout(r, 0));

    const threshold = parseFloat(document.getElementById('uncertainty-threshold')?.value ?? 60) / 100;
    AppState.scene.buildUncertaintySurface(AppState.voxelGrid, threshold, 0.35);

    uncertBuilt = true;
    uncertVisible = true;
    AppState.scene.setUncertaintySurfaceVisible(true);
    uncertBtn.disabled = false;
    uncertBtn.textContent = '◈ Uncertainty';
    uncertBtn.classList.add('active');
    log(`Uncertainty surface built (threshold: ${(threshold * 100).toFixed(0)}% entropy).`, 'ok');
  });

  const uncertThresh = document.getElementById('uncertainty-threshold');
  const uncertThreshLbl = document.getElementById('uncertainty-threshold-val');
  uncertThresh?.addEventListener('input', () => {
    if (uncertThreshLbl) uncertThreshLbl.textContent = uncertThresh.value + '%';
  });
  uncertThresh?.addEventListener('change', () => {
    if (!uncertBuilt) return;
    // rebuild with new threshold
    uncertBuilt = false;
    uncertVisible = false;
    uncertBtn?.classList.remove('active');
    uncertBtn?.click();
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

      // Auto-encode conceptual descriptions from semantic model into ConceptStore
      const conceptDescs = result.conceptual_descriptions ?? [];
      if (conceptDescs.length > 0 && AppState.conceptStore) {
        let encoded = 0;
        for (const cd of conceptDescs) {
          const stmt = cd.statement ?? cd;
          if (!stmt?.trim()) continue;
          try {
            const emb = await encodeGeologicalConcept(stmt, AppState.apiKey, AppState.demoMode, {
              siteContext: { units: AppState.geoUnits.map(u => ({ code: u.code, name: u.name })), description: AppState.siteContext?.description ?? '' },
            });
            AppState.conceptStore.add({
              description:  stmt,
              embedding:    emb,
              confidence:   cd.confidence ?? 0.7,
              domain:       { type: 'global' },
              unitAffinity: Array.isArray(cd.unit_codes) ? cd.unit_codes : [],
            });
            encoded++;
          } catch (e) { log(`Concept auto-encode: ${e.message}`, 'warn'); }
        }
        if (encoded > 0) {
          _renderConceptList();
          _saveConceptStore();
          log(`Auto-encoded ${encoded} semantic model concept(s) → ConceptStore`, 'ok');
        }
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
    boundary:           'Boundary uncertainty (blend ratio 0–1)',
    concept_influence:  'Concept semantic influence (0=data, 1=concept)',
    coverage_density:   'Borehole coverage density (0=sparse, 1=data-dense)',
    dominant_concept:   'Dominant concept (colour by active geological concept)',
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

    if (paramName === 'entropy') {
      const nUnits = AppState.geoUnits.length || 4;
      const ok = AppState.scene.colorByEntropy(nUnits);
      if (!ok) { log('Entropy data not available — build the model first.', 'warn'); return; }
      document.getElementById('param-scale-min').textContent = '0.0 bits';
      document.getElementById('param-scale-mid').textContent = `${(Math.log2(nUnits)/2).toFixed(1)} bits`;
      document.getElementById('param-scale-max').textContent = `${Math.log2(nUnits).toFixed(1)} bits`;
      document.getElementById('param-scale-label').textContent = 'Classification entropy — blue=certain, red=highly uncertain';
      const cs = document.getElementById('param-colorscale');
      cs.querySelector('div').style.background = 'linear-gradient(to right,#2244cc,#00aacc,#22cc66,#ddcc00,#dd2222)';
      cs.style.display = 'block';
      log(`Parameter view: classification entropy (max ${Math.log2(nUnits).toFixed(2)} bits for ${nUnits} units)`, 'ok');
      return;
    }

    if (paramName === 'concept_influence') {
      const ok = AppState.scene.colorByConceptInfluence();
      if (!ok) {
        log('Concept influence data not available — rebuild the model with concepts loaded.', 'warn');
        return;
      }
      document.getElementById('param-scale-min').textContent = '0.0';
      document.getElementById('param-scale-mid').textContent = '0.5';
      document.getElementById('param-scale-max').textContent = '1.0';
      document.getElementById('param-scale-label').textContent = 'Concept semantic influence — purple=data-driven, amber=concept-driven';
      const cs = document.getElementById('param-colorscale');
      cs.querySelector('div').style.background = 'linear-gradient(to right,#6a2fce,#3070cc,#1ab8b8,#40cc40,#e0b020)';
      cs.style.display = 'block';
      log('Parameter view: concept semantic influence', 'ok');
      return;
    }

    if (paramName === 'coverage_density') {
      const ok = AppState.scene.colorByCoverage();
      if (!ok) {
        log('Coverage density not available — rebuild the model using Neural Implicit method.', 'warn');
        return;
      }
      document.getElementById('param-scale-min').textContent = '0.0';
      document.getElementById('param-scale-mid').textContent = '0.5';
      document.getElementById('param-scale-max').textContent = '1.0';
      document.getElementById('param-scale-label').textContent = 'Borehole coverage density — red=sparse/extrapolated, green=data-dense';
      const cs = document.getElementById('param-colorscale');
      cs.querySelector('div').style.background = 'linear-gradient(to right,#cc2222,#e08020,#e0cc00,#40cc40)';
      cs.style.display = 'block';
      log('Parameter view: borehole coverage density', 'ok');
      return;
    }

    if (paramName === 'dominant_concept') {
      const store = AppState.conceptStore;
      if (!store || store.isEmpty) {
        log('No concepts encoded — add concepts in the Concepts tab first.', 'warn');
        return;
      }
      const ok = AppState.scene.colorByDominantConcept(store);
      if (!ok) { log('Dominant concept coloring unavailable.', 'warn'); return; }
      // Build a legend showing concept → hue
      const cs     = document.getElementById('param-colorscale');
      const csDiv  = cs.querySelector('div');
      const stops  = store.concepts.map((c, i) => {
        const hue = i / store.concepts.length;
        return `hsl(${(hue * 360).toFixed(0)},75%,45%)`;
      });
      csDiv.style.background = `linear-gradient(to right,${stops.join(',')})`;
      cs.style.display = 'block';
      document.getElementById('param-scale-min').textContent = store.concepts[0]?.description.slice(0, 12) ?? '';
      document.getElementById('param-scale-max').textContent = store.concepts[store.concepts.length - 1]?.description.slice(0, 12) ?? '';
      document.getElementById('param-scale-mid').textContent = '·';
      document.getElementById('param-scale-label').textContent = `Dominant concept — grey=no concept, ${store.concepts.length} concept(s) shown`;
      log(`Parameter view: dominant concept (${store.concepts.length} concepts)`, 'ok');
      return;
    }

    if (paramName === 'geological_age') {
      const hasPeriods = AppState.geoUnits.some(u => u.period);
      if (!hasPeriods) {
        log('No geological periods assigned — set periods in the Properties tab first.', 'warn');
        return;
      }
      // Build period → colour map from GEO_PERIODS
      const periodColorMap = {};
      GEO_PERIODS.forEach(p => { if (p.color) periodColorMap[p.code] = p.color; });
      AppState.scene.colorByGeologicalAge(AppState.geoUnits, periodColorMap);
      // Build ICS legend gradient from assigned periods
      const used = [...new Set(AppState.geoUnits.filter(u => u.period).map(u => u.period))];
      const stops = used.map(code => GEO_PERIODS.find(p => p.code === code)?.color ?? '#888');
      const cs2 = document.getElementById('param-colorscale');
      cs2.querySelector('div').style.background = stops.length > 1
        ? `linear-gradient(to right,${stops.join(',')})` : `${stops[0] ?? '#888'}`;
      cs2.style.display = 'block';
      document.getElementById('param-scale-min').textContent = '';
      document.getElementById('param-scale-mid').textContent = '';
      document.getElementById('param-scale-max').textContent = '';
      document.getElementById('param-scale-label').textContent = `ICS geological age — ${used.map(c => GEO_PERIODS.find(p=>p.code===c)?.name ?? c).join(', ')}`;
      log(`Parameter view: geological age (${used.length} period(s) assigned)`, 'ok');
      return;
    }

    // Use spatially-interpolated 3D parameter volume if available
    const paramGrid = AppState.voxelGrid?.paramVolumes?.get(paramName) ?? null;
    const range = AppState.scene.colorByParameter(paramName, AppState.geoUnits, paramGrid);
    if (!range) {
      log(`No unit has the parameter "${paramName}" defined. Use Auto-Fill Parameters first.`, 'warn');
      return;
    }
    const { min, max } = range;
    const mid = ((min + max) / 2).toFixed(1);
    document.getElementById('param-scale-min').textContent = min.toFixed(1);
    document.getElementById('param-scale-mid').textContent = mid;
    document.getElementById('param-scale-max').textContent = max.toFixed(1);
    document.getElementById('param-scale-label').textContent =
      (paramGrid ? '3D IDW — ' : '') + (PARAM_LABELS[paramName] ?? paramName);
    document.getElementById('param-colorscale').style.display = 'block';
    log(`Parameter view: ${paramGrid ? '3D IDW ' : ''}${PARAM_LABELS[paramName] ?? paramName} (${min.toFixed(1)} – ${max.toFixed(1)})`, 'ok');
  });

  document.getElementById('btn-param-reset')?.addEventListener('click', () => {
    if (!AppState.scene) return;
    AppState.scene.resetUnitColors();
    const cs = document.getElementById('param-colorscale');
    cs.style.display = 'none';
    // Restore default blue-cyan-green-yellow-red gradient
    cs.querySelector('div').style.background = 'linear-gradient(to right,#0000ff,#00ffff,#00ff00,#ffff00,#ff0000)';
    log('Restored unit colours.', 'info');
  });
}

// ── Volume by Depth Slice ─────────────────────────────────────────────────────
function initVolumeByDepth() {
  const btn     = document.getElementById('btn-depth-vol-compute');
  const results = document.getElementById('depth-vol-results');

  btn?.addEventListener('click', () => {
    const grid = AppState.voxelGrid;
    if (!grid) { log('Build the 3D model first.', 'warn'); return; }

    const depthFrom = parseFloat(document.getElementById('depth-vol-from')?.value ?? '0');
    const depthTo   = parseFloat(document.getElementById('depth-vol-to')?.value ?? '10');
    if (!isFinite(depthFrom) || !isFinite(depthTo) || depthTo <= depthFrom) {
      log('Enter valid depth range (from < to).', 'warn'); return;
    }

    const { nx, ny, nz, cellSize: cs, cellHeight: ch, origin: O, unitIds } = grid;
    const cellVol = cs * cs * ch;

    // Ground reference: max collar elevation
    const groundRef = AppState.classifiedBH.reduce((m, b) => Math.max(m, b.groundLevel ?? 0), O.y + nz * ch);

    // Elevation range for this depth slice
    const elevTop    = groundRef - depthFrom;
    const elevBottom = groundRef - depthTo;

    const izTop    = Math.max(0, Math.min(nz - 1, Math.floor((elevTop    - O.y) / ch)));
    const izBottom = Math.max(0, Math.min(nz - 1, Math.floor((elevBottom - O.y) / ch)));
    const izMin = Math.min(izTop, izBottom);
    const izMax = Math.max(izTop, izBottom);

    const counts = {};
    AppState.geoUnits.forEach(u => { counts[u.id] = 0; });
    let total = 0;

    for (let iz = izMin; iz <= izMax; iz++) {
      for (let iy = 0; iy < ny; iy++) {
        for (let ix = 0; ix < nx; ix++) {
          const uid = unitIds[ix + iy * nx + iz * nx * ny];
          if (uid && counts[uid] !== undefined) { counts[uid]++; total++; }
        }
      }
    }

    if (!total) {
      if (results) { results.style.display = 'block'; results.innerHTML = '<p class="hint" style="font-size:10px">No model data in this depth range.</p>'; }
      return;
    }

    const sliceVol = total * cellVol;
    let html = `<div style="font-size:10px;color:var(--text-mid);margin-bottom:4px">
      Depth ${depthFrom}–${depthTo} m bgl · Total slice: ${sliceVol >= 1000 ? (sliceVol/1000).toFixed(1)+'k' : sliceVol.toFixed(0)} m³</div>`;

    AppState.geoUnits.forEach(u => {
      const n = counts[u.id] ?? 0;
      if (!n) return;
      const vol = n * cellVol;
      const pct = (n / total * 100).toFixed(1);
      html += `<div style="display:flex;align-items:center;gap:5px;margin-bottom:3px">
        <span style="width:9px;height:9px;border-radius:2px;background:${u.color};flex-shrink:0;display:inline-block"></span>
        <span style="font-size:10px;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${escHtml(u.code)}</span>
        <span style="font-size:9px;font-family:var(--font-mono);color:var(--accent)">${vol >= 1000 ? (vol/1000).toFixed(1)+'k' : vol.toFixed(0)} m³</span>
        <div style="width:40px;height:5px;background:var(--bg-surface);border-radius:2px;overflow:hidden">
          <div style="width:${pct}%;height:100%;background:${u.color}"></div>
        </div>
        <span style="font-size:9px;font-family:var(--font-mono);color:var(--text-dim);width:28px;text-align:right">${pct}%</span>
      </div>`;
    });

    if (results) { results.style.display = 'block'; results.innerHTML = html; }
    log(`Depth slice ${depthFrom}–${depthTo}m: ${sliceVol.toFixed(0)} m³ total (${izMax - izMin + 1} Z levels)`, 'ok');
  });

  document.addEventListener('geomodel:model-built', () => {
    if (btn) btn.disabled = false;
  });
}

// ── Grade Shell ───────────────────────────────────────────────────────────────
function initGradeShell() {
  const modeEl     = document.getElementById('grade-mode');
  const maxWrap    = document.getElementById('grade-max-wrap');
  const applyBtn   = document.getElementById('btn-grade-apply');
  const clearBtn   = document.getElementById('btn-grade-clear');
  const statsEl    = document.getElementById('grade-stats');

  // Show/hide max input based on mode
  modeEl?.addEventListener('change', () => {
    if (maxWrap) maxWrap.style.display = modeEl.value === 'between' ? 'flex' : 'none';
  });
  if (maxWrap) maxWrap.style.display = 'none';

  applyBtn?.addEventListener('click', () => {
    if (!AppState.voxelGrid || !AppState.scene) { log('Build the 3D model first.', 'warn'); return; }
    const param     = document.getElementById('grade-param')?.value;
    if (!param) { log('Select a parameter for the grade shell.', 'warn'); return; }
    const mode      = modeEl?.value ?? 'above';
    const minVal    = parseFloat(document.getElementById('grade-threshold-min')?.value ?? 'NaN');
    const maxVal    = parseFloat(document.getElementById('grade-threshold-max')?.value ?? 'NaN');
    const highlight = document.getElementById('grade-color')?.value ?? '#ff4040';
    const dimOthers = document.getElementById('grade-dim-rest')?.checked ?? true;

    if (!isFinite(minVal)) { log('Enter a threshold value.', 'warn'); return; }
    if (mode === 'between' && !isFinite(maxVal)) { log('Enter a max value for "between" mode.', 'warn'); return; }

    const result = AppState.scene.colorByGradeShell(
      param, minVal, maxVal, mode, highlight, dimOthers, AppState.geoUnits
    );
    if (!result) { log(`No data for parameter "${param}" — assign parameters or build model first.`, 'warn'); return; }

    const pct = result.total > 0 ? ((result.matched / result.total) * 100).toFixed(1) : '0';
    if (statsEl) statsEl.textContent = `${result.matched.toLocaleString()} / ${result.total.toLocaleString()} voxels highlighted (${pct}%)`;
    const label = document.getElementById('grade-param')?.selectedOptions[0]?.text ?? param;
    const condStr = mode === 'between' ? `${minVal}–${maxVal}` : `${mode === 'above' ? '≥' : '≤'}${minVal}`;
    log(`Grade shell: ${label} ${condStr} — ${result.matched.toLocaleString()} voxels (${pct}%)`, 'ok');
  });

  clearBtn?.addEventListener('click', () => {
    if (!AppState.scene) return;
    AppState.scene.resetUnitColors();
    if (statsEl) statsEl.textContent = '';
    log('Grade shell cleared.', 'info');
  });

  // Enable buttons when model is built
  document.addEventListener('geomodel:built', () => {
    if (applyBtn) applyBtn.disabled = false;
  });
}

// ── Cross-Plot (scatter plot) ─────────────────────────────────────────────────
function initCrossPlot() {
  document.getElementById('btn-crossplot-draw')?.addEventListener('click', () => _drawCrossPlot());
}

function _drawCrossPlot() {
  const canvas = document.getElementById('crossplot-canvas');
  const wrap   = document.getElementById('crossplot-wrap');
  const legend = document.getElementById('crossplot-legend');
  if (!canvas || !AppState.classifiedBH.length) {
    log('Load borehole data first.', 'warn'); return;
  }

  const xParam = document.getElementById('crossplot-x')?.value ?? 'depth';
  const yParam = document.getElementById('crossplot-y')?.value ?? 'N_spt';

  const PARAM_LABELS = {
    depth: 'Depth (m bgl)', N_spt: 'SPT N', cu: 'Cu (kPa)',
    phi: 'φ′ (°)', E: 'E (MPa)', gamma: 'γ (kN/m³)', Cc: 'Cc',
  };

  // Collect data points: {x, y, unitCode, unitColor}
  const points = [];
  for (const bh of AppState.classifiedBH) {
    if (bh.synthetic || !bh.layers?.length) continue;
    for (const layer of bh.layers) {
      if (!layer.unitCode) continue;
      const unit = AppState.geoUnits.find(u => u.code === layer.unitCode);
      const depth = (layer.top + layer.base) / 2;

      const getVal = (p) => {
        if (p === 'depth') return depth;
        if (p === 'N_spt') return layer.sptN ?? null;
        return unit?.params?.[p] ?? null;
      };

      const xv = getVal(xParam), yv = getVal(yParam);
      if (xv != null && yv != null && isFinite(xv) && isFinite(yv)) {
        points.push({ x: xv, y: yv, unitCode: layer.unitCode, color: unit?.color ?? '#888888' });
      }
    }
  }

  if (!points.length) {
    log(`No data for ${PARAM_LABELS[xParam]} vs ${PARAM_LABELS[yParam]}. Check borehole layers have SPT/parameter values.`, 'warn');
    return;
  }

  const W = 260, H = 200, PAD = 36, PAD_R = 10, PAD_T = 10;
  canvas.width = W; canvas.height = H;
  if (wrap) wrap.style.display = 'block';
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, W, H);

  const xs = points.map(p => p.x), ys = points.map(p => p.y);
  const xMin = Math.min(...xs), xMax = Math.max(...xs) || xMin + 1;
  const yMin = Math.min(...ys), yMax = Math.max(...ys) || yMin + 1;
  const xR = xMax - xMin || 1, yR = yMax - yMin || 1;

  const toCanX = (v) => PAD + (v - xMin) / xR * (W - PAD - PAD_R);
  const toCanY = (v) => PAD_T + (1 - (v - yMin) / yR) * (H - PAD - PAD_T);

  // Axes
  ctx.strokeStyle = '#999'; ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(PAD, PAD_T); ctx.lineTo(PAD, H - PAD);
  ctx.moveTo(PAD, H - PAD); ctx.lineTo(W - PAD_R, H - PAD);
  ctx.stroke();

  // Axis labels
  ctx.fillStyle = '#555'; ctx.font = '9px sans-serif'; ctx.textAlign = 'center';
  ctx.fillText(PARAM_LABELS[xParam] ?? xParam, (PAD + W - PAD_R) / 2, H - 4);
  ctx.save(); ctx.translate(10, (PAD_T + H - PAD) / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(PARAM_LABELS[yParam] ?? yParam, 0, 0);
  ctx.restore();

  // Tick marks + values
  ctx.font = '8px sans-serif'; ctx.fillStyle = '#888';
  const nTicks = 4;
  for (let i = 0; i <= nTicks; i++) {
    const xv = xMin + i / nTicks * xR;
    const xc = toCanX(xv);
    ctx.textAlign = 'center';
    ctx.fillText(xv.toFixed(xR < 1 ? 2 : 0), xc, H - PAD + 10);
    ctx.strokeStyle = '#ddd'; ctx.lineWidth = 0.5;
    ctx.beginPath(); ctx.moveTo(xc, PAD_T); ctx.lineTo(xc, H - PAD); ctx.stroke();
  }
  for (let i = 0; i <= nTicks; i++) {
    const yv = yMin + i / nTicks * yR;
    const yc = toCanY(yv);
    ctx.textAlign = 'right';
    ctx.fillText(yv.toFixed(yR < 1 ? 2 : 0), PAD - 3, yc + 3);
    ctx.strokeStyle = '#ddd'; ctx.lineWidth = 0.5;
    ctx.beginPath(); ctx.moveTo(PAD, yc); ctx.lineTo(W - PAD_R, yc); ctx.stroke();
  }

  // Points
  ctx.lineWidth = 0;
  for (const p of points) {
    ctx.fillStyle = p.color;
    ctx.globalAlpha = 0.72;
    ctx.beginPath();
    ctx.arc(toCanX(p.x), toCanY(p.y), 3.5, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.globalAlpha = 1;

  // Legend
  const units = [...new Map(points.map(p => [p.unitCode, { code: p.unitCode, color: p.color }])).values()];
  if (legend) {
    legend.innerHTML = units.map(u => {
      const label = AppState.geoUnits.find(g => g.code === u.code)?.name ?? u.code;
      return `<span style="display:flex;align-items:center;gap:3px">
        <span style="width:9px;height:9px;border-radius:50%;background:${u.color};flex-shrink:0;display:inline-block"></span>
        <span>${escHtml(label)}</span>
      </span>`;
    }).join('');
  }

  log(`Cross-plot: ${PARAM_LABELS[xParam]} vs ${PARAM_LABELS[yParam]} — ${points.length} data points`, 'ok');
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

  // ── Orientation → Concept ─────────────────────────────────────────────────────
  // Convert the parsed mean strike/dip into a 32-dim concept embedding and add it
  // to the ConceptStore. This bridges structural geology observations (measured dip
  // and azimuth) with the semantic embedding space that shapes neural field geometry.
  document.getElementById('btn-orientation-to-concept')?.addEventListener('click', () => {
    const text = document.getElementById('orientation-text')?.value ?? '';
    if (!text.trim()) { log('Enter strike/dip measurements first.', 'warn'); return; }

    // Parse measurements: same logic as the parse button but also extract dip
    const lines   = text.split(/\r?\n/).map(l => l.trim()).filter(Boolean);
    const strikes = [], dips = [];
    for (const line of lines) {
      const parts = line.replace(/[°,]/g, ' ').split(/\s+/);
      // Try: "azimuth dip" plain pair
      if (parts.length >= 2 && isFinite(parts[0]) && isFinite(parts[1])) {
        strikes.push(parseFloat(parts[0]));
        dips.push(parseFloat(parts[1]));
        continue;
      }
      // Try: "strike NNN, dip DD [dir]"
      const sm = line.match(/strike\s+(\d+)/i);
      const dm = line.match(/dip\s+(\d+)/i);
      if (sm) strikes.push(parseFloat(sm[1]));
      if (dm) dips.push(parseFloat(dm[1]));
    }
    if (!strikes.length) { log('No valid measurements found.', 'warn'); return; }

    // Circular mean strike azimuth
    const sinS = strikes.reduce((s, a) => s + Math.sin(a * Math.PI / 180), 0);
    const cosS = strikes.reduce((s, a) => s + Math.cos(a * Math.PI / 180), 0);
    let meanStrike = Math.atan2(sinS, cosS) * 180 / Math.PI;
    if (meanStrike < 0) meanStrike += 360;
    const meanDip  = dips.length ? dips.reduce((a, b) => a + b, 0) / dips.length : AppState.anisoAzimuth ? 5 : 0;

    // Build 32-dim embedding from strike/dip geometry
    const emb = new Float32Array(32);

    // Strike direction → elongation axes [3]=EW, [4]=NS
    // Bodies are elongated ALONG strike (perpendicular to dip direction)
    const strikeRad = meanStrike * Math.PI / 180;
    const ewComponent = Math.abs(Math.sin(strikeRad)); // sin of strike = EW component
    const nsComponent = Math.abs(Math.cos(strikeRad)); // cos of strike = NS component
    emb[3] = Math.min(0.9, ewComponent * 1.0);          // east_west_elongation
    emb[4] = Math.min(0.9, nsComponent * 1.0);          // north_south_elongation
    emb[27] = 0.7;                                        // lateral_anisotropy

    // Dip magnitude
    const dipNorm = Math.min(1, meanDip / 90);
    emb[1] = Math.min(0.9, dipNorm * 1.2);             // inclined_bedding
    emb[2] = dipNorm;                                    // dip_magnitude

    // Deepening direction: dip direction = strike + 90° (right-hand rule)
    const dipDirRad = (meanStrike + 90) * Math.PI / 180;
    const dipEast   = Math.sin(dipDirRad);
    const dipNorth  = Math.cos(dipDirRad);
    if (dipNorm > 0.05) {
      if (dipEast  >  0.4) emb[14] = dipEast  * dipNorm;   // deepens_east
      if (dipEast  < -0.4) emb[15] = -dipEast * dipNorm;   // deepens_west
      if (dipNorth >  0.4) emb[16] = dipNorth * dipNorm;   // deepens_north
      if (dipNorth < -0.4) emb[17] = -dipNorth * dipNorm;  // deepens_south
    }

    // Horizontal beds for very low dip
    if (meanDip < 5) { emb[0] = 0.7; emb[9] = 0.6; }    // horizontal_layering, lateral_continuity

    // Confidence based on number of measurements
    const conf = Math.min(0.9, 0.5 + Math.min(strikes.length / 10, 0.35));

    // Compose description
    const ewStr = ewComponent > nsComponent ? 'E-W elongated' : 'N-S elongated';
    const dipStr = meanDip > 5 ? `, dipping ${meanDip.toFixed(0)}° toward ${(meanStrike + 90) % 360 < 180 ? 'E' : 'W'}/${(meanStrike + 90) % 360 < 90 || (meanStrike + 90) % 360 > 270 ? 'N' : 'S'}` : ', sub-horizontal';
    const desc = `Structural fabric: strike ${meanStrike.toFixed(0)}°${dipStr} — ${ewStr}`;

    if (!AppState.conceptStore) AppState.conceptStore = new ConceptStore();
    AppState.conceptStore.add({ description: desc, embedding: emb, confidence: conf, domain: { type: 'global' } });
    _renderConceptList();
    _saveConceptStore();
    switchTab('concepts');
    log(`Structural orientation encoded as concept: ${desc.slice(0, 80)}`, 'ok');
    document.getElementById('orientation-result').textContent =
      `✓ Encoded as concept: "${desc.slice(0, 60)}…"`;
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

  // ── Unit similarity analysis ───────────────────────────────────────────────
  document.getElementById('btn-unit-similarity')?.addEventListener('click', async () => {
    const units = AppState.geoUnits.filter(u => u.code !== 'UNKN');
    if (units.length < 2) { log('Need at least 2 units for similarity analysis.', 'warn'); return; }
    const btn = document.getElementById('btn-unit-similarity');
    const resEl = document.getElementById('unit-similarity-results');
    if (!resEl) { log('UI element missing: unit-similarity-results', 'error'); return; }
    setEnabled('btn-unit-similarity', false);
    if (btn) btn.textContent = '⊛ Analysing…';
    const apiKey = sessionStorage.getItem('anthropic_api_key') ?? '';
    try {
      const { pairs } = await analyseUnitSimilarity(units, apiKey, !apiKey || AppState.demoMode);
      resEl.style.display = '';
      if (!pairs.length) {
        resEl.innerHTML = '<p style="color:var(--text-muted);font-style:italic">No similar unit pairs detected — all units appear distinct.</p>';
        log('Unit similarity: all units appear distinct.', 'ok');
      } else {
        resEl.innerHTML = pairs.map(p => {
          const pct = Math.round(p.similarity * 100);
          const col = pct >= 88 ? 'var(--error)' : pct >= 72 ? 'var(--warn)' : 'var(--text-mid)';
          return `<div style="border-left:3px solid ${col};padding:4px 6px;margin-bottom:5px;background:var(--bg-surface)">
            <div style="font-weight:600;color:${col}">${p.codeA} ↔ ${p.codeB} <span style="font-weight:400;float:right">${pct}%</span></div>
            <div style="color:var(--text-mid);margin-top:2px">${p.nameA} / ${p.nameB}</div>
            ${p.sharedTokens?.length ? `<div style="color:var(--text-muted);font-size:9px;margin-top:2px">Shared: ${p.sharedTokens.slice(0, 6).join(', ')}</div>` : ''}
            <div style="color:var(--text-dim);font-style:italic;margin-top:2px;font-size:9px">${p.suggestion}</div>
          </div>`;
        }).join('');
        log(`Unit similarity: ${pairs.length} similar pair(s) found — highest ${Math.round(pairs[0].similarity * 100)}% (${pairs[0].codeA} ↔ ${pairs[0].codeB})`, pairs[0].similarity > 0.88 ? 'warn' : 'info');
      }
    } catch (e) {
      resEl.style.display = '';
      resEl.innerHTML = `<p style="color:var(--error);font-size:10px">Error: ${e.message}</p>`;
      log(`Unit similarity error: ${e.message}`, 'error');
    } finally {
      if (btn) btn.textContent = '⊛ Analyse Unit Similarity';
      setEnabled('btn-unit-similarity', true);
    }
  });

  document.getElementById('btn-detect-pinchouts')?.addEventListener('click', () => {
    const resEl = document.getElementById('pinchout-results');
    if (!resEl) return;
    if (!AppState.voxelGrid) { log('Build the 3D model first.', 'warn'); return; }
    let pinchMap;
    try {
      pinchMap = detectPinchouts(AppState.voxelGrid, AppState.geoUnits);
    } catch (e) { log(`Pinch-out detection failed: ${e.message}`, 'error'); return; }

    if (!pinchMap.size) {
      resEl.style.display = 'block';
      resEl.innerHTML = '<em style="color:#aaa">No pinch-outs detected — all units span the full model extent.</em>';
      return;
    }

    const rows = [];
    for (const [code, info] of pinchMap.entries()) {
      const pct = (info.coverageFraction * 100).toFixed(1);
      const edgeCount = info.pinchoutEdges.length;
      const unit = AppState.geoUnits.find(u => u.code === code);
      const col = unit?.color ?? '#888';
      rows.push(`<div style="display:flex;align-items:center;gap:6px;margin-bottom:4px">
        <span style="width:10px;height:10px;border-radius:2px;background:${col};flex-shrink:0"></span>
        <span style="flex:1"><strong>${code}</strong></span>
        <span style="color:#ccc">${pct}% coverage</span>
        <span style="color:#f5a623">${edgeCount} edge col${edgeCount !== 1 ? 's' : ''}</span>
      </div>`);
    }
    resEl.style.display = 'block';
    resEl.innerHTML = `<div style="margin-bottom:6px;color:#aaa">${pinchMap.size} unit${pinchMap.size !== 1 ? 's' : ''} with lateral terminations:</div>${rows.join('')}`;
    log(`Pinch-out detection: ${pinchMap.size} units with lateral terminations`, 'info');
  });

  document.getElementById('btn-seq-surfaces')?.addEventListener('click', () => {
    const grid = AppState.voxelGrid;
    if (!grid) { log('Build the 3D model first.', 'warn'); return; }
    const resEl = document.getElementById('seq-surfaces-results');

    try {
      const surfaces = identifySequenceSurfaces(grid, AppState.geoUnits, AppState.conceptStore);
      if (!surfaces.length) {
        if (resEl) { resEl.style.display = 'block'; resEl.innerHTML = '<p class="hint" style="font-size:10px">No sequence surfaces identified. Add temporally ordered concepts (set temporal rank in concept controls) to enable this analysis.</p>'; }
        return;
      }

      const unitById = {};
      AppState.geoUnits.forEach(u => { unitById[u.code] = u; });

      const rows = surfaces.slice(0, 12).map(s => {
        const youngUnit = unitById[s.youngerCode];
        const oldUnit   = unitById[s.olderCode];
        const youngCol  = youngUnit?.color ?? '#888';
        const oldCol    = oldUnit?.color   ?? '#666';
        const isBoundary = s.type === 'boundary';
        const typeLabel  = isBoundary ? 'Sequence boundary' : '⚠ Inversion';
        const typeColor  = isBoundary ? 'var(--accent)' : 'var(--red)';
        const elevRange  = s.elevMin !== s.elevMax
          ? `${s.elevMin} – ${s.elevMax} m AOD` : `${s.elevation} m AOD`;
        return `<div style="padding:5px;border:1px solid var(--border);border-radius:4px;margin-bottom:3px">
          <div style="display:flex;align-items:center;gap:4px;margin-bottom:2px">
            <span style="font-size:10px;font-weight:600;color:${typeColor}">${typeLabel}</span>
            <span style="margin-left:auto;font-size:9px;color:var(--text-dim)">${s.voxelCount} contacts</span>
          </div>
          <div style="display:flex;align-items:center;gap:3px;font-size:9.5px">
            <span style="width:8px;height:8px;border-radius:1px;background:${youngCol};flex-shrink:0"></span>
            <span style="color:var(--text)">${escHtml(s.youngerCode)}</span>
            <span style="color:var(--text-dim)"> overlies </span>
            <span style="width:8px;height:8px;border-radius:1px;background:${oldCol};flex-shrink:0"></span>
            <span style="color:var(--text)">${escHtml(s.olderCode)}</span>
          </div>
          <div style="font-size:9px;color:var(--text-mid);margin-top:2px">${elevRange}</div>
        </div>`;
      });

      const nBound = surfaces.filter(s => s.type === 'boundary').length;
      const nRev   = surfaces.filter(s => s.type === 'reversal').length;
      if (resEl) {
        resEl.style.display = 'block';
        resEl.innerHTML = `<div style="font-size:10px;font-weight:600;color:var(--text-mid);margin-bottom:5px">
          ${nBound} surface${nBound !== 1 ? 's' : ''} · ${nRev > 0 ? `${nRev} inversion${nRev !== 1 ? 's' : ''} ⚠` : 'no inversions'}
        </div>${rows.join('')}`;
      }
      log(`Sequence stratigraphy: ${nBound} surface(s) identified${nRev > 0 ? `, ${nRev} inversion(s)` : ''}`, nRev > 0 ? 'warn' : 'ok');
    } catch (e) {
      log(`Sequence surface identification failed: ${e.message}`, 'error');
    }
  });

  document.getElementById('btn-recommend-drilling')?.addEventListener('click', () => {
    const grid = AppState.voxelGrid;
    if (!grid) { log('Build the 3D model first.', 'warn'); return; }
    const resEl = document.getElementById('drill-rec-results');
    const n = parseInt(document.getElementById('drill-rec-n')?.value ?? '5') || 5;

    try {
      const recs = recommendDrillingLocations(grid, AppState.geoUnits, AppState.conceptStore, n);
      if (!recs.length) {
        if (resEl) { resEl.style.display = 'block'; resEl.innerHTML = '<p class="hint" style="font-size:10px">No recommendations — model uncertainty is uniformly low.</p>'; }
        return;
      }

      const unitById = {};
      AppState.geoUnits.forEach(u => { unitById[u.id] = u; });

      const rows = recs.map((r, i) => {
        const uncPct = Math.round(r.uncert * 100);
        const bar = `<div style="display:inline-block;width:${uncPct}px;max-width:80px;height:5px;background:var(--red);border-radius:2px;vertical-align:middle"></div>`;
        return `<div style="padding:5px;border:1px solid var(--border);border-radius:4px;margin-bottom:4px">
          <div style="display:flex;align-items:center;gap:5px;margin-bottom:2px">
            <span style="font-size:11px;font-weight:600;color:var(--accent)">${i + 1}</span>
            <span style="font-family:var(--font-mono);font-size:9.5px;color:var(--text-primary)">E ${r.x.toFixed(0)}m &nbsp; N ${r.y.toFixed(0)}m</span>
            <span style="margin-left:auto;font-size:9px;color:var(--text-dim)">${bar} ${uncPct}% uncert</span>
          </div>
          <div style="font-size:9px;color:var(--text-mid);line-height:1.4">${escHtml(r.reason)}</div>
        </div>`;
      });

      if (resEl) {
        resEl.style.display = 'block';
        resEl.innerHTML = `<div style="font-size:10px;font-weight:600;color:var(--text-mid);margin-bottom:5px">
          ${recs.length} recommended location(s) <span style="font-size:9px;font-weight:400">(uncertainty + concept geometry + BH coverage)</span>
        </div>${rows.join('')}`;
      }
      log(`Drilling recommendation: ${recs.length} optimal location(s) identified`, 'ok');
    } catch (e) {
      log(`Drilling recommendation failed: ${e.message}`, 'error');
    }
  });

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

  // ── Investigation Planning — suggest drill locations ──────────────────────
  document.getElementById('btn-drill-plan')?.addEventListener('click', () => {
    const grid = AppState.voxelGrid;
    if (!grid) { log('Build the 3D model first.', 'warn'); return; }
    const nSuggest = parseInt(document.getElementById('drill-plan-n')?.value ?? '5') || 5;
    const { nx, ny, nz, cellSize: cs, cellHeight: ch, origin: O, unitIds, certainty, blendRatios, probVolumes } = grid;
    const LOG2 = Math.log(2);
    const nUnits = Math.max(2, AppState.geoUnits.length);

    // Compute per-column mean entropy — use exact Shannon entropy from probVolumes if available
    const colEntropy = new Float32Array(nx * ny);
    const probArrays = probVolumes?.size > 0 ? [...probVolumes.values()] : null;
    const xEnt = p => (p > 1e-6 && p < 1 - 1e-6) ? -p * Math.log(p) / LOG2 : 0;
    for (let iy = 0; iy < ny; iy++) {
      for (let ix = 0; ix < nx; ix++) {
        let sumH = 0, cnt = 0;
        for (let iz = 0; iz < nz; iz++) {
          const flat = ix + iy * nx + iz * nx * ny;
          if (!unitIds[flat]) continue;
          let H;
          if (probArrays) {
            H = 0;
            for (const arr of probArrays) H += xEnt(arr[flat]);
          } else {
            const p1 = Math.max(0.001, Math.min(0.999, certainty[flat]));
            const p2 = Math.max(0, Math.min(1 - p1, blendRatios ? (blendRatios[flat] ?? 0) : 0));
            const pR = Math.max(0, 1 - p1 - p2);
            H = xEnt(p1) + xEnt(p2) + xEnt(pR);
          }
          sumH += H;
          cnt++;
        }
        colEntropy[ix + iy * nx] = cnt > 0 ? sumH / cnt : 0;
      }
    }

    // Compute distance penalty: existing boreholes reduce priority of nearby cells
    const realBHs = AppState.classifiedBH.filter(b => !b.synthetic && b.layers?.length);
    const SIGMA_SQ = Math.pow(nx * cs * 0.2, 2); // 20% of site width
    const distPenalty = new Float32Array(nx * ny);
    for (let iy = 0; iy < ny; iy++) {
      for (let ix = 0; ix < nx; ix++) {
        const wx = O.x + (ix + 0.5) * cs;
        const wy = O.z + (iy + 0.5) * cs;
        let pen = 0;
        for (const bh of realBHs) {
          const d2 = (bh.x - wx) ** 2 + (bh.y - wy) ** 2;
          pen = Math.max(pen, Math.exp(-d2 / SIGMA_SQ));
        }
        distPenalty[ix + iy * nx] = pen;
      }
    }

    // Per-column mean concept influence (0 = data-driven, 1 = concept-driven)
    const colConceptInf = new Float32Array(nx * ny);
    const hasConceptInf = !!(grid.conceptInfluence);
    if (hasConceptInf) {
      for (let iy = 0; iy < ny; iy++) {
        for (let ix = 0; ix < nx; ix++) {
          let sumInf = 0, cnt = 0;
          for (let iz = 0; iz < nz; iz++) {
            const flat = ix + iy * nx + iz * nx * ny;
            if (!unitIds[flat]) continue;
            sumInf += grid.conceptInfluence[flat];
            cnt++;
          }
          colConceptInf[ix + iy * nx] = cnt > 0 ? sumInf / cnt : 0;
        }
      }
    }

    // Score = entropy × (1 − proximity) × (1 + concept_influence × 0.5)
    // The concept-influence boost prioritises locations where the model is relying on
    // semantic knowledge rather than borehole data — drilling there replaces inference
    // with real observations (maximum information gain for this level of uncertainty).
    const scores = [];
    for (let iy = 0; iy < ny; iy++) {
      for (let ix = 0; ix < nx; ix++) {
        const col = ix + iy * nx;
        const conceptBoostFactor = hasConceptInf ? (1 + colConceptInf[col] * 0.5) : 1;
        const score = colEntropy[col] * (1 - distPenalty[col]) * conceptBoostFactor;
        if (score > 0) scores.push({ ix, iy, score, entropy: colEntropy[col], conceptInf: colConceptInf[col] });
      }
    }
    scores.sort((a, b) => b.score - a.score);

    // Select top N with min spacing (greedy diverse selection)
    const selected = [];
    const minSpacing = cs * Math.max(2, Math.floor(Math.min(nx, ny) / (nSuggest + 1)));
    for (const s of scores) {
      if (selected.length >= nSuggest) break;
      const wx = O.x + (s.ix + 0.5) * cs;
      const wy = O.z + (s.iy + 0.5) * cs;
      if (selected.every(sel => Math.hypot(sel.wx - wx, sel.wy - wy) >= minSpacing)) {
        selected.push({ ...s, wx, wy });
      }
    }

    const out = document.getElementById('drill-plan-results');
    if (!out) return;
    out.style.display = 'block';
    if (!selected.length) {
      out.innerHTML = '<p class="hint" style="font-size:10px">No high-uncertainty locations found.</p>';
      return;
    }

    const maxEnt = Math.log2(nUnits);
    const showConceptCol = hasConceptInf && selected.some(s => s.conceptInf > 0.1);
    out.innerHTML = `<table style="width:100%;font-size:10px;border-collapse:collapse">
      <thead><tr style="color:var(--text-muted)">
        <th style="text-align:left;padding:2px 4px">Location</th>
        <th style="padding:2px 4px">E (m)</th>
        <th style="padding:2px 4px">N (m)</th>
        <th style="padding:2px 4px">Uncertainty</th>
        ${showConceptCol ? '<th style="padding:2px 4px" title="Fraction of prediction driven by conceptual model rather than borehole data — high = concept-reliant = highest priority for drilling">Concept</th>' : ''}
        <th style="padding:2px 4px;font-size:9px">Reason</th>
      </tr></thead>
      <tbody>
        ${selected.map((s, i) => {
          const pct = (s.entropy / maxEnt * 100).toFixed(0);
          const bar = '▓'.repeat(Math.round(pct / 10)) + '░'.repeat(10 - Math.round(pct / 10));
          const infPct = showConceptCol ? (s.conceptInf * 100).toFixed(0) : '0';
          const reasons = [];
          if (s.entropy / maxEnt > 0.5) reasons.push('high uncertainty');
          if (s.conceptInf > 0.4) reasons.push('concept-reliant');
          else if (s.conceptInf > 0.2) reasons.push('concept-influenced');
          if (distPenalty[s.ix + s.iy * nx] < 0.15) reasons.push('data-sparse');
          const reason = reasons.join(' + ') || 'uncertain';
          const infColor = s.conceptInf > 0.5 ? 'color:#e8a020' : s.conceptInf > 0.25 ? 'color:#d4c020' : 'color:var(--text-dim)';
          return `<tr style="border-top:1px solid var(--border)">
            <td style="padding:2px 4px;color:var(--accent);font-weight:600">BH-${i+1}</td>
            <td style="padding:2px 4px;font-family:var(--font-mono)">${s.wx.toFixed(0)}</td>
            <td style="padding:2px 4px;font-family:var(--font-mono)">${s.wy.toFixed(0)}</td>
            <td style="padding:2px 4px;font-family:var(--font-mono);font-size:9px" title="${pct}% of max entropy">${bar} ${pct}%</td>
            ${showConceptCol ? `<td style="padding:2px 4px;font-family:var(--font-mono);font-size:9px;${infColor}">${infPct}%</td>` : ''}
            <td style="padding:2px 4px;font-size:9px;color:var(--text-dim)">${reason}</td>
          </tr>`;
        }).join('')}
      </tbody>
    </table>
    <p class="hint" style="font-size:9px;margin-top:4px">Ranked by classification entropy × (1 − BH proximity)${hasConceptInf ? ' × concept-reliance factor' : ''}. ${hasConceptInf ? 'Amber = concept-driven areas where drilling replaces semantic inference with real data.' : 'Build with neural-implicit to see concept-reliance column.'}</p>`;

    log(`Investigation planning: ${selected.length} suggested locations. Top location: E${selected[0].wx.toFixed(0)} N${selected[0].wy.toFixed(0)} (${(selected[0].entropy / maxEnt * 100).toFixed(0)}% max entropy)`, 'ok');
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

  // Probabilistic volume bounds from MC probability volumes (when available).
  // E[V] = Σ P(unit) × voxelVol; Var[V] = Σ P(unit)(1-P(unit)) × voxelVol²
  // P10 ≈ E - 1.28σ, P90 ≈ E + 1.28σ (normal approximation of Bernoulli sum)
  const probVolMap = grid.probVolumes; // Map<unitCode, Float32Array> or undefined
  const probStats = {};
  if (probVolMap?.size > 0) {
    const total = nx * ny * nz;
    for (const [code, probArr] of probVolMap) {
      let eVol = 0, varVol = 0;
      for (let i = 0; i < total; i++) {
        const p = probArr[i];
        eVol   += p;
        varVol += p * (1 - p);
      }
      eVol   *= voxelVol;
      varVol *= voxelVol * voxelVol;
      const sigma = Math.sqrt(Math.max(0, varVol));
      probStats[code] = {
        mean: eVol,
        p10:  Math.max(0, eVol - 1.28 * sigma),
        p90:  eVol + 1.28 * sigma,
      };
    }
  }

  el.innerHTML = '';
  geoUnits.forEach(unit => {
    const s = stats[unit.id];
    if (!s || !s.count) return;
    const avgCert  = (s.certSum / s.count * 100).toFixed(0);
    const thick    = (s.maxElev - s.minElev).toFixed(1);
    const volume   = (s.count * voxelVol).toFixed(0);
    const ps       = probStats[unit.code];
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
          ${ps
            ? `<span class="stat-lbl">Vol (P50)</span><span class="stat-val">${Math.round(ps.mean).toLocaleString()} m³</span>
               <span class="stat-lbl" title="10th–90th percentile from MC uncertainty">P10–P90</span><span class="stat-val" style="font-size:9px">${Math.round(ps.p10).toLocaleString()}–${Math.round(ps.p90).toLocaleString()} m³</span>`
            : `<span class="stat-lbl">Vol</span><span class="stat-val">${Number(volume).toLocaleString()} m³</span>`
          }
        </div>
      </div>`;
    el.appendChild(div);
  });
}

// ── Build progress bar helpers ─────────────────────────────────────────────────
// Ring buffer for loss curve (max 120 points)
const _lossHistory = { data: [], maxLen: 120 };

function showBuildProgress(visible) {
  const el = document.getElementById('build-progress-wrap');
  if (el) el.hidden = !visible;
  if (!visible) {
    // Clear loss history when hiding so next build starts fresh
    _lossHistory.data = [];
    const canvas = document.getElementById('build-loss-canvas');
    const meta   = document.getElementById('build-loss-meta');
    if (canvas) canvas.hidden = true;
    if (meta)   meta.hidden   = true;
  }
}

function _drawLossCurve() {
  const canvas = document.getElementById('build-loss-canvas');
  if (!canvas) return;
  const pts = _lossHistory.data;
  if (pts.length < 2) return;

  canvas.hidden = false;
  const dpr = window.devicePixelRatio || 1;
  const W   = canvas.offsetWidth  || 260;
  const H   = canvas.offsetHeight || 48;
  canvas.width  = W * dpr;
  canvas.height = H * dpr;

  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, W, H);

  // Background gradient
  const bg = ctx.createLinearGradient(0, 0, 0, H);
  bg.addColorStop(0, '#f0f4ff');
  bg.addColorStop(1, '#f8f9fa');
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, W, H);

  const losses = pts.map(p => p.loss);
  const minL = Math.min(...losses);
  const maxL = Math.max(...losses);
  const range = maxL - minL || 1;

  const pad = { t: 6, b: 14, l: 6, r: 6 };
  const plotW = W - pad.l - pad.r;
  const plotH = H - pad.t - pad.b;

  const toX = i  => pad.l + (i / (pts.length - 1)) * plotW;
  const toY = l  => pad.t + (1 - (l - minL) / range) * plotH;

  // Horizontal guide lines (3 levels)
  ctx.strokeStyle = 'rgba(180,190,220,0.5)';
  ctx.lineWidth = 0.5;
  for (let k = 0; k <= 2; k++) {
    const y = pad.t + (k / 2) * plotH;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + plotW, y); ctx.stroke();
  }

  // Area fill under curve
  const grad = ctx.createLinearGradient(0, pad.t, 0, pad.t + plotH);
  grad.addColorStop(0, 'rgba(79,140,255,0.25)');
  grad.addColorStop(1, 'rgba(79,140,255,0.02)');
  ctx.beginPath();
  ctx.moveTo(toX(0), toY(losses[0]));
  for (let i = 1; i < pts.length; i++) ctx.lineTo(toX(i), toY(losses[i]));
  ctx.lineTo(toX(pts.length - 1), pad.t + plotH);
  ctx.lineTo(toX(0), pad.t + plotH);
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();

  // Loss line
  ctx.beginPath();
  ctx.moveTo(toX(0), toY(losses[0]));
  for (let i = 1; i < pts.length; i++) ctx.lineTo(toX(i), toY(losses[i]));
  ctx.strokeStyle = '#4f8cff';
  ctx.lineWidth = 1.5;
  ctx.lineJoin = 'round';
  ctx.stroke();

  // Endpoint dot
  const lastX = toX(pts.length - 1);
  const lastY = toY(losses[pts.length - 1]);
  ctx.beginPath();
  ctx.arc(lastX, lastY, 2.5, 0, Math.PI * 2);
  ctx.fillStyle = '#4f8cff';
  ctx.fill();

  // Axis labels (min/max loss and epoch)
  ctx.fillStyle = 'rgba(80,90,110,0.85)';
  ctx.font = `${9 * dpr / dpr}px var(--font-mono, monospace)`;
  ctx.textBaseline = 'bottom';
  ctx.fillText(maxL.toFixed(3), pad.l + 1, pad.t + plotH - 1);
  ctx.textBaseline = 'top';
  ctx.fillText(minL.toFixed(3), pad.l + 1, pad.t + 1);
  ctx.textBaseline = 'bottom';
  ctx.textAlign = 'right';
  ctx.fillText(`ep ${pts[pts.length - 1].epoch}`, pad.l + plotW, pad.t + plotH + 12);
  ctx.textAlign = 'left';
  ctx.fillText(`ep ${pts[0].epoch}`, pad.l, pad.t + plotH + 12);
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
  // Accumulate loss history and redraw sparkline
  if (loss != null && meta?.epoch != null) {
    const buf = _lossHistory.data;
    buf.push({ epoch: meta.epoch, loss });
    if (buf.length > _lossHistory.maxLen) buf.shift();
    _drawLossCurve();
    // Update meta line
    const metaEl = document.getElementById('build-loss-meta');
    if (metaEl) {
      const first = buf[0].loss;
      const pctDrop = first > 0 ? ((first - loss) / first * 100).toFixed(0) : '—';
      metaEl.hidden = false;
      metaEl.innerHTML = `<span>loss ${loss.toFixed(4)}</span><span>↓ ${pctDrop}%</span>`;
    }
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
    setEnabled('btn-export-bh-csv', true); setEnabled('btn-export-las', true);
      setEnabled('btn-export-ags', true); setEnabled('btn-export-geojson-tops', true);
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
  document.getElementById('show-orient-symbols')?.addEventListener('change', e => {
    if (AppState.scene) AppState.scene.toggleOrientationSymbols(e.target.checked);
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
  _initStratLockToggle();
  initStratOrderButtons();
  _initGeoEventTimeline();
  initFenceSection();
  initScreenshot();
  initBackgroundToggle();
  initMeasureTool();
  initModelReport();
  initAnnotations();
  initPlanView();
  initStratCorrelation();
  initPropertiesTab();
  initLegendRename();
  initTopoClip();
  initCursorCoords();
  initShortcutsModal();
  initGWT();
  initCameraPresets();
  initViewBookmarks();
  initVolumeByDepth();
  initGradeShell();
  initCrossPlot();
  initVariogramControls();
  initCRSTools();
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
  initSPTProfile();
  initBHDataSubTabs();
  initCPTImport();
  initWelcomeOverlay();
  initStereonet();
  initSlopeStability();
  initGeoFeatures();
  initLiquefaction();
  initMohrCircle();
  initSectionInterpreter();
  initConceptPanel();
  initProbVolPanel();

  // Sample tile buttons (left panel)
  document.querySelectorAll('.sample-tile').forEach(btn => {
    btn.addEventListener('click', () => loadDemoSite(btn.dataset.demo));
  });

  window.addEventListener('geomodel:api-key-set', e => {
    AppState.apiKey = e.detail.key;
    AppState.demoMode = !e.detail.key;
    log(e.detail.key ? '✓ API key configured' : 'Demo mode active', 'ok');
  });

  // Plan view click: show concept-based stratigraphic prediction popup
  window.addEventListener('planview:click', e => {
    const d        = e.detail;
    const store    = AppState.conceptStore;
    const geoUnits = AppState.geoUnits;
    const grid     = AppState.voxelGrid;
    if (!d || !geoUnits.length) return;
    // Show popup if we have a grid OR active concepts
    if (!grid && (!store || store.isEmpty)) return;

    // ── Extract vertical column from the built grid ──────────────────────────
    let column = null;
    if (grid) {
      const { nx, ny, nz, cellSize: cs, cellHeight: ch, origin: O,
              unitIds, certainty, conceptInfluence, coverageDensity } = grid;
      const ix = Math.round((d.worldX - O.x) / cs - 0.5);
      const iy = Math.round((d.worldY - O.z) / cs - 0.5);
      if (ix >= 0 && ix < nx && iy >= 0 && iy < ny) {
        column = [];
        for (let iz = nz - 1; iz >= 0; iz--) { // top-down
          const idx  = ix + iy * nx + iz * nx * ny;
          const uid  = unitIds[idx];
          const unit = geoUnits.find(u => u.id === uid);
          const elev = O.y + iz * ch + ch * 0.5;
          const ctx  = store ? store.computeAt(d.worldX, d.worldY, elev) : null;
          column.push({
            elev,
            unit,
            cert:    certainty?.[idx] ?? 0,
            ci:      conceptInfluence?.[idx] ?? 0,
            cov:     coverageDensity?.[idx] ?? 0,
            topConceptDesc: ctx?.weights?.[0] ? store.concepts.find(c => c.id === ctx.weights[0].id)?.description?.slice(0, 28) : null,
          });
        }
      }
    }

    // ── Concept-only predictions if no grid ──────────────────────────────────
    let conceptRows = [];
    if (!column && store && !store.isEmpty) {
      const DR = { min: d.elev - 20, max: d.elev + 5 };
      for (let i = 0; i < 6; i++) {
        const wz  = DR.min + (i / 5) * (DR.max - DR.min);
        const ctx = store.computeAt(d.worldX, d.worldY, wz);
        if (ctx.totalWeight < 0.05) continue;
        const tc   = store.concepts.find(c => c.id === ctx.weights[0]?.id);
        const affU = tc?.unitAffinity?.length ? geoUnits.find(u => tc.unitAffinity.includes(u.code)) : null;
        conceptRows.push({ z: wz, totalWeight: ctx.totalWeight, unit: affU, desc: tc?.description?.slice(0, 30) ?? '—' });
      }
    }

    // ── Render popup ─────────────────────────────────────────────────────────
    let popup = document.getElementById('plan-concept-popup');
    if (!popup) {
      popup = document.createElement('div');
      popup.id = 'plan-concept-popup';
      popup.style.cssText = `position:fixed;z-index:9999;background:var(--bg-panel);border:1px solid var(--border);
        border-radius:6px;padding:8px 10px;font-size:10px;color:var(--text);
        box-shadow:0 4px 20px rgba(0,0,0,0.45);max-width:270px;pointer-events:auto;min-width:220px`;
      document.body.appendChild(popup);
    }
    // Position near cursor but keep on-screen
    const px = Math.min(d.canvasX + 18, window.innerWidth - 290);
    const py = Math.max(Math.min(d.canvasY - 10, window.innerHeight - 350), 10);
    popup.style.left = `${px}px`;
    popup.style.top  = `${py}px`;
    popup.style.display = 'block';

    let inner = `<div style="font-weight:600;margin-bottom:4px;font-size:10.5px">
      Vertical profile &thinsp; (${d.worldX.toFixed(0)}, ${d.worldY.toFixed(0)})
    </div>`;

    if (column) {
      // Compact column log with unit color bar, certainty, CI
      inner += `<div style="font-size:9px;color:var(--text-dim);margin-bottom:4px">
        Built model · ${column.length} depth intervals ↑ top to bottom ↓</div>`;
      // Group consecutive voxels with same unit into bands
      const bands = [];
      let cur = null;
      for (const row of column) {
        const code = row.unit?.code ?? '?';
        if (!cur || cur.code !== code) {
          cur = { code, color: row.unit?.color ?? '#888', topElev: row.elev, botElev: row.elev, cert: row.cert, ci: row.ci, desc: row.topConceptDesc };
          bands.push(cur);
        } else {
          cur.botElev  = row.elev;
          cur.cert     = (cur.cert + row.cert) / 2;
          cur.ci       = (cur.ci + row.ci) / 2;
        }
      }
      inner += bands.map(b => {
        const thick = Math.abs(b.topElev - b.botElev).toFixed(1);
        const ciCol = b.ci > 0.6 ? '#e07020' : b.ci > 0.3 ? '#d4a843' : 'var(--accent)';
        return `<div style="display:flex;align-items:center;gap:4px;margin-bottom:2px">
          <div style="width:10px;height:14px;border-radius:2px;background:${b.color};flex-shrink:0"></div>
          <div style="flex:1;min-width:0">
            <div style="font-size:10px;font-weight:600">${escHtml(b.code)}</div>
            <div style="font-size:9px;color:var(--text-dim)">${b.topElev.toFixed(1)}→${b.botElev.toFixed(1)}m &nbsp; ${thick}m</div>
            ${b.desc ? `<div style="font-size:9px;color:var(--text-dim);font-style:italic;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${escHtml(b.desc)}</div>` : ''}
          </div>
          <div style="text-align:right;flex-shrink:0;font-size:9px;font-family:var(--font-mono)">
            <div style="color:var(--text-mid)">${(b.cert*100).toFixed(0)}%</div>
            <div style="color:${ciCol}">${(b.ci*100).toFixed(0)}%CI</div>
          </div>
        </div>`;
      }).join('');
      inner += `<div style="font-size:9px;color:var(--text-dim);margin-top:4px;border-top:1px solid var(--border);padding-top:4px">
        Certainty / CI (concept influence)
      </div>`;
    } else if (conceptRows.length) {
      inner += `<div style="font-size:9px;color:var(--text-dim);margin-bottom:4px">Concept prediction only — build model for full stratigraphy</div>`;
      inner += conceptRows.map(p =>
        `<div style="display:flex;align-items:center;gap:5px;margin-bottom:2px">
          ${p.unit ? `<span style="width:9px;height:9px;border-radius:2px;background:${p.unit.color};flex-shrink:0"></span>` : '<span style="width:9px;height:9px;flex-shrink:0"></span>'}
          <span style="color:var(--text-mid);min-width:38px;font-family:var(--font-mono);font-size:9px">${p.z.toFixed(1)}m</span>
          <span style="flex:1;color:var(--text-dim);font-size:9px">${escHtml(p.desc)}</span>
          <span style="color:var(--accent);font-size:9px">${(p.totalWeight*100).toFixed(0)}%</span>
        </div>`
      ).join('');
    } else {
      inner += `<div style="color:var(--text-dim);font-size:10px">No data at this location.</div>`;
    }

    inner += `<button onclick="this.closest('#plan-concept-popup').style.display='none'"
      style="margin-top:6px;font-size:9px;padding:1px 7px;background:var(--bg-el);border:1px solid var(--border);border-radius:3px;cursor:pointer;color:var(--text-mid);width:100%">✕ Close</button>`;

    popup.innerHTML = inner;
    clearTimeout(popup._autoClose);
    popup._autoClose = setTimeout(() => { if (popup) popup.style.display = 'none'; }, 12000);
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

    // Wire neural sensitivity scan button (rendered inside attribution HTML)
    const sensBtn = document.getElementById('btn-trace-sensitivity');
    if (sensBtn) {
      sensBtn.addEventListener('click', () => {
        const sensEl = document.getElementById('trace-sensitivity-content');
        if (!sensEl) return;
        sensBtn.disabled = true;
        sensBtn.textContent = '⟳ Scanning…';
        // Run in next tick to allow UI update
        setTimeout(() => {
          try {
            const result = _computeVoxelSensitivity(d.worldX, d.worldY, d.worldZ);
            if (!result) {
              sensEl.innerHTML = '<span style="color:var(--text-dim)">No neural model available.</span>';
              return;
            }
            const { sensitivity, dominantUnitCode, basePDom } = result;
            const maxAbs = Math.max(0.01, ...Array.from(sensitivity).map(Math.abs));
            const bars = Array.from(sensitivity).map((v, i) => {
              const pct = Math.round(Math.abs(v) / maxAbs * 100);
              const col = v > 0.01 ? 'var(--accent)' : v < -0.01 ? 'var(--red)' : 'var(--border)';
              const sign = v > 0.005 ? '↑' : v < -0.005 ? '↓' : '·';
              return `<div style="display:flex;align-items:center;gap:3px;margin-bottom:1px">
                <span style="width:14px;text-align:center;font-size:9px;color:${col}">${sign}</span>
                <div style="flex:1;height:5px;background:var(--bg-deep);border-radius:2px;overflow:hidden">
                  <div style="width:${pct}%;height:100%;background:${col}"></div>
                </div>
                <span style="font-size:8.5px;font-family:var(--font-mono);color:var(--text-dim);width:36px;text-align:right">${v >= 0 ? '+' : ''}${v.toFixed(2)}</span>
                <span style="font-size:8.5px;color:var(--text-dim);width:110px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis" title="${CONCEPT_AXES[i]}">${CONCEPT_AXES[i]}</span>
              </div>`;
            }).join('');
            sensEl.innerHTML = `<div style="font-size:9px;color:var(--text-mid);margin-bottom:4px">
              ∂P(${dominantUnitCode ?? '?'}) / ∂axis (base P=${(basePDom * 100).toFixed(0)}%) — how each axis shifts the dominant unit probability
            </div>${bars}`;
          } catch (err) {
            if (sensEl) sensEl.innerHTML = `<span style="color:var(--red);font-size:9px">Sensitivity scan failed: ${err.message}</span>`;
          }
        }, 20);
      });
    }
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

    // Show 3D dip symbols — sample ~50 representative positions per unit
    _showDipSymbols3D(allOrientations, filterCode);

    log(`Stereonet: ${displayOrientations.length} orientation measurements computed`, 'ok');
  });
}

function _showDipSymbols3D(allOrientations, filterCode) {
  if (!AppState.scene || !AppState.voxelGrid) return;
  const { nx, ny, nz, cellSize: cs, cellHeight: ch, origin: O, unitIds } = AppState.voxelGrid;

  const symbols = [];
  const targetCodes = filterCode ? [filterCode] : Object.keys(allOrientations);
  const MAX_PER_UNIT = 30;

  for (const code of targetCodes) {
    const orients = allOrientations[code];
    if (!orients?.length) continue;
    const unit = AppState.geoUnits.find(u => u.code === code);
    const color = unit?.color ?? '#ff8800';

    // Find surface cells for this unit and sample them
    const cells = [];
    for (let iy = 1; iy < ny - 1; iy++) {
      for (let ix = 1; ix < nx - 1; ix++) {
        for (let iz = nz - 1; iz >= 0; iz--) {
          if (unitIds[ix + iy * nx + iz * nx * ny] === unit?.id) {
            cells.push({ ix, iy, iz }); break;
          }
        }
      }
    }

    const step = Math.max(1, Math.ceil(cells.length / MAX_PER_UNIT));
    let oi = 0;
    for (let ci = 0; ci < cells.length; ci += step) {
      if (oi >= orients.length) oi = 0;
      const { ix, iy, iz } = cells[ci];
      const x    = O.x + (ix + 0.5) * cs;
      const y    = O.z + (iy + 0.5) * cs;
      const elev = O.y + (iz + 0.5) * ch;
      const { dip, dipDir } = orients[oi++];
      symbols.push({ x, y, elev, dip, dipDir, color });
    }
  }

  AppState.scene.showOrientationSymbols(symbols);
  if (symbols.length) log(`Showing ${symbols.length} dip symbols in 3D view.`, 'info');
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

      // Assign a spatial domain matching the fence footprint so section-derived
      // concepts influence only the region around the section, not the whole site.
      const sectionLen = fence
        ? Math.hypot(fence.endX - fence.startX, fence.endY - fence.startY)
        : 0;
      const sectionBuf = Math.max(20, sectionLen * 0.25);
      const conceptDomain = fence && sectionLen > 0
        ? {
            type: 'bbox',
            minX: Math.min(fence.startX, fence.endX) - sectionBuf,
            maxX: Math.max(fence.startX, fence.endX) + sectionBuf,
            minY: Math.min(fence.startY, fence.endY) - sectionBuf,
            maxY: Math.max(fence.startY, fence.endY) + sectionBuf,
            sigma: Math.max(30, sectionLen * 0.4),
          }
        : { type: 'global' };

      // Normalise statements: support both old string[] and new {statement, unit_codes, confidence}[]
      const normStatements = statements.map(s =>
        typeof s === 'string'
          ? { statement: s, unit_codes: [], confidence: parsed.confidence ?? 0.72 }
          : { statement: s.statement ?? s, unit_codes: s.unit_codes ?? [], confidence: s.confidence ?? parsed.confidence ?? 0.72 }
      ).filter(s => s.statement?.trim());

      const conceptIds = [];
      for (const ns of normStatements) {
        try {
          const emb = await encodeGeologicalConcept(ns.statement, AppState.apiKey, AppState.demoMode);
          const id  = AppState.conceptStore.add({
            description:  ns.statement,
            embedding:    emb,
            confidence:   ns.confidence,
            domain:       conceptDomain,
            unitAffinity: ns.unit_codes,
          });
          conceptIds.push(id);
        } catch (e) { log(`Concept encode warning: ${e.message}`, 'warn'); }
      }
      _renderConceptList();

      const kws = normStatements.map(s => s.statement).join(', ') || '—';
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
  // ── Fluvial / Alluvial ──────────────────────────────────────────────────────
  { label: 'Palaeochannel E-W',    axes: { east_west_elongation: 0.9, north_south_elongation: -0.5, channel_morphology: 1.0, erosional_contact: 0.9, gravel_basal_lag: 0.8, incision_depth_ratio: 0.8, lateral_anisotropy: 0.9, horizontal_layering: -0.7 } },
  { label: 'Palaeochannel N-S',    axes: { north_south_elongation: 0.9, east_west_elongation: -0.5, channel_morphology: 1.0, erosional_contact: 0.9, gravel_basal_lag: 0.8, incision_depth_ratio: 0.8, lateral_anisotropy: 0.9, horizontal_layering: -0.7 } },
  { label: 'Palaeochannel NE-SW',  axes: { east_west_elongation: 0.65, north_south_elongation: 0.65, channel_morphology: 1.0, erosional_contact: 0.9, gravel_basal_lag: 0.8, incision_depth_ratio: 0.8, lateral_anisotropy: 0.9, horizontal_layering: -0.7 } },
  { label: 'Palaeochannel NW-SE',  axes: { east_west_elongation: 0.65, north_south_elongation: 0.65, channel_morphology: 1.0, erosional_contact: 0.8, gravel_basal_lag: 0.7, incision_depth_ratio: 0.7, lateral_anisotropy: 0.8, horizontal_layering: -0.5 } },
  { label: 'Nested Channels',      axes: { channel_morphology: 0.9, nested_channels: 1.0, erosional_contact: 0.9, gravel_basal_lag: 0.7, lateral_continuity: 0.3 } },
  { label: 'River Terrace',      axes: { horizontal_layering: 0.7, lateral_continuity: 0.8, gravel_basal_lag: 0.7, fining_upward: 0.4, erosional_contact: 0.6, lateral_anisotropy: 0.6 } },
  { label: 'Alluvial Fan',       axes: { lateral_thinning_east: 0.4, lateral_thinning_west: 0.4, lateral_thinning_north: 0.4, coarsening_upward: -0.5, gravel_basal_lag: 0.5 } },
  { label: 'Floodplain',         axes: { horizontal_layering: 0.8, lateral_continuity: 0.8, fining_upward: 0.5, lateral_anisotropy: 0.5 } },
  // ── Glacial ─────────────────────────────────────────────────────────────────
  { label: 'Esker (E-W)',        axes: { east_west_elongation: 0.7, channel_morphology: 0.5, lateral_anisotropy: 0.8, gravel_basal_lag: 0.6, dome_anticline: 0.4 } },
  { label: 'Drumlin',            axes: { dome_anticline: 0.6, lateral_anisotropy: 0.7, lateral_continuity: 0.6, horizontal_layering: 0.3 } },
  { label: 'Glacial Till',       axes: { horizontal_layering: -0.5, lateral_continuity: 0.5, structural_complexity: 0.4, overburden_control: 0.5 } },
  { label: 'Ice-contact / Kame', axes: { irregular_base: 0.7, lateral_continuity: -0.3, structural_complexity: 0.6, gravel_basal_lag: 0.4 } },
  // ── Structural ──────────────────────────────────────────────────────────────
  { label: 'Fault E-W (stepped)', axes: { fault_controlled: 1.0, stepped_boundary: 0.9, structural_complexity: 0.7, deepens_north: 0.4 } },
  { label: 'Fault N-S (stepped)', axes: { fault_controlled: 1.0, stepped_boundary: 0.9, structural_complexity: 0.7, deepens_east: 0.5 } },
  { label: 'Deepening North',    axes: { deepens_north: 0.85, inclined_bedding: 0.6, dip_magnitude: 0.5 } },
  { label: 'Deepening South',    axes: { deepens_south: 0.85, inclined_bedding: 0.6, dip_magnitude: 0.5 } },
  { label: 'Deepening East',     axes: { deepens_east: 0.85, inclined_bedding: 0.6, dip_magnitude: 0.5 } },
  { label: 'Deepening West',     axes: { deepens_west: 0.85, inclined_bedding: 0.6, dip_magnitude: 0.5 } },
  { label: 'Deepening NE',       axes: { deepens_north: 0.6, deepens_east: 0.6, inclined_bedding: 0.5, dip_magnitude: 0.5 } },
  { label: 'Deepening SW',       axes: { deepens_south: 0.6, deepens_west: 0.6, inclined_bedding: 0.5, dip_magnitude: 0.5 } },
  { label: 'Dome / Anticline',   axes: { dome_anticline: 0.9, lateral_continuity: 0.6, horizontal_layering: -0.3 } },
  // ── Dissolution / Karst ─────────────────────────────────────────────────────
  { label: 'Karst / Dissolution', axes: { dissolution_features: 1.0, irregular_base: 0.9, structural_complexity: 0.6, lateral_continuity: -0.4 } },
  { label: 'Rockhead surface',   axes: { erosional_contact: 0.7, irregular_base: 0.5, stepped_boundary: 0.3, data_confidence: 0.6 } },
  // ── Simple geometry ─────────────────────────────────────────────────────────
  { label: 'Bedded / Tabular',   axes: { horizontal_layering: 0.9, lateral_continuity: 0.9, vertical_anisotropy: 0.7 } },
  { label: 'Massive (no fabric)', axes: { horizontal_layering: -0.7, inclined_bedding: -0.4, vertical_anisotropy: -0.5, structural_complexity: 0.3 } },
  { label: 'Lacustrine clay',    axes: { horizontal_layering: 0.9, lateral_continuity: 0.9, fining_upward: 0.3, overburden_control: 0.3, lateral_anisotropy: 0.3 } },
  { label: 'Sand Lens / Pod',    axes: { channel_morphology: 0.5, lateral_thinning_north: 0.6, lateral_thinning_south: 0.6, lateral_continuity: -0.5 } },
  // ── Coastal / Estuarine ─────────────────────────────────────────────────────
  { label: 'Tidal flat / mudflat',axes: { horizontal_layering: 0.9, lateral_continuity: 0.8, coarsening_upward: -0.4, fining_upward: 0.3, overburden_control: 0.4 } },
  { label: 'Estuarine channel',  axes: { east_west_elongation: 0.6, channel_morphology: 0.8, erosional_contact: 0.7, gravel_basal_lag: 0.5, lateral_continuity: 0.4, horizontal_layering: -0.4 } },
  { label: 'Beach / Shoreface',  axes: { east_west_elongation: 0.8, horizontal_layering: 0.5, lateral_continuity: 0.6, coarsening_upward: 0.3, gravel_basal_lag: 0.4 } },
  // ── Urban / Made Ground ──────────────────────────────────────────────────────
  { label: 'Made ground',        axes: { horizontal_layering: -0.5, lateral_continuity: -0.3, structural_complexity: 0.7, irregular_base: 0.6, overburden_control: 0.8 } },
  { label: 'Mining spoil',       axes: { structural_complexity: 0.9, irregular_base: 0.7, lateral_continuity: -0.5, horizontal_layering: -0.7, overburden_control: 0.6 } },
  // ── Chalk / Weak Rock ────────────────────────────────────────────────────────
  { label: 'Chalk rockhead',     axes: { dissolution_features: 0.5, irregular_base: 0.7, stepped_boundary: 0.4, erosional_contact: 0.5, data_confidence: 0.7 } },
  { label: 'Weathered chalk',    axes: { structural_complexity: 0.5, lateral_continuity: 0.5, vertical_anisotropy: 0.7, overburden_control: 0.5, data_confidence: 0.6 } },
  // ── Soft Soils ──────────────────────────────────────────────────────────────
  { label: 'Peat / Organics',    axes: { horizontal_layering: 0.7, lateral_continuity: 0.5, irregular_base: 0.5, fining_upward: 0.2, overburden_control: 0.3 } },
  { label: 'Soft alluvial clay', axes: { horizontal_layering: 0.8, lateral_continuity: 0.7, fining_upward: 0.5, overburden_control: 0.4, data_confidence: 0.6 } },
  // ── Intrusive / Igneous ──────────────────────────────────────────────────────
  { label: 'Igneous intrusion',  axes: { dome_anticline: 0.7, structural_complexity: 0.8, lateral_continuity: -0.4, inclined_bedding: 0.5, dip_magnitude: 0.7 } },
  { label: 'Dyke / Sill',       axes: { fault_controlled: 0.6, stepped_boundary: 0.5, structural_complexity: 0.8, lateral_continuity: -0.6, dip_magnitude: 0.8 } },
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
  const amaj = t.Amaj ?? Math.max(t.Ax, t.Ay);
  const amin = t.Amin ?? Math.min(t.Ax, t.Ay);
  const tDeg = ((t.theta ?? 0) * 180 / Math.PI).toFixed(0);
  // Compass direction: 0°=E-W, 45°=NE-SW, 90°=N-S, 135°=NW-SE
  const dirs = ['E-W', 'NE-SW', 'N-S', 'NW-SE'];
  const dirIdx = Math.round((parseFloat(tDeg) % 180 + 180) / 45) % 4;
  const hasAniso = amaj > 1.1;
  el.textContent = hasAniso
    ? `Warp: ${dirs[dirIdx]} major ×${amaj.toFixed(1)} · minor ×${amin.toFixed(2)} · Z ×${t.Az.toFixed(2)}`
    : `Warp: isotropic (no strong horizontal elongation) · Z ×${t.Az.toFixed(2)}`;
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

window._exportConceptGeoJSON = function() {
  if (!AppState.conceptStore || AppState.conceptStore.isEmpty) {
    log('No concepts to export', 'warn'); return;
  }
  const concepts = AppState.conceptStore.concepts;

  // Determine site bounding box for global concepts (use borehole extents or 100×100m fallback)
  const bhs = AppState.classifiedBH ?? [];
  const xs  = bhs.map(b => b.x), ys = bhs.map(b => b.y);
  const siteMinX = xs.length ? Math.min(...xs) - 20 : 0;
  const siteMaxX = xs.length ? Math.max(...xs) + 20 : 100;
  const siteMinY = ys.length ? Math.min(...ys) - 20 : 0;
  const siteMaxY = ys.length ? Math.max(...ys) + 20 : 100;

  const features = concepts.map(c => {
    // Build geometry: polygon for bbox domain, point for global/radius
    let geometry;
    if (c.domain?.type === 'bbox') {
      const { minX = siteMinX, maxX = siteMaxX, minY = siteMinY, maxY = siteMaxY } = c.domain;
      geometry = {
        type: 'Polygon',
        coordinates: [[[minX, minY], [maxX, minY], [maxX, maxY], [minX, maxY], [minX, minY]]],
      };
    } else if (c.domain?.type === 'radius') {
      const { centreX = (siteMinX + siteMaxX) / 2, centreY = (siteMinY + siteMaxY) / 2, radius = 50 } = c.domain;
      // Approximate circle as 24-point polygon
      const pts = Array.from({ length: 24 }, (_, i) => {
        const a = (i / 24) * 2 * Math.PI;
        return [centreX + Math.cos(a) * radius, centreY + Math.sin(a) * radius];
      });
      pts.push(pts[0]);
      geometry = { type: 'Polygon', coordinates: [pts] };
    } else {
      // Global concept: use site bbox
      geometry = {
        type: 'Polygon',
        coordinates: [[[siteMinX, siteMinY], [siteMaxX, siteMinY], [siteMaxX, siteMaxY], [siteMinX, siteMaxY], [siteMinX, siteMinY]]],
      };
    }

    // Build properties: all 32 axes + meta
    const axisProps = {};
    CONCEPT_AXES.forEach((name, i) => { axisProps[`ax_${i}_${name}`] = +c.embedding[i].toFixed(3); });
    return {
      type: 'Feature',
      geometry,
      properties: {
        id: c.id,
        description: c.description,
        confidence: c.confidence,
        domain_type: c.domain?.type ?? 'global',
        temporal_order: c.temporalOrder ?? null,
        unit_affinity: (c.unitAffinity ?? []).join(', '),
        depth_min_m: c.domain?.minZ ?? null,
        depth_max_m: c.domain?.maxZ ?? null,
        ...axisProps,
      },
    };
  });

  const geojson = { type: 'FeatureCollection', features, _source: 'GeoModel AI Concepts' };
  const blob = new Blob([JSON.stringify(geojson, null, 2)], { type: 'application/geo+json' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href = url; a.download = 'geomodel-concepts.geojson'; a.click();
  URL.revokeObjectURL(url);
  log(`Exported ${concepts.length} concept(s) as GeoJSON`, 'ok');
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

  // Concept influence overlays — enabled after model build
  document.getElementById('btn-show-concept-influence')?.addEventListener('click', () => {
    if (!AppState.scene) return;
    const ok = AppState.scene.colorByConceptInfluence();
    if (!ok) log('Concept influence data not available — build with Neural Implicit method first.', 'warn');
    else log('3D view: coloured by concept semantic influence (blue=data-driven, red=concept-driven)', 'ok');
  });
  document.getElementById('btn-show-dominant-concept')?.addEventListener('click', () => {
    if (!AppState.scene || !AppState.conceptStore) return;
    const ok = AppState.scene.colorByDominantConcept(AppState.conceptStore);
    if (!ok) log('Dominant concept data not available — build with Neural Implicit method first.', 'warn');
    else log('3D view: coloured by dominant geological concept', 'ok');
  });
  document.getElementById('btn-show-concept-effect')?.addEventListener('click', async () => {
    const btn = document.getElementById('btn-show-concept-effect');
    if (!AppState.trainedModel || !AppState.voxelGrid || !AppState.scene) {
      log('Build neural-implicit model first.', 'warn'); return;
    }
    if (!AppState.conceptStore || AppState.conceptStore.isEmpty) {
      log('No active concepts — add concepts first to see their effect.', 'warn'); return;
    }
    btn.disabled = true; btn.textContent = '⊕ Computing…';
    await new Promise(r => setTimeout(r, 0));
    try {
      const grid = AppState.voxelGrid;
      const gridMeta = {
        nx: grid.nx, ny: grid.ny, nz: grid.nz,
        cellSize: grid.cellSize, cellHeight: grid.cellHeight, origin: grid.origin,
      };
      // Re-infer with no concepts to get the pure-data baseline
      const { ConceptStore: CS } = await import('./concept-store.js');
      const emptyStore = new CS();
      const baseline = inferGeoImplicit(AppState.trainedModel, gridMeta, AppState.geoUnits, emptyStore);

      // Build effect map: 1 where concept changed prediction, 0 where same
      const current = grid.unitIds;
      const total   = current.length;
      const effectMap = new Float32Array(total);
      let changedCount = 0;
      for (let i = 0; i < total; i++) {
        if (current[i] !== baseline.unitIds[i]) { effectMap[i] = 1; changedCount++; }
      }
      const pct = (changedCount / total * 100).toFixed(1);
      AppState.scene.colorByParameter(null, AppState.geoUnits, effectMap);
      log(`Concept effect map: ${changedCount.toLocaleString()} / ${total.toLocaleString()} voxels (${pct}%) changed by semantic embedding. Yellow=concept-changed, dark=unchanged.`, 'ok');
    } catch (e) {
      log(`Concept effect map failed: ${e.message}`, 'error');
    } finally {
      btn.disabled = false; btn.textContent = '⊕ Concept effect map';
    }
  });

  document.getElementById('btn-concept-interp-summary')?.addEventListener('click', () => {
    const outEl = document.getElementById('concept-interp-summary-output');
    if (!outEl) return;
    const store = AppState.conceptStore;
    if (!store || store.isEmpty) {
      outEl.style.display = 'block';
      outEl.innerHTML = '<em style="color:var(--text-dim)">No active concepts. Add concepts first.</em>';
      return;
    }
    const concepts = store.concepts;

    // Aggregate embedding weighted by confidence
    const aggEmb = new Float32Array(32);
    let totalW = 0;
    for (const c of concepts) {
      for (let i = 0; i < 32; i++) aggEmb[i] += c.embedding[i] * c.confidence;
      totalW += c.confidence;
    }
    if (totalW > 0) for (let i = 0; i < 32; i++) aggEmb[i] /= totalW;

    // Translate top axes to geological statements
    const statements = [];
    if (aggEmb[3] > 0.4) statements.push(`E-W elongated bodies (×${Math.exp(aggEmb[3] * 1.4).toFixed(1)} stretch)`);
    if (aggEmb[4] > 0.4) statements.push(`N-S elongated bodies (×${Math.exp(aggEmb[4] * 1.4).toFixed(1)} stretch)`);
    if (aggEmb[5] > 0.4) statements.push(`Channel morphology: concave-up contacts, narrow bodies`);
    if (aggEmb[0] > 0.4) statements.push(`Horizontally layered formations with good lateral continuity`);
    if (aggEmb[8] > 0.4) statements.push(`Erosional contacts expected at unit bases`);
    if (aggEmb[18] > 0.4) statements.push(`Stepped or fault-controlled boundaries — sharp contacts`);
    if (aggEmb[7] > 0.4) statements.push(`Fault-controlled geometry — lateral discontinuities`);
    if (aggEmb[23] > 0.4) statements.push(`Basal gravel lag expected below key contacts`);
    if (aggEmb[22] > 0.3) statements.push(`Fining-upward grading within units`);
    if (aggEmb[29] > 0.4) statements.push(`Deep incision — significant relief on formation base`);
    if (aggEmb[9] > 0.5) statements.push(`High lateral continuity — units likely continuous across site`);
    if (aggEmb[25] > 0.5) statements.push(`High structural complexity — irregular or variable contacts`);
    if (aggEmb[19] > 0.4) statements.push(`Irregular formation base — variable rockhead or contact`);
    if (aggEmb[6] > 0.4) statements.push(`Doming or anticlinal structure`);
    if (aggEmb[24] > 0.3) statements.push(`Possible dissolution / karst features`);

    // Depth trend
    const deepE = aggEmb[14] - aggEmb[15], deepN = aggEmb[16] - aggEmb[17];
    if (Math.abs(deepE) > 0.3) statements.push(`Formation deepens ${deepE > 0 ? 'to the east' : 'to the west'}`);
    if (Math.abs(deepN) > 0.3) statements.push(`Formation deepens ${deepN > 0 ? 'to the north' : 'to the south'}`);

    // Units involved
    const allAffinity = [...new Set(concepts.flatMap(c => c.unitAffinity ?? []))];
    const unitsSentence = allAffinity.length > 0
      ? `This model targets unit${allAffinity.length > 1 ? 's' : ''} <b>${allAffinity.map(escHtml).join(', ')}</b>.`
      : 'Concepts apply to all geological units globally.';

    // Temporal summary
    const withTemporal = concepts.filter(c => c.temporalOrder != null).sort((a, b) => a.temporalOrder - b.temporalOrder);
    const chronologySentence = withTemporal.length >= 2
      ? `Temporal order: ${withTemporal.map(c => `<b>${escHtml(c.description.slice(0, 30))}</b>`).join(' → ')} (oldest → youngest).`
      : '';

    const summaryLines = statements.length > 0
      ? statements.map(s => `<li>${escHtml(s)}</li>`).join('')
      : '<li style="color:var(--text-dim)">No dominant geometric patterns detected</li>';

    outEl.style.display = 'block';
    outEl.innerHTML = `
      <div style="font-weight:600;color:var(--accent);margin-bottom:4px">Conceptual model predicts:</div>
      <ul style="margin:0 0 6px 14px;padding:0;list-style:disc">${summaryLines}</ul>
      <div style="color:var(--text-mid);margin-bottom:3px">${unitsSentence}</div>
      ${chronologySentence ? `<div style="color:var(--text-mid)">${chronologySentence}</div>` : ''}
      <div style="color:var(--text-dim);margin-top:5px;font-style:italic">${concepts.length} active concept${concepts.length !== 1 ? 's' : ''} · avg confidence ${(totalW / concepts.length * 100).toFixed(0)}%</div>`;
  });

  document.addEventListener('geomodel:model-built', () => {
    setEnabled('btn-show-concept-influence', !!(AppState.voxelGrid?.conceptInfluence));
    setEnabled('btn-show-dominant-concept', !!(AppState.voxelGrid?.conceptInfluence));
    setEnabled('btn-show-concept-effect', !!(AppState.trainedModel && AppState.voxelGrid?.conceptInfluence));
    // Auto-run sensitivity and warn about low-influence concepts
    _warnLowInfluenceConcepts();
  });

  // Render concept library chips and scenario list
  _initConceptLibrary();

  // Wire concept correlation matrix collapsible
  const corrToggle = document.getElementById('concept-corr-toggle');
  const corrBody   = document.getElementById('concept-corr-body');
  if (corrToggle && corrBody) {
    corrToggle.addEventListener('click', () => {
      const hidden = corrBody.hasAttribute('hidden');
      if (hidden) { corrBody.removeAttribute('hidden'); _drawConceptCorrelationMatrix(); }
      else corrBody.setAttribute('hidden', '');
      const arrow = corrToggle.querySelector('.collapse-arrow');
      if (arrow) arrow.textContent = hidden ? '⌄' : '›';
    });
  }

  // Wire concept manifold (2D PCA) collapsible
  const manifoldToggle = document.getElementById('concept-manifold-toggle');
  const manifoldBody   = document.getElementById('concept-manifold-body');
  if (manifoldToggle && manifoldBody) {
    manifoldToggle.addEventListener('click', () => {
      const hidden = manifoldBody.hasAttribute('hidden');
      if (hidden) { manifoldBody.removeAttribute('hidden'); _drawConceptManifold(); }
      else manifoldBody.setAttribute('hidden', '');
      const arrow = manifoldToggle.querySelector('.collapse-arrow');
      if (arrow) arrow.textContent = hidden ? '⌄' : '›';
    });
  }

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

  // Real-time preview: run demo encoding as user types, so they see which axes light up
  const previewWrap = document.getElementById('concept-preview-wrap');
  const previewAxes = document.getElementById('concept-preview-axes');
  let _previewTimer = null;
  textarea?.addEventListener('input', () => {
    clearTimeout(_previewTimer);
    const text = textarea.value.trim();
    if (!text || text.length < 5) {
      if (previewWrap) previewWrap.style.display = 'none';
      return;
    }
    _previewTimer = setTimeout(async () => {
      // Use demo encoding (instant) for live preview — API encoding happens on submit
      const emb = await encodeGeologicalConcept(text, null, true);
      if (!previewAxes || !previewWrap) return;
      previewAxes.innerHTML = Array.from(emb).map((v, i) => {
        const pct = Math.round(Math.abs(v) * 100);
        const col = v >= 0 ? 'var(--accent)' : 'var(--red)';
        return `<div class="concept-bar-wrap" title="${CONCEPT_AXES[i]}: ${v.toFixed(2)}">
          <div class="concept-bar" style="width:${pct}%;background:${col}"></div>
        </div>`;
      }).join('');
      previewWrap.style.display = 'block';
    }, 250);  // debounce 250ms
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

  // Auto-suggest concepts from borehole patterns
  document.getElementById('btn-suggest-concepts')?.addEventListener('click', async () => {
    const btn = document.getElementById('btn-suggest-concepts');
    const out = document.getElementById('concept-suggestions');
    if (!AppState.classifiedBH.length) { log('Load borehole data first.', 'warn'); return; }
    if (btn) { btn.disabled = true; btn.textContent = '⟳ Analysing patterns…'; }
    try {
      const apiKey = sessionStorage.getItem('anthropic_api_key') ?? '';

      // Run geometric analysis immediately (no API) + Claude suggestions in parallel.
      // geomSuggestions have pre-computed embeddings; Claude suggestions need encode step.
      const [geomSuggestions, aiSuggestions] = await Promise.all([
        Promise.resolve(analyzeBoreholeGeometry(AppState.classifiedBH.filter(b => !b.synthetic), AppState.geoUnits)),
        suggestConceptsFromBoreholes(AppState.classifiedBH, AppState.geoUnits, apiKey, !apiKey),
      ]);

      // Merge: geometry-derived first (instant add), then Claude-suggested (need encode)
      const suggestions = [
        ...geomSuggestions.map(s => ({ ...s, _preEncoded: true })),
        ...aiSuggestions.filter(s => !geomSuggestions.some(g => g.unitCode === s.unit_codes?.[0])),
      ];
      if (!suggestions.length) { log('No concept suggestions generated.', 'info'); return; }
      if (out) {
        out.style.display = 'block';
        out.innerHTML = suggestions.map((s, i) => `
          <div class="suggestion-row" style="border:1px solid var(--border);border-radius:4px;padding:5px 6px;margin-bottom:4px;font-size:10px">
            <div style="font-weight:600;color:var(--text-primary);margin-bottom:2px">${escHtml(s.description)}</div>
            <div style="color:var(--text-dim);margin-bottom:4px;font-size:9px">${escHtml(s.reason ?? '')}${s._preEncoded ? ' <span style="color:var(--accent);opacity:.7">[geometric analysis]</span>' : ''}</div>
            <button class="btn-ghost btn-sm" style="font-size:9px;padding:1px 6px" data-sug-idx="${i}">+ Add${s._preEncoded ? ' (instant)' : ''}</button>
          </div>
        `).join('');
        out.querySelectorAll('[data-sug-idx]').forEach(btn => {
          btn.addEventListener('click', async () => {
            const s = suggestions[parseInt(btn.dataset.sugIdx)];
            if (!s) return;
            btn.disabled = true; btn.textContent = '⟳';
            const apiKey2 = sessionStorage.getItem('anthropic_api_key') ?? '';
            try {
              // Pre-encoded geometric suggestions use stored embedding directly;
              // Claude-suggested descriptions still need the encode step.
              const emb = s._preEncoded && s.embedding
                ? s.embedding
                : await encodeGeologicalConcept(s.description, apiKey2, !apiKey2);
              if (!AppState.conceptStore) AppState.conceptStore = new ConceptStore();
              AppState.conceptStore.add({
                description:  s.description,
                embedding:    emb,
                confidence:   s.confidence ?? 0.7,
                domain:       { type: 'global' },
                unitAffinity: s.unit_codes ?? s.unitCode ? [s.unitCode] : [],
              });
              _renderConceptList();
              _saveConceptStore();
              log(`Concept added from suggestion: "${s.description.slice(0, 60)}"`, 'ok');
              btn.textContent = '✓ Added';
            } catch (err) { log(`Encode failed: ${err.message}`, 'error'); btn.disabled = false; btn.textContent = '+ Add'; }
          });
        });
        log(`${suggestions.length} concept suggestion(s) generated from borehole patterns`, 'ok');
      }
    } catch (err) {
      log(`Concept suggestion failed: ${err.message}`, 'error');
    } finally {
      if (btn) { btn.disabled = false; btn.textContent = '✦ Auto-suggest concepts from data'; }
    }
  });

  // ── One-shot site concept setup ───────────────────────────────────────────
  document.getElementById('btn-site-setup')?.addEventListener('click', async () => {
    const btn    = document.getElementById('btn-site-setup');
    const txtEl  = document.getElementById('site-setup-text');
    const resEl  = document.getElementById('site-setup-result');
    const txt    = txtEl?.value?.trim();
    if (!txt) { log('Enter a site description first.', 'warn'); return; }
    if (!AppState.geoUnits.length) { log('Run AI Analysis first to define geological units.', 'warn'); return; }

    btn.disabled = true; btn.textContent = '⟳ Setting up…';
    if (resEl) { resEl.style.display = 'none'; resEl.innerHTML = ''; }

    try {
      const apiKey = sessionStorage.getItem('anthropic_api_key') ?? '';
      const setup  = await setupConceptsFromSiteDescription(txt, AppState.geoUnits, apiKey, !apiKey);

      if (!AppState.conceptStore) AppState.conceptStore = new ConceptStore();
      let nConcepts = 0;
      for (const c of setup.concepts) {
        try {
          const emb = await encodeGeologicalConcept(c.description, apiKey, !apiKey);
          const embArr = emb instanceof Float32Array ? emb : emb.embedding;
          AppState.conceptStore.add({
            description:  c.description,
            embedding:    embArr,
            confidence:   c.confidence,
            domain:       { type: 'global' },
            unitAffinity: c.unitAffinity ?? [],
          });
          nConcepts++;
        } catch (_) { /* skip failed encodes */ }
      }

      // Apply geological event timeline
      for (const evt of setup.events) {
        const id = `evt_${Date.now()}_${Math.random().toString(36).slice(2,6)}`;
        AppState.geoEvents.push({ id, type: evt.type, name: evt.name, unitCodes: evt.unitCodes });
      }

      // Apply strat order if provided
      if (setup.stratOrder.length) {
        AppState.stratOrder = [...setup.stratOrder];
        AppState._stratDisplayOrder = [...setup.stratOrder];
        const lockEl = document.getElementById('strat-manual-lock');
        if (lockEl) lockEl.checked = true;
        updateStratColumn?.();
      }

      _renderConceptList();
      _saveConceptStore();
      _renderGeoEventList?.();

      const summary = [
        nConcepts ? `${nConcepts} concept(s) added` : null,
        setup.events.length ? `${setup.events.length} event(s) added to timeline` : null,
        setup.stratOrder.length ? `Strat order set (${setup.stratOrder.join(' → ')})` : null,
      ].filter(Boolean).join(' · ');

      if (resEl) {
        resEl.style.display = 'block';
        resEl.innerHTML = `<span style="color:var(--accent)">✓ ${escHtml(summary)}</span>`;
      }
      log(`Site setup complete: ${summary}`, 'ok');
    } catch (err) {
      log(`Site setup failed: ${err.message}`, 'error');
    } finally {
      btn.disabled = false; btn.textContent = '✦ Setup concepts & events from description';
    }
  });

  // ── Collapsible: extract concepts from pasted text ────────────────────────
  (() => {
    const toggle  = document.getElementById('concept-extract-toggle');
    const body    = document.getElementById('concept-extract-body');
    const extBtn  = document.getElementById('btn-extract-concepts');
    const extArea = document.getElementById('concept-extract-text');
    const extOut  = document.getElementById('concept-extract-results');

    toggle?.addEventListener('click', () => {
      const hidden = body?.hasAttribute('hidden');
      if (hidden) body?.removeAttribute('hidden'); else body?.setAttribute('hidden', '');
      const arrow = toggle.querySelector('.collapse-arrow');
      if (arrow) arrow.textContent = hidden ? '⌄' : '›';
    });

    extBtn?.addEventListener('click', async () => {
      const txt = extArea?.value?.trim();
      if (!txt) { log('Paste some geological text first.', 'warn'); return; }
      extBtn.disabled   = true;
      extBtn.textContent = '⟳ Extracting…';
      if (extOut) { extOut.style.display = 'none'; extOut.innerHTML = ''; }

      try {
        const concepts = await extractConceptsFromText(
          txt, AppState.geoUnits,
          sessionStorage.getItem('anthropic_api_key') ?? '',
          AppState.demoMode,
        );
        if (!concepts.length) { log('No geological concepts found in that text.', 'info'); return; }

        if (extOut) {
          extOut.style.display = 'block';
          extOut.innerHTML = concepts.map((c, i) => `
            <div style="border:1px solid var(--border);border-radius:4px;padding:5px 6px;margin-bottom:4px;font-size:10px">
              <div style="font-weight:600;color:var(--text-primary);margin-bottom:2px">${escHtml(c.description)}</div>
              <div style="display:flex;align-items:center;gap:6px;margin-bottom:4px">
                <span style="font-size:9px;color:var(--text-dim)">Confidence: ${(c.confidence * 100).toFixed(0)}%</span>
                ${c.unitAffinity?.length ? `<span style="font-size:9px;color:var(--accent)">${c.unitAffinity.join(', ')}</span>` : ''}
              </div>
              <button class="btn-ghost btn-sm" style="font-size:9px;padding:1px 6px" data-ext-idx="${i}">+ Add to concepts</button>
            </div>
          `).join('');

          extOut.querySelectorAll('[data-ext-idx]').forEach(btn => {
            btn.addEventListener('click', async () => {
              const c = concepts[parseInt(btn.dataset.extIdx)];
              if (!c) return;
              btn.disabled = true; btn.textContent = '⟳ Encoding…';
              const apiKey2 = sessionStorage.getItem('anthropic_api_key') ?? '';
              try {
                const encodeResult = await encodeGeologicalConcept(
                  c.description, apiKey2, AppState.demoMode,
                  { siteContext: { units: AppState.geoUnits.map(u => ({ code: u.code, name: u.name })) },
                    withRationale: false });
                const emb = encodeResult instanceof Float32Array ? encodeResult : encodeResult.embedding;
                if (!AppState.conceptStore) AppState.conceptStore = new ConceptStore();
                AppState.conceptStore.add({
                  description:  c.description,
                  embedding:    emb,
                  confidence:   c.confidence,
                  domain:       { type: 'global' },
                  unitAffinity: c.unitAffinity ?? [],
                });
                _renderConceptList();
                _saveConceptStore();
                log(`Concept added: "${c.description.slice(0, 60)}"`, 'ok');
                btn.textContent = '✓ Added';
              } catch (err) {
                log(`Encode failed: ${err.message}`, 'error');
                btn.disabled = false; btn.textContent = '+ Add to concepts';
              }
            });
          });
          log(`${concepts.length} concept(s) extracted from text`, 'ok');
        }
      } catch (err) {
        log(`Text concept extraction failed: ${err.message}`, 'error');
      } finally {
        extBtn.disabled = false;
        extBtn.textContent = '✦ Extract concepts from text';
      }
    });
  })();

  // ── Geological Laws Compiler ─────────────────────────────────────────────────
  (() => {
    const toggle    = document.getElementById('geo-rules-toggle');
    const body      = document.getElementById('geo-rules-body');
    const compileBtn = document.getElementById('btn-compile-geo-rules');
    const rulesArea  = document.getElementById('geo-rules-input');
    const rulesOut   = document.getElementById('geo-rules-output');

    toggle?.addEventListener('click', () => {
      const hidden = body?.hasAttribute('hidden');
      if (hidden) body?.removeAttribute('hidden'); else body?.setAttribute('hidden', '');
      const arrow = toggle.querySelector('.collapse-arrow');
      if (arrow) arrow.textContent = hidden ? '⌄' : '›';
    });

    compileBtn?.addEventListener('click', async () => {
      const text = rulesArea?.value?.trim();
      if (!text) { log('Enter geological rules first.', 'warn'); return; }
      compileBtn.disabled = true;
      compileBtn.textContent = '⟳ Compiling rules…';
      if (rulesOut) { rulesOut.style.display = 'none'; rulesOut.innerHTML = ''; }

      try {
        const grid = AppState.voxelGrid;
        const bounds = grid ? {
          minX: grid.origin.x, maxX: grid.origin.x + grid.nx * grid.cellSize,
          minY: grid.origin.z, maxY: grid.origin.z + grid.ny * grid.cellSize,
        } : null;
        const compiled = await compileGeologicalRules(
          text, AppState.geoUnits, bounds, AppState.apiKey, AppState.demoMode
        );
        if (!compiled.length) {
          if (rulesOut) { rulesOut.style.display = 'block'; rulesOut.innerHTML = '<p class="hint" style="font-size:10px">No rules could be parsed from the input.</p>'; }
          return;
        }
        _renderCompiledRules(compiled, rulesOut);
        log(`${compiled.length} geological rule(s) compiled`, 'ok');
      } catch (err) {
        log(`Rule compilation failed: ${err.message}`, 'error');
      } finally {
        compileBtn.disabled = false;
        compileBtn.textContent = '⚡ Compile Geological Rules → Concepts';
      }
    });
  })();

  encodeBtn?.addEventListener('click', async () => {
    const text = textarea?.value?.trim();
    if (!text) return;

    encodeBtn.disabled   = true;
    encodeBtn.textContent = '⟳ Encoding…';
    const warnEl = document.getElementById('concept-encode-warnings');
    if (warnEl) { warnEl.style.display = 'none'; warnEl.innerHTML = ''; }
    try {
      const siteContext = {
        units: AppState.geoUnits.map(u => ({ code: u.code, name: u.name })),
        description: AppState.siteContext?.description ?? '',
      };
      // Request rationale when using live API (not demo mode) — adds ~100 extra tokens
      const encodeResult = await encodeGeologicalConcept(text, AppState.apiKey, AppState.demoMode, {
        siteContext, withRationale: !AppState.demoMode && !!AppState.apiKey,
      });
      const emb  = encodeResult instanceof Float32Array ? encodeResult : encodeResult.embedding;
      const encodeRationale = encodeResult?.rationale ?? null;
      const conf = parseFloat(confidence?.value ?? 0.7);

      // ── Pre-add checks ───────────────────────────────────────────────────────
      const encodeWarnings = [];

      // 1. Intra-concept axis contradictions on the new embedding
      const intraIssues = ConceptStore.detectIntraConflicts(emb);
      for (const msg of intraIssues) encodeWarnings.push({ sev: 'warning', text: msg });

      // 2. Similarity to existing concepts
      const similar = AppState.conceptStore.findSimilar(emb, 0.80);
      for (const { concept: sc, similarity: sim } of similar) {
        encodeWarnings.push({
          sev: sim >= 0.93 ? 'error' : 'warning',
          text: `Very similar to existing concept "${sc.description.slice(0, 40)}" (cosine ${(sim * 100).toFixed(0)}%) — consider increasing that concept's confidence instead of adding a duplicate`,
        });
      }

      // 3. Geological implication gap check: flag axes that this concept implies
      // but which are not covered by ANY existing concept (below threshold).
      const GEO_IMPLICATIONS = [
        { trigger: [5, 0.7],  implies: [23, 0.3], msg: 'channel_morphology implies a basal gravel lag (axis gravel_basal_lag) — consider adding a complementary concept' },
        { trigger: [5, 0.7],  implies: [8,  0.5], msg: 'channel_morphology implies an erosional contact (axis erosional_contact)' },
        { trigger: [7, 0.7],  implies: [18, 0.5], msg: 'fault_controlled implies a stepped boundary (axis stepped_boundary)' },
        { trigger: [24, 0.6], implies: [19, 0.4], msg: 'dissolution_features implies an irregular base (axis irregular_base)' },
        { trigger: [29, 0.6], implies: [5,  0.4], msg: 'incision_depth_ratio implies a channel form (axis channel_morphology)' },
        { trigger: [3, 0.6],  implies: [27, 0.4], msg: 'east_west_elongation implies lateral anisotropy (axis lateral_anisotropy)' },
        { trigger: [4, 0.6],  implies: [27, 0.4], msg: 'north_south_elongation implies lateral anisotropy' },
        { trigger: [21, 0.6], implies: [23, 0.3], msg: 'coarsening_upward implies a coarse base (gravel_basal_lag likely)' },
        { trigger: [25, 0.6], implies: [31, 0.3], msg: 'structural_complexity implies a complexity gradient (axis complexity_gradient)' },
      ];
      for (const rule of GEO_IMPLICATIONS) {
        const [trigAx, trigThresh] = rule.trigger;
        const [impAx,  impThresh]  = rule.implies;
        if ((emb[trigAx] ?? 0) >= trigThresh) {
          // Check if ANY existing concept covers the implied axis at threshold
          const covered = AppState.conceptStore.concepts.some(c => (c.embedding?.[impAx] ?? 0) >= impThresh);
          if (!covered && (emb[impAx] ?? 0) < impThresh) {
            encodeWarnings.push({ sev: 'info', text: `ℹ Geological implication: ${rule.msg}` });
          }
        }
      }

      if (warnEl && encodeWarnings.length) {
        warnEl.style.display = 'block';
        warnEl.innerHTML = encodeWarnings.map(w => {
          const col  = w.sev === 'error' ? 'var(--red)' : w.sev === 'info' ? 'var(--accent)' : '#e0a020';
          const icon = w.sev === 'error' ? '⚠' : w.sev === 'info' ? 'ℹ' : '△';
          const text = w.sev === 'info' ? w.text.replace(/^ℹ\s*/, '') : w.text;
          return `<div style="border-left:2px solid ${col};padding:3px 6px;margin-bottom:3px;font-size:9.5px;color:var(--text-mid)">
            <span style="color:${col};margin-right:4px">${icon}</span>${escHtml(text)}
          </div>`;
        }).join('');
        if (encodeWarnings.some(w => w.sev === 'error')) {
          log(`Concept has issues: ${encodeWarnings.map(w => w.text).join('; ')}`, 'warn');
        }
      }

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
      // Depth range: optionally restrict concept to a vertical band (minZ..maxZ in AOD)
      const minZEl = document.getElementById('concept-minz');
      const maxZEl = document.getElementById('concept-maxz');
      const sigZEl = document.getElementById('concept-sigmaz');
      const minZVal = minZEl?.value !== '' ? parseFloat(minZEl.value) : undefined;
      const maxZVal = maxZEl?.value !== '' ? parseFloat(maxZEl.value) : undefined;
      const sigZVal = sigZEl?.value !== '' ? parseFloat(sigZEl.value) : 10;
      if (minZVal !== undefined || maxZVal !== undefined) {
        // Merge depth constraints into domain object
        if (minZVal !== undefined) domain.minZ = minZVal;
        if (maxZVal !== undefined) domain.maxZ = maxZVal;
        domain.sigmaZ = sigZVal;
      }

      // Parent concept (inheritance): child blends in 40% of parent's embedding
      const parentSel = document.getElementById('concept-parent');
      const parentId  = parentSel?.value || null;

      AppState.conceptStore.add({ description: text, embedding: emb, confidence: conf, domain, unitAffinity, parentId });
      _renderConceptList();
      _saveConceptStore();

      // Show AI rationale if returned (API mode only)
      if (encodeRationale && warnEl) {
        warnEl.style.display = 'block';
        const rationaleHtml = `<div style="border-left:2px solid var(--accent);padding:3px 6px;margin-bottom:3px;font-size:9.5px;color:var(--text-mid);font-style:italic">
          <span style="color:var(--accent);margin-right:4px">ℹ</span>${escHtml(encodeRationale)}
        </div>`;
        warnEl.innerHTML = rationaleHtml + warnEl.innerHTML;
      }

      if (textarea) textarea.value = '';
      // Clear selection after encoding
      if (unitAffinitySel) Array.from(unitAffinitySel.options).forEach(o => o.selected = false);
      // Reset drawn bbox after encoding so it's not accidentally reused
      _drawnBboxDomain = null;
      if (bboxPreview) bboxPreview.textContent = '';
      // Clear depth range inputs
      if (minZEl) minZEl.value = '';
      if (maxZEl) maxZEl.value = '';
      const depthStr = (minZVal !== undefined || maxZVal !== undefined)
        ? ` Z=[${minZVal ?? '∞'}..${maxZVal ?? '∞'}]±${sigZVal}m` : '';
      log(`Concept encoded: "${text.slice(0, 60)}"${depthStr} — ${AppState.conceptStore.concepts.length} total`, 'ok');
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

  // Wire Live Axis Perturbation panel
  _initAxisPerturbation();
}

// ── Live Axis Perturbation ─────────────────────────────────────────────────────
// Renders 32 concept-axis sliders for a selected concept. On "Re-infer", clones
// the concept store with the modified embedding, runs inferGeoImplicit with the
// cached trained model (no retraining), and updates the 3D scene live.
function _initAxisPerturbation() {
  const toggle   = document.getElementById('axis-perturb-toggle');
  const body     = document.getElementById('axis-perturb-body');
  const selEl    = document.getElementById('axis-perturb-concept');
  const sliderEl = document.getElementById('axis-perturb-sliders');
  const statusEl = document.getElementById('axis-perturb-status');
  if (!toggle || !body) return;

  // Collapse toggle
  toggle.addEventListener('click', () => {
    body.hidden = !body.hidden;
    const arrow = toggle.querySelector('.collapse-arrow');
    if (arrow) arrow.textContent = body.hidden ? '›' : '⌄';
    if (!body.hidden) _refreshPerturbList();
  });

  function _refreshPerturbList() {
    if (!selEl || !AppState.conceptStore) return;
    const prev = selEl.value;
    selEl.innerHTML = '<option value="">— select concept —</option>';
    for (const c of (AppState.conceptStore?.concepts ?? [])) {
      if (!c?.id || !c?.description) continue;
      const opt = document.createElement('option');
      opt.value = c.id;
      opt.textContent = c.description.slice(0, 48);
      selEl.appendChild(opt);
    }
    if (prev && [...selEl.options].some(o => o.value === prev)) {
      selEl.value = prev;
    }
  }

  let _origEmb = null;

  function _loadSliders(conceptId) {
    if (!sliderEl) return;
    if (!AppState.conceptStore || !conceptId) { sliderEl.innerHTML = ''; return; }
    const concept = AppState.conceptStore.concepts.find(c => c.id === conceptId);
    if (!concept) { sliderEl.innerHTML = ''; _origEmb = null; return; }
    _origEmb = new Float32Array(concept.embedding);

    sliderEl.innerHTML = CONCEPT_AXES.map((name, i) => {
      const v = +(concept.embedding[i] ?? 0);
      const col = v > 0.2 ? '#4caf50' : v < -0.2 ? '#f44336' : '#888';
      return `<div style="display:grid;grid-template-columns:1fr 72px 30px;gap:2px;align-items:center;margin-bottom:2px">
        <label style="font-size:8.5px;color:${col};white-space:nowrap;overflow:hidden;text-overflow:ellipsis" title="${name}">${name.replace(/_/g,' ')}</label>
        <input type="range" id="axsl-${i}" data-idx="${i}" min="-1" max="1" step="0.05" value="${v.toFixed(2)}" style="height:12px;accent-color:${col}">
        <span id="axv-${i}" style="font-size:8.5px;color:var(--text-dim);text-align:right">${v.toFixed(2)}</span>
      </div>`;
    }).join('');

    let _previewTimer = null;
    sliderEl.querySelectorAll('input[type=range]').forEach(sl => {
      sl.addEventListener('input', () => {
        const idx = parseInt(sl.dataset.idx);
        const v = parseFloat(sl.value);
        const vEl = document.getElementById(`axv-${idx}`);
        if (vEl) vEl.textContent = v.toFixed(2);
        sl.style.accentColor = v > 0.2 ? '#4caf50' : v < -0.2 ? '#f44336' : '#888';
        // Debounce live preview
        clearTimeout(_previewTimer);
        _previewTimer = setTimeout(() => _updateLivePreview(selEl?.value), 250);
      });
    });
  }

  // Live cross-section preview using a tiny mini-grid (20 x 1 x 10 = 200 voxels)
  function _updateLivePreview(conceptId) {
    const canvas = document.getElementById('axis-perturb-preview');
    const label  = document.getElementById('axis-perturb-preview-label');
    if (!canvas || !AppState.trainedModel || !AppState.voxelGrid || !conceptId) return;

    // Build modified concept store from current slider values
    const modStore = AppState.conceptStore?.cloneScaled(1.0);
    if (!modStore) return;
    const mc = modStore.concepts.find(c => c.id === conceptId);
    if (!mc) return;
    CONCEPT_AXES.forEach((_, i) => {
      const sl = document.getElementById(`axsl-${i}`);
      if (sl) mc.embedding[i] = parseFloat(sl.value);
    });

    const grid = AppState.voxelGrid;
    // Mini grid: 22 E-W × 1 N-S (centre) × 10 depth levels
    const MINI_NX = 22, MINI_NY = 1, MINI_NZ = 10;
    const cs = grid.cellSize, ch = grid.cellHeight;
    const centreY = grid.origin.z + (grid.ny / 2) * cs;
    const miniGrid = {
      nx: MINI_NX, ny: MINI_NY, nz: MINI_NZ,
      cellSize: cs * (grid.nx / MINI_NX),
      cellHeight: ch * (grid.nz / MINI_NZ),
      origin: { x: grid.origin.x, y: grid.origin.y, z: centreY },
    };
    const result = inferGeoImplicit(AppState.trainedModel, miniGrid, AppState.geoUnits, modStore);
    if (!result?.unitIds) return;

    // Render to canvas: columns are E-W, rows are depth (top=shallow)
    canvas.style.display = 'block';
    if (label) label.style.display = 'block';
    const cw = canvas.width / MINI_NX, ch2 = canvas.height / MINI_NZ;
    const ctx2d = canvas.getContext('2d');
    ctx2d.clearRect(0, 0, canvas.width, canvas.height);
    const unitById = {};
    AppState.geoUnits.forEach(u => { unitById[u.id] = u; });
    for (let iz = 0; iz < MINI_NZ; iz++) {
      for (let ix = 0; ix < MINI_NX; ix++) {
        const unitId = result.unitIds[ix + 0 * MINI_NX + iz * MINI_NX];
        const unit   = unitById[unitId];
        ctx2d.fillStyle = unit?.color ?? '#333';
        ctx2d.fillRect(ix * cw, (MINI_NZ - 1 - iz) * ch2, Math.ceil(cw), Math.ceil(ch2));
      }
    }
    // Overlay "W" and "E" labels
    ctx2d.fillStyle = 'rgba(255,255,255,0.5)';
    ctx2d.font = '9px monospace';
    ctx2d.fillText('W', 2, 10);
    ctx2d.fillText('E', canvas.width - 12, 10);
  }

  selEl?.addEventListener('change', () => _loadSliders(selEl.value));

  async function _doReInfer(useOriginal = false) {
    const conceptId = selEl?.value;
    if (!conceptId) { if (statusEl) statusEl.textContent = 'Select a concept first.'; return; }
    if (!AppState.trainedModel || !AppState.voxelGrid) {
      if (statusEl) statusEl.textContent = 'Need neural-implicit model (not IDW/kriging).';
      return;
    }
    const concept = AppState.conceptStore.concepts.find(c => c.id === conceptId);
    if (!concept) return;

    if (statusEl) { statusEl.textContent = 'Re-inferring…'; statusEl.style.color = 'var(--text-muted)'; }
    setEnabled('btn-axis-perturb-apply', false);
    setEnabled('btn-axis-perturb-reset', false);

    try {
      const modStore = AppState.conceptStore.cloneScaled(1.0);
      const mc = modStore._concepts.find(c => c.id === conceptId);
      if (mc) {
        if (useOriginal && _origEmb) {
          mc.embedding = new Float32Array(_origEmb);
        } else {
          const newEmb = new Float32Array(32);
          for (let i = 0; i < 32; i++) {
            const sl = document.getElementById(`axsl-${i}`);
            const raw = sl?.value != null ? parseFloat(sl.value) : NaN;
            newEmb[i] = isFinite(raw) ? raw : (concept.embedding[i] ?? 0);
          }
          mc.embedding = newEmb;
        }
      }

      const grid = AppState.voxelGrid;
      const gridMeta = {
        nx: grid.nx, ny: grid.ny, nz: grid.nz,
        cellSize: grid.cellSize, cellHeight: grid.cellHeight, origin: grid.origin,
      };
      const newResult = inferGeoImplicit(AppState.trainedModel, gridMeta, AppState.geoUnits, modStore);

      let changed = 0;
      const n = grid.unitIds.length;
      for (let i = 0; i < n; i++) {
        if (newResult.unitIds[i] !== grid.unitIds[i]) changed++;
      }

      if (typeof grid.unitIds.set === 'function' && newResult.unitIds.length === grid.unitIds.length)
        grid.unitIds.set(newResult.unitIds);
      else grid.unitIds = newResult.unitIds;
      if (typeof grid.certainty.set === 'function' && newResult.certainty.length === grid.certainty.length)
        grid.certainty.set(newResult.certainty);
      else grid.certainty = newResult.certainty;
      AppState.scene.buildVoxels(grid, AppState.geoUnits, AppState.classifiedBH);

      const pct = (changed / n * 100).toFixed(1);
      if (statusEl) {
        statusEl.textContent = useOriginal
          ? 'Reset to original embedding.'
          : `${changed.toLocaleString()} voxels changed (${pct}%)`;
        statusEl.style.color = changed > 100 ? 'var(--ok)' : 'var(--text-muted)';
      }
    } catch (e) {
      if (statusEl) { statusEl.textContent = `Error: ${e.message}`; statusEl.style.color = 'var(--error)'; }
      console.error('Axis perturbation error:', e);
    } finally {
      setEnabled('btn-axis-perturb-apply', true);
      setEnabled('btn-axis-perturb-reset', true);
    }
  }

  document.getElementById('btn-axis-perturb-apply')?.addEventListener('click', () => _doReInfer(false));
  document.getElementById('btn-axis-perturb-reset')?.addEventListener('click', async () => {
    const conceptId = selEl?.value;
    if (!conceptId || !_origEmb) return;
    for (let i = 0; i < 32; i++) {
      const sl = document.getElementById(`axsl-${i}`);
      const vEl = document.getElementById(`axv-${i}`);
      if (sl) sl.value = _origEmb[i].toFixed(2);
      if (vEl) vEl.textContent = _origEmb[i].toFixed(2);
      if (sl) sl.style.accentColor = _origEmb[i] > 0.2 ? '#4caf50' : _origEmb[i] < -0.2 ? '#f44336' : '#888';
    }
    await _doReInfer(true);
  });

  window.addEventListener('geomodel:model-built', _refreshPerturbList);
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
    document.getElementById('btn-compare-scenarios')?.style.setProperty('display','none');
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
  const compareBtn = document.getElementById('btn-compare-scenarios');
  if (compareBtn) compareBtn.style.display = scenarios.length >= 2 ? '' : 'none';
}

window._compareConceptScenarios = function() {
  const scenarios = _loadScenarios();
  const out = document.getElementById('concept-scenario-compare');
  if (!out || scenarios.length < 2) return;

  // Compare first two saved scenarios — plus current
  const current = AppState.conceptStore && !AppState.conceptStore.isEmpty
    ? { name: '(current)', store: AppState.conceptStore }
    : null;
  const stores = [
    { name: scenarios[0].name, store: ConceptStore.deserialize(scenarios[0].json) },
    { name: scenarios[1].name, store: ConceptStore.deserialize(scenarios[1].json) },
    ...(current ? [current] : []),
  ];

  // Build comparison: per-axis aggregate embedding for each scenario
  const aggEmb = stores.map(s => {
    const vec = new Float32Array(32);
    for (const c of s.store.concepts) {
      for (let i = 0; i < 32; i++) vec[i] += c.embedding[i] * c.confidence;
    }
    const n = Math.max(1, s.store.concepts.length);
    for (let i = 0; i < 32; i++) vec[i] /= n;
    return vec;
  });

  // Top-5 most divergent axes across scenarios
  const diffs = CONCEPT_AXES.map((name, i) => {
    const vals = aggEmb.map(e => e[i]);
    const mn   = vals.reduce((a,b) => a+b, 0) / vals.length;
    const variance = vals.reduce((s,v) => s + (v-mn)**2, 0) / vals.length;
    return { i, name, vals, variance };
  }).sort((a,b) => b.variance - a.variance);

  let html = `<div style="font-size:10px;font-weight:600;margin-bottom:5px;color:var(--text-mid)">
    Scenario comparison — aggregate embedding (${stores.length} scenarios)</div>`;

  // Legend
  const HUE = [0.6, 0.95, 0.42]; // blue, red, green
  html += `<div style="display:flex;gap:6px;margin-bottom:6px;flex-wrap:wrap">` +
    stores.map((s,i) => {
      const col = `hsl(${Math.round(HUE[i]*360)},70%,50%)`;
      return `<span style="font-size:9px"><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:${col};vertical-align:middle;margin-right:3px"></span>${escHtml(s.name)}</span>`;
    }).join('') + `</div>`;

  // Top divergent axes
  html += `<div style="font-size:9px;color:var(--text-dim);margin-bottom:4px">Top divergent axes:</div>`;
  for (const { name, vals } of diffs.slice(0, 8)) {
    const bars = vals.map((v, si) => {
      const pct = Math.abs(v) * 100;
      const col = v >= 0 ? `hsl(${Math.round(HUE[si]*360)},70%,45%)` : `hsl(${Math.round(HUE[si]*360)},70%,45%)`;
      const dir = v >= 0 ? '' : '-';
      return `<div title="${name}: ${dir}${Math.abs(v).toFixed(2)}" style="flex:1;height:6px;border-radius:2px;background:var(--bg-deep);overflow:hidden">
        <div style="width:${pct}%;height:100%;background:${col}"></div>
      </div>`;
    }).join('');
    const valStr = vals.map((v, si) => `<span style="color:hsl(${Math.round(HUE[si]*360)},70%,50%)">${v >= 0 ? '+' : ''}${v.toFixed(2)}</span>`).join(' ');
    html += `<div style="margin-bottom:3px">
      <div style="font-size:9px;color:var(--text-primary);margin-bottom:1px">${escHtml(name)}</div>
      <div style="display:flex;gap:2px;margin-bottom:1px">${bars}</div>
      <div style="font-size:8px;font-family:var(--font-mono)">${valStr}</div>
    </div>`;
  }

  // Concept overlap analysis
  html += `<div style="font-size:9px;color:var(--text-dim);margin-top:6px;margin-bottom:3px">Concept overlap:</div>`;
  const A = stores[0].store.concepts.map(c => c.description.slice(0, 40).toLowerCase());
  const B = stores[1].store.concepts.map(c => c.description.slice(0, 40).toLowerCase());
  const onlyA   = A.filter(d => !B.some(b => b.includes(d.slice(0,15)) || d.includes(b.slice(0,15))));
  const onlyB   = B.filter(d => !A.some(a => a.includes(d.slice(0,15)) || d.includes(a.slice(0,15))));
  const shared  = A.filter(d => B.some(b => b.includes(d.slice(0,15)) || d.includes(b.slice(0,15))));
  html += `<div style="font-size:9px;color:var(--text-mid)">
    Unique to <b>${escHtml(stores[0].name)}</b>: ${onlyA.length} &nbsp;
    Unique to <b>${escHtml(stores[1].name)}</b>: ${onlyB.length} &nbsp;
    Shared: ${shared.length}
  </div>`;

  // Inference accuracy comparison (only when trainedModel is available)
  if (AppState.trainedModel && AppState.voxelGrid) {
    html += `<div style="font-size:9px;color:var(--text-dim);margin-top:8px;margin-bottom:3px">Running inference accuracy comparison…</div>`;
    out.style.display = 'block';
    out.innerHTML = html;

    const grid = AppState.voxelGrid;
    const gridMeta = {
      nx: grid.nx, ny: grid.ny, nz: grid.nz,
      cellSize: grid.cellSize, cellHeight: grid.cellHeight,
      origin: grid.origin,
    };

    const results = [];
    for (const { name, store } of stores) {
      try {
        const r = inferGeoImplicit(AppState.trainedModel, gridMeta, AppState.geoUnits, store);
        const acc = _bhAccuracy(r.unitIds);
        // Count voxels per dominant unit to measure distribution
        const unitCount = new Map();
        for (const id of r.unitIds) {
          unitCount.set(id, (unitCount.get(id) ?? 0) + 1);
        }
        results.push({ name, acc, unitIds: r.unitIds, unitCount });
      } catch (e) {
        results.push({ name, acc: null, error: e.message });
      }
    }

    // Voxel divergence between scenario 0 and 1
    let divergedVoxels = 0;
    if (results[0]?.unitIds && results[1]?.unitIds) {
      const len = Math.min(results[0].unitIds.length, results[1].unitIds.length);
      for (let i = 0; i < len; i++) {
        if (results[0].unitIds[i] !== results[1].unitIds[i]) divergedVoxels++;
      }
    }
    const totalV = grid.nx * grid.ny * grid.nz;
    const divergePct = totalV > 0 ? (divergedVoxels / totalV * 100).toFixed(1) : '—';

    html += `<div style="font-size:9px;color:var(--text-dim);margin-top:6px;margin-bottom:3px">Inference accuracy vs boreholes:</div>`;
    const best = results.reduce((b, r) => (r.acc != null && (b == null || r.acc > b.acc)) ? r : b, null);
    for (const r of results) {
      const isBest = r === best;
      const accStr = r.acc != null ? `${(r.acc * 100).toFixed(1)}%` : `error: ${r.error}`;
      html += `<div style="display:flex;gap:6px;align-items:center;margin-bottom:3px;font-size:9px">
        <span style="flex:1;font-weight:${isBest ? 600 : 400};color:${isBest ? '#7fcfb0' : 'var(--text-primary)'}">${escHtml(r.name)}</span>
        <span style="font-family:var(--font-mono);color:${r.acc != null ? '#f5a623' : '#e05'}">BH acc: ${accStr}</span>
        ${isBest && results.length > 1 ? '<span style="color:#7fcfb0;font-size:8px">best</span>' : ''}
      </div>`;
    }
    if (results[0]?.unitIds && results[1]?.unitIds) {
      html += `<div style="font-size:9px;color:var(--text-dim);margin-top:4px">Voxel divergence: <span style="color:var(--text-primary);font-family:var(--font-mono)">${divergePct}%</span> of model differs between scenarios</div>`;
    }
    if (best) {
      html += `<div style="font-size:9px;color:#7fcfb0;margin-top:6px;padding:4px 6px;background:rgba(127,207,176,0.08);border-radius:3px">
        Recommendation: <b>${escHtml(best.name)}</b> best fits borehole observations (${best.acc != null ? (best.acc*100).toFixed(1) : '?'}% accuracy).
      </div>`;
    }
  } else if (!AppState.trainedModel) {
    html += `<div style="font-size:9px;color:var(--text-dim);margin-top:6px;font-style:italic">Build neural model first to enable inference accuracy comparison.</div>`;
  }

  out.style.display = 'block';
  out.innerHTML = html;
  log(`Compared scenarios: "${stores[0].name}" vs "${stores[1].name}"`, 'info');
};

// ── Geological Event Timeline ─────────────────────────────────────────────────
// Leapfrog-style ordered list of geological events (oldest → youngest).
// Each event: { id, type, name, unitCodes[] }
// The list drives stratOrder and can auto-encode associated concepts.

const GEO_EVENT_TYPES = {
  deposition: { icon: '⬤', label: 'Deposition',      axes: { horizontal_layering: 0.7, lateral_continuity: 0.8, fining_upward: 0.3 } },
  erosion:    { icon: '≈',  label: 'Erosion/Incision', axes: { erosional_contact: 0.9, irregular_base: 0.7, incision_depth_ratio: 0.7 } },
  fault:      { icon: '/',  label: 'Faulting',         axes: { fault_controlled: 1.0, stepped_boundary: 0.8, structural_complexity: 0.7 } },
  intrusion:  { icon: '▲',  label: 'Intrusion',        axes: { dome_anticline: 0.7, structural_complexity: 0.6, dip_magnitude: 0.5 } },
  folding:    { icon: '~',  label: 'Folding',           axes: { inclined_bedding: 0.8, dip_magnitude: 0.7, structural_complexity: 0.6 } },
  karst:      { icon: '◉',  label: 'Karst dissolution', axes: { dissolution_features: 1.0, irregular_base: 0.8, structural_complexity: 0.5 } },
  fill:       { icon: '▥',  label: 'Channel/Valley fill', axes: { channel_morphology: 1.0, erosional_contact: 0.8, gravel_basal_lag: 0.7, fining_upward: 0.5 } },
  terrace:    { icon: '—',  label: 'Terrace formation', axes: { horizontal_layering: 0.7, lateral_continuity: 0.8, gravel_basal_lag: 0.6, erosional_contact: 0.5 } },
};

// Axis name → index mapping (matches CONCEPT_AXES in geo-implicit.js)
const AXIS_NAME_TO_IDX = {
  horizontal_layering:0, inclined_bedding:1, dip_magnitude:2, east_west_elongation:3,
  north_south_elongation:4, channel_morphology:5, dome_anticline:6, fault_controlled:7,
  erosional_contact:8, lateral_continuity:9, lateral_thinning_east:10, lateral_thinning_west:11,
  lateral_thinning_north:12, lateral_thinning_south:13, deepens_east:14, deepens_west:15,
  deepens_north:16, deepens_south:17, stepped_boundary:18, irregular_base:19,
  nested_channels:20, coarsening_upward:21, fining_upward:22, gravel_basal_lag:23,
  dissolution_features:24, structural_complexity:25, data_confidence:26, lateral_anisotropy:27,
  vertical_anisotropy:28, incision_depth_ratio:29, overburden_control:30, complexity_gradient:31,
};

function _renderGeoEventList() {
  const list = document.getElementById('geo-event-list');
  if (!list) return;
  const events = AppState.geoEvents; // oldest first in state; displayed oldest at bottom
  if (!events.length) {
    list.innerHTML = '<li style="font-size:10px;color:var(--text-muted);padding:4px 6px">No events added yet.</li>';
    return;
  }

  list.innerHTML = '';
  // Display: youngest at top → reverse the state array for display
  const displayed = [...events].reverse();
  displayed.forEach((evt, dispIdx) => {
    const stateIdx = events.length - 1 - dispIdx; // actual index in AppState.geoEvents
    const typeInfo = GEO_EVENT_TYPES[evt.type] ?? { icon: '?', label: evt.type };
    const li = document.createElement('li');
    li.className = 'geo-event-item';
    li.draggable = true;
    li.dataset.id = evt.id;
    li.dataset.stateIdx = stateIdx;
    li.innerHTML = `
      <span class="geo-event-drag" title="Drag to reorder">⠿</span>
      <span class="geo-event-type-icon" title="${typeInfo.label}">${typeInfo.icon}</span>
      <span class="geo-event-name-text">${evt.name || typeInfo.label}</span>
      ${evt.unitCodes?.length ? `<span class="geo-event-units-text">${evt.unitCodes.join(',')}</span>` : ''}
      <button class="geo-event-del" title="Remove" data-id="${evt.id}">×</button>
    `;
    list.appendChild(li);
  });

  // ── Drag-and-drop ─────────────────────────────────────────────────────────
  let dragSrc = null;
  list.querySelectorAll('.geo-event-item').forEach(item => {
    item.addEventListener('dragstart', e => {
      dragSrc = item; e.dataTransfer.effectAllowed = 'move';
      item.classList.add('dragging');
    });
    item.addEventListener('dragend', () => {
      item.classList.remove('dragging');
      list.querySelectorAll('.geo-event-item').forEach(i => i.classList.remove('drag-over'));
    });
    item.addEventListener('dragover', e => {
      e.preventDefault();
      if (item !== dragSrc) {
        list.querySelectorAll('.geo-event-item').forEach(i => i.classList.remove('drag-over'));
        item.classList.add('drag-over');
      }
    });
    item.addEventListener('drop', e => {
      e.preventDefault();
      if (dragSrc && dragSrc !== item) {
        const allItems = [...list.querySelectorAll('.geo-event-item')];
        const srcDispIdx = allItems.indexOf(dragSrc);
        const dstDispIdx = allItems.indexOf(item);
        // Reorder in AppState (events are oldest-first; display is reversed)
        const srcStateIdx = AppState.geoEvents.length - 1 - srcDispIdx;
        const dstStateIdx = AppState.geoEvents.length - 1 - dstDispIdx;
        const [moved] = AppState.geoEvents.splice(srcStateIdx, 1);
        AppState.geoEvents.splice(dstStateIdx, 0, moved);
        _renderGeoEventList();
      }
      list.querySelectorAll('.geo-event-item').forEach(i => i.classList.remove('drag-over'));
    });
  });

  // Delete buttons
  list.querySelectorAll('.geo-event-del').forEach(btn => {
    btn.addEventListener('click', () => {
      const id = btn.dataset.id;
      AppState.geoEvents = AppState.geoEvents.filter(e => e.id !== id);
      _renderGeoEventList();
    });
  });
}

function _initGeoEventTimeline() {
  const toggle = document.getElementById('geo-events-toggle');
  const body   = document.getElementById('geo-events-body');
  if (toggle && body) {
    toggle.addEventListener('click', () => {
      body.hidden = !body.hidden;
      const arrow = toggle.querySelector('.collapse-arrow');
      if (arrow) arrow.textContent = body.hidden ? '›' : '⌄';
    });
  }

  document.getElementById('btn-add-geo-event')?.addEventListener('click', () => {
    const type  = document.getElementById('geo-event-type')?.value ?? 'deposition';
    const name  = document.getElementById('geo-event-name')?.value?.trim() ?? '';
    const rawU  = document.getElementById('geo-event-units')?.value ?? '';
    const units = rawU.split(/[,\s]+/).map(s => s.trim().toUpperCase()).filter(Boolean);
    const evt = {
      id: `evt_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
      type, name, unitCodes: units,
    };
    AppState.geoEvents.push(evt); // push to end = youngest
    document.getElementById('geo-event-name').value = '';
    document.getElementById('geo-event-units').value = '';
    _renderGeoEventList();
  });

  // Apply: set stratOrder from event list (youngest unit codes first = top)
  document.getElementById('btn-apply-geo-events')?.addEventListener('click', () => {
    const events = AppState.geoEvents; // oldest first
    const orderCodes = [];
    // Walk oldest → youngest; collect unit codes in that order (oldest = deepest = last in strat)
    for (const evt of events) {
      for (const code of (evt.unitCodes ?? [])) {
        if (!orderCodes.includes(code)) orderCodes.push(code);
      }
    }
    // stratOrder = youngest → oldest (top → bottom)
    const stratCodes = [...orderCodes].reverse();
    AppState.stratOrder = stratCodes;
    AppState._stratDisplayOrder = stratCodes;
    // Lock the strat column
    const lockEl = document.getElementById('strat-manual-lock');
    if (lockEl) lockEl.checked = true;
    updateStratColumn();
    log(`Event timeline applied → stratigraphic order: ${stratCodes.join(' → ')}`, 'ok');
  });

  // Auto-encode: for each event, build a 32-dim embedding from type axes and encode it
  document.getElementById('btn-events-to-concepts')?.addEventListener('click', async () => {
    const events = AppState.geoEvents;
    if (!events.length) { log('Add geological events first.', 'warn'); return; }
    let added = 0;
    for (const evt of events) {
      const typeInfo = GEO_EVENT_TYPES[evt.type];
      if (!typeInfo) continue;
      const vec = new Float32Array(32).fill(0);
      for (const [axisName, value] of Object.entries(typeInfo.axes)) {
        const idx = AXIS_NAME_TO_IDX[axisName];
        if (idx != null) vec[idx] = value;
      }
      const description = `${typeInfo.label}${evt.name ? ': ' + evt.name : ''}`;
      const unitCodes = evt.unitCodes?.length ? evt.unitCodes : null;
      const conceptId = AppState.conceptStore.add({
        description, embedding: vec, confidence: 0.70,
        domain: { type: 'global' }, unitAffinity: unitCodes,
      });
      added++;
    }
    _renderConceptList();
    _saveConceptStore();
    log(`Auto-encoded ${added} event-type concept(s) from timeline.`, 'ok');
  });

  _renderGeoEventList();
}

// ── Concept conflict detection ────────────────────────────────────────────────
// Returns an array of conflict objects describing contradictions within or
// between active concepts. Each entry has:
//   type: 'intra' | 'inter'
//   description: human-readable explanation
//   severity: 'warning' | 'error'
//   conceptIds: array of involved concept IDs
export function detectConceptConflicts() {
  const store = AppState.conceptStore;
  if (!store || store.isEmpty) return [];

  const conflicts = [];
  const concepts  = store.concepts.filter(c => c.confidence >= 0.15);

  // ── 1. Intra-concept contradictions ─────────────────────────────────────────
  // Axes that are geometrically incompatible within the same concept embedding.
  const INTRA_PAIRS = [
    { a: 0,  b: 5,  label: 'horizontal_layering vs channel_morphology',
      desc: 'Flat-bedded (axis 0) and concave-up channel (axis 5) are geometrically incompatible. One should be negative.' },
    { a: 6,  b: 0,  label: 'dome_anticline vs horizontal_layering',
      desc: 'A dome/anticline (axis 6) and strong horizontal layering (axis 0) conflict — domed bodies are not flat.' },
    { a: 7,  b: 5,  label: 'fault_controlled vs channel_morphology',
      desc: 'Fault-controlled geometry (axis 7) and channel morphology (axis 5) have different spatial scales and styles.' },
    { a: 14, b: 15, label: 'deepens_east vs deepens_west',
      desc: 'Cannot deepen in both east and west simultaneously — these are opposing dip directions.' },
    { a: 16, b: 17, label: 'deepens_north vs deepens_south',
      desc: 'Cannot deepen in both north and south simultaneously — these are opposing dip directions.' },
    { a: 10, b: 11, label: 'thinning_east vs thinning_west',
      desc: 'Cannot thin in both east and west simultaneously.' },
    { a: 12, b: 13, label: 'thinning_north vs thinning_south',
      desc: 'Cannot thin in both north and south simultaneously.' },
    { a: 21, b: 22, label: 'coarsening_upward vs fining_upward',
      desc: 'A sequence cannot simultaneously coarsen and fine upward.' },
  ];

  for (const c of concepts) {
    const emb = c.embedding;

    // Check intra-pairs: both strongly positive (same direction) = contradiction
    for (const { a, b, label, desc } of INTRA_PAIRS) {
      if (emb[a] > 0.45 && emb[b] > 0.45) {
        conflicts.push({
          type: 'intra', severity: 'warning',
          conceptIds: [c.id],
          description: `"${c.description.slice(0, 40)}": ${label} — ${desc}`,
        });
      }
    }

    // E-W and N-S elongation both strongly positive → isotropic (concept has no directional effect)
    if (emb[3] > 0.5 && emb[4] > 0.5) {
      conflicts.push({
        type: 'intra', severity: 'warning',
        conceptIds: [c.id],
        description: `"${c.description.slice(0, 40)}": both E-W and N-S elongation are high — body will be isotropic, not directionally elongated. Choose the dominant direction.`,
      });
    }
  }

  // ── 2. Inter-concept contradictions ──────────────────────────────────────────
  // Between pairs of concepts: axes where embeddings strongly oppose.
  // Only flag when both concepts have the same (or nearby) spatial domain and
  // both have significant weight (confidence × relevance).
  const CONFLICT_AXES = [3, 4, 0, 6, 5, 7, 14, 15, 16, 17]; // key geometry axes
  for (let i = 0; i < concepts.length; i++) {
    for (let j = i + 1; j < concepts.length; j++) {
      const ca = concepts[i], cb = concepts[j];
      // Skip if domains are spatially separated (both bbox with no overlap)
      const aGlobal = !ca.domain || ca.domain.type === 'global';
      const bGlobal = !cb.domain || cb.domain.type === 'global';
      if (!aGlobal && !bGlobal) {
        const ad = ca.domain, bd = cb.domain;
        // Simple AABB overlap check
        if (ad.maxX < bd.minX || bd.maxX < ad.minX || ad.maxY < bd.minY || bd.maxY < ad.minY) continue;
      }

      const opposingAxes = [];
      for (const ax of CONFLICT_AXES) {
        const va = ca.embedding[ax], vb = cb.embedding[ax];
        if (va * vb < 0 && Math.abs(va) > 0.4 && Math.abs(vb) > 0.4) {
          opposingAxes.push({ ax, axName: CONCEPT_AXES[ax], va, vb });
        }
      }
      // Only flag when 2+ axes conflict, or 1 axis conflicts at high magnitude
      const strongConflicts = opposingAxes.filter(o => Math.abs(o.va) > 0.6 && Math.abs(o.vb) > 0.6);
      if (opposingAxes.length >= 2 || strongConflicts.length >= 1) {
        const axNames = opposingAxes.slice(0, 3).map(o => o.axName).join(', ');
        conflicts.push({
          type: 'inter', severity: strongConflicts.length ? 'error' : 'warning',
          conceptIds: [ca.id, cb.id],
          description: `"${ca.description.slice(0, 25)}" vs "${cb.description.slice(0, 25)}": opposing values on [${axNames}] — their combined effect will partially cancel. Consider separating to different spatial domains.`,
        });
      }
    }
  }

  // ── 3. Temporal order conflicts ───────────────────────────────────────────────
  // If two concepts both have temporalOrder AND unitAffinity, check whether the
  // borehole data actually supports the declared order (younger unit appears above
  // the older one in observed sections).
  const temporalPairs = AppState.conceptStore?.temporallyOrderedPairs?.() ?? [];
  for (const { younger, older } of temporalPairs) {
    if (!younger.unitAffinity?.length || !older.unitAffinity?.length) continue;
    // Gather observed vertical ordering from boreholes
    const boreholes = AppState.classifiedBH ?? [];
    let correctCount = 0, totalContacts = 0;
    for (const bh of boreholes) {
      const layers = bh.layers ?? [];
      for (let i = 0; i < layers.length; i++) {
        for (let j = i + 1; j < layers.length; j++) {
          const la = layers[i], lb = layers[j]; // la is shallower (higher elevation)
          const laIsYounger = younger.unitAffinity.includes(la.unitCode) && older.unitAffinity.includes(lb.unitCode);
          const lbIsYounger = younger.unitAffinity.includes(lb.unitCode) && older.unitAffinity.includes(la.unitCode);
          if (laIsYounger || lbIsYounger) {
            totalContacts++;
            if (laIsYounger) correctCount++; // younger above older = correct
          }
        }
      }
    }
    if (totalContacts >= 3 && correctCount / totalContacts < 0.4) {
      conflicts.push({
        type: 'temporal', severity: 'error',
        conceptIds: [younger.id, older.id],
        description: `Temporal order conflict: "${younger.description.slice(0, 25)}" declared younger than "${older.description.slice(0, 25)}" but borehole data shows the opposite in ${totalContacts} observed contacts. Check your age rank assignments.`,
      });
    } else if (totalContacts >= 3 && correctCount / totalContacts < 0.6) {
      conflicts.push({
        type: 'temporal', severity: 'warning',
        conceptIds: [younger.id, older.id],
        description: `Temporal order uncertain: "${younger.description.slice(0, 25)}" vs "${older.description.slice(0, 25)}" — only ${Math.round(correctCount/totalContacts*100)}% of observed contacts agree with declared age rank.`,
      });
    }
  }

  return conflicts;
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

      // E-W vs N-S elongation axes
      const ewAxis = emb[3]; // east_west_elongation
      const nsAxis = emb[4]; // north_south_elongation
      const netEW = ewAxis - nsAxis; // positive → should be E-W elongated
      if (Math.abs(netEW) > 0.25) {
        const actualRatio = spanX / Math.max(spanY, 0.1);
        const expectedRatio = Math.exp(netEW * 1.4);
        const match = 1 - Math.min(1, Math.abs(Math.log(actualRatio / Math.max(expectedRatio, 0.05))) / 2.0);
        details.push({ axis: 'E/W vs N/S elongation', expected: netEW.toFixed(2), actual: actualRatio.toFixed(1), match });
        score += (match - 0.5) * 0.35 * Math.abs(netEW);
      }

      // Lateral anisotropy (any direction): concept says max(spanX,spanY) >> min
      const latAxis = emb[27]; // lateral_anisotropy
      if (latAxis > 0.3) {
        const horizRatio = Math.max(spanX, spanY) / Math.max(Math.min(spanX, spanY), 0.1);
        const match = Math.min(1, (horizRatio - 1) / (Math.exp(latAxis * 1.4) - 1 + 0.1));
        details.push({ axis: 'lateral anisotropy', expected: latAxis.toFixed(2), actual: horizRatio.toFixed(1), match });
        score += (match - 0.5) * 0.2 * latAxis;
      }

      // Depth incision: concept says spanZ proportional to incision_depth_ratio
      const inciAxis = emb[29]; // incision_depth_ratio
      if (Math.abs(inciAxis) > 0.3) {
        const actualDepth = spanZ / Math.max(Math.min(spanX, spanY), 0.1);
        const expectedDepth = Math.exp(inciAxis * 0.8);
        const match = 1 - Math.min(1, Math.abs(Math.log(actualDepth / Math.max(expectedDepth, 0.05))) / 1.5);
        details.push({ axis: 'incision depth', expected: inciAxis.toFixed(2), actual: actualDepth.toFixed(2), match });
        score += (match - 0.5) * 0.2 * Math.abs(inciAxis);
      }

      // Horizontal layering: concept says spanX/spanY >> spanZ (flat body)
      const layerAxis = emb[0]; // horizontal_layering
      if (Math.abs(layerAxis) > 0.3) {
        const actualFlat = Math.max(spanX, spanY) / Math.max(spanZ, 0.01);
        const match = layerAxis > 0 ? (actualFlat > 3 ? 1 : actualFlat / 3) : (actualFlat < 3 ? 1 : 3 / actualFlat);
        details.push({ axis: 'horizontal layering', expected: layerAxis.toFixed(2), actual: actualFlat.toFixed(1), match });
        score += (match - 0.5) * 0.2 * Math.abs(layerAxis);
      }

      // Channel morphology: concept says body should be narrow relative to incision
      const chanAxis = emb[5]; // channel_morphology
      if (chanAxis > 0.3) {
        // High channel_morphology → body should be deep relative to width
        const narrowness = spanZ / Math.max(Math.max(spanX, spanY), 0.1);
        const match = Math.min(1, narrowness / Math.exp(chanAxis * 0.8));
        details.push({ axis: 'channel morphology', expected: chanAxis.toFixed(2), actual: narrowness.toFixed(2), match });
        score += (match - 0.5) * 0.15 * chanAxis;
      }
    }

    score = Math.max(0, Math.min(1, score));
    return { conceptId: c.id, description: c.description, score, details };
  });
}

// Show coherence scores when triggered (button in concept panel)
// ── Section line suggestion from concept anisotropy ───────────────────────────
document.getElementById('btn-suggest-sections')?.addEventListener('click', () => {
  const out = document.getElementById('concept-section-suggestions');
  if (!out) return;
  if (!AppState.conceptStore || AppState.conceptStore.isEmpty) {
    log('Add concepts first to get section suggestions.', 'warn'); return;
  }

  const tensor  = AppState.conceptStore.globalTensor();
  const geoUnits = AppState.geoUnits;
  const grid     = AppState.voxelGrid;
  const bhs      = AppState.classifiedBH?.filter(b => !b.synthetic) ?? [];

  // Compute site centre from grid or boreholes
  let cx, cz, halfSpan;
  if (grid) {
    cx = grid.origin.x + grid.nx * grid.cellSize / 2;
    cz = grid.origin.z + grid.ny * grid.cellSize / 2;
    halfSpan = Math.max(grid.nx, grid.ny) * grid.cellSize / 2;
  } else if (bhs.length) {
    cx = bhs.reduce((s, b) => s + b.x, 0) / bhs.length;
    cz = bhs.reduce((s, b) => s + (b.y ?? b.z ?? 0), 0) / bhs.length;
    const xs = bhs.map(b => b.x), zs = bhs.map(b => b.y ?? b.z ?? 0);
    halfSpan = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...zs) - Math.min(...zs)) / 2;
  } else {
    log('Add boreholes or build a model first.', 'warn'); return;
  }
  halfSpan = Math.max(halfSpan, 20);

  // Derive dominant elongation direction from global tensor
  const theta = tensor.theta ?? 0; // angle of major axis from North (radians)
  const ew    = (AppState.conceptStore.concepts.reduce((s, c) => s + c.embedding[3] * c.confidence, 0) /
                 Math.max(1, AppState.conceptStore.concepts.reduce((s,c) => s + c.confidence, 0)));
  const ns    = (AppState.conceptStore.concepts.reduce((s, c) => s + c.embedding[4] * c.confidence, 0) /
                 Math.max(1, AppState.conceptStore.concepts.reduce((s,c) => s + c.confidence, 0)));
  const hasFault = AppState.conceptStore.concepts.some(c => c.embedding[7] > 0.5);

  // Generate 3 section suggestions:
  // 1. Perpendicular to elongation (crosses the body — shows thickness variation)
  // 2. Along elongation (shows along-strike continuity)
  // 3. 45° (catches NE-SW structural features)
  const suggestions = [];
  const elongationAngle = Math.atan2(ew, ns); // angle from North to elongation direction
  const perpAngle = elongationAngle + Math.PI / 2; // perpendicular cuts across the body

  suggestions.push({
    name:   'Cross-strike section',
    reason: `Perpendicular to ${Math.abs(ew) > Math.abs(ns) ? 'E-W' : 'N-S'} predicted elongation — best reveals unit thickness and base geometry`,
    x1: cx - Math.sin(perpAngle) * halfSpan,
    z1: cz - Math.cos(perpAngle) * halfSpan,
    x2: cx + Math.sin(perpAngle) * halfSpan,
    z2: cz + Math.cos(perpAngle) * halfSpan,
  });
  suggestions.push({
    name:   'Along-strike section',
    reason: `Along the predicted ${Math.abs(ew) > Math.abs(ns) ? 'E-W' : 'N-S'} elongation direction — verifies lateral continuity and pinch-out`,
    x1: cx - Math.sin(elongationAngle) * halfSpan,
    z1: cz - Math.cos(elongationAngle) * halfSpan,
    x2: cx + Math.sin(elongationAngle) * halfSpan,
    z2: cz + Math.cos(elongationAngle) * halfSpan,
  });
  if (hasFault) {
    suggestions.push({
      name:   'Fault-perpendicular section',
      reason: 'Perpendicular to predicted fault trend — shows vertical offset and stepped contacts',
      x1: cx - Math.sin(elongationAngle + Math.PI / 4) * halfSpan,
      z1: cz - Math.cos(elongationAngle + Math.PI / 4) * halfSpan,
      x2: cx + Math.sin(elongationAngle + Math.PI / 4) * halfSpan,
      z2: cz + Math.cos(elongationAngle + Math.PI / 4) * halfSpan,
    });
  }

  out.style.display = 'block';
  out.innerHTML = suggestions.map((s, i) =>
    `<div style="border:1px solid var(--border);border-radius:4px;padding:5px 6px;margin-bottom:4px;font-size:10px">
      <div style="font-weight:600;color:var(--text-primary);margin-bottom:1px">${escHtml(s.name)}</div>
      <div style="font-size:9px;color:var(--text-dim);margin-bottom:3px">${escHtml(s.reason)}</div>
      <div style="font-family:var(--font-mono);font-size:9px;color:var(--text-mid);margin-bottom:3px">
        (${s.x1.toFixed(0)}, ${s.z1.toFixed(0)}) → (${s.x2.toFixed(0)}, ${s.z2.toFixed(0)})
      </div>
      <button class="btn-ghost btn-sm" style="font-size:9px" data-sec-i="${i}">⌖ Apply section in 3D</button>
    </div>`
  ).join('');

  out.querySelectorAll('[data-sec-i]').forEach(btn => {
    btn.addEventListener('click', () => {
      const s = suggestions[parseInt(btn.dataset.secI)];
      if (!s || !AppState.scene?.slicer) { log('Open the 3D view first.', 'warn'); return; }
      AppState.scene.slicer.setByWorldPoints(s.x1, s.z1, s.x2, s.z2);
      log(`Section line set: ${s.name}`, 'ok');
      btn.textContent = '✓ Applied';
      btn.disabled = true;
    });
  });
  log(`${suggestions.length} section suggestions based on concept anisotropy`, 'info');
});

// ── Borehole concept validation ───────────────────────────────────────────────
// For each borehole, samples concept context at observed depths and checks whether
// the dominant concept's unitAffinity matches what was actually observed.
// Uses the built model's column predictions when available; falls back to concept-only.
document.getElementById('btn-bh-concept-validation')?.addEventListener('click', () => {
  const out = document.getElementById('bh-concept-validation-output');
  if (!out) return;
  const store = AppState.conceptStore;
  if (!store || store.isEmpty) { log('Add concepts first.', 'warn'); return; }
  const bhs = (AppState.classifiedBH ?? []).filter(b => !b.synthetic);
  if (!bhs.length) { log('Load borehole data first.', 'warn'); return; }
  const grid = AppState.voxelGrid;

  const results = [];
  for (const bh of bhs) {
    let matched = 0, total = 0;
    const layers = (bh.layers ?? []).filter(l => l.unitCode && l.unitCode !== 'UNKN');
    for (const layer of layers) {
      const nSamples = Math.max(1, Math.round((layer.base - layer.top) / 2));
      for (let s = 0; s < nSamples; s++) {
        const wz = bh.groundLevel - (layer.top + (s + 0.5) / nSamples * (layer.base - layer.top));
        total++;

        // Get model prediction at this voxel if grid available
        let modelPredCode = null;
        if (grid) {
          const { nx, ny, nz, cellSize: cs, cellHeight: ch, origin: O, unitIds } = grid;
          const ix = Math.round((bh.x - O.x) / cs - 0.5);
          const iy = Math.round((bh.y - O.z) / cs - 0.5);
          const iz = Math.round((wz - O.y) / ch - 0.5);
          if (ix >= 0 && ix < nx && iy >= 0 && iy < ny && iz >= 0 && iz < nz) {
            const uid = unitIds[ix + iy * nx + iz * nx * ny];
            modelPredCode = AppState.geoUnits.find(u => u.id === uid)?.code ?? null;
          }
        }

        // Compare model prediction with observation
        if (modelPredCode !== null) {
          if (modelPredCode === layer.unitCode) matched++;
        } else {
          // Fallback: concept-only prediction — check if dominant concept's unitAffinity matches
          const ctx = store.computeAt(bh.x, bh.y, wz, layer.unitCode);
          if (ctx.totalWeight < 0.05) { matched++; continue; } // no active concepts = neutral
          const topConcept = store.concepts.find(c => c.id === ctx.weights[0]?.id);
          if (!topConcept?.unitAffinity?.length || topConcept.unitAffinity.includes(layer.unitCode)) {
            matched++;
          }
        }
      }
    }
    if (total > 0) results.push({ id: bh.id, score: matched / total, total, matched, x: bh.x, y: bh.y });
  }

  if (!results.length) { log('No data to validate.', 'warn'); return; }
  const globalScore = results.reduce((s,r) => s + r.score, 0) / results.length;

  out.style.display = 'block';
  const scoreCol = s => s >= 0.8 ? 'var(--accent)' : s >= 0.6 ? '#d4a843' : 'var(--red)';
  out.innerHTML = `
    <div style="font-size:10px;font-weight:600;margin-bottom:5px;color:var(--text-mid)">
      ${grid ? 'Model' : 'Concept'} prediction accuracy vs borehole observations
    </div>
    <div style="margin-bottom:6px;padding:5px;background:var(--bg-surface);border-radius:4px;text-align:center">
      <div style="font-size:20px;font-weight:700;color:${scoreCol(globalScore)}">${(globalScore*100).toFixed(0)}%</div>
      <div style="font-size:9px;color:var(--text-dim)">Mean accuracy across ${results.length} boreholes</div>
    </div>
    ${results.sort((a,b) => a.score - b.score).map(r => `
      <div style="display:flex;align-items:center;gap:5px;margin-bottom:2px;font-size:10px">
        <span style="width:30px;font-family:var(--font-mono);font-size:9px">${escHtml(r.id ?? '?')}</span>
        <div style="flex:1;height:5px;background:var(--bg-deep);border-radius:2px;overflow:hidden">
          <div style="width:${(r.score*100).toFixed(0)}%;height:100%;background:${scoreCol(r.score)}"></div>
        </div>
        <span style="font-family:var(--font-mono);font-size:9px;color:${scoreCol(r.score)};min-width:28px;text-align:right">${(r.score*100).toFixed(0)}%</span>
        <span style="font-size:8px;color:var(--text-dim)">${r.matched}/${r.total}</span>
      </div>`).join('')}
    ${globalScore < 0.6 ? `<div style="margin-top:5px;font-size:9px;color:#d4a843;border-left:2px solid #d4a843;padding-left:5px">
      Low accuracy: consider refining concept unit affinities or adding more targeted concepts for poorly predicted boreholes.
    </div>` : ''}
  `;
  log(`Concept validation: ${(globalScore*100).toFixed(0)}% mean accuracy across ${results.length} boreholes`, globalScore >= 0.7 ? 'ok' : 'warn');
});

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
    const dom       = c.domain?.type === 'bbox' ? '⬛ bbox' : '🌐 global';
    const affText   = c.unitAffinity?.length ? ` · ${c.unitAffinity.join(',')}` : '';
    const depthText = (c.domain?.minZ !== undefined || c.domain?.maxZ !== undefined)
      ? ` · Z [${c.domain.minZ ?? '?'}..${c.domain.maxZ ?? '?'}]m`
      : '';
    const confPct   = (c.confidence * 100).toFixed(0);
    const parentC   = c.parentId ? concepts.find(p => p.id === c.parentId) : null;
    const parentTag = parentC
      ? `<span style="font-size:9px;color:var(--accent);opacity:.8;margin-left:4px" title="Inherits 40% of parent embedding">↳ ${parentC.description.slice(0, 28)}</span>`
      : '';
    return `<div class="concept-entry" data-id="${c.id}">
      <div class="concept-header">
        <span class="concept-desc" title="${c.description}">${c.description.slice(0, 55)}${c.description.length > 55 ? '…' : ''}</span>
        <div style="display:flex;gap:2px">
          <button class="concept-radar-btn" title="Show radar chart" onclick="_toggleConceptRadar('${c.id}', this)">◎</button>
          <button class="concept-hl-btn" title="Highlight this concept's influence in 3D" onclick="_highlightConcept3D('${c.id}', this)" style="font-size:10px;padding:1px 4px;background:none;border:1px solid var(--border);border-radius:3px;color:var(--text-dim);cursor:pointer">🔦</button>
          <button class="concept-hl-btn" title="Score this concept's coherence with BH data" onclick="_checkConceptCoherence('${c.id}')" style="font-size:10px;padding:1px 4px;background:none;border:1px solid var(--border);border-radius:3px;color:var(--text-dim);cursor:pointer" title="Check how well this concept is supported by borehole data">≈</button>
          <button class="concept-remove" title="Remove concept" onclick="_removeConcept('${c.id}')">×</button>
        </div>
      </div>
      <div class="concept-meta">
        <span class="concept-dom-tag">${dom}${affText}${depthText}${parentTag}</span>
        <label class="concept-conf-row">
          conf <input type="range" class="concept-conf-slider" min="0" max="100" value="${confPct}"
            oninput="this.nextElementSibling.textContent=this.value+'%'; _updateConceptConf('${c.id}', this.value/100)"
          ><span class="concept-conf-val">${confPct}%</span>
        </label>
      </div>
      <div class="concept-temporal-row">
        <span class="concept-temporal-label" title="Geological age rank — lower = older (deposited first). Used to inject training constraints so younger units appear above older ones in data-sparse areas.">Age rank:</span>
        <button class="concept-temporal-btn" onclick="_conceptTemporalDec('${c.id}')" title="Older (decrease rank)">−</button>
        <span class="concept-temporal-val" id="trank-${c.id}">${c.temporalOrder !== null && c.temporalOrder !== undefined ? c.temporalOrder : '—'}</span>
        <button class="concept-temporal-btn" onclick="_conceptTemporalInc('${c.id}')" title="Younger (increase rank)">+</button>
        <button class="concept-temporal-btn" onclick="_conceptTemporalClear('${c.id}')" title="Clear rank" style="font-size:9px;padding:0 3px">✕</button>
        ${c.temporalOrder !== null && c.temporalOrder !== undefined
          ? `<span style="font-size:9px;color:var(--text-dim);margin-left:3px">${c.temporalOrder === 0 ? '(oldest)' : ''}</span>`
          : ''}
      </div>
      <div class="concept-axes">${bars}</div>
      <canvas class="concept-radar-canvas" id="radar-${c.id}" width="200" height="200" style="display:none;margin:4px auto 0;border-radius:4px;background:#f3f5f8"></canvas>
      <div id="coherence-${c.id}" class="concept-coherence" style="display:none"></div>
    </div>`;
  }).join('');
  _updateConceptInfluenceBar();
  // Refresh parent concept selector with current concept list
  const parentSel = document.getElementById('concept-parent');
  if (parentSel) {
    const prevVal = parentSel.value;
    parentSel.innerHTML = '<option value="">None (standalone concept)</option>'
      + concepts.map(c => `<option value="${c.id}">${escHtml(c.description.slice(0, 50))}</option>`).join('');
    if (concepts.some(c => c.id === prevVal)) parentSel.value = prevVal;
  }
  // Conflict detection: run live after every concept change so warnings appear inline
  _renderConceptConflicts();
  // Update 3D scene concept domain boxes (only bbox concepts show a 3D marker)
  AppState.scene?.drawConceptDomains?.(AppState.conceptStore);
  // Refresh correlation matrix and manifold map if panels are open
  if (!document.getElementById('concept-corr-body')?.hasAttribute('hidden')) {
    _drawConceptCorrelationMatrix();
  }
  if (!document.getElementById('concept-manifold-body')?.hasAttribute('hidden')) {
    _drawConceptManifold();
  }
}

window._checkConceptCoherence = function(id) {
  const concept = AppState.conceptStore?.concepts.find(c => c.id === id);
  const el = document.getElementById(`coherence-${id}`);
  if (!concept || !el) return;

  const bhs = AppState.classifiedBH?.filter(b => !b.synthetic && b.layers?.length >= 1) ?? [];
  if (!bhs.length) {
    el.style.display = 'block';
    el.innerHTML = '<div style="font-size:9px;color:var(--text-dim);font-style:italic">No borehole data to check against.</div>';
    return;
  }

  const result = scoreConceptCoherence(concept, bhs, AppState.geoUnits);
  if (!result) {
    el.style.display = 'block';
    el.innerHTML = '<div style="font-size:9px;color:var(--text-dim)">Coherence check unavailable.</div>';
    return;
  }

  const gradeCol = result.grade === 'strong' ? '#7fcfb0' : result.grade === 'moderate' ? '#f5a623' : '#e06c75';
  const scoreBar = `<div style="display:inline-block;width:${(result.score*60).toFixed(0)}px;height:5px;background:${gradeCol};border-radius:2px;vertical-align:middle;margin:0 4px"></div>`;

  let html = `<div style="font-size:9px;margin-top:4px;padding:4px 6px;background:rgba(255,255,255,0.04);border-radius:3px;border-left:2px solid ${gradeCol}">
    <div style="font-weight:600;color:${gradeCol};margin-bottom:3px">BH coherence: ${result.grade} ${scoreBar} ${(result.score*100).toFixed(0)}%</div>`;

  for (const d of result.details) {
    html += `<div style="color:var(--text-mid);margin-bottom:2px">• ${escHtml(d)}</div>`;
  }
  for (const s of result.suggestions) {
    html += `<div style="color:#f5a623;margin-bottom:2px;font-style:italic">⚠ ${escHtml(s)}</div>`;
  }
  html += '</div>';

  el.style.display = 'block';
  el.innerHTML = html;
};

function _renderConceptConflicts() {
  const el = document.getElementById('concept-conflicts');
  if (!el) return;
  const conflicts = detectConceptConflicts();
  if (!conflicts.length) {
    el.style.display = 'none';
    el.innerHTML = '';
    return;
  }
  el.style.display = 'block';
  el.innerHTML = conflicts.map(cf => {
    const icon = cf.severity === 'error' ? '⚠' : '△';
    const col  = cf.severity === 'error' ? 'var(--red)' : '#e0a020';
    return `<div class="conflict-row" style="border-left:2px solid ${col};padding:3px 6px;margin-bottom:4px;font-size:9.5px;color:var(--text-mid)">
      <span style="color:${col};margin-right:4px">${icon}</span>${escHtml(cf.description)}
    </div>`;
  }).join('');
}

// 8 grouped axes for radar chart (index into CONCEPT_AXES, sign-weighted mean)
const _RADAR_GROUPS = [
  { label: 'E-W Elongation',  axes: [3, 27],    signs: [1, 1]  },
  { label: 'N-S Elongation',  axes: [4, 27],    signs: [1, -1] },
  { label: 'Channel Form',    axes: [5, 19, 20], signs: [1, 1, 1] },
  { label: 'Continuity',      axes: [9, 8],      signs: [1, -0.5] },
  { label: 'Incision / Dip',  axes: [29, 2, 1],  signs: [1, 1, 1] },
  { label: 'Steps / Faults',  axes: [18, 7, 25], signs: [1, 1, 1] },
  { label: 'Sequence',        axes: [22, 21, 23], signs: [1, 1, 1] },
  { label: 'Complexity',      axes: [31, 24],    signs: [1, 1]  },
];

window._toggleConceptRadar = function(id, btn) {
  const canvas = document.getElementById(`radar-${id}`);
  if (!canvas) return;
  const visible = canvas.style.display !== 'none';
  canvas.style.display = visible ? 'none' : 'block';
  btn.style.opacity = visible ? '' : '1';
  btn.style.color   = visible ? '' : 'var(--accent)';
  if (!visible) {
    const c = AppState.conceptStore?.concepts.find(x => x.id === id);
    if (c) _drawConceptRadar(canvas, c.embedding);
  }
};

function _drawConceptRadar(canvas, embedding) {
  const W = 200, H = 200;
  const cx = W / 2, cy = H / 2;
  const R  = 76;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#f3f5f8';
  ctx.fillRect(0, 0, W, H);

  const n = _RADAR_GROUPS.length;
  const angle = (i) => -Math.PI / 2 + (i / n) * 2 * Math.PI;

  // Compute group values (clamped 0–1 absolute, signed for coloring)
  const groupVals = _RADAR_GROUPS.map(g => {
    let sum = 0, wsum = 0;
    g.axes.forEach((axIdx, k) => {
      const w = 1 / g.axes.length;
      sum  += (embedding[axIdx] ?? 0) * g.signs[k] * w;
      wsum += w;
    });
    return sum; // signed, −1 to +1
  });

  // Grid rings
  ctx.strokeStyle = '#d0d8e4'; ctx.lineWidth = 0.5;
  [0.25, 0.5, 0.75, 1.0].forEach(frac => {
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const a = angle(i);
      const x = cx + Math.cos(a) * R * frac;
      const y = cy + Math.sin(a) * R * frac;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.closePath(); ctx.stroke();
  });

  // Spokes
  ctx.strokeStyle = '#c5cdd8';
  for (let i = 0; i < n; i++) {
    const a = angle(i);
    ctx.beginPath(); ctx.moveTo(cx, cy);
    ctx.lineTo(cx + Math.cos(a) * R, cy + Math.sin(a) * R);
    ctx.stroke();
  }

  // Positive polygon (blue fill)
  const posVals = groupVals.map(v => Math.max(0, v));
  const negVals = groupVals.map(v => Math.max(0, -v));

  const drawPoly = (vals, fillColor, strokeColor) => {
    ctx.beginPath();
    vals.forEach((v, i) => {
      const a = angle(i);
      const r = v * R;
      const x = cx + Math.cos(a) * r;
      const y = cy + Math.sin(a) * r;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.fillStyle = fillColor; ctx.fill();
    ctx.strokeStyle = strokeColor; ctx.lineWidth = 1.5; ctx.stroke();
  };

  drawPoly(posVals, 'rgba(66,114,196,0.25)', 'rgba(66,114,196,0.85)');
  drawPoly(negVals, 'rgba(200,60,60,0.18)', 'rgba(200,60,60,0.6)');

  // Dots at vertices
  groupVals.forEach((v, i) => {
    const a = angle(i);
    const r = Math.abs(v) * R;
    const x = cx + Math.cos(a) * r;
    const y = cy + Math.sin(a) * r;
    ctx.beginPath();
    ctx.arc(x, y, 2.5, 0, Math.PI * 2);
    ctx.fillStyle = v >= 0 ? 'rgba(66,114,196,0.9)' : 'rgba(200,60,60,0.85)';
    ctx.fill();
  });

  // Labels
  ctx.font = '7.5px Inter, sans-serif'; ctx.fillStyle = '#334455';
  ctx.textBaseline = 'middle';
  _RADAR_GROUPS.forEach((g, i) => {
    const a   = angle(i);
    const x   = cx + Math.cos(a) * (R + 14);
    const y   = cy + Math.sin(a) * (R + 14);
    ctx.textAlign = Math.cos(a) > 0.1 ? 'left' : Math.cos(a) < -0.1 ? 'right' : 'center';
    ctx.fillText(g.label, x, y);
  });

  // Centre label
  ctx.fillStyle = '#7a8a9a'; ctx.font = '7px monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText('◎ concept', cx, cy);
}

window._removeConcept = function(id) {
  AppState.conceptStore?.remove(id);
  _renderConceptList();
  _saveConceptStore();
  log(`Concept removed`, 'info');
};

window._highlightConcept3D = function(id, btn) {
  if (!AppState.scene || !AppState.conceptStore) {
    log('Build the 3D model first to highlight concept influence.', 'warn');
    return;
  }
  // Toggle: if already highlighted for this id, reset to unit colors
  if (btn._activeId === id) {
    AppState.scene.resetUnitColors();
    btn._activeId = null;
    btn.style.color = 'var(--text-dim)';
    btn.style.background = 'none';
    log('Concept highlight cleared.', 'info');
    return;
  }
  // Highlight this concept
  const ok = AppState.scene.colorBySingleConcept(id, AppState.conceptStore);
  if (!ok) {
    log('No model built yet — build with Neural Implicit method to see concept volumes.', 'warn');
    return;
  }
  // Clear previous highlight button state
  document.querySelectorAll('.concept-hl-btn').forEach(b => {
    b._activeId = null;
    b.style.color = 'var(--text-dim)';
    b.style.background = 'none';
  });
  btn._activeId = id;
  btn.style.color = 'var(--accent)';
  btn.style.background = 'rgba(var(--accent-rgb,67,133,245),0.15)';
  const c = AppState.conceptStore.concepts.find(x => x.id === id);
  log(`3D: highlighting influence of "${c?.description?.slice(0,50) ?? id}"`, 'ok');
};

window._conceptTemporalInc = function(id) {
  const store = AppState.conceptStore;
  if (!store) return;
  const c = store.concepts.find(x => x.id === id);
  if (!c) return;
  const cur = c.temporalOrder ?? -1;
  store.setTemporalOrder(id, cur + 1);
  _updateConceptTemporalDisplay(id, store.concepts.find(x => x.id === id)?.temporalOrder);
  _saveConceptStore();
  _renderConceptConflicts();
};

window._conceptTemporalDec = function(id) {
  const store = AppState.conceptStore;
  if (!store) return;
  const c = store.concepts.find(x => x.id === id);
  if (!c) return;
  const cur = c.temporalOrder ?? 1;
  const newRank = Math.max(0, cur - 1);
  store.setTemporalOrder(id, newRank);
  _updateConceptTemporalDisplay(id, newRank);
  _saveConceptStore();
  _renderConceptConflicts();
};

window._conceptTemporalClear = function(id) {
  const store = AppState.conceptStore;
  if (!store) return;
  store.setTemporalOrder(id, null);
  _updateConceptTemporalDisplay(id, null);
  _saveConceptStore();
  _renderConceptConflicts();
};

function _updateConceptTemporalDisplay(id, rank) {
  const el = document.getElementById(`trank-${id}`);
  if (el) el.textContent = (rank !== null && rank !== undefined) ? rank : '—';
}

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

  // Coverage density at this (x,y) position from the voxel grid
  let coverageDensityVal = null;
  const grid = AppState.voxelGrid;
  if (grid?.coverageDensity) {
    const { nx, ny, nz, cellSize: cs, cellHeight: ch, origin: O } = grid;
    const ix = Math.floor((worldX - O.x) / cs);
    const iy = Math.floor((worldY - O.z) / cs);
    const iz = Math.floor((worldZ - O.y) / ch);
    if (ix >= 0 && ix < nx && iy >= 0 && iy < ny && iz >= 0 && iz < nz) {
      coverageDensityVal = grid.coverageDensity[ix + iy * nx + iz * nx * ny];
    }
  }

  const tensor = concepts.tensor;
  const trend  = concepts.trend ?? { dz_dxN: 0, dz_dyN: 0 };
  return {
    conceptWeights:    concepts.weights.slice(0, 4),
    bhWeights,
    tensor: {
      Ax:    +tensor.Ax.toFixed(2),
      Ay:    +tensor.Ay.toFixed(2),
      Az:    +tensor.Az.toFixed(2),
      Amaj:  +(tensor.Amaj ?? Math.max(tensor.Ax, tensor.Ay)).toFixed(2),
      Amin:  +(tensor.Amin ?? Math.min(tensor.Ax, tensor.Ay)).toFixed(2),
      theta: +(tensor.theta ?? 0).toFixed(3),
      thetaDeg: +((tensor.theta ?? 0) * 180 / Math.PI).toFixed(1),
    },
    semanticDominance: concepts.totalWeight > 0 ? Math.min(1, concepts.totalWeight) : 0,
    activeAxes:        concepts.activeAxes ?? [],
    trend:             { dz_dxN: trend.dz_dxN, dz_dyN: trend.dz_dyN },
    coverageDensity:   coverageDensityVal,
    sharpnessT:        (() => {
      if (!grid?.sharpnessT) return null;
      const { nx, ny, nz, cellSize: cs, cellHeight: ch, origin: O } = grid;
      const ix = Math.floor((worldX - O.x) / cs);
      const iy = Math.floor((worldY - O.z) / cs);
      const iz = Math.floor((worldZ - O.y) / ch);
      if (ix < 0 || ix >= nx || iy < 0 || iy >= ny || iz < 0 || iz >= nz) return null;
      return grid.sharpnessT[ix + iy * nx + iz * nx * ny];
    })(),
    // 3-component uncertainty decomposition for this voxel
    uncertaintyDecomp: (() => {
      // dataUncertainty: how poorly covered by BH data (1 - coverageDensity)
      const dataUncert = coverageDensityVal != null ? Math.max(0, 1 - coverageDensityVal) : null;

      // modelUncertainty: from network's softmax confidence via certainty field
      let modelUncert = null;
      if (grid?.certainty) {
        const { nx, ny, nz, cellSize: cs, cellHeight: ch, origin: O } = grid;
        const ix = Math.floor((worldX - O.x) / cs);
        const iy = Math.floor((worldY - O.z) / cs);
        const iz = Math.floor((worldZ - O.y) / ch);
        if (ix >= 0 && ix < nx && iy >= 0 && iy < ny && iz >= 0 && iz < nz) {
          modelUncert = Math.max(0, 1 - grid.certainty[ix + iy * nx + iz * nx * ny]);
        }
      }

      // conceptUncertainty: inverse of weighted-average concept confidence at this location
      let conceptUncert = null;
      if (concepts.weights.length > 0 && AppState.conceptStore) {
        const wtdConf = concepts.weights.reduce((sum, w) => {
          const c = AppState.conceptStore.concepts.find(c => c.id === w.id);
          return sum + w.weight * (c?.confidence ?? 0.7);
        }, 0);
        const totalWt = concepts.weights.reduce((s, w) => s + w.weight, 0);
        conceptUncert = totalWt > 0 ? Math.max(0, 1 - wtdConf / totalWt) : 0.3;
      }

      return { dataUncert, modelUncert, conceptUncert };
    })(),
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

// ── Concept sensitivity analysis ──────────────────────────────────────────────
// For each concept, compute the fraction of voxels whose dominant unit would
// change if that concept were removed (concept contribution = how many voxels
// it significantly influences, based on the conceptInfluence array).
// Returns [{conceptId, description, influencedVoxels, pctOfTotal}]
export function computeConceptSensitivity() {
  const grid  = AppState.voxelGrid;
  const store = AppState.conceptStore;
  if (!grid?.conceptInfluence || !store || store.isEmpty) return [];

  const total = grid.nx * grid.ny * grid.nz;
  const { conceptInfluence } = grid;

  return store.concepts.map(c => {
    // Estimate: count voxels where this concept's relevance is >25% of total weight
    let influenced = 0;
    let totalW     = 0;
    const centerX = c.domain?.type === 'bbox'
      ? (c.domain.minX + c.domain.maxX) / 2 : (grid.origin.x + grid.nx * grid.cellSize / 2);
    const centerY = c.domain?.type === 'bbox'
      ? (c.domain.minY + c.domain.maxY) / 2 : (grid.origin.z + grid.ny * grid.cellSize / 2);

    for (let iz = 0; iz < grid.nz; iz++) {
      for (let iy = 0; iy < grid.ny; iy++) {
        const worldY = grid.origin.z + (iy + 0.5) * grid.cellSize;
        for (let ix = 0; ix < grid.nx; ix++) {
          const worldX = grid.origin.x + (ix + 0.5) * grid.cellSize;
          const idx    = ix + iy * grid.nx + iz * grid.nx * grid.ny;
          const inf    = conceptInfluence[idx] ?? 0;
          if (inf < 0.05) continue;
          // Check spatial relevance of this specific concept at this position
          const dx = c.domain?.type === 'bbox' ? Math.max(0, Math.abs(worldX - centerX) - (c.domain.maxX - c.domain.minX) / 2) : 0;
          const dy = c.domain?.type === 'bbox' ? Math.max(0, Math.abs(worldY - centerY) - (c.domain.maxY - c.domain.minY) / 2) : 0;
          const sigma = c.domain?.sigma ?? 50;
          const relevance = c.confidence * Math.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma));
          if (relevance > 0.1) { influenced++; totalW += relevance; }
        }
      }
    }
    return {
      conceptId:       c.id,
      description:     c.description,
      influencedVoxels: influenced,
      pctOfTotal:      total > 0 ? (influenced / total * 100) : 0,
      meanRelevance:   influenced > 0 ? totalW / influenced : 0,
    };
  }).sort((a, b) => b.pctOfTotal - a.pctOfTotal);
}

// ── Show concept sensitivity in panel ────────────────────────────────────────
window._showConceptSensitivity = function() {
  const results = computeConceptSensitivity();
  const el = document.getElementById('concept-sensitivity-output');
  if (!el) return;
  if (!results.length) {
    el.innerHTML = '<div style="font-size:10px;color:var(--text-dim)">Build a model with neural-implicit and concepts first.</div>';
    el.style.display = 'block';
    return;
  }
  el.innerHTML = results.map(r => {
    const pct = r.pctOfTotal.toFixed(1);
    const col = r.pctOfTotal > 30 ? 'var(--accent)' : r.pctOfTotal > 10 ? '#f0b429' : 'var(--text-mid)';
    return `<div class="coherence-row" style="margin-bottom:3px">
      <span class="coherence-name" title="${r.description}">${r.description.slice(0, 32)}…</span>
      <div class="coherence-bar-wrap"><div class="coherence-bar" style="width:${Math.min(100, r.pctOfTotal * 2)}%;background:${col}"></div></div>
      <span class="coherence-pct" style="color:${col}">${pct}%</span>
    </div>`;
  }).join('');
  el.style.display = 'block';
};

// ── Low-influence concept warning ─────────────────────────────────────────────
// After model build: auto-run sensitivity, warn about concepts with <5% influence.
// The warning appears in the concept-conflicts panel (reuses same styling).
function _warnLowInfluenceConcepts() {
  const store = AppState.conceptStore;
  if (!store || store.isEmpty) return;

  // Only available when conceptInfluence grid exists (neural-implicit with concepts)
  const hasSensitivity = !!(AppState.voxelGrid?.conceptInfluence);
  const confEl = document.getElementById('concept-conflicts');
  if (!confEl) return;

  // Run sensitivity if grid is available; otherwise check cosine similarity between concepts
  const sensitivityWarnings = [];

  if (hasSensitivity) {
    const results = computeConceptSensitivity();
    for (const r of results) {
      if (r.pctOfTotal < 3 && r.meanRelevance < 0.05) {
        sensitivityWarnings.push({
          severity: 'warning',
          description: `"${r.description.slice(0, 45)}" influenced only ${r.pctOfTotal.toFixed(1)}% of voxels — check domain, confidence, or opposing concepts`,
        });
      }
    }
  }

  // Pairwise cosine similarity warnings between stored concepts (always available)
  const concepts = store.concepts;
  const simWarnings = [];
  for (let i = 0; i < concepts.length; i++) {
    for (let j = i + 1; j < concepts.length; j++) {
      const sim = ConceptStore.cosineSimilarity(concepts[i].embedding, concepts[j].embedding);
      if (sim >= 0.90) {
        simWarnings.push({
          severity: 'warning',
          description: `"${concepts[i].description.slice(0, 28)}" and "${concepts[j].description.slice(0, 28)}" are ${(sim * 100).toFixed(0)}% similar — consider merging or using different domains`,
        });
      }
    }
  }

  // Always run base conflict detection first (populates the panel)
  _renderConceptConflicts();

  // Then append sensitivity + similarity warnings
  const allWarnings = [...sensitivityWarnings, ...simWarnings];
  if (!allWarnings.length) return;

  const newRows = allWarnings.map(w => {
    const col  = w.severity === 'error' ? 'var(--red)' : '#e0a020';
    const icon = w.severity === 'error' ? '⚠' : '△';
    return `<div class="conflict-row" style="border-left:2px solid ${col};padding:3px 6px;margin-bottom:4px;font-size:9.5px;color:var(--text-mid)">
      <span style="color:${col};margin-right:4px">${icon}</span>${escHtml(w.description)}
    </div>`;
  }).join('');
  confEl.style.display = 'block';
  confEl.innerHTML += newRows;
}

// ── Concept-driven Engineering Hazard Map ────────────────────────────────────
// Derives a 2D engineering hazard probability surface directly from concept
// embeddings — without requiring a 3D model build.
//
// Hazard categories derived from concept axes:
//   Karst/Dissolution:  axis 24 (dissolution_features) + axis 19 (irregular_base)
//   Fault/Structural:   axis  7 (fault_controlled)     + axis 25 (structural_complexity) + axis 18 (stepped_boundary)
//   Ground Instability: axis  5 (channel_morphology)   + axis 29 (incision_depth_ratio)
//   Settlement risk:    –axis 9 (lateral_continuity)  + –axis 0 (horizontal_layering)
//   Data uncertainty:   –axis 26 (data_confidence)
//
// Renders a mini SVG heatmap in the concept panel.
window._showConceptHazardMap = function() {
  const el = document.getElementById('concept-hazard-output');
  if (!el) return;
  el.style.display = 'block';

  const store = AppState.conceptStore;
  if (!store || store.isEmpty) {
    el.innerHTML = '<p class="hint" style="font-size:10px">No concepts encoded — add concepts to generate a hazard map.</p>';
    return;
  }

  // Get site bounds from voxelGrid or borehole extents
  let minX, maxX, minY, maxY;
  const grid = AppState.voxelGrid;
  if (grid) {
    minX = grid.origin.x; maxX = grid.origin.x + grid.nx * grid.cellSize;
    minY = grid.origin.z; maxY = grid.origin.z + grid.ny * grid.cellSize;
  } else {
    const bhs = (AppState.classifiedBH ?? []).filter(b => isFinite(b.x) && isFinite(b.y));
    if (!bhs.length) {
      el.innerHTML = '<p class="hint" style="font-size:10px">No boreholes or model — load data first to define site extent.</p>';
      return;
    }
    minX = Math.min(...bhs.map(b => b.x)); maxX = Math.max(...bhs.map(b => b.x));
    minY = Math.min(...bhs.map(b => b.y)); maxY = Math.max(...bhs.map(b => b.y));
    const pad = Math.max(10, (maxX - minX) * 0.1, (maxY - minY) * 0.1);
    minX -= pad; maxX += pad; minY -= pad; maxY += pad;
  }

  const COLS = 32, ROWS = 24;  // mini raster resolution
  const dx = (maxX - minX) / COLS, dy = (maxY - minY) / ROWS;

  const HAZARD_TYPES = [
    { name: 'Karst/Dissolution', axes: [[24, 1.0], [19, 0.7]], color: '#b35c00' },
    { name: 'Fault/Structure',   axes: [[7,  1.0], [25, 0.6], [18, 0.5]], color: '#e06c75' },
    { name: 'Ground Instability',axes: [[5,  0.8], [29, 0.7]], color: '#d7a020' },
    { name: 'Settlement Risk',   axes: [[-9, 0.7], [-0, 0.5]], color: '#a070d0' },
    { name: 'Data Uncertainty',  axes: [[-26, 1.0]], color: '#507090' },
  ];

  const SVG_W = 200, SVG_H = Math.round(200 * ROWS / COLS);
  const cellW = SVG_W / COLS, cellH = SVG_H / ROWS;

  // For each hazard type, compute 2D grid of hazard scores
  const hazardGrids = HAZARD_TYPES.map(() => new Float32Array(COLS * ROWS));

  for (let row = 0; row < ROWS; row++) {
    const wy = minY + (row + 0.5) * dy;
    for (let col = 0; col < COLS; col++) {
      const wx = minX + (col + 0.5) * dx;
      const ctx = store.computeAt(wx, wy, 0);
      if (!ctx || ctx.totalWeight < 0.05) continue;
      const v = ctx.vec;

      for (let h = 0; h < HAZARD_TYPES.length; h++) {
        let score = 0;
        for (const [axIdx, weight] of HAZARD_TYPES[h].axes) {
          const neg = axIdx < 0;
          const i   = Math.abs(axIdx);
          const val = v[i] ?? 0;
          score += (neg ? -val : val) * weight;
        }
        hazardGrids[h][col + row * COLS] = Math.max(0, Math.min(1, score));
      }
    }
  }

  // Build composite (max hazard per cell) and per-type SVG layers
  const compositeMax = new Float32Array(COLS * ROWS);
  const dominantType = new Uint8Array(COLS * ROWS);
  for (let i = 0; i < COLS * ROWS; i++) {
    let mx = 0, mIdx = 0;
    for (let h = 0; h < HAZARD_TYPES.length; h++) {
      if (hazardGrids[h][i] > mx) { mx = hazardGrids[h][i]; mIdx = h; }
    }
    compositeMax[i] = mx;
    dominantType[i] = mIdx;
  }

  // Helper to convert hex color with opacity
  const hexToRgba = (hex, alpha) => {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r},${g},${b},${alpha.toFixed(2)})`;
  };

  const cells = [];
  for (let row = 0; row < ROWS; row++) {
    for (let col = 0; col < COLS; col++) {
      const i = col + row * COLS;
      const score = compositeMax[i];
      if (score < 0.1) continue;
      const hType = HAZARD_TYPES[dominantType[i]];
      const alpha = Math.min(0.9, score * 0.85 + 0.1);
      const wx    = minX + (col + 0.5) * dx;
      const wy    = minY + (row + 0.5) * dy;
      cells.push(`<rect x="${(col * cellW).toFixed(1)}" y="${(row * cellH).toFixed(1)}" width="${cellW.toFixed(1)}" height="${cellH.toFixed(1)}" fill="${hexToRgba(hType.color, alpha)}" title="${hType.name} at E${wx.toFixed(0)} N${wy.toFixed(0)}: ${(score * 100).toFixed(0)}%"/>`);
    }
  }

  // BH dots
  const bhDots = (AppState.classifiedBH ?? []).filter(b => !b.synthetic && isFinite(b.x) && isFinite(b.y)).map(bh => {
    const cx = ((bh.x - minX) / (maxX - minX) * SVG_W).toFixed(1);
    const cy = ((bh.y - minY) / (maxY - minY) * SVG_H).toFixed(1);
    return `<circle cx="${cx}" cy="${cy}" r="2.5" fill="white" stroke="#222" stroke-width="0.8" opacity="0.9" title="${bh.id}"/>`;
  });

  // Legend
  const legendRows = HAZARD_TYPES.map((h, i) => `
    <div style="display:flex;align-items:center;gap:4px;margin-bottom:2px">
      <div style="width:10px;height:10px;border-radius:2px;background:${h.color};flex-shrink:0;opacity:0.85"></div>
      <span style="font-size:9px;color:var(--text-mid)">${h.name}</span>
    </div>`).join('');

  // Find high-hazard zones summary
  const highHazardCells = Array.from(compositeMax).filter(v => v > 0.5).length;
  const totalCells = COLS * ROWS;
  const pctHigh = (highHazardCells / totalCells * 100).toFixed(0);

  el.innerHTML = `
    <div style="font-size:9.5px;font-weight:600;color:var(--text-mid);margin-bottom:4px">
      Concept-predicted hazard zones <span style="font-weight:400">(no model build required)</span>
    </div>
    <svg width="${SVG_W}" height="${SVG_H}" style="display:block;border-radius:4px;background:var(--bg-deep);border:1px solid var(--border);margin-bottom:4px">
      ${cells.join('')}
      ${bhDots.join('')}
    </svg>
    <div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:4px">${legendRows}</div>
    <div style="font-size:9px;color:var(--text-dim)">
      ${pctHigh}% of site has moderate–high hazard scores from active concepts.
      BH locations shown as white circles.
    </div>`;
};

// ── Concept ensemble uncertainty analysis ────────────────────────────────────
// Runs 3 inference passes with concept scales [0, 1, 1.5] using the cached
// trained neural model. Colors voxels by prediction stability across scales.
// For non-neural methods, shows an explanation and offers the 3-run rebuild.
window._runConceptEnsemble = async function() {
  const el = document.getElementById('concept-ensemble-output');
  if (el) { el.style.display = 'block'; el.innerHTML = '<div style="font-size:10px;color:var(--text-mid)">Running…</div>'; }

  const grid  = AppState.voxelGrid;
  const store = AppState.conceptStore;
  if (!grid) {
    if (el) el.innerHTML = '<div style="font-size:10px;color:#e06c75">Build a 3D model first.</div>';
    return;
  }

  // Neural method: fast re-inference with concept confidence scaling
  if (AppState.trainedModel && store && !store.isEmpty) {
    const gridMeta = {
      nx: grid.nx, ny: grid.ny, nz: grid.nz,
      cellSize: grid.cellSize, cellHeight: grid.cellHeight,
      origin: grid.origin,
    };
    const scales = [0, 1, 1.5];
    const labels = ['Concepts OFF (pure borehole)', 'Baseline (×1.0)', 'Amplified (×1.5)'];
    const runs = [];
    for (let s = 0; s < scales.length; s++) {
      if (el) el.innerHTML = `<div style="font-size:10px;color:var(--text-mid)">Inferring run ${s + 1}/3…</div>`;
      const scaledStore = store.cloneScaled(scales[s]);
      const inferred = inferGeoImplicit(AppState.trainedModel, gridMeta, AppState.geoUnits, scaledStore);
      runs.push(inferred.unitIds);
    }

    // Color by stability
    const stability = AppState.scene.colorByConceptStability(runs);
    if (!stability) {
      if (el) el.innerHTML = '<div style="font-size:10px;color:#e06c75">Color failed — rebuild model first.</div>';
      return;
    }

    // Compute statistics
    const total = stability.length;
    let nStable = 0, nPartial = 0, nUnstable = 0;
    for (let i = 0; i < total; i++) {
      if      (stability[i] >= 0.99) nStable++;
      else if (stability[i] >= 0.67) nPartial++;
      else                            nUnstable++;
    }
    const pStable   = (nStable   / total * 100).toFixed(0);
    const pPartial  = (nPartial  / total * 100).toFixed(0);
    const pUnstable = (nUnstable / total * 100).toFixed(0);

    if (el) el.innerHTML = `
      <div style="font-size:9.5px;color:var(--text-mid);margin-bottom:6px">
        3 runs: ${labels.join(' · ')}<br>
        <b style="color:var(--text)">Voxel stability across concept scales:</b>
      </div>
      <div class="coherence-row">
        <span class="coherence-name" style="color:#5ab97d">Stable (all 3 agree)</span>
        <div class="coherence-bar-wrap"><div class="coherence-bar" style="width:${pStable}%;background:#5ab97d"></div></div>
        <span class="coherence-pct" style="color:#5ab97d">${pStable}%</span>
      </div>
      <div class="coherence-row">
        <span class="coherence-name" style="color:#e6b84a">Partial (2/3 agree)</span>
        <div class="coherence-bar-wrap"><div class="coherence-bar" style="width:${pPartial}%;background:#e6b84a"></div></div>
        <span class="coherence-pct" style="color:#e6b84a">${pPartial}%</span>
      </div>
      <div class="coherence-row">
        <span class="coherence-name" style="color:#e06c75">Unstable (no consensus)</span>
        <div class="coherence-bar-wrap"><div class="coherence-bar" style="width:${pUnstable}%;background:#e06c75"></div></div>
        <span class="coherence-pct" style="color:#e06c75">${pUnstable}%</span>
      </div>
      <div style="font-size:9px;color:var(--text-dim);margin-top:4px">
        Green = borehole-anchored, safe to rely on.<br>
        Red = concept-driven, interpretation-dependent.
      </div>`;
  } else if (!AppState.trainedModel) {
    // Non-neural method: show analytical sensitivity proxy
    if (!grid.certainty || !grid.conceptInfluence) {
      if (el) el.innerHTML = `<div style="font-size:10px;color:var(--text-mid)">
        Switch to <b>neural-implicit</b> method and build model to enable ensemble analysis.
        The cached trained network enables instant re-inference at different concept strengths.
      </div>`;
      return;
    }
    // Proxy: sensitivity = conceptInfluence × (1 - certainty) × (1 - coverage)
    const { certainty: cert, conceptInfluence: ci, coverageDensity: cov } = grid;
    const total = cert.length;
    const stability = new Float32Array(total);
    for (let i = 0; i < total; i++) {
      const sens = (ci?.[i] ?? 0) * (1 - cert[i]) * (1 - (cov?.[i] ?? 0));
      stability[i] = 1 - Math.min(1, sens * 3);
    }
    AppState.scene.colorByConceptStability([
      // Simulate 3 runs by slight blurring of existing unitIds
      grid.unitIds,
      // Proxy "run with amplified concepts" via blendUnitIds in concept-influenced voxels
      new Uint8Array(total).map((_, i) => ci?.[i] > 0.4 ? (grid.blendUnitIds[i] || grid.unitIds[i]) : grid.unitIds[i]),
      grid.unitIds,
    ]);
    if (el) el.innerHTML = `<div style="font-size:10px;color:var(--text-mid)">
      <b>Concept sensitivity proxy</b> — shown on model (green=stable, red=concept-sensitive).<br>
      For full ensemble use <b>neural-implicit</b> method.
    </div>`;
  }
};

// ── Concept axis correlation matrix ──────────────────────────────────────────
// Renders a 32×32 heatmap on #concept-corr-canvas showing how geological axes
// co-vary across the active concept store.
// r[i][j] = Pearson correlation of axis-i vs axis-j values across all concepts.
function _drawConceptCorrelationMatrix() {
  const canvas = document.getElementById('concept-corr-canvas');
  if (!canvas) return;
  const ctx2 = canvas.getContext('2d');
  const store = AppState.conceptStore;
  if (!store || store.isEmpty) {
    ctx2.clearRect(0, 0, canvas.width, canvas.height);
    ctx2.fillStyle = '#2a3848';
    ctx2.fillRect(0, 0, canvas.width, canvas.height);
    ctx2.fillStyle = '#4a6275';
    ctx2.font = '11px Inter, sans-serif';
    ctx2.textAlign = 'center';
    ctx2.fillText('Add concepts to see correlation', canvas.width / 2, canvas.height / 2);
    return;
  }

  const concepts = store.concepts;
  const DIM = 32;
  const N = concepts.length;
  if (N < 2) {
    ctx2.clearRect(0, 0, canvas.width, canvas.height);
    ctx2.fillStyle = '#2a3848';
    ctx2.fillRect(0, 0, canvas.width, canvas.height);
    ctx2.fillStyle = '#4a6275';
    ctx2.font = '11px Inter, sans-serif';
    ctx2.textAlign = 'center';
    ctx2.fillText('Add ≥2 concepts to see correlation', canvas.width / 2, canvas.height / 2);
    return;
  }

  // Build DIM×DIM correlation matrix
  // Each concept contributes one sample vector of length 32
  const means = new Float32Array(DIM);
  for (const c of concepts) for (let i = 0; i < DIM; i++) means[i] += c.embedding[i];
  for (let i = 0; i < DIM; i++) means[i] /= N;

  const corr = new Float32Array(DIM * DIM);
  for (let i = 0; i < DIM; i++) {
    for (let j = i; j < DIM; j++) {
      let num = 0, si = 0, sj = 0;
      for (const c of concepts) {
        const di = c.embedding[i] - means[i];
        const dj = c.embedding[j] - means[j];
        num += di * dj;
        si  += di * di;
        sj  += dj * dj;
      }
      const r = (si > 1e-9 && sj > 1e-9) ? num / Math.sqrt(si * sj) : (i === j ? 1 : 0);
      corr[i * DIM + j] = r;
      corr[j * DIM + i] = r;
    }
  }

  const W = canvas.width, H = canvas.height;
  const MARGIN = 18;
  const cellW = (W - MARGIN) / DIM;
  const cellH = (H - MARGIN) / DIM;

  ctx2.clearRect(0, 0, W, H);
  ctx2.fillStyle = '#1a2230';
  ctx2.fillRect(0, 0, W, H);

  // Draw cells
  for (let i = 0; i < DIM; i++) {
    for (let j = 0; j < DIM; j++) {
      const r = corr[i * DIM + j];
      let color;
      if (r >= 0) {
        const t = r;
        color = `rgb(${Math.round(20 + t * 180)},${Math.round(30 + t * 30)},${Math.round(30)})`;
      } else {
        const t = -r;
        color = `rgb(${Math.round(30)},${Math.round(30 + t * 30)},${Math.round(20 + t * 180)})`;
      }
      ctx2.fillStyle = color;
      ctx2.fillRect(MARGIN + j * cellW, i * cellH, Math.ceil(cellW + 0.3), Math.ceil(cellH + 0.3));
    }
  }

  // Axis tick labels (every 4th axis, abbreviated)
  ctx2.fillStyle = '#8898a8';
  ctx2.font = `${Math.max(5, Math.min(7, cellW * 0.85))}px Inter, sans-serif`;
  ctx2.textAlign = 'right';
  for (let i = 0; i < DIM; i += 4) {
    const label = CONCEPT_AXES[i]?.slice(0, 6) ?? String(i);
    const y = i * cellH + cellH * 0.5 + 3;
    ctx2.fillText(label, MARGIN - 1, y);
  }
  ctx2.textAlign = 'center';
  ctx2.save();
  for (let j = 0; j < DIM; j += 4) {
    const label = CONCEPT_AXES[j]?.slice(0, 6) ?? String(j);
    const x = MARGIN + j * cellW + cellW * 0.5;
    ctx2.save();
    ctx2.translate(x, DIM * cellH + 2);
    ctx2.rotate(-Math.PI / 2);
    ctx2.fillText(label, 0, 0);
    ctx2.restore();
  }
  ctx2.restore();

  // Color scale legend
  const legEl = document.getElementById('concept-corr-legend');
  if (legEl) legEl.textContent = `${N} concepts · red=positive · blue=negative correlation across axes`;
}

// ── Concept similarity map — 2D PCA of 32-dim embeddings ─────────────────────
// Projects each concept's effective embedding (with inheritance) to 2D via PCA.
// Points close together = semantically similar concepts.
function _drawConceptManifold() {
  const canvas = document.getElementById('concept-manifold-canvas');
  const tooltip = document.getElementById('concept-manifold-tooltip');
  if (!canvas) return;
  const store = AppState.conceptStore;
  const ctx2  = canvas.getContext('2d');
  ctx2.clearRect(0, 0, canvas.width, canvas.height);
  ctx2.fillStyle = '#192330';
  ctx2.fillRect(0, 0, canvas.width, canvas.height);

  if (!store || store.isEmpty) {
    ctx2.fillStyle = '#4a6275';
    ctx2.font = '11px Inter, sans-serif';
    ctx2.textAlign = 'center';
    ctx2.fillText('No concepts encoded', canvas.width / 2, canvas.height / 2);
    return;
  }

  const concepts = store.concepts;
  const DIM = 32;

  // Collect effective embeddings (with inheritance)
  const vecs = concepts.map(c => Array.from(store._effectiveEmbedding(c)));

  // ── 2D PCA: compute covariance, find top 2 eigenvectors by power iteration ──
  const N = vecs.length;
  if (N < 2) {
    ctx2.fillStyle = '#4a6275';
    ctx2.font = '11px Inter, sans-serif';
    ctx2.textAlign = 'center';
    ctx2.fillText('Need ≥ 2 concepts for PCA', canvas.width / 2, canvas.height / 2);
    return;
  }

  // Mean-centre the data
  const mean = new Array(DIM).fill(0);
  for (const v of vecs) for (let i = 0; i < DIM; i++) mean[i] += v[i] / N;
  const centred = vecs.map(v => v.map((x, i) => x - mean[i]));

  // Power iteration to find first 2 eigenvectors of covariance matrix
  function powerIter(data, iters = 50) {
    let v = new Array(DIM).fill(0); v[0] = 1;
    for (let it = 0; it < iters; it++) {
      const next = new Array(DIM).fill(0);
      for (const x of data) {
        const dot = x.reduce((s, xi, i) => s + xi * v[i], 0);
        for (let i = 0; i < DIM; i++) next[i] += dot * x[i];
      }
      const norm = Math.sqrt(next.reduce((s, x) => s + x * x, 0)) + 1e-9;
      v = next.map(x => x / norm);
    }
    return v;
  }

  const pc1 = powerIter(centred);
  // Deflate data for PC2
  const deflated = centred.map(x => {
    const d = x.reduce((s, xi, i) => s + xi * pc1[i], 0);
    return x.map((xi, i) => xi - d * pc1[i]);
  });
  const pc2 = powerIter(deflated);

  // Project
  const proj = centred.map(x => [
    x.reduce((s, xi, i) => s + xi * pc1[i], 0),
    x.reduce((s, xi, i) => s + xi * pc2[i], 0),
  ]);

  const xs = proj.map(p => p[0]), ys = proj.map(p => p[1]);
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const yMin = Math.min(...ys), yMax = Math.max(...ys);
  const xRange = xMax - xMin || 1, yRange = yMax - yMin || 1;

  const PAD = 20;
  const toCanv = (px, py) => [
    PAD + (px - xMin) / xRange * (canvas.width  - PAD * 2),
    PAD + (py - yMin) / yRange * (canvas.height - PAD * 2),
  ];

  // Draw axis labels
  ctx2.fillStyle = '#3a4a5a';
  ctx2.font = '9px Inter, sans-serif';
  ctx2.textAlign = 'left';
  ctx2.fillText('PC1 →', 4, canvas.height - 5);
  ctx2.save(); ctx2.translate(12, canvas.height - 20);
  ctx2.rotate(-Math.PI / 2);
  ctx2.fillText('PC2 ↑', 0, 0);
  ctx2.restore();

  // Draw concept points with golden-angle hue per index
  const pointData = [];
  concepts.forEach((c, ci) => {
    const [cx, cy] = toCanv(proj[ci][0], proj[ci][1]);
    const hue = (ci * 137.508 % 360) / 360;
    ctx2.beginPath();
    ctx2.arc(cx, cy, 6, 0, Math.PI * 2);
    ctx2.fillStyle = `hsl(${(hue * 360).toFixed(0)},80%,58%)`;
    ctx2.fill();
    // Draw inheritance arrow if parent
    if (c.parentId) {
      const pi = concepts.findIndex(p => p.id === c.parentId);
      if (pi >= 0) {
        const [px, py] = toCanv(proj[pi][0], proj[pi][1]);
        ctx2.strokeStyle = 'rgba(255,255,255,0.25)';
        ctx2.lineWidth = 1;
        ctx2.setLineDash([3, 3]);
        ctx2.beginPath(); ctx2.moveTo(px, py); ctx2.lineTo(cx, cy); ctx2.stroke();
        ctx2.setLineDash([]);
      }
    }
    pointData.push({ cx, cy, concept: c });
  });

  // Short labels (first 12 chars)
  ctx2.font = '8px Inter, sans-serif';
  concepts.forEach((c, ci) => {
    const [cx, cy] = toCanv(proj[ci][0], proj[ci][1]);
    ctx2.fillStyle = 'rgba(180,200,220,0.9)';
    ctx2.textAlign = 'center';
    ctx2.fillText(c.description.slice(0, 14), cx, cy - 9);
  });

  // Hover: show full concept name
  canvas.onmousemove = e => {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const hit = pointData.find(p => Math.hypot(p.cx - mx, p.cy - my) < 10);
    if (tooltip) tooltip.textContent = hit
      ? `${hit.concept.description} (conf ${(hit.concept.confidence * 100).toFixed(0)}%)`
      : '';
  };
  canvas.onmouseleave = () => { if (tooltip) tooltip.textContent = ''; };
}

// ── Concept interpretation narrative ─────────────────────────────────────────
// Generates a report-ready paragraph describing what the conceptual model predicts,
// suitable for inclusion in a ground investigation report or factual statement.
function _generateConceptNarrative() {
  const store = AppState.conceptStore;
  if (!store || store.isEmpty) return 'No concepts encoded in the conceptual model.';

  const concepts = store.concepts;
  const grid = AppState.voxelGrid;
  const hasModel = !!(grid?.conceptInfluence);
  const coherenceResults = hasModel ? computeConceptCoherence() : [];
  const sensitivityResults = hasModel ? computeConceptSensitivity() : [];

  const siteName = AppState.siteContext?.name ?? 'the site';
  const nBHs = (AppState.classifiedBH ?? []).filter(b => !b.synthetic).length;

  const parts = [];

  // Opening
  parts.push(
    `The three-dimensional ground model for ${siteName} incorporates ${concepts.length} geological concept${concepts.length > 1 ? 's' : ''} ` +
    `encoded as dense 32-axis semantic embeddings that geometrically shape the neural implicit field, ` +
    `complementing ${nBHs} borehole observation${nBHs !== 1 ? 's' : ''}.`
  );

  // Per-concept description
  const conceptParts = concepts.map((c, i) => {
    const conf = (c.confidence * 100).toFixed(0);
    const emb = c.embedding;

    // Derive key geometric properties from embedding
    const geomDescriptions = [];
    const ew = emb[3] ?? 0, ns = emb[4] ?? 0;
    const chan = emb[5] ?? 0, dome = emb[6] ?? 0;
    const horiz = emb[0] ?? 0, stepped = emb[18] ?? 0, fault = emb[7] ?? 0;
    const vert = emb[29] ?? 0, erosional = emb[8] ?? 0;
    const deepE = emb[14] ?? 0, deepW = emb[15] ?? 0, deepN = emb[16] ?? 0, deepS = emb[17] ?? 0;

    if (ew > 0.5 && ew > ns + 0.2) geomDescriptions.push('E-W elongated geometry');
    else if (ns > 0.5 && ns > ew + 0.2) geomDescriptions.push('N-S elongated geometry');
    else if (ew > 0.4 && ns > 0.4) geomDescriptions.push('NE-SW to NW-SE elongated geometry');
    if (chan > 0.6) geomDescriptions.push(`concave-up channel morphology${vert > 0.5 ? ' with significant incision' : ''}`);
    if (dome > 0.6) geomDescriptions.push('convex-up dome or anticline form');
    if (horiz > 0.6) geomDescriptions.push('laterally continuous bedded geometry');
    if (stepped > 0.6 || fault > 0.7) geomDescriptions.push('stepped fault-controlled contact');
    if (erosional > 0.6) geomDescriptions.push('erosional basal contact');
    if (deepE > 0.4) geomDescriptions.push('surface deepening to east');
    else if (deepW > 0.4) geomDescriptions.push('surface deepening to west');
    if (deepN > 0.4) geomDescriptions.push('surface deepening to north');
    else if (deepS > 0.4) geomDescriptions.push('surface deepening to south');

    const domainStr = c.domain?.type === 'bbox'
      ? ' with a spatially restricted horizontal domain'
      : '';
    const depthStr = (c.domain?.minZ !== undefined || c.domain?.maxZ !== undefined)
      ? ` in the depth range ${c.domain.minZ ?? '?'} to ${c.domain.maxZ ?? '?'} m AOD`
      : '';
    const affStr = c.unitAffinity?.length ? ` applied to ${c.unitAffinity.join(', ')} deposits` : '';
    const tempStr = c.temporalOrder !== null && c.temporalOrder !== undefined
      ? ` (geological age rank ${c.temporalOrder})` : '';

    // Coherence and sensitivity for this concept
    const coh = coherenceResults.find(r => concepts.some(cc => cc.id === c.id));
    const sen = sensitivityResults.find(r => r.conceptId === c.id);
    const infStr = sen ? ` and influenced ${sen.pctOfTotal.toFixed(0)}% of model voxels` : '';

    const geomStr = geomDescriptions.length
      ? `characterised by ${geomDescriptions.join(', ')}`
      : 'encoded geometry';

    return `(${i + 1}) ${c.description.slice(0, 80)} — ` +
      `a ${geomStr}${affStr}${depthStr}${domainStr}${tempStr}, ` +
      `with a confidence weighting of ${conf}%${infStr}.`;
  });

  parts.push('The conceptual model comprises: ' + conceptParts.join(' '));

  // Temporal ordering
  const tempPairs = store.temporallyOrderedPairs?.() ?? [];
  if (tempPairs.length) {
    const tStr = tempPairs.map(({ younger, older }) =>
      `${younger.description.slice(0, 40)} is younger than ${older.description.slice(0, 40)}`
    ).join('; ');
    parts.push(`Geological time ordering constraints were applied: ${tStr}. These relationships injected synthetic training samples to enforce the stratigraphic sequence in data-sparse regions.`);
  }

  // Post-build coherence
  if (hasModel && coherenceResults.length) {
    const matched = coherenceResults.filter(r => r.conceptMatch >= 0.5);
    const mismatched = coherenceResults.filter(r => r.conceptMatch < 0.5);
    if (matched.length) {
      parts.push(
        `Concept-geometry verification confirmed that ${matched.length} geological unit${matched.length > 1 ? 's' : ''} (` +
        matched.map(r => r.unitCode).join(', ') +
        `) exhibited geometry consistent with the encoded conceptual model (match score ≥50%). ` +
        (mismatched.length ? `${mismatched.length} unit${mismatched.length > 1 ? 's' : ''} showed geometry that diverged from concept predictions, suggesting the borehole data provides strong geometric constraints that override the conceptual model in those areas.` : '')
      );
    }
  }

  // Conflict warnings
  const conflicts = detectConceptConflicts();
  if (conflicts.filter(c => c.severity === 'error').length) {
    parts.push(`Note: ${conflicts.filter(c => c.severity === 'error').length} concept conflict${conflicts.filter(c => c.severity === 'error').length > 1 ? 's were' : ' was'} detected during model build. These should be reviewed before using the model for engineering decisions.`);
  }

  return parts.join('\n\n');
}

// Wire concept narrative button
document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('btn-concept-narrative')?.addEventListener('click', () => {
    const store = AppState.conceptStore;
    if (!store || store.isEmpty) { log('Encode at least one concept first.', 'warn'); return; }
    const narrative = _generateConceptNarrative();
    const el = document.getElementById('concept-narrative-output');
    const copyBtn = document.getElementById('btn-copy-narrative');
    if (el) {
      el.style.display = 'block';
      el.textContent = narrative;
    }
    if (copyBtn) {
      copyBtn.style.display = 'inline-block';
      copyBtn.onclick = () => {
        navigator.clipboard?.writeText(narrative).then(() => { copyBtn.textContent = '✓ Copied'; setTimeout(() => { copyBtn.textContent = '⧉ Copy to clipboard'; }, 1500); });
      };
    }
    log('Concept interpretation narrative generated', 'ok');
  });
});

// Internal alias for traceability (same as exported getVoxelAttribution)
function _computeAttribution(worldX, worldY, worldZ) {
  return getVoxelAttribution(worldX, worldY, worldZ);
}

// Numerically differentiate the neural implicit field w.r.t. each concept axis.
// For the voxel at (wx,wy,wz), perturbs each concept axis by +DELTA and measures
// the change in P(dominantUnit). Returns Float32Array(32) of ∂P/∂axis_i values.
function _computeVoxelSensitivity(wx, wy, wz) {
  const trained = AppState.trainedModel;
  if (!trained?.net || !trained.fourierEnc) return null;
  const { net, fourierEnc, bounds, warpedBounds, unitCodes, CONCEPT_DIM } = trained;
  const cdim  = CONCEPT_DIM ?? 32;
  const nIn   = fourierEnc.outDim + cdim;
  const store = AppState.conceptStore;

  const ctx     = store?.computeAt?.(wx, wy, wz) ?? null;
  const ctxVec  = ctx?.vec ?? new Float32Array(cdim);
  const tensor  = ctx?.tensor ?? { Ax: 1, Ay: 1, Az: 1, Amaj: 1, Amin: 1, theta: 0, cosT: 1, sinT: 0 };
  const useB    = store ? computeWarpedBounds(bounds, tensor) : (warpedBounds ?? bounds);

  const warped  = warpPoint(wx, wy, wz, tensor);
  // Handle dip trend if present (mirrors inferGeoImplicit logic)
  let wz2 = warped.z;
  const iTrend = ctx?.trend;
  if (iTrend && (Math.abs(iTrend.dz_dxN) > 0.005 || Math.abs(iTrend.dz_dyN) > 0.005)) {
    const xN = 2 * (warped.x - useB.minX) / Math.max(1e-6, useB.maxX - useB.minX) - 1;
    const yN = 2 * (warped.y - useB.minY) / Math.max(1e-6, useB.maxY - useB.minY) - 1;
    wz2 += iTrend.dz_dxN * xN + iTrend.dz_dyN * yN;
  }

  const pos = fourierEnc.encode(warped.x, warped.y, wz2, useB);

  // Base prediction
  const baseInp = new Float32Array(nIn);
  baseInp.set(pos);
  for (let i = 0; i < cdim; i++) baseInp[fourierEnc.outDim + i] = ctxVec[i] ?? 0;
  let baseResult;
  try { baseResult = net.forward(baseInp, 0); } catch { return null; }
  const baseProbs  = baseResult.probs;
  let domIdx = 0;
  for (let u = 1; u < baseProbs.length; u++) { if (baseProbs[u] > baseProbs[domIdx]) domIdx = u; }
  const basePDom   = baseProbs[domIdx];

  // Perturb each axis
  const DELTA = 0.2;
  const sensitivity = new Float32Array(Math.min(32, cdim));
  for (let i = 0; i < sensitivity.length; i++) {
    const pertInp = new Float32Array(nIn);
    pertInp.set(pos);
    for (let j = 0; j < cdim; j++) pertInp[fourierEnc.outDim + j] = ctxVec[j] ?? 0;
    pertInp[fourierEnc.outDim + i] = Math.min(1, (ctxVec[i] ?? 0) + DELTA);
    try {
      const pertProbs = net.forward(pertInp, 0).probs;
      sensitivity[i] = (pertProbs[domIdx] - basePDom) / DELTA;
    } catch { sensitivity[i] = 0; }
  }

  return { sensitivity, dominantUnitCode: unitCodes[domIdx] ?? null, basePDom };
}

function _renderAttribution(attr, unitCode) {
  if (!attr) return '';
  const { conceptWeights, bhWeights, tensor, semanticDominance, activeAxes, trend, coverageDensity, sharpnessT } = attr;

  const semPct = (semanticDominance * 100).toFixed(0);
  const datPct = Math.max(0, 100 - semPct).toFixed(0);
  const covPct = coverageDensity != null ? (coverageDensity * 100).toFixed(0) : null;

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

  const covBar = covPct != null
    ? `<div class="trace-cov-bar-wrap" title="Borehole coverage density">
        <div class="trace-cov-fill" style="width:${covPct}%;background:hsl(${covPct * 1.2},75%,38%)"></div>
      </div><span class="trace-weight">${covPct}%</span>`
    : '';

  // Plain-language geometry narrative from active axes + warp
  const narrativeParts = [];
  const ax   = +tensor.Ax, ay = +tensor.Ay, az = +tensor.Az;
  const amaj = tensor.Amaj ?? Math.max(ax, ay);
  const tDeg = tensor.thetaDeg ?? 0;
  if (amaj > 1.5) {
    // Describe elongation direction from theta in compass terms
    const dirs = ['E-W', 'NE-SW', 'N-S', 'NW-SE'];
    const dirIdx = Math.round(((tDeg % 180) + 180) / 45) % 4;
    narrativeParts.push(`Bodies elongated ${dirs[dirIdx]} (×${amaj.toFixed(1)})`);
  }
  if (az < 0.7) narrativeParts.push(`Sharp vertical contacts`);
  else if (az > 1.5) narrativeParts.push(`Gradational vertical contacts`);
  if (activeAxes?.some(a => a.name === 'channel_morphology' && a.value > 0.4)) narrativeParts.push('Channel geometry active');
  if (activeAxes?.some(a => a.name === 'stepped_boundary' && a.value > 0.4)) narrativeParts.push(`Stepped contact active${sharpnessT != null && sharpnessT < 0.85 ? ` (sharpened T=${sharpnessT.toFixed(2)})` : ''}`);
  if (activeAxes?.some(a => a.name === 'erosional_contact' && a.value > 0.4)) narrativeParts.push('Erosional base predicted');
  const trendLines = [];
  if (trend && Math.abs(trend.dz_dxN) > 0.01) trendLines.push(`dips ${trend.dz_dxN > 0 ? 'E' : 'W'}`);
  if (trend && Math.abs(trend.dz_dyN) > 0.01) trendLines.push(`dips ${trend.dz_dyN > 0 ? 'N' : 'S'}`);
  if (trendLines.length) narrativeParts.push(`Surface ${trendLines.join(' + ')}`);
  const narrative = narrativeParts.length
    ? `<div class="trace-narrative">${narrativeParts.join(' · ')}</div>`
    : '';

  // Uncertainty decomposition section
  const ud = attr.uncertaintyDecomp;
  const uncertSection = (ud && (ud.dataUncert != null || ud.modelUncert != null || ud.conceptUncert != null)) ? (() => {
    const fmt = v => v != null ? `${(v * 100).toFixed(0)}%` : '—';
    const bar = (v, col) => v != null
      ? `<div style="display:inline-block;width:${(v * 50).toFixed(0)}px;height:5px;border-radius:2px;background:${col};vertical-align:middle;margin:0 4px 0 2px"></div>`
      : '';
    return `<div class="trace-section">
      <div class="trace-section-hdr">Uncertainty decomposition</div>
      <div class="trace-row"><span class="trace-label">Data</span>${bar(ud.dataUncert,'hsl(30,80%,55%)')}<span class="trace-weight" style="color:hsl(30,80%,55%)">${fmt(ud.dataUncert)}</span></div>
      <div class="trace-row"><span class="trace-label">Model</span>${bar(ud.modelUncert,'hsl(200,70%,55%)')}<span class="trace-weight" style="color:hsl(200,70%,55%)">${fmt(ud.modelUncert)}</span></div>
      <div class="trace-row"><span class="trace-label">Concept</span>${bar(ud.conceptUncert,'hsl(280,60%,60%)')}<span class="trace-weight" style="color:hsl(280,60%,60%)">${fmt(ud.conceptUncert)}</span></div>
    </div>`;
  })() : '';

  return `
    <div class="trace-section">
      <div class="trace-section-hdr">Semantic influence: ${semPct}% · Data: ${datPct}%
        ${covPct != null ? `· Coverage: ${covBar}` : ''}
      </div>
      ${narrative}
    </div>
    ${uncertSection}
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
      <div class="trace-warp">Major ×${(tensor.Amaj ?? Math.max(tensor.Ax, tensor.Ay)).toFixed(2)} at ${(tensor.thetaDeg ?? 0).toFixed(0)}° · Minor ×${(tensor.Amin ?? Math.min(tensor.Ax, tensor.Ay)).toFixed(2)} · Z ×${tensor.Az}</div>
      ${trend && (Math.abs(trend.dz_dxN) > 0.01 || Math.abs(trend.dz_dyN) > 0.01) ? `<div class="trace-warp" style="margin-top:2px;color:var(--text-mid)">Depth trend: E ${trend.dz_dxN >= 0 ? '↘' : '↗'} ${Math.abs(trend.dz_dxN).toFixed(3)} · N ${trend.dz_dyN >= 0 ? '↘' : '↗'} ${Math.abs(trend.dz_dyN).toFixed(3)}</div>` : ''}
      ${sharpnessT != null && sharpnessT < 0.95 ? `<div class="trace-warp" style="margin-top:2px;color:#e8a020">Contact sharpness T=${sharpnessT.toFixed(2)}${sharpnessT < 0.5 ? ' (sharp/stepped boundary)' : ' (slightly sharpened)'}</div>` : ''}
    </div>
    ${AppState.trainedModel ? `<div class="trace-section">
      <div class="trace-section-hdr">Neural Sensitivity <span style="font-weight:400;font-size:9px;color:var(--text-dim)">∂P(unit)/∂axis</span></div>
      <div id="trace-sensitivity-content" style="font-size:9.5px;color:var(--text-dim)">
        <button style="font-size:9px;padding:2px 6px;border:1px solid var(--border);border-radius:3px;background:var(--bg-el);color:var(--text);cursor:pointer" id="btn-trace-sensitivity">
          ⟳ Scan 32 axes
        </button>
      </div>
    </div>` : ''}`;
}

// ── Model QC Dashboard ───────────────────────────────────────────────────────
// Summarises model quality metrics in the right panel after every build.
// Shows: mean certainty per unit, concept vs data influence breakdown,
// coverage hole percentage, and geometry match score.
function _renderModelQC() {
  const panel   = document.getElementById('model-qc-panel');
  const content = document.getElementById('model-qc-content');
  if (!panel || !content) return;
  const grid  = AppState.voxelGrid;
  if (!grid) return;

  const { nx, ny, nz, unitIds, certainty, conceptInfluence, coverageDensity } = grid;
  const total = nx * ny * nz;

  // Per-unit certainty stats
  const unitStats = {};
  for (const u of AppState.geoUnits) unitStats[u.id] = { sum: 0, cnt: 0, code: u.code, color: u.color };
  let sumCert = 0, certCnt = 0;
  let conceptVoxels = 0, dataVoxels = 0;
  let coverHoles = 0;

  for (let idx = 0; idx < total; idx++) {
    const uid = unitIds[idx];
    if (!uid) continue;
    const cert = certainty[idx];
    sumCert += cert; certCnt++;
    if (unitStats[uid]) { unitStats[uid].sum += cert; unitStats[uid].cnt++; }
    if (conceptInfluence) {
      if (conceptInfluence[idx] > 0.5) conceptVoxels++;
      else dataVoxels++;
    }
    if (coverageDensity && coverageDensity[idx] < 0.2) coverHoles++;
  }

  const meanCert     = certCnt > 0 ? (sumCert / certCnt * 100).toFixed(0) : '—';
  const certColor    = sumCert / certCnt > 0.65 ? 'var(--accent)' : sumCert / certCnt > 0.45 ? '#d4a843' : 'var(--red)';
  const conceptPct   = certCnt > 0 && conceptInfluence ? (conceptVoxels / certCnt * 100).toFixed(0) : '—';
  const dataPct      = certCnt > 0 && conceptInfluence ? (dataVoxels   / certCnt * 100).toFixed(0) : '—';
  const holePct      = certCnt > 0 && coverageDensity  ? (coverHoles   / certCnt * 100).toFixed(0) : '—';

  // Geometry match score from last concept check
  const geoCheck = AppState._lastGeoCheck ?? [];
  const matchScore = geoCheck.length
    ? (geoCheck.reduce((s, r) => s + r.conceptMatch, 0) / geoCheck.length * 100).toFixed(0)
    : null;

  // Unit rows
  const unitRows = AppState.geoUnits
    .map(u => {
      const s = unitStats[u.id];
      if (!s || s.cnt === 0) return '';
      const pct = (s.sum / s.cnt * 100).toFixed(0);
      const col = s.sum / s.cnt > 0.65 ? 'var(--accent)' : s.sum / s.cnt > 0.45 ? '#d4a843' : 'var(--red)';
      return `<div style="display:flex;align-items:center;gap:5px;margin-bottom:3px">
        <span style="width:8px;height:8px;border-radius:2px;background:${u.color};flex-shrink:0"></span>
        <span style="flex:1;font-size:9.5px;color:var(--text-mid)">${u.code}</span>
        <div style="width:60px;height:5px;background:var(--bg-deep);border-radius:2px;overflow:hidden">
          <div style="width:${pct}%;height:100%;background:${col}"></div>
        </div>
        <span style="font-size:9px;color:${col};font-family:var(--font-mono);min-width:22px;text-align:right">${pct}%</span>
      </div>`;
    }).join('');

  let html = `
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:8px">
      <div style="background:var(--bg-surface);border:1px solid var(--border);border-radius:4px;padding:5px;text-align:center">
        <div style="font-size:18px;font-weight:700;color:${certColor}">${meanCert}%</div>
        <div style="font-size:9px;color:var(--text-dim)">Mean Certainty</div>
      </div>
      ${conceptPct !== '—' ? `
      <div style="background:var(--bg-surface);border:1px solid var(--border);border-radius:4px;padding:5px;text-align:center">
        <div style="font-size:18px;font-weight:700;color:var(--accent)">${conceptPct}%</div>
        <div style="font-size:9px;color:var(--text-dim)">Concept-driven</div>
      </div>` : ''}
      ${holePct !== '—' ? `
      <div style="background:var(--bg-surface);border:1px solid var(--border);border-radius:4px;padding:5px;text-align:center">
        <div style="font-size:18px;font-weight:700;color:${parseFloat(holePct) > 30 ? 'var(--red)' : '#d4a843'}">${holePct}%</div>
        <div style="font-size:9px;color:var(--text-dim)">Coverage holes</div>
      </div>` : ''}
      ${matchScore !== null ? `
      <div style="background:var(--bg-surface);border:1px solid var(--border);border-radius:4px;padding:5px;text-align:center">
        <div style="font-size:18px;font-weight:700;color:${parseFloat(matchScore) >= 70 ? 'var(--accent)' : parseFloat(matchScore) >= 50 ? '#d4a843' : 'var(--red)'}">${matchScore}%</div>
        <div style="font-size:9px;color:var(--text-dim)">Concept match</div>
      </div>` : ''}
    </div>
    <div style="font-size:9.5px;color:var(--text-dim);margin-bottom:4px;font-weight:600">Certainty by unit</div>
    ${unitRows}`;

  content.innerHTML = html;
  panel.hidden = false;
}

// ── AI Borehole Gap Analysis ──────────────────────────────────────────────────
(function _initBoreholeGapAnalysis() {
  document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('btn-borehole-gap-analysis')?.addEventListener('click', async () => {
      const btn = document.getElementById('btn-borehole-gap-analysis');
      const out = document.getElementById('borehole-gap-results');
      if (!AppState.voxelGrid) { log('Build the 3D model first.', 'warn'); return; }
      btn.disabled = true; btn.textContent = '⟳ Analysing…';
      if (out) { out.style.display = 'none'; out.innerHTML = ''; }
      try {
        const apiKey = sessionStorage.getItem('anthropic_api_key') ?? '';
        const results = await analyseBoreholeGaps(
          AppState.voxelGrid, AppState.classifiedBH, AppState.geoUnits,
          AppState.conceptStore, AppState._lastGeoCheck ?? [], apiKey, !apiKey,
        );
        if (!results.length) { log('No borehole gap suggestions — model appears well-constrained.', 'info'); return; }
        if (out) {
          out.style.display = 'block';
          out.innerHTML = results.map((r, i) => {
            const col = r.priority === 'high' ? 'var(--red)' : '#d4a843';
            return `<div style="border:1px solid var(--border);border-radius:4px;padding:5px 6px;margin-bottom:4px;font-size:10px">
              <div style="display:flex;align-items:center;gap:5px;margin-bottom:2px">
                <span style="font-size:9px;font-weight:700;color:${col};text-transform:uppercase;letter-spacing:.5px">${r.priority}</span>
                <span style="font-size:10px;font-weight:600;color:var(--text-primary)">BH location ${i + 1}</span>
              </div>
              <div style="font-family:var(--font-mono);font-size:9px;color:var(--text-mid);margin-bottom:3px">
                x=${r.x.toFixed(0)}m &nbsp; y=${r.y.toFixed(0)}m &nbsp; depth≥${r.depth_m.toFixed(0)}m
              </div>
              <div style="font-size:9px;color:var(--text-mid);line-height:1.4">${escHtml(r.reason)}</div>
              <button class="btn-ghost btn-sm" style="font-size:9px;margin-top:4px" data-gap-i="${i}">
                + Add as virtual BH
              </button>
            </div>`;
          }).join('');
          // Wire "Add as virtual BH" buttons
          out.querySelectorAll('[data-gap-i]').forEach(btn2 => {
            btn2.addEventListener('click', () => {
              const r = results[parseInt(btn2.dataset.gapI)];
              if (!r) return;
              const gl = Math.max(...(AppState.classifiedBH?.map(b => b.groundLevel ?? 0) ?? [0]), 0);
              const synth = {
                id: `GAP${Date.now()}`, x: r.x, y: r.y, z: r.y,
                groundLevel: gl, synthetic: true,
                layers: [{ unitCode: AppState.geoUnits[0]?.code ?? 'UNKN', top: 0, base: r.depth_m, confidence: 0.3, description: 'Suggested investigation depth' }],
                classified: true, isGapProbe: true,
              };
              if (!AppState.classifiedBH) AppState.classifiedBH = [];
              AppState.classifiedBH.push(synth);
              AppState.scene?.addBoreholeSticks?.([synth], AppState.geoUnits);
              log(`Gap probe added at (${r.x.toFixed(0)}, ${r.y.toFixed(0)}) — rebuild model to incorporate.`, 'ok');
              btn2.textContent = '✓ Added'; btn2.disabled = true;
            });
          });
        }
        log(`Borehole gap analysis: ${results.length} location(s) suggested.`, 'ok');
      } catch (e) {
        log(`Borehole gap analysis failed: ${e.message}`, 'error');
      } finally {
        btn.disabled = false; btn.textContent = '✦ Suggest new borehole locations';
      }
    });
  });
})();

// ── Concept geometry verification report ─────────────────────────────────────
// Called after neural-implicit build when concepts are active.
// Shows measured E-W/N-S elongation per unit vs concept predictions.
function _showConceptGeometryReport(geoCheck) {
  const el = document.getElementById('concept-coherence-output');
  if (!el) {
    geoCheck.forEach(r => log(
      `${r.unitCode}: E-W ×${r.ewRatio} / N-S ×${r.nsRatio} (predicted E-W ×${r.predictedEW}) — ${r.conceptMatch >= 0.9 ? '✓' : r.conceptMatch >= 0.5 ? '~' : '✗'}`,
      r.conceptMatch >= 0.5 ? 'ok' : 'warn'));
    return;
  }

  const hasMismatch = geoCheck.some(r => r.conceptMatch < 0.9);
  let html = `<div style="font-size:10px;color:var(--text-mid);margin-bottom:6px;font-weight:600">Concept → Geometry Verification</div>`;

  for (const r of geoCheck) {
    const match = r.conceptMatch >= 0.9 ? '✓' : r.conceptMatch >= 0.5 ? '~' : '✗';
    const matchColor = r.conceptMatch >= 0.9 ? 'var(--accent)' : r.conceptMatch >= 0.5 ? '#d4a843' : 'var(--red)';
    const ewPct = Math.min(100, r.ewRatio * 30).toFixed(0);
    const nsPct = Math.min(100, r.nsRatio * 30).toFixed(0);
    html += `<div style="margin-bottom:6px;padding:5px;background:var(--bg-surface);border-radius:4px;border:1px solid var(--border)">
      <div style="display:flex;align-items:center;gap:5px;margin-bottom:3px">
        <span style="width:8px;height:8px;border-radius:2px;background:${r.unitColor};flex-shrink:0"></span>
        <span style="font-size:10px;font-weight:600;flex:1">${escHtml(r.unitCode)}</span>
        <span style="font-size:12px;color:${matchColor}" title="Concept-geometry match">${match}</span>
      </div>
      <div style="font-size:9px;color:var(--text-dim);margin-bottom:3px">${r.count.toLocaleString()} voxels · ${r.extentX.toFixed(0)}m E-W × ${r.extentY.toFixed(0)}m N-S × ${r.extentZ.toFixed(1)}m depth</div>
      <div style="display:flex;gap:6px;font-size:9px;font-family:var(--font-mono)">
        <div style="flex:1">
          <div style="color:var(--text-mid);margin-bottom:1px">E-W elongation</div>
          <div style="height:5px;background:var(--bg-deep);border-radius:2px;overflow:hidden;margin-bottom:1px"><div style="width:${ewPct}%;height:100%;background:var(--accent)"></div></div>
          <div style="color:var(--accent)">actual ×${r.ewRatio} · concept ×${r.predictedEW}</div>
        </div>
        <div style="flex:1">
          <div style="color:var(--text-mid);margin-bottom:1px">N-S elongation</div>
          <div style="height:5px;background:var(--bg-deep);border-radius:2px;overflow:hidden;margin-bottom:1px"><div style="width:${nsPct}%;height:100%;background:hsl(200,70%,50%)"></div></div>
          <div style="color:hsl(200,70%,50%)">actual ×${r.nsRatio} · concept ×${r.predictedNS}</div>
        </div>
      </div>
    </div>`;
  }

  // AI Refinement button — only when there are mismatches
  if (hasMismatch) {
    html += `<button id="btn-refine-concepts-ai" class="btn btn-secondary" style="width:100%;font-size:10px;margin-top:4px">
      ✦ Ask AI to suggest concept refinements
    </button>
    <div id="concept-refinement-output" style="margin-top:6px"></div>`;
  }

  el.style.display = 'block';
  el.innerHTML = html;

  // Wire the refinement button
  document.getElementById('btn-refine-concepts-ai')?.addEventListener('click', async () => {
    const btn = document.getElementById('btn-refine-concepts-ai');
    const out = document.getElementById('concept-refinement-output');
    if (btn) { btn.disabled = true; btn.textContent = '⟳ Asking AI…'; }
    try {
      const apiKey = sessionStorage.getItem('anthropic_api_key') ?? '';
      const suggestions = await refineConceptsWithClaude(
        geoCheck, AppState.conceptStore.concepts, apiKey, !apiKey
      );
      if (!suggestions.length) {
        if (out) out.innerHTML = '<p class="hint" style="font-size:10px">No refinements suggested — model geometry is consistent with concepts.</p>';
        return;
      }
      _renderConceptRefinements(suggestions, out);
      log(`AI suggested ${suggestions.length} concept refinement(s).`, 'ok');
    } catch (e) {
      log(`Concept refinement failed: ${e.message}`, 'error');
    } finally {
      if (btn) { btn.disabled = false; btn.textContent = '✦ Ask AI to suggest concept refinements'; }
    }
  });
}

function _renderCompiledRules(rules, container) {
  if (!container) return;
  const ruleTypeIcon = { superposition: '⇅', morphological: '⬡', facies: '⊟', contact: '⚡', regional: '▣' };
  const ruleTypeLabel = { superposition: 'Superposition', morphological: 'Morphological', facies: 'Facies/Depth', contact: 'Contact', regional: 'Regional' };

  let html = `<div style="font-size:10px;font-weight:600;color:var(--text-mid);margin-bottom:5px">${rules.length} compiled rule(s)</div>`;
  rules.forEach((r, i) => {
    const icon  = ruleTypeIcon[r.ruleType] ?? '⬡';
    const label = ruleTypeLabel[r.ruleType] ?? r.ruleType;
    const domainStr = r.domain?.type === 'global'
      ? (r.domain.minZ !== undefined || r.domain.maxZ !== undefined
          ? `Depth ${r.domain.minZ ?? '?'}–${r.domain.maxZ ?? '?'} m AOD` : 'Global')
      : `Spatial bbox`;
    const affinityStr = r.unitAffinity?.length ? ` · Units: ${r.unitAffinity.join(', ')}` : '';
    const tempStr = r.temporalOrder !== null && r.temporalOrder !== undefined
      ? ` · Temporal rank ${r.temporalOrder}` : '';

    // Embedding sparkline
    const bars = Array.from(r.embedding).map((v, ai) => {
      const pct = Math.round(Math.abs(v) * 100);
      const col = v >= 0 ? 'var(--accent)' : 'var(--red)';
      return `<div style="width:3px;height:${pct}%;background:${col};flex-shrink:0" title="${CONCEPT_AXES[ai]}: ${v.toFixed(2)}"></div>`;
    }).join('');

    html += `<div style="padding:5px;border:1px solid var(--border);border-radius:4px;margin-bottom:4px;font-size:10px">
      <div style="display:flex;align-items:center;gap:4px;margin-bottom:2px">
        <span style="font-size:11px">${icon}</span>
        <span style="font-weight:600;color:var(--accent)">${escHtml(label)}</span>
        <span style="font-size:9px;color:var(--text-dim);margin-left:auto">${escHtml(domainStr)}${affinityStr}${tempStr}</span>
      </div>
      <div style="font-style:italic;color:var(--text-mid);font-size:9.5px;margin-bottom:3px">"${escHtml(r.ruleText)}"</div>
      <div style="font-size:9.5px;color:var(--text-primary);margin-bottom:3px">→ ${escHtml(r.description)}</div>
      <div style="display:flex;align-items:flex-end;gap:1px;height:20px;margin-bottom:4px;background:var(--bg-deep);border-radius:2px;padding:2px">
        ${bars}
      </div>
      <button class="btn-ghost btn-sm" style="font-size:9px;width:100%" data-rule-i="${i}">+ Add concept to model</button>
    </div>`;
  });

  container.style.display = 'block';
  container.innerHTML = html;

  container.querySelectorAll('[data-rule-i]').forEach(btn => {
    btn.addEventListener('click', () => {
      const r = rules[parseInt(btn.dataset.ruleI)];
      if (!r || !AppState.conceptStore) return;
      btn.disabled = true;
      const opts = {
        description:   r.description,
        embedding:     r.embedding,
        confidence:    r.confidence,
        domain:        r.domain,
        unitAffinity:  r.unitAffinity,
        temporalOrder: r.temporalOrder,
      };
      AppState.conceptStore.add(opts);
      _renderConceptList();
      _updateConceptInfluenceBar();
      _saveConceptStore();
      log(`Rule compiled → concept added: "${r.description.slice(0, 60)}"`, 'ok');
      btn.textContent = '✓ Added';
    });
  });
}

function _renderConceptRefinements(suggestions, container) {
  if (!container) return;
  let html = '<div style="font-size:10px;font-weight:600;color:var(--text-mid);margin-bottom:5px">AI Concept Refinements</div>';
  suggestions.forEach((s, i) => {
    const axes = (s.adjustments ?? []).map(a => {
      const name = (CONCEPT_AXES ?? [])[a.axis] ?? `axis${a.axis}`;
      return `${name}: ${a.delta > 0 ? '+' : ''}${a.delta.toFixed(2)}`;
    }).join(', ');
    html += `<div style="padding:5px;border:1px solid var(--border);border-radius:4px;margin-bottom:4px;font-size:10px">
      <div style="font-style:italic;color:var(--text-primary);margin-bottom:2px">"${escHtml(s.description)}"</div>
      <div style="font-size:9px;color:var(--text-dim);margin-bottom:3px">${escHtml(s.reason)}</div>
      ${axes ? `<div style="font-size:9px;font-family:var(--font-mono);color:var(--accent);margin-bottom:3px">Adjustments: ${escHtml(axes)}</div>` : ''}
      <div style="display:flex;gap:4px">
        <button class="btn-ghost btn-sm" style="font-size:9px;flex:1" data-sug-i="${i}">+ Apply this refinement</button>
      </div>
    </div>`;
  });
  container.innerHTML = html;
  container.querySelectorAll('[data-sug-i]').forEach(btn => {
    btn.addEventListener('click', async () => {
      const s = suggestions[parseInt(btn.dataset.sugI)];
      if (!s || !AppState.conceptStore) return;
      btn.disabled = true; btn.textContent = '⟳';
      try {
        if (s.newEmbedding) {
          // Apply full replacement embedding
          if (s.conceptId) {
            const c = AppState.conceptStore.concepts.find(c => c.id === s.conceptId);
            if (c) { c.embedding = new Float32Array(s.newEmbedding); c.description = s.description; }
          } else {
            AppState.conceptStore.add({ description: s.description, embedding: new Float32Array(s.newEmbedding), confidence: 0.75, domain: { type: 'global' } });
          }
        } else if (s.adjustments?.length && s.conceptId) {
          const c = AppState.conceptStore.concepts.find(c => c.id === s.conceptId);
          if (c) {
            for (const { axis, delta } of s.adjustments) {
              if (axis >= 0 && axis < 32) c.embedding[axis] = Math.max(-1, Math.min(1, c.embedding[axis] + delta));
            }
          }
        } else {
          // Encode new concept via API or demo
          const apiKey = sessionStorage.getItem('anthropic_api_key') ?? '';
          const emb = await encodeGeologicalConcept(s.description, apiKey, !apiKey);
          AppState.conceptStore.add({ description: s.description, embedding: emb, confidence: 0.75, domain: { type: 'global' } });
        }
        _renderConceptList?.();
        _saveConceptStore?.();
        log(`Applied refinement: "${s.description.slice(0, 60)}"`, 'ok');
        btn.textContent = '✓ Applied';
      } catch (e) { log(`Apply refinement failed: ${e.message}`, 'error'); btn.textContent = '✗'; }
    });
  });
}

// ── Probability Volume Panel ──────────────────────────────────────────────────
// Wires the indicator-kriging probability volume viewer in the Analysis tab.
// Shows per-unit P(unit) across the voxel grid as a heat-map colour overlay,
// with a threshold slider to isolate high-confidence zones.
function initProbVolPanel() {
  const panel       = document.getElementById('prob-vol-panel');
  const unitSel     = document.getElementById('prob-vol-unit-sel');
  const showBtn     = document.getElementById('btn-prob-vol-show');
  const clearBtn    = document.getElementById('btn-prob-vol-clear');
  const threshSlide = document.getElementById('prob-vol-threshold');
  const threshVal   = document.getElementById('prob-vol-threshold-val');
  const statsEl     = document.getElementById('prob-vol-stats');
  if (!panel || !unitSel || !showBtn) return;

  let _activeUnit = null;

  function _populate() {
    const grid = AppState.voxelGrid;
    const hasProb = grid?.probVolumes?.size > 0;
    panel.style.display = hasProb ? '' : 'none';
    if (!hasProb) return;

    unitSel.innerHTML = '';
    for (const code of grid.probVolumes.keys()) {
      const unit = AppState.geoUnits.find(u => u.code === code);
      const opt  = document.createElement('option');
      opt.value       = code;
      opt.textContent = unit ? `${unit.name} (${code})` : code;
      unitSel.appendChild(opt);
    }
    _activeUnit = [...grid.probVolumes.keys()][0] ?? null;
    if (_activeUnit) unitSel.value = _activeUnit;
    showBtn.disabled = false;
    statsEl.textContent = '';
  }

  function _applyOverlay(code, threshold) {
    const grid = AppState.voxelGrid;
    if (!grid?.probVolumes || !AppState.scene) return;
    const probVol = grid.probVolumes.get(code);
    if (!probVol) return;

    // Create display grid: probability if >= threshold, NaN otherwise (→ grey)
    const total    = probVol.length;
    const display  = new Float32Array(total);
    let nAbove = 0;
    for (let i = 0; i < total; i++) {
      if (probVol[i] >= threshold) { display[i] = probVol[i]; nAbove++; }
      else display[i] = NaN;
    }
    AppState.scene.colorByParameter(null, AppState.geoUnits, display);
    const pct = total > 0 ? ((nAbove / total) * 100).toFixed(1) : 0;
    const unit = AppState.geoUnits.find(u => u.code === code);
    const label = unit ? unit.name : code;
    statsEl.textContent =
      `P(${label}) ≥ ${threshold.toFixed(2)}: ${nAbove.toLocaleString()} voxels (${pct}%)`;
  }

  document.addEventListener('geomodel:model-built', _populate);

  showBtn.addEventListener('click', () => {
    const code      = unitSel.value;
    const threshold = parseFloat(threshSlide.value);
    if (!code) return;
    _activeUnit = code;
    _applyOverlay(code, threshold);
    log(`Probability overlay: P(${code}) ≥ ${threshold.toFixed(2)}`, 'ok');
  });

  unitSel.addEventListener('change', () => {
    if (_activeUnit) _applyOverlay(unitSel.value, parseFloat(threshSlide.value));
  });

  threshSlide.addEventListener('input', () => {
    const v = parseFloat(threshSlide.value);
    threshVal.textContent = v.toFixed(2);
    if (_activeUnit && AppState.voxelGrid?.probVolumes) {
      _applyOverlay(_activeUnit, v);
    }
  });

  clearBtn.addEventListener('click', () => {
    _activeUnit = null;
    AppState.scene?.resetUnitColors();
    statsEl.textContent = '';
    log('Probability overlay cleared — restored unit colours.', 'info');
  });
}

// ── Concept contribution report ────────────────────────────────────────────────
// Ablation study: removes each concept one at a time, re-infers without it,
// and measures (1) % of voxels changed vs. baseline and (2) accuracy change at
// borehole observations. Identifies which concepts genuinely improve the model.
// Module-level BH accuracy helper — shared by scenario comparison and contribution report.
// Returns fraction of BH layer observations predicted correctly by unitIds.
function _bhAccuracyVsGrid(unitIds, grid) {
  if (!unitIds?.length || !grid) return null;
  const realBHs = (AppState.classifiedBH ?? []).filter(b => !b.synthetic && b.layers?.length >= 1);
  if (!realBHs.length) return null;
  let correct = 0, count = 0;
  const unitById = {};
  (AppState.geoUnits ?? []).forEach(u => { unitById[u.id] = u; });
  for (const bh of realBHs) {
    if (!isFinite(bh.x) || !isFinite(bh.y)) continue;
    const ix = Math.max(0, Math.min(grid.nx - 1, Math.round((bh.x - grid.origin.x) / grid.cellSize - 0.5)));
    const iy = Math.max(0, Math.min(grid.ny - 1, Math.round((bh.y - grid.origin.z) / grid.cellSize - 0.5)));
    for (const layer of bh.layers) {
      if (!layer.unitCode) continue;
      const elev = (bh.groundLevel ?? 0) - ((layer.top ?? 0) + (layer.base ?? 0)) / 2;
      const iz = Math.max(0, Math.min(grid.nz - 1, Math.round((elev - grid.origin.y) / grid.cellHeight - 0.5)));
      const flat = ix + iy * grid.nx + iz * grid.nx * grid.ny;
      if (flat >= 0 && flat < unitIds.length) {
        const pred = unitById[unitIds[flat]];
        count++;
        if (pred?.code === layer.unitCode) correct++;
      }
    }
  }
  return count > 0 ? correct / count : null;
}

// ── Bayesian Concept Confidence Calibration ───────────────────────────────────
// After building, computes for each concept how strongly its semantic influence
// correlates with correct vs incorrect predictions at real boreholes.
//
// Likelihood model (per concept c):
//   - For each BH observation, get concept relevance r_c at that (x,y,z)
//   - If model CORRECT:   positive_signal += r_c × certainty
//   - If model INCORRECT: negative_signal += r_c × certainty
// Bayesian update:
//   log_odds_update = alpha × (positive_signal - negative_signal) / total_signal
//   new_conf = sigmoid(logit(prior_conf) + log_odds_update)
// This makes concept confidence track BH evidence without dramatic swings.
//
// Returns [{conceptId, priorConf, posteriorConf, delta, posSignal, negSignal, bhCount}]
window._calibrateConceptConfidences = function(applyUpdates = false) {
  const store = AppState.conceptStore;
  const grid  = AppState.voxelGrid;
  const bhs   = (AppState.classifiedBH ?? []).filter(b => !b.synthetic && b.layers?.length >= 1);
  const el    = document.getElementById('concept-calib-output');

  if (!store || store.isEmpty || !grid?.unitIds || !bhs.length) {
    if (el) { el.style.display = 'block'; el.innerHTML = '<p class="hint" style="font-size:10px">Requires: built model + active concepts + real boreholes.</p>'; }
    return;
  }

  const unitById  = {};
  AppState.geoUnits.forEach(u => { unitById[u.id] = u; });

  const ALPHA     = 1.2;  // update step strength
  const MIN_CONF  = 0.15;
  const MAX_CONF  = 0.97;

  const logit   = p => Math.log(Math.max(1e-6, p) / Math.max(1e-6, 1 - p));
  const sigmoid = x => 1 / (1 + Math.exp(-x));

  const results = store.concepts.map(concept => {
    let posSignal = 0, negSignal = 0, totalSignal = 0;
    let bhCount = 0;

    for (const bh of bhs) {
      if (!isFinite(bh.x) || !isFinite(bh.y)) continue;
      const ix = Math.max(0, Math.min(grid.nx - 1, Math.round((bh.x - grid.origin.x) / grid.cellSize - 0.5)));
      const iy = Math.max(0, Math.min(grid.ny - 1, Math.round((bh.y - grid.origin.z) / grid.cellSize - 0.5)));

      for (const layer of bh.layers) {
        if (!layer.unitCode) continue;
        const elev = (bh.groundLevel ?? 0) - ((layer.top ?? 0) + (layer.base ?? 0)) / 2;
        const iz   = Math.max(0, Math.min(grid.nz - 1, Math.round((elev - grid.origin.y) / grid.cellHeight - 0.5)));
        const flat = ix + iy * grid.nx + iz * grid.nx * grid.ny;
        if (flat < 0 || flat >= grid.unitIds.length) continue;

        // Concept relevance at this BH sample
        const rel = store._relevance(concept, bh.x, bh.y, elev);
        if (rel < 0.05) continue; // concept not active here

        const cert    = grid.certainty ? grid.certainty[flat] : 0.5;
        const predId  = grid.unitIds[flat];
        const pred    = unitById[predId];
        const correct = pred?.code === layer.unitCode;

        const weight = rel * cert;
        if (correct)  posSignal += weight;
        else          negSignal += weight;
        totalSignal  += weight;
        bhCount++;
      }
    }

    if (totalSignal < 0.1) {
      // Not enough signal to update — concept has no spatial overlap with BH data
      return { conceptId: concept.id, description: concept.description,
               priorConf: concept.confidence, posteriorConf: concept.confidence,
               delta: 0, posSignal, negSignal, bhCount, hasSignal: false };
    }

    const logOddsUpdate = ALPHA * (posSignal - negSignal) / totalSignal;
    const posteriorConf = Math.max(MIN_CONF, Math.min(MAX_CONF, sigmoid(logit(concept.confidence) + logOddsUpdate)));
    const delta = posteriorConf - concept.confidence;

    return { conceptId: concept.id, description: concept.description,
             priorConf: concept.confidence, posteriorConf, delta,
             posSignal: +posSignal.toFixed(3), negSignal: +negSignal.toFixed(3),
             bhCount, hasSignal: true };
  });

  if (applyUpdates) {
    for (const r of results) {
      if (!r.hasSignal) continue;
      const c = store.concepts.find(c => c.id === r.conceptId);
      if (c) c.confidence = r.posteriorConf;
    }
    _renderConceptList();
    _updateConceptInfluenceBar();
    _saveConceptStore();
    log(`Bayesian calibration: ${results.filter(r => r.hasSignal).length} concept(s) updated`, 'ok');
  }

  // Render results
  if (el) {
    const signalResults = results.filter(r => r.hasSignal);
    if (!signalResults.length) {
      el.style.display = 'block';
      el.innerHTML = '<p class="hint" style="font-size:10px">Concepts have no spatial overlap with borehole locations — no calibration signal available. Add spatially bounded domain concepts to get meaningful calibration.</p>';
      return;
    }

    const rows = results.map(r => {
      const col  = r.delta > 0.03 ? 'var(--accent)' : r.delta < -0.03 ? 'var(--red)' : 'var(--text-mid)';
      const sign = r.delta >= 0 ? '+' : '';
      const bar  = r.hasSignal
        ? `<div style="display:flex;gap:2px;height:5px;margin:2px 0">
             <div style="flex:${r.posSignal.toFixed(2)};background:#5ab97d;border-radius:1px" title="Positive signal (helped BH predictions)"></div>
             <div style="flex:${r.negSignal.toFixed(2)};background:#e06c75;border-radius:1px" title="Negative signal (hurt BH predictions)"></div>
           </div>` : '';
      return `<div style="padding:4px 5px;border:1px solid var(--border);border-radius:4px;margin-bottom:3px;font-size:10px">
        <div style="display:flex;align-items:center;gap:4px">
          <span style="flex:1;color:var(--text-mid);overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${escHtml(r.description)}">${escHtml(r.description.slice(0, 40))}</span>
          <span style="font-family:var(--font-mono);font-size:9px;color:var(--text-dim)">${(r.priorConf * 100).toFixed(0)}%→</span>
          <span style="font-family:var(--font-mono);font-size:9px;color:${col}">${(r.posteriorConf * 100).toFixed(0)}%</span>
          <span style="font-size:9px;color:${col}">(${sign}${(r.delta * 100).toFixed(0)}%)</span>
        </div>
        ${bar}
        ${r.hasSignal ? `<div style="font-size:8.5px;color:var(--text-dim)">${r.bhCount} BH points · +${r.posSignal} / -${r.negSignal}</div>` : '<div style="font-size:8.5px;color:var(--text-dim)">No spatial overlap with BH data</div>'}
      </div>`;
    }).join('');

    el.style.display = 'block';
    el.innerHTML = `<div style="font-size:10px;font-weight:600;color:var(--text-mid);margin-bottom:5px">Bayesian confidence updates</div>
      ${rows}
      <button id="btn-apply-calib" class="btn btn-secondary btn-sm" style="width:100%;margin-top:5px;font-size:10px">Apply updates to concept confidences</button>`;

    document.getElementById('btn-apply-calib')?.addEventListener('click', () => {
      window._calibrateConceptConfidences(true);
      const btn = document.getElementById('btn-apply-calib');
      if (btn) { btn.disabled = true; btn.textContent = '✓ Applied'; }
    });
  }
};

window._runConceptContributionReport = async function() {
  const el    = document.getElementById('concept-contrib-output');
  const grid  = AppState.voxelGrid;
  const store = AppState.conceptStore;
  if (!el) return;
  el.style.display = 'block';

  if (!AppState.trainedModel || !grid) {
    el.innerHTML = '<div style="font-size:10px;color:#e06c75">Requires neural-implicit method — build the model with neural-implicit first.</div>';
    return;
  }
  if (!store || store.isEmpty) {
    el.innerHTML = '<div style="font-size:10px;color:#e06c75">No active concepts. Add concepts first.</div>';
    return;
  }

  const concepts = store.concepts;
  if (!concepts.length) return;

  el.innerHTML = '<div style="font-size:10px;color:var(--text-mid)">Running ablation study…</div>';

  const gridMeta = {
    nx: grid.nx, ny: grid.ny, nz: grid.nz,
    cellSize: grid.cellSize, cellHeight: grid.cellHeight, origin: grid.origin,
  };
  const total = grid.unitIds.length;

  function _bhAccuracy(unitIds) {
    return _bhAccuracyVsGrid(unitIds, grid);
  }
  // Baseline: full concept store
  const baselineResult = inferGeoImplicit(AppState.trainedModel, gridMeta, AppState.geoUnits, store);
  const baselineAcc    = _bhAccuracy(baselineResult.unitIds);

  const rows = [];

  for (let ci = 0; ci < concepts.length; ci++) {
    el.innerHTML = `<div style="font-size:10px;color:var(--text-mid)">Ablating concept ${ci + 1}/${concepts.length}…</div>`;
    await new Promise(r => setTimeout(r, 0));

    // Clone store without concept ci
    const ablatedStore = store.cloneScaled(1.0);
    ablatedStore._concepts.splice(ci, 1);

    const ablResult = inferGeoImplicit(AppState.trainedModel, gridMeta, AppState.geoUnits, ablatedStore);

    let changed = 0;
    const compareLen = Math.min(total, ablResult.unitIds?.length ?? 0, baselineResult.unitIds?.length ?? 0);
    for (let i = 0; i < compareLen; i++) {
      if (ablResult.unitIds[i] !== baselineResult.unitIds[i]) changed++;
    }
    const influence = compareLen > 0 ? changed / compareLen : 0;

    const ablAcc    = _bhAccuracy(ablResult.unitIds);
    const accDelta  = (baselineAcc != null && ablAcc != null) ? (baselineAcc - ablAcc) : null;

    rows.push({
      concept: concepts[ci],
      influence,
      accDelta,
      ablAcc,
    });
  }

  // Sort by influence (most impactful first)
  rows.sort((a, b) => b.influence - a.influence);

  const baseAccStr = baselineAcc != null
    ? `${(baselineAcc * 100).toFixed(1)}%`
    : 'N/A (no BH data)';

  const rowsHtml = rows.map(r => {
    const infPct  = (r.influence * 100).toFixed(1);
    const infBar  = '▓'.repeat(Math.round(r.influence * 20)).padEnd(20, '░');
    const infCol  = r.influence > 0.1 ? 'var(--ok)' : r.influence > 0.03 ? '#c8a855' : 'var(--text-muted)';

    let accHtml = '';
    if (r.accDelta != null) {
      const sign  = r.accDelta >= 0 ? '+' : '';
      const col   = r.accDelta > 0.01 ? 'var(--ok)' : r.accDelta < -0.01 ? 'var(--error)' : 'var(--text-muted)';
      accHtml = `<span style="color:${col};margin-left:6px">${sign}${(r.accDelta * 100).toFixed(1)}% acc</span>`;
    }

    const desc = r.concept.description.length > 40
      ? r.concept.description.slice(0, 40) + '…'
      : r.concept.description;

    return `<div style="margin-bottom:6px;padding:5px 6px;background:var(--bg-surface);border-radius:3px;border-left:3px solid ${infCol}">
      <div style="font-size:9.5px;font-weight:600;color:var(--text)">${desc}</div>
      <div style="font-size:9px;color:${infCol};margin-top:2px;font-family:monospace">${infBar} ${infPct}%${accHtml}</div>
      <div style="font-size:8.5px;color:var(--text-muted);margin-top:1px">Removing shifts ${infPct}% of voxels${r.ablAcc != null ? ` · BH accuracy without: ${(r.ablAcc * 100).toFixed(1)}%` : ''}</div>
    </div>`;
  }).join('');

  el.innerHTML = `
    <div style="font-size:10px;font-weight:600;margin-bottom:6px">
      Baseline accuracy: ${baseAccStr}
      <span style="font-weight:400;color:var(--text-muted)"> (${realBHs.length} BHs)</span>
    </div>
    ${rowsHtml}
    <p style="font-size:9px;color:var(--text-muted);margin-top:4px;font-style:italic">
      Higher influence = more voxels shift when concept is removed.<br>
      Positive accuracy delta = concept improves BH prediction accuracy.
    </p>`;

  const topInfluence = rows[0]?.influence ?? 0;
  const avgAccBoost  = rows.filter(r => r.accDelta != null).reduce((s, r) => s + r.accDelta, 0) / (rows.filter(r => r.accDelta != null).length || 1);
  log(
    `Concept contribution: ${rows.length} concepts · top influence ${(topInfluence * 100).toFixed(1)}%` +
    (baselineAcc != null ? ` · baseline accuracy ${(baselineAcc * 100).toFixed(1)}%` : ''),
    'ok'
  );
};

// Geological Knowledge Uncertainty — stochastic concept embedding sampling
// Runs K inference passes with Box-Muller-perturbed concept embeddings, then
// computes per-voxel entropy across realisations and colours the model.
window._runKnowledgeUncertainty = async function(K = 6, baseNoise = 0.12) {
  const el    = document.getElementById('knowledge-uncert-output');
  const grid  = AppState.voxelGrid;
  const store = AppState.conceptStore;
  if (!el) return;
  el.style.display = 'block';

  if (!AppState.trainedModel || !grid) {
    el.innerHTML = '<div style="font-size:10px;color:#e06c75">Requires neural-implicit method — build model first.</div>';
    return;
  }
  if (!store || store.isEmpty) {
    el.innerHTML = '<div style="font-size:10px;color:#e06c75">No active concepts. Add concepts to quantify knowledge uncertainty.</div>';
    return;
  }

  el.innerHTML = `<div style="font-size:10px;color:var(--text-mid)">Sampling concept space… 0/${K}</div>`;

  const gridMeta = {
    nx: grid.nx, ny: grid.ny, nz: grid.nz,
    cellSize: grid.cellSize, cellHeight: grid.cellHeight, origin: grid.origin,
  };
  const nVox = grid.unitIds.length;
  const nUnits = AppState.geoUnits.length;

  // Accumulate per-voxel unit-assignment counts across K realisations
  const counts = new Float32Array(nVox * nUnits); // counts[vox*nUnits + unitIdx]

  const unitIdxMap = {};
  AppState.geoUnits.forEach((u, i) => { unitIdxMap[u.id] = i; });

  for (let k = 0; k < K; k++) {
    el.innerHTML = `<div style="font-size:10px;color:var(--text-mid)">Sampling concept space… ${k + 1}/${K}</div>`;
    await new Promise(r => setTimeout(r, 0)); // yield to UI

    const pertStore = store.clonePerturbed(baseNoise);
    const result    = inferGeoImplicit(AppState.trainedModel, gridMeta, AppState.geoUnits, pertStore);
    const ids       = result.unitIds;

    for (let v = 0; v < nVox; v++) {
      const ui = unitIdxMap[ids[v]];
      if (ui !== undefined) counts[v * nUnits + ui] += 1;
    }
  }

  // Per-voxel entropy H = -Σ p_k log2(p_k)
  const entropy = new Float32Array(nVox);
  let sumH = 0, nHigh = 0;
  for (let v = 0; v < nVox; v++) {
    let h = 0;
    for (let u = 0; u < nUnits; u++) {
      const p = counts[v * nUnits + u] / K;
      if (p > 0) h -= p * Math.log2(p);
    }
    entropy[v] = h;
    sumH += h;
    if (h > 0.5) nHigh++;
  }
  const maxH     = Math.log2(nUnits) || 1;
  const meanH    = nVox > 0 ? sumH / nVox : 0;
  const pctHigh  = nVox > 0 ? (nHigh / nVox * 100) : 0;

  // Normalise 0-1 for colour mapping
  const normEntropy = new Float32Array(nVox);
  for (let v = 0; v < nVox; v++) normEntropy[v] = entropy[v] / maxH;

  // Colour model: low entropy = green (borehole-anchored), high entropy = red (concept-sensitive)
  if (AppState.scene?.colorByParameter) {
    AppState.scene.colorByParameter('knowledge_uncertainty', AppState.geoUnits, normEntropy);
  }

  // Distribution histogram: 10 bins
  const BINS = 10;
  const hist = new Array(BINS).fill(0);
  for (let v = 0; v < nVox; v++) {
    const bin = Math.min(BINS - 1, Math.floor(normEntropy[v] * BINS));
    hist[bin]++;
  }
  const histMax = Math.max(...hist, 1);
  const histBars = hist.map((n, i) => {
    const pct = (n / histMax * 40).toFixed(0);
    const label = (i / BINS * 100).toFixed(0) + '%';
    return `<div title="${n} voxels · entropy ${label}–${((i + 1) / BINS * 100).toFixed(0)}%" style="display:flex;flex-direction:column;align-items:center;gap:1px">
      <div style="width:16px;height:${pct}px;background:hsl(${120 - i * 12},60%,45%);border-radius:1px 1px 0 0"></div>
      ${i === 0 || i === BINS - 1 ? `<div style="font-size:7px;color:var(--text-dim)">${label}</div>` : '<div style="font-size:7px;color:transparent">·</div>'}
    </div>`;
  }).join('');

  el.innerHTML = `
    <div style="font-size:10px;font-weight:600;color:var(--text);margin-bottom:5px">Knowledge uncertainty — ${K} concept-space realisations</div>
    <div style="display:flex;gap:8px;margin-bottom:8px">
      <div style="flex:1;padding:5px;background:var(--bg-surface);border:1px solid var(--border);border-radius:4px;text-align:center">
        <div style="font-size:16px;font-weight:700;color:var(--accent)">${meanH.toFixed(2)}</div>
        <div style="font-size:9px;color:var(--text-dim)">mean entropy (bits)</div>
      </div>
      <div style="flex:1;padding:5px;background:var(--bg-surface);border:1px solid var(--border);border-radius:4px;text-align:center">
        <div style="font-size:16px;font-weight:700;color:${pctHigh > 20 ? '#e06c75' : '#5ab97d'}">${pctHigh.toFixed(1)}%</div>
        <div style="font-size:9px;color:var(--text-dim)">high-uncertainty voxels</div>
      </div>
    </div>
    <div style="display:flex;align-items:flex-end;height:44px;gap:1px;padding:0 2px;margin-bottom:4px">${histBars}</div>
    <div style="font-size:9px;color:var(--text-dim);margin-bottom:6px">Distribution: low entropy (left) = stable interpretation; high (right) = concept-sensitive</div>
    <div style="font-size:9px;color:var(--text-mid);padding:4px 6px;background:var(--bg-surface);border-left:2px solid var(--accent);border-radius:0 3px 3px 0">
      Model coloured by knowledge uncertainty. <strong>Green</strong> = interpretation stable across concept perturbations. <strong>Red</strong> = high sensitivity to how you have described the geology.
    </div>
    <div style="margin-top:6px;display:flex;gap:4px">
      <button class="btn btn-ghost btn-sm" style="font-size:9px;flex:1" onclick="AppState.scene?.colorByParameter(null, AppState.geoUnits, null)">Reset colour</button>
      <button class="btn btn-ghost btn-sm" style="font-size:9px;flex:1" onclick="_runKnowledgeUncertainty(${K}, 0.20)">Re-run (noise ×1.7)</button>
    </div>`;

  log(`Knowledge uncertainty: K=${K} realisations · mean entropy ${meanH.toFixed(3)} bits · ${pctHigh.toFixed(1)}% high-uncertainty voxels`, 'ok');
};

// Predictive Borehole Log — render a synthetic log at any (x,y) from the voxel grid.
// Shows unit sequence, certainty, concept context samples, and nearest real BH comparison.
window._predictBoreholeLog = function() {
  const el = document.getElementById('pred-bh-output');
  const grid  = AppState.voxelGrid;
  const store = AppState.conceptStore;
  if (!el) return;

  if (!grid) {
    el.style.display = 'block';
    el.innerHTML = '<div style="font-size:10px;color:#e06c75">Build the model first.</div>';
    return;
  }

  const xIn = parseFloat(document.getElementById('pred-bh-x')?.value ?? 'NaN');
  const yIn = parseFloat(document.getElementById('pred-bh-y')?.value ?? 'NaN');
  if (isNaN(xIn) || isNaN(yIn)) {
    el.style.display = 'block';
    el.innerHTML = '<div style="font-size:10px;color:#e06c75">Enter valid X, Y coordinates (world units).</div>';
    return;
  }

  el.style.display = 'block';
  el.innerHTML = '<div style="font-size:10px;color:var(--text-mid)">Extracting log…</div>';

  const realBHs = AppState.boreholes ?? [];
  const result  = predictBoreholeLog(xIn, yIn, grid, AppState.geoUnits, store, realBHs);

  if (!result || !result.runs.length) {
    el.innerHTML = '<div style="font-size:10px;color:#e06c75">Location is outside the model grid.</div>';
    return;
  }

  const { runs, conceptSamples, nearestBH } = result;
  const topZ = runs[0].fromZ;
  const botZ = runs[runs.length - 1].toZ;
  const range = topZ - botZ || 1;

  // SVG dimensions
  const SVG_H     = Math.max(200, Math.min(480, range * 7));
  const SVG_W     = nearestBH ? 290 : 180;
  const DEPTH_W   = 38;  // left depth axis
  const LOG_W     = 70;  // predicted log column
  const GAP       = 8;
  const COMP_X    = DEPTH_W + LOG_W + GAP + 8; // nearest BH comparison start
  const COMP_W    = nearestBH ? 60 : 0;
  const SPARK_X   = DEPTH_W + LOG_W + GAP;

  const zToY = z => ((topZ - z) / range) * SVG_H;
  const escSvg = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;');

  // Predicted log blocks
  let logBlocks = runs.map(r => {
    const y1  = zToY(r.fromZ);
    const y2  = zToY(r.toZ);
    const h   = Math.max(1, y2 - y1);
    const col = r.unit.color ?? '#888';
    const opa = 0.35 + r.certainty * 0.65;
    const midY = (y1 + y2) / 2;
    const label = h > 10 ? r.unit.code : '';
    const certPct = Math.round(r.certainty * 100);
    return `<rect x="${DEPTH_W}" y="${y1.toFixed(1)}" width="${LOG_W}" height="${h.toFixed(1)}"
              fill="${escSvg(col)}" opacity="${opa.toFixed(2)}" stroke="none"/>
            <rect x="${DEPTH_W}" y="${y1.toFixed(1)}" width="${(LOG_W * r.certainty).toFixed(1)}" height="2"
              fill="rgba(255,255,255,0.4)" />
            ${label ? `<text x="${DEPTH_W + LOG_W / 2}" y="${midY + 3}" font-size="8" text-anchor="middle"
              fill="rgba(0,0,0,0.75)" font-family="monospace">${escSvg(r.unit.code)}</text>` : ''}
            <title>${escSvg(r.unit.name)} ${r.fromZ.toFixed(1)}–${r.toZ.toFixed(1)}m AOD · ${certPct}% certainty · ${r.thickness.toFixed(1)}m thick</title>`;
  }).join('');

  // Depth tick marks every 2m
  let ticks = '';
  const tickStep = range < 15 ? 1 : range < 40 ? 2 : 5;
  const firstTick = Math.ceil(botZ / tickStep) * tickStep;
  for (let z = firstTick; z <= topZ; z += tickStep) {
    const ty = zToY(z).toFixed(1);
    ticks += `<line x1="${DEPTH_W - 4}" y1="${ty}" x2="${DEPTH_W + LOG_W}" y2="${ty}" stroke="rgba(255,255,255,0.12)" stroke-width="0.5"/>
              <text x="${DEPTH_W - 6}" y="${parseFloat(ty) + 3}" font-size="7" text-anchor="end" fill="var(--text-dim)">${z.toFixed(0)}</text>`;
  }

  // Concept sparklines at sampled depths (right side of log)
  let sparkLines = '';
  if (conceptSamples.length && store && !store.isEmpty) {
    const SBAR_W  = 2;
    const SBAR_GAP = 0.5;
    const SPARK_TOTAL_W = 32 * (SBAR_W + SBAR_GAP);
    conceptSamples.forEach(s => {
      if (!s.vec) return;
      const sy = zToY(s.z).toFixed(1);
      let bars = '';
      for (let i = 0; i < 32; i++) {
        const v   = s.vec[i] ?? 0;
        const bh  = Math.abs(v) * 12;
        const bx  = DEPTH_W + LOG_W + GAP + i * (SBAR_W + SBAR_GAP);
        const col2 = v >= 0 ? '#5ab97d' : '#e06c75';
        bars += `<rect x="${bx.toFixed(1)}" y="${(parseFloat(sy) - bh / 2).toFixed(1)}" width="${SBAR_W}" height="${bh.toFixed(1)}" fill="${col2}" opacity="0.8"/>`;
      }
      sparkLines += `<g title="Concept context at ${s.z.toFixed(1)}m AOD">${bars}
        <line x1="${DEPTH_W + LOG_W + 2}" y1="${sy}" x2="${DEPTH_W + LOG_W + GAP + SPARK_TOTAL_W}" y2="${sy}" stroke="rgba(255,255,255,0.1)" stroke-width="0.5"/>
      </g>`;
    });
  }

  // Nearest BH comparison column
  let bhCol = '';
  if (nearestBH?.layers?.length) {
    const bhTopZ = Math.max(...nearestBH.layers.map(l => l.fromDepth ?? topZ));
    const bhBotZ = Math.min(...nearestBH.layers.map(l => l.toDepth   ?? botZ));
    const unitByCode = Object.fromEntries(AppState.geoUnits.map(u => [u.code, u]));
    nearestBH.layers.forEach(layer => {
      const unit = unitByCode[layer.unitCode];
      if (!unit) return;
      const ly1 = zToY(layer.fromDepth ?? topZ);
      const ly2 = zToY(layer.toDepth   ?? botZ);
      const lh  = Math.max(1, ly2 - ly1);
      bhCol += `<rect x="${COMP_X}" y="${ly1.toFixed(1)}" width="${COMP_W}" height="${lh.toFixed(1)}"
        fill="${escSvg(unit.color ?? '#888')}" opacity="0.8" stroke="none"/>
        <title>${escSvg(unit.code)} (real BH: ${escSvg(nearestBH.id)})</title>`;
    });
    bhCol += `<text x="${COMP_X + COMP_W / 2}" y="${SVG_H + 12}" font-size="7" text-anchor="middle" fill="var(--text-dim)">BH ${escSvg(nearestBH.id)}</text>
              <text x="${COMP_X + COMP_W / 2}" y="${SVG_H + 20}" font-size="6.5" text-anchor="middle" fill="var(--text-dim)">${nearestBH.dist}m away</text>`;
  }

  const SPARK_LABEL_W = conceptSamples.length && store && !store.isEmpty ? 32 * 2.5 : 0;
  const totalW = DEPTH_W + LOG_W + (SPARK_LABEL_W ? GAP + SPARK_LABEL_W + 4 : 0) + (nearestBH ? GAP + COMP_W + 4 : 0);

  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${totalW}" height="${SVG_H + 30}"
      style="display:block;background:var(--bg-deep);border-radius:4px;overflow:visible">
    <!-- depth axis -->
    <line x1="${DEPTH_W}" y1="0" x2="${DEPTH_W}" y2="${SVG_H}" stroke="rgba(255,255,255,0.25)" stroke-width="1"/>
    <text x="${DEPTH_W - 6}" y="-4" font-size="7" text-anchor="end" fill="var(--text-dim)">mAOD</text>
    ${ticks}
    <!-- log blocks -->
    ${logBlocks}
    <!-- column header -->
    <text x="${DEPTH_W + LOG_W / 2}" y="-4" font-size="7.5" text-anchor="middle" fill="var(--text-mid)" font-weight="600">Predicted</text>
    <!-- concept sparklines -->
    ${sparkLines}
    ${conceptSamples.length ? `<text x="${DEPTH_W + LOG_W + GAP}" y="-4" font-size="7" fill="var(--text-dim)">Concept context →</text>` : ''}
    <!-- nearest BH -->
    ${bhCol}
    ${nearestBH ? `<text x="${COMP_X + COMP_W / 2}" y="-4" font-size="7.5" text-anchor="middle" fill="var(--text-mid)" font-weight="600">Nearest BH</text>` : ''}
    <!-- base line -->
    <line x1="${DEPTH_W}" y1="${SVG_H}" x2="${DEPTH_W + LOG_W}" y2="${SVG_H}" stroke="rgba(255,255,255,0.3)" stroke-width="1"/>
    <text x="${DEPTH_W + LOG_W / 2}" y="${SVG_H + 12}" font-size="7" text-anchor="middle" fill="var(--text-dim)">X ${result.x} Y ${result.y}</text>
  </svg>`;

  // Legend table
  const seen = new Set();
  const legendItems = runs
    .filter(r => { if (seen.has(r.unit.id)) return false; seen.add(r.unit.id); return true; })
    .map(r => `<span style="display:inline-flex;align-items:center;gap:3px;margin-right:6px;font-size:9px">
      <span style="display:inline-block;width:10px;height:10px;background:${r.unit.color ?? '#888'};border-radius:2px"></span>
      ${escHtml(r.unit.code)}
    </span>`).join('');

  el.innerHTML = `
    <div style="font-size:10px;font-weight:600;margin-bottom:6px;color:var(--text)">
      Predicted log @ X=${result.x} Y=${result.y}
      ${nearestBH ? `<span style="font-weight:400;color:var(--text-dim)"> · nearest BH: ${escHtml(nearestBH.id ?? '?')} (${nearestBH.dist}m)</span>` : ''}
    </div>
    <div style="overflow-x:auto;padding:4px 0 16px">${svg}</div>
    <div style="margin-top:4px;display:flex;flex-wrap:wrap">${legendItems}</div>
    <div style="margin-top:6px;font-size:9px;color:var(--text-dim)">
      ${runs.length} unit run(s) · ${topZ.toFixed(1)}m → ${botZ.toFixed(1)}m AOD · ${range.toFixed(1)}m total depth
      ${store && !store.isEmpty ? ' · concept context bars show active axes at each sampled depth' : ''}
    </div>`;

  log(`Predictive BH log: ${runs.length} units at (${result.x}, ${result.y}) · ${range.toFixed(1)}m depth`, 'ok');
};

// Concept-Annotated Cross-Section Generator
// Samples the voxel grid along a section plane (any azimuth) and renders:
//   - Coloured unit pixel grid with certainty shading
//   - Concept annotation overlays (orientation arrows, channel profile, fault symbols)
//   - Intersecting real borehole sticks
window._generateCrossSection = function() {
  const canvas = document.getElementById('xs-canvas');
  const el     = document.getElementById('xs-output');
  const grid   = AppState.voxelGrid;
  const store  = AppState.conceptStore;
  if (!canvas || !el) return;

  if (!grid) {
    el.style.display = 'block';
    el.innerHTML = '<div style="font-size:10px;color:#e06c75">Build the model first.</div>';
    return;
  }

  const azimuth = parseFloat(document.getElementById('xs-azimuth')?.value ?? '90');
  const length  = parseFloat(document.getElementById('xs-length')?.value  ?? '200');
  const rawMidX = document.getElementById('xs-midx')?.value?.trim();
  const rawMidY = document.getElementById('xs-midy')?.value?.trim();
  const midX = rawMidX ? parseFloat(rawMidX) : (grid.origin[0] + grid.nx * grid.cellSize / 2);
  const midY = rawMidY ? parseFloat(rawMidY) : (grid.origin[1] + grid.ny * grid.cellSize / 2);

  const NCOLS = 120;
  const result = generateCrossSection(
    { azimuthDeg: azimuth, midX, midY, length, nCols: NCOLS },
    grid, AppState.geoUnits, store, AppState.boreholes ?? []
  );

  if (!result) {
    el.style.display = 'block';
    el.innerHTML = '<div style="font-size:10px;color:#e06c75">Could not generate section — check coordinates.</div>';
    return;
  }

  const { pixels, nCols, nRows, topZ, botZ, conceptSamples, bhIntersections } = result;
  const PIXEL_W = 4;
  const PIXEL_H = Math.max(3, Math.round(320 / nRows));
  const canW    = nCols * PIXEL_W;
  const canH    = nRows * PIXEL_H;

  canvas.width  = canW;
  canvas.height = canH;
  canvas.style.width  = Math.min(canW, 360) + 'px';
  canvas.style.height = (canH * Math.min(canW, 360) / canW) + 'px';
  canvas.style.display = 'block';

  const ctx2d = canvas.getContext('2d');
  ctx2d.clearRect(0, 0, canW, canH);

  // Draw unit pixels
  for (let row = 0; row < nRows; row++) {
    for (let col = 0; col < nCols; col++) {
      const px = pixels[col + row * nCols];
      if (!px?.unit) {
        ctx2d.fillStyle = 'rgba(20,28,36,0.4)';
      } else {
        const alpha = 0.35 + px.certainty * 0.65;
        ctx2d.fillStyle = px.unit.color
          ? (px.unit.color + Math.round(alpha * 255).toString(16).padStart(2, '0'))
          : `rgba(100,100,100,${alpha.toFixed(2)})`;
      }
      ctx2d.fillRect(col * PIXEL_W, row * PIXEL_H, PIXEL_W, PIXEL_H);
    }
  }

  // Draw unit contact lines (scan columns for vertical changes)
  ctx2d.strokeStyle = 'rgba(255,255,255,0.35)';
  ctx2d.lineWidth   = 0.8;
  for (let col = 0; col < nCols; col++) {
    for (let row = 1; row < nRows; row++) {
      const above = pixels[col + (row - 1) * nCols];
      const here  = pixels[col +  row      * nCols];
      if (above?.unitId !== here?.unitId) {
        ctx2d.beginPath();
        ctx2d.moveTo(col * PIXEL_W,              row * PIXEL_H);
        ctx2d.lineTo((col + 1) * PIXEL_W,        row * PIXEL_H);
        ctx2d.stroke();
      }
    }
  }

  // Draw intersecting boreholes as white lines with depth ticks
  if (bhIntersections.length) {
    for (const bh of bhIntersections) {
      const colPx = Math.round((bh.tProj / length + 0.5) * nCols);
      const cx    = colPx * PIXEL_W + PIXEL_W / 2;
      ctx2d.strokeStyle = 'rgba(255,255,255,0.8)';
      ctx2d.lineWidth   = 1.5;
      ctx2d.beginPath();
      ctx2d.moveTo(cx, 0);
      ctx2d.lineTo(cx, canH);
      ctx2d.stroke();
      ctx2d.fillStyle = 'rgba(255,255,255,0.9)';
      ctx2d.font = '8px monospace';
      ctx2d.textAlign = 'center';
      ctx2d.fillText(bh.id ?? '?', cx, 9);
    }
  }

  // ── Concept annotation overlay ──────────────────────────────────────────────
  // Compute mean concept vector across section samples
  const meanVec = new Float32Array(32);
  let sCount = 0;
  for (const s of conceptSamples) {
    if (!s.vec) continue;
    for (let i = 0; i < 32; i++) meanVec[i] += s.vec[i];
    sCount++;
  }
  if (sCount > 0) for (let i = 0; i < 32; i++) meanVec[i] /= sCount;

  const ew_elong  = meanVec[3]  ?? 0;
  const ns_elong  = meanVec[4]  ?? 0;
  const channel   = meanVec[5]  ?? 0;
  const fault     = meanVec[7]  ?? 0;
  const erosional = meanVec[8]  ?? 0;
  const stepped   = meanVec[18] ?? 0;
  const karst     = meanVec[24] ?? 0;
  const complex   = meanVec[25] ?? 0;

  // Draw section-plane alignment arrow if concept has strong elongation
  // (elongation axis that is ALONG the section → shows as continuation; perpendicular → shows pinch)
  const azR          = ((azimuth - 90) * Math.PI) / 180;
  const sectionDotEW = Math.abs(Math.cos(azR));  // how much section cuts E-W
  const sectionDotNS = Math.abs(Math.sin(azR));

  // Orientation banner text
  let annotLines = [];
  if (Math.abs(ew_elong) > 0.4 || Math.abs(ns_elong) > 0.4) {
    const align = ew_elong > ns_elong
      ? (sectionDotEW > 0.7 ? '↔ E-W body cut along trend (long axis)' : '↑ E-W body cut across trend (short axis)')
      : (sectionDotNS > 0.7 ? '↕ N-S body cut along trend (long axis)' : '↔ N-S body cut across trend (short axis)');
    annotLines.push(align);
  }
  if (channel > 0.5)   annotLines.push('⌣ Channel morphology — concave-up expected');
  if (stepped > 0.6)   annotLines.push('⇅ Stepped boundary — fault-controlled contacts');
  if (fault > 0.5)     annotLines.push('⚡ Fault-controlled geometry');
  if (karst > 0.5)     annotLines.push('⊙ Dissolution/karst features');
  if (erosional > 0.6) annotLines.push('∿ Erosional basal contact');

  // Draw channel profile arc if channel morphology is strong
  if (channel > 0.5 && nRows > 4) {
    const arcDepth  = Math.min(canH * 0.25, channel * 40);
    const arcWidth  = canW * (0.5 + ew_elong * 0.3);
    const arcCentreX = canW / 2;
    // Find approximate base of topmost sand/gravel unit in section
    let baseRow = Math.round(nRows * 0.5);
    ctx2d.strokeStyle = `rgba(255,220,80,${(channel * 0.6).toFixed(2)})`;
    ctx2d.lineWidth = 1.5;
    ctx2d.setLineDash([3, 3]);
    ctx2d.beginPath();
    ctx2d.arc(arcCentreX, baseRow * PIXEL_H - arcDepth * 0.5, arcWidth / 2, 0, Math.PI);
    ctx2d.stroke();
    ctx2d.setLineDash([]);
  }

  // Draw stepped boundary lines if stepped_boundary is strong
  if (stepped > 0.6) {
    ctx2d.strokeStyle = `rgba(224,108,117,${(stepped * 0.7).toFixed(2)})`;
    ctx2d.lineWidth = 1.5;
    ctx2d.setLineDash([4, 2]);
    const stepY1 = Math.round(canH * 0.35);
    const stepY2 = Math.round(canH * 0.55);
    const stepX  = Math.round(canW * 0.52);
    ctx2d.beginPath();
    ctx2d.moveTo(0,     stepY1); ctx2d.lineTo(stepX,  stepY1);
    ctx2d.moveTo(stepX, stepY1); ctx2d.lineTo(stepX,  stepY2);
    ctx2d.moveTo(stepX, stepY2); ctx2d.lineTo(canW,   stepY2);
    ctx2d.stroke();
    ctx2d.setLineDash([]);
    // Downthrow tick
    ctx2d.fillStyle = `rgba(224,108,117,0.9)`;
    ctx2d.font = '9px sans-serif';
    ctx2d.fillText('⇓', stepX - 4, stepY1 + 10);
  }

  // Render annotation text beneath canvas
  el.style.display = 'block';
  if (annotLines.length) {
    el.innerHTML = annotLines.map(a =>
      `<div style="font-size:9px;color:var(--accent);margin-bottom:2px">${escHtml(a)}</div>`
    ).join('');
  } else {
    el.innerHTML = '<div style="font-size:9px;color:var(--text-dim)">No strong concept geometry detected along section.</div>';
  }

  // Legend row
  const seen = new Set();
  const units = [];
  for (const px of pixels) { if (px?.unit && !seen.has(px.unit.id)) { seen.add(px.unit.id); units.push(px.unit); } }
  el.innerHTML += '<div style="display:flex;flex-wrap:wrap;gap:4px;margin-top:4px">' +
    units.map(u => `<span style="font-size:9px;display:inline-flex;align-items:center;gap:3px">
      <span style="width:10px;height:10px;display:inline-block;background:${u.color ?? '#888'};border-radius:2px"></span>${escHtml(u.code)}</span>`
    ).join('') + '</div>';
  el.innerHTML += `<div style="font-size:8.5px;color:var(--text-dim);margin-top:3px">
    Section: Az ${azimuth}° · L=${length}m · ${bhIntersections.length} BH(s) intersect · midpoint (${midX.toFixed(0)}, ${midY.toFixed(0)})</div>`;

  log(`Cross-section: Az${azimuth}° · ${nCols}×${nRows}px · ${bhIntersections.length} BH(s) · ${annotLines.length} concept annotation(s)`, 'ok');
};
