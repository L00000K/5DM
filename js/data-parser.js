import { log, setEnabled } from './app.js';

// ── AGS 4.x parser ────────────────────────────────────────────────────────────
// Reads key groups: LOCA, GEOL, ISPT
// Returns BHLog[]

export function parseAGS(text) {
  const lines  = text.split(/\r?\n/);
  const groups = {};
  let current  = null;
  let headings = [];

  for (const raw of lines) {
    const line = raw.trim();
    if (!line || line.startsWith('!')) continue;

    const cells = parseAGSLine(line);
    if (!cells.length) continue;
    const tag = cells[0].toUpperCase().replace(/"/g, '');

    if (tag === 'GROUP') {
      current = stripQuotes(cells[1] || '').toUpperCase();
      groups[current] = [];
      headings = [];
    } else if (tag === 'HEADING') {
      headings = cells.slice(1).map(stripQuotes);
    } else if (tag === 'DATA' && current && headings.length) {
      const row = {};
      cells.slice(1).forEach((c, i) => { row[headings[i]] = stripQuotes(c); });
      groups[current].push(row);
    }
  }

  return buildBoreholes(groups);
}

// ── CPT parser — AGS CPTG group or CSV ────────────────────────────────────────
// Returns CPTLog[]: { id, x, y, groundLevel, depths[], qc[], fs[], Rf[], Ic[] }
export function parseCPT(text) {
  const isAGS = text.slice(0, 200).includes('"GROUP"') || text.slice(0, 200).includes('"CPTG"');
  if (isAGS) return _parseCPTfromAGS(text);
  return _parseCPTfromCSV(text);
}

function _parseCPTfromAGS(text) {
  const lines  = text.split(/\r?\n/);
  const groups = {};
  let current  = null;
  let headings = [];

  for (const raw of lines) {
    const line = raw.trim();
    if (!line || line.startsWith('!')) continue;
    const cells = parseAGSLine(line);
    if (!cells.length) continue;
    const tag = cells[0].toUpperCase().replace(/"/g, '');
    if (tag === 'GROUP') { current = stripQuotes(cells[1]||'').toUpperCase(); groups[current]=[]; headings=[]; }
    else if (tag === 'HEADING') { headings = cells.slice(1).map(stripQuotes); }
    else if (tag === 'DATA' && current && headings.length) {
      const row = {};
      cells.slice(1).forEach((c,i) => { row[headings[i]] = stripQuotes(c); });
      groups[current].push(row);
    }
  }

  const loca = groups['LOCA'] ?? [];
  const cptg = groups['CPTG'] ?? [];
  const logs  = {};

  loca.forEach(loc => {
    const id = loc['LOCA_ID'] || loc['HOLE_ID'] || 'CPT';
    logs[id] = {
      id, x: parseFloat(loc['LOCA_NATE']||'0'), y: parseFloat(loc['LOCA_NATN']||'0'),
      groundLevel: parseFloat(loc['LOCA_GL']||'0'),
      depths: [], qc: [], fs: [], Rf: [], Ic: [],
    };
  });

  cptg.forEach(r => {
    const id  = r['LOCA_ID'] || r['HOLE_ID'];
    let log = logs[id];
    if (!log) { log = logs[id] = { id, x: 0, y: 0, groundLevel: 0, depths: [], qc: [], fs: [], Rf: [], Ic: [] }; }
    const d  = parseFloat(r['CPTG_DPTH'] || r['DEPT'] || '0');
    const qc = parseFloat(r['CPTG_RES']  || r['QC']   || '0');  // MPa
    const fs = parseFloat(r['CPTG_FRES'] || r['FS']   || '0');  // kPa
    const Rf = qc > 0 ? (fs / 1000 / qc) * 100 : 0;             // %
    const Ic = _calcIc(qc * 1000, fs, 1.0); // normalised SBT index (approx)
    log.depths.push(d); log.qc.push(qc); log.fs.push(fs/1000); log.Rf.push(Rf); log.Ic.push(Ic);
  });

  return Object.values(logs).filter(l => l.depths.length > 0);
}

function _parseCPTfromCSV(text) {
  const rows = text.split(/\r?\n/).map(r => r.trim()).filter(Boolean);
  if (!rows.length) return [];
  const sep = rows[0].includes('\t') ? '\t' : ',';
  const hdr = rows[0].split(sep).map(h => h.trim().toLowerCase().replace(/[^a-z0-9_]/g,'_'));
  const col = k => hdr.indexOf(k);

  const logMap = {};
  for (let i = 1; i < rows.length; i++) {
    const c   = rows[i].split(sep).map(v => v.trim().replace(/^"|"$/g,''));
    const id  = c[col('id')] || c[col('cpt_id')] || c[col('hole_id')] || 'CPT-1';
    if (!logMap[id]) {
      logMap[id] = {
        id,
        x: parseFloat(c[col('x')]||c[col('easting')]||'0'),
        y: parseFloat(c[col('y')]||c[col('northing')]||'0'),
        groundLevel: parseFloat(c[col('ground_level')]||c[col('gl')]||'0'),
        depths: [], qc: [], fs: [], Rf: [], Ic: [],
      };
    }
    const log = logMap[id];
    const d   = parseFloat(c[col('depth')]||c[col('d')]||c[col('z')]||'0');
    const qc  = parseFloat(c[col('qc')]||c[col('qc_mpa')]||'0');
    const fs  = parseFloat(c[col('fs')]||c[col('fs_kpa')]||'0') / 1000; // convert to MPa if in kPa
    const Rf  = qc > 0 ? (fs / qc) * 100 : 0;
    log.depths.push(d); log.qc.push(qc); log.fs.push(fs);
    log.Rf.push(Rf); log.Ic.push(_calcIc(qc * 1000, fs * 1000, 1.0));
  }
  return Object.values(logMap).filter(l => l.depths.length > 0);
}

// Robertson SBT index Ic (approximate, σ'v ≈ 100 kPa)
function _calcIc(qcKpa, fsKpa, sigmaV_atm) {
  const Qt = (qcKpa / 1000 - sigmaV_atm) / sigmaV_atm;  // normalised tip
  const Fr = Qt > 0.001 ? (fsKpa / (qcKpa - sigmaV_atm * 1000)) * 100 : 0;
  const Ic = Math.sqrt((3.47 - Math.log10(Math.max(0.001, Qt))) ** 2 +
                       (1.22 + Math.log10(Math.max(0.001, Fr))) ** 2);
  return isFinite(Ic) ? Math.min(4, Ic) : 2;
}

function parseAGSLine(line) {
  const cells = [];
  let inQ = false, cur = '';
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') { inQ = !inQ; cur += ch; }
    else if (ch === ',' && !inQ) { cells.push(cur); cur = ''; }
    else cur += ch;
  }
  cells.push(cur);
  return cells;
}

function stripQuotes(s) {
  return s?.replace(/^"|"$/g, '').trim() ?? '';
}

function buildBoreholes(groups) {
  const boreholes = [];
  const locos = groups['LOCA'] || [];

  locos.forEach(loc => {
    const id  = loc['LOCA_ID'] || loc['HOLE_ID'] || 'BH-?';
    const x   = parseFloat(loc['LOCA_NATE'] || loc['LOCA_X'] || '0');
    const y   = parseFloat(loc['LOCA_NATN'] || loc['LOCA_Y'] || '0');
    const gl  = parseFloat(loc['LOCA_GL']   || loc['LOCA_GRND'] || '0');
    const dep = parseFloat(loc['LOCA_FDEP'] || loc['LOCA_TDEP'] || '20');

    const geolRows = (groups['GEOL'] || []).filter(r =>
      (r['LOCA_ID'] || r['HOLE_ID']) === id
    );

    const layers = geolRows.map(r => ({
      top:         parseFloat(r['GEOL_TOP']  || r['DEPT_TOP'] || '0'),
      base:        parseFloat(r['GEOL_BASE'] || r['DEPT_BOT'] || '1'),
      description: r['GEOL_DESC'] || r['GEOL_LEG'] || '',
      unitCode:    r['GEOL_UNIT'] || null,
      certainty:   null,
    }));

    // Attach SPT N values
    const isptRows = (groups['ISPT'] || []).filter(r =>
      (r['LOCA_ID'] || r['HOLE_ID']) === id
    );
    isptRows.forEach(spt => {
      const depth = parseFloat(spt['ISPT_TOP'] || spt['DEPT'] || '0');
      const nval  = parseInt(spt['ISPT_NVAL'] || spt['N'] || '0');
      layers.forEach(l => {
        if (depth >= l.top && depth < l.base) l.sptN = nval;
      });
    });

    // Groundwater strike depth from WSTB group
    let gwtDepth = null;
    const wstbRows = (groups['WSTB'] || []).filter(r =>
      (r['LOCA_ID'] || r['HOLE_ID']) === id
    );
    if (wstbRows.length) {
      const depths = wstbRows.map(r => parseFloat(r['WSTB_DPTH'] || r['DEPT'] || '')).filter(isFinite);
      if (depths.length) gwtDepth = Math.min(...depths);
    }

    if (layers.length > 0) {
      const bh = { id, x, y, groundLevel: gl, depth: dep, layers };
      if (gwtDepth != null) bh.gwtDepth = gwtDepth;

      // Drillhole deviation from TRAN group (minimum curvature survey data)
      const tranRows = (groups['TRAN'] || []).filter(r =>
        (r['LOCA_ID'] || r['HOLE_ID']) === id
      );
      if (tranRows.length >= 2) {
        bh.deviation = tranRows.map(r => ({
          depth: parseFloat(r['TRAN_DPTH'] || r['DEPT'] || '0'),
          incl:  parseFloat(r['TRAN_INCL'] || '0'),
          azim:  parseFloat(r['TRAN_AZMH'] || r['TRAN_AZIM'] || '0'),
        })).filter(s => isFinite(s.depth));
      }

      boreholes.push(bh);
    }
  });

  return boreholes;
}

// ── CSV parser ─────────────────────────────────────────────────────────────────
// Expected header row containing: id/bh_id, x/easting, y/northing, gl/ground_level,
//                                  top/depth_from, base/depth_to, description
// Flexible aliases, tab or comma separated.

export function parseCSV(text) {
  const rows = text.split(/\r?\n/).filter(l => l.trim());
  if (!rows.length) return [];

  const sep = rows[0].includes('\t') ? '\t' : ',';
  const header = rows[0].split(sep).map(h => h.trim().toLowerCase().replace(/[^a-z0-9_]/g, '_'));

  const ALIASES = {
    id:          ['bh_id','hole_id','borehole_id','id','name','location','loca_id'],
    x:           ['x','easting','x_m','local_x','coord_x','loca_nate'],
    y:           ['y','northing','y_m','local_y','coord_y','loca_natn'],
    gl:          ['ground_level','gl','ground_level_maod','reduced_level','rl','elevation','loca_gl'],
    depth:       ['total_depth','depth','final_depth','end_depth','bh_depth','loca_fdep'],
    top:         ['depth_from','top','from_m','depth_top','start_depth','from','geol_top'],
    base:        ['depth_to','base','to_m','depth_base','end_depth_m','to','geol_base'],
    description: ['description','desc','geology','lithology','log','material','unit_desc','geol_desc'],
    unit_code:   ['unit_code','unitcode','code','legend','geol_unit','unit','geol_leg','geol_geol'],
    certainty:   ['certainty','confidence','cert'],
    gwt_depth:   ['gwt_depth','gwt','water_depth','water_level','swl','standing_water_level','wstb_dpth'],
  };

  const col = name => {
    for (const alias of (ALIASES[name] || [name])) {
      const idx = header.indexOf(alias);
      if (idx !== -1) return idx;
    }
    return -1;
  };

  const boreholeMap = {};
  const _pf = (s) => { const v = parseFloat(s ?? ''); return isFinite(v) ? v : null; };

  for (let i = 1; i < rows.length; i++) {
    const cells = rows[i].split(sep).map(c => c.trim().replace(/^"|"$/g, ''));
    const id    = cells[col('id')] || `BH-${i}`;
    if (!id.trim()) continue;

    const x    = parseFloat(cells[col('x')]     ?? '') || 0;
    const y    = parseFloat(cells[col('y')]     ?? '') || 0;
    const gl   = parseFloat(cells[col('gl')]    ?? '') || 0;
    const dep  = parseFloat(cells[col('depth')] ?? '') || 20;
    const top  = parseFloat(cells[col('top')]   ?? '') || 0;
    const base = parseFloat(cells[col('base')]  ?? '') || (top + 1);
    const desc = cells[col('description')] || '';
    const code = cells[col('unit_code')]   || null;
    const cert = _pf(cells[col('certainty')]);
    const gwtD = _pf(cells[col('gwt_depth')]);
    if (!boreholeMap[id]) {
      const bh = { id, x, y, groundLevel: gl, depth: dep, layers: [] };
      if (gwtD != null && isFinite(gwtD)) bh.gwtDepth = gwtD;
      boreholeMap[id] = bh;
    } else if (gwtD != null && isFinite(gwtD) && boreholeMap[id].gwtDepth == null) {
      boreholeMap[id].gwtDepth = gwtD;
    }
    if (desc || code) {
      const layer = { top, base, description: desc, unitCode: code, certainty: cert };
      if (code) layer.classified = true;
      boreholeMap[id].layers.push(layer);
    }
  }

  const bhs = Object.values(boreholeMap);
  // A BH with unit_code on all layers can skip AI analysis
  bhs.forEach(bh => {
    if (bh.layers.length && bh.layers.every(l => l.unitCode)) bh.classified = true;
  });
  return bhs;
}

// ── Open Ground / AGS-column CSV table detection ──────────────────────────────
// Returns the AGS group name ('LOCA','GEOL','ISPT','WSTB','CPTG','TRAN') when a
// CSV header row contains AGS-style column names.  Returns null for plain CSVs
// and for flat combined exports (LOCA + GEOL cols in one file — handled by parseCSV).
function _detectTableType(headers) {
  const h = new Set(headers.map(c => c.toUpperCase().trim()));
  const hasLocaCoords = h.has('LOCA_NATE') || h.has('LOCA_NATN');
  const hasGeolDepths = h.has('GEOL_TOP')  || h.has('GEOL_BASE');
  if (hasLocaCoords && hasGeolDepths) return null; // flat combined → use parseCSV
  if (hasGeolDepths || h.has('GEOL_DESC'))                     return 'GEOL';
  if (hasLocaCoords || h.has('LOCA_GL') || h.has('LOCA_FDEP')) return 'LOCA';
  if (h.has('ISPT_TOP') || h.has('ISPT_NVAL'))                 return 'ISPT';
  if (h.has('WSTB_DPTH'))                                      return 'WSTB';
  if (h.has('CPTG_DPTH') || h.has('CPTG_RES'))                return 'CPTG';
  if (h.has('TRAN_DPTH') || h.has('TRAN_INCL'))               return 'TRAN';
  return null;
}

function _parseTableRows(text) {
  const lines = text.split(/\r?\n/).filter(l => l.trim());
  if (lines.length < 2) return null;
  const sep     = lines[0].includes('\t') ? '\t' : ',';
  const headers = lines[0].split(sep).map(h => h.trim().replace(/^"|"$/g, ''));
  const type    = _detectTableType(headers);
  if (!type) return null;
  const rows = [];
  for (let i = 1; i < lines.length; i++) {
    const cells = lines[i].split(sep).map(c => c.trim().replace(/^"|"$/g, ''));
    if (!cells.some(v => v)) continue;
    const row = {};
    headers.forEach((h, idx) => { row[h] = cells[idx] ?? ''; });
    rows.push(row);
  }
  return { type, rows };
}

// Accepts [{text, name}] — each an Open Ground table CSV export.
// Merges LOCA + GEOL + ISPT + WSTB tables and calls buildBoreholes.
function _parseOpenGroundBatch(tableDataArr) {
  const groups = {};
  for (const { text, name } of tableDataArr) {
    const parsed = _parseTableRows(text);
    if (!parsed) continue;
    if (!groups[parsed.type]) groups[parsed.type] = [];
    groups[parsed.type].push(...parsed.rows);
  }
  if (!groups['LOCA'] && !groups['GEOL']) return [];
  if (!groups['LOCA']) log('GEOL table found but no LOCA table — drop the LOCA CSV too for borehole coordinates.', 'warn');
  if (!groups['GEOL']) log('LOCA table found but no GEOL table — drop the GEOL CSV too for geology layers.', 'warn');
  return buildBoreholes(groups);
}

// ── File reading utility ───────────────────────────────────────────────────────
function readFileText(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload  = e => resolve(e.target.result);
    reader.onerror = () => reject(new Error(`Cannot read ${file.name}`));
    reader.readAsText(file);
  });
}

// ── Uploader init ─────────────────────────────────────────────────────────────
// Auto-parses when files are dropped or selected (no separate "Parse" button).
export function initUploader({ onParsed }) {
  const dropZone  = document.getElementById('drop-bh');
  const fileInput = document.getElementById('file-bh');
  const fileList  = document.getElementById('bh-file-list');

  let pendingFiles = [];

  function renderFileList() {
    if (!fileList) return;
    fileList.innerHTML = '';
    pendingFiles.forEach((f, i) => {
      const item = document.createElement('div');
      item.className = 'file-item';
      item.innerHTML = `
        <span class="file-name">${escHtml(f.name)}</span>
        <span class="file-size">${(f.size / 1024).toFixed(1)} KB</span>
        <button class="file-remove" data-i="${i}" title="Remove">×</button>`;
      fileList.appendChild(item);
    });
    fileList.querySelectorAll('.file-remove').forEach(btn => {
      btn.addEventListener('click', e => {
        pendingFiles.splice(parseInt(e.currentTarget.dataset.i), 1);
        renderFileList();
        if (dropZone) dropZone.classList.toggle('has-files', pendingFiles.length > 0);
      });
    });
  }

  async function parseAll(files) {
    let allBoreholes = [];

    // Phase 1: read all files upfront
    const readFiles = [];
    for (const file of files) {
      try {
        readFiles.push({ file, text: await readFileText(file) });
      } catch (err) {
        log(`Error reading ${escHtml(file.name)}: ${err.message}`, 'error');
      }
    }

    // Phase 2: separate Open Ground AGS-column tables from standard files
    const ogTables = [], standardFiles = [];
    for (const { file, text } of readFiles) {
      const ext = file.name.split('.').pop().toLowerCase();
      if (ext === 'csv' || ext === 'txt') {
        const firstLine = text.split(/\r?\n/).find(l => l.trim()) ?? '';
        const sep       = firstLine.includes('\t') ? '\t' : ',';
        const headers   = firstLine.split(sep).map(h => h.trim().replace(/^"|"$/g, ''));
        if (_detectTableType(headers)) { ogTables.push({ text, name: file.name }); continue; }
      }
      standardFiles.push({ file, text });
    }

    // Phase 3: batch-process Open Ground tables (LOCA + GEOL + ISPT + WSTB merged)
    if (ogTables.length) {
      const names = ogTables.map(t => escHtml(t.name)).join(', ');
      const bhs   = _parseOpenGroundBatch(ogTables);
      log(`Open Ground tables (${names}): ${bhs.length} borehole(s) parsed`, bhs.length ? 'ok' : 'warn');
      allBoreholes = allBoreholes.concat(bhs);
    }

    // Phase 4: standard AGS or generic CSV files
    for (const { file, text } of standardFiles) {
      try {
        const ext   = file.name.split('.').pop().toLowerCase();
        const isAGS = ext === 'ags' || text.slice(0, 200).includes('"GROUP"') || text.slice(0, 200).includes('"LOCA"');
        const bhs   = isAGS ? parseAGS(text) : parseCSV(text);
        log(`${escHtml(file.name)}: ${bhs.length} borehole(s) parsed`, bhs.length ? 'ok' : 'warn');
        allBoreholes = allBoreholes.concat(bhs);
      } catch (err) {
        log(`Error parsing ${escHtml(file.name)}: ${err.message}`, 'error');
      }
    }

    if (allBoreholes.length) {
      window.dispatchEvent(new CustomEvent('geomodel:data-loaded', { detail: { boreholes: allBoreholes } }));
      onParsed(allBoreholes);
    } else {
      log('No boreholes extracted from dropped files.', 'warn');
    }
  }

  function addFiles(files) {
    for (const f of files) {
      if (!pendingFiles.find(p => p.name === f.name)) pendingFiles.push(f);
    }
    renderFileList();
    if (dropZone) dropZone.classList.toggle('has-files', pendingFiles.length > 0);
    if (files.length) parseAll(files);
  }

  if (dropZone) {
    dropZone.addEventListener('click', () => fileInput?.click());
    dropZone.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') fileInput?.click(); });
    dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
    dropZone.addEventListener('drop', e => {
      e.preventDefault();
      dropZone.classList.remove('drag-over');
      addFiles(Array.from(e.dataTransfer?.files ?? []));
    });
  }

  if (fileInput) {
    fileInput.addEventListener('change', () => {
      addFiles(Array.from(fileInput.files));
      fileInput.value = '';
    });
  }
}

function escHtml(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
