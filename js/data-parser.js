import { log, setEnabled, populateDataTable, drawPlanView } from './app.js';

// ── AGS 4.x parser ─────────────────────────────────────────────────────────────
// Reads key groups: LOCA, GEOL, ISPT
// Returns BHLog[]

export function parseAGS(text) {
  const lines  = text.split(/\r?\n/);
  const groups = {};
  let current  = null;
  let headings = [];
  let units    = [];
  let types    = [];

  for (const raw of lines) {
    const line = raw.trim();
    if (!line || line.startsWith('!')) continue; // blank / comment

    const cells = parseAGSLine(line);
    if (!cells.length) continue;

    const tag = cells[0].toUpperCase();

    if (tag === '"GROUP"' || tag === 'GROUP') {
      current = stripQuotes(cells[1] || '').toUpperCase();
      groups[current] = [];
      headings = []; units = []; types = [];
    } else if (tag === '"HEADING"' || tag === 'HEADING') {
      headings = cells.slice(1).map(stripQuotes);
    } else if (tag === '"UNIT"' || tag === 'UNIT') {
      units = cells.slice(1).map(stripQuotes);
    } else if (tag === '"TYPE"' || tag === 'TYPE') {
      types = cells.slice(1).map(stripQuotes);
    } else if ((tag === '"DATA"' || tag === 'DATA') && current && headings.length) {
      const row = {};
      cells.slice(1).forEach((c, i) => {
        row[headings[i]] = stripQuotes(c);
      });
      groups[current].push(row);
    }
  }

  return buildBoreholes(groups);
}

function parseAGSLine(line) {
  // AGS uses comma-separated, values may be quoted
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

    // Get geological layers for this borehole
    const geolRows = (groups['GEOL'] || []).filter(r =>
      (r['LOCA_ID'] || r['HOLE_ID']) === id
    );

    const layers = geolRows.map(r => ({
      top:         parseFloat(r['GEOL_TOP'] || r['DEPT_TOP'] || '0'),
      base:        parseFloat(r['GEOL_BASE'] || r['DEPT_BOT'] || '1'),
      description: r['GEOL_DESC'] || r['GEOL_LEG'] || '',
      unitCode:    null,
      certainty:   null,
    }));

    // Attach SPT N values if present
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

    boreholes.push({ id, x, y, groundLevel: gl, depth: dep, layers });
  });

  return boreholes;
}

// ── CSV parser ─────────────────────────────────────────────────────────────────
// Expected columns: BH_ID, x, y, ground_level, depth_from, depth_to, description
// (header row detection is flexible)

export function parseCSV(text) {
  const lines = text.split(/\r?\n/).filter(l => l.trim());
  if (!lines.length) return [];

  const header = lines[0].split(',').map(h => h.trim().toLowerCase().replace(/[^a-z0-9_]/g, '_'));

  const col = name => {
    const aliases = {
      id:          ['bh_id','hole_id','borehole_id','id','name'],
      x:           ['x','easting','x_m','northing_e','local_x'],
      y:           ['y','northing','y_m','northing_n','local_y'],
      gl:          ['ground_level','gl','ground_level_maod','reduced_level','rl'],
      depth:       ['total_depth','depth','final_depth','end_depth'],
      top:         ['depth_from','top','from_m','depth_top','start_depth'],
      base:        ['depth_to','base','to_m','depth_base','end_depth_m'],
      description: ['description','desc','geology','lithology','log','material'],
    };
    for (const alias of (aliases[name] || [name])) {
      const idx = header.indexOf(alias);
      if (idx !== -1) return idx;
    }
    return -1;
  };

  const boreholeMap = {};

  for (let i = 1; i < lines.length; i++) {
    const cells = lines[i].split(',').map(c => c.trim());
    const id   = cells[col('id')]  || `BH-${i}`;
    const x    = parseFloat(cells[col('x')]  || '0');
    const y    = parseFloat(cells[col('y')]  || '0');
    const gl   = parseFloat(cells[col('gl')] || '0');
    const dep  = parseFloat(cells[col('depth')] || '20');
    const top  = parseFloat(cells[col('top')]  || '0');
    const base = parseFloat(cells[col('base')] || '1');
    const desc = cells[col('description')] || '';

    if (!boreholeMap[id]) {
      boreholeMap[id] = { id, x, y, groundLevel: gl, depth: dep, layers: [] };
    }
    if (desc) {
      boreholeMap[id].layers.push({ top, base, description: desc, unitCode: null, certainty: null });
    }
  }

  return Object.values(boreholeMap);
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

// ── Uploader init ──────────────────────────────────────────────────────────────
export function initUploader({ onParsed }) {
  const dropZone  = document.getElementById('drop-bh');
  const fileInput = document.getElementById('file-bh');
  const fileList  = document.getElementById('bh-file-list');
  const btnParse  = document.getElementById('btn-parse');

  let pendingFiles = [];

  function renderFileList() {
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
        const i = parseInt(e.currentTarget.dataset.i);
        pendingFiles.splice(i, 1);
        renderFileList();
        dropZone.classList.toggle('has-files', pendingFiles.length > 0);
        setEnabled('btn-parse', pendingFiles.length > 0);
      });
    });
  }

  function addFiles(files) {
    for (const f of files) {
      if (!pendingFiles.find(p => p.name === f.name)) pendingFiles.push(f);
    }
    renderFileList();
    dropZone.classList.toggle('has-files', pendingFiles.length > 0);
    setEnabled('btn-parse', pendingFiles.length > 0);
  }

  dropZone.addEventListener('click',  () => fileInput.click());
  dropZone.addEventListener('keydown', e => { if (e.key === 'Enter') fileInput.click(); });

  dropZone.addEventListener('dragover', e => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
  });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    addFiles(Array.from(e.dataTransfer.files));
  });

  fileInput.addEventListener('change', () => {
    addFiles(Array.from(fileInput.files));
    fileInput.value = '';
  });

  btnParse.addEventListener('click', async () => {
    if (!pendingFiles.length) return;
    setEnabled('btn-parse', false);
    log(`Parsing ${pendingFiles.length} file(s)…`, 'info');

    let allBoreholes = [];
    for (const file of pendingFiles) {
      try {
        const text = await readFileText(file);
        const ext  = file.name.split('.').pop().toLowerCase();
        const bhs  = ext === 'ags' || text.includes('"GROUP"') || text.includes('"LOCA"')
          ? parseAGS(text)
          : parseCSV(text);

        log(`${file.name}: ${bhs.length} borehole(s) parsed`, 'ok');
        allBoreholes = allBoreholes.concat(bhs);
      } catch (err) {
        log(`Error parsing ${file.name}: ${err.message}`, 'error');
      }
    }

    if (allBoreholes.length) {
      window.dispatchEvent(new CustomEvent('geomodel:data-loaded', {
        detail: { boreholes: allBoreholes }
      }));
      onParsed(allBoreholes);
    } else {
      log('No boreholes extracted from files.', 'warn');
    }
    setEnabled('btn-parse', true);
  });
}

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
