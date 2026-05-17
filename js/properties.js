// ── Geotechnical Unit Properties Editor ───────────────────────────────────────
// Renders an editable table of engineering parameters per geological unit.
// Parameters are stored on geoUnit objects as unit.params = {…}.

// ICS standard geological period colours (simplified)
export const GEO_PERIODS = [
  { name: '—',             code: '',    color: null    },
  { name: 'Quaternary',    code: 'Q',   color: '#f9f97f' },
  { name: 'Holocene',      code: 'Qh',  color: '#fff2ae' },
  { name: 'Pleistocene',   code: 'Qp',  color: '#fff987' },
  { name: 'Neogene',       code: 'Ng',  color: '#ffe619' },
  { name: 'Palaeogene',    code: 'Pg',  color: '#fd9a52' },
  { name: 'Cretaceous',    code: 'K',   color: '#7fc64e' },
  { name: 'Jurassic',      code: 'J',   color: '#34b2c9' },
  { name: 'Triassic',      code: 'Tr',  color: '#812b92' },
  { name: 'Permian',       code: 'P',   color: '#f04028' },
  { name: 'Carboniferous', code: 'C',   color: '#67a599' },
  { name: 'Devonian',      code: 'D',   color: '#cb8c37' },
  { name: 'Silurian',      code: 'S',   color: '#b3e1b6' },
  { name: 'Ordovician',    code: 'O',   color: '#009270' },
  { name: 'Cambrian',      code: 'Cm',  color: '#7fa056' },
  { name: 'Precambrian',   code: 'pC',  color: '#f74370' },
  { name: 'Made Ground',   code: 'MG',  color: '#b5b5b5' },
  { name: 'Alluvium',      code: 'Al',  color: '#daf0e3' },
  { name: 'River Terrace', code: 'RT',  color: '#f9ddb9' },
];

const DEFAULT_PARAMS = {
  gamma:  null,   // bulk unit weight (kN/m³)
  cu:     null,   // undrained shear strength (kPa)
  cprime: null,   // effective cohesion (kPa)
  phi:    null,   // effective friction angle (°)
  E:      null,   // Young's modulus (MPa)
  nu:     null,   // Poisson's ratio
  Cc:     null,   // compression index
  e0:     null,   // initial void ratio (for settlement)
  cv:     null,   // coefficient of consolidation (m²/yr)
  N_spt:  null,   // characteristic SPT N value
};

export function initProperties() {
  // Properties tab is rendered on demand
}

export function renderPropertiesTable(geoUnits, onUpdate) {
  const wrap = document.getElementById('props-table-wrap');
  if (!wrap) return;
  if (!geoUnits.length) {
    wrap.innerHTML = '<p class="hint" style="padding:8px">Load data to see unit properties.</p>';
    return;
  }

  // Ensure all units have a params object and a geom descriptor object
  geoUnits.forEach(u => {
    if (!u.params) u.params = { ...DEFAULT_PARAMS };
    if (!u.geom)   u.geom   = {};
  });

  const cols = [
    { key: 'gamma',  label: 'γ (kN/m³)',  placeholder: '19'  },
    { key: 'cu',     label: 'Cu (kPa)',    placeholder: '—'   },
    { key: 'cprime', label: "c' (kPa)",    placeholder: '0'   },
    { key: 'phi',    label: "φ' (°)",      placeholder: '—'   },
    { key: 'E',      label: 'E (MPa)',     placeholder: '—'   },
    { key: 'Cc',     label: 'Cc',          placeholder: '—'   },
    { key: 'e0',     label: 'e₀',          placeholder: '0.8' },
    { key: 'cv',     label: 'cv (m²/yr)', placeholder: '1'   },
    { key: 'N_spt',  label: 'N (SPT)',     placeholder: '—'   },
  ];

  // Geometry descriptor columns (stored in unit.geom)
  const geomCols = [
    { key: 'corrLength',  label: 'L (m)',    placeholder: 'auto', title: 'Per-unit correlation length (m) — overrides global variogram range for this unit' },
    { key: 'anisoRatio',  label: 'AR',       placeholder: '1',    title: 'Anisotropy ratio for this unit (along-strike vs across-strike search)' },
    { key: 'anisoAzimuth',label: 'Az°',      placeholder: '0',    title: 'Anisotropy strike azimuth for this unit (° from North)' },
  ];

  const periodOpts = GEO_PERIODS.map(p =>
    `<option value="${p.code}">${p.name}</option>`).join('');

  let html = `<table class="props-table">
    <thead><tr>
      <th>Code</th><th>Name</th><th title="Geological age / period (ICS)">Period</th>
      ${cols.map(c => `<th>${c.label}</th>`).join('')}
      <th class="props-geom-sep" colspan="${geomCols.length}" title="Per-unit geostatistical geometry overrides">Geometry ▾</th>
    </tr><tr class="props-geom-subrow">
      <th colspan="3"></th>
      ${cols.map(() => '<th></th>').join('')}
      ${geomCols.map(c => `<th title="${c.title}">${c.label}</th>`).join('')}
    </tr></thead>
    <tbody>`;

  for (const u of geoUnits) {
    const curPeriod = u.period ?? '';
    const pDef = GEO_PERIODS.find(p => p.code === curPeriod);
    const pColor = pDef?.color ? `background:${pDef.color};` : '';
    html += `<tr data-uid="${u.id}">
      <td><span class="props-swatch" style="background:${u.color}"></span>${_esc(u.code)}</td>
      <td class="props-name-cell" contenteditable="true" data-field="name">${_esc(u.name)}</td>
      <td><select class="props-period-sel" data-uid="${u.id}" style="font-size:9px;${pColor}">
        ${GEO_PERIODS.map(p => `<option value="${p.code}"${p.code===curPeriod?' selected':''}>${p.name}</option>`).join('')}
      </select></td>
      ${cols.map(c => {
        const val = u.params[c.key] ?? '';
        return `<td><input class="props-input" type="number" data-uid="${u.id}"
          data-field="${c.key}" data-store="params" value="${val}" placeholder="${c.placeholder}"
          min="0" step="any"></td>`;
      }).join('')}
      ${geomCols.map(c => {
        const val = u.geom?.[c.key] ?? '';
        return `<td class="props-geom-cell"><input class="props-input" type="number" data-uid="${u.id}"
          data-field="${c.key}" data-store="geom" value="${val}" placeholder="${c.placeholder}"
          min="0" step="any" title="${c.title}"></td>`;
      }).join('')}
    </tr>`;
  }
  html += '</tbody></table>';
  wrap.innerHTML = html;

  // Wire period selectors
  wrap.querySelectorAll('.props-period-sel').forEach(sel => {
    sel.addEventListener('change', () => {
      const uid  = parseInt(sel.dataset.uid);
      const unit = geoUnits.find(u => u.id === uid);
      if (!unit) return;
      unit.period = sel.value;
      const pDef = GEO_PERIODS.find(p => p.code === sel.value);
      sel.style.background = pDef?.color ?? '';
      onUpdate?.();
    });
  });

  // Wire name editing
  wrap.querySelectorAll('[contenteditable][data-field="name"]').forEach(cell => {
    const row  = cell.closest('tr');
    const uid  = parseInt(row.dataset.uid);
    const unit = geoUnits.find(u => u.id === uid);
    if (!unit) return;
    cell.addEventListener('blur', () => {
      const newName = cell.textContent.trim();
      if (newName && newName !== unit.name) {
        unit.name = newName;
        onUpdate?.();
      }
    });
    cell.addEventListener('keydown', e => {
      if (e.key === 'Enter') { e.preventDefault(); cell.blur(); }
    });
  });

  // Wire parameter + geometry inputs
  wrap.querySelectorAll('.props-input').forEach(input => {
    input.addEventListener('change', () => {
      const uid   = parseInt(input.dataset.uid);
      const field = input.dataset.field;
      const store = input.dataset.store ?? 'params';
      const unit  = geoUnits.find(u => u.id === uid);
      if (!unit) return;
      const val = input.value.trim();
      const num = val === '' ? null : parseFloat(val);
      if (store === 'geom') {
        if (!unit.geom) unit.geom = {};
        unit.geom[field] = num;
      } else {
        unit.params[field] = num;
      }
      onUpdate?.();
    });
  });
}

export function exportPropertiesCSV(geoUnits) {
  const cols = ['code','name','gamma_kNm3','cu_kPa','cprime_kPa','phi_deg','E_MPa','Cc','e0','N_spt'];
  const rows = [cols.join(',')];
  for (const u of geoUnits) {
    const p = u.params ?? {};
    rows.push([
      u.code,
      `"${(u.name ?? '').replace(/"/g, '""')}"`,
      p.gamma ?? '',
      p.cu     ?? '',
      p.cprime ?? '',
      p.phi    ?? '',
      p.E      ?? '',
      p.Cc     ?? '',
      p.e0     ?? '',
      p.N_spt  ?? '',
    ].join(','));
  }
  const blob = new Blob([rows.join('\n')], { type: 'text/csv' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href = url; a.download = 'unit-properties.csv'; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
}

// ── BS5930 / BGS colour presets ────────────────────────────────────────────────
// Applies standard UK geological display colours based on unit code and name.
// Matches by code prefix (case-insensitive), then by geological keyword in name.
const _BS_BY_CODE = {
  mg: '#7a6a42', fill: '#7a6a42', fil: '#7a6a42', made: '#7a6a42',
  lc: '#5b6a8a', lcla: '#5b6a8a',
  rtd: '#D4A843', tg: '#c8a855',
  grvl: '#c8aa60', grv: '#c8aa60', sag: '#c8aa60',
  sa: '#e8d5a0', snd: '#e8d5a0', sand: '#e8d5a0',
  acl: '#4A7C59', al: '#4A7C59', all: '#4A7C59',
  ch: '#f0ede0', ck: '#e8e4d8', chlk: '#e8e4d8',
  wck: '#c8b89a', wch: '#c8b89a',
  ts: '#5c3d1e', top: '#5c3d1e',
  ss: '#b8603a', sdst: '#b8603a',
  ms: '#6a7a8a', mdst: '#6a7a8a', mds: '#6a7a8a',
  ls: '#b8c8d0', lmst: '#b8c8d0',
  pt: '#3a2a1a', peat: '#3a2a1a',
  cl: '#8a90a0', cly: '#8a90a0',
  sl: '#a0b090', slt: '#a0b090',
  rk: '#808080', rock: '#808080',
  scl: '#7a88a0',
  hd: '#c8b090', head: '#c8b090',
};
const _BS_BY_KEYWORD = [
  ['made ground', '#7a6a42'], ['fill', '#7a6a42'],
  ['london clay', '#5b6a8a'],
  ['river terrace', '#D4A843'],
  ['gravel', '#c8aa60'], ['sand', '#e8d5a0'],
  ['alluvial', '#4A7C59'], ['alluvium', '#4A7C59'],
  ['chalk', '#e8e4d8'],
  ['sandstone', '#b8603a'],
  ['mudstone', '#6a7a8a'],
  ['limestone', '#b8c8d0'],
  ['peat', '#3a2a1a'],
  ['silt', '#a0b090'],
  ['topsoil', '#5c3d1e'],
  ['clay', '#8a90a0'],
  ['rock', '#808080'],
];

export function applyBS5930Colors(geoUnits) {
  let matched = 0;
  for (const u of geoUnits) {
    const codeKey = u.code.toLowerCase().replace(/[^a-z]/g, '');
    if (_BS_BY_CODE[codeKey]) {
      u.color = _BS_BY_CODE[codeKey];
      matched++;
      continue;
    }
    const nameLow = (u.name ?? '').toLowerCase();
    const found = _BS_BY_KEYWORD.find(([kw]) => nameLow.includes(kw));
    if (found) { u.color = found[1]; matched++; }
  }
  return matched;
}

function _esc(s) {
  return String(s ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
