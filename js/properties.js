// ── Geotechnical Unit Properties Editor ───────────────────────────────────────
// Renders an editable table of engineering parameters per geological unit.
// Parameters are stored on geoUnit objects as unit.params = {…}.

const DEFAULT_PARAMS = {
  gamma:  null,   // bulk unit weight (kN/m³)
  cu:     null,   // undrained shear strength (kPa)
  cprime: null,   // effective cohesion (kPa)
  phi:    null,   // effective friction angle (°)
  E:      null,   // Young's modulus (MPa)
  nu:     null,   // Poisson's ratio
  Cc:     null,   // compression index
  e0:     null,   // initial void ratio (for settlement)
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

  // Ensure all units have a params object
  geoUnits.forEach(u => {
    if (!u.params) u.params = { ...DEFAULT_PARAMS };
  });

  const cols = [
    { key: 'gamma',  label: 'γ (kN/m³)',  placeholder: '19'  },
    { key: 'cu',     label: 'Cu (kPa)',    placeholder: '—'   },
    { key: 'cprime', label: "c' (kPa)",    placeholder: '0'   },
    { key: 'phi',    label: "φ' (°)",      placeholder: '—'   },
    { key: 'E',      label: 'E (MPa)',     placeholder: '—'   },
    { key: 'Cc',     label: 'Cc',          placeholder: '—'   },
    { key: 'e0',     label: 'e₀',          placeholder: '0.8' },
    { key: 'N_spt',  label: 'N (SPT)',     placeholder: '—'   },
  ];

  let html = `<table class="props-table">
    <thead><tr>
      <th>Code</th><th>Name</th>
      ${cols.map(c => `<th>${c.label}</th>`).join('')}
    </tr></thead>
    <tbody>`;

  for (const u of geoUnits) {
    html += `<tr data-uid="${u.id}">
      <td><span class="props-swatch" style="background:${u.color}"></span>${_esc(u.code)}</td>
      <td class="props-name-cell" contenteditable="true" data-field="name">${_esc(u.name)}</td>
      ${cols.map(c => {
        const val = u.params[c.key] ?? '';
        return `<td><input class="props-input" type="number" data-uid="${u.id}"
          data-field="${c.key}" value="${val}" placeholder="${c.placeholder}"
          min="0" step="any"></td>`;
      }).join('')}
    </tr>`;
  }
  html += '</tbody></table>';
  wrap.innerHTML = html;

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

  // Wire parameter inputs
  wrap.querySelectorAll('.props-input').forEach(input => {
    input.addEventListener('change', () => {
      const uid   = parseInt(input.dataset.uid);
      const field = input.dataset.field;
      const unit  = geoUnits.find(u => u.id === uid);
      if (!unit) return;
      const val = input.value.trim();
      unit.params[field] = val === '' ? null : parseFloat(val);
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

function _esc(s) {
  return String(s ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
