// ── Laboratory Data CSV Importer ───────────────────────────────────────────────
// CSV format (one value per row):
//   Unit_Code, Test_Type, Value
//
// Supported Test_Type values (case-insensitive):
//   Cu / su / undrained        → unit.params.cu   (kPa)
//   phi / friction / friction_angle → unit.params.phi (°)
//   gamma / unit_weight / density   → unit.params.gamma (kN/m³)
//   cprime / c_prime / cohesion     → unit.params.cprime (kPa)
//   E / youngs / modulus            → unit.params.E (MPa)
//   Cc / comp_index                 → unit.params.Cc
//   e0 / void_ratio                 → unit.params.e0
//   N_spt / spt / N                 → unit.params.N_spt
//   nu / poisson                    → unit.params.nu
//
// Multiple rows per unit+test are averaged.

const FIELD_MAP = [
  [/^(cu|su|undrained)/i,       'cu'],
  [/^(phi|friction)/i,          'phi'],
  [/^(gamma|unit.?weight|density)/i, 'gamma'],
  [/^(c[_-]?prime|cohesion)/i,  'cprime'],
  [/^(e|youngs|modulus)/i,      'E'],
  [/^(cc|comp)/i,               'Cc'],
  [/^(e0|void)/i,               'e0'],
  [/^(n[_-]?spt|spt|^n$)/i,    'N_spt'],
  [/^(nu|poisson)/i,            'nu'],
];

function mapField(raw) {
  const r = raw.trim();
  for (const [re, field] of FIELD_MAP) {
    if (re.test(r)) return field;
  }
  return null;
}

export function parseLabCSV(text, geoUnits) {
  const lines  = text.split('\n').map(l => l.trim()).filter(Boolean);
  const unitByCode = {};
  geoUnits.forEach(u => { unitByCode[u.code.toLowerCase()] = u; });

  // accumulators[unitId][field] = [value, ...]
  const acc = {};
  geoUnits.forEach(u => { acc[u.id] = {}; });

  let parsed = 0, skipped = 0;

  for (const line of lines) {
    if (line.startsWith('#') || line.startsWith('//')) continue;

    const parts = line.split(/[,\t;]+/).map(p => p.trim());
    if (parts.length < 3) { skipped++; continue; }

    // Try to find the header row and skip it
    if (/unit.?code|unit_code|code/i.test(parts[0])) continue;

    const code  = parts[0].toLowerCase();
    const type  = parts[1];
    const val   = parseFloat(parts[2]);

    if (isNaN(val)) { skipped++; continue; }

    const unit = unitByCode[code];
    if (!unit) { skipped++; continue; }

    const field = mapField(type);
    if (!field) { skipped++; continue; }

    if (!acc[unit.id][field]) acc[unit.id][field] = [];
    acc[unit.id][field].push(val);
    parsed++;
  }

  // Average and apply to unit params
  const updated = [];
  geoUnits.forEach(u => {
    if (!u.params) u.params = {};
    const fields = acc[u.id] ?? {};
    const changedFields = [];
    for (const [field, vals] of Object.entries(fields)) {
      if (!vals.length) continue;
      const avg = vals.reduce((a, b) => a + b, 0) / vals.length;
      u.params[field] = +avg.toFixed(4);
      changedFields.push(`${field}=${avg.toFixed(2)}`);
    }
    if (changedFields.length) updated.push(`${u.code}: ${changedFields.join(', ')}`);
  });

  return { parsed, skipped, updated };
}
