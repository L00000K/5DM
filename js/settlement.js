// ── 1D Consolidation Settlement Estimator ─────────────────────────────────────
// Terzaghi/Skempton 1D consolidation:
//   ρ_i = (Cc_i / (1+e0_i)) × H_i × log10((σ'v0 + Δσ) / σ'v0)
//
// Inputs per unit: Cc (compression index), e0 (initial void ratio), gamma (kN/m³)
// Δσ approximated as net applied pressure (uniform load) throughout.

export function calculateSettlement(grid, geoUnits, foundElev, appliedKPa) {
  if (!grid || !geoUnits.length) return null;
  const { nx, ny, nz, cellSize: cs, cellHeight: ch, origin: O, unitIds } = grid;

  const cx = Math.floor(nx / 2);
  const cy = Math.floor(ny / 2);

  const unitById = {};
  geoUnits.forEach(u => { unitById[u.id] = u; });

  // Collect layers from foundation depth to base of model
  const rawLayers = [];
  let sigmaV0 = 0;

  // Work from the top down through the profile at the model centre
  for (let iz = nz - 1; iz >= 0; iz--) {
    const layerTopZ = O.y + (iz + 1) * ch;
    const layerMidZ = O.y + iz * ch + ch * 0.5;
    if (layerMidZ >= foundElev) {
      // Above foundation — only contributes to overburden
      const uid  = unitIds[cx + cy * nx + iz * nx * ny];
      const unit = unitById[uid];
      const gam  = unit?.params?.gamma ?? 19;
      sigmaV0 += gam * ch;
      continue;
    }
    const uid  = unitIds[cx + cy * nx + iz * nx * ny];
    const unit = unitById[uid];
    if (!unit) continue;

    rawLayers.push({
      unit,
      H:       ch,
      topZ:    layerTopZ,
      sigmaV0: sigmaV0 + (unit.params?.gamma ?? 19) * ch * 0.5,
    });
    sigmaV0 += (unit.params?.gamma ?? 19) * ch;
  }

  // Merge consecutive same-unit layers
  const merged = [];
  for (const l of rawLayers) {
    const prev = merged[merged.length - 1];
    if (prev && prev.unit.id === l.unit.id) {
      prev.H       += l.H;
      prev.sigmaV0  = (prev.sigmaV0 + l.sigmaV0) / 2;
    } else {
      merged.push({ ...l });
    }
  }

  // Calculate settlement per layer
  let total = 0;
  const layers = merged.map(l => {
    const Cc = l.unit.params?.Cc ?? null;
    const e0 = l.unit.params?.e0 ?? null;
    let settlement = null;
    if (Cc !== null && e0 !== null && l.sigmaV0 > 0) {
      const ratio = (l.sigmaV0 + appliedKPa) / l.sigmaV0;
      if (ratio > 1) {
        settlement = (Cc / (1 + e0)) * l.H * 1000 * Math.log10(ratio); // mm
        total += settlement;
      }
    }
    return { ...l, Cc, e0, settlement };
  });

  return { layers, total, foundElev, appliedKPa };
}

export function renderSettlementResults(result, containerEl) {
  if (!containerEl) return;
  if (!result) {
    containerEl.innerHTML = '<p class="hint">Set unit Cc and e0 in the Props tab, then calculate.</p>';
    return;
  }

  const { layers, total } = result;
  const hasCalc = layers.some(l => l.settlement !== null);

  let html = `<div class="sett-total ${hasCalc ? '' : 'sett-no-data'}">
    <span class="sett-total-lbl">Total settlement</span>
    <span class="sett-total-val">${hasCalc ? total.toFixed(1) + ' mm' : '— (set Cc, e0)'}</span>
  </div>
  <table class="sett-table">
    <tr><th>Unit</th><th>H (m)</th><th>σ'v0 (kPa)</th><th>Cc</th><th>e0</th><th>ρ (mm)</th></tr>`;

  for (const l of layers) {
    const sVal = l.settlement !== null ? l.settlement.toFixed(1) : '—';
    const missing = l.Cc === null || l.e0 === null;
    html += `<tr class="${missing ? 'sett-missing' : ''}">
      <td><span class="sett-swatch" style="background:${l.unit.color}"></span>${_esc(l.unit.code)}</td>
      <td>${l.H.toFixed(2)}</td>
      <td>${l.sigmaV0.toFixed(0)}</td>
      <td>${l.Cc ?? '—'}</td>
      <td>${l.e0 ?? '—'}</td>
      <td class="sett-val">${sVal}</td>
    </tr>`;
  }
  html += '</table>';
  containerEl.innerHTML = html;
}

function _esc(s) {
  return String(s ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
