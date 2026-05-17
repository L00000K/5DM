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

// ── Terzaghi 1D consolidation time-rate ────────────────────────────────────────
// Uses Tv = cv × t / Hdr² → degree of consolidation U (Terzaghi approximation)
// Returns { times[], U[], t50, t90 } where U is 0–1, times in years.
export function consolidationTimeCurve(settlementResult) {
  if (!settlementResult) return null;

  // Find cv-bearing compressible layers
  const compLayers = settlementResult.layers.filter(l =>
    l.settlement > 0 && l.unit.params?.cv != null);
  if (!compLayers.length) return null;

  // Weighted-average cv and total drainage height (one-way drainage assumed)
  let totalH = 0, cvNum = 0;
  for (const l of compLayers) {
    totalH += l.H;
    cvNum  += l.unit.params.cv * l.H;
  }
  const cvEff = cvNum / totalH;   // m²/yr
  const Hdr   = totalH;           // one-way drainage path = full thickness

  // Terzaghi Tv → U approximation (Barron 1948 two-term)
  function _U(Tv) {
    if (Tv <= 0) return 0;
    if (Tv < 0.217) return Math.sqrt(4 * Tv / Math.PI);      // parabolic (early)
    return 1 - Math.exp(-Math.PI * Math.PI / 4 * Tv) * 1.781; // exponential (late)
  }
  function _clamp(v) { return Math.min(1, Math.max(0, v)); }

  // Log-spaced time points 0.001–100 years
  const times = [];
  for (let exp = -3; exp <= 2; exp += 0.05) {
    times.push(Math.pow(10, exp));
  }
  const U = times.map(t => _clamp(_U(cvEff * t / (Hdr * Hdr))));

  const t50 = times.find((t, i) => U[i] >= 0.5) ?? null;
  const t90 = times.find((t, i) => U[i] >= 0.9) ?? null;

  return { times, U, t50, t90, cvEff, Hdr, totalSettlement: settlementResult.total };
}

export function renderConsolidationCurve(curve, width = 260, height = 140) {
  if (!curve) return '';
  const { times, U, t50, t90, totalSettlement } = curve;

  const pad = { t: 10, r: 10, b: 28, l: 36 };
  const W = width - pad.l - pad.r, H = height - pad.t - pad.b;

  const xScale = t => pad.l + (Math.log10(t) + 3) / 5 * W; // 0.001–100 yr
  const yScale = u => pad.t + (1 - u) * H;

  let path = times.map((t, i) => `${i===0?'M':'L'}${xScale(t)},${yScale(U[i])}`).join('');

  const xTicks = [0.01, 0.1, 1, 10, 100];
  const yTicks = [0, 0.25, 0.5, 0.75, 1.0];

  const t50X = t50 ? xScale(t50) : null;
  const t90X = t90 ? xScale(t90) : null;

  return `<svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg"
    style="font-family:monospace;font-size:8.5px;display:block">
    ${yTicks.map(u => `<line x1="${pad.l}" y1="${yScale(u)}" x2="${pad.l+W}" y2="${yScale(u)}"
      stroke="#333" stroke-width="0.5"/>`).join('')}
    ${t50X ? `<line x1="${t50X}" y1="${pad.t}" x2="${t50X}" y2="${pad.t+H}"
      stroke="#f1c40f" stroke-width="1" stroke-dasharray="3,2"/>
      <text x="${t50X}" y="${pad.t+H+10}" text-anchor="middle" fill="#f1c40f">${t50<1?t50.toFixed(2):t50.toFixed(1)}yr</text>` : ''}
    ${t90X ? `<line x1="${t90X}" y1="${pad.t}" x2="${t90X}" y2="${pad.t+H}"
      stroke="#e67e22" stroke-width="1" stroke-dasharray="3,2"/>
      <text x="${t90X}" y="${pad.t+H+10}" text-anchor="middle" fill="#e67e22">${t90<1?t90.toFixed(2):t90.toFixed(1)}yr</text>` : ''}
    <path d="${path}" fill="none" stroke="#5b9bd5" stroke-width="2"/>
    ${yTicks.map(u => `<text x="${pad.l-4}" y="${yScale(u)+3}" text-anchor="end" fill="#aaa">${(u*100).toFixed(0)}%</text>`).join('')}
    ${xTicks.map(t => `<text x="${xScale(t)}" y="${pad.t+H+9}" text-anchor="middle" fill="#aaa">${t<1?t:t+'y'}</text>`).join('')}
    <text x="${pad.l-28}" y="${pad.t+H/2}" transform="rotate(-90,${pad.l-28},${pad.t+H/2})"
      text-anchor="middle" fill="#ccc" font-size="8">Consolidation U</text>
    <text x="${pad.l+W/2}" y="${height}" text-anchor="middle" fill="#ccc" font-size="8">Time (years)</text>
    <text x="${pad.l+W}" y="${pad.t+10}" text-anchor="end" fill="#5b9bd5" font-size="8">ρ_total=${totalSettlement.toFixed(0)}mm</text>
  </svg>`;
}

function _esc(s) {
  return String(s ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
