// ── Bearing Capacity Calculator ────────────────────────────────────────────────
// Undrained:  Skempton's Nc method (φ=0 condition)
// Drained:    Meyerhof's general bearing capacity equation with shape/depth factors

export function calculateBearingCapacity(grid, geoUnits, foundElev, B, L, z, FS) {
  if (!grid || !geoUnits.length) return null;
  const { nx, ny, nz, cellHeight: ch, origin: O, unitIds } = grid;

  // Identify bearing unit at foundation level from model centre column
  const cx = Math.floor(nx / 2);
  const cy = Math.floor(ny / 2);
  let bearingUnit = null;
  for (let iz = nz - 1; iz >= 0; iz--) {
    if (O.y + iz * ch + ch * 0.5 <= foundElev) {
      const uid  = unitIds[cx + cy * nx + iz * nx * ny];
      const unit = geoUnits.find(u => u.id === uid);
      if (unit) { bearingUnit = unit; break; }
    }
  }
  if (!bearingUnit) {
    return { error: 'No geological unit found at the foundation level — adjust Foundation Z (mAOD).' };
  }

  const p      = bearingUnit.params ?? {};
  const gamma  = p.gamma  ?? 19;     // bulk unit weight kN/m³
  const cu     = p.cu     ?? null;   // undrained shear strength kPa
  const cprime = p.cprime ?? null;   // effective cohesion kPa
  const phi    = p.phi    ?? null;   // effective friction angle °

  const q = gamma * z;               // overburden at foundation base (kPa)
  const result = { unit: bearingUnit, B, L, z, FS, q_overburden: q };

  // ── Undrained — Skempton (1951) ─────────────────────────────────────────────
  if (cu !== null) {
    // Shape factor: 1 + 0.2(B/L)  [Skempton]
    const sf   = 1 + 0.2 * (B / Math.max(L, B));
    // Depth factor: 1 + 0.4(z/B), capped at z/B = 1
    const df   = 1 + 0.4 * Math.min(z / B, 1.0);
    // Nc_eff = 5.14 × sf × df, capped per Skempton's original table (max ≈ 9)
    const Nc   = Math.min(5.14 * sf * df, 9.0);
    const qnet = cu * Nc;
    const qa   = qnet / FS;
    result.undrained = {
      cu, Nc: +Nc.toFixed(2),
      qnet_ult: +qnet.toFixed(1),
      qa:       +qa.toFixed(1),
      FS,
    };
  }

  // ── Drained — Meyerhof (1963) ───────────────────────────────────────────────
  if (phi !== null) {
    const phiR = phi * Math.PI / 180;
    // Bearing capacity factors
    const Nq     = Math.exp(Math.PI * Math.tan(phiR)) * Math.pow(Math.tan(Math.PI / 4 + phiR / 2), 2);
    const Nc     = phi > 0 ? (Nq - 1) / Math.tan(phiR) : 5.14;
    const Ngamma = 2 * (Nq + 1) * Math.tan(phiR);

    // Shape factors (Meyerhof rectangular)
    const BL = B / Math.max(L, B);
    const Fcs  = 1 + BL * (Nq / Nc);
    const Fqs  = 1 + BL * Math.tan(phiR);
    const Fgs  = Math.max(1 - 0.4 * BL, 0.6);

    // Depth factors (Meyerhof, z/B ≤ 1 simplified)
    const zB  = Math.min(z / B, 1.0);
    const Fcd = 1 + 0.4 * zB;
    const Fqd = 1 + 2 * Math.tan(phiR) * Math.pow(1 - Math.sin(phiR), 2) * zB;
    const Fgd = 1;

    const c    = cprime ?? 0;
    const qult = c * Nc * Fcs * Fcd
               + q * Nq * Fqs * Fqd
               + 0.5 * gamma * B * Ngamma * Fgs * Fgd;
    const qnet = qult - q;
    const qa   = qnet / FS;

    result.drained = {
      phi, cprime: c,
      Nq:     +Nq.toFixed(2),
      Nc:     +Nc.toFixed(2),
      Ngamma: +Ngamma.toFixed(2),
      qult:   +qult.toFixed(1),
      qnet_ult: +qnet.toFixed(1),
      qa:     +qa.toFixed(1),
      FS,
    };
  }

  if (!result.undrained && !result.drained) {
    return {
      error: `Unit "${bearingUnit.code}" has no Cu or φ' set — enter values in the Props tab.`,
      unit: bearingUnit,
    };
  }

  return result;
}

export function renderBearingResults(result, containerEl) {
  if (!containerEl) return;
  if (!result) {
    containerEl.innerHTML = '<p class="hint">Set Cu or φ\' in the Props tab, then calculate.</p>';
    return;
  }
  if (result.error) {
    containerEl.innerHTML = `<p class="hint bearing-error">${_esc(result.error)}</p>`;
    return;
  }

  const { unit, B, L, z, q_overburden } = result;

  let html = `
    <div class="bearing-unit-badge">
      <span class="sett-swatch" style="background:${unit.color}"></span>
      <span>Bearing unit: <strong>${_esc(unit.code)}</strong> — ${_esc(unit.name)}</span>
    </div>
    <div class="bearing-params-row">B=${B} m · L=${L} m · z=${z} m · q₀=${q_overburden.toFixed(0)} kPa</div>`;

  if (result.undrained) {
    const u = result.undrained;
    html += `
    <div class="bearing-section">
      <div class="bearing-mode-hdr">Undrained — Skempton (φ=0)</div>
      <table class="sett-table">
        <tr><td>Cu</td><td class="sett-val">${u.cu} kPa</td></tr>
        <tr><td>Nc (eff.)</td><td class="sett-val">${u.Nc}</td></tr>
        <tr><td>q<sub>net,ult</sub></td><td class="sett-val">${u.qnet_ult} kPa</td></tr>
        <tr class="bearing-highlight">
          <td>q<sub>allow</sub> (FS ${u.FS})</td>
          <td class="sett-val bearing-qa">${u.qa} kPa</td>
        </tr>
      </table>
    </div>`;
  }

  if (result.drained) {
    const d = result.drained;
    html += `
    <div class="bearing-section">
      <div class="bearing-mode-hdr">Drained — Meyerhof</div>
      <table class="sett-table">
        <tr><td>φ' / c'</td><td class="sett-val">${d.phi}° / ${d.cprime} kPa</td></tr>
        <tr><td>Nq / Nc / Nγ</td><td class="sett-val">${d.Nq} / ${d.Nc} / ${d.Ngamma}</td></tr>
        <tr><td>q<sub>ult</sub> (gross)</td><td class="sett-val">${d.qult} kPa</td></tr>
        <tr><td>q<sub>net,ult</sub></td><td class="sett-val">${d.qnet_ult} kPa</td></tr>
        <tr class="bearing-highlight">
          <td>q<sub>allow</sub> (FS ${d.FS})</td>
          <td class="sett-val bearing-qa">${d.qa} kPa</td>
        </tr>
      </table>
    </div>`;
  }

  containerEl.innerHTML = html;
}

function _esc(s) {
  return String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
