// ── Pile Capacity Calculator ───────────────────────────────────────────────────
// Skin friction: Alpha method (undrained, clays) or Beta method (drained, granular)
// Tip resistance: 9×Cu×A (undrained) or Nq×σ'v×A (drained)
// Sampling: scans model centre column from pile toe to pile head

export function calculatePileCapacity(grid, geoUnits, headElev, toeElev, D, FS) {
  if (!grid || !geoUnits.length) return null;
  if (headElev <= toeElev) return { error: 'Pile head elevation must be above pile toe.' };

  const { nx, ny, nz, cellHeight: ch, origin: O, unitIds } = grid;
  const cx = Math.floor(nx / 2);
  const cy = Math.floor(ny / 2);
  const unitById = {};
  geoUnits.forEach(u => { unitById[u.id] = u; });

  const pilePerim = Math.PI * D;
  const pileArea  = Math.PI * D * D / 4;

  // Accumulate overburden above pile head and collect shaft layers
  const rawLayers = [];
  let sigmaV = 0;

  for (let iz = nz - 1; iz >= 0; iz--) {
    const topZ = O.y + (iz + 1) * ch;
    const botZ = O.y + iz * ch;
    const midZ = botZ + ch * 0.5;
    const uid  = unitIds[cx + cy * nx + iz * nx * ny];
    const unit = unitById[uid];
    const gam  = unit?.params?.gamma ?? 19;

    if (midZ >= headElev) {
      // Above pile head — overburden only
      sigmaV += gam * ch;
      continue;
    }
    if (topZ <= toeElev) break; // below pile toe

    // Clip layer to shaft interval
    const clipTop = Math.min(topZ, headElev);
    const clipBot = Math.max(botZ, toeElev);
    const h = clipTop - clipBot;
    if (h <= 0) continue;

    if (!unit) { sigmaV += gam * h; continue; }

    const sigmaV_mid = sigmaV + gam * h * 0.5;
    rawLayers.push({ unit, h, topZ: clipTop, midZ: (clipTop + clipBot) / 2, sigmaV_mid });
    sigmaV += gam * h;
  }

  if (!rawLayers.length) {
    return { error: 'No model voxels found in the pile shaft interval — check elevations.' };
  }

  // Merge consecutive same-unit layers
  const layers = [];
  for (const l of rawLayers) {
    const prev = layers[layers.length - 1];
    if (prev && prev.unit.id === l.unit.id) {
      prev.h        += l.h;
      prev.sigmaV_mid = (prev.sigmaV_mid + l.sigmaV_mid) / 2;
    } else {
      layers.push({ ...l });
    }
  }

  // Tip unit
  let tipUnit = null;
  for (let iz = nz - 1; iz >= 0; iz--) {
    if (O.y + iz * ch + ch * 0.5 <= toeElev) {
      const uid = unitIds[cx + cy * nx + iz * nx * ny];
      tipUnit = unitById[uid] ?? null;
      break;
    }
  }
  const sigmaV_tip = Math.min(sigmaV, 200); // Meyerhof effective stress cap (kPa)

  // ── Shaft friction per layer ─────────────────────────────────────────────────
  let Qsk_total = 0;
  const layerResults = layers.map(l => {
    const p   = l.unit.params ?? {};
    const cu  = p.cu  ?? null;
    const phi = p.phi ?? null;
    let qs = null, Qs = null, method = null;

    if (cu !== null) {
      // Alpha method (Randolph & Wroth 1978): α = 0.5 for stiff, ≤1.0 for soft
      const alpha = cu <= 25 ? 1.0
                  : cu <= 75 ? (1.0 - 0.5 * (cu - 25) / 50)
                  : 0.5;
      qs = alpha * cu;
      method = `α=${alpha.toFixed(2)}`;
    } else if (phi !== null) {
      // Beta method
      const phiR  = phi * Math.PI / 180;
      const K0    = 1 - Math.sin(phiR);
      const delta = (2 / 3) * phiR;
      const beta  = K0 * Math.tan(delta);
      qs = beta * Math.max(l.sigmaV_mid, 1);
      method = `β=${(K0 * Math.tan(delta)).toFixed(3)}`;
    }

    if (qs !== null) {
      Qs = qs * pilePerim * l.h;
      Qsk_total += Qs;
    }
    return { unit: l.unit, h: l.h, midZ: l.midZ, qs, Qs, method };
  });

  // ── Tip resistance ────────────────────────────────────────────────────────────
  let Qtip = null;
  let tipMethod = null;
  if (tipUnit) {
    const p  = tipUnit.params ?? {};
    const cu = p.cu  ?? null;
    const phi = p.phi ?? null;
    if (cu !== null) {
      Qtip      = 9 * cu * pileArea;
      tipMethod = `9×Cu (${cu} kPa)`;
    } else if (phi !== null) {
      const phiR = phi * Math.PI / 180;
      const Nq   = Math.exp(Math.PI * Math.tan(phiR)) * Math.pow(Math.tan(Math.PI / 4 + phiR / 2), 2);
      Qtip       = Nq * sigmaV_tip * pileArea;
      tipMethod  = `Nq=${Nq.toFixed(1)}`;
    }
  }

  const hasData = layerResults.some(l => l.Qs !== null) || Qtip !== null;
  if (!hasData) {
    return {
      error: 'No Cu or φ\' set for units in pile profile — enter values in the Props tab.',
      layers: layerResults, tipUnit,
    };
  }

  const Qult = Qsk_total + (Qtip ?? 0);
  const Qa   = Qult / FS;
  // Tension capacity = Qsk / (FS_tension=2) — compression FS usually 3
  const Qt   = Qsk_total / 2;

  return {
    layers: layerResults,
    tipUnit, tipMethod,
    Qsk:  +Qsk_total.toFixed(1),
    Qtip: Qtip !== null ? +Qtip.toFixed(1) : null,
    Qult: +Qult.toFixed(1),
    Qa:   +Qa.toFixed(1),
    Qt:   +Qt.toFixed(1),
    FS, D, headElev, toeElev,
    pileLength: +(headElev - toeElev).toFixed(2),
  };
}

export function renderPileResults(result, containerEl) {
  if (!containerEl) return;
  if (!result) {
    containerEl.innerHTML = '<p class="hint">Set Cu or φ\' in Props tab, then calculate.</p>';
    return;
  }
  if (result.error) {
    containerEl.innerHTML = `<p class="hint bearing-error">${_esc(result.error)}</p>`;
    return;
  }

  const { D, pileLength, Qsk, Qtip, Qult, Qa, Qt, FS, layers, tipUnit, tipMethod } = result;

  let html = `
    <div class="bearing-params-row">
      D=${D} m · L=${pileLength} m · FS=${FS}
    </div>
    <table class="sett-table">
      <tr><th>Layer</th><th>h (m)</th><th>qs (kPa)</th><th>Qs (kN)</th></tr>`;

  for (const l of layers) {
    const hasData = l.Qs !== null;
    html += `<tr class="${hasData ? '' : 'sett-missing'}">
      <td><span class="sett-swatch" style="background:${l.unit.color}"></span>${_esc(l.unit.code)}</td>
      <td>${l.h.toFixed(2)}</td>
      <td>${l.qs !== null ? l.qs.toFixed(1) : '—'}${l.method ? ` <small>${_esc(l.method)}</small>` : ''}</td>
      <td class="sett-val">${l.Qs !== null ? l.Qs.toFixed(1) : '—'}</td>
    </tr>`;
  }

  html += `</table>
    <div class="bearing-section">
      <table class="sett-table">
        <tr><td>Q<sub>sk</sub> (skin)</td><td class="sett-val">${Qsk} kN</td></tr>`;
  if (Qtip !== null) {
    html += `<tr><td>Q<sub>tip</sub> ${tipMethod ? `<small>${_esc(tipMethod)}</small>` : ''}</td>
              <td class="sett-val">${Qtip} kN</td></tr>`;
  } else {
    html += `<tr class="sett-missing"><td>Q<sub>tip</sub></td><td class="sett-val">—</td></tr>`;
  }
  html += `
        <tr><td>Q<sub>ult</sub></td><td class="sett-val">${Qult} kN</td></tr>
        <tr class="bearing-highlight">
          <td>Q<sub>a</sub> (FS ${FS})</td>
          <td class="sett-val bearing-qa">${Qa} kN</td>
        </tr>
        <tr><td>Q<sub>tension</sub> (FS 2)</td><td class="sett-val">${Qt} kN</td></tr>
      </table>
    </div>`;

  containerEl.innerHTML = html;
}

function _esc(s) {
  return String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
