// ── Isopach / thickness map + depth / elevation / settlement / certainty maps ──
// Modes:
//   thick   — per-column unit thickness (isopach)
//   topZ    — elevation of unit top contact (mAOD)
//   baseZ   — elevation of unit base contact (mAOD)
//   settle  — total 1D consolidation settlement per column (mm)
//   cert    — mean certainty across all voxels per column (0–1)

export class IsopachMap {
  constructor() {
    this._panel     = document.getElementById('isopach-panel');
    this._canvas    = document.getElementById('isopach-canvas');
    this._closeBtn  = document.getElementById('isopach-close');
    this._exportBtn = document.getElementById('isopach-export');
    this._select    = document.getElementById('isopach-unit-select');
    this._modeSelect = document.getElementById('isopach-mode-select');
    this._ctx       = this._canvas?.getContext('2d');
    this._visible   = false;
    this._lastArgs  = null;

    this._settleInputs = document.getElementById('isopach-settle-inputs');
    this._foundLevel   = document.getElementById('isopach-found-level');
    this._loadKpa      = document.getElementById('isopach-load-kpa');

    this._closeBtn?.addEventListener('click', () => this.hide());
    this._exportBtn?.addEventListener('click', () => this._exportPNG());
    this._select?.addEventListener('change', () => {
      if (this._lastArgs) this._redraw();
    });
    this._modeSelect?.addEventListener('change', () => {
      const mode = this._modeSelect.value;
      if (this._settleInputs) {
        this._settleInputs.style.display = mode === 'settle' ? 'flex' : 'none';
      }
      if (this._lastArgs) this._redraw();
    });
    this._foundLevel?.addEventListener('change', () => { if (this._lastArgs) this._redraw(); });
    this._loadKpa?.addEventListener('change',    () => { if (this._lastArgs) this._redraw(); });

    window.addEventListener('keydown', e => {
      if (e.key === 'Escape' && this._visible) this.hide();
    });

    const ro = new ResizeObserver(() => { if (this._visible) this._redraw(); });
    if (this._panel) ro.observe(this._panel);
  }

  draw(grid, geoUnits, boreholes, conceptStore = null) {
    if (!grid) return;
    this._lastArgs = { grid, geoUnits, boreholes, conceptStore };

    if (this._select) {
      this._select.innerHTML = geoUnits
        .map(u => `<option value="${u.id}">${u.code} — ${u.name}</option>`)
        .join('');
    }

    this.show();
    this._redraw();
  }

  _redraw() {
    const args = this._lastArgs;
    if (!args || !this._canvas || !this._ctx) return;
    const { grid, geoUnits, boreholes } = args;
    const { nx, ny, nz, cellSize: cs, cellHeight: ch, origin: O, unitIds, certainty } = grid;

    const mode = this._modeSelect?.value ?? 'thick';

    // For unit-specific modes, require a valid unit selection
    const targetId = parseInt(this._select?.value ?? geoUnits[0]?.id ?? 0);
    const unit     = geoUnits.find(u => u.id === targetId);
    if (!unit && ['thick', 'thick-p10', 'thick-p90', 'topZ', 'baseZ', 'depth-to-top'].includes(mode)) return;

    // Build unitById lookup for settle mode
    const unitById = {};
    geoUnits.forEach(u => { unitById[u.id] = u; });

    // Foundation params for settle mode
    const foundElev  = parseFloat(this._foundLevel?.value ?? '') || (O.y + nz * ch * 0.5);
    const appliedKPa = parseFloat(this._loadKpa?.value ?? '50') || 50;

    // ── Compute per-column values ───────────────────────────────────────────
    const vals = new Float32Array(nx * ny).fill(NaN);
    let vMin = Infinity, vMax = -Infinity;

    for (let iy = 0; iy < ny; iy++) {
      for (let ix = 0; ix < nx; ix++) {
        let v = NaN;
        if (mode === 'thick') {
          let cnt = 0;
          for (let iz = 0; iz < nz; iz++) {
            if (unitIds[ix + iy * nx + iz * nx * ny] === targetId) cnt++;
          }
          v = cnt > 0 ? cnt * ch : NaN;
        } else if (mode === 'thick-p10' || mode === 'thick-p90') {
          // P10 = thickness exceeded with 90% probability (pessimistic/thin end)
          // P90 = thickness exceeded with 10% probability (optimistic/thick end)
          // Uses MC probability volume: count voxels where P(unit) >= threshold
          const probVol = grid.probVolumes?.get(unit.code);
          if (probVol) {
            const threshold = mode === 'thick-p10' ? 0.90 : 0.10;
            let cnt = 0;
            for (let iz = 0; iz < nz; iz++) {
              if (probVol[ix + iy * nx + iz * nx * ny] >= threshold) cnt++;
            }
            v = cnt > 0 ? cnt * ch : NaN;
          } else {
            // Fall back to P50 if no probability volumes
            let cnt = 0;
            for (let iz = 0; iz < nz; iz++) {
              if (unitIds[ix + iy * nx + iz * nx * ny] === targetId) cnt++;
            }
            v = cnt > 0 ? cnt * ch : NaN;
          }
        } else if (mode === 'topZ') {
          for (let iz = nz - 1; iz >= 0; iz--) {
            if (unitIds[ix + iy * nx + iz * nx * ny] === targetId) {
              v = O.y + (iz + 1) * ch; break;
            }
          }
        } else if (mode === 'baseZ') {
          for (let iz = 0; iz < nz; iz++) {
            if (unitIds[ix + iy * nx + iz * nx * ny] === targetId) {
              v = O.y + iz * ch; break;
            }
          }
        } else if (mode === 'depth-to-top') {
          // Find surface elevation (highest non-empty voxel in column)
          let surfElev = NaN;
          for (let iz = nz - 1; iz >= 0; iz--) {
            if (unitIds[ix + iy * nx + iz * nx * ny]) {
              surfElev = O.y + iz * ch + ch; break;
            }
          }
          // Find top of target unit
          for (let iz = nz - 1; iz >= 0; iz--) {
            if (unitIds[ix + iy * nx + iz * nx * ny] === targetId) {
              const topElev = O.y + iz * ch + ch;
              v = isNaN(surfElev) ? NaN : surfElev - topElev;
              break;
            }
          }
        } else if (mode === 'settle') {
          v = _settlementAtCol(grid, unitById, ix, iy, foundElev, appliedKPa);
        } else if (mode === 'cert') {
          let sum = 0, cnt = 0;
          for (let iz = 0; iz < nz; iz++) {
            const flat = ix + iy * nx + iz * nx * ny;
            if (unitIds[flat]) { sum += certainty[flat]; cnt++; }
          }
          v = cnt > 0 ? sum / cnt : NaN;
        }
        vals[ix + iy * nx] = v;
        if (!isNaN(v)) { if (v < vMin) vMin = v; if (v > vMax) vMax = v; }
      }
    }
    if (vMin === Infinity) vMin = 0;
    if (vMax === -Infinity) vMax = vMin + 1;
    const vRange = vMax - vMin || 1;

    // ── Canvas setup ────────────────────────────────────────────────────────
    const panelW = this._canvas.parentElement?.clientWidth  ?? 400;
    const panelH = this._canvas.parentElement?.clientHeight ?? 360;
    this._canvas.width  = panelW;
    this._canvas.height = panelH;

    const PAD   = 40;
    const drawW = panelW - PAD * 2;
    const drawH = panelH - PAD - 56;
    if (drawW < 40 || drawH < 40) return;

    const cellPxW = drawW / nx;
    const cellPxH = drawH / ny;
    const ctx = this._ctx;

    ctx.clearRect(0, 0, panelW, panelH);
    ctx.fillStyle = '#f0f2f5'; ctx.fillRect(0, 0, panelW, panelH);
    ctx.fillStyle = '#ffffff'; ctx.fillRect(PAD, PAD, drawW, drawH);

    // ── Draw cells ──────────────────────────────────────────────────────────
    for (let iy = 0; iy < ny; iy++) {
      for (let ix = 0; ix < nx; ix++) {
        const v = vals[ix + iy * nx];
        if (isNaN(v)) continue;
        const norm = (v - vMin) / vRange;
        ctx.fillStyle = mode === 'thick'         ? isopachColor(unit.color, norm)
                      : mode === 'depth-to-top' ? isopachColor(unit.color, 1 - norm) // shallow = bright
                      : mode === 'cert'         ? certColor(norm)
                      : mode === 'settle'       ? settleColor(norm)
                      : elevColor(norm);
        ctx.fillRect(
          PAD + ix * cellPxW,
          PAD + (ny - 1 - iy) * cellPxH,
          Math.ceil(cellPxW + 0.5),
          Math.ceil(cellPxH + 0.5)
        );
      }
    }

    // ── Borehole dots ────────────────────────────────────────────────────────
    (boreholes ?? []).filter(b => !b.synthetic).forEach(bh => {
      const ix = Math.floor((bh.x - O.x) / cs);
      const iy = Math.floor((bh.y - O.z) / cs);
      if (ix < 0 || ix >= nx || iy < 0 || iy >= ny) return;
      const px = PAD + (ix + 0.5) * cellPxW;
      const py = PAD + (ny - 1 - iy + 0.5) * cellPxH;
      ctx.fillStyle = '#1c2a38';
      ctx.beginPath(); ctx.arc(px, py, 3, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = 'rgba(255,255,255,0.85)';
      ctx.beginPath(); ctx.arc(px, py, 1.5, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = '#1c2a38';
      ctx.font = '8px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(bh.id, px, py - 5);
    });

    // ── Concept influence overlay ─────────────────────────────────────────────
    // If a conceptStore is provided in args, overlay translucent amber cells
    // where semantic context weight is high — shows where the conceptual model
    // is actively shaping the interpolation vs raw borehole data.
    const conceptStore = args.conceptStore ?? null;
    if (conceptStore && !conceptStore.isEmpty) {
      for (let iy = 0; iy < ny; iy++) {
        const worldY = O.z + (iy + 0.5) * cs;
        for (let ix = 0; ix < nx; ix++) {
          const worldX = O.x + (ix + 0.5) * cs;
          // computeAt is cheap for global concepts; use midpoint elevation
          const ctxC = conceptStore.computeAt(worldX, O.y + (nz * 0.5) * ch, worldY);
          const w    = Math.min(1, ctxC.totalWeight);
          if (w < 0.05) continue;
          const px = PAD + ix * cellPxW;
          const py = PAD + (ny - 1 - iy) * cellPxH;
          ctx.fillStyle = `rgba(255,160,30,${(w * 0.25).toFixed(3)})`;
          ctx.fillRect(px, py, Math.ceil(cellPxW + 0.5), Math.ceil(cellPxH + 0.5));
        }
      }
      // Legend note
      ctx.fillStyle = 'rgba(200,120,20,0.85)';
      ctx.font = '8px Inter, sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText('▣ concept influence', PAD + drawW - 2, PAD + 10);
    }

    // ── Marching-squares contour lines (7 levels) ──────────────────────────────
    // Each contour level is traced as a connected polyline using the standard
    // marching-squares lookup table (16 cases). This produces smooth, closed
    // isolines rather than the previous disconnected tick-marks.
    if (vMin < vMax) {
      const nContours = 7;
      ctx.font = '7px monospace';

      // Interpolate the crossing point between v0 and v1 for isovalue cL
      const lerp = (v0, v1, cL) => (cL - v0) / (v1 - v0);

      for (let ci = 1; ci < nContours; ci++) {
        const cLevel = vMin + (vMax - vMin) * ci / nContours;
        const isMajor = ci % 2 === 0;
        ctx.strokeStyle = isMajor ? 'rgba(255,255,255,0.70)' : 'rgba(255,255,255,0.38)';
        ctx.lineWidth   = isMajor ? 1.2 : 0.6;
        ctx.beginPath();

        for (let iy = 0; iy < ny - 1; iy++) {
          for (let ix = 0; ix < nx - 1; ix++) {
            const v00 = vals[ ix      +  iy      * nx];
            const v10 = vals[(ix + 1) +  iy      * nx];
            const v01 = vals[ ix      + (iy + 1) * nx];
            const v11 = vals[(ix + 1) + (iy + 1) * nx];
            if (isNaN(v00) || isNaN(v10) || isNaN(v01) || isNaN(v11)) continue;

            // Cell origin in canvas coords (bottom-left is iy=0, y flipped)
            const cx0 = PAD +  ix      * cellPxW;
            const cx1 = PAD + (ix + 1) * cellPxW;
            const cy0 = PAD + (ny - 1 -  iy     ) * cellPxH;
            const cy1 = PAD + (ny - 1 - (iy + 1)) * cellPxH;

            // Marching squares 4-bit index (above/below cLevel)
            const b00 = v00 >= cLevel ? 1 : 0;
            const b10 = v10 >= cLevel ? 2 : 0;
            const b01 = v01 >= cLevel ? 8 : 0;
            const b11 = v11 >= cLevel ? 4 : 0;
            const mc  = b00 | b10 | b11 | b01;

            // Edge midpoints (linearly interpolated)
            const eB = (t => [cx0 + t * cellPxW, cy0])(lerp(v00, v10, cLevel)); // bottom
            const eT = (t => [cx0 + t * cellPxW, cy1])(lerp(v01, v11, cLevel)); // top
            const eL = (t => [cx0, cy0 - t * cellPxH])(lerp(v00, v01, cLevel)); // left
            const eR = (t => [cx1, cy0 - t * cellPxH])(lerp(v10, v11, cLevel)); // right

            // Draw segments from marching-squares table (15-mc handles inverted cases)
            const segs = _mcSegs(mc, eB, eT, eL, eR);
            for (const [p0, p1] of segs) {
              ctx.moveTo(p0[0], p0[1]);
              ctx.lineTo(p1[0], p1[1]);
            }
          }
        }
        ctx.stroke();

        // Label: place at a point near the left edge where the contour crosses
        if (isMajor) {
          const labelVal = mode === 'settle' ? `${cLevel.toFixed(0)}`
                         : mode === 'cert'   ? `${(cLevel * 100).toFixed(0)}%`
                         : `${cLevel.toFixed(1)}`;
          // Find a crossing near the left of the grid
          for (let iy = 1; iy < ny - 1; iy++) {
            const v0 = vals[0 + iy * nx], v1 = vals[0 + (iy + 1) * nx];
            if (!isNaN(v0) && !isNaN(v1) && (v0 - cLevel) * (v1 - cLevel) < 0) {
              const t  = lerp(v0, v1, cLevel);
              const ly = PAD + (ny - 1 - iy) * cellPxH - t * cellPxH;
              ctx.fillStyle = 'rgba(255,255,255,0.85)';
              ctx.textAlign = 'left';
              ctx.fillText(labelVal, PAD + 2, ly - 2);
              break;
            }
          }
        }
      }
    }

    // ── Frame ────────────────────────────────────────────────────────────────
    ctx.strokeStyle = '#c8cdd6'; ctx.lineWidth = 1;
    ctx.strokeRect(PAD, PAD, drawW, drawH);

    // ── Legend bar ───────────────────────────────────────────────────────────
    const lgX = PAD, lgY = PAD + drawH + 8, lgW = drawW, lgH = 12;
    for (let i = 0; i < lgW; i++) {
      const t = i / lgW;
      ctx.fillStyle = mode === 'thick'  ? isopachColor(unit?.color ?? '#4a6fa5', t)
                    : mode === 'cert'   ? certColor(t)
                    : mode === 'settle' ? settleColor(t)
                    : elevColor(t);
      ctx.fillRect(lgX + i, lgY, 1, lgH);
    }
    ctx.strokeStyle = '#c8cdd6'; ctx.strokeRect(lgX, lgY, lgW, lgH);
    ctx.fillStyle = '#4a6275'; ctx.font = '10px Inter, sans-serif';

    const modeLabel = mode === 'thick'         ? `${unit?.code} thickness`
                    : mode === 'topZ'          ? `${unit?.code} top Z (mAOD)`
                    : mode === 'baseZ'         ? `${unit?.code} base Z (mAOD)`
                    : mode === 'depth-to-top'  ? `Depth to top of ${unit?.code} (m)`
                    : mode === 'settle'        ? `Settlement (mm) · FD=${foundElev.toFixed(1)} mAOD · q=${appliedKPa} kPa`
                    : 'Mean certainty (0–1)';

    const loStr = mode === 'settle' ? `${vMin.toFixed(0)} mm`
                : mode === 'cert'   ? `${(vMin*100).toFixed(0)}%`
                : `${vMin.toFixed(1)} m`;
    const hiStr = mode === 'settle' ? `${vMax.toFixed(0)} mm`
                : mode === 'cert'   ? `${(vMax*100).toFixed(0)}%`
                : `${vMax.toFixed(1)} m`;

    ctx.textAlign = 'left';
    ctx.fillText(loStr, lgX, lgY + lgH + 12);
    ctx.textAlign = 'right';
    ctx.fillText(hiStr, lgX + lgW, lgY + lgH + 12);
    ctx.textAlign = 'center';
    ctx.fillText(modeLabel, lgX + lgW * 0.5, lgY + lgH + 12);

    // ── Title ─────────────────────────────────────────────────────────────────
    ctx.fillStyle = '#8898a8';
    ctx.font = 'bold 11px Inter, sans-serif';
    ctx.textAlign = 'left';
    const titleUnit = (mode === 'thick' || mode === 'topZ' || mode === 'baseZ') && unit
      ? ` — ${unit.name}` : '';
    ctx.fillText(`${modeLabel}${titleUnit}`, PAD, PAD - 6);
  }

  show() { this._visible = true;  if (this._panel) this._panel.hidden = false; }
  hide() { this._visible = false; if (this._panel) this._panel.hidden = true;  }
  get visible() { return this._visible; }

  _exportPNG() {
    if (!this._canvas) return;
    const url = this._canvas.toDataURL('image/png');
    const a   = document.createElement('a');
    a.href     = url;
    a.download = `isopach-${new Date().toISOString().slice(0, 10)}.png`;
    a.click();
  }

  // Export current horizon / isopach values as CSV
  // Columns: X (Easting), Y (Northing), value, unit
  exportCSV() {
    const args = this._lastArgs;
    if (!args) return;
    const { grid, geoUnits } = args;
    const { nx, ny, nz, cellSize: cs, cellHeight: ch, origin: O, unitIds, certainty } = grid;
    const mode = this._modeSelect?.value ?? 'thick';
    const targetId = parseInt(this._select?.value ?? geoUnits[0]?.id ?? 0);
    const unit     = geoUnits.find(u => u.id === targetId);
    const unitById = {};
    geoUnits.forEach(u => { unitById[u.id] = u; });

    const modeLabel = mode === 'thick'        ? `${unit?.code ?? ''}_thickness_m`
                    : mode === 'topZ'         ? `${unit?.code ?? ''}_topZ_mAOD`
                    : mode === 'baseZ'        ? `${unit?.code ?? ''}_baseZ_mAOD`
                    : mode === 'depth-to-top' ? `${unit?.code ?? ''}_depth_to_top_m`
                    : mode === 'cert'         ? 'mean_certainty'
                    : 'settlement_mm';

    const rows = ['X_easting,Y_northing,' + modeLabel];
    for (let iy = 0; iy < ny; iy++) {
      const worldY = O.z + (iy + 0.5) * cs;
      for (let ix = 0; ix < nx; ix++) {
        const worldX = O.x + (ix + 0.5) * cs;
        let v = NaN;
        if (mode === 'thick') {
          let cnt = 0;
          for (let iz = 0; iz < nz; iz++) { if (unitIds[ix + iy * nx + iz * nx * ny] === targetId) cnt++; }
          v = cnt > 0 ? cnt * ch : NaN;
        } else if (mode === 'topZ') {
          for (let iz = nz - 1; iz >= 0; iz--) {
            if (unitIds[ix + iy * nx + iz * nx * ny] === targetId) { v = O.y + (iz + 1) * ch; break; }
          }
        } else if (mode === 'baseZ') {
          for (let iz = 0; iz < nz; iz++) {
            if (unitIds[ix + iy * nx + iz * nx * ny] === targetId) { v = O.y + iz * ch; break; }
          }
        } else if (mode === 'depth-to-top') {
          let surfElev = NaN;
          for (let iz = nz - 1; iz >= 0; iz--) {
            if (unitIds[ix + iy * nx + iz * nx * ny]) { surfElev = O.y + iz * ch + ch; break; }
          }
          for (let iz = nz - 1; iz >= 0; iz--) {
            if (unitIds[ix + iy * nx + iz * nx * ny] === targetId) {
              v = isNaN(surfElev) ? NaN : surfElev - (O.y + iz * ch + ch); break;
            }
          }
        } else if (mode === 'cert') {
          let sum = 0, cnt = 0;
          for (let iz = 0; iz < nz; iz++) {
            const flat = ix + iy * nx + iz * nx * ny;
            if (unitIds[flat]) { sum += certainty[flat]; cnt++; }
          }
          v = cnt > 0 ? sum / cnt : NaN;
        }
        if (!isNaN(v)) rows.push(`${worldX.toFixed(2)},${worldY.toFixed(2)},${v.toFixed(3)}`);
      }
    }

    const blob = new Blob([rows.join('\n')], { type: 'text/csv' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = `horizon-${modeLabel}-${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }
}

// ── Per-column settlement (mm) — Terzaghi 1D consolidation ────────────────────
function _settlementAtCol(grid, unitById, cx, cy, foundElev, appliedKPa) {
  const { nx, ny, nz, cellHeight: ch, origin: O, unitIds } = grid;
  let sigmaV = 0, total = 0, hasData = false;
  for (let iz = nz - 1; iz >= 0; iz--) {
    const midZ = O.y + iz * ch + ch * 0.5;
    const unit  = unitById[unitIds[cx + cy * nx + iz * nx * ny]];
    const gam   = unit?.params?.gamma ?? 19;
    if (midZ >= foundElev) { sigmaV += gam * ch; continue; }
    if (!unit)             { sigmaV += gam * ch; continue; }
    const Cc = unit.params?.Cc ?? null;
    const e0 = unit.params?.e0 ?? null;
    if (Cc !== null && e0 !== null && sigmaV > 0) {
      const ratio = (sigmaV + appliedKPa) / sigmaV;
      if (ratio > 1) {
        total += (Cc / (1 + e0)) * ch * 1000 * Math.log10(ratio);
        hasData = true;
      }
    }
    sigmaV += gam * ch;
  }
  return hasData ? total : NaN;
}

// ── Marching-squares segment table ───────────────────────────────────────────
// Returns an array of [p0, p1] pairs (each point is [cx, cy]) for the given
// 4-bit marching-squares case (mc 0–15). Edges:
//   eB = bottom, eT = top, eL = left, eR = right  (canvas coordinates)
function _mcSegs(mc, eB, eT, eL, eR) {
  switch (mc) {
    case  0: case 15: return [];
    case  1: case 14: return [[eB, eL]];
    case  2: case 13: return [[eB, eR]];
    case  3: case 12: return [[eL, eR]];
    case  4: case 11: return [[eT, eR]];
    case  5:          return [[eB, eL], [eT, eR]];
    case  6: case  9: return [[eB, eT]];
    case  7: case  8: return [[eT, eL]];
    case 10:          return [[eB, eR], [eT, eL]];
    default:          return [];
  }
}

// ── Colour ramps ──────────────────────────────────────────────────────────────
function isopachColor(hex, norm) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgb(${Math.round(255+(r-255)*norm)},${Math.round(255+(g-255)*norm)},${Math.round(255+(b-255)*norm)})`;
}

function certColor(norm) {
  // Red (low) → yellow → green (high certainty)
  const r = norm < 0.5 ? 220 : Math.round(220 - (norm - 0.5) * 2 * 170);
  const g = norm < 0.5 ? Math.round(norm * 2 * 200) : 200;
  const b = 60;
  return `rgb(${r},${g},${b})`;
}

function settleColor(norm) {
  // White (low settlement) → yellow → red (high settlement)
  const r = 255;
  const g = Math.round(255 - norm * 220);
  const b = Math.round(255 - norm * 255);
  return `rgb(${r},${g},${b})`;
}

function elevColor(norm) {
  // Cool (deep blue) → warm (yellow-red) ramp for elevation
  const stops = [
    [0.00, [  8,  48, 107]],
    [0.25, [ 33, 113, 181]],
    [0.50, [107, 174, 214]],
    [0.65, [186, 228, 179]],
    [0.80, [253, 205, 90]],
    [1.00, [215,  48,  39]],
  ];
  let lo = stops[0], hi = stops[stops.length - 1];
  for (let i = 1; i < stops.length; i++) {
    if (norm <= stops[i][0]) { lo = stops[i-1]; hi = stops[i]; break; }
  }
  const t = lo[0] === hi[0] ? 0 : (norm - lo[0]) / (hi[0] - lo[0]);
  const lerp = (a, b) => Math.round(a + (b - a) * t);
  return `rgb(${lerp(lo[1][0],hi[1][0])},${lerp(lo[1][1],hi[1][1])},${lerp(lo[1][2],hi[1][2])})`;
}
