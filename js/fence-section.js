// ── 2D Fence / Cross-Section renderer ────────────────────────────────────────
// Samples the voxel grid along the slicer's section line and draws a
// lithology panel with elevation grid, BH ticks and unit legend.

// ── Geological lithological pattern system ────────────────────────────────────
// Each pattern is a 16×16 offscreen canvas tile; drawn using fillPattern (repeat).
// Pattern type is inferred from unit code/name keyword matching.

const _patternCache = new Map(); // key: patternType → CanvasPattern

function _inferLithType(unit) {
  const s = ((unit.code ?? '') + ' ' + (unit.name ?? '')).toLowerCase();
  if (/gravel|grv|gvl|cobble|bould/.test(s))  return 'gravel';
  if (/sand|snd|sa(?!\w)/.test(s))             return 'sand';
  if (/silt|slt/.test(s))                      return 'silt';
  if (/clay|cl(?!\w)|alluvial|alc/.test(s))    return 'clay';
  if (/peat|org|pt(?!\w)/.test(s))             return 'peat';
  if (/chalk|cha|chl/.test(s))                 return 'chalk';
  if (/limestone|lst|ls(?!\w)/.test(s))        return 'limestone';
  if (/made.?ground|mg(?!\w)|fill|topsoil|ts(?!\w)/.test(s)) return 'fill';
  if (/bedrock|rock|granite|gneiss|mudstone|sandstone|mudst/.test(s)) return 'rock';
  return 'default';
}

function _buildPattern(ctx, type, color) {
  const key = `${type}:${color}`;
  if (_patternCache.has(key)) return _patternCache.get(key);

  const sz = 16;
  const off = document.createElement('canvas');
  off.width = sz; off.height = sz;
  const c = off.getContext('2d');

  // Base fill
  c.fillStyle = color;
  c.fillRect(0, 0, sz, sz);
  c.strokeStyle = 'rgba(0,0,0,0.18)';
  c.fillStyle   = 'rgba(0,0,0,0.15)';
  c.lineWidth   = 0.6;

  switch (type) {
    case 'clay':
      // Horizontal wavy lines (hachure)
      for (let y = 3; y < sz; y += 4) {
        c.beginPath();
        for (let x = 0; x < sz; x++) {
          const yy = y + Math.sin(x * 0.8) * 0.7;
          if (x === 0) c.moveTo(x, yy); else c.lineTo(x, yy);
        }
        c.stroke();
      }
      break;
    case 'sand':
      // Stipple dots
      for (let dy = 0; dy < sz; dy += 4) {
        for (let dx = (dy % 8 === 0 ? 1 : 3); dx < sz; dx += 6) {
          c.beginPath(); c.arc(dx, dy + 2, 0.9, 0, Math.PI * 2); c.fill();
        }
      }
      break;
    case 'gravel':
      // Larger irregular blobs / ellipses
      const blobs = [[3,4,2.5,1.5],[9,2,2,1.2],[12,8,1.8,2.2],[5,11,2.2,1.5],[11,13,1.8,1.2]];
      blobs.forEach(([x,y,rx,ry]) => {
        c.beginPath(); c.ellipse(x, y, rx, ry, 0.4, 0, Math.PI * 2);
        c.stroke();
      });
      break;
    case 'silt':
      // Fine dots + faint wavy line
      for (let dy = 0; dy < sz; dy += 3) {
        for (let dx = (dy % 6 === 0 ? 0 : 2); dx < sz; dx += 5) {
          c.beginPath(); c.arc(dx, dy + 1.5, 0.5, 0, Math.PI * 2); c.fill();
        }
      }
      break;
    case 'peat':
      // Horizontal dashes + diagonal cross
      for (let y = 2; y < sz; y += 5) {
        c.beginPath(); c.moveTo(1, y); c.lineTo(6, y);
        c.moveTo(9, y); c.lineTo(14, y); c.stroke();
      }
      c.beginPath();
      c.moveTo(0, sz); c.lineTo(sz, 0);
      c.moveTo(0, sz/2); c.lineTo(sz/2, 0);
      c.globalAlpha = 0.4; c.stroke(); c.globalAlpha = 1;
      break;
    case 'chalk':
      // V-hachures (classic chalk pattern)
      for (let y = 0; y < sz; y += 5) {
        c.beginPath();
        c.moveTo(0, y); c.lineTo(sz/2, y + 3); c.lineTo(sz, y);
        c.stroke();
      }
      break;
    case 'limestone':
      // Brick-like blocks
      for (let row = 0; row < 3; row++) {
        const yOff = row % 2 === 0 ? 0 : sz / 4;
        for (let col = 0; col < 3; col++) {
          c.strokeRect(col * sz/2 - yOff, row * sz/3, sz/2 - 1, sz/3 - 1);
        }
      }
      break;
    case 'fill':
      // Diagonal hatching (made ground)
      for (let d = -sz; d < sz * 2; d += 5) {
        c.beginPath();
        c.moveTo(d, 0); c.lineTo(d + sz, sz); c.stroke();
      }
      break;
    case 'rock':
      // Cross-hatch (strong diagonal both ways)
      c.lineWidth = 0.5;
      for (let d = -sz; d < sz * 2; d += 6) {
        c.beginPath(); c.moveTo(d, 0); c.lineTo(d + sz, sz); c.stroke();
        c.beginPath(); c.moveTo(d, 0); c.lineTo(d - sz, sz); c.stroke();
      }
      break;
    default:
      // No overlay
      break;
  }

  const pattern = ctx.createPattern(off, 'repeat');
  _patternCache.set(key, pattern);
  return pattern;
}

export class FenceSection {
  constructor() {
    this._panel    = document.getElementById('fence-panel');
    this._canvas   = document.getElementById('fence-canvas');
    this._closeBtn = document.getElementById('fence-close');
    this._titleEl  = document.getElementById('fence-title');
    this._exportBtn = document.getElementById('fence-export');
    this._ctx      = this._canvas?.getContext('2d');
    this._visible  = false;
    this._lastArgs = null;

    this._closeBtn?.addEventListener('click',  () => this.hide());
    this._exportBtn?.addEventListener('click', () => this._exportPNG());

    window.addEventListener('keydown', e => {
      if (e.key === 'Escape' && this._visible) this.hide();
    });

    const ro = new ResizeObserver(() => { if (this._visible) this._redraw(); });
    if (this._panel) ro.observe(this._panel);
  }

  // ── Public API ────────────────────────────────────────────────────────────
  draw(grid, geoUnits, normal, centerD, thickness, boreholes, conceptStore = null) {
    if (!grid) return;
    this._lastArgs = { grid, geoUnits, normal, centerD, thickness, boreholes, conceptStore };
    this.show();
    this._redraw();
  }

  show() {
    this._visible = true;
    if (this._panel) this._panel.hidden = false;
  }

  hide() {
    this._visible = false;
    if (this._panel) this._panel.hidden = true;
  }

  get visible() { return this._visible; }

  // ── Rendering ─────────────────────────────────────────────────────────────
  _redraw() {
    const args = this._lastArgs;
    if (!args || !this._canvas || !this._ctx) return;

    const showUnc      = document.getElementById('fence-show-uncertainty')?.checked ?? false;
    const showCov      = document.getElementById('fence-show-coverage')?.checked ?? false;
    const showPatterns = document.getElementById('fence-show-patterns')?.checked ?? false;
    const showRibbons  = document.getElementById('fence-show-ribbons')?.checked ?? false;

    const { grid, geoUnits, normal, centerD, thickness, boreholes } = args;
    const { nx, ny, nz, cellSize: cs, cellHeight: ch, origin: O, unitIds, certainty, blendRatios } = grid;

    const unitById = {};
    geoUnits.forEach(u => { unitById[u.id] = u; });

    // Along-axis: perpendicular to normal in XZ plane
    const along = { x: normal.z, z: -normal.x };

    // Project model centre onto section plane
    const cx0 = O.x + nx * cs * 0.5;
    const cz0 = O.z + ny * cs * 0.5;
    const proj = centerD - (normal.x * cx0 + normal.z * cz0);
    const sx0  = cx0 + normal.x * proj;
    const sz0  = cz0 + normal.z * proj;

    const panelW = this._canvas.parentElement?.clientWidth  ?? 600;
    const panelH = this._canvas.parentElement?.clientHeight ?? 400;
    this._canvas.width  = panelW;
    this._canvas.height = panelH;

    const PAD_L = 52, PAD_R = 20, PAD_T = 36, PAD_B = 50;
    const drawW = panelW - PAD_L - PAD_R;
    const drawH = panelH - PAD_T  - PAD_B;
    if (drawW < 50 || drawH < 50) return;

    const ctx = this._ctx;
    ctx.clearRect(0, 0, panelW, panelH);
    ctx.fillStyle = '#f0f2f5';
    ctx.fillRect(0, 0, panelW, panelH);
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(PAD_L, PAD_T, drawW, drawH);

    const worldW = Math.max(nx * cs, ny * cs) * 1.1;
    const botY   = O.y;
    const topY   = O.y + nz * ch;
    const worldH = topY - botY;

    const N_COLS = Math.max(60, Math.min(300, Math.floor(drawW / 2)));
    const colPx  = drawW / N_COLS;

    // ── Draw column strips ────────────────────────────────────────────────
    for (let ci = 0; ci < N_COLS; ci++) {
      const t    = (ci / (N_COLS - 1)) - 0.5;
      const dist = t * worldW;
      const wx   = sx0 + along.x * dist;
      const wz   = sz0 + along.z * dist;

      const ix = Math.floor((wx - O.x) / cs);
      const iy = Math.floor((wz - O.z) / cs);
      if (ix < 0 || ix >= nx || iy < 0 || iy >= ny) continue;

      const colX = PAD_L + ci * colPx;

      for (let iz = 0; iz < nz; iz++) {
        const flat   = ix + iy * nx + iz * nx * ny;
        const uid    = unitIds[flat];
        const unit   = unitById[uid];
        if (!unit) continue;
        const cert   = certainty[flat];
        const blend  = blendRatios ? blendRatios[flat] : 0;
        const yPx    = PAD_T + drawH - ((iz * ch + ch * 0.5) / worldH) * drawH;
        const hPx    = Math.max(1, (ch / worldH) * drawH);

        ctx.globalAlpha = Math.max(0.4, cert);
        ctx.fillStyle   = unit.color;
        ctx.fillRect(colX, yPx - hPx, Math.ceil(colPx + 0.5), Math.ceil(hPx + 0.5));

        // Geological lithological pattern overlay
        if (showPatterns && hPx > 2) {
          const lithType = _inferLithType(unit);
          if (lithType !== 'default') {
            const pat = _buildPattern(ctx, lithType, unit.color);
            if (pat) {
              ctx.globalAlpha = 0.55;
              ctx.fillStyle = pat;
              ctx.fillRect(colX, yPx - hPx, Math.ceil(colPx + 0.5), Math.ceil(hPx + 0.5));
            }
          }
        }

        // Fuzzy boundary: if high blend ratio, overlay with a semi-transparent
        // hatching zone at the top of the cell to indicate uncertain contact
        if (blend > 0.25) {
          ctx.globalAlpha = blend * 0.5;
          ctx.fillStyle   = '#ffffff';
          ctx.fillRect(colX, yPx - hPx, Math.ceil(colPx + 0.5), Math.max(1, hPx * 0.3));
        }
      }
    }
    ctx.globalAlpha = 1;

    // ── Uncertainty / coverage overlay ────────────────────────────────────
    if ((showUnc || showCov) && (certainty || grid.coverageDensity)) {
      const coverageData = grid.coverageDensity;
      const LOG2 = Math.log(2);
      const nUnits = Math.max(2, geoUnits.length);
      const maxH_ent = Math.log2(nUnits);

      for (let ci = 0; ci < N_COLS; ci++) {
        const t  = (ci / (N_COLS - 1)) - 0.5;
        const wx = sx0 + along.x * t * worldW;
        const wz = sz0 + along.z * t * worldW;
        const ix = Math.floor((wx - O.x) / cs);
        const iy = Math.floor((wz - O.z) / cs);
        if (ix < 0 || ix >= nx || iy < 0 || iy >= ny) continue;
        const colX = PAD_L + ci * colPx;

        for (let iz = 0; iz < nz; iz++) {
          const flat  = ix + iy * nx + iz * nx * ny;
          if (!unitIds[flat]) continue;
          const yPx = PAD_T + drawH - ((iz * ch + ch * 0.5) / worldH) * drawH;
          const hPx = Math.max(1, (ch / worldH) * drawH);

          if (showUnc && certainty && blendRatios) {
            const p1 = Math.max(0.001, Math.min(0.999, certainty[flat]));
            const p2 = Math.max(0, Math.min(1 - p1, blendRatios[flat] ?? 0));
            const xE = (p) => p > 0 && p < 1 ? -p * Math.log(p) / LOG2 : 0;
            const ent  = xE(p1) + xE(p2) + xE(Math.max(0, 1 - p1 - p2));
            const t_unc = Math.min(1, ent / maxH_ent);
            // Red tint proportional to uncertainty
            ctx.globalAlpha = t_unc * 0.45;
            ctx.fillStyle = `hsl(${(1 - t_unc) * 120},80%,45%)`;
            ctx.fillRect(colX, yPx - hPx, Math.ceil(colPx + 0.5), Math.ceil(hPx + 0.5));
          }

          if (showCov && coverageData) {
            const cov   = coverageData[flat] ?? 0;
            const t_cov = Math.min(1, cov);
            // Blue tint proportional to data coverage
            ctx.globalAlpha = (1 - t_cov) * 0.35;
            ctx.fillStyle = 'rgba(40,60,180,1)';
            ctx.fillRect(colX, yPx - hPx, Math.ceil(colPx + 0.5), Math.ceil(hPx + 0.5));
          }
        }
      }
      ctx.globalAlpha = 1;

      // Overlay legend
      const legY = PAD_T + 6;
      if (showUnc) {
        const grd = ctx.createLinearGradient(PAD_L, 0, PAD_L + 80, 0);
        grd.addColorStop(0, 'hsla(120,80%,45%,0.5)');
        grd.addColorStop(0.5, 'hsla(60,80%,45%,0.5)');
        grd.addColorStop(1, 'hsla(0,80%,45%,0.5)');
        ctx.fillStyle = grd;
        ctx.fillRect(PAD_L, legY, 80, 8);
        ctx.fillStyle = '#333'; ctx.font = '8px monospace'; ctx.textAlign = 'left';
        ctx.fillText('certain', PAD_L + 1, legY + 17);
        ctx.textAlign = 'right';
        ctx.fillText('uncertain', PAD_L + 80, legY + 17);
        ctx.textAlign = 'left';
      }
    }

    // ── Elevation grid lines ──────────────────────────────────────────────
    ctx.strokeStyle = '#dde1e7';
    ctx.lineWidth   = 0.5;
    ctx.fillStyle   = '#4a6275';
    ctx.font        = '10px Inter, sans-serif';
    ctx.textAlign   = 'right';

    const tickStep = worldH <= 15 ? 1 : worldH <= 50 ? 5 : worldH <= 200 ? 10 : 25;
    for (let ev = Math.ceil(botY / tickStep) * tickStep; ev <= topY + 0.01; ev += tickStep) {
      const yPx = PAD_T + drawH - ((ev - botY) / worldH) * drawH;
      ctx.beginPath(); ctx.moveTo(PAD_L, yPx); ctx.lineTo(PAD_L + drawW, yPx); ctx.stroke();
      ctx.fillText(`${ev.toFixed(0)}`, PAD_L - 6, yPx + 3);
    }

    // ── Geological contact lines ──────────────────────────────────────────
    // Trace unit boundaries horizontally: for each column, detect where the
    // unit changes from iz-1 to iz and draw a thin line at that elevation.
    // Use a semi-transparent dark line so the contacts are readable but subtle.
    // Group collinear segments and draw one path per contact elevation for speed.
    {
      // colContacts[ci] = array of {iz, uid_above} where unit changes
      const contactY = new Map(); // key: iz → [ci,...] columns where contact occurs
      for (let ci = 0; ci < N_COLS; ci++) {
        const t   = (ci / (N_COLS - 1)) - 0.5;
        const wx  = sx0 + along.x * t * worldW;
        const wz  = sz0 + along.z * t * worldW;
        const ix  = Math.floor((wx - O.x) / cs);
        const iy  = Math.floor((wz - O.z) / cs);
        if (ix < 0 || ix >= nx || iy < 0 || iy >= ny) continue;
        for (let iz = 1; iz < nz; iz++) {
          const uid0 = unitIds[ix + iy * nx + (iz - 1) * nx * ny];
          const uid1 = unitIds[ix + iy * nx + iz * nx * ny];
          if (uid0 !== uid1 && uid0 && uid1) {
            if (!contactY.has(iz)) contactY.set(iz, []);
            contactY.get(iz).push(ci);
          }
        }
      }

      ctx.strokeStyle = 'rgba(20,30,45,0.55)';
      ctx.lineWidth   = 0.75;
      for (const [iz, cols] of contactY) {
        const yPx = PAD_T + drawH - ((iz * ch) / worldH) * drawH;
        // Draw a connected segment for each run of consecutive columns
        let runStart = null;
        for (let k = 0; k <= cols.length; k++) {
          const ci = cols[k];
          if (ci === undefined || (k > 0 && ci !== cols[k - 1] + 1)) {
            // End of run — draw it
            if (runStart !== null) {
              const x0 = PAD_L + runStart * colPx;
              const x1 = PAD_L + (cols[k - 1] + 1) * colPx;
              ctx.beginPath();
              ctx.moveTo(x0, yPx); ctx.lineTo(x1, yPx);
              ctx.stroke();
            }
            runStart = ci ?? null;
          } else if (k === 0) {
            runStart = ci;
          }
        }
      }
    }

    // ── P10–P90 probability contact ribbons (MC uncertainty) ─────────────────
    // For each unit with a probability volume, draw a shaded ribbon between the
    // P10 and P90 positions of its top contact along the section.
    if (showRibbons && grid.probVolumes?.size > 0) {
      const probVolumes = grid.probVolumes;

      for (const unit of geoUnits) {
        const probVol = probVolumes.get(unit.code);
        if (!probVol) continue;

        // Collect P10 and P90 top contact positions per section column
        const p10Y = new Float32Array(N_COLS).fill(NaN);
        const p90Y = new Float32Array(N_COLS).fill(NaN);

        for (let ci = 0; ci < N_COLS; ci++) {
          const t  = (ci / (N_COLS - 1)) - 0.5;
          const wx = sx0 + along.x * t * worldW;
          const wz = sz0 + along.z * t * worldW;
          const ix = Math.floor((wx - O.x) / cs);
          const iy = Math.floor((wz - O.z) / cs);
          if (ix < 0 || ix >= nx || iy < 0 || iy >= ny) continue;

          let topP10 = -1, topP90 = -1;
          // Scan top→bottom; find highest iz where P >= threshold
          for (let iz = nz - 1; iz >= 0; iz--) {
            const p = probVol[ix + iy * nx + iz * nx * ny];
            if (topP10 < 0 && p >= 0.10) topP10 = iz;
            if (topP90 < 0 && p >= 0.90) topP90 = iz;
            if (topP10 >= 0 && topP90 >= 0) break;
          }

          if (topP10 >= 0) {
            p10Y[ci] = PAD_T + drawH - ((topP10 * ch + ch) / worldH) * drawH;
          }
          if (topP90 >= 0) {
            p90Y[ci] = PAD_T + drawH - ((topP90 * ch + ch) / worldH) * drawH;
          }
        }

        // Draw ribbon: filled polygon from P10 top line to P90 top line
        const validCols = [];
        for (let ci = 0; ci < N_COLS; ci++) {
          if (!isNaN(p10Y[ci]) && !isNaN(p90Y[ci])) validCols.push(ci);
        }
        if (validCols.length < 2) continue;

        // Split into contiguous runs
        let runStart = validCols[0];
        for (let k = 0; k <= validCols.length; k++) {
          const ci  = validCols[k];
          const end = k > 0 && (ci === undefined || ci !== validCols[k - 1] + 1);
          if (end) {
            const runEnd = validCols[k - 1];
            if (runEnd > runStart + 1) {
              ctx.beginPath();
              // Top edge: P10 (shallower, wider uncertainty)
              ctx.moveTo(PAD_L + runStart * colPx, p10Y[runStart]);
              for (let ci2 = runStart + 1; ci2 <= runEnd; ci2++) {
                ctx.lineTo(PAD_L + ci2 * colPx, p10Y[ci2]);
              }
              // Bottom edge: P90 (deeper, inside the unit)
              for (let ci2 = runEnd; ci2 >= runStart; ci2--) {
                ctx.lineTo(PAD_L + ci2 * colPx, p90Y[ci2]);
              }
              ctx.closePath();
              ctx.globalAlpha = 0.28;
              ctx.fillStyle = unit.color;
              ctx.fill();
              // P50 contact line: solid mid-tone
              ctx.beginPath();
              ctx.moveTo(PAD_L + runStart * colPx, p90Y[runStart]);
              for (let ci2 = runStart + 1; ci2 <= runEnd; ci2++) {
                ctx.lineTo(PAD_L + ci2 * colPx, p90Y[ci2]);
              }
              ctx.globalAlpha = 0.65;
              ctx.strokeStyle = unit.color;
              ctx.lineWidth = 1.2;
              ctx.stroke();
            }
            runStart = ci;
          }
        }
        ctx.globalAlpha = 1;
      }

      // Add ribbon legend
      const ribY = PAD_T + drawH - 12;
      ctx.globalAlpha = 0.45;
      ctx.fillStyle = '#7090a0';
      ctx.fillRect(PAD_L + 4, ribY - 4, 18, 8);
      ctx.globalAlpha = 1;
      ctx.fillStyle = '#4a6275';
      ctx.font = '8px Inter, sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText('P10–P90 contact ribbon', PAD_L + 26, ribY + 2);
    }

    // ── BH ticks with SPT N-value bars ───────────────────────────────────
    (boreholes ?? []).filter(b => !b.synthetic).forEach(bh => {
      const distToPlane = Math.abs(normal.x * bh.x + normal.z * bh.y - centerD);
      if (distToPlane > thickness * 0.55) return;

      const sDist = along.x * (bh.x - sx0) + along.z * (bh.y - sz0);
      const ci = Math.floor((sDist / worldW + 0.5) * N_COLS);
      if (ci < 0 || ci >= N_COLS) return;

      const bhX    = PAD_L + ci * colPx + colPx * 0.5;
      const gl     = bh.groundLevel ?? topY;
      const colYPx = PAD_T + drawH - ((gl - botY) / worldH) * drawH;

      // BH stick
      ctx.strokeStyle = '#1c2a38';
      ctx.lineWidth   = 1.5;
      ctx.beginPath(); ctx.moveTo(bhX, colYPx); ctx.lineTo(bhX, PAD_T + drawH); ctx.stroke();

      // SPT N-value bars: if any layer has sptN, draw small horizontal bars
      // Max bar width = 18px at N=50, clipped to colPx*3
      const sptLayers = (bh.layers ?? []).filter(l => l.sptN != null && l.sptN > 0);
      if (sptLayers.length) {
        const maxSPT = Math.max(...sptLayers.map(l => l.sptN), 1);
        const BAR_SCALE = Math.min(18, colPx * 4) / Math.max(maxSPT, 50);
        for (const layer of sptLayers) {
          const zMid   = bh.groundLevel - (layer.top + layer.base) / 2;
          const yMid   = PAD_T + drawH - ((zMid - botY) / worldH) * drawH;
          const barW   = layer.sptN * BAR_SCALE;
          ctx.fillStyle   = 'rgba(60,120,200,0.65)';
          ctx.fillRect(bhX + 2, yMid - 1.5, barW, 3);
        }
      }

      ctx.fillStyle = '#ffffff';
      ctx.beginPath(); ctx.arc(bhX, colYPx, 4, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = '#1c2a38'; ctx.lineWidth = 1;
      ctx.stroke();

      ctx.fillStyle = '#1c2a38';
      ctx.font = 'bold 9px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(bh.id, bhX, colYPx - 7);
    });

    // ── Frame ─────────────────────────────────────────────────────────────
    ctx.strokeStyle = '#c8cdd6';
    ctx.lineWidth = 1;
    ctx.strokeRect(PAD_L, PAD_T, drawW, drawH);

    // ── Axis labels ───────────────────────────────────────────────────────
    ctx.fillStyle = '#8898a8';
    ctx.font = 'bold 10px Inter, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('Elevation (mAOD)', 4, PAD_T + 12);
    ctx.textAlign = 'center';
    ctx.fillText('Distance along section (m)', PAD_L + drawW * 0.5, panelH - 4);

    // X-axis distance ticks
    ctx.fillStyle = '#8898a8';
    ctx.font = '9px Inter, sans-serif';
    ctx.textAlign = 'center';
    const xTickStep = worldW <= 30 ? 5 : worldW <= 100 ? 10 : worldW <= 500 ? 50 : 100;
    for (let d = Math.ceil(-worldW * 0.5 / xTickStep) * xTickStep;
         d <= worldW * 0.5 + 0.01; d += xTickStep) {
      const xPx = PAD_L + (d / worldW + 0.5) * drawW;
      if (xPx < PAD_L || xPx > PAD_L + drawW) continue;
      ctx.fillText(`${d.toFixed(0)}`, xPx, PAD_T + drawH + 13);
    }

    // ── Unit legend strip ─────────────────────────────────────────────────
    const presentUnits = geoUnits.filter(u =>
      unitIds.some(id => unitById[id]?.code === u.code)
    );
    const lgW  = Math.min(54, Math.floor(drawW / (presentUnits.length + 0.5)));
    const lgX0 = PAD_L + (drawW - presentUnits.length * lgW) * 0.5;
    const lgY  = panelH - PAD_B + 14;

    presentUnits.forEach((u, i) => {
      const lx = lgX0 + i * lgW;
      ctx.fillStyle = u.color;
      ctx.fillRect(lx, lgY, lgW - 3, 11);
      ctx.fillStyle = '#4a6275';
      ctx.font = '9px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(u.code, lx + (lgW - 3) * 0.5, lgY + 22);
    });

    // ── Concept influence overlay on section ──────────────────────────────────
    // Translucent blue ribbon per column proportional to active concept weight at
    // that horizontal position. Drawn before legend so it stays behind text.
    if (conceptStore && !conceptStore.isEmpty) {
      const midElev = O.y + (nz * ch) * 0.5;
      let maxW = 0;
      const wts = new Float32Array(N_COLS);
      for (let ci = 0; ci < N_COLS; ci++) {
        const t    = (ci / Math.max(N_COLS - 1, 1)) - 0.5;
        const dist = t * worldW;
        const wx   = sx0 + along.x * dist;
        const wz   = sz0 + along.z * dist;
        const ctxC = conceptStore.computeAt(wx, midElev, wz);
        wts[ci] = ctxC.totalWeight;
        if (ctxC.totalWeight > maxW) maxW = ctxC.totalWeight;
      }
      if (maxW > 0.01) {
        for (let ci = 0; ci < N_COLS; ci++) {
          const tt = wts[ci] / maxW;
          const px = PAD_L + ci * colPx;
          ctx.fillStyle = `rgba(64,180,255,${(tt * 0.22).toFixed(3)})`;
          ctx.fillRect(px, PAD_T, Math.ceil(colPx + 0.5), drawH);
        }
        // Label at top-right of section
        ctx.fillStyle = 'rgba(64,180,255,0.75)';
        ctx.font      = '8px Inter, sans-serif';
        ctx.textAlign = 'right';
        ctx.fillText('concept influence', PAD_L + drawW - 2, PAD_T + 9);
      }
    }

    if (this._titleEl) this._titleEl.textContent = 'Geological Cross-Section';
  }

  // ── Export PNG ────────────────────────────────────────────────────────────
  _exportPNG() {
    if (!this._canvas) return;
    const url = this._canvas.toDataURL('image/png');
    const a   = document.createElement('a');
    a.href     = url;
    a.download = `section-${new Date().toISOString().slice(0, 10)}.png`;
    a.click();
  }

  // ── Export DXF ────────────────────────────────────────────────────────────
  // Exports the cross-section as an AutoCAD R12 DXF file.
  // Each geological unit gets its own LAYER; unit contacts are drawn as filled
  // SOLID entities. BH stick centrelines are exported as LINE entities.
  exportDXF() {
    const args = this._lastArgs;
    if (!args) return;
    const { grid, geoUnits, normal, centerD, thickness, boreholes } = args;
    const { nx, ny, nz, cellSize: cs, cellHeight: ch, origin: O, unitIds, certainty } = grid;

    const unitById = {};
    geoUnits.forEach(u => { unitById[u.id] = u; });

    const along = { x: normal.z, z: -normal.x };
    const cx0   = O.x + nx * cs * 0.5;
    const cz0   = O.z + ny * cs * 0.5;
    const proj  = centerD - (normal.x * cx0 + normal.z * cz0);
    const sx0   = cx0 + normal.x * proj;
    const sz0   = cz0 + normal.z * proj;
    const worldW = Math.max(nx * cs, ny * cs) * 1.1;
    const botY   = O.y;
    const topY   = O.y + nz * ch;

    // DXF helper
    const lines = [];
    const d = (code, value) => lines.push(`${code}\n${value}`);

    // Header
    d(0, 'SECTION'); d(2, 'HEADER');
    d(9,'$ACADVER'); d(1,'AC1009'); // R12
    d(9,'$INSUNITS'); d(70,6); // meters
    d(0,'ENDSEC');

    // Tables (minimal: just layer defs)
    d(0,'SECTION'); d(2,'TABLES');
    d(0,'TABLE'); d(2,'LAYER'); d(70, geoUnits.length + 3);
    for (const u of geoUnits) {
      // Convert hex color to AutoCAD color index (ACI) — approximate
      const ci = _hexToACI(u.color ?? '#888888');
      d(0,'LAYER'); d(2, u.code ?? 'UNKN'); d(70,0); d(62, ci); d(6,'CONTINUOUS');
    }
    d(0,'LAYER'); d(2,'BOREHOLES');  d(70,0); d(62,7); d(6,'CONTINUOUS');
    d(0,'LAYER'); d(2,'GRID');       d(70,0); d(62,8); d(6,'CONTINUOUS');
    d(0,'ENDTAB'); d(0,'ENDSEC');

    // Entities
    d(0,'SECTION'); d(2,'ENTITIES');

    const N_COLS = 150;
    const colW   = worldW / N_COLS;

    for (let ci = 0; ci < N_COLS; ci++) {
      const t    = (ci / (N_COLS - 1)) - 0.5;
      const dist = t * worldW;
      const wx   = sx0 + along.x * dist;
      const wz   = sz0 + along.z * dist;
      const ix = Math.floor((wx - O.x) / cs);
      const iy = Math.floor((wz - O.z) / cs);
      if (ix < 0 || ix >= nx || iy < 0 || iy >= ny) continue;

      const sPos = dist; // position along section line

      for (let iz = 0; iz < nz; iz++) {
        const uid  = unitIds[ix + iy * nx + iz * nx * ny];
        const unit = unitById[uid];
        if (!unit) continue;

        const y0 = botY + iz * ch;
        const y1 = y0 + ch;
        const x0 = sPos - colW / 2;
        const x1 = sPos + colW / 2;

        // SOLID entity (filled quad) — DXF SOLID uses 4 corner points
        d(0,'SOLID'); d(8, unit.code ?? 'UNKN');
        d(10, x0.toFixed(3)); d(20, y0.toFixed(3)); d(30,'0');
        d(11, x1.toFixed(3)); d(21, y0.toFixed(3)); d(31,'0');
        d(12, x0.toFixed(3)); d(22, y1.toFixed(3)); d(32,'0');
        d(13, x1.toFixed(3)); d(23, y1.toFixed(3)); d(33,'0');
      }
    }

    // BH sticks
    (boreholes ?? []).filter(b => !b.synthetic).forEach(bh => {
      const distToPlane = Math.abs(normal.x * bh.x + normal.z * bh.y - centerD);
      if (distToPlane > thickness * 0.55) return;
      const sDist = along.x * (bh.x - sx0) + along.z * (bh.y - sz0);
      const gl   = bh.groundLevel ?? topY;
      const dep  = bh.depth ?? (bh.layers?.length ? Math.max(...bh.layers.map(l => l.base)) : 10);
      const bhBot = gl - dep;
      d(0,'LINE'); d(8,'BOREHOLES');
      d(10, sDist.toFixed(3)); d(20, bhBot.toFixed(3)); d(30,'0');
      d(11, sDist.toFixed(3)); d(21, gl.toFixed(3)); d(31,'0');
      // Label
      d(0,'TEXT'); d(8,'BOREHOLES');
      d(10, sDist.toFixed(3)); d(20, (gl + 0.5).toFixed(3)); d(30,'0');
      d(40, 1.0); d(1, bh.id ?? 'BH');
    });

    // Grid lines
    const tickStep = (topY - botY) <= 15 ? 1 : (topY - botY) <= 50 ? 5 : 10;
    for (let ev = Math.ceil(botY / tickStep) * tickStep; ev <= topY + 0.01; ev += tickStep) {
      const halfW = worldW * 0.5;
      d(0,'LINE'); d(8,'GRID');
      d(10,(-halfW).toFixed(1)); d(20, ev.toFixed(1)); d(30,'0');
      d(11, halfW.toFixed(1)); d(21, ev.toFixed(1)); d(31,'0');
    }

    d(0,'ENDSEC'); d(0,'EOF');

    const dxfText = lines.join('\n');
    const blob = new Blob([dxfText], { type: 'application/dxf' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href = url; a.download = `section-${new Date().toISOString().slice(0,10)}.dxf`; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 5000);
  }
}

// AutoCAD Color Index approximation from hex string
function _hexToACI(hex) {
  const r = parseInt(hex.slice(1,3),16), g = parseInt(hex.slice(3,5),16), b = parseInt(hex.slice(5,7),16);
  // Simple mapping to 7 basic ACI colors
  const colors = [
    [255,0,0,1],[255,255,0,2],[0,255,0,3],[0,255,255,4],
    [0,0,255,5],[255,0,255,6],[255,255,255,7],
    [128,128,128,8],[128,64,0,9],[0,128,128,4],
  ];
  let best = 7, bestD = Infinity;
  for (const [cr,cg,cb,ci] of colors) {
    const d = (r-cr)**2 + (g-cg)**2 + (b-cb)**2;
    if (d < bestD) { bestD = d; best = ci; }
  }
  return best;
}
