// ── Isopach / thickness map + depth / elevation maps ─────────────────────────
// Modes:
//   thick  — per-column unit thickness (isopach)
//   topZ   — elevation of unit top contact (mAOD)
//   baseZ  — elevation of unit base contact (mAOD)

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

    this._closeBtn?.addEventListener('click', () => this.hide());
    this._exportBtn?.addEventListener('click', () => this._exportPNG());
    this._select?.addEventListener('change', () => {
      if (this._lastArgs) this._redraw();
    });
    this._modeSelect?.addEventListener('change', () => {
      if (this._lastArgs) this._redraw();
    });

    window.addEventListener('keydown', e => {
      if (e.key === 'Escape' && this._visible) this.hide();
    });

    const ro = new ResizeObserver(() => { if (this._visible) this._redraw(); });
    if (this._panel) ro.observe(this._panel);
  }

  draw(grid, geoUnits, boreholes) {
    if (!grid) return;
    this._lastArgs = { grid, geoUnits, boreholes };

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
    const { nx, ny, nz, cellSize: cs, cellHeight: ch, origin: O, unitIds } = grid;

    const targetId = parseInt(this._select?.value ?? geoUnits[0]?.id ?? 0);
    const unit     = geoUnits.find(u => u.id === targetId);
    if (!unit) return;

    const mode = this._modeSelect?.value ?? 'thick';

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
        } else if (mode === 'topZ') {
          for (let iz = nz - 1; iz >= 0; iz--) {
            if (unitIds[ix + iy * nx + iz * nx * ny] === targetId) {
              v = O.y + (iz + 1) * ch;
              break;
            }
          }
        } else if (mode === 'baseZ') {
          for (let iz = 0; iz < nz; iz++) {
            if (unitIds[ix + iy * nx + iz * nx * ny] === targetId) {
              v = O.y + iz * ch;
              break;
            }
          }
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
        ctx.fillStyle = mode === 'thick'
          ? isopachColor(unit.color, norm)
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

    // ── Frame ────────────────────────────────────────────────────────────────
    ctx.strokeStyle = '#c8cdd6'; ctx.lineWidth = 1;
    ctx.strokeRect(PAD, PAD, drawW, drawH);

    // ── Legend bar ───────────────────────────────────────────────────────────
    const lgX = PAD, lgY = PAD + drawH + 8, lgW = drawW, lgH = 12;
    for (let i = 0; i < lgW; i++) {
      const t = i / lgW;
      ctx.fillStyle = mode === 'thick'
        ? isopachColor(unit.color, t)
        : elevColor(t);
      ctx.fillRect(lgX + i, lgY, 1, lgH);
    }
    ctx.strokeStyle = '#c8cdd6'; ctx.strokeRect(lgX, lgY, lgW, lgH);
    ctx.fillStyle = '#4a6275'; ctx.font = '10px Inter, sans-serif';
    ctx.textAlign = 'left';
    const lo = mode === 'thick' ? '0 m' : `${vMin.toFixed(1)} m`;
    const hi = mode === 'thick' ? `${vMax.toFixed(1)} m` : `${vMax.toFixed(1)} m`;
    ctx.fillText(lo, lgX, lgY + lgH + 12);
    ctx.textAlign = 'right';
    ctx.fillText(hi, lgX + lgW, lgY + lgH + 12);
    ctx.textAlign = 'center';
    const modeLabel = mode === 'thick' ? `${unit.code} thickness`
                    : mode === 'topZ'  ? `${unit.code} top Z (mAOD)`
                    : `${unit.code} base Z (mAOD)`;
    ctx.fillText(modeLabel, lgX + lgW * 0.5, lgY + lgH + 12);

    // ── Title ─────────────────────────────────────────────────────────────────
    ctx.fillStyle = '#8898a8';
    ctx.font = 'bold 11px Inter, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(`${modeLabel} — ${unit.name}`, PAD, PAD - 6);
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
}

// ── Colour ramps ──────────────────────────────────────────────────────────────
function isopachColor(hex, norm) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgb(${Math.round(255+(r-255)*norm)},${Math.round(255+(g-255)*norm)},${Math.round(255+(b-255)*norm)})`;
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
