// ── Isopach / thickness map ────────────────────────────────────────────────────
// Computes per-column thickness of each geological unit and renders as a
// colour-ramp plan-view canvas overlay.

export class IsopachMap {
  constructor() {
    this._panel    = document.getElementById('isopach-panel');
    this._canvas   = document.getElementById('isopach-canvas');
    this._closeBtn = document.getElementById('isopach-close');
    this._exportBtn = document.getElementById('isopach-export');
    this._select    = document.getElementById('isopach-unit-select');
    this._ctx      = this._canvas?.getContext('2d');
    this._visible  = false;
    this._lastArgs = null;

    this._closeBtn?.addEventListener('click', () => this.hide());
    this._exportBtn?.addEventListener('click', () => this._exportPNG());
    this._select?.addEventListener('change', () => {
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

    // Populate unit selector
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

    // Compute thickness per XY column
    const thick = new Float32Array(nx * ny).fill(0);
    let maxThick = 0;

    for (let iy = 0; iy < ny; iy++) {
      for (let ix = 0; ix < nx; ix++) {
        let cnt = 0;
        for (let iz = 0; iz < nz; iz++) {
          if (unitIds[ix + iy * nx + iz * nx * ny] === targetId) cnt++;
        }
        const t = cnt * ch;
        thick[ix + iy * nx] = t;
        if (t > maxThick) maxThick = t;
      }
    }

    const panelW = this._canvas.parentElement?.clientWidth  ?? 400;
    const panelH = this._canvas.parentElement?.clientHeight ?? 360;
    this._canvas.width  = panelW;
    this._canvas.height = panelH;

    const PAD = 40;
    const drawW = panelW - PAD * 2;
    const drawH = panelH - PAD - 56;
    if (drawW < 40 || drawH < 40) return;

    const cellPxW = drawW / nx;
    const cellPxH = drawH / ny;

    const ctx = this._ctx;
    ctx.clearRect(0, 0, panelW, panelH);
    ctx.fillStyle = '#f0f2f5';
    ctx.fillRect(0, 0, panelW, panelH);
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(PAD, PAD, drawW, drawH);

    // Draw thickness colour map
    for (let iy = 0; iy < ny; iy++) {
      for (let ix = 0; ix < nx; ix++) {
        const t = thick[ix + iy * nx];
        if (t < 0.01) continue;
        const norm = maxThick > 0 ? t / maxThick : 0;
        ctx.fillStyle = isopachColor(unit.color, norm);
        ctx.fillRect(
          PAD + ix * cellPxW,
          PAD + (ny - 1 - iy) * cellPxH,  // flip Y (north at top)
          Math.ceil(cellPxW + 0.5),
          Math.ceil(cellPxH + 0.5)
        );
      }
    }

    // Draw BH positions
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

    // Frame
    ctx.strokeStyle = '#c8cdd6';
    ctx.lineWidth = 1;
    ctx.strokeRect(PAD, PAD, drawW, drawH);

    // Colour legend bar
    const lgX = PAD, lgY = PAD + drawH + 8, lgW = drawW, lgH = 12;
    for (let i = 0; i < lgW; i++) {
      ctx.fillStyle = isopachColor(unit.color, i / lgW);
      ctx.fillRect(lgX + i, lgY, 1, lgH);
    }
    ctx.strokeStyle = '#c8cdd6';
    ctx.strokeRect(lgX, lgY, lgW, lgH);
    ctx.fillStyle = '#4a6275';
    ctx.font = '10px Inter, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('0 m', lgX, lgY + lgH + 12);
    ctx.textAlign = 'right';
    ctx.fillText(`${maxThick.toFixed(1)} m`, lgX + lgW, lgY + lgH + 12);
    ctx.textAlign = 'center';
    ctx.fillText(`${unit.code} thickness`, lgX + lgW * 0.5, lgY + lgH + 12);

    // Title
    ctx.fillStyle = '#8898a8';
    ctx.font = 'bold 11px Inter, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(`Isopach — ${unit.code}: ${unit.name}`, PAD, PAD - 6);
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

  _exportPNG() {
    if (!this._canvas) return;
    const url = this._canvas.toDataURL('image/png');
    const a   = document.createElement('a');
    a.href     = url;
    a.download = `isopach-${new Date().toISOString().slice(0, 10)}.png`;
    a.click();
  }
}

// ── Colour ramp: white → unit colour (low → high thickness) ──────────────────
function isopachColor(hex, norm) {
  // Parse hex to RGB
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  // Blend white (255,255,255) → unit colour
  const ir = Math.round(255 + (r - 255) * norm);
  const ig = Math.round(255 + (g - 255) * norm);
  const ib = Math.round(255 + (b - 255) * norm);
  return `rgb(${ir},${ig},${ib})`;
}
