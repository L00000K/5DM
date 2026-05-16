// ── 2D Fence / Cross-Section renderer ────────────────────────────────────────
// Samples the voxel grid along the slicer's section line and draws a
// lithology panel with elevation grid, BH ticks and unit legend.

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
  draw(grid, geoUnits, normal, centerD, thickness, boreholes) {
    if (!grid) return;
    this._lastArgs = { grid, geoUnits, normal, centerD, thickness, boreholes };
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

    const { grid, geoUnits, normal, centerD, thickness, boreholes } = args;
    const { nx, ny, nz, cellSize: cs, cellHeight: ch, origin: O, unitIds, certainty } = grid;

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
        const flat = ix + iy * nx + iz * nx * ny;
        const unit = unitById[unitIds[flat]];
        if (!unit) continue;
        const cert = certainty[flat];
        const yPx  = PAD_T + drawH - ((iz * ch + ch * 0.5) / worldH) * drawH;
        const hPx  = Math.max(1, (ch / worldH) * drawH);
        ctx.globalAlpha = Math.max(0.4, cert);
        ctx.fillStyle   = unit.color;
        ctx.fillRect(colX, yPx - hPx, Math.ceil(colPx + 0.5), Math.ceil(hPx + 0.5));
      }
    }
    ctx.globalAlpha = 1;

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

    // ── BH ticks ──────────────────────────────────────────────────────────
    (boreholes ?? []).filter(b => !b.synthetic).forEach(bh => {
      const distToPlane = Math.abs(normal.x * bh.x + normal.z * bh.y - centerD);
      if (distToPlane > thickness * 0.55) return;

      const sDist = along.x * (bh.x - sx0) + along.z * (bh.y - sz0);
      const ci = Math.floor((sDist / worldW + 0.5) * N_COLS);
      if (ci < 0 || ci >= N_COLS) return;

      const bhX    = PAD_L + ci * colPx + colPx * 0.5;
      const gl     = bh.groundLevel ?? topY;
      const colYPx = PAD_T + drawH - ((gl - botY) / worldH) * drawH;

      ctx.strokeStyle = '#1c2a38';
      ctx.lineWidth   = 1.5;
      ctx.beginPath(); ctx.moveTo(bhX, colYPx); ctx.lineTo(bhX, PAD_T + drawH); ctx.stroke();

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
}
