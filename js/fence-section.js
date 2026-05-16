// ── 2D Fence Section renderer ─────────────────────────────────────────────────
// Samples the voxel grid along a user-defined section line and draws a
// lithology column panel on a Canvas 2D element.

import { log } from './app.js';

export class FenceSection {
  constructor() {
    this._panel    = document.getElementById('fence-panel');
    this._canvas   = document.getElementById('fence-canvas');
    this._closeBtn = document.getElementById('fence-close');
    this._titleEl  = document.getElementById('fence-title');
    this._ctx      = this._canvas?.getContext('2d');
    this._visible  = false;

    this._closeBtn?.addEventListener('click', () => this.hide());
    window.addEventListener('keydown', e => {
      if (e.key === 'Escape' && this._visible) this.hide();
    });

    const resizeObs = new ResizeObserver(() => this._redraw());
    if (this._panel) resizeObs.observe(this._panel);
  }

  // ── Draw from slice normal+centerD, voxel grid, geo units ────────────────
  draw(grid, geoUnits, normal, centerD, thickness, boreholes) {
    if (!grid || !this._canvas || !this._ctx) return;
    this._lastArgs = { grid, geoUnits, normal, centerD, thickness, boreholes };
    this.show();
    this._redraw();
  }

  _redraw() {
    if (!this._lastArgs) return;
    const { grid, geoUnits, normal, centerD, thickness, boreholes } = this._lastArgs;
    const { nx, ny, nz, cellSize: cs, cellHeight: ch,
            origin: O, unitIds, certainty } = grid;

    // Build a lookup for geo units
    const unitById = {};
    geoUnits.forEach(u => { unitById[u.id] = u; });

    // Section orientation: along is perpendicular to normal in XZ plane
    const along = { x: normal.z, z: -normal.x };  // 90° CW
    const aLen  = Math.hypot(along.x, along.z) || 1;
    along.x /= aLen; along.z /= aLen;

    // Sample columns along the section line
    const panelW = this._canvas.parentElement?.clientWidth ?? 600;
    const panelH = this._canvas.parentElement?.clientHeight ?? 400;
    this._canvas.width  = panelW;
    this._canvas.height = panelH;

    const PAD_LEFT = 50, PAD_RIGHT = 20, PAD_TOP = 36, PAD_BOT = 40;
    const drawW = panelW - PAD_LEFT - PAD_RIGHT;
    const drawH = panelH - PAD_TOP  - PAD_BOT;

    const ctx = this._ctx;
    ctx.clearRect(0, 0, panelW, panelH);

    // Background
    ctx.fillStyle = '#f0f2f5';
    ctx.fillRect(0, 0, panelW, panelH);
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(PAD_LEFT, PAD_TOP, drawW, drawH);

    const worldW = Math.max(grid.worldWidth, grid.worldDepth) * 1.2;
    const worldH = grid.worldHeight;
    const botY   = grid.origin.y;
    const topY   = botY + worldH;

    // Number of sample columns
    const N_COLS = Math.max(40, Math.min(200, Math.floor(drawW / 3)));
    const colPx  = drawW / N_COLS;

    // For each column, project onto the section line and sample the voxel column
    for (let ci = 0; ci < N_COLS; ci++) {
      const t     = (ci / (N_COLS - 1)) - 0.5;  // -0.5 to +0.5
      const sWorld = t * worldW;
      const wx    = O.x + grid.worldWidth * 0.5 + along.x * sWorld + normal.x * (centerD - normal.x * (O.x + grid.worldWidth * 0.5) - normal.z * (O.z + grid.worldDepth * 0.5));
      const wz    = O.z + grid.worldDepth * 0.5 + along.z * sWorld + normal.z * (centerD - normal.x * (O.x + grid.worldWidth * 0.5) - normal.z * (O.z + grid.worldDepth * 0.5));

      const ix    = Math.floor((wx - O.x) / cs);
      const iy    = Math.floor((wz - O.z) / cs);
      if (ix < 0 || ix >= nx || iy < 0 || iy >= ny) continue;

      const colX = PAD_LEFT + ci * colPx;

      for (let iz = 0; iz < nz; iz++) {
        const flat = ix + iy * nx + iz * nx * ny;
        const unit = unitById[unitIds[flat]];
        if (!unit) continue;

        const cert   = certainty[flat];
        const elev   = O.y + iz * ch;
        const yPx    = PAD_TOP + drawH - ((elev - botY) / worldH) * drawH;
        const hPx    = Math.max(1, (ch / worldH) * drawH);

        ctx.globalAlpha = Math.max(0.35, cert);
        ctx.fillStyle   = unit.color;
        ctx.fillRect(colX, yPx - hPx, Math.ceil(colPx + 0.5), Math.ceil(hPx + 0.5));
      }
    }
    ctx.globalAlpha = 1;

    // Draw BH sticks on section (within half-thickness of the plane)
    if (boreholes?.length) {
      boreholes.filter(b => !b.synthetic).forEach(bh => {
        // Distance from bh to section plane
        const distToPlane = Math.abs(normal.x * bh.x + normal.z * bh.y - centerD);
        if (distToPlane > thickness * 0.55) return;

        // Along-axis coordinate
        const s = along.x * (bh.x - (O.x + grid.worldWidth * 0.5)) +
                  along.z * (bh.y - (O.z + grid.worldDepth * 0.5));
        const ciBH = Math.floor((s / worldW + 0.5) * N_COLS);
        if (ciBH < 0 || ciBH >= N_COLS) return;
        const bhX = PAD_LEFT + ciBH * colPx + colPx * 0.5;
        const gl  = bh.groundLevel ?? topY;

        // BH collar tick
        const collarY = PAD_TOP + drawH - ((gl - botY) / worldH) * drawH;
        ctx.fillStyle = '#1c2a38';
        ctx.fillRect(bhX - 1, collarY - 6, 2, 6);

        // Label
        ctx.font = 'bold 9px Inter, sans-serif';
        ctx.fillStyle = '#1c2a38';
        ctx.textAlign = 'center';
        ctx.fillText(bh.id, bhX, collarY - 8);
      });
    }

    // Y axis (elevation) ticks
    ctx.globalAlpha = 1;
    ctx.strokeStyle = '#c8cdd6';
    ctx.lineWidth   = 1;
    ctx.fillStyle   = '#4a6275';
    ctx.font        = '10px Inter, sans-serif';
    ctx.textAlign   = 'right';

    const depthRange = worldH;
    const tickStep   = depthRange <= 15 ? 1 : depthRange <= 50 ? 5 : depthRange <= 200 ? 10 : 25;
    const firstTick  = Math.ceil(botY / tickStep) * tickStep;
    for (let ev = firstTick; ev <= topY + 0.01; ev += tickStep) {
      const yPx = PAD_TOP + drawH - ((ev - botY) / worldH) * drawH;
      ctx.beginPath();
      ctx.moveTo(PAD_LEFT - 4, yPx);
      ctx.lineTo(PAD_LEFT + drawW, yPx);
      ctx.strokeStyle = '#e0e4ea';
      ctx.stroke();
      ctx.fillText(`${ev.toFixed(0)}`, PAD_LEFT - 6, yPx + 3);
    }

    // Frame
    ctx.strokeStyle = '#c8cdd6';
    ctx.lineWidth = 1;
    ctx.strokeRect(PAD_LEFT, PAD_TOP, drawW, drawH);

    // Title
    ctx.fillStyle = '#8898a8';
    ctx.font = 'bold 10px Inter, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('Section (mAOD)', 6, PAD_TOP + 12);

    // X-axis label
    ctx.textAlign = 'center';
    ctx.fillText('Distance along section →', PAD_LEFT + drawW * 0.5, panelH - 6);

    // Legend strip at bottom
    const legendUnits = geoUnits.filter(u => {
      return grid.unitIds.some(id => unitById[id]?.code === u.code);
    });
    const lgW  = Math.min(60, drawW / (legendUnits.length + 1));
    const lgX0 = PAD_LEFT + (drawW - legendUnits.length * lgW) * 0.5;
    legendUnits.forEach((u, i) => {
      const lx = lgX0 + i * lgW;
      ctx.fillStyle = u.color;
      ctx.fillRect(lx, panelH - PAD_BOT + 8, lgW - 4, 10);
      ctx.fillStyle = '#4a6275';
      ctx.font = '9px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(u.code, lx + (lgW - 4) * 0.5, panelH - PAD_BOT + 28);
    });

    if (this._titleEl) this._titleEl.textContent = 'Geological Cross-Section';
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
}
