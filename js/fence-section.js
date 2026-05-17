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
