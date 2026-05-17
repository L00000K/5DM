// ── Plan View — horizontal slice through the voxel model ──────────────────────
// Shows colour-coded unit distribution at a specified elevation (mAOD)
// with BH position overlay, unit legend, and PNG export.

export class PlanView {
  constructor() {
    this._panel      = document.getElementById('plan-view-panel');
    this._canvas     = document.getElementById('plan-view-canvas');
    this._ctx        = this._canvas?.getContext('2d');
    this._slider     = document.getElementById('plan-view-elevation');
    this._elevLabel  = document.getElementById('plan-view-elev-lbl');
    this._closeBtn   = document.getElementById('plan-view-close');
    this._exportBtn  = document.getElementById('plan-view-export');
    this._modeSelect = document.getElementById('plan-view-mode');
    this._visible    = false;
    this._lastArgs   = null;

    // ── Bbox / polygon drawing state ─────────────────────────────────────────
    this._drawMode   = null;      // null | 'bbox'
    this._drawStart  = null;      // {canvasX, canvasY, worldX, worldY}
    this._drawRect   = null;      // {x1,y1,x2,y2} in world coords (live preview)
    this._onBboxDone = null;      // callback({minX,maxX,minY,maxY})

    this._closeBtn?.addEventListener('click', () => this.hide());
    this._exportBtn?.addEventListener('click', () => this._exportPNG());
    this._slider?.addEventListener('input', () => {
      if (this._lastArgs) this._redraw();
    });
    this._modeSelect?.addEventListener('change', () => {
      if (this._lastArgs) this._redraw();
    });

    window.addEventListener('keydown', e => {
      if (e.key === 'Escape') {
        if (this._drawMode) { this._cancelDraw(); }
        else if (this._visible) { this.hide(); }
      }
    });

    // Canvas interaction for bbox drawing
    this._canvas?.addEventListener('mousedown',  e => this._onMouseDown(e));
    this._canvas?.addEventListener('mousemove',  e => this._onMouseMove(e));
    this._canvas?.addEventListener('mouseup',    e => this._onMouseUp(e));
    this._canvas?.addEventListener('mouseleave', e => { if (this._drawMode === 'bbox' && !this._drawStart) {} });

    const ro = new ResizeObserver(() => { if (this._visible) this._redraw(); });
    if (this._panel) ro.observe(this._panel);
  }

  // ── Bbox draw API: call startBboxDraw(callback) to let user drag a rectangle
  startBboxDraw(onDone) {
    this._drawMode  = 'bbox';
    this._drawStart = null;
    this._drawRect  = null;
    this._onBboxDone = onDone;
    if (this._canvas) this._canvas.style.cursor = 'crosshair';
  }

  _cancelDraw() {
    this._drawMode = null; this._drawStart = null; this._drawRect = null; this._onBboxDone = null;
    if (this._canvas) this._canvas.style.cursor = '';
    this._redraw();
  }

  _canvasToWorld(canvasX, canvasY) {
    const args = this._lastArgs;
    if (!args) return null;
    const { grid: { nx, ny, cellSize: cs, origin: O } } = args;
    const W   = this._canvas.clientWidth  || this._canvas.width;
    const H   = this._canvas.clientHeight || this._canvas.height;
    const PAD = 24;
    const drawW = W - 2 * PAD, drawH = H - 2 * PAD;
    const cellPxW = drawW / nx, cellPxH = drawH / ny;
    const ix = (canvasX - PAD) / cellPxW;
    const iy = ny - (canvasY - PAD) / cellPxH;   // flip Y (plan view has Y=0 at bottom)
    return { worldX: O.x + ix * cs, worldY: O.z + iy * cs };
  }

  _onMouseDown(e) {
    if (this._drawMode !== 'bbox') return;
    const rect = this._canvas.getBoundingClientRect();
    const cx = (e.clientX - rect.left) * (this._canvas.width / rect.width);
    const cy = (e.clientY - rect.top)  * (this._canvas.height / rect.height);
    const world = this._canvasToWorld(cx, cy);
    if (!world) return;
    this._drawStart = { cx, cy, ...world };
    this._drawRect  = { x1: world.worldX, y1: world.worldY, x2: world.worldX, y2: world.worldY };
  }

  _onMouseMove(e) {
    if (this._drawMode !== 'bbox' || !this._drawStart) return;
    const rect = this._canvas.getBoundingClientRect();
    const cx = (e.clientX - rect.left) * (this._canvas.width / rect.width);
    const cy = (e.clientY - rect.top)  * (this._canvas.height / rect.height);
    const world = this._canvasToWorld(cx, cy);
    if (!world) return;
    this._drawRect = { x1: this._drawStart.worldX, y1: this._drawStart.worldY, x2: world.worldX, y2: world.worldY };
    this._redraw();
  }

  _onMouseUp(e) {
    if (this._drawMode !== 'bbox' || !this._drawStart || !this._drawRect) return;
    const { x1, y1, x2, y2 } = this._drawRect;
    const domain = {
      minX: Math.min(x1, x2), maxX: Math.max(x1, x2),
      minY: Math.min(y1, y2), maxY: Math.max(y1, y2),
    };
    const cb = this._onBboxDone;
    this._cancelDraw();
    if (cb && Math.abs(domain.maxX - domain.minX) > 0.5) cb(domain);
  }

  draw(grid, geoUnits, boreholes, conceptStore = null) {
    if (!grid) return;
    this._lastArgs = { grid, geoUnits, boreholes, conceptStore };

    const { origin: O, nz, cellHeight: ch } = grid;
    const minElev = O.y;
    const maxElev = O.y + nz * ch;

    if (this._slider) {
      this._slider.min  = minElev.toFixed(2);
      this._slider.max  = maxElev.toFixed(2);
      this._slider.step = (ch * 0.5).toFixed(2);
      const cur = parseFloat(this._slider.value);
      if (cur < minElev || cur > maxElev) {
        this._slider.value = ((minElev + maxElev) / 2).toFixed(2);
      }
    }

    this.show();
    this._redraw();
  }

  // Jet-like colormap: t ∈ [0,1] → {r,g,b} 0-255
  static _jet(t) {
    const r = t < 0.5 ? Math.round(t * 510) : 255;
    const g = t < 0.25 ? Math.round(t * 1020)
            : t < 0.75 ? 255
            : Math.round((1 - t) * 1020);
    const b = t > 0.5 ? Math.round((1 - t) * 510) : 255;
    return `rgb(${r},${g},${b})`;
  }

  _redraw() {
    const args = this._lastArgs;
    if (!args || !this._canvas || !this._ctx) return;
    const { grid, geoUnits, boreholes, conceptStore } = args;
    const { nx, ny, nz, cellSize: cs, cellHeight: ch, origin: O, unitIds, certainty } = grid;

    const mode = this._modeSelect?.value ?? 'unit';
    const elev = parseFloat(this._slider?.value ?? O.y);
    const iz   = Math.max(0, Math.min(nz - 1, Math.floor((elev - O.y) / ch)));
    if (this._elevLabel) this._elevLabel.textContent = `${elev.toFixed(1)} mAOD`;

    const unitMap = {};
    geoUnits.forEach(u => { unitMap[u.id] = u; });

    // Populate / show probability unit selector
    const probUnitSel = document.getElementById('plan-view-prob-unit');
    if (probUnitSel) {
      const show = mode === 'probability';
      probUnitSel.parentElement.style.display = show ? 'flex' : 'none';
      if (show && probUnitSel.options.length !== geoUnits.length) {
        const cur = probUnitSel.value;
        probUnitSel.innerHTML = geoUnits.map(u => `<option value="${u.code}">${u.code} — ${u.name}</option>`).join('');
        if (cur) probUnitSel.value = cur;
      }
    }
    const probUnitCode = probUnitSel?.value ?? geoUnits[0]?.code;

    const PAD = 40;
    const W = this._canvas.parentElement?.clientWidth  ?? 420;
    const H = this._canvas.parentElement?.clientHeight ?? 370;
    this._canvas.width  = W;
    this._canvas.height = H;

    const drawW   = W - PAD * 2;
    const drawH   = H - PAD - 52;
    if (drawW < 40 || drawH < 40) return;
    const cellPxW = drawW / nx;
    const cellPxH = drawH / ny;

    const ctx = this._ctx;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#f0f2f5'; ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = '#ffffff'; ctx.fillRect(PAD, PAD, drawW, drawH);

    // ── Compute per-column geotechnical parameters for settlement/bearing modes ──
    // For each (ix, iy), scan top 10m of voxels to accumulate Cu, N_spt, Cc
    let colCu = null, colN = null, colCc = null;
    let minCu = Infinity, maxCu = 0, minN = Infinity, maxN = 0, minCc = Infinity, maxCc = 0;
    if (['cu', 'N_spt', 'bearing', 'settlement'].includes(mode)) {
      colCu = new Float32Array(nx * ny).fill(-1);
      colN  = new Float32Array(nx * ny).fill(-1);
      colCc = new Float32Array(nx * ny).fill(-1);

      // Find surface iz for each column
      for (let iy = 0; iy < ny; iy++) {
        for (let ix = 0; ix < nx; ix++) {
          let surfIz = -1;
          for (let jz = nz - 1; jz >= 0; jz--) {
            if (unitIds[ix + iy*nx + jz*nx*ny]) { surfIz = jz; break; }
          }
          if (surfIz < 0) continue;
          const surfElev = O.y + surfIz * ch + ch;
          let cuMin = Infinity, nSum = 0, nCnt = 0, ccSum = 0, ccCnt = 0;

          for (let jz = surfIz; jz >= 0; jz--) {
            const vElev = O.y + jz * ch + ch * 0.5;
            const depth = surfElev - vElev;
            if (depth > 10) break;
            const uid  = unitIds[ix + iy*nx + jz*nx*ny];
            const unit = unitMap[uid];
            if (!unit) continue;
            const p = unit.params ?? {};
            if (depth <= 5 && p.cu != null) cuMin = Math.min(cuMin, p.cu);
            if (p.N_spt != null) { nSum += p.N_spt; nCnt++; }
            if (p.Cc  != null) { ccSum += p.Cc;    ccCnt++; }
          }
          const ci = ix + iy * nx;
          if (cuMin < Infinity) { colCu[ci] = cuMin; minCu = Math.min(minCu, cuMin); maxCu = Math.max(maxCu, cuMin); }
          if (nCnt)             { colN[ci]  = nSum/nCnt; minN = Math.min(minN, nSum/nCnt); maxN = Math.max(maxN, nSum/nCnt); }
          if (ccCnt)            { colCc[ci] = ccSum/ccCnt; minCc = Math.min(minCc, ccSum/ccCnt); maxCc = Math.max(maxCc, ccSum/ccCnt); }
        }
      }
    }

    const cuRange  = Math.max(maxCu  - minCu,  1);
    const nRange   = Math.max(maxN   - minN,   1);
    const ccRange  = Math.max(maxCc  - minCc,  0.001);

    let paramMin = 0, paramMax = 100, paramLabel = '';

    for (let iy = 0; iy < ny; iy++) {
      for (let ix = 0; ix < nx; ix++) {
        const flat  = ix + iy * nx + iz * nx * ny;
        const uid   = unitIds[flat];
        const unit  = unitMap[uid];
        if (!uid) continue;

        let color;
        if (mode === 'unit') {
          color = unit?.color ?? '#888';
        } else if (mode === 'cert') {
          const cert = certainty[flat];
          const g = Math.round(cert * 200);
          color = `rgb(${Math.round(255 - cert * 150)},${g + 55},${Math.round(cert * 100 + 50)})`;
        } else if (mode === 'cu' || mode === 'bearing') {
          const v = colCu?.[ix + iy*nx] ?? -1;
          if (v < 0) { color = '#cccccc'; }
          else {
            const t = mode === 'bearing'
              ? 1 - Math.min((v - minCu) / cuRange, 1)   // low Cu = high risk = red
              : (v - minCu) / cuRange;                    // high Cu = blue
            color = PlanView._jet(t);
            paramMin = minCu; paramMax = maxCu; paramLabel = 'Cu (kPa) — blue=high';
          }
        } else if (mode === 'N_spt') {
          const v = colN?.[ix + iy*nx] ?? -1;
          if (v < 0) { color = '#cccccc'; }
          else {
            color = PlanView._jet((v - minN) / nRange);
            paramMin = minN; paramMax = maxN; paramLabel = 'SPT N — blue=low';
          }
        } else if (mode === 'settlement') {
          const v = colCc?.[ix + iy*nx] ?? -1;
          if (v < 0) { color = '#cccccc'; }
          else {
            const t = (v - minCc) / ccRange; // high Cc = high settlement = red
            color = PlanView._jet(1 - t);
            paramMin = minCc; paramMax = maxCc; paramLabel = 'Cc — red=high compressibility';
          }
        } else if (mode === 'probability') {
          // Probability approximation: certainty × (1 if winning unit, blendRatio if runner-up)
          const { blendUnitIds, blendRatios } = grid;
          const targetId = geoUnits.find(u => u.code === probUnitCode)?.id;
          if (targetId === undefined) { color = '#cccccc'; }
          else {
            let p = 0;
            if (uid === targetId)                  p = certainty[flat];
            else if (blendUnitIds[flat] === targetId) p = blendRatios[flat] * certainty[flat];
            const targetUnit = geoUnits.find(u => u.code === probUnitCode);
            const hex = targetUnit?.color ?? '#4472c4';
            const r = parseInt(hex.slice(1, 3), 16);
            const g = parseInt(hex.slice(3, 5), 16);
            const b = parseInt(hex.slice(5, 7), 16);
            color = `rgba(${r},${g},${b},${Math.max(0.05, p)})`;
            paramMin = 0; paramMax = 100;
            paramLabel = `P(${probUnitCode}) %`;
          }
        } else if (mode === 'concept' && conceptStore) {
          // Concept influence heatmap — rendered after main cells below
          color = null; // skip normal fill; handled in post-pass
        } else {
          color = '#888';
        }

        if (color) {
          ctx.fillStyle = color;
          ctx.fillRect(
            PAD + ix * cellPxW,
            PAD + (ny - 1 - iy) * cellPxH,
            Math.ceil(cellPxW + 0.5),
            Math.ceil(cellPxH + 0.5)
          );
        }
      }
    }

    // Concept influence heatmap — sampled at each grid cell's world position
    if (mode === 'concept' && conceptStore && !conceptStore.isEmpty) {
      // Single-pass: cache weights and find max simultaneously
      const wCache = new Float32Array(nx * ny);
      let maxW = 0;
      for (let iy = 0; iy < ny; iy++) {
        for (let ix = 0; ix < nx; ix++) {
          const wx = O.x + (ix + 0.5) * cs;
          const wy = O.z + (iy + 0.5) * cs;
          const w = conceptStore.computeAt(wx, wy, elev).totalWeight;
          wCache[ix + iy * nx] = w;
          if (w > maxW) maxW = w;
        }
      }
      if (maxW < 0.01) maxW = 1;

      // Render from cache
      for (let iy = 0; iy < ny; iy++) {
        for (let ix = 0; ix < nx; ix++) {
          const t   = Math.min(1, wCache[ix + iy * nx] / maxW);
          // Color: low influence = light grey, high = deep blue-purple
          const r = Math.round(240 - t * 180);
          const g = Math.round(242 - t * 160);
          const b = Math.round(245 - t * 50 + t * 100); // more blue
          ctx.fillStyle = `rgb(${r},${g},${b})`;
          ctx.fillRect(
            PAD + ix * cellPxW,
            PAD + (ny - 1 - iy) * cellPxH,
            Math.ceil(cellPxW + 0.5),
            Math.ceil(cellPxH + 0.5)
          );
        }
      }

      // Draw bbox domains of spatially-constrained concepts
      conceptStore.concepts.forEach((c, ci) => {
        if (c.domain?.type !== 'bbox') return;
        const { minX, maxX, minY, maxY } = c.domain;
        const px1 = PAD + ((minX - O.x) / cs) * cellPxW;
        const px2 = PAD + ((maxX - O.x) / cs) * cellPxW;
        const py1 = PAD + (ny - (maxY - O.z) / cs) * cellPxH;
        const py2 = PAD + (ny - (minY - O.z) / cs) * cellPxH;
        const hue = (ci * 137) % 360;
        ctx.strokeStyle = `hsla(${hue},80%,45%,0.85)`;
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 3]);
        ctx.strokeRect(px1, py1, px2 - px1, py2 - py1);
        ctx.setLineDash([]);
        ctx.fillStyle = `hsla(${hue},80%,45%,0.9)`;
        ctx.font = 'bold 9px Inter, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText(c.description.slice(0, 24), px1 + 3, py1 - 3);
      });

      paramLabel = 'Concept Influence';
      paramMin = 0; paramMax = 100;
    }

    // Certainty threshold contour overlay
    // Draws a dashed red boundary around zones where certainty < 0.4
    const CERT_THRESHOLD = 0.4;
    ctx.strokeStyle = 'rgba(220,50,50,0.65)';
    ctx.lineWidth   = 0.8;
    ctx.setLineDash([3, 2]);
    for (let iy = 0; iy < ny; iy++) {
      for (let ix = 0; ix < nx; ix++) {
        const flat  = ix + iy * nx + iz * nx * ny;
        const c     = certainty[flat];
        if (!unitIds[flat] || c >= CERT_THRESHOLD) continue;
        const px = PAD + ix * cellPxW;
        const py = PAD + (ny - 1 - iy) * cellPxH;
        const w  = Math.ceil(cellPxW + 0.5);
        const h  = Math.ceil(cellPxH + 0.5);
        // Only draw the edges that border a confident cell (to form a contour)
        const left  = ix === 0       || certainty[ix-1 + iy*nx + iz*nx*ny] >= CERT_THRESHOLD;
        const right = ix === nx - 1  || certainty[ix+1 + iy*nx + iz*nx*ny] >= CERT_THRESHOLD;
        const down  = iy === 0       || certainty[ix + (iy-1)*nx + iz*nx*ny] >= CERT_THRESHOLD;
        const up    = iy === ny - 1  || certainty[ix + (iy+1)*nx + iz*nx*ny] >= CERT_THRESHOLD;
        ctx.beginPath();
        if (left)  { ctx.moveTo(px,   py);   ctx.lineTo(px,   py+h); }
        if (right) { ctx.moveTo(px+w, py);   ctx.lineTo(px+w, py+h); }
        if (up)    { ctx.moveTo(px,   py);   ctx.lineTo(px+w, py);   }
        if (down)  { ctx.moveTo(px,   py+h); ctx.lineTo(px+w, py+h); }
        ctx.stroke();
      }
    }
    ctx.setLineDash([]);
    ctx.lineWidth = 1;

    // Concept anisotropy ellipse — shows global warp tensor as a reference ellipse
    // centred in the plan view. Helps verify that E-W palaeochannel → wide E-W ellipse.
    if (conceptStore && !conceptStore.isEmpty) {
      const gT = conceptStore.globalTensor();
      // Only draw if meaningful anisotropy
      if (Math.max(gT.Ax, gT.Ay) > 1.15 || Math.min(gT.Ax, gT.Ay) < 0.88) {
        const ex = (drawW / 2) + PAD;
        const ey = (drawH / 2) + PAD;
        // Radius in pixels proportional to grid size, scaled by tensor
        const baseR = Math.min(drawW, drawH) * 0.12;
        const rx = baseR * Math.min(gT.Ax, 5);  // E-W radius
        const ry = baseR * Math.min(gT.Ay, 5);  // N-S radius
        // Filled ellipse (faint)
        ctx.beginPath();
        ctx.ellipse(ex, ey, rx, ry, 0, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(64,180,255,0.08)';
        ctx.fill();
        ctx.strokeStyle = 'rgba(64,180,255,0.55)';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 3]);
        ctx.stroke();
        ctx.setLineDash([]);
        // Axis labels
        ctx.fillStyle = 'rgba(64,180,255,0.8)';
        ctx.font = 'bold 9px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(`E-W ×${gT.Ax.toFixed(1)}`, ex, ey - ry - 4);
        ctx.textAlign = 'right';
        ctx.fillText(`N-S ×${gT.Ay.toFixed(1)}`, ex - rx - 3, ey + 3);
      }
    }

    // Live bbox drawing preview
    if (this._drawMode === 'bbox' && this._drawRect) {
      const { x1, y1, x2, y2 } = this._drawRect;
      const px1 = PAD + ((Math.min(x1,x2) - O.x) / cs) * cellPxW;
      const px2 = PAD + ((Math.max(x1,x2) - O.x) / cs) * cellPxW;
      const py1 = PAD + (ny - (Math.max(y1,y2) - O.z) / cs) * cellPxH;
      const py2 = PAD + (ny - (Math.min(y1,y2) - O.z) / cs) * cellPxH;
      ctx.fillStyle = 'rgba(255,180,30,0.12)';
      ctx.fillRect(px1, py1, px2 - px1, py2 - py1);
      ctx.strokeStyle = 'rgba(255,160,10,0.9)';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 3]);
      ctx.strokeRect(px1, py1, px2 - px1, py2 - py1);
      ctx.setLineDash([]);
      ctx.fillStyle = 'rgba(255,160,10,0.9)';
      ctx.font = 'bold 10px Inter, sans-serif';
      ctx.textAlign = 'center';
      const w = Math.abs(x2 - x1).toFixed(0), d = Math.abs(y2 - y1).toFixed(0);
      ctx.fillText(`${w}×${d}m`, (px1 + px2) / 2, (py1 + py2) / 2);
    } else if (this._drawMode === 'bbox') {
      // Show crosshair hint before first click
      ctx.fillStyle = 'rgba(255,160,10,0.85)';
      ctx.font = '11px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Click and drag to define spatial domain', PAD + drawW / 2, PAD + drawH / 2);
    }

    // Borehole markers
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

    ctx.strokeStyle = '#c8cdd6'; ctx.lineWidth = 1;
    ctx.strokeRect(PAD, PAD, drawW, drawH);

    const MODE_LABELS = { unit: 'Geology', cert: 'Certainty', cu: 'Undrained Strength (Cu)',
      N_spt: 'SPT N', settlement: 'Settlement Risk (Cc)', bearing: 'Bearing Capacity Risk (Cu)',
      probability: `P(${probUnitCode})`, concept: 'Concept Influence (semantic warp)' };
    const modeLabel = MODE_LABELS[mode] ?? mode;
    ctx.fillStyle = '#8898a8';
    ctx.font = 'bold 11px Inter, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(`Plan View — ${modeLabel}`, PAD, PAD - 6);

    // Color scale bar for geotechnical modes
    if (!['unit', 'cert'].includes(mode) && paramLabel) {
      const scaleW = 120, scaleH = 10;
      const scX = W - PAD - scaleW, scY = PAD - 18;
      const grad = ctx.createLinearGradient(scX, 0, scX + scaleW, 0);
      grad.addColorStop(0, PlanView._jet(0));
      grad.addColorStop(0.5, PlanView._jet(0.5));
      grad.addColorStop(1, PlanView._jet(1));
      ctx.fillStyle = grad;
      ctx.fillRect(scX, scY, scaleW, scaleH);
      ctx.strokeStyle = '#c8cdd6'; ctx.lineWidth = 0.5;
      ctx.strokeRect(scX, scY, scaleW, scaleH);
      ctx.fillStyle = '#8898a8'; ctx.font = '8px Inter, sans-serif';
      ctx.textAlign = 'left'; ctx.fillText(paramMin.toFixed(1), scX, scY + scaleH + 10);
      ctx.textAlign = 'right'; ctx.fillText(paramMax.toFixed(1), scX + scaleW, scY + scaleH + 10);
    }

    // Legend
    const lgY = PAD + drawH + 10;
    if (mode === 'unit') {
      let lx = PAD;
      ctx.font = '9px Inter, sans-serif';
      ctx.textBaseline = 'middle';
      for (const u of geoUnits) {
        if (lx > W - 50) break;
        ctx.fillStyle = u.color;
        ctx.fillRect(lx, lgY - 4, 10, 8);
        ctx.fillStyle = '#4a6275';
        ctx.textAlign = 'left';
        ctx.fillText(u.code, lx + 13, lgY);
        lx += 14 + ctx.measureText(u.code).width + 8;
      }
    } else {
      // Certainty gradient bar
      const lgW = drawW, lgH = 10;
      for (let i = 0; i < lgW; i++) {
        const cert = i / lgW;
        const g = Math.round(cert * 200);
        ctx.fillStyle = `rgb(${Math.round(255-cert*150)},${g+55},${Math.round(cert*100+50)})`;
        ctx.fillRect(PAD + i, lgY - 2, 1, lgH);
      }
      ctx.strokeStyle = '#c8cdd6'; ctx.strokeRect(PAD, lgY - 2, lgW, lgH);
      ctx.fillStyle = '#4a6275'; ctx.font = '9px Inter, sans-serif';
      ctx.textAlign = 'left';  ctx.fillText('Low', PAD, lgY + lgH + 8);
      ctx.textAlign = 'right'; ctx.fillText('High certainty', PAD + lgW, lgY + lgH + 8);
    }
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
    a.download = `plan-view-${new Date().toISOString().slice(0, 10)}.png`;
    a.click();
  }
}
