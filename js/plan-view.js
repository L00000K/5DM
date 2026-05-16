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

    this._closeBtn?.addEventListener('click', () => this.hide());
    this._exportBtn?.addEventListener('click', () => this._exportPNG());
    this._slider?.addEventListener('input', () => {
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
    const { grid, geoUnits, boreholes } = args;
    const { nx, ny, nz, cellSize: cs, cellHeight: ch, origin: O, unitIds, certainty } = grid;

    const mode = this._modeSelect?.value ?? 'unit';
    const elev = parseFloat(this._slider?.value ?? O.y);
    const iz   = Math.max(0, Math.min(nz - 1, Math.floor((elev - O.y) / ch)));
    if (this._elevLabel) this._elevLabel.textContent = `${elev.toFixed(1)} mAOD`;

    const unitMap = {};
    geoUnits.forEach(u => { unitMap[u.id] = u; });

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
        } else {
          color = '#888';
        }

        ctx.fillStyle = color;
        ctx.fillRect(
          PAD + ix * cellPxW,
          PAD + (ny - 1 - iy) * cellPxH,
          Math.ceil(cellPxW + 0.5),
          Math.ceil(cellPxH + 0.5)
        );
      }
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
      N_spt: 'SPT N', settlement: 'Settlement Risk (Cc)', bearing: 'Bearing Capacity Risk (Cu)' };
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
