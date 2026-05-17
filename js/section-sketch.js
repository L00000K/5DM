// ── Section Sketch Canvas ──────────────────────────────────────────────────────
// 2D canvas for the user to draw geological unit boundary polylines on a
// cross-section view. Strokes are converted to virtual boreholes for interpolation.

import { sketchToVirtualBoreholes, fenceLength } from './section-interpreter.js';

export class SectionSketch {
  constructor(canvas, infoEl) {
    this.canvas  = canvas;
    this.infoEl  = infoEl;
    this.ctx     = canvas.getContext('2d');
    this.fence   = null;  // {startX, startY, endX, endY}
    this.grid    = null;  // voxelGrid for background rendering
    this.geoUnits = [];
    this.strokes  = [];   // [{unitCode, color, points: [{distM, depthM}]}]
    this.current  = null; // stroke in progress
    this.activeUnit = null;
    this.maxDepth = 30;   // m
    this._pad = { l: 40, r: 10, t: 10, b: 28 };
    this._bound = this._onMouseDown.bind(this);
    this._bmove = this._onMouseMove.bind(this);
    this._bup   = this._onMouseUp.bind(this);
    canvas.addEventListener('mousedown', this._bound);
    canvas.addEventListener('mousemove', this._bmove);
    canvas.addEventListener('mouseup',   this._bup);
    canvas.addEventListener('mouseleave', this._bup);
    canvas.addEventListener('touchstart',  e => { e.preventDefault(); this._onMouseDown(e.touches[0]); }, { passive: false });
    canvas.addEventListener('touchmove',   e => { e.preventDefault(); this._onMouseMove(e.touches[0]); }, { passive: false });
    canvas.addEventListener('touchend',    e => { e.preventDefault(); this._onMouseUp(); });
  }

  // Set the fence + background data and re-render
  setContext(fence, geoUnits, maxDepth = 30, grid = null) {
    this.fence    = fence;
    this.geoUnits = geoUnits;
    this.maxDepth = maxDepth;
    this.grid     = grid;
    this.strokes  = [];
    this.current  = null;
    this._render();
  }

  setActiveUnit(unitCode) {
    this.activeUnit = unitCode;
  }

  // ── Coordinate transforms ─────────────────────────────────────────────────

  _W() { return this.canvas.width  - this._pad.l - this._pad.r; }
  _H() { return this.canvas.height - this._pad.t - this._pad.b; }

  _canvasX(distM) {
    return this._pad.l + (distM / (fenceLength(this.fence) || 1)) * this._W();
  }
  _canvasY(depthM) {
    return this._pad.t + (depthM / this.maxDepth) * this._H();
  }
  _distM(cx) {
    return Math.max(0, Math.min(fenceLength(this.fence), (cx - this._pad.l) / this._W() * fenceLength(this.fence)));
  }
  _depthM(cy) {
    return Math.max(0, Math.min(this.maxDepth, (cy - this._pad.t) / this._H() * this.maxDepth));
  }
  _clientToCanvas(e) {
    const r  = this.canvas.getBoundingClientRect();
    const sx = this.canvas.width  / r.width;
    const sy = this.canvas.height / r.height;
    return { cx: (e.clientX - r.left) * sx, cy: (e.clientY - r.top) * sy };
  }

  // ── Mouse handlers ────────────────────────────────────────────────────────

  _onMouseDown(e) {
    if (!this.fence || !this.activeUnit) return;
    const { cx, cy } = this._clientToCanvas(e);
    const unit = this.geoUnits.find(u => u.code === this.activeUnit);
    this.current = {
      unitCode: this.activeUnit,
      color:    unit?.color ?? '#888',
      points:   [{ distM: this._distM(cx), depthM: this._depthM(cy) }],
    };
  }

  _onMouseMove(e) {
    if (!this.current) return;
    const { cx, cy } = this._clientToCanvas(e);
    const last = this.current.points[this.current.points.length - 1];
    const d = this._distM(cx), z = this._depthM(cy);
    // Only add point if moved ≥ 3 canvas px
    if (Math.hypot(this._canvasX(d) - this._canvasX(last.distM),
                   this._canvasY(z) - this._canvasY(last.depthM)) >= 3) {
      this.current.points.push({ distM: d, depthM: z });
      this._render();
    }
  }

  _onMouseUp() {
    if (!this.current) return;
    if (this.current.points.length >= 2) {
      this.strokes.push(this.current);
    }
    this.current = null;
    this._render();
    this._updateInfo();
  }

  // ── Rendering ─────────────────────────────────────────────────────────────

  _render() {
    const { ctx, canvas, _pad: pad } = this;
    const W = this._W(), H = this._H();
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!this.fence) {
      ctx.fillStyle = '#888';
      ctx.font = '12px monospace';
      ctx.textAlign = 'center';
      ctx.fillText('Select a fence line first', canvas.width / 2, canvas.height / 2);
      return;
    }

    // Background — gradient for "ground"
    const bg = ctx.createLinearGradient(0, pad.t, 0, pad.t + H);
    bg.addColorStop(0, '#1a2233');
    bg.addColorStop(1, '#0d1520');
    ctx.fillStyle = bg;
    ctx.fillRect(pad.l, pad.t, W, H);

    // Ground surface line
    ctx.strokeStyle = '#4a7c59';
    ctx.lineWidth   = 2;
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(pad.l, pad.t);
    ctx.lineTo(pad.l + W, pad.t);
    ctx.stroke();

    // Depth grid lines
    ctx.strokeStyle = '#ffffff18';
    ctx.lineWidth   = 0.5;
    const depthStep = this.maxDepth > 20 ? 5 : 2;
    for (let d = depthStep; d < this.maxDepth; d += depthStep) {
      const y = this._canvasY(d);
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + W, y); ctx.stroke();
    }

    // Distance grid lines
    const fLen = fenceLength(this.fence);
    const distStep = Math.pow(10, Math.floor(Math.log10(fLen / 4)));
    for (let d = distStep; d < fLen; d += distStep) {
      const x = this._canvasX(d);
      ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, pad.t + H); ctx.stroke();
    }

    // Axis labels — depth
    ctx.fillStyle = '#888';
    ctx.font      = '9px monospace';
    ctx.textAlign = 'right';
    for (let d = 0; d <= this.maxDepth; d += depthStep) {
      ctx.fillText(d + 'm', pad.l - 4, this._canvasY(d) + 3);
    }

    // Axis labels — distance
    ctx.textAlign = 'center';
    for (let d = 0; d <= fLen; d += distStep) {
      ctx.fillText(Math.round(d) + 'm', this._canvasX(d), pad.t + H + 12);
    }

    // Axis titles
    ctx.save();
    ctx.translate(10, pad.t + H / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillStyle = '#aaa';
    ctx.fillText('Depth (m)', 0, 0);
    ctx.restore();

    // Committed strokes
    for (const stroke of this.strokes) {
      this._drawStroke(stroke, false);
    }

    // Active stroke in progress
    if (this.current) {
      this._drawStroke(this.current, true);
    }

    // Unit legend overlay
    const units = this.geoUnits.filter(u => u.code !== 'UNKN');
    units.forEach((u, i) => {
      const isActive = u.code === this.activeUnit;
      ctx.fillStyle = u.color + (isActive ? 'ff' : '99');
      ctx.fillRect(pad.l + 4 + i * 50, pad.t + 4, 12, 12);
      ctx.fillStyle = isActive ? '#fff' : '#aaa';
      ctx.font = isActive ? 'bold 9px monospace' : '9px monospace';
      ctx.textAlign = 'left';
      ctx.fillText(u.code, pad.l + 18 + i * 50, pad.t + 13);
    });
  }

  _drawStroke(stroke, active) {
    const { ctx } = this;
    if (stroke.points.length < 1) return;
    ctx.strokeStyle = stroke.color;
    ctx.lineWidth   = active ? 2.5 : 2;
    ctx.setLineDash(active ? [4, 3] : []);
    ctx.lineCap     = 'round';
    ctx.lineJoin    = 'round';
    ctx.globalAlpha = active ? 0.8 : 1.0;
    ctx.beginPath();
    stroke.points.forEach((p, i) => {
      const x = this._canvasX(p.distM), y = this._canvasY(p.depthM);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    // Label at end of stroke
    if (stroke.points.length > 0 && !active) {
      const last = stroke.points[stroke.points.length - 1];
      ctx.fillStyle = stroke.color;
      ctx.font = 'bold 9px monospace';
      ctx.textAlign = 'left';
      ctx.fillText(stroke.unitCode, this._canvasX(last.distM) + 3, this._canvasY(last.depthM) - 2);
    }
    ctx.globalAlpha = 1.0;
    ctx.setLineDash([]);
  }

  _updateInfo() {
    if (!this.infoEl) return;
    if (!this.strokes.length) {
      this.infoEl.textContent = 'No strokes yet. Select a unit and draw boundary lines.';
      return;
    }
    const counts = {};
    for (const s of this.strokes) counts[s.unitCode] = (counts[s.unitCode] ?? 0) + 1;
    this.infoEl.textContent = Object.entries(counts)
      .map(([c, n]) => `${c}×${n}`).join('  ') + `  (${this.strokes.length} stroke${this.strokes.length>1?'s':''})`;
  }

  // ── Public API ────────────────────────────────────────────────────────────

  undoLast() {
    this.strokes.pop();
    this._render();
    this._updateInfo();
  }

  clearAll() {
    this.strokes = [];
    this.current = null;
    this._render();
    this._updateInfo();
  }

  hasStrokes() { return this.strokes.length > 0; }

  toVirtualBoreholes(groundLevel = 0, weight = 0.92) {
    if (!this.fence || !this.strokes.length) return [];
    return sketchToVirtualBoreholes(this.strokes, this.fence, this.geoUnits, groundLevel, weight);
  }

  destroy() {
    this.canvas.removeEventListener('mousedown', this._bound);
    this.canvas.removeEventListener('mousemove', this._bmove);
    this.canvas.removeEventListener('mouseup',   this._bup);
  }
}
