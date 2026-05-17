// ── Stratigraphic Correlation Panel ──────────────────────────────────────────
// Renders multiple BH logs side by side with horizontal correlation lines
// connecting matching geological units at the same elevation.

export class StratCorrelation {
  constructor() {
    this._panel   = document.getElementById('strat-corr-panel');
    this._canvas  = document.getElementById('strat-corr-canvas');
    this._close   = document.getElementById('strat-corr-close');
    this._export  = document.getElementById('strat-corr-export');
    this._ctx     = this._canvas?.getContext('2d');
    this._visible = false;
    this._lastArgs = null;

    this._close?.addEventListener('click', () => this.hide());
    this._export?.addEventListener('click', () => this._exportPNG());

    ['strat-corr-elev', 'strat-corr-lines'].forEach(id => {
      document.getElementById(id)?.addEventListener('change', () => {
        if (this._visible && this._lastArgs) this._redraw();
      });
    });

    window.addEventListener('keydown', e => {
      if (e.key === 'Escape' && this._visible) this.hide();
    });

    const ro = new ResizeObserver(() => { if (this._visible) this._redraw(); });
    if (this._panel) ro.observe(this._panel);
  }

  draw(boreholes, geoUnits) {
    if (!boreholes?.length || !geoUnits?.length) return;
    this._lastArgs = { boreholes, geoUnits };
    this.show();
    this._redraw();
  }

  show()  { this._visible = true;  if (this._panel) this._panel.hidden = false; }
  hide()  { this._visible = false; if (this._panel) this._panel.hidden = true; }
  get visible() { return this._visible; }

  _redraw() {
    if (!this._canvas || !this._ctx || !this._lastArgs) return;
    const { boreholes, geoUnits } = this._lastArgs;
    const useElev  = document.getElementById('strat-corr-elev')?.checked ?? true;
    const showLines = document.getElementById('strat-corr-lines')?.checked ?? true;

    // Filter to real BHs with layers
    const bhs = boreholes.filter(b => !b.synthetic && b.layers?.length);
    if (!bhs.length) return;

    // Sort boreholes W→E (by Easting)
    const sorted = [...bhs].sort((a, b) => a.x - b.x);

    const unitByCode = {};
    geoUnits.forEach(u => { unitByCode[u.code] = u; });

    const PAD_L = 10, PAD_R = 20, PAD_T = 40, PAD_B = 30;
    const BH_W  = 24;   // borehole log width in px
    const BH_GAP = 80;  // gap between boreholes

    const totalW = PAD_L + PAD_R + sorted.length * (BH_W + BH_GAP);
    const containerH = this._canvas.parentElement?.clientHeight ?? 420;
    const H  = containerH;

    this._canvas.width  = Math.max(totalW, 300);
    this._canvas.height = H;
    const ctx = this._ctx;
    ctx.clearRect(0, 0, this._canvas.width, H);
    ctx.fillStyle = '#f5f7fa';
    ctx.fillRect(0, 0, this._canvas.width, H);

    // Elevation range across all BHs
    let globalTop = -Infinity, globalBot = Infinity;
    sorted.forEach(bh => {
      const gl = bh.groundLevel ?? 0;
      globalTop = Math.max(globalTop, gl);
      sorted.forEach(bh2 => {
        bh2.layers.forEach(l => {
          const base = gl - (l.base ?? 0);
          globalBot = Math.min(globalBot, base);
        });
      });
    });
    // Also check non-sorted for min base
    sorted.forEach(bh => {
      const gl = bh.groundLevel ?? 0;
      bh.layers.forEach(l => {
        globalBot = Math.min(globalBot, gl - (l.base ?? 0));
      });
    });
    if (globalTop === -Infinity) globalTop = 0;
    if (globalBot === Infinity)  globalBot = globalTop - 20;
    const worldH = Math.max(1, globalTop - globalBot);

    const drawH = H - PAD_T - PAD_B;
    const toY = elev => PAD_T + drawH - ((elev - globalBot) / worldH) * drawH;

    // Elevation grid
    const tickStep = worldH <= 10 ? 1 : worldH <= 50 ? 5 : worldH <= 200 ? 10 : 25;
    ctx.strokeStyle = '#dde3ea'; ctx.lineWidth = 0.5;
    ctx.fillStyle = '#6a7a8a'; ctx.font = '9px monospace'; ctx.textAlign = 'right';
    for (let ev = Math.ceil(globalBot / tickStep) * tickStep; ev <= globalTop + 0.01; ev += tickStep) {
      const yPx = toY(ev);
      ctx.beginPath(); ctx.moveTo(PAD_L + 5, yPx); ctx.lineTo(this._canvas.width - PAD_R, yPx); ctx.stroke();
      ctx.fillText(`${ev.toFixed(0)}`, PAD_L + 35, yPx + 3);
    }

    // Per-BH log rendering
    const bhX = (i) => PAD_L + 40 + i * (BH_W + BH_GAP);

    // Build per-unit contact elevation maps for correlation lines
    const unitContacts = new Map(); // unitCode → [{bhIdx, topElev, botElev}]

    sorted.forEach((bh, i) => {
      const gl  = bh.groundLevel ?? 0;
      const cx  = bhX(i);

      // BH stick (grey background)
      ctx.fillStyle = '#e0e4ea';
      ctx.fillRect(cx, toY(gl), BH_W, toY(globalBot) - toY(gl));

      // Layers
      bh.layers.forEach(l => {
        const unit  = unitByCode[l.unitCode];
        if (!unit) return;
        const topZ  = gl - (l.top ?? 0);
        const botZ  = gl - (l.base ?? 0);
        const y1 = toY(topZ), y2 = toY(botZ);
        const h  = Math.max(1, y2 - y1);

        ctx.fillStyle = unit.color ?? '#aaa';
        ctx.globalAlpha = 0.88;
        ctx.fillRect(cx, y1, BH_W, h);
        ctx.globalAlpha = 1;

        // Unit code label if tall enough
        if (h > 10) {
          ctx.fillStyle = '#fff';
          ctx.font = `bold ${Math.min(9, h - 2)}px Inter, sans-serif`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(unit.code.slice(0, 3), cx + BH_W / 2, y1 + h / 2);
        }

        // Accumulate contacts
        if (!unitContacts.has(unit.code)) unitContacts.set(unit.code, []);
        unitContacts.get(unit.code).push({ bhIdx: i, topElev: topZ, botElev: botZ, bh });
      });

      // BH ID label
      ctx.fillStyle = '#334455';
      ctx.font = 'bold 9px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';
      ctx.fillText(bh.id.slice(0, 8), cx + BH_W / 2, PAD_T - 4);

      // Ground level tick
      const glY = toY(gl);
      ctx.strokeStyle = '#334455'; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(cx - 3, glY); ctx.lineTo(cx + BH_W + 3, glY); ctx.stroke();

      // GL label
      ctx.fillStyle = '#334455'; ctx.font = '8px monospace'; ctx.textAlign = 'left';
      ctx.textBaseline = 'bottom';
      ctx.fillText(`${gl.toFixed(1)}m`, cx + BH_W + 2, glY - 1);

      // Outline
      ctx.strokeStyle = 'rgba(40,60,80,0.2)'; ctx.lineWidth = 0.7;
      ctx.strokeRect(cx, toY(gl), BH_W, toY(globalBot) - toY(gl));
    });

    // ── Correlation lines ───────────────────────────────────────────────────
    if (showLines && sorted.length > 1) {
      geoUnits.forEach(unit => {
        const contacts = unitContacts.get(unit.code);
        if (!contacts || contacts.length < 2) return;

        // Sort by BH index and draw spline through top contacts
        contacts.sort((a, b) => a.bhIdx - b.bhIdx);

        // Top contact correlation line
        const pts = contacts.map(c => ({
          x: bhX(c.bhIdx) + BH_W / 2,
          y: toY(c.topElev),
        }));

        if (pts.length >= 2) {
          const col = unit.color ?? '#4472c4';
          ctx.strokeStyle = col;
          ctx.lineWidth = 1.2;
          ctx.setLineDash([4, 3]);
          ctx.globalAlpha = 0.65;
          ctx.beginPath();
          ctx.moveTo(pts[0].x, pts[0].y);
          for (let k = 1; k < pts.length; k++) {
            // Catmull-Rom-like smoothed line through control points
            const p0 = pts[Math.max(0, k - 1)];
            const p1 = pts[k];
            const cp1x = p0.x + (p1.x - p0.x) / 3;
            const cp1y = p0.y;
            const cp2x = p1.x - (p1.x - p0.x) / 3;
            const cp2y = p1.y;
            ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, p1.x, p1.y);
          }
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.globalAlpha = 1;
        }
      });
    }

    // ── Legend ────────────────────────────────────────────────────────────
    const legendY = H - PAD_B + 5;
    let lx = PAD_L + 40;
    ctx.textBaseline = 'top';
    ctx.font = '8px Inter, sans-serif';
    geoUnits.forEach(u => {
      if (lx > this._canvas.width - 60) return;
      ctx.fillStyle = u.color ?? '#888';
      ctx.fillRect(lx, legendY, 10, 10);
      ctx.fillStyle = '#334455';
      ctx.fillText(u.code, lx + 13, legendY + 1);
      lx += 14 + ctx.measureText(u.code).width + 8;
    });
  }

  _exportPNG() {
    if (!this._canvas) return;
    const url = this._canvas.toDataURL('image/png');
    const a   = document.createElement('a');
    a.href = url; a.download = 'strat-correlation.png'; a.click();
  }
}
