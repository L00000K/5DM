// ── Borehole Log Strip Viewer ─────────────────────────────────────────────────
// Renders vertical graphical log strips (one per borehole) in an SVG panel.
// Displays: geological unit layers (colour-filled), SPT N-value bar chart,
// certainty annotations, ground level datum line.

export class BHLogView {
  constructor() {
    this._svg       = document.getElementById('bh-log-svg');
    this._exportBtn = document.getElementById('btn-bh-log-export');
    this._scaleIn   = document.getElementById('bh-log-scale');
    this._scaleLbl  = document.getElementById('bh-log-scale-lbl');
    this._hint      = document.getElementById('bh-log-hint');
    this._lastArgs  = null;

    this._exportBtn?.addEventListener('click', () => this._exportSVG());
    this._scaleIn?.addEventListener('input', () => {
      if (this._scaleLbl) this._scaleLbl.textContent = this._scaleIn.value;
      if (this._lastArgs) this._render(...this._lastArgs);
    });
  }

  draw(boreholes, geoUnits) {
    if (!boreholes?.length) return;
    this._lastArgs = [boreholes, geoUnits];
    if (this._hint) this._hint.hidden = true;
    if (this._exportBtn) this._exportBtn.disabled = false;
    this._render(boreholes, geoUnits);
  }

  // ── Render ────────────────────────────────────────────────────────────────
  _render(boreholes, geoUnits) {
    const svg = this._svg;
    if (!svg) return;

    // Scale: pixels per metre depth (from slider, default 20)
    const pxPerM  = parseFloat(this._scaleIn?.value ?? 20) || 20;
    const colW    = 90;   // unit colour column width px
    const sptW    = 60;   // SPT chart width px
    const lhsW    = 60;   // left-hand side axis width
    const colGap  = 28;
    const padTop  = 50;

    const unitByCode = {};
    geoUnits.forEach(u => { unitByCode[u.code] = u; });

    // Maximum depth across all boreholes
    const maxDepth = Math.max(...boreholes.map(bh =>
      bh.depth ?? (bh.layers.length ? Math.max(...bh.layers.map(l => l.base)) : 10)
    ));

    const svgH = padTop + maxDepth * pxPerM + 40;
    const svgW = lhsW + boreholes.length * (colW + sptW + colGap) + 20;

    // ── SVG namespace helper ──────────────────────────────────────────────
    const NS = 'http://www.w3.org/2000/svg';
    const mk = (tag, attrs) => {
      const el = document.createElementNS(NS, tag);
      Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, v));
      return el;
    };
    const txt = (x, y, content, attrs = {}) => {
      const el = mk('text', { x, y, ...attrs });
      el.textContent = content;
      return el;
    };

    // Clear and resize
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    svg.setAttribute('width', svgW);
    svg.setAttribute('height', svgH);
    svg.setAttribute('viewBox', `0 0 ${svgW} ${svgH}`);

    // Background
    svg.appendChild(mk('rect', { x: 0, y: 0, width: svgW, height: svgH, fill: '#1a2230' }));

    // Depth axis (left side)
    const axisX = lhsW - 8;
    for (let d = 0; d <= maxDepth; d += (maxDepth > 30 ? 5 : 1)) {
      const y = padTop + d * pxPerM;
      svg.appendChild(mk('line', { x1: lhsW - 5, y1: y, x2: svgW - 10, y2: y,
        stroke: '#2e3a4a', 'stroke-width': d % 5 === 0 ? 1 : 0.3 }));
      if (d % (maxDepth > 30 ? 5 : 2) === 0) {
        svg.appendChild(txt(axisX, y + 4, `${d}m`, {
          fill: '#8a9bb0', 'font-size': 9, 'text-anchor': 'end',
          'font-family': 'monospace',
        }));
      }
    }
    svg.appendChild(txt(8, padTop - 12, 'Depth (m)', {
      fill: '#8a9bb0', 'font-size': 10, 'font-family': 'sans-serif',
    }));

    // SPT N max for scale
    const maxN = Math.max(60, ...boreholes.flatMap(bh =>
      (bh.layers ?? []).map(l => l.sptN ?? 0)
    ));

    // ── Per-borehole columns ──────────────────────────────────────────────
    boreholes.forEach((bh, bi) => {
      const cx = lhsW + bi * (colW + sptW + colGap);

      // BH header
      svg.appendChild(txt(cx + colW * 0.5, padTop - 28, bh.id ?? `BH-${bi + 1}`, {
        fill: '#e0e8f0', 'font-size': 10, 'font-weight': 'bold',
        'text-anchor': 'middle', 'font-family': 'sans-serif',
      }));
      const glLabel = bh.groundLevel != null ? `GL=${bh.groundLevel.toFixed(1)}mAOD` : '';
      svg.appendChild(txt(cx + colW * 0.5, padTop - 16, glLabel, {
        fill: '#6a7d90', 'font-size': 8, 'text-anchor': 'middle',
        'font-family': 'sans-serif',
      }));

      // Ground level line
      svg.appendChild(mk('line', {
        x1: cx, y1: padTop, x2: cx + colW, y2: padTop,
        stroke: '#4fba6f', 'stroke-width': 2,
      }));

      // Layer rectangles
      const layers = bh.layers ?? [];
      layers.forEach(l => {
        const top  = l.top  ?? 0;
        const base = l.base ?? top + 1;
        const y1   = padTop + top  * pxPerM;
        const y2   = padTop + base * pxPerM;
        const h    = Math.max(1, y2 - y1);

        const unit = unitByCode[l.unitCode];
        const fill = unit?.color ?? '#555';
        const cert = l.certainty ?? 0.8;
        const opacity = 0.35 + cert * 0.65;

        svg.appendChild(mk('rect', {
          x: cx, y: y1, width: colW, height: h,
          fill, opacity, stroke: '#283040', 'stroke-width': 0.5,
        }));

        // Unit code label if tall enough
        if (h >= 12) {
          svg.appendChild(txt(cx + colW * 0.5, y1 + Math.min(h * 0.5, 10) + 3,
            l.unitCode ?? '?', {
              fill: '#fff', 'font-size': 9, 'font-weight': 'bold',
              'text-anchor': 'middle', 'font-family': 'monospace',
              'pointer-events': 'none',
            }));
        }
      });

      // Column outline
      const bhDepth = bh.depth ?? (layers.length ? Math.max(...layers.map(l => l.base)) : 10);
      svg.appendChild(mk('rect', {
        x: cx, y: padTop, width: colW, height: bhDepth * pxPerM,
        fill: 'none', stroke: '#3a4d62', 'stroke-width': 1,
      }));

      // ── SPT N-value bar chart ─────────────────────────────────────────
      const sptX = cx + colW + 2;
      svg.appendChild(txt(sptX + sptW * 0.5, padTop - 28, 'SPT N', {
        fill: '#8a9bb0', 'font-size': 8, 'text-anchor': 'middle',
        'font-family': 'sans-serif',
      }));
      svg.appendChild(mk('line', {
        x1: sptX, y1: padTop, x2: sptX, y2: padTop + bhDepth * pxPerM,
        stroke: '#2e3a4a', 'stroke-width': 1,
      }));

      layers.forEach(l => {
        if (l.sptN == null) return;
        const top = l.top ?? 0;
        const mid = ((l.top ?? 0) + (l.base ?? l.top + 1)) / 2;
        const y   = padTop + mid * pxPerM;
        const bw  = Math.min(sptW - 4, (l.sptN / maxN) * (sptW - 4));
        svg.appendChild(mk('rect', {
          x: sptX + 1, y: y - 4, width: Math.max(1, bw), height: 7,
          fill: '#5ab8e0', opacity: 0.8,
        }));
        svg.appendChild(txt(sptX + Math.max(1, bw) + 3, y + 2.5, `${l.sptN}`, {
          fill: '#5ab8e0', 'font-size': 7.5, 'font-family': 'monospace',
        }));
      });
    });

    // ── Unit legend strip at bottom ───────────────────────────────────────
    const legY = svgH - 28;
    svg.appendChild(txt(lhsW, legY - 2, 'Units:', {
      fill: '#8a9bb0', 'font-size': 9, 'font-family': 'sans-serif',
    }));
    let lx = lhsW + 36;
    geoUnits.forEach(u => {
      svg.appendChild(mk('rect', { x: lx, y: legY - 8, width: 12, height: 10, fill: u.color ?? '#888' }));
      svg.appendChild(txt(lx + 15, legY, u.code, {
        fill: '#c8d5e0', 'font-size': 8.5, 'font-family': 'monospace',
      }));
      lx += 14 + (u.code?.length ?? 2) * 6 + 8;
    });
  }

  // ── Export SVG ────────────────────────────────────────────────────────────
  _exportSVG() {
    const svg = this._svg;
    if (!svg) return;
    const src  = '<?xml version="1.0" encoding="UTF-8"?>\n' +
                 new XMLSerializer().serializeToString(svg);
    const blob = new Blob([src], { type: 'image/svg+xml' });
    const url  = URL.createObjectURL(blob);
    const a    = Object.assign(document.createElement('a'), { href: url, download: 'bh-log-strips.svg' });
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 5000);
  }
}
