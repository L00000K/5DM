// ── CPT Profile Viewer ─────────────────────────────────────────────────────────
// Renders qc (cone resistance) and fs (sleeve friction) profiles as SVG.
// Colour-codes each depth interval by Robertson SBT Ic zone.

// Robertson SBT zone colours (Ic ranges)
const SBT_ZONES = [
  { Ic: [0,    1.31], label: 'Dense sand/gravel', color: '#D4A843' },
  { Ic: [1.31, 2.05], label: 'Sand',              color: '#f0e060' },
  { Ic: [2.05, 2.60], label: 'Sand-silt mix',     color: '#a8c060' },
  { Ic: [2.60, 2.95], label: 'Silt-clay mix',     color: '#60a878' },
  { Ic: [2.95, 3.60], label: 'Clays',             color: '#4A7C9A' },
  { Ic: [3.60, 4.00], label: 'Organic/peat',      color: '#6a4a30' },
];

function sbtColor(Ic) {
  for (const z of SBT_ZONES) {
    if (Ic >= z.Ic[0] && Ic < z.Ic[1]) return z.color;
  }
  return '#666';
}

export class CPTLogView {
  constructor() {
    this._svg      = document.getElementById('cpt-log-svg');
    this._exportBtn= document.getElementById('btn-cpt-log-export');
    this._scaleIn  = document.getElementById('cpt-log-scale');
    this._scaleLbl = document.getElementById('cpt-log-scale-lbl');
    this._hint     = document.getElementById('cpt-log-hint');
    this._lastData = null;

    this._exportBtn?.addEventListener('click', () => this._exportSVG());
    this._scaleIn?.addEventListener('input', () => {
      if (this._scaleLbl) this._scaleLbl.textContent = this._scaleIn.value;
      if (this._lastData) this._render(this._lastData);
    });
  }

  draw(cptLogs) {
    if (!cptLogs?.length) return;
    this._lastData = cptLogs;
    if (this._hint) this._hint.hidden = true;
    if (this._exportBtn) this._exportBtn.disabled = false;
    this._render(cptLogs);
  }

  _render(logs) {
    const svg = this._svg;
    if (!svg) return;

    const pxPerM = parseFloat(this._scaleIn?.value ?? 20) || 20;
    const NS = 'http://www.w3.org/2000/svg';
    const mk = (tag, attrs) => {
      const el = document.createElementNS(NS, tag);
      Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, v));
      return el;
    };
    const txt = (x, y, s, attrs = {}) => {
      const el = mk('text', { x, y, ...attrs });
      el.textContent = s;
      return el;
    };

    const padTop = 55;
    const padLeft = 12;
    const colW_sbt = 16;  // SBT Ic colour strip
    const colW_qc  = 80;  // qc chart
    const colW_fs  = 50;  // fs chart
    const colGap   = 20;
    const colTotal = colW_sbt + colW_qc + colW_fs + colGap;
    const lhsW = 55;

    const maxDepth = Math.max(...logs.map(l => l.depths.length > 0 ? Math.max(...l.depths) : 10));
    const maxQc    = Math.max(5, ...logs.flatMap(l => l.qc));
    const maxFs    = Math.max(0.1, ...logs.flatMap(l => l.fs));

    const svgH = padTop + maxDepth * pxPerM + 60;
    const svgW = padLeft + lhsW + logs.length * colTotal + 20;

    while (svg.firstChild) svg.removeChild(svg.firstChild);
    svg.setAttribute('width', svgW);
    svg.setAttribute('height', svgH);
    svg.setAttribute('viewBox', `0 0 ${svgW} ${svgH}`);
    svg.appendChild(mk('rect', { x:0, y:0, width:svgW, height:svgH, fill:'#1a2230' }));

    // Depth axis
    for (let d = 0; d <= maxDepth; d += (maxDepth > 30 ? 5 : 1)) {
      const y = padTop + d * pxPerM;
      svg.appendChild(mk('line', { x1: lhsW, y1: y, x2: svgW - 10, y2: y,
        stroke: '#2e3a4a', 'stroke-width': d % 5 === 0 ? 1 : 0.3 }));
      if (d % (maxDepth > 30 ? 5 : 2) === 0) {
        svg.appendChild(txt(lhsW - 4, y + 4, `${d}m`, {
          fill: '#8a9bb0', 'font-size': 9, 'text-anchor': 'end', 'font-family': 'monospace',
        }));
      }
    }

    logs.forEach((log, li) => {
      const cx = padLeft + lhsW + li * colTotal;

      // Header
      svg.appendChild(txt(cx + colW_sbt + colW_qc * 0.5, padTop - 40, log.id ?? `CPT-${li+1}`, {
        fill: '#e0e8f0', 'font-size': 10, 'font-weight': 'bold',
        'text-anchor': 'middle', 'font-family': 'sans-serif',
      }));
      const glLabel = log.groundLevel != null ? `GL=${log.groundLevel.toFixed(1)}m` : '';
      svg.appendChild(txt(cx + colW_sbt + colW_qc * 0.5, padTop - 28, glLabel, {
        fill: '#6a7d90', 'font-size': 8, 'text-anchor': 'middle', 'font-family': 'sans-serif',
      }));

      // Column headers
      svg.appendChild(txt(cx + colW_sbt + colW_qc * 0.5, padTop - 14, `qc (MPa)`, {
        fill: '#8ab8d0', 'font-size': 8, 'text-anchor': 'middle', 'font-family': 'sans-serif',
      }));
      svg.appendChild(txt(cx + colW_sbt + colW_qc + colW_fs * 0.5, padTop - 14, `fs (MPa)`, {
        fill: '#90d8a0', 'font-size': 8, 'text-anchor': 'middle', 'font-family': 'sans-serif',
      }));

      // qc scale
      svg.appendChild(txt(cx + colW_sbt, padTop - 4, '0', {
        fill: '#556677', 'font-size': 7, 'text-anchor': 'start', 'font-family': 'monospace',
      }));
      svg.appendChild(txt(cx + colW_sbt + colW_qc, padTop - 4, maxQc.toFixed(0), {
        fill: '#556677', 'font-size': 7, 'text-anchor': 'end', 'font-family': 'monospace',
      }));

      // GL line
      svg.appendChild(mk('line', { x1: cx, y1: padTop, x2: cx + colTotal - colGap, y2: padTop,
        stroke: '#4fba6f', 'stroke-width': 1.5 }));

      // Per-depth-increment rendering
      const n = log.depths.length;
      for (let i = 0; i < n - 1; i++) {
        const d0 = log.depths[i], d1 = log.depths[i + 1];
        const y0 = padTop + d0 * pxPerM;
        const y1 = padTop + d1 * pxPerM;
        const h  = Math.max(0.5, y1 - y0);

        // SBT Ic colour strip
        const sbtC = sbtColor(log.Ic[i]);
        svg.appendChild(mk('rect', {
          x: cx, y: y0, width: colW_sbt, height: h,
          fill: sbtC, opacity: 0.85,
        }));

        // qc bar (horizontal, rightward)
        const qcW = Math.max(0.5, (log.qc[i] / maxQc) * colW_qc);
        svg.appendChild(mk('rect', {
          x: cx + colW_sbt, y: y0 + 0.5, width: qcW, height: Math.max(0.5, h - 0.5),
          fill: '#5ab8e0', opacity: 0.75,
        }));

        // fs bar
        const fsW = Math.max(0.5, (log.fs[i] / maxFs) * colW_fs);
        svg.appendChild(mk('rect', {
          x: cx + colW_sbt + colW_qc, y: y0 + 0.5, width: fsW, height: Math.max(0.5, h - 0.5),
          fill: '#70d890', opacity: 0.75,
        }));
      }

      // Column outlines
      svg.appendChild(mk('rect', {
        x: cx, y: padTop, width: colW_sbt, height: maxDepth * pxPerM,
        fill: 'none', stroke: '#3a4d62', 'stroke-width': 0.5,
      }));
      svg.appendChild(mk('rect', {
        x: cx + colW_sbt, y: padTop, width: colW_qc, height: maxDepth * pxPerM,
        fill: 'none', stroke: '#3a4d62', 'stroke-width': 0.5,
      }));
      svg.appendChild(mk('rect', {
        x: cx + colW_sbt + colW_qc, y: padTop, width: colW_fs, height: maxDepth * pxPerM,
        fill: 'none', stroke: '#3a4d62', 'stroke-width': 0.5,
      }));
    });

    // SBT legend
    const legY = svgH - 30;
    let lx = padLeft + lhsW;
    svg.appendChild(txt(lx - 8, legY + 8, 'SBT:', {
      fill: '#8a9bb0', 'font-size': 9, 'text-anchor': 'end', 'font-family': 'sans-serif',
    }));
    SBT_ZONES.forEach(z => {
      svg.appendChild(mk('rect', { x: lx, y: legY, width: 12, height: 10, fill: z.color }));
      svg.appendChild(txt(lx + 14, legY + 8, z.label, {
        fill: '#c0ccda', 'font-size': 8, 'font-family': 'sans-serif',
      }));
      lx += 14 + z.label.length * 5 + 8;
    });
  }

  _exportSVG() {
    const svg = this._svg;
    if (!svg) return;
    const src  = '<?xml version="1.0" encoding="UTF-8"?>\n' +
                 new XMLSerializer().serializeToString(svg);
    const blob = new Blob([src], { type: 'image/svg+xml' });
    const url  = URL.createObjectURL(blob);
    Object.assign(document.createElement('a'), { href: url, download: 'cpt-profiles.svg' }).click();
    setTimeout(() => URL.revokeObjectURL(url), 5000);
  }
}
