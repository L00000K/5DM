// ── Model Quality & Coverage Report ───────────────────────────────────────────

export class ModelReport {
  constructor() {
    this._canvas  = document.getElementById('coverage-canvas');
    this._ctx     = this._canvas?.getContext('2d');
    this._statsEl = document.getElementById('quality-stats');
  }

  compute(grid, classifiedBH, geoUnits) {
    if (!grid) return null;
    const { nx, ny, cellSize: cs, origin: O, unitIds, certainty } = grid;
    const bhs = classifiedBH.filter(b => !b.synthetic);

    // Per-column nearest BH distance
    const searchR     = cs * 3;
    const covered     = new Uint8Array(nx * ny);
    const nearestDist = new Float32Array(nx * ny).fill(Infinity);

    for (let iy = 0; iy < ny; iy++) {
      for (let ix = 0; ix < nx; ix++) {
        const wx = O.x + (ix + 0.5) * cs;
        const wz = O.z + (iy + 0.5) * cs;
        let minD = Infinity;
        for (const bh of bhs) {
          const d = Math.hypot(bh.x - wx, bh.y - wz);
          if (d < minD) minD = d;
        }
        nearestDist[ix + iy * nx] = minD;
        covered[ix + iy * nx]     = minD <= searchR ? 1 : 0;
      }
    }

    const coveredCols = covered.reduce((s, v) => s + v, 0);
    const coveragePct = (coveredCols / (nx * ny) * 100).toFixed(0);

    // Mean certainty of filled voxels
    let certSum = 0, certCnt = 0;
    for (let i = 0; i < unitIds.length; i++) {
      if (unitIds[i]) { certSum += certainty[i]; certCnt++; }
    }
    const meanCert = certCnt > 0 ? (certSum / certCnt * 100).toFixed(0) : 0;

    // Suggest new BH in largest data gaps
    const suggested = this._gapCentroids(nx, ny, covered, O, cs, 4);

    this._renderMap(nx, ny, covered, bhs, O, cs, suggested);
    this._renderStats({ coveragePct, meanCert, bhCount: bhs.length, suggested });

    return { coveragePct, meanCert, suggested };
  }

  _gapCentroids(nx, ny, covered, O, cs, max) {
    const visited  = new Uint8Array(nx * ny);
    const clusters = [];

    for (let start = 0; start < nx * ny; start++) {
      if (covered[start] || visited[start]) continue;
      const cluster = [];
      const queue   = [start];
      visited[start] = 1;
      while (queue.length) {
        const cur = queue.shift();
        cluster.push(cur);
        const cx = cur % nx, cy = (cur / nx) | 0;
        for (const [dx, dy] of [[1,0],[-1,0],[0,1],[0,-1]]) {
          const nx2 = cx + dx, ny2 = cy + dy;
          if (nx2 < 0 || nx2 >= nx || ny2 < 0 || ny2 >= ny) continue;
          const ni = nx2 + ny2 * nx;
          if (!covered[ni] && !visited[ni]) { visited[ni] = 1; queue.push(ni); }
        }
      }
      if (cluster.length >= 4) clusters.push(cluster);
    }

    clusters.sort((a, b) => b.length - a.length);
    return clusters.slice(0, max).map((cl, i) => {
      const sumX = cl.reduce((s, idx) => s + (idx % nx),       0);
      const sumY = cl.reduce((s, idx) => s + ((idx / nx) | 0), 0);
      return {
        x:     O.x + (sumX / cl.length + 0.5) * cs,
        y:     O.z + (sumY / cl.length + 0.5) * cs,
        size:  cl.length,
        label: `P${i + 1}`,
      };
    });
  }

  _renderMap(nx, ny, covered, bhs, O, cs, suggested) {
    if (!this._canvas || !this._ctx) return;
    const W = this._canvas.parentElement?.clientWidth ?? 220;
    const H = Math.max(60, Math.round(W * ny / nx)) + 20;
    this._canvas.width  = W;
    this._canvas.height = H;
    const ctx = this._ctx;
    ctx.clearRect(0, 0, W, H);

    const PAD = 4;
    const dW  = W - PAD * 2;
    const dH  = H - PAD - 18;
    const pW  = dW / nx;
    const pH  = dH / ny;

    for (let iy = 0; iy < ny; iy++) {
      for (let ix = 0; ix < nx; ix++) {
        ctx.fillStyle = covered[ix + iy * nx]
          ? 'rgba(74,124,89,0.35)' : 'rgba(200,60,40,0.22)';
        ctx.fillRect(PAD + ix * pW, PAD + (ny - 1 - iy) * pH,
          Math.ceil(pW + 0.5), Math.ceil(pH + 0.5));
      }
    }

    ctx.strokeStyle = '#c8cdd6'; ctx.lineWidth = 1;
    ctx.strokeRect(PAD, PAD, dW, dH);

    ctx.fillStyle = '#1c2a38';
    for (const bh of bhs) {
      const px = PAD + (bh.x - O.x) / cs * pW;
      const py = PAD + (ny - (bh.y - O.z) / cs - 1) * pH;
      ctx.beginPath();
      ctx.arc(px, py, Math.max(2, pW * 0.4), 0, Math.PI * 2);
      ctx.fill();
    }

    for (const s of suggested) {
      const px = PAD + (s.x - O.x) / cs * pW;
      const py = PAD + (ny - (s.y - O.z) / cs - 1) * pH;
      ctx.strokeStyle = '#f07030'; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(px, py, 4.5, 0, Math.PI * 2); ctx.stroke();
      ctx.fillStyle = '#f07030';
      ctx.font = '8px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(s.label, px, py - 7);
    }

    ctx.fillStyle = '#8898a8';
    ctx.font = '9px Inter, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('■ Covered  ■ Gap  ● BH', PAD, H - 4);
  }

  _renderStats({ coveragePct, meanCert, bhCount, suggested }) {
    if (!this._statsEl) return;
    const cColor = +coveragePct > 75 ? '#4A7C59' : +coveragePct > 50 ? '#c8a855' : '#d04040';
    let html = `
      <div class="quality-row">
        <span class="quality-lbl">Coverage</span>
        <span class="quality-val" style="color:${cColor}">${coveragePct}%</span>
      </div>
      <div class="quality-row">
        <span class="quality-lbl">Mean certainty</span>
        <span class="quality-val">${meanCert}%</span>
      </div>
      <div class="quality-row">
        <span class="quality-lbl">Boreholes</span>
        <span class="quality-val">${bhCount}</span>
      </div>`;
    if (suggested.length) {
      html += `<div class="quality-row quality-gap">
        <span class="quality-lbl">Suggested new BH</span>
        <span class="quality-val quality-orange">${suggested.length}</span>
      </div>`;
      for (const s of suggested) {
        html += `<div class="quality-row quality-suggest">
          <span class="quality-lbl">${s.label} (${s.size} cells)</span>
          <span class="quality-val">(${s.x.toFixed(0)}, ${s.y.toFixed(0)})</span>
        </div>`;
      }
    } else {
      html += `<div class="quality-row" style="margin-top:4px">
        <span class="quality-lbl" style="color:#4A7C59">Good coverage ✓</span>
      </div>`;
    }
    this._statsEl.innerHTML = html;
  }

  exportHTML(grid, classifiedBH, geoUnits) {
    if (!grid) return;
    const { nx, ny, nz, cellSize: cs, cellHeight: ch, origin: O, unitIds, certainty } = grid;
    const bhs = classifiedBH.filter(b => !b.synthetic);

    const counts = {}, certSums = {};
    geoUnits.forEach(u => { counts[u.id] = 0; certSums[u.id] = 0; });
    for (let i = 0; i < unitIds.length; i++) {
      const uid = unitIds[i];
      if (uid && counts[uid] !== undefined) { counts[uid]++; certSums[uid] += certainty[i]; }
    }
    const total    = Object.values(counts).reduce((a, b) => a + b, 0);
    const cellVol  = cs * cs * ch;
    const dateStr  = new Date().toLocaleDateString('en-GB', { year:'numeric', month:'long', day:'numeric' });

    const unitRows = geoUnits.map(u => {
      const n    = counts[u.id] ?? 0;
      const vol  = Math.round(n * cellVol).toLocaleString();
      const pct  = total > 0 ? (n / total * 100).toFixed(1) : '0';
      const cert = n > 0 ? ((certSums[u.id] / n) * 100).toFixed(0) + '%' : '—';
      return `<tr>
        <td><span class="swatch" style="background:${u.color}"></span>${u.code}</td>
        <td>${u.name}</td>
        <td>${u.description ?? '—'}</td>
        <td>${vol}</td>
        <td>${pct}%</td>
        <td>${cert}</td>
      </tr>`;
    }).join('');

    const bhRows = bhs.map(bh => {
      const maxBase = bh.layers.length ? Math.max(...bh.layers.map(l => l.base)) : (bh.depth ?? 0);
      const units   = [...new Set(bh.layers.map(l => l.unitCode).filter(Boolean))].join(', ');
      return `<tr>
        <td><strong>${bh.id}</strong></td>
        <td>${bh.x?.toFixed(1) ?? '—'}</td>
        <td>${bh.y?.toFixed(1) ?? '—'}</td>
        <td>${bh.groundLevel?.toFixed(1) ?? '—'}</td>
        <td>${maxBase.toFixed(1)}</td>
        <td>${bh.layers.length}</td>
        <td>${units}</td>
      </tr>`;
    }).join('');

    const html = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>GeoModel AI — Site Investigation Report</title>
<style>
  *{box-sizing:border-box}
  body{font-family:Arial,Helvetica,sans-serif;color:#1c2a38;padding:40px;max-width:960px;margin:0 auto;font-size:13px}
  h1{font-size:22px;color:#1c2a38;border-bottom:3px solid #4a6275;padding-bottom:10px;margin-bottom:4px}
  .meta{color:#8898a8;font-size:11px;margin-bottom:28px}
  h2{font-size:13px;text-transform:uppercase;letter-spacing:0.8px;color:#4a6275;margin:28px 0 8px;border-bottom:1px solid #e4e8ec;padding-bottom:4px}
  table{width:100%;border-collapse:collapse;margin-bottom:16px}
  th{background:#f0f2f5;padding:6px 10px;text-align:left;border-bottom:2px solid #c8cdd6;font-size:11px;text-transform:uppercase;letter-spacing:0.4px}
  td{padding:5px 10px;border-bottom:1px solid #eaeef2;vertical-align:middle}
  tr:last-child td{border-bottom:none}
  .swatch{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:middle}
  .badge{display:inline-block;background:#d04040;color:#fff;font-size:10px;padding:2px 7px;border-radius:3px;margin-left:10px;vertical-align:middle}
  .footer{margin-top:48px;padding-top:12px;border-top:1px solid #e4e8ec;color:#8898a8;font-size:10px}
  .info-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:8px}
  .info-cell{background:#f8f9fa;border-radius:4px;padding:10px 14px}
  .info-label{font-size:10px;text-transform:uppercase;letter-spacing:0.5px;color:#8898a8}
  .info-value{font-size:18px;font-weight:700;color:#1c2a38;margin-top:2px}
</style>
</head>
<body>
<h1>Site Investigation Report <span class="badge">PROOF OF CONCEPT</span></h1>
<p class="meta">Generated by GeoModel AI · ${dateStr}</p>

<div class="info-grid">
  <div class="info-cell"><div class="info-label">Total Voxels</div><div class="info-value">${(nx*ny*nz).toLocaleString()}</div></div>
  <div class="info-cell"><div class="info-label">Boreholes</div><div class="info-value">${bhs.length}</div></div>
  <div class="info-cell"><div class="info-label">Grid</div><div class="info-value">${nx}×${ny}×${nz}</div></div>
  <div class="info-cell"><div class="info-label">Cell Size</div><div class="info-value">${cs}×${cs}×${ch.toFixed(2)} m</div></div>
</div>

<h2>Geological Units</h2>
<table>
  <tr><th>Code</th><th>Name</th><th>Description</th><th>Volume (m³)</th><th>Model %</th><th>Mean Certainty</th></tr>
  ${unitRows}
</table>

<h2>Borehole Register</h2>
<table>
  <tr><th>BH ID</th><th>X (m)</th><th>Y (m)</th><th>GL (mAOD)</th><th>Depth (m)</th><th>Layers</th><th>Units</th></tr>
  ${bhRows}
</table>

<h2>Model Grid</h2>
<table>
  <tr><th>Parameter</th><th>Value</th></tr>
  <tr><td>Dimensions (nx × ny × nz)</td><td>${nx} × ${ny} × ${nz}</td></tr>
  <tr><td>Horizontal cell size</td><td>${cs} m</td></tr>
  <tr><td>Vertical cell height</td><td>${ch.toFixed(2)} m</td></tr>
  <tr><td>Grid origin (X, Y, Z)</td><td>(${O.x.toFixed(1)}, ${O.y.toFixed(1)}, ${O.z.toFixed(1)})</td></tr>
  <tr><td>Geological units</td><td>${geoUnits.length}</td></tr>
</table>

<p class="footer">This report is automatically generated from a probabilistic 3D interpolation model. It is intended for illustrative purposes only and must not be used for design, construction, or contractual purposes without independent verification by a qualified geotechnical engineer.</p>
</body>
</html>`;

    const blob = new Blob([html], { type: 'text/html' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href = url;
    a.download = `geomodel-report-${new Date().toISOString().slice(0, 10)}.html`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 5000);
  }
}
