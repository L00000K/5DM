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

  // ── Cross-validation: compare predicted voxel units vs observed BH layers ────
  validateModel(grid, classifiedBH, geoUnits) {
    if (!grid) return null;
    const { nx, ny, nz, cellSize: cs, cellHeight: ch, origin: O, unitIds } = grid;
    const bhs = classifiedBH.filter(b => !b.synthetic && b.layers?.length);

    const unitById = {}, unitByCode = {};
    geoUnits.forEach(u => { unitById[u.id] = u; unitByCode[u.code] = u; });

    let correct = 0, total = 0;
    const perUnit = {}; // { code: { correct, total } }
    const mismatches = [];

    for (const bh of bhs) {
      // Grid column for this BH
      const ix = Math.max(0, Math.min(nx - 1, Math.round((bh.x - O.x) / cs - 0.5)));
      const iy = Math.max(0, Math.min(ny - 1, Math.round((bh.y - O.z) / cs - 0.5)));

      for (const layer of bh.layers) {
        if (!layer.unitCode) continue;
        const midDepth = (layer.top + layer.base) / 2;
        const elev     = (bh.groundLevel ?? 0) - midDepth;
        const iz       = Math.max(0, Math.min(nz - 1,
          Math.round((elev - O.y) / ch - 0.5)));

        const predId   = unitIds[ix + iy * nx + iz * nx * ny];
        const predUnit = unitById[predId];
        const obsCode  = layer.unitCode;

        if (!perUnit[obsCode]) perUnit[obsCode] = { correct: 0, total: 0 };
        perUnit[obsCode].total++;
        total++;

        if (predUnit?.code === obsCode) {
          correct++;
          perUnit[obsCode].correct++;
        } else if (predUnit) {
          mismatches.push({
            bh: bh.id, depth: midDepth.toFixed(1),
            observed: obsCode, predicted: predUnit.code,
          });
        }
      }
    }

    const accuracy = total > 0 ? (correct / total * 100).toFixed(1) : null;
    this._renderValidation({ accuracy, total, correct, perUnit, mismatches, geoUnits });
    return { accuracy, total, correct, perUnit, mismatches };
  }

  _renderValidation({ accuracy, total, correct, perUnit, mismatches, geoUnits }) {
    const el = document.getElementById('validation-results');
    if (!el) return;

    if (accuracy == null) { el.innerHTML = '<p class="hint">No classified layers to validate.</p>'; return; }

    const color = +accuracy >= 80 ? 'var(--green)' : +accuracy >= 60 ? '#c8a855' : '#d04040';

    let html = `
      <div class="quality-row" style="margin-bottom:6px">
        <span class="quality-lbl">Overall accuracy</span>
        <span class="quality-val" style="color:${color};font-size:15px;font-weight:700">${accuracy}%</span>
      </div>
      <div class="quality-row">
        <span class="quality-lbl">Points checked</span>
        <span class="quality-val">${correct} / ${total}</span>
      </div>
      <div style="margin:8px 0 4px;font-size:11px;color:var(--text-mid);font-weight:600">Per unit accuracy:</div>`;

    for (const [code, stats] of Object.entries(perUnit)) {
      const unitObj = geoUnits.find(u => u.code === code);
      const acc = (stats.correct / stats.total * 100).toFixed(0);
      const c   = +acc >= 80 ? 'var(--green)' : +acc >= 50 ? '#c8a855' : '#d04040';
      html += `<div class="quality-row">
        <span class="quality-lbl">
          <span style="display:inline-block;width:8px;height:8px;border-radius:2px;background:${unitObj?.color ?? '#888'};margin-right:4px;vertical-align:middle"></span>
          ${code}
        </span>
        <span class="quality-val" style="color:${c}">${acc}% (${stats.correct}/${stats.total})</span>
      </div>`;
    }

    if (mismatches.length) {
      html += `<div style="margin:8px 0 4px;font-size:11px;color:var(--text-mid);font-weight:600">Mismatches (first 10):</div>`;
      mismatches.slice(0, 10).forEach(m => {
        html += `<div class="quality-row" style="font-size:10px">
          <span style="color:var(--text-dim)">${m.bh} @${m.depth}m</span>
          <span><span style="color:#d04040">${m.observed}</span> → <span style="color:var(--green)">${m.predicted}</span></span>
        </div>`;
      });
    } else if (total > 0) {
      html += `<div class="quality-row"><span style="color:var(--green);font-size:11px">No mismatches ✓</span></div>`;
    }

    el.innerHTML = html;
  }

  exportHTML(grid, classifiedBH, geoUnits, riskReport = null) {
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
    const unitByCode = {};
    geoUnits.forEach(u => { unitByCode[u.code] = u; });

    // ── Unit table rows ────────────────────────────────────────────────────────
    const unitRows = geoUnits.map(u => {
      const n    = counts[u.id] ?? 0;
      const vol  = Math.round(n * cellVol).toLocaleString();
      const pct  = total > 0 ? (n / total * 100).toFixed(1) : '0';
      const cert = n > 0 ? ((certSums[u.id] / n) * 100).toFixed(0) + '%' : '—';
      const p    = u.params ?? {};
      const cu   = p.cu    != null ? p.cu.toFixed(0)    : '—';
      const phi  = p.phi   != null ? p.phi.toFixed(0)   : '—';
      const Cc   = p.Cc    != null ? p.Cc.toFixed(3)    : '—';
      const E    = p.E     != null ? p.E.toFixed(0)     : '—';
      const gam  = p.gamma != null ? p.gamma.toFixed(1) : '—';
      const nspt = p.N_spt != null ? p.N_spt.toFixed(0) : '—';
      const period = u.period ? `<span style="background:${_periodColor(u.period)};padding:1px 5px;border-radius:3px;font-size:10px">${u.period}</span>` : '—';
      return `<tr>
        <td><span class="swatch" style="background:${u.color}"></span>${u.code}</td>
        <td>${u.name}</td><td>${period}</td>
        <td style="font-size:11px;color:#677">${u.description ?? '—'}</td>
        <td>${vol}</td><td>${pct}%</td><td>${cert}</td>
        <td>${gam}</td><td>${cu}</td><td>${phi}</td><td>${Cc}</td><td>${E}</td><td>${nspt}</td>
      </tr>`;
    }).join('');

    // ── Formation tops matrix ──────────────────────────────────────────────────
    const formationTopsHTML = (() => {
      const matrix = {};
      bhs.forEach(bh => {
        matrix[bh.id] = {};
        bh.layers.forEach(l => {
          if (!matrix[bh.id][l.unitCode]) {
            matrix[bh.id][l.unitCode] = l.top;
          }
        });
      });
      const usedUnits = geoUnits.filter(u => bhs.some(b => matrix[b.id]?.[u.code] !== undefined));
      if (!usedUnits.length) return '';
      let t = `<table class="data-table" style="font-size:10px"><thead><tr>
        <th>Unit</th>${bhs.map(b => `<th>${b.id}</th>`).join('')}
      </tr></thead><tbody>`;
      usedUnits.forEach(u => {
        t += `<tr><td><span class="swatch" style="background:${u.color}"></span>${u.code}</td>
          ${bhs.map(b => {
            const v = matrix[b.id]?.[u.code];
            return `<td style="text-align:right;color:${v !== undefined ? '#223' : '#bbb'}">${v !== undefined ? v.toFixed(1)+'m' : '—'}</td>`;
          }).join('')}</tr>`;
      });
      t += '</tbody></table>';
      return `<h2>Formation Top Depths (m below GL)</h2>${t}`;
    })();

    // ── Borehole register rows ─────────────────────────────────────────────────
    const bhRows = bhs.map(bh => {
      const maxBase  = bh.layers.length ? Math.max(...bh.layers.map(l => l.base)) : (bh.depth ?? 0);
      const units    = [...new Set(bh.layers.map(l => l.unitCode).filter(Boolean))].join(', ');
      const meanSPT  = (() => {
        const vals = bh.layers.filter(l => l.sptN != null).map(l => l.sptN);
        return vals.length ? (vals.reduce((a,b)=>a+b,0)/vals.length).toFixed(0) : '—';
      })();
      return `<tr>
        <td><strong>${bh.id}</strong></td>
        <td>${bh.x?.toFixed(1) ?? '—'}</td><td>${bh.y?.toFixed(1) ?? '—'}</td>
        <td>${bh.groundLevel?.toFixed(1) ?? '—'}</td>
        <td>${maxBase.toFixed(1)}</td><td>${bh.layers.length}</td>
        <td style="font-size:11px">${units}</td><td>${meanSPT}</td>
      </tr>`;
    }).join('');

    // ── Site plan SVG ──────────────────────────────────────────────────────────
    const planSVG = (() => {
      if (!bhs.length) return '<p style="color:#999">No borehole data</p>';
      const margin = 24;
      const svgW = 520, svgH = 320;
      const xs = bhs.map(b => b.x), ys = bhs.map(b => b.y);
      const minX = Math.min(...xs), maxX = Math.max(...xs);
      const minY = Math.min(...ys), maxY = Math.max(...ys);
      const rangeX = Math.max(maxX - minX, 1), rangeY = Math.max(maxY - minY, 1);
      const scX = (svgW - margin*2) / rangeX, scY = (svgH - margin*2) / rangeY;
      const sc = Math.min(scX, scY);
      const offX = margin + ((svgW - margin*2) - rangeX * sc) / 2;
      const offY = margin + ((svgH - margin*2) - rangeY * sc) / 2;
      const px = x => (offX + (x - minX) * sc).toFixed(1);
      const py = y => (offY + (maxY - y) * sc).toFixed(1); // flip Y

      const dots = bhs.map(bh => {
        const glStr = bh.groundLevel != null ? ` GL:${bh.groundLevel.toFixed(1)}mAOD` : '';
        const unitList = [...new Set(bh.layers.map(l=>l.unitCode).filter(Boolean))];
        const topUnit  = unitByCode[unitList[0]];
        const fill = topUnit?.color ?? '#6688aa';
        return `<circle cx="${px(bh.x)}" cy="${py(bh.y)}" r="5" fill="${fill}" stroke="#fff" stroke-width="1.5"/>
<text x="${(parseFloat(px(bh.x))+8).toFixed(1)}" y="${(parseFloat(py(bh.y))+4).toFixed(1)}" font-size="9" fill="#334">${bh.id}${glStr}</text>`;
      }).join('\n');

      // Grid extent rectangle
      const ox = (offX + (O.x - minX) * sc).toFixed(1);
      const oy = (offY + (maxY - (O.z + ny*cs)) * sc).toFixed(1);
      const gw = (nx * cs * sc).toFixed(1);
      const gh = (ny * cs * sc).toFixed(1);
      const gridRect = `<rect x="${ox}" y="${oy}" width="${gw}" height="${gh}" fill="none" stroke="#4a6275" stroke-width="1" stroke-dasharray="4,3" opacity="0.5"/>`;

      return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${svgW} ${svgH}" style="width:100%;max-width:${svgW}px;border:1px solid #e0e4ea;border-radius:4px;background:#f8f9fa">
  <rect width="${svgW}" height="${svgH}" fill="#f8f9fa"/>
  ${gridRect}
  ${dots}
  <text x="8" y="${svgH-6}" font-size="9" fill="#999">Model grid boundary (dashed)  ·  Site Plan — Not to scale</text>
</svg>`;
    })();

    // ── Borehole log strips SVG ────────────────────────────────────────────────
    const logStripsSVG = (() => {
      if (!bhs.length) return '';
      const SCALE   = 14;    // px per metre
      const COL_W   = 22;    // unit colour column width
      const SPT_W   = 28;    // SPT N bar column width
      const AXIS_W  = 22;    // depth axis width
      const BH_W    = AXIS_W + COL_W + SPT_W + 4;
      const maxDepth = Math.max(...bhs.map(bh =>
        bh.layers.length ? Math.max(...bh.layers.map(l => l.base)) : 10));
      const svgH = Math.min(maxDepth * SCALE + 36, 600);
      const svgW = bhs.length * (BH_W + 8) + 8;

      const strips = bhs.map((bh, bi) => {
        const gl    = bh.groundLevel ?? 0;
        const bhX   = 8 + bi * (BH_W + 8);
        let rows    = '';

        // Unit colour blocks
        bh.layers.forEach(l => {
          const unit = unitByCode[l.unitCode];
          if (!unit) return;
          const y0  = Math.min(l.top  * SCALE + 26, svgH - 2);
          const y1  = Math.min(l.base * SCALE + 26, svgH - 2);
          if (y1 <= y0) return;
          const col = unit.color ?? '#888';
          rows += `<rect x="${bhX + AXIS_W}" y="${y0.toFixed(1)}" width="${COL_W}" height="${(y1-y0).toFixed(1)}" fill="${col}"/>`;
          // Label if tall enough
          if (y1 - y0 > 10) {
            rows += `<text x="${(bhX + AXIS_W + COL_W/2).toFixed(1)}" y="${((y0+y1)/2+3).toFixed(1)}" font-size="7" fill="#fff" text-anchor="middle">${unit.code}</text>`;
          }
        });

        // SPT N bars
        bh.layers.forEach(l => {
          if (l.sptN == null) return;
          const y    = Math.min(((l.top + l.base) / 2) * SCALE + 26, svgH - 2);
          const barW = Math.min(l.sptN / 60 * SPT_W, SPT_W);
          rows += `<rect x="${(bhX + AXIS_W + COL_W + 2).toFixed(1)}" y="${(y-2.5).toFixed(1)}" width="${barW.toFixed(1)}" height="5" fill="#4a6275" opacity="0.7"/>`;
          rows += `<text x="${(bhX + AXIS_W + COL_W + 4 + barW).toFixed(1)}" y="${(y+2.5).toFixed(1)}" font-size="7" fill="#334">${l.sptN}</text>`;
        });

        // Depth axis labels every 5m
        for (let d = 0; d <= maxDepth; d += 5) {
          const y = d * SCALE + 26;
          if (y > svgH - 2) break;
          rows += `<line x1="${bhX + AXIS_W - 4}" y1="${y.toFixed(1)}" x2="${bhX + BH_W}" y2="${y.toFixed(1)}" stroke="#ddd" stroke-width="0.5"/>`;
          rows += `<text x="${(bhX + AXIS_W - 6).toFixed(1)}" y="${(y+3).toFixed(1)}" font-size="7" fill="#889" text-anchor="end">${d}</text>`;
        }

        // BH ID header
        rows += `<text x="${(bhX + AXIS_W + COL_W/2).toFixed(1)}" y="14" font-size="8" font-weight="bold" fill="#334" text-anchor="middle">${bh.id}</text>`;
        rows += `<text x="${(bhX + AXIS_W + COL_W/2).toFixed(1)}" y="23" font-size="7" fill="#889" text-anchor="middle">${gl.toFixed(1)}mAOD</text>`;

        return rows;
      }).join('');

      return `<div style="overflow-x:auto;margin-bottom:8px">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${svgW} ${svgH}" style="height:${svgH}px;width:${svgW}px;background:#f8f9fa;border:1px solid #e0e4ea;border-radius:4px">
  <rect width="${svgW}" height="${svgH}" fill="#f8f9fa"/>
  <!-- Depth axis label -->
  <text x="4" y="40" font-size="8" fill="#889" transform="rotate(-90,8,40)">Depth (m)</text>
  ${strips}
</svg>
<p style="font-size:10px;color:#889;margin-top:4px">Left column: geological unit (coloured) · Right column: SPT N value bars (max scale=60)</p>
</div>`;
    })();

    // ── Risk assessment section ────────────────────────────────────────────────
    const riskHTML = (() => {
      if (!riskReport?.zones?.length) return '';
      const COLORS = { low:'#4fba6f', medium:'#d4a843', high:'#e06040' };
      const zoneCards = riskReport.zones.map(z => {
        const c = COLORS[z.level] ?? '#888';
        return `<tr>
          <td style="border-left:4px solid ${c};padding-left:8px"><strong>${z.icon} ${z.name}</strong></td>
          <td><span style="background:${c};color:#fff;padding:1px 6px;border-radius:3px;font-size:11px">${z.level.toUpperCase()}</span></td>
          <td>${z.pct}%</td>
          <td style="font-size:11px;color:#677">${z.description}</td>
        </tr>`;
      }).join('');
      const oc = COLORS[riskReport.overallLevel] ?? '#888';
      return `<h2>Geotechnical Risk Assessment</h2>
<p style="font-size:12px;margin-bottom:8px">Overall Risk: <strong style="color:${oc}">${(riskReport.overallLevel??'').toUpperCase()}</strong></p>
<table><tr><th>Hazard</th><th>Level</th><th>Site %</th><th>Description</th></tr>${zoneCards}</table>`;
    })();

    const meanCert = (() => {
      let s=0,n=0;
      for (let i=0;i<unitIds.length;i++) if(unitIds[i]){s+=certainty[i];n++;}
      return n ? (s/n*100).toFixed(0)+'%' : '—';
    })();

    // Inject formation tops and period column header
    const periodColHeader = '<th>Period</th>';
    const html = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>GeoModel AI — Site Investigation Report</title>
<style>
  *{box-sizing:border-box}
  body{font-family:Arial,Helvetica,sans-serif;color:#1c2a38;padding:40px;max-width:980px;margin:0 auto;font-size:13px}
  h1{font-size:22px;color:#1c2a38;border-bottom:3px solid #4a6275;padding-bottom:10px;margin-bottom:4px}
  .meta{color:#8898a8;font-size:11px;margin-bottom:24px}
  h2{font-size:12px;text-transform:uppercase;letter-spacing:0.8px;color:#4a6275;margin:28px 0 8px;border-bottom:1px solid #e4e8ec;padding-bottom:4px}
  table{width:100%;border-collapse:collapse;margin-bottom:16px;font-size:12px}
  th{background:#f0f2f5;padding:5px 8px;text-align:left;border-bottom:2px solid #c8cdd6;font-size:10px;text-transform:uppercase;letter-spacing:0.4px}
  td{padding:4px 8px;border-bottom:1px solid #eaeef2;vertical-align:middle}
  tr:last-child td{border-bottom:none}
  .swatch{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:middle}
  .badge{display:inline-block;background:#d04040;color:#fff;font-size:10px;padding:2px 7px;border-radius:3px;margin-left:10px;vertical-align:middle}
  .footer{margin-top:48px;padding-top:12px;border-top:1px solid #e4e8ec;color:#8898a8;font-size:10px;line-height:1.5}
  .info-grid{display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:8px;margin-bottom:16px}
  .info-cell{background:#f8f9fa;border-radius:4px;padding:10px 14px}
  .info-label{font-size:10px;text-transform:uppercase;letter-spacing:0.5px;color:#8898a8}
  .info-value{font-size:18px;font-weight:700;color:#1c2a38;margin-top:2px}
  @media print{.footer{page-break-before:always}}
</style>
</head>
<body>
<h1>Site Investigation Report <span class="badge">PROOF OF CONCEPT</span></h1>
<p class="meta">Generated by GeoModel AI — probabilistic 3D ground model · ${dateStr}</p>

<div class="info-grid">
  <div class="info-cell"><div class="info-label">Boreholes</div><div class="info-value">${bhs.length}</div></div>
  <div class="info-cell"><div class="info-label">Geo. Units</div><div class="info-value">${geoUnits.length}</div></div>
  <div class="info-cell"><div class="info-label">Voxels</div><div class="info-value">${(nx*ny*nz).toLocaleString()}</div></div>
  <div class="info-cell"><div class="info-label">Mean Certainty</div><div class="info-value">${meanCert}</div></div>
</div>

<h2>Site Plan</h2>
${planSVG}

<h2>Geological Units — Volumes &amp; Parameters</h2>
<table>
  <tr><th>Code</th><th>Name</th><th>Period</th><th>Description</th><th>Vol. (m³)</th><th>%</th><th>Cert.</th>
    <th>γ (kN/m³)</th><th>Cu (kPa)</th><th>φ′ (°)</th><th>Cc</th><th>E (MPa)</th><th>SPT N</th></tr>
  ${unitRows}
</table>

${formationTopsHTML}

<h2>Borehole Log Strips</h2>
${logStripsSVG}

<h2>Borehole Register</h2>
<table>
  <tr><th>BH ID</th><th>X (m)</th><th>Y (m)</th><th>GL (mAOD)</th><th>Depth (m)</th><th>Layers</th><th>Units</th><th>Mean SPT N</th></tr>
  ${bhRows}
</table>

${riskHTML}

<h2>Model Grid</h2>
<table>
  <tr><th>Parameter</th><th>Value</th></tr>
  <tr><td>Dimensions (nx × ny × nz)</td><td>${nx} × ${ny} × ${nz}</td></tr>
  <tr><td>Horizontal cell size</td><td>${cs} m</td></tr>
  <tr><td>Vertical cell height</td><td>${ch.toFixed(2)} m</td></tr>
  <tr><td>Grid origin (X, Y, Z)</td><td>(${O.x.toFixed(1)}, ${O.y.toFixed(1)}, ${O.z.toFixed(1)})</td></tr>
  <tr><td>Grid world extent</td><td>${(nx*cs).toFixed(0)} × ${(ny*cs).toFixed(0)} × ${(nz*ch).toFixed(0)} m</td></tr>
</table>

<p class="footer"><strong>DISCLAIMER:</strong> This report is automatically generated from a probabilistic 3D interpolation model built from limited site investigation data. It is intended for project management and preliminary design purposes only. All ground conditions, geotechnical parameters, and risk assessments must be independently verified by a suitably qualified and experienced geotechnical engineer before being used for design, specification, or construction. This document does not constitute a geotechnical investigation report and should not be used as such.</p>
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

// ICS period colour lookup (shared with properties.js GEO_PERIODS)
function _periodColor(code) {
  const MAP = {
    Q:'#f9f97f',Qh:'#fff2ae',Qp:'#fff987',Ng:'#ffe619',Pg:'#fd9a52',
    K:'#7fc64e',J:'#34b2c9',Tr:'#812b92',P:'#f04028',C:'#67a599',
    D:'#cb8c37',S:'#b3e1b6',O:'#009270',Cm:'#7fa056',pC:'#f74370',
    MG:'#b5b5b5',Al:'#daf0e3',RT:'#f9ddb9',
  };
  return MAP[code] ?? '#e0e0e0';
}
