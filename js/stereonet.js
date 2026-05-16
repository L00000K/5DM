// js/stereonet.js — Surface orientation analysis
// Schmidt equal-area lower hemisphere stereonet + strike rose diagram.
// Orientations are derived from elevation gradients of unit top surfaces.

export function computeOrientations(grid, geoUnits) {
  const { nx, ny, nz, cellSize: cs, cellHeight: ch, origin: O, unitIds } = grid;
  const result = {};

  for (const unit of geoUnits) {
    if (!unit.id) continue;

    // Build top-elevation grid for this unit
    const topZ = new Float32Array(nx * ny).fill(NaN);
    for (let iy = 0; iy < ny; iy++) {
      for (let ix = 0; ix < nx; ix++) {
        for (let iz = nz - 1; iz >= 0; iz--) {
          if (unitIds[ix + iy * nx + iz * nx * ny] === unit.id) {
            topZ[ix + iy * nx] = O.y + (iz + 0.5) * ch;
            break;
          }
        }
      }
    }

    // Central-difference gradients → dip / strike per cell
    const orientations = [];
    for (let iy = 1; iy < ny - 1; iy++) {
      for (let ix = 1; ix < nx - 1; ix++) {
        const zC = topZ[ix     + iy     * nx];
        const zE = topZ[(ix+1) + iy     * nx];
        const zW = topZ[(ix-1) + iy     * nx];
        const zN = topZ[ix     + (iy+1) * nx]; // +Y = north in grid
        const zS = topZ[ix     + (iy-1) * nx];
        if ([zC,zE,zW,zN,zS].some(isNaN)) continue;

        const dZdE = (zE - zW) / (2 * cs); // east gradient
        const dZdN = (zN - zS) / (2 * cs); // north gradient
        const grad  = Math.hypot(dZdE, dZdN);
        const dip   = Math.atan(grad) * 180 / Math.PI;

        // Dip direction: steepest downward (invert gradient vector)
        const dipDir = (Math.atan2(-dZdE, -dZdN) * 180 / Math.PI + 360) % 360;
        const strike = (dipDir - 90 + 360) % 360;
        orientations.push({ dip, dipDir, strike });
      }
    }

    if (orientations.length) result[unit.code] = orientations;
  }
  return result;
}

// Fisher mean of a set of dip/dipDir vectors (directional statistics)
function _fisherMean(orientations) {
  let Sx = 0, Sy = 0, Sz = 0;
  for (const { dip, dipDir } of orientations) {
    const dr = dip * Math.PI / 180;
    const ar = dipDir * Math.PI / 180;
    Sx += Math.sin(dr) * Math.sin(ar);
    Sy += Math.sin(dr) * Math.cos(ar);
    Sz += Math.cos(dr);
  }
  const N = orientations.length;
  const R = Math.sqrt(Sx*Sx + Sy*Sy + Sz*Sz);
  if (R < 1e-9) return { dip: 0, dipDir: 0, R: 0, N };
  return {
    dip:    Math.acos(Math.min(1, Sz / R)) * 180 / Math.PI,
    dipDir: (Math.atan2(Sx, Sy) * 180 / Math.PI + 360) % 360,
    R: R / N,
    N,
  };
}

export function orientationStats(allOrientations) {
  const stats = {};
  for (const [code, orients] of Object.entries(allOrientations)) {
    if (!orients.length) continue;
    const mean = _fisherMean(orients);
    const dips = orients.map(o => o.dip);
    const avg  = dips.reduce((a, b) => a + b, 0) / dips.length;
    const std  = Math.sqrt(dips.reduce((s, d) => s + (d - avg) ** 2, 0) / dips.length);
    stats[code] = {
      n: orients.length,
      meanDip:    mean.dip.toFixed(1),
      meanDipDir: mean.dipDir.toFixed(0),
      stdDip:     std.toFixed(1),
      R:          mean.R.toFixed(3),
    };
  }
  return stats;
}

// Lambert equal-area lower hemisphere stereonet
export function renderStereonet(canvas, orientations, color = '#5ab8e0') {
  const W = canvas.width, H = canvas.height;
  const ctx = canvas.getContext('2d');
  const cx = W / 2, cy = H / 2;
  const R = Math.min(cx, cy) - 6;

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#151f2c';
  ctx.beginPath(); ctx.arc(cx, cy, R, 0, Math.PI * 2); ctx.fill();

  // Reference circles at 30° and 60° from horizontal
  ctx.strokeStyle = '#1e3048'; ctx.lineWidth = 0.7;
  [30, 60].forEach(d => {
    const r = 2 * R * Math.sin(d / 2 * Math.PI / 180);
    ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.stroke();
  });
  // Radial lines every 45°
  for (let az = 0; az < 360; az += 45) {
    const a = az * Math.PI / 180;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + R * Math.sin(a), cy - R * Math.cos(a));
    ctx.stroke();
  }
  ctx.strokeStyle = '#2a4060'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.arc(cx, cy, R, 0, Math.PI * 2); ctx.stroke();

  // Compass labels
  ctx.fillStyle = '#6a7f96'; ctx.font = `${Math.max(7, Math.round(R * 0.14))}px monospace`;
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText('N', cx, cy - R - 5);
  ctx.fillText('S', cx, cy + R + 6);
  ctx.fillText('E', cx + R + 6, cy);
  ctx.fillText('W', cx - R - 5, cy);

  if (!orientations?.length) return;

  // Plot poles — Lambert r = 2R sin(dip/2)
  const alpha = Math.max(0.12, Math.min(0.75, 30 / orientations.length));
  ctx.fillStyle = color;
  ctx.globalAlpha = alpha;
  for (const { dip, dipDir } of orientations) {
    const r  = 2 * R * Math.sin(dip / 2 * Math.PI / 180);
    const a  = dipDir * Math.PI / 180; // dip direction in Schmidt net = downward direction
    ctx.beginPath(); ctx.arc(cx + r * Math.sin(a), cy - r * Math.cos(a), 2, 0, Math.PI * 2); ctx.fill();
  }
  ctx.globalAlpha = 1;

  // Mean pole (red circle)
  const mean = _fisherMean(orientations);
  const mr = 2 * R * Math.sin(mean.dip / 2 * Math.PI / 180);
  const ma = mean.dipDir * Math.PI / 180;
  ctx.fillStyle = '#ff6b6b';
  ctx.beginPath(); ctx.arc(cx + mr * Math.sin(ma), cy - mr * Math.cos(ma), 4, 0, Math.PI * 2); ctx.fill();
}

// Bidirectional strike rose diagram
export function renderRoseDiagram(canvas, orientations, color = '#5ab8e0') {
  const W = canvas.width, H = canvas.height;
  const ctx = canvas.getContext('2d');
  const cx = W / 2, cy = H / 2;
  const R = Math.min(cx, cy) - 6;

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#151f2c';
  ctx.beginPath(); ctx.arc(cx, cy, R, 0, Math.PI * 2); ctx.fill();

  const BINS = 18;  // 10° bins over 0–180° (bidirectional)
  const counts = new Array(BINS).fill(0);
  let total = 0;
  if (orientations?.length) {
    for (const { strike } of orientations) {
      const b = Math.floor(((strike % 180) / 180) * BINS);
      counts[Math.min(b, BINS - 1)]++;
      total++;
    }
  }
  const maxC = Math.max(...counts, 1);

  ctx.strokeStyle = '#1e3048'; ctx.lineWidth = 0.7;
  [0.5, 1].forEach(f => { ctx.beginPath(); ctx.arc(cx, cy, R * f, 0, Math.PI * 2); ctx.stroke(); });

  if (total > 0) {
    ctx.fillStyle = color; ctx.globalAlpha = 0.7;
    for (let b = 0; b < BINS; b++) {
      if (!counts[b]) continue;
      const r  = (counts[b] / maxC) * R;
      const a0 = (b / BINS) * Math.PI - Math.PI / 2;
      const a1 = ((b + 1) / BINS) * Math.PI - Math.PI / 2;
      for (const offset of [0, Math.PI]) {
        ctx.beginPath(); ctx.moveTo(cx, cy);
        ctx.arc(cx, cy, r, a0 + offset, a1 + offset);
        ctx.closePath(); ctx.fill();
      }
    }
    ctx.globalAlpha = 1;
  }

  ctx.strokeStyle = '#2a4060'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.arc(cx, cy, R, 0, Math.PI * 2); ctx.stroke();

  ctx.fillStyle = '#6a7f96'; ctx.font = `${Math.max(7, Math.round(R * 0.14))}px monospace`;
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText('N', cx, cy - R - 5);
  ctx.fillText('S', cx, cy + R + 6);
  ctx.fillText('E', cx + R + 6, cy);
  ctx.fillText('W', cx - R - 5, cy);
}
