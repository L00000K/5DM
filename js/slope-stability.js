// js/slope-stability.js — Bishop simplified circular slip stability analysis
// Implements the Bishop (1955) simplified method with grid search over slip circles.
// Used for cut slopes and embankments in geotechnical assessment.

// Bishop simplified: iterative solution for Factor of Safety
function _bishopFs(slices, phiRad, cPrime, gamma, gwtDepth) {
  const n = slices.length;
  let Fs = 1.5; // initial estimate
  for (let iter = 0; iter < 20; iter++) {
    let numerator = 0, denominator = 0;
    for (const s of slices) {
      const { b, h, alpha, hW } = s;
      const W = gamma * b * h;
      const N = W - gamma * b * Math.max(0, hW); // buoyant for submerged slice
      const mAlpha = Math.cos(alpha) + (Math.sin(alpha) * Math.tan(phiRad)) / Fs;
      if (Math.abs(mAlpha) < 1e-9) continue;
      numerator  += (cPrime * b + N * Math.tan(phiRad)) / mAlpha;
      denominator += W * Math.sin(alpha);
    }
    const newFs = denominator > 0 ? numerator / denominator : Fs;
    if (Math.abs(newFs - Fs) < 0.001) { Fs = newFs; break; }
    Fs = newFs;
  }
  return Fs;
}

// Divide a slip arc into N vertical slices
function _makeSlices(cx, cy, R, xLeft, xRight, surfFn, gwtDepth, nSlices) {
  const slices = [];
  const b = (xRight - xLeft) / nSlices;
  for (let i = 0; i < nSlices; i++) {
    const xM = xLeft + (i + 0.5) * b;
    // Base of slice on circle
    const dx = xM - cx;
    if (Math.abs(dx) > R) continue;
    const yBase  = cy - Math.sqrt(Math.max(0, R * R - dx * dx));
    const ySurf  = surfFn(xM);
    const h = ySurf - yBase;
    if (h <= 0) continue;
    const alpha = Math.atan2(dx, Math.sqrt(Math.max(0, R * R - dx * dx))); // base angle from horizontal
    const hW = gwtDepth != null ? Math.max(0, h - gwtDepth) : 0;
    slices.push({ b, h, alpha, hW });
  }
  return slices;
}

// Extract a simple 2D slope profile from the voxel grid at a given iy row (EW section)
function _extractEWSurface(grid, iy) {
  const { nx, nz, cellSize: cs, cellHeight: ch, origin: O, unitIds } = grid;
  const n2 = nx * grid.ny;
  const profile = [];
  for (let ix = 0; ix < nx; ix++) {
    const wx = O.x + ix * cs + cs / 2;
    let topElev = O.y; // default to base
    for (let iz = nz - 1; iz >= 0; iz--) {
      if (unitIds[ix + iy * nx + iz * n2] !== 0) {
        topElev = O.y + (iz + 1) * ch;
        break;
      }
    }
    profile.push({ x: wx, y: topElev });
  }
  return profile;
}

// Linear interpolation of profile at x
function _interpSurf(profile, x) {
  if (!profile.length) return 0;
  if (x <= profile[0].x)  return profile[0].y;
  if (x >= profile[profile.length - 1].x) return profile[profile.length - 1].y;
  for (let i = 1; i < profile.length; i++) {
    if (profile[i].x >= x) {
      const t = (x - profile[i-1].x) / (profile[i].x - profile[i-1].x);
      return profile[i-1].y + t * (profile[i].y - profile[i-1].y);
    }
  }
  return profile[profile.length - 1].y;
}

// Grid search for minimum Fs slip circle.
// Returns { Fs, cx, cy, R, slices, profile }
export function bishopAnalysis(grid, geoUnits, options = {}) {
  const { ny, cellSize: cs, cellHeight: ch, origin: O } = grid;
  const iy = Math.floor(ny / 2); // midpoint section (E-W)
  const profile = _extractEWSurface(grid, iy);
  if (profile.length < 3) return null;

  // Dominant unit (most voxels in mid-section, excluding UNKN)
  const unitCount = {};
  const { nx, nz, unitIds } = grid;
  const n2 = nx * ny;
  for (let ix = 0; ix < nx; ix++) {
    for (let iz = 0; iz < nz; iz++) {
      const uid = unitIds[ix + iy * nx + iz * n2];
      if (uid) unitCount[uid] = (unitCount[uid] ?? 0) + 1;
    }
  }
  const dominantUid = Object.entries(unitCount).sort((a,b) => b[1]-a[1])[0]?.[0];
  const dominantUnit = geoUnits.find(u => String(u.id) === String(dominantUid));
  const p = dominantUnit?.params ?? {};

  const cPrime  = options.cPrime  ?? (p.cu ?? 0) * 0.4; // effective c' estimate
  const phiDeg  = options.phiDeg  ?? (p.phi ?? 25);
  const gamma   = options.gamma   ?? (p.gamma ?? 18);
  const gwtD    = options.gwtDepth;

  const phiRad = phiDeg * Math.PI / 180;

  // Profile extents
  const xMin = profile[0].x, xMax = profile[profile.length - 1].x;
  const yMin = Math.min(...profile.map(p => p.y));
  const yMax = Math.max(...profile.map(p => p.y));
  const H = yMax - yMin; // slope height

  const surfFn = x => _interpSurf(profile, x);

  let bestFs = Infinity, bestCircle = null;
  const STEPS = 12;

  // Grid search: circle centres above and behind slope crest
  for (let ci = 0; ci <= STEPS; ci++) {
    const cxC = xMin + (ci / STEPS) * (xMax - xMin);
    for (let cj = 0; cj <= STEPS; cj++) {
      const cyC = yMin + H * 0.5 + (cj / STEPS) * H * 1.5;
      const minR = H * 0.3;
      const maxR = H * 3.0;
      for (let rk = 0; rk <= 8; rk++) {
        const R = minR + (rk / 8) * (maxR - minR);
        // Check circle intersects slope surface
        const yEntry = cyC - Math.sqrt(Math.max(0, R * R - (xMin - cxC) ** 2));
        const yExit  = cyC - Math.sqrt(Math.max(0, R * R - (xMax - cxC) ** 2));
        if (yEntry > yMax + H * 0.1 || yExit > yMax + H * 0.1) continue;

        const xLeft  = Math.max(xMin, cxC - R);
        const xRight = Math.min(xMax, cxC + R);
        if (xRight - xLeft < cs) continue;

        const slices = _makeSlices(cxC, cyC, R, xLeft, xRight, surfFn, gwtD, 20);
        if (slices.length < 3) continue;

        const fs = _bishopFs(slices, phiRad, cPrime, gamma, gwtD);
        if (fs > 0.3 && fs < bestFs) {
          bestFs = fs;
          bestCircle = { cx: cxC, cy: cyC, R, slices, profile };
        }
      }
    }
  }

  if (!bestCircle) return null;
  return {
    Fs: Math.max(0.1, bestFs),
    cx: bestCircle.cx,
    cy: bestCircle.cy,
    R:  bestCircle.R,
    profile: bestCircle.profile,
    unitName: dominantUnit?.name ?? 'Unknown',
    params: { cPrime, phiDeg, gamma, gwtD },
  };
}

// Render the slope cross-section with slip circle on an SVG string
export function renderSlopeSection(result, width = 400, height = 200) {
  if (!result) return '<p class="hint">No valid slip circle found.</p>';
  const { Fs, cx, cy, R, profile, unitName } = result;
  if (!profile?.length) return '';

  const xMin = profile[0].x, xMax = profile[profile.length - 1].x;
  const yMin = Math.min(...profile.map(p => p.y), cy - R);
  const yMax = Math.max(...profile.map(p => p.y), cy + R * 0.1);
  const pad = 20;
  const w = width - 2 * pad, h = height - 2 * pad;

  const sx = x => pad + ((x - xMin) / (xMax - xMin)) * w;
  const sy = y => pad + h - ((y - yMin) / (yMax - yMin)) * h;

  // Profile path
  const profilePts = profile.map(p => `${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(' ');
  const xL = sx(xMin), xR = sx(xMax), yBot = sy(yMin);
  const profilePath = `${xL},${yBot} ${profilePts} ${xR},${yBot}`;

  // Slip circle arc
  const scx = sx(cx), scy = sy(cy), sR = (R / (xMax - xMin)) * w;
  const color = Fs < 1.2 ? '#e84040' : Fs < 1.5 ? '#e8924a' : '#4ae87a';

  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" style="background:#151f2c;border-radius:4px">
  <polygon points="${profilePath}" fill="#2a6040" stroke="#3a8060" stroke-width="1"/>
  <circle cx="${scx.toFixed(1)}" cy="${scy.toFixed(1)}" r="${sR.toFixed(1)}"
    fill="none" stroke="${color}" stroke-width="1.5" stroke-dasharray="4 2"/>
  <text x="${width/2}" y="14" text-anchor="middle" font-size="10" fill="${color}" font-family="monospace">
    Bishop Fs = ${Fs.toFixed(2)} (${Fs < 1.2 ? 'UNSAFE' : Fs < 1.5 ? 'MARGINAL' : 'STABLE'})
  </text>
  <text x="${width/2}" y="${height-4}" text-anchor="middle" font-size="9" fill="#6a7f96" font-family="monospace">
    ${unitName} · c′=${result.params.cPrime.toFixed(0)}kPa · φ=${result.params.phiDeg}° · γ=${result.params.gamma}kN/m³
  </text>
</svg>`;
}
