import { log } from './app.js';
import { buildStratRankMap, stratigraphicConsistencyPenalty,
         descriptionJaccard, meanDescriptionSimilarity, buildTransitionMatrix } from './semantic-engine.js';
import { trainGeoImplicit, inferGeoImplicit, finetuneGeoImplicit,
         findUncertainClusters, patchWithOracle } from './geo-implicit.js';

const MIN_BH_DIST = 0.1;

// ── Minimum-curvature drillhole trajectory ─────────────────────────────────────
// survey = [{depth, incl (° from vertical), azim (° from North)}] sorted ascending
// Returns [{depth, dx, dy}] cumulative horizontal offsets from collar (dx=Easting, dy=Northing)
function computeTrajectory(bh) {
  const survey = bh.deviation;
  if (!survey || survey.length < 2) return null;
  const st = [...survey].sort((a, b) => a.depth - b.depth);
  if (st[0].depth > 0.01) st.unshift({ depth: 0, incl: 0, azim: st[0].azim });

  const result = [{ depth: st[0].depth, dx: 0, dy: 0 }];
  let dx = 0, dy = 0;
  for (let i = 1; i < st.length; i++) {
    const i1 = st[i - 1].incl * Math.PI / 180, i2 = st[i].incl * Math.PI / 180;
    const a1 = st[i - 1].azim * Math.PI / 180, a2 = st[i].azim * Math.PI / 180;
    const di = st[i].depth - st[i - 1].depth;
    // Dogleg severity → ratio factor for minimum curvature
    const cosDL = Math.cos(i2 - i1) - Math.sin(i1) * Math.sin(i2) * (1 - Math.cos(a2 - a1));
    const dl = Math.acos(Math.max(-1, Math.min(1, cosDL)));
    const RF = dl < 0.0001 ? 1 : 2 * Math.tan(dl / 2) / dl;
    // Easting (+x) and Northing (+y) increments
    dx += di / 2 * (Math.sin(i1) * Math.sin(a1) + Math.sin(i2) * Math.sin(a2)) * RF;
    dy += di / 2 * (Math.sin(i1) * Math.cos(a1) + Math.sin(i2) * Math.cos(a2)) * RF;
    result.push({ depth: st[i].depth, dx, dy });
  }
  return result;
}

// Returns {dx, dy} — horizontal offset from collar at measured depth d
function getDeviatedXY(bh, d) {
  const traj = bh._trajectory;
  if (!traj) return { dx: 0, dy: 0 };
  if (d <= traj[0].depth) return { dx: traj[0].dx, dy: traj[0].dy };
  const last = traj[traj.length - 1];
  if (d >= last.depth) return { dx: last.dx, dy: last.dy };
  for (let i = 1; i < traj.length; i++) {
    if (traj[i].depth >= d) {
      const t = (d - traj[i - 1].depth) / (traj[i].depth - traj[i - 1].depth);
      return {
        dx: traj[i - 1].dx + t * (traj[i].dx - traj[i - 1].dx),
        dy: traj[i - 1].dy + t * (traj[i].dy - traj[i - 1].dy),
      };
    }
  }
  return { dx: last.dx, dy: last.dy };
}

// ── Gaussian elimination with partial pivoting ────────────────────────────────
function solveLinear(Ain, bin) {
  const n = bin.length;
  const A = Ain.map((row, i) => [...row, bin[i]]);
  for (let col = 0; col < n; col++) {
    let maxRow = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(A[r][col]) > Math.abs(A[maxRow][col])) maxRow = r;
    }
    [A[col], A[maxRow]] = [A[maxRow], A[col]];
    const pivot = A[col][col];
    if (Math.abs(pivot) < 1e-12) continue;
    for (let r = col + 1; r < n; r++) {
      const f = A[r][col] / pivot;
      for (let c = col; c <= n; c++) A[r][c] -= f * A[col][c];
    }
  }
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    x[i] = A[i][n];
    for (let j = i + 1; j < n; j++) x[i] -= A[i][j] * x[j];
    if (Math.abs(A[i][i]) > 1e-12) x[i] /= A[i][i];
  }
  return x;
}

// ── Variogram / kernel functions ──────────────────────────────────────────────
function gammaSpherical(d, range, sill) {
  if (d <= 0) return 0;
  if (d >= range) return sill;
  const h = d / range;
  return sill * (1.5 * h - 0.5 * h * h * h);
}

function kernelRBF(d, l) {
  return Math.exp(-(d * d) / (2 * l * l));
}

// ── Tiny two-layer MLP for neural-network interpolation ───────────────────────
class TinyMLP {
  constructor(nIn, nHidden, nOut) {
    const he = (rows, cols) =>
      Array.from({ length: rows }, () =>
        Array.from({ length: cols }, () => (Math.random() * 2 - 1) * Math.sqrt(2 / cols)));
    this.W1 = he(nHidden, nIn);
    this.b1 = new Array(nHidden).fill(0);
    this.W2 = he(nOut, nHidden);
    this.b2 = new Array(nOut).fill(0);
    this.nH = nHidden;
    this.nO = nOut;
  }

  forward(x) {
    const h = this.W1.map((row, i) =>
      Math.max(0, row.reduce((s, w, j) => s + w * x[j], 0) + this.b1[i]));
    const raw = this.W2.map((row, i) =>
      row.reduce((s, w, j) => s + w * h[j], 0) + this.b2[i]);
    const m = Math.max(...raw);
    const e = raw.map(v => Math.exp(v - m));
    const sm = e.reduce((a, b) => a + b, 0);
    return { h, probs: e.map(v => v / sm) };
  }

  step(x, labelIdx, lr) {
    const { h, probs } = this.forward(x);
    const dOut = probs.map((p, i) => p - (i === labelIdx ? 1 : 0));
    for (let i = 0; i < this.nO; i++) {
      for (let j = 0; j < this.nH; j++) this.W2[i][j] -= lr * dOut[i] * h[j];
      this.b2[i] -= lr * dOut[i];
    }
    const dH = new Array(this.nH).fill(0);
    for (let j = 0; j < this.nH; j++) {
      for (let i = 0; i < this.nO; i++) dH[j] += dOut[i] * this.W2[i][j];
      dH[j] *= h[j] > 0 ? 1 : 0;
    }
    for (let i = 0; i < this.nH; i++) {
      for (let j = 0; j < x.length; j++) this.W1[i][j] -= lr * dH[i] * x[j];
      this.b1[i] -= lr * dH[i];
    }
  }
}

// ── BH data at a query depth ──────────────────────────────────────────────────
// Uses deviated (minimum-curvature) position when trajectory data is available.
function getCandidates(boreholes, x, y, z, sinAz = 0, cosAz = 1, anisoRatio = 1) {
  const out = [];
  for (const bh of boreholes) {
    const depth = (bh.groundLevel ?? 0) - z;
    if (!bh.layers.length || depth < 0) continue;
    let layer;
    if      (depth < bh.layers[0].top)                       layer = bh.layers[0];
    else if (depth > bh.layers[bh.layers.length - 1].base)   layer = bh.layers[bh.layers.length - 1];
    else layer = bh.layers.find(l => depth >= l.top && depth <= l.base);
    if (!layer?.unitCode) continue;

    // Apply deviation correction at layer midpoint
    const layerMid = (layer.top + layer.base) * 0.5;
    const { dx: devX, dy: devY } = getDeviatedXY(bh, layerMid);
    const bhX = bh.x + devX, bhY = bh.y + devY;

    const ddx = bhX - x, ddy = bhY - y;
    const dAlong = ddx * sinAz + ddy * cosAz;
    const dPerp  = ddx * cosAz - ddy * sinAz;
    const dist2d = Math.hypot(dAlong, dPerp / anisoRatio);
    out.push({ dist: Math.max(dist2d, MIN_BH_DIST), x: bhX, y: bhY,
               unitCode: layer.unitCode, layerCert: layer.certainty ?? 0.8,
               description: layer.description ?? '' });
  }
  return out;
}

// ── Result shape helper ───────────────────────────────────────────────────────
function makeResult(bestCode, secondCode, bestShare, totalW, secondW, certainty, unitIndex, unknownId) {
  return {
    unitId:      unitIndex[bestCode]    ?? unknownId,
    certainty:   Math.min(1, certainty),
    blendUnitId: secondCode
      ? (unitIndex[secondCode] ?? unknownId)
      : (unitIndex[bestCode]   ?? unknownId),
    blendRatio: totalW > 0 ? secondW / totalW : 0,
  };
}

// ── IDW ───────────────────────────────────────────────────────────────────────
function idwVote(neighbours, power, unitIndex, unknownId, typicalSpacing) {
  const votes = {};
  let totalW = 0;
  for (const n of neighbours) {
    const w = n.layerCert / Math.pow(n.dist, power);
    votes[n.unitCode] = (votes[n.unitCode] ?? 0) + w;
    totalW += w;
  }
  const sorted = Object.entries(votes).sort((a, b) => b[1] - a[1]);
  const [bc, bw] = sorted[0];
  const [sc, sw = 0] = sorted[1] ?? [];
  const dDecay = Math.exp(-neighbours[0].dist / typicalSpacing);
  const mCert  = neighbours.reduce((s, n) => s + n.layerCert, 0) / neighbours.length;
  const cert   = (bw / totalW) * 0.55 + dDecay * 0.30 + mCert * 0.15;
  // Semantic cohesion bonus for winning unit
  const winnerDescs = neighbours.filter(n => n.unitCode === bc).map(n => n.description);
  const semScore = meanDescriptionSimilarity(winnerDescs);
  const semAdj = 0.8 + 0.4 * semScore;
  return makeResult(bc, sc, bw / totalW, totalW, sw, Math.min(1, cert * semAdj), unitIndex, unknownId);
}

// ── Topographic surface elevation at (x, y) via IDW from scattered points ────
// Returns the interpolated elevation. If topoPoints is empty, returns +Infinity
// so all voxels pass the "below ground" check.
function _topoElevAt(x, y, topoPoints) {
  if (!topoPoints?.length) return Infinity;
  const k = Math.min(6, topoPoints.length);
  // Find k nearest by Euclidean XY distance
  let maxHeap = []; // will hold [negDist2, idx]
  for (let i = 0; i < topoPoints.length; i++) {
    const dx = topoPoints[i].x - x, dy = topoPoints[i].y - y;
    const d2 = dx * dx + dy * dy;
    if (maxHeap.length < k) {
      maxHeap.push([d2, i]);
      maxHeap.sort((a, b) => b[0] - a[0]); // max at front
    } else if (d2 < maxHeap[0][0]) {
      maxHeap[0] = [d2, i];
      maxHeap.sort((a, b) => b[0] - a[0]);
    }
  }
  let wSum = 0, zSum = 0;
  for (const [d2, i] of maxHeap) {
    const w = 1 / (d2 + 1e-9);
    wSum += w; zSum += w * topoPoints[i].z;
  }
  return wSum > 0 ? zSum / wSum : Infinity;
}

// Apply topographic masking to a completed grid in-place.
// Sets unitIds[idx]=0 and certainty[idx]=0 for voxels whose centre elevation
// exceeds the IDW-interpolated topo surface at that (x,y) column.
function _applyTopoMask(unitIds, certainty, gridMeta, topoPoints) {
  if (!topoPoints?.length) return;
  const { nx, ny, nz, cellSize, cellHeight, origin } = gridMeta;
  const ox = origin.x, oy = origin.z, oz = origin.y; // note: origin.z = min Northing
  for (let iy = 0; iy < ny; iy++) {
    const wy = oy + iy * cellSize + cellSize * 0.5;
    for (let ix = 0; ix < nx; ix++) {
      const wx   = ox + ix * cellSize + cellSize * 0.5;
      const elev = _topoElevAt(wx, wy, topoPoints);
      for (let iz = 0; iz < nz; iz++) {
        const wz  = oz + iz * cellHeight + cellHeight * 0.5;
        if (wz > elev) {
          const idx = ix + iy * nx + iz * nx * ny;
          unitIds[idx]   = 0;
          certainty[idx] = 0;
        }
      }
    }
  }
}

// ── Anisotropic distance transform for Kriging ────────────────────────────────
// When aniso is provided ({sinAz, cosAz, ratio}), the major axis (azimuth Az) uses
// full range; the minor axis uses range/ratio. This gives an ellipsoidal variogram
// that matches the concept's predicted directional elongation.
function _anisoD(dx, dy, aniso) {
  if (!aniso) return Math.hypot(dx, dy);
  const { sinAz, cosAz, ratio } = aniso;
  const major = dx * sinAz + dy * cosAz;  // projection onto major axis
  const minor = dx * cosAz - dy * sinAz;  // projection onto minor axis
  return Math.sqrt(major * major + (minor * ratio) * (minor * ratio));
}

// ── Ordinary Kriging (spherical variogram, indicator approach) ────────────────
//   Augments the covariance system with a Lagrange multiplier to enforce
//   the unbiasedness constraint (weights sum to 1), giving optimal linear
//   unbiased prediction at the query location.
//   aniso: optional {sinAz, cosAz, ratio} for anisotropic variogram (from concept store)
function krigingVote(neighbours, qx, qy, unitIndex, unknownId, range, sill, nugget = null, aniso = null) {
  const n = neighbours.length;
  if (nugget === null) nugget = sill * 0.05;
  const sz = n + 1;
  const K = Array.from({ length: sz }, () => new Array(sz).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const dx = neighbours[i].x - neighbours[j].x, dy = neighbours[i].y - neighbours[j].y;
      const d = _anisoD(dx, dy, aniso);
      K[i][j] = i === j ? sill + nugget : sill - gammaSpherical(d, range, sill);
    }
    K[i][n] = K[n][i] = 1;
  }
  const k = new Array(sz).fill(1);
  for (let i = 0; i < n; i++) {
    const dx = neighbours[i].x - qx, dy = neighbours[i].y - qy;
    const d = _anisoD(dx, dy, aniso);
    k[i] = sill - gammaSpherical(d, range, sill);
  }
  k[n] = 1; // Lagrange RHS

  let w;
  try { w = solveLinear(K, k); } catch { return null; }

  const krigVar = Math.max(0,
    sill - w.slice(0, n).reduce((s, wi, i) => s + wi * k[i], 0) - w[n]);
  const certainty = Math.max(0.05, 1 - Math.sqrt(krigVar / (sill + 1e-9)));

  const votes = {};
  let wPos = 0;
  for (let i = 0; i < n; i++) {
    const wi = Math.max(0, w[i]) * neighbours[i].layerCert;
    votes[neighbours[i].unitCode] = (votes[neighbours[i].unitCode] ?? 0) + wi;
    wPos += wi;
  }
  if (wPos < 1e-12) return null;
  const sorted = Object.entries(votes).sort((a, b) => b[1] - a[1]);
  const [bc, bv] = sorted[0];
  const [sc, sv = 0] = sorted[1] ?? [];
  return makeResult(bc, sc, bv / wPos, wPos, sv, certainty * (bv / wPos), unitIndex, unknownId);
}

// ── Gaussian Process (squared-exponential / RBF kernel, Simple Kriging) ───────
//   Probabilistic ML approach: fits a GP to the borehole observations, giving
//   a posterior distribution over unit membership and a principled variance
//   estimate for uncertainty quantification.
function gpVote(neighbours, qx, qy, unitIndex, unknownId, lengthScale) {
  const n = neighbours.length;
  const jitter = 0.01;
  const K = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => {
      const d = Math.hypot(neighbours[i].x - neighbours[j].x, neighbours[i].y - neighbours[j].y);
      return kernelRBF(d, lengthScale) + (i === j ? jitter : 0);
    }));
  const kStar = neighbours.map(nb =>
    kernelRBF(Math.hypot(nb.x - qx, nb.y - qy), lengthScale));

  let alpha;
  try { alpha = solveLinear(K, kStar); } catch { return null; }

  const varPost = Math.max(0, 1 - kStar.reduce((s, k, i) => s + k * alpha[i], 0));
  const certainty = Math.max(0.05, 1 - varPost);

  const votes = {};
  let wPos = 0;
  for (let i = 0; i < n; i++) {
    const wi = Math.max(0, alpha[i]) * neighbours[i].layerCert;
    votes[neighbours[i].unitCode] = (votes[neighbours[i].unitCode] ?? 0) + wi;
    wPos += wi;
  }
  if (wPos < 1e-12) return null;
  const sorted = Object.entries(votes).sort((a, b) => b[1] - a[1]);
  const [bc, bv] = sorted[0];
  const [sc, sv = 0] = sorted[1] ?? [];
  return makeResult(bc, sc, bv / wPos, wPos, sv, certainty, unitIndex, unknownId);
}

// ── Universal Kriging (polynomial drift removal, order 1 = linear trend) ──────
//   Fits a polynomial trend surface to the domain, then Kriges residuals.
//   Order 0 = Ordinary Kriging. Order 1 = linear (regional dip/tilt).
//   Order 2 = quadratic (fold structures, bowl-shaped stratigraphy).
function ukVote(neighbours, qx, qy, unitIndex, unknownId, range, sill, trendOrder, aniso = null) {
  const n = neighbours.length;
  const nugget = sill * 0.05;

  const basis = (x, y) => {
    if (trendOrder <= 0) return [1];
    if (trendOrder === 1) return [1, x, y];
    return [1, x, y, x * y, x * x, y * y];
  };
  const f0 = basis(qx, qy);
  const p  = f0.length;
  if (n < p) return null; // underdetermined

  const sz = n + p;
  const K = Array.from({ length: sz }, () => new Array(sz).fill(0));

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const dx = neighbours[i].x - neighbours[j].x, dy = neighbours[i].y - neighbours[j].y;
      const d = _anisoD(dx, dy, aniso);
      K[i][j] = i === j ? sill + nugget : sill - gammaSpherical(d, range, sill);
    }
    const fi = basis(neighbours[i].x, neighbours[i].y);
    for (let j = 0; j < p; j++) {
      K[i][n + j] = fi[j];
      K[n + j][i] = fi[j];
    }
  }

  const rhs = new Array(sz).fill(0);
  for (let i = 0; i < n; i++) {
    const dx = neighbours[i].x - qx, dy = neighbours[i].y - qy;
    const d = _anisoD(dx, dy, aniso);
    rhs[i] = sill - gammaSpherical(d, range, sill);
  }
  for (let j = 0; j < p; j++) rhs[n + j] = f0[j];

  let sol;
  try { sol = solveLinear(K, rhs); } catch { return null; }

  const krigVar = Math.max(0,
    sill
    - sol.slice(0, n).reduce((s, wi, i) => s + wi * rhs[i], 0)
    - sol.slice(n).reduce((s, li, j) => s + li * f0[j], 0));
  const certainty = Math.max(0.05, 1 - Math.sqrt(krigVar / (sill + 1e-9)));

  const votes = {};
  let wPos = 0;
  for (let i = 0; i < n; i++) {
    const wi = Math.max(0, sol[i]) * neighbours[i].layerCert;
    votes[neighbours[i].unitCode] = (votes[neighbours[i].unitCode] ?? 0) + wi;
    wPos += wi;
  }
  if (wPos < 1e-12) return null;
  const sorted = Object.entries(votes).sort((a, b) => b[1] - a[1]);
  const [bc, bv] = sorted[0];
  const [sc, sv = 0] = sorted[1] ?? [];
  return makeResult(bc, sc, bv / wPos, wPos, sv, certainty * (bv / wPos), unitIndex, unknownId);
}

// ── Neural Network: train on BH observations, return per-voxel predict fn ────
//   A two-layer MLP (3 → 32 → nUnits) trained with mini-batch SGD and a
//   cosine-annealed learning rate. Each BH layer contributes three training
//   samples (top, mid, base) so depth boundaries are well represented.
function trainNN(boreholes, geoUnits, bounds, epochs = 500) {
  const { minX, maxX, minY, maxY, botZ, topZ } = bounds;
  const rx = (maxX - minX) || 1;
  const ry = (maxY - minY) || 1;
  const rz = (topZ - botZ) || 1;
  const norm = (x, y, z) => [(x - minX) / rx, (y - minY) / ry, (z - botZ) / rz];

  const codeToIdx = {};
  geoUnits.forEach((u, i) => { codeToIdx[u.code] = i; });

  const samples = [];
  for (const bh of boreholes) {
    const gl = bh.groundLevel ?? 0;
    for (const l of bh.layers) {
      const idx = codeToIdx[l.unitCode];
      if (idx === undefined) continue;
      const pts = [l.top + 0.05, (l.top + l.base) * 0.5, l.base - 0.05];
      for (const d of pts) {
        samples.push({ inp: norm(bh.x, bh.y, gl - d), label: idx });
      }
    }
  }
  if (!samples.length) return null;

  const mlp = new TinyMLP(3, 32, geoUnits.length);

  for (let e = 0; e < epochs; e++) {
    // Fisher-Yates shuffle
    for (let i = samples.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [samples[i], samples[j]] = [samples[j], samples[i]];
    }
    // Cosine-annealed LR: 0.08 → 0.005
    const lr = 0.005 + 0.5 * (0.08 - 0.005) * (1 + Math.cos(Math.PI * e / epochs));
    for (const s of samples) mlp.step(s.inp, s.label, lr);
  }

  return (x, y, z) => {
    const { probs } = mlp.forward(norm(x, y, z));
    let best = 0, second = -1;
    for (let i = 1; i < probs.length; i++) {
      if (probs[i] > probs[best]) { second = best; best = i; }
      else if (second < 0 || probs[i] > probs[second]) second = i;
    }
    if (second < 0) second = best;
    return {
      unitId:      geoUnits[best]?.id   ?? 0,
      certainty:   Math.min(1, probs[best]),
      blendUnitId: geoUnits[second]?.id ?? geoUnits[best]?.id ?? 0,
      blendRatio:  probs[second],
    };
  };
}

// ── RBF (multiquadric) interpolation ─────────────────────────────────────────
// Fits a smooth indicator surface per unit using multiquadric RBF weights
// then takes the unit with the largest positive indicator value at query.
// More accurate than IDW for capturing smooth geological contacts.
function rbfVote(neighbours, qx, qy, unitIndex, unknownId, epsilon = null) {
  const n = neighbours.length;
  if (n < 2) return null;

  // Collect unique unit codes
  const codes = [...new Set(neighbours.map(nb => nb.unitCode))];
  if (!codes.length) return null;

  // RBF epsilon: if not provided, use mean inter-point distance
  if (epsilon === null) {
    let dSum = 0, dCnt = 0;
    for (let i = 0; i < n; i++)
      for (let j = i + 1; j < n; j++) {
        dSum += Math.hypot(neighbours[i].x - neighbours[j].x, neighbours[i].y - neighbours[j].y);
        dCnt++;
      }
    epsilon = dCnt > 0 ? dSum / dCnt : 1;
  }

  // Multiquadric RBF: φ(r) = sqrt(1 + (r/ε)²)
  const phi = (r) => Math.sqrt(1 + (r / epsilon) ** 2);

  // Build Gram matrix K[i][j] = φ(dist(i,j))
  const K = [];
  for (let i = 0; i < n; i++) {
    K[i] = new Array(n);
    for (let j = 0; j < n; j++) {
      const d = Math.hypot(neighbours[i].x - neighbours[j].x, neighbours[i].y - neighbours[j].y);
      K[i][j] = phi(d);
    }
  }

  // Evaluate each unit's indicator: 1 at points of that unit, 0 elsewhere
  const unitScores = {};
  let totalPositive = 0;

  for (const code of codes) {
    // RHS indicator: 1 if neighbour is this code, 0 otherwise
    const f = neighbours.map(nb => nb.unitCode === code ? nb.layerCert : 0);
    // Solve K*w = f (via simple Jacobi/direct for small n)
    let w;
    try { w = solveLinear(K.map(r => [...r]), f); } catch { continue; }
    // Evaluate at query point
    const kq = neighbours.map(nb => phi(Math.hypot(nb.x - qx, nb.y - qy)));
    const val = kq.reduce((s, k, i) => s + k * w[i], 0);
    unitScores[code] = Math.max(0, val);
    totalPositive += unitScores[code];
  }

  if (!totalPositive) return null;

  const sorted = Object.entries(unitScores).sort((a, b) => b[1] - a[1]);
  const [bc, bv] = sorted[0];
  const [sc, sv = 0] = sorted[1] ?? [];
  const cert = Math.min(1, (bv / totalPositive) * 0.8 + 0.1);
  return makeResult(bc, sc, bv / (bv + sv + 1e-9), totalPositive, sv, cert, unitIndex, unknownId);
}

// ── Nearest-BH surface fallback ───────────────────────────────────────────────
function nearestFallback(boreholes, x, y, unitIndex, unknownId) {
  const best = boreholes
    .filter(b => b.layers.length)
    .map(b => ({ d: Math.hypot(b.x - x, b.y - y), b }))
    .sort((a, b) => a.d - b.d)[0];
  const code = best?.b.layers[0].unitCode;
  const uid  = code ? (unitIndex[code] ?? unknownId) : unknownId;
  return { unitId: uid, certainty: 0.05, blendUnitId: uid, blendRatio: 0 };
}

// ── buildVoxelGrid ────────────────────────────────────────────────────────────
export async function buildVoxelGrid(boreholes, geoUnits, cellSizeParam, options = {}) {
  if (!boreholes.length) throw new Error('No borehole data to interpolate');

  const kNeighbors  = Math.max(1, options.kNeighbors ?? 5);
  const idwPower    = Math.max(0.5, options.idwPower ?? 2);
  const method      = options.method ?? 'idw';
  const trendOrder  = options.trendOrder ?? 1;
  const onProgress  = options.onProgress ?? null;
  const stratRanks  = buildStratRankMap(options.stratOrder ?? []);
  // Anisotropy: anisoAzimuth = strike direction (degrees from North), anisoRatio > 1 = elongated along strike
  const anisoAz    = ((options.anisoAzimuth ?? 0) * Math.PI) / 180;
  const anisoRatio = Math.max(1, options.anisoRatio ?? 1);
  const anisoSinAz = Math.sin(anisoAz);
  const anisoCosAz = Math.cos(anisoAz);

  const semanticModel  = options.semanticModel ?? null;
  const semanticWeight = Math.max(0, Math.min(1, options.semanticWeight ?? 0.3));
  const topoPoints     = options.topoPoints ?? null;

  // Pre-compute deviation trajectories (minimum curvature)
  for (const bh of boreholes) {
    bh._trajectory = (bh.deviation?.length >= 2) ? computeTrajectory(bh) : null;
  }

  // ── 1. Bounding box ────────────────────────────────────────────────────────
  const xs  = boreholes.map(b => b.x);
  const ys  = boreholes.map(b => b.y);
  const gls = boreholes.map(b => b.groundLevel ?? 0);
  const maxDepths = boreholes.map(b =>
    b.depth ?? (b.layers.length ? Math.max(...b.layers.map(l => l.base)) : 10));

  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const maxGL = Math.max(...gls);
  const maxDep = Math.max(...maxDepths);
  const topZ = maxGL, botZ = maxGL - maxDep;

  const siteDiag       = Math.hypot(maxX - minX + 1, maxY - minY + 1);
  const typicalSpacing = siteDiag / Math.sqrt(boreholes.length);

  const marginX = Math.max((maxX - minX) * 0.15, cellSizeParam * 2);
  const marginY = Math.max((maxY - minY) * 0.15, cellSizeParam * 2);
  const ox = minX - marginX, oy = minY - marginY, oz = botZ;

  // Add synthetic anchor boreholes from semantic model
  const allBoreholes = [...boreholes];
  if (semanticModel?.synthetic_anchors?.length) {
    const bboxW = maxX - minX || 100, bboxH = maxY - minY || 100;
    for (const anchor of semanticModel.synthetic_anchors) {
      const ax = minX + (anchor.x_frac ?? 0.5) * bboxW;
      const ay = minY + (anchor.y_frac ?? 0.5) * bboxH;
      if (!isFinite(ax) || !isFinite(ay)) continue;
      allBoreholes.push({
        id: `SYN-${anchor.label ?? 'AI'}`,
        x: ax, y: ay,
        groundLevel: maxGL,
        depth: 20,
        synthetic: true,
        _trajectory: null,
        layers: (anchor.layers ?? []).map(l => ({
          top: l.top ?? 0, base: l.base ?? 1,
          unitCode: l.unit_code,
          certainty: (l.certainty ?? 0.5) * (1 - semanticWeight * 0.5),
          description: '',
        })),
      });
    }
    if (semanticModel.synthetic_anchors.length) {
      log(`Semantic model: added ${semanticModel.synthetic_anchors.length} synthetic anchor point(s)`, 'info');
    }
  }

  // Depth exclusion priors from semantic model (penalise unlikely depth assignments)
  const depthExclusions = semanticModel?.depth_exclusions ?? [];

  // Transition matrix (from real boreholes only)
  const transMatrix = buildTransitionMatrix(boreholes, geoUnits);
  const unitCodeToIdx = {};
  geoUnits.forEach((u, i) => { unitCodeToIdx[u.code] = i; });

  // ── 2. Grid dimensions ─────────────────────────────────────────────────────
  const MAX_VOXELS = 500_000;
  const userCellH = options.cellSizeZ;
  let cellSize = cellSizeParam, cellH = userCellH ?? cellSize / 5;
  let nx = Math.ceil((maxX + marginX - ox) / cellSize);
  let ny = Math.ceil((maxY + marginY - oy) / cellSize);
  let nz = Math.ceil((topZ - botZ) / cellH);

  while (nx * ny * nz > MAX_VOXELS) {
    cellSize += 1;
    if (!userCellH) cellH = cellSize / 5;
    nx = Math.ceil((maxX + marginX - ox) / cellSize);
    ny = Math.ceil((maxY + marginY - oy) / cellSize);
    nz = Math.ceil((topZ - botZ) / cellH);
    log(`Cell size auto-increased to ${cellSize} m (>500 K voxel cap)`, 'warn');
  }

  const methodLabel = {
    idw: 'IDW', kriging: 'Kriging', gp: 'Gauss. Process', nn: 'Neural Net',
    uk: `UK (order ${trendOrder})`, 'neural-implicit': 'Neural Implicit Field',
    rbf: 'RBF (multiquadric)',
  }[method] ?? method;
  log(`Grid ${nx}×${ny}×${nz} = ${(nx * ny * nz).toLocaleString()} voxels @ ${cellSize} m | ${methodLabel} K=${kNeighbors}`, 'info');

  const total       = nx * ny * nz;
  const unitIds      = new Uint8Array(total);
  const certainty    = new Float32Array(total);
  const blendUnitIds = new Uint8Array(total);
  const blendRatios  = new Float32Array(total);

  const unitIndex = {};
  geoUnits.forEach(u => { unitIndex[u.code] = u.id; });
  const unknownId = geoUnits.find(u => u.code === 'UNKN')?.id ?? 0;

  // Geostatistical parameters — use auto-fitted variogram if available
  const range = options.varRange  ?? typicalSpacing * 1.5;
  const sill  = options.varSill   ?? 1.0;
  const gpLen = options.varRange  ? options.varRange * 0.6 : typicalSpacing * 0.8;
  if (options.varRange) {
    log(`Using fitted variogram: range=${range.toFixed(1)}m sill=${sill.toFixed(3)} nugget=${(options.varNugget ?? 0).toFixed(3)}`, 'info');
  }

  // Per-unit geometry descriptors: corrLength, anisoRatio, anisoAzimuth
  const unitGeomMap = {};
  for (const u of geoUnits) {
    if (u.geom && (u.geom.corrLength != null || u.geom.anisoRatio != null || u.geom.anisoAzimuth != null)) {
      unitGeomMap[u.code] = u.geom;
    }
  }
  const hasPerUnitGeom = Object.keys(unitGeomMap).length > 0;
  if (hasPerUnitGeom) {
    log(`Per-unit geometry active for: ${Object.keys(unitGeomMap).join(', ')}`, 'info');
  }
  if (topoPoints?.length) {
    log(`Topographic masking: ${topoPoints.length} surface point(s) — voxels above ground will be hidden`, 'info');
  }

  // ── Neural Implicit Geological Field ──────────────────────────────────────
  if (method === 'neural-implicit') {
    const conceptStore = options.conceptStore ?? null;
    if (conceptStore && !conceptStore.isEmpty) {
      log(`Concept store: ${conceptStore.concepts.length} geological concept(s) active — coordinate warping enabled`, 'info');
    }

    log(`Training neural implicit field (${options.niEpochs ?? 600} epochs)…`, 'info');
    if (onProgress) onProgress(0.02);

    const trainedModel = await trainGeoImplicit(
      allBoreholes, geoUnits, conceptStore,
      {
        epochs:          options.niEpochs ?? 600,
        lr:              0.01,
        lrMin:           0.001,
        l2:              0.001,
        samplesPerLayer: 8,
        stratOrder:      options.stratOrder ?? null,
        onProgress: (frac, loss, info) => {
          if (onProgress) onProgress(0.02 + frac * 0.7, loss, info);
          if (frac < 1 && info?.epoch != null) log(`  …epoch ${info.epoch} loss=${loss?.toFixed(4) ?? '–'}`, 'info');
          if (info?.stratContactsFound) log(`  Stratigraphic contacts used for training: ${info.stratContactsFound}`, 'info');
        },
      },
    );

    if (!trainedModel) {
      log('Neural implicit training failed (no samples) — falling back to IDW', 'warn');
      // Fall through to IDW below
    } else {
      log('Inferring voxel grid from neural implicit field…', 'info');
      const nMCPasses = options.nMCPasses ?? 8;
      if (nMCPasses > 1) log(`MC uncertainty: ${nMCPasses} passes`, 'info');
      const gridMeta = { nx, ny, nz, cellSize, cellHeight: cellH, origin: { x: ox, y: oz, z: oy } };
      const inferred = inferGeoImplicit(trainedModel, gridMeta, geoUnits, conceptStore, { nMCPasses });
      unitIds.set(inferred.unitIds);
      certainty.set(inferred.certainty);
      blendUnitIds.set(inferred.blendUnitIds);
      blendRatios.set(inferred.blendRatios);
      const conceptInfluence = inferred.conceptInfluence ?? null;
      const probVolumes      = inferred.probVolumes ?? null;
      const sharpnessT       = inferred.sharpnessT ?? null;

      // ── Concept-driven iterative refinement ──────────────────────────────────
      // After the first inference pass, identify voxel columns where:
      //   (a) concept influence is active (semantics are driving prediction)
      //   (b) borehole coverage is sparse (no nearby real data to anchor it)
      //   (c) certainty is low (model is genuinely uncertain)
      // Inject virtual "expectation" observations at those positions using the
      // concept store's predicted unit profile, then fine-tune the network for
      // a short pass to strengthen the concept-anchored regions.
      if (conceptStore && !conceptStore.isEmpty && options.conceptRefinement !== false) {
        const refineSamples = [];
        const realBHs = allBoreholes.filter(b => !b.synthetic && b.layers?.length);
        const bhSigmaSq = (() => {
          if (realBHs.length < 2) return (Math.max(nx, ny) * cellSize * 0.3) ** 2;
          let totalNN = 0;
          for (const a of realBHs) {
            let minD = Infinity;
            for (const b of realBHs) { if (a !== b) { const d = Math.hypot(a.x - b.x, a.y - b.y); if (d < minD) minD = d; } }
            totalNN += minD;
          }
          return ((totalNN / realBHs.length) * 1.5) ** 2;
        })();

        // Scan columns at coarse spacing (every 3rd cell) to find refinement targets
        const stride = 3;
        const targets = [];
        for (let ciy = 0; ciy < ny; ciy += stride) {
          for (let cix = 0; cix < nx; cix += stride) {
            const wx = ox + (cix + 0.5) * cellSize;
            const wy = oy + (ciy + 0.5) * cellSize;
            // Borehole coverage density at this XY
            const cov = realBHs.length ? realBHs.reduce((s, b) => {
              const d2 = (b.x - wx) ** 2 + (b.y - wy) ** 2;
              return s + Math.exp(-d2 / bhSigmaSq);
            }, 0) / Math.max(1, realBHs.length) : 0;
            if (cov > 0.4) continue; // well-constrained by boreholes — skip
            // Average certainty and concept influence in this column
            let sumCert = 0, sumCInf = 0, cnt = 0;
            for (let ciz = 0; ciz < nz; ciz++) {
              const f = cix + ciy * nx + ciz * nx * ny;
              if (!inferred.unitIds[f]) continue;
              sumCert  += inferred.certainty[f];
              sumCInf  += conceptInfluence ? conceptInfluence[f] : 0;
              cnt++;
            }
            if (!cnt) continue;
            const avgCert = sumCert / cnt;
            const avgCInf = sumCInf / cnt;
            if (avgCert > 0.55) continue;    // already confident — skip
            if (avgCInf < 0.2) continue;     // concept not driving this area — skip
            targets.push({ cix, ciy, wx, wy, avgCert, avgCInf });
          }
        }

        // Sort by (low certainty × high concept influence) and take top 20
        targets.sort((a, b) => (b.avgCInf * (1 - b.avgCert)) - (a.avgCInf * (1 - a.avgCert)));
        const topTargets = targets.slice(0, 20);

        for (const { wx, wy } of topTargets) {
          // Sample 3 depths in the model column
          for (let zi = 0; zi < 3; zi++) {
            const wz = oz + cellH * (nz * (zi + 0.5) / 3);
            const ctx = conceptStore.computeAt(wx, wy, wz);
            if (ctx.totalWeight < 0.15 || !ctx.weights.length) continue;
            // Predict unit from concept: use dominant unit affinity of top concept
            const topConcept = conceptStore.concepts.find(c => c.id === ctx.weights[0]?.id);
            if (!topConcept) continue;
            const affinityUnits = topConcept.unitAffinity?.length
              ? topConcept.unitAffinity
              : geoUnits.map(u => u.code);
            // Pick the unit that has highest expected probability at this depth
            // (favour units whose affinity matches the concept, weighted by vertical position)
            const midUnit = affinityUnits[Math.floor(zi / 3 * affinityUnits.length)] ?? affinityUnits[0];
            if (!midUnit) continue;
            // Weight = concept influence × (1 - certainty), capped low to stay soft constraint
            refineSamples.push({ x: wx, y: wy, z: wz, unitCode: midUnit, weight: 0.12 });
          }
        }

        if (refineSamples.length > 0) {
          log(`Concept refinement: fine-tuning on ${refineSamples.length} virtual samples in ${topTargets.length} concept-driven uncertain zone(s)…`, 'info');
          if (onProgress) onProgress(0.74);
          await finetuneGeoImplicit(trainedModel, geoUnits, refineSamples, {
            epochs: 80,
            lr: 0.002,
            onProgress: (frac) => { if (onProgress) onProgress(0.74 + frac * 0.04); },
          });
          // Re-infer after refinement
          const inferred2 = inferGeoImplicit(trainedModel, gridMeta, geoUnits, conceptStore, { nMCPasses });
          unitIds.set(inferred2.unitIds);
          certainty.set(inferred2.certainty);
          blendUnitIds.set(inferred2.blendUnitIds);
          blendRatios.set(inferred2.blendRatios);
          // Update inferred reference for downstream certainty calibration
          Object.assign(inferred, inferred2);
          log(`Concept refinement complete — re-inferred ${nx * ny * nz} voxels`, 'info');
        }
      }

      // Oracle refinement: find uncertain clusters and pass to injected oracle fn
      const oracleFn = options.oracleRefineFn;
      if (oracleFn && options.oracleApiKey) {
        log('Running LLM oracle on uncertain regions…', 'info');
        if (onProgress) onProgress(0.78);
        const clusters = findUncertainClusters(certainty, gridMeta, 0.45, 12);
        log(`Oracle: found ${clusters.length} uncertain cluster(s)`, 'info');
        if (clusters.length) {
          const patches = await oracleFn(clusters, geoUnits, options.oracleApiKey, options.demoMode);
          patchWithOracle(unitIds, certainty, blendUnitIds, blendRatios, patches, geoUnits);
          log(`Oracle: applied ${patches.length} probability patch(es)`, 'info');
        }
      }

      // ── Borehole coverage density ─────────────────────────────────────────
      // For each (ix,iy) column compute how well-constrained by real borehole
      // data this horizontal position is: sum of exp(-dist²/σ²) for each BH.
      // σ = mean inter-borehole spacing (or site span / √N if <3 BHs).
      // Stored as a 3D Float32Array (same for all iz) for uniform access in
      // the voxel builder and traceability panel.
      const coverageDensity = (() => {
        const realBHs = allBoreholes.filter(b => !b.synthetic && b.layers?.length);
        if (!realBHs.length) return null;
        // Estimate σ from mean nearest-neighbour distance
        let sigmaSq = 0;
        if (realBHs.length >= 2) {
          let totalNN = 0;
          for (const a of realBHs) {
            let minD = Infinity;
            for (const b of realBHs) {
              if (a === b) continue;
              const d = Math.hypot(a.x - b.x, a.y - b.y);
              if (d < minD) minD = d;
            }
            totalNN += minD;
          }
          const meanNN = totalNN / realBHs.length;
          sigmaSq = meanNN * meanNN * 1.5; // 1.5× mean NN gives smooth falloff
        } else {
          // Single borehole — use quarter of site span
          const span = Math.max(nx * cellSize, ny * cellSize) * 0.25;
          sigmaSq = span * span;
        }
        sigmaSq = Math.max(sigmaSq, 1); // avoid zero sigma

        const arr = new Float32Array(nx * ny * nz);
        let maxVal = 0;
        // Compute 2D density first (only varies horizontally)
        const density2d = new Float32Array(nx * ny);
        for (let iy = 0; iy < ny; iy++) {
          const worldY = oy + iy * cellSize + cellSize * 0.5;
          for (let ix = 0; ix < nx; ix++) {
            const worldX = ox + ix * cellSize + cellSize * 0.5;
            let sum = 0;
            for (const bh of realBHs) {
              const dx = bh.x - worldX, dy = bh.y - worldY;
              sum += Math.exp(-(dx * dx + dy * dy) / sigmaSq);
            }
            density2d[ix + iy * nx] = sum;
            if (sum > maxVal) maxVal = sum;
          }
        }
        // Normalise and replicate vertically
        if (maxVal > 0) {
          for (let iz = 0; iz < nz; iz++) {
            const base = iz * nx * ny;
            for (let i = 0; i < nx * ny; i++) arr[base + i] = density2d[i] / maxVal;
          }
        }
        return arr;
      })();

      // ── Concept-aware certainty calibration ──────────────────────────────────
      // In data-sparse areas where the model is driven by concept embeddings rather
      // than borehole observations, we deflate certainty to prevent the model from
      // appearing artificially confident about unvalidated semantic interpretations.
      // The penalty scales with conceptInfluence and inversely with borehole coverage:
      //   penalty = conceptInfluence × (1 − coverage) × 0.22
      // This keeps certainty high near boreholes (where FiLM conditioning is validated)
      // and appropriately uncertain in pure-concept regions.
      if (conceptInfluence && coverageDensity) {
        const total3D = nx * ny * nz;
        for (let idx = 0; idx < total3D; idx++) {
          const inf = conceptInfluence[idx];
          const cov = coverageDensity[idx];
          if (inf > 0.25 && cov < 0.35) {
            const penalty = inf * (1 - cov) * 0.22;
            certainty[idx] = Math.max(0.05, certainty[idx] - penalty);
          }
        }
      }

      // Topographic masking (neural path): mask voxels above ground surface
      // Pre-compute per-column topo elevation to avoid repeated IDW for each iz layer.
      if (topoPoints?.length) {
        const colTopoElev = new Float32Array(nx * ny);
        for (let iy = 0; iy < ny; iy++) {
          const wy = oy + iy * cellSize + cellSize * 0.5;
          for (let ix = 0; ix < nx; ix++) {
            const wx = ox + ix * cellSize + cellSize * 0.5;
            colTopoElev[ix + iy * nx] = _topoElevAt(wx, wy, topoPoints);
          }
        }
        for (let iz = 0; iz < nz; iz++) {
          const wz = oz + iz * cellH + cellH * 0.5;
          const base = iz * nx * ny;
          for (let iy = 0; iy < ny; iy++) {
            for (let ix = 0; ix < nx; ix++) {
              if (wz > colTopoElev[ix + iy * nx]) {
                unitIds[base + ix + iy * nx]   = 0;
                certainty[base + ix + iy * nx] = 0;
              }
            }
          }
        }
      }

      if (onProgress) onProgress(1);
      return {
        nx, ny, nz,
        cellSize, cellHeight: cellH,
        origin: { x: ox, y: oz, z: oy },
        worldWidth:  nx * cellSize,
        worldHeight: nz * cellH,
        worldDepth:  ny * cellSize,
        unitIds, certainty, blendUnitIds, blendRatios,
        // trainedModel exposed so callers can re-infer with different concept stores
        // (concept ensemble uncertainty) without re-training the network.
        trainedModel,
        ...(conceptInfluence ? { conceptInfluence } : {}),
        ...(coverageDensity   ? { coverageDensity }  : {}),
        ...(probVolumes       ? { probVolumes }       : {}),
        ...(sharpnessT        ? { sharpnessT }        : {}),
      };
    }
  }

  // Neural network: train once before the main voxel loop (classic tiny MLP)
  let nnPredict = null;
  if (method === 'nn') {
    log('Training neural network on borehole data…', 'info');
    nnPredict = trainNN(allBoreholes, geoUnits,
      { minX, maxX, minY, maxY, botZ, topZ });
    if (!nnPredict) log('NN training yielded no samples — falling back to IDW', 'warn');
  }

  // ── Concept-steered anisotropy for IDW/kriging ────────────────────────────────
  // When a conceptStore is active, compute per-column local anisotropy from the
  // concept embedding (axes 3=EW, 4=NS elongation) and use it to steer the
  // search ellipsoid in getCandidates(). This extends concept influence to all
  // interpolation methods, not just neural-implicit.
  // Cache is keyed by (ix, iy) column since concept relevance is horizontal-only.
  const conceptStore = options.conceptStore ?? null;
  const conceptAnisoCache = new Map(); // key = ix*ny+iy → {sinAz, cosAz, ratio}
  function _conceptAniso(ix, iy, worldX, worldY) {
    const key = ix * ny + iy;
    let cached = conceptAnisoCache.get(key);
    if (cached !== undefined) return cached;
    if (!conceptStore || conceptStore.isEmpty) {
      cached = null;
      conceptAnisoCache.set(key, cached);
      return null;
    }
    const ctx = conceptStore.computeAt(worldX, worldY, 0);
    if (ctx.totalWeight < 0.1) {
      cached = null; conceptAnisoCache.set(key, cached); return null;
    }
    const ew = ctx.vec[3] ?? 0;  // east_west_elongation
    const ns = ctx.vec[4] ?? 0;  // north_south_elongation
    const maxAx = Math.max(Math.abs(ew), Math.abs(ns));
    if (maxAx < 0.2) {
      cached = null; conceptAnisoCache.set(key, cached); return null;
    }
    // Derive azimuth: direction of maximum elongation
    // ew > ns → body elongated E-W → azimuth = 90° (East)
    // ns > ew → body elongated N-S → azimuth = 0° (North)
    const az = Math.atan2(ew, ns); // angle from North clockwise
    const ratio = Math.max(1, Math.min(6, Math.exp(maxAx * 1.2)));
    cached = { sinAz: Math.sin(az), cosAz: Math.cos(az), ratio };
    conceptAnisoCache.set(key, cached);
    return cached;
  }

  // Pre-compute per-column topographic elevation for efficiency (avoids O(nx*ny*nz) IDW calls)
  const colTopoElev = topoPoints?.length ? (() => {
    const arr = new Float32Array(nx * ny);
    for (let iy = 0; iy < ny; iy++) {
      const wy = oy + iy * cellSize + cellSize * 0.5;
      for (let ix = 0; ix < nx; ix++) {
        arr[ix + iy * nx] = _topoElevAt(ox + ix * cellSize + cellSize * 0.5, wy, topoPoints);
      }
    }
    return arr;
  })() : null;

  // ── 3. Classify every voxel (top-down so transition matrix can look upward) ─
  for (let iz = nz - 1; iz >= 0; iz--) {
    const z = oz + iz * cellH + cellH * 0.5;
    if (onProgress && iz % 3 === 0) {
      onProgress((nz - 1 - iz) / nz);
      await new Promise(r => setTimeout(r, 0));
    }
    for (let iy = 0; iy < ny; iy++) {
      const y = oy + iy * cellSize + cellSize * 0.5;
      for (let ix = 0; ix < nx; ix++) {
        const x   = ox + ix * cellSize + cellSize * 0.5;
        const idx = ix + iy * nx + iz * nx * ny;

        let result;

        if (method === 'nn' && nnPredict) {
          result = nnPredict(x, y, z);
        } else {
          // Apply concept-steered anisotropy if available, else fall back to global setting
          const cAniso = _conceptAniso(ix, iy, x, y);
          const effSinAz = cAniso ? cAniso.sinAz * (1 - anisoRatio / 10) + anisoSinAz * (anisoRatio / 10)
                                  : anisoSinAz;
          const effCosAz = cAniso ? cAniso.cosAz * (1 - anisoRatio / 10) + anisoCosAz * (anisoRatio / 10)
                                  : anisoCosAz;
          const effRatio = cAniso ? Math.max(anisoRatio, cAniso.ratio) : anisoRatio;
          let cands = getCandidates(allBoreholes, x, y, z, effSinAz, effCosAz, effRatio);
          // Fault plane filtering: restrict candidates to same side of each fault as the voxel
          if (options.faultPlanes?.length) {
            cands = cands.filter(c => options.faultPlanes.every(f => {
              const vSide = (x - f.px) * f.fnx + (y - f.py) * f.fny >= 0;
              const bSide = (c.x - f.px) * f.fnx + (c.y - f.py) * f.fny >= 0;
              return vSide === bSide;
            }));
          }
          cands.sort((a, b) => a.dist - b.dist);
          const nb = cands.slice(0, kNeighbors);

          if (!nb.length) {
            result = nearestFallback(allBoreholes, x, y, unitIndex, unknownId);
          } else {
            // Per-unit corrLength: IDW-weighted mean of each candidate's unit corrLength
            let effRange = range, effGpLen = gpLen;
            if (hasPerUnitGeom) {
              let wSum = 0, wRange = 0;
              for (const n of nb) {
                const w = n.layerCert / (n.dist * n.dist + 1e-6);
                const r = unitGeomMap[n.unitCode]?.corrLength ?? range;
                wRange += w * r; wSum += w;
              }
              if (wSum > 0) { effRange = wRange / wSum; effGpLen = effRange * 0.6; }
            }

            if (method === 'kriging') {
              const kAniso = (effSinAz || effCosAz || effRatio !== 1)
                ? { sinAz: effSinAz, cosAz: effCosAz, ratio: effRatio } : null;
              result = krigingVote(nb, x, y, unitIndex, unknownId, effRange, sill, options.varNugget ?? null, kAniso)
                    ?? idwVote(nb, idwPower, unitIndex, unknownId, typicalSpacing);
            } else if (method === 'uk') {
              const ukAniso = (effSinAz || effCosAz || effRatio !== 1)
                ? { sinAz: effSinAz, cosAz: effCosAz, ratio: effRatio } : null;
              result = ukVote(nb, x, y, unitIndex, unknownId, effRange, sill, trendOrder, ukAniso)
                    ?? idwVote(nb, idwPower, unitIndex, unknownId, typicalSpacing);
            } else if (method === 'gp') {
              result = gpVote(nb, x, y, unitIndex, unknownId, effGpLen)
                    ?? idwVote(nb, idwPower, unitIndex, unknownId, typicalSpacing);
            } else if (method === 'rbf') {
              const effEps = hasPerUnitGeom ? effRange * 0.4 : null;
              result = rbfVote(nb, x, y, unitIndex, unknownId, effEps)
                    ?? idwVote(nb, idwPower, unitIndex, unknownId, typicalSpacing);
            } else {
              result = idwVote(nb, idwPower, unitIndex, unknownId, typicalSpacing);
            }
          }
        }

        let cert = result.certainty;
        if (stratRanks.size && result.unitId) {
          const winCode = geoUnits.find(u => u.id === result.unitId)?.code ?? '';
          const depth   = topZ - z; // depth below max ground level
          const penalty = stratigraphicConsistencyPenalty(
            winCode, depth, x, y, boreholes, stratRanks, typicalSpacing * 2,
          );
          cert *= penalty;
        }

        // Transition probability prior (Markov chain top-down)
        if (iz < nz - 1 && semanticWeight > 0) {
          const aboveUid  = unitIds[ix + iy * nx + (iz + 1) * nx * ny];
          const aboveCode = aboveUid ? geoUnits.find(u => u.id === aboveUid)?.code : null;
          if (aboveCode) {
            const fromI = unitCodeToIdx[aboveCode];
            const toI   = unitCodeToIdx[geoUnits.find(u => u.id === result.unitId)?.code];
            if (fromI !== undefined && toI !== undefined) {
              const transProb   = transMatrix[fromI][toI];
              const uniformProb = 1 / geoUnits.length;
              // Scale: above-average transition boosts cert; below-average penalises
              const transScale = 0.5 + (transProb / uniformProb) * 0.5;
              cert = Math.min(1, cert * (1 - semanticWeight + semanticWeight * Math.min(transScale, 2)));
            }
          }
        }

        // Depth exclusion priors from semantic model
        if (semanticWeight > 0) {
          const vDepth = maxGL - (oz + iz * cellH + cellH * 0.5);
          const winCode = geoUnits.find(u => u.id === result.unitId)?.code ?? '';
          for (const ex of depthExclusions) {
            if (ex.unit_code !== winCode) continue;
            const tooShallow = ex.exclude_above_m != null && vDepth < ex.exclude_above_m;
            const tooDeep    = ex.exclude_below_m != null && vDepth > ex.exclude_below_m;
            if (tooShallow || tooDeep) {
              cert = Math.min(1, cert * (1 - semanticWeight * (ex.confidence ?? 0.5)));
            }
          }
        }

        // Topographic masking: voxels whose centre is above the ground surface are air
        if (colTopoElev && z > colTopoElev[ix + iy * nx]) {
          certainty[idx] = 0;
          continue;
        }

        unitIds[idx]      = result.unitId;
        certainty[idx]    = Math.max(0.05, cert);
        blendUnitIds[idx] = result.blendUnitId;
        blendRatios[idx]  = result.blendRatio;
      }
    }
  }

  if (onProgress) onProgress(1);
  const idwBase = {
    nx, ny, nz, cellSize, cellHeight: cellH,
    origin: { x: ox, y: oz, z: oy },
    worldWidth: nx * cellSize, worldHeight: nz * cellH, worldDepth: ny * cellSize,
  };
  _applyTopoMask(unitIds, certainty, idwBase, topoPoints);
  return { ...idwBase, unitIds, certainty, blendUnitIds, blendRatios };
}

// ── Monte Carlo Uncertainty Quantification ───────────────────────────────────
// Runs N realisations of the IDW model with perturbed layer boundaries
// Returns { unitIds, certainty, blendUnitIds, blendRatios } — certainty is
// now the fraction of realisations that agreed with the majority vote.
export async function buildVoxelGridMonteCarlo(boreholes, geoUnits, cellSizeH, options = {}) {
  const N = options.nRealisations ?? 20;
  const perturbSigma = options.perturbSigmaM ?? 0.5; // boundary perturbation std dev (m)
  const onProgress = options.onProgress;

  // Run N realisations with perturbed layer boundaries
  log(`Monte Carlo: running ${N} realisations (σ=${perturbSigma}m boundary perturbation)…`, 'info');

  // Build first realisation normally to get grid dimensions
  const base = await buildVoxelGrid(boreholes, geoUnits, cellSizeH, {
    ...options, method: 'idw', onProgress: null,
  });
  const { nx, ny, nz } = base;
  const total = nx * ny * nz;
  const nUnits = geoUnits.length + 1;

  // Vote accumulators: per voxel, per unit index, count of realisations
  const votes = new Array(nUnits).fill(null).map(() => new Float32Array(total));

  // First realisation
  for (let i = 0; i < total; i++) {
    const uid = base.unitIds[i];
    if (uid < nUnits) votes[uid][i]++;
  }

  // Subsequent realisations with perturbed boundaries
  for (let r = 1; r < N; r++) {
    const perturbed = boreholes.map(bh => {
      const newLayers = bh.layers.map((l, li) => {
        const dTop  = li === 0 ? 0 : (Math.random() - 0.5) * 2 * perturbSigma;
        const dBase = (Math.random() - 0.5) * 2 * perturbSigma;
        return {
          ...l,
          top:  Math.max(0, l.top  + dTop),
          base: Math.max(l.top + 0.01, l.base + dBase),
        };
      });
      return { ...bh, layers: newLayers };
    });

    const grid = await buildVoxelGrid(perturbed, geoUnits, cellSizeH, {
      ...options, method: 'idw', onProgress: null,
    });

    for (let i = 0; i < total; i++) {
      const uid = grid.unitIds[i];
      if (uid < nUnits) votes[uid][i]++;
    }

    if (onProgress) onProgress(r / N);
    await new Promise(res => setTimeout(res, 0));
  }

  // Extract majority vote and certainty (= fraction of realisations in agreement)
  const unitIds      = new Uint8Array(total);
  const certainty    = new Float32Array(total);
  const blendUnitIds = new Uint8Array(total);
  const blendRatios  = new Float32Array(total);

  for (let i = 0; i < total; i++) {
    let best1 = 0, best2 = 0;
    let cnt1  = -1, cnt2  = -1;
    for (let u = 0; u < nUnits; u++) {
      const v = votes[u][i];
      if      (v > cnt1) { cnt2 = cnt1; best2 = best1; cnt1 = v; best1 = u; }
      else if (v > cnt2) { cnt2 = v;    best2 = u; }
    }
    unitIds[i]      = best1;
    certainty[i]    = Math.max(0.05, cnt1 / N);
    blendUnitIds[i] = best2;
    blendRatios[i]  = cnt2 / N;
  }

  if (onProgress) onProgress(1);
  log(`Monte Carlo complete — mean certainty ${(Array.from(certainty).reduce((a,b)=>a+b,0)/total*100).toFixed(0)}%`, 'ok');

  _applyTopoMask(unitIds, certainty, base, options.topoPoints);
  return { ...base, unitIds, certainty, blendUnitIds, blendRatios };
}

// ── Indicator Kriging — per-unit probability volumes ──────────────────────────
// For each geological unit u, kriging is applied to the indicator function
//   I_u(x,y,z) = 1 if the point belongs to unit u, 0 otherwise.
// The result is nUnits Float32Arrays of shape [nx*ny*nz], each holding P(unit=u).
// Rows are clipped to [0,1] and row-normalised so probabilities sum to 1 per voxel.
// This is the canonical Leapfrog / SGeMS uncertainty-quantification approach.
//
// Returns: { unitIds, certainty, blendUnitIds, blendRatios, probVolumes }
// probVolumes: Map<unitCode, Float32Array(nx*ny*nz)>
export async function buildIndicatorKriging(boreholes, geoUnits, cellSizeParam, options = {}) {
  if (!boreholes.length) throw new Error('No borehole data');

  const onProgress = options.onProgress ?? null;
  const kNeighbors = Math.max(2, options.kNeighbors ?? 6);

  // Pre-compute deviation trajectories
  for (const bh of boreholes) {
    bh._trajectory = (bh.deviation?.length >= 2) ? computeTrajectory(bh) : null;
  }

  // ── Bounding box + grid ──────────────────────────────────────────────────────
  const xs  = boreholes.map(b => b.x);
  const ys  = boreholes.map(b => b.y);
  const gls = boreholes.map(b => b.groundLevel ?? 0);
  const maxDepths = boreholes.map(b =>
    b.depth ?? (b.layers.length ? Math.max(...b.layers.map(l => l.base)) : 10));

  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const maxGL = Math.max(...gls);
  const maxDep = Math.max(...maxDepths);

  const marginX = Math.max((maxX - minX) * 0.15, cellSizeParam * 2);
  const marginY = Math.max((maxY - minY) * 0.15, cellSizeParam * 2);
  const ox = minX - marginX, oy = minY - marginY, oz = maxGL - maxDep;

  const rawW = maxX - minX + 2 * marginX;
  const rawD = maxY - minY + 2 * marginY;
  const rawH = maxDep;
  const cellSize = cellSizeParam;
  const cellH    = cellSizeParam * (options.verticalExaggeration ?? 1.0);

  const nx = Math.max(2, Math.ceil(rawW / cellSize));
  const ny = Math.max(2, Math.ceil(rawD / cellSize));
  const nz = Math.max(2, Math.ceil(rawH / cellH));
  const total = nx * ny * nz;

  // ── Build sample set: one entry per borehole layer mid-point ─────────────────
  const samples = [];
  for (const bh of boreholes) {
    const gl = bh.groundLevel ?? 0;
    for (const layer of bh.layers) {
      if (!layer.unitCode) continue;
      const midDepth = (layer.top + layer.base) / 2;
      const wz = gl - midDepth;
      const { dx, dy } = getDeviatedXY(bh, midDepth);
      samples.push({ x: bh.x + dx, y: bh.y + dy, z: wz, unitCode: layer.unitCode, cert: layer.certainty ?? 0.9 });
    }
  }
  if (!samples.length) throw new Error('No layer samples for indicator kriging');

  // ── Per-unit variogram parameters (auto-fit from sample spacing) ──────────────
  const siteDiag = Math.hypot(maxX - minX + 1, maxY - minY + 1);
  const autoRange = siteDiag * 0.5;
  const autoSill  = 0.25; // variance of Bernoulli(0.5)
  const nugget    = options.nugget ?? autoSill * 0.05;
  const sill      = options.sill   ?? autoSill;
  const range     = options.range  ?? autoRange;

  // ── Probability accumulators — one Float32Array per unit ──────────────────────
  const unitCodes = geoUnits.map(u => u.code);
  const nUnits    = unitCodes.length;
  const codeIdx   = {};
  unitCodes.forEach((c, i) => { codeIdx[c] = i; });

  // Flat [total * nUnits] accumulator for P(u|voxel) — stored interleaved per voxel
  const probs = new Float32Array(total * nUnits);

  // Simple 3-D kd-style nearest-sample lookup using sorted index arrays
  // (full kd-tree is overkill for typical <500 borehole sample counts)
  const sxArr = Float64Array.from(samples.map(s => s.x));
  const syArr = Float64Array.from(samples.map(s => s.y));
  const szArr = Float64Array.from(samples.map(s => s.z));

  function kNearest(qx, qy, qz, k) {
    const dists = [];
    for (let i = 0; i < samples.length; i++) {
      const dx = sxArr[i] - qx, dy = syArr[i] - qy, dz = szArr[i] - qz;
      dists.push([dx * dx + dy * dy + dz * dz * 4, i]); // 2× vertical stretch
    }
    dists.sort((a, b) => a[0] - b[0]);
    return dists.slice(0, k).map(([d2, i]) => ({ ...samples[i], dist: Math.sqrt(d2) }));
  }

  // ── Kriging per voxel ──────────────────────────────────────────────────────────
  const batchSize = nx * ny; // one Z-layer per progress tick
  for (let iz = 0; iz < nz; iz++) {
    const wz = oz + iz * cellH + cellH * 0.5;
    for (let iy = 0; iy < ny; iy++) {
      const wy = oy + iy * cellSize + cellSize * 0.5;
      for (let ix = 0; ix < nx; ix++) {
        const wx = ox + ix * cellSize + cellSize * 0.5;
        const vIdx = (ix + iy * nx + iz * nx * ny) * nUnits;

        const nbrs = kNearest(wx, wy, wz, kNeighbors);
        const n    = nbrs.length;
        if (!n) { // no samples → uniform prior
          for (let u = 0; u < nUnits; u++) probs[vIdx + u] = 1 / nUnits;
          continue;
        }

        // Kriging weight matrix (augmented for Lagrange multiplier)
        const sz = n + 1;
        const K = new Float32Array(sz * sz);
        for (let i = 0; i < n; i++) {
          for (let j = 0; j < n; j++) {
            const d = Math.hypot(nbrs[i].x - nbrs[j].x, nbrs[i].y - nbrs[j].y, (nbrs[i].z - nbrs[j].z) * 2);
            K[i * sz + j] = i === j ? sill + nugget : sill - gammaSpherical(d, range, sill);
          }
          K[i * sz + n] = K[n * sz + i] = 1;
        }
        const kVec = new Array(sz).fill(1);
        for (let i = 0; i < n; i++) {
          const d = Math.hypot(nbrs[i].x - wx, nbrs[i].y - wy, (nbrs[i].z - wz) * 2);
          kVec[i] = sill - gammaSpherical(d, range, sill);
        }

        // Solve KW = k (flat arrays for speed, Gaussian elim)
        let weights;
        try {
          const KRows = Array.from({ length: sz }, (_, r) => Array.from({ length: sz }, (_, c) => K[r * sz + c]));
          weights = solveLinear(KRows, kVec).slice(0, n);
        } catch {
          // Fallback to IDW if kriging system is singular
          const wSum = nbrs.reduce((s, nb) => s + (1 / (nb.dist + 0.01)), 0);
          weights = nbrs.map(nb => (1 / (nb.dist + 0.01)) / wSum);
        }

        // Accumulate indicator for each unit
        let pSum = 0;
        for (let u = 0; u < nUnits; u++) {
          let p = 0;
          for (let i = 0; i < n; i++) {
            const ind = nbrs[i].unitCode === unitCodes[u] ? nbrs[i].cert : 0;
            p += Math.max(0, weights[i]) * ind;
          }
          p = Math.max(0, Math.min(1, p));
          probs[vIdx + u] = p;
          pSum += p;
        }
        // Normalise so probabilities sum to 1
        if (pSum > 1e-9) {
          for (let u = 0; u < nUnits; u++) probs[vIdx + u] /= pSum;
        } else {
          for (let u = 0; u < nUnits; u++) probs[vIdx + u] = 1 / nUnits;
        }
      }
    }
    if (onProgress) onProgress(iz / nz);
  }

  // ── Extract winner + certainty from probability volumes ──────────────────────
  const unitIds      = new Uint8Array(total);
  const certainty    = new Float32Array(total);
  const blendUnitIds = new Uint8Array(total);
  const blendRatios  = new Float32Array(total);

  const unitIdsMap = {};
  geoUnits.forEach(u => { unitIdsMap[u.code] = u.id; });

  for (let v = 0; v < total; v++) {
    const base = v * nUnits;
    let b1 = 0, b2 = 1;
    for (let u = 2; u < nUnits; u++) {
      if      (probs[base + u] > probs[base + b1]) { b2 = b1; b1 = u; }
      else if (probs[base + u] > probs[base + b2])  b2 = u;
    }
    unitIds[v]      = unitIdsMap[unitCodes[b1]] ?? 0;
    certainty[v]    = Math.max(0.05, probs[base + b1]);
    blendUnitIds[v] = unitIdsMap[unitCodes[b2]] ?? 0;
    blendRatios[v]  = probs[base + b2];
  }

  // ── Build per-unit probability maps (separate Float32Arrays) ─────────────────
  const probVolumes = new Map();
  for (let u = 0; u < nUnits; u++) {
    const vol = new Float32Array(total);
    for (let v = 0; v < total; v++) vol[v] = probs[v * nUnits + u];
    probVolumes.set(unitCodes[u], vol);
  }

  if (onProgress) onProgress(1);
  log(`Indicator kriging complete — ${nUnits} units · ${total.toLocaleString()} voxels`, 'ok');

  const ikBase = {
    nx, ny, nz, cellSize, cellHeight: cellH,
    origin: { x: ox, y: oz, z: oy },
    worldWidth: nx * cellSize, worldHeight: nz * cellH, worldDepth: ny * cellSize,
  };
  _applyTopoMask(unitIds, certainty, ikBase, options.topoPoints);
  return { ...ikBase, unitIds, certainty, blendUnitIds, blendRatios, probVolumes, method: 'indicator-kriging' };
}

// ── Stratigraphic inversion detection + correction ─────────────────────────────
// Scans each vertical column for stratigraphic inversions — voxels where a younger
// unit (lower stratOrder index) appears BELOW an older unit (higher index).
// stratOrder: array of unit codes youngest-first [MADE, FILL, CLAY, …, BEDROCK]
// Modifies grid.unitIds and grid.certainty in-place.
// Returns { invertedCount, corrections, invertedFraction }
export function detectAndCorrectInversions(grid, geoUnits, stratOrder) {
  if (!grid || !stratOrder?.length || !geoUnits?.length) return { invertedCount: 0, corrections: 0 };

  const { nx, ny, nz, unitIds, certainty } = grid;
  if (!unitIds || !certainty) return { invertedCount: 0, corrections: 0 };
  const n2 = nx * ny;

  const rankByCode = new Map(stratOrder.map((code, i) => [code, i]));
  const rankById   = new Map();
  for (const u of geoUnits) {
    const r = rankByCode.get(u.code);
    if (r != null) rankById.set(u.id, r);
  }
  if (!rankById.size) return { invertedCount: 0, corrections: 0 };

  let invertedCount = 0, corrections = 0;

  for (let iy = 0; iy < ny; iy++) {
    for (let ix = 0; ix < nx; ix++) {
      // Top → bottom: rank should non-decrease (older units deeper)
      let maxRank = -1;
      let unitAtMaxRank = 0;

      for (let iz = nz - 1; iz >= 0; iz--) {
        const idx = ix + iy * nx + iz * n2;
        const uid = unitIds[idx];
        if (uid === 0) continue;
        const rank = rankById.get(uid);
        if (rank == null) continue;

        if (rank >= maxRank) {
          maxRank = rank;
          unitAtMaxRank = uid;
        } else {
          // Inversion: younger unit below an older unit already seen above
          invertedCount++;
          if (idx < unitIds.length) {
            unitIds[idx] = unitAtMaxRank;
            certainty[idx] = Math.min((certainty[idx] ?? 1) * 0.65, 0.45);
            corrections++;
          }
        }
      }
    }
  }

  return {
    invertedCount,
    corrections,
    invertedFraction: n2 * nz > 0 ? invertedCount / (n2 * nz) : 0,
  };
}

// ── 3D geotechnical parameter volumes ─────────────────────────────────────────
// Builds IDW-interpolated 3D volumes for engineering test parameters:
//   N_spt — SPT blow count (from borehole layers)
//   cu    — undrained shear strength (from unit params, unit-constant)
//   phi   — friction angle (from unit params)
//   gamma — unit weight (from unit params)
// Returns Map<paramName, Float32Array(nx*ny*nz)>
export function buildParamVolumes(boreholes, geoUnits, grid) {
  if (!grid || !boreholes?.length || !geoUnits?.length) return new Map();
  const { nx, ny, nz, cellSize: cs, cellHeight: ch, origin, unitIds } = grid;
  if (!unitIds || !nx || !ny || !nz) return new Map();
  const ox = origin.x, oy = origin.y, oz = origin.z;
  const total = nx * ny * nz;
  const n2    = nx * ny;
  const vols  = new Map();

  // ── SPT N-value: point observations from borehole layers ─────────────────
  const sptObs = [];
  for (const bh of boreholes) {
    if (!bh.layers) continue;
    let depthTop = 0;
    for (const layer of bh.layers) {
      if (layer.sptN != null && layer.sptN > 0) {
        const depthMid = (depthTop + (layer.base ?? depthTop + 1)) / 2;
        const zWorld   = (bh.groundLevel ?? 0) - depthMid;
        sptObs.push({ x: bh.x, y: bh.y, z: zWorld, v: Math.min(layer.sptN, 100) });
      }
      depthTop = layer.base ?? depthTop + 1;
    }
  }

  if (sptObs.length >= 2) {
    vols.set('N_spt', _idw3D(sptObs, nx, ny, nz, ox, oy, oz, cs, ch, 6, 2.0));
  }

  // ── Unit-parameter volumes: fill each voxel from its unit's params ─────────
  const UNIT_PARAMS = ['cu', 'phi', 'gamma', 'E', 'Cc'];
  const unitParamMaps = {};
  for (const pname of UNIT_PARAMS) {
    const map = new Map();
    for (const u of geoUnits) {
      const v = u.params?.[pname];
      if (v != null && isFinite(v)) map.set(u.id, v);
    }
    if (map.size > 0) unitParamMaps[pname] = map;
  }

  for (const [pname, map] of Object.entries(unitParamMaps)) {
    const vol = new Float32Array(total).fill(NaN);
    for (let i = 0; i < total; i++) {
      const uid = unitIds[i];
      const val = map.get(uid);
      if (val != null) vol[i] = val;
    }
    vols.set(pname, vol);
  }

  return vols;
}

function _idw3D(obs, nx, ny, nz, ox, oy, oz, cs, ch, k, power) {
  const n2   = nx * ny;
  const vol  = new Float32Array(nx * ny * nz).fill(NaN);
  if (!obs?.length) return vol;

  // Bucket by (ix, iy) column for fast 2D neighbor search
  const buckets = new Map();
  for (const o of obs) {
    if (!isFinite(o.x) || !isFinite(o.y) || !isFinite(o.z) || !isFinite(o.v)) continue;
    const ix = Math.floor((o.x - ox) / cs);
    const iy = Math.floor((o.y - oz) / cs);
    const key = `${ix},${iy}`;
    if (!buckets.has(key)) buckets.set(key, []);
    buckets.get(key).push(o);
  }

  const validObs = obs.length || 1;
  const SEARCH_R = Math.max(3, Math.ceil(Math.sqrt(nx * ny / validObs) * 2));

  for (let iz = 0; iz < nz; iz++) {
    const wz = oy + iz * ch + ch * 0.5;
    for (let iy = 0; iy < ny; iy++) {
      const wy = oz + iy * cs + cs * 0.5;
      for (let ix = 0; ix < nx; ix++) {
        const wx = ox + ix * cs + cs * 0.5;

        // Collect candidate observations within column-search radius
        const cands = [];
        for (let diy = -SEARCH_R; diy <= SEARCH_R; diy++) {
          for (let dix = -SEARCH_R; dix <= SEARCH_R; dix++) {
            const key = `${ix + dix},${iy + diy}`;
            const b = buckets.get(key);
            if (b) for (const o of b) cands.push(o);
          }
        }
        if (!cands.length) continue;

        // k-nearest in 3D and IDW
        cands.sort((a, b) => {
          const da = (a.x - wx) ** 2 + (a.y - wy) ** 2 + (a.z - wz) ** 2;
          const db = (b.x - wx) ** 2 + (b.y - wy) ** 2 + (b.z - wz) ** 2;
          return da - db;
        });
        const near = cands.slice(0, k);
        if (!near.length) continue;
        const d0 = Math.sqrt((near[0].x - wx) ** 2 + (near[0].y - wy) ** 2 + (near[0].z - wz) ** 2);
        if (d0 < 0.001) { vol[ix + iy * nx + iz * n2] = near[0].v; continue; }

        let sum = 0, wsum = 0;
        for (const n of near) {
          const d = Math.sqrt((n.x - wx) ** 2 + (n.y - wy) ** 2 + (n.z - wz) ** 2);
          const w = 1 / Math.pow(Math.max(d, 0.01), power);
          sum += w * n.v; wsum += w;
        }
        const result = wsum > 0 ? sum / wsum : NaN;
        vol[ix + iy * nx + iz * n2] = isFinite(result) ? result : NaN;
      }
    }
  }
  return vol;
}

// ── Formation pinch-out / wedge-out detection ─────────────────────────────────
// For each geological unit, finds columns where the unit is present and adjacent
// to columns where it is absent (zero thickness). These edges are "pinch-outs" —
// where the formation wedges out laterally.
// Returns Map<unitCode, { pinchoutEdges: [{ix, iy, wx, wy}], presentCols, totalCols }>
export function detectPinchouts(grid, geoUnits) {
  if (!grid || !geoUnits?.length) return new Map();
  const { nx, ny, nz, cellSize: cs, origin, unitIds } = grid;
  if (!unitIds) return new Map();
  const n2 = nx * ny;
  const result = new Map();

  for (const unit of geoUnits) {
    // Build presence map: present[ix + iy*nx] = true if unit exists in column
    const present = new Uint8Array(n2);
    let presentCount = 0;
    for (let iy = 0; iy < ny; iy++) {
      for (let ix = 0; ix < nx; ix++) {
        for (let iz = 0; iz < nz; iz++) {
          if (unitIds[ix + iy * nx + iz * n2] === unit.id) {
            present[ix + iy * nx] = 1;
            presentCount++;
            break;
          }
        }
      }
    }
    if (!presentCount) continue;

    // Find present columns adjacent to absent columns (4-connectivity)
    const edges = [];
    const DIRS = [[1, 0], [-1, 0], [0, 1], [0, -1]];
    for (let iy = 0; iy < ny; iy++) {
      for (let ix = 0; ix < nx; ix++) {
        if (!present[ix + iy * nx]) continue;
        for (const [dix, diy] of DIRS) {
          const nx2 = ix + dix, ny2 = iy + diy;
          if (nx2 < 0 || nx2 >= nx || ny2 < 0 || ny2 >= ny) continue;
          if (!present[nx2 + ny2 * nx]) {
            edges.push({
              ix, iy,
              wx: origin.x + (ix + 0.5) * cs,
              wy: origin.z + (iy + 0.5) * cs,
            });
            break; // one edge per column is enough
          }
        }
      }
    }

    if (edges.length) {
      result.set(unit.code, {
        pinchoutEdges: edges,
        presentCols: presentCount,
        totalCols: n2,
        coverageFraction: presentCount / n2,
      });
    }
  }
  return result;
}

export function voxelIndex(ix, iy, iz, grid) {
  return ix + iy * grid.nx + iz * grid.nx * grid.ny;
}

export function voxelWorldPos(ix, iy, iz, grid) {
  return {
    x: grid.origin.x + ix * grid.cellSize   + grid.cellSize   * 0.5,
    y: grid.origin.y + iz * grid.cellHeight + grid.cellHeight * 0.5,
    z: grid.origin.z + iy * grid.cellSize   + grid.cellSize   * 0.5,
  };
}
