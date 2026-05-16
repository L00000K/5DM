import { log } from './app.js';
import { buildStratRankMap, stratigraphicConsistencyPenalty,
         descriptionJaccard, meanDescriptionSimilarity, buildTransitionMatrix } from './semantic-engine.js';
import { trainGeoImplicit, inferGeoImplicit, buildGeoContext,
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

// ── Ordinary Kriging (spherical variogram, indicator approach) ────────────────
//   Augments the covariance system with a Lagrange multiplier to enforce
//   the unbiasedness constraint (weights sum to 1), giving optimal linear
//   unbiased prediction at the query location.
function krigingVote(neighbours, qx, qy, unitIndex, unknownId, range, sill, nugget = null) {
  const n = neighbours.length;
  if (nugget === null) nugget = sill * 0.05;
  const sz = n + 1;
  const K = Array.from({ length: sz }, () => new Array(sz).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const d = Math.hypot(neighbours[i].x - neighbours[j].x, neighbours[i].y - neighbours[j].y);
      K[i][j] = i === j ? sill + nugget : sill - gammaSpherical(d, range, sill);
    }
    K[i][n] = K[n][i] = 1;
  }
  const k = new Array(sz).fill(1);
  for (let i = 0; i < n; i++) {
    const d = Math.hypot(neighbours[i].x - qx, neighbours[i].y - qy);
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
function ukVote(neighbours, qx, qy, unitIndex, unknownId, range, sill, trendOrder) {
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
      const d = Math.hypot(neighbours[i].x - neighbours[j].x, neighbours[i].y - neighbours[j].y);
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
    const d = Math.hypot(neighbours[i].x - qx, neighbours[i].y - qy);
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

  // ── Neural Implicit Geological Field ──────────────────────────────────────
  if (method === 'neural-implicit') {
    log('Building geological context from unit descriptions…', 'info');
    const siteHistory = options.siteHistory ?? '';
    const unitDescs   = options.unitDescriptions ?? [];
    const geoCtx  = buildGeoContext(geoUnits, siteHistory, unitDescs);

    log(`Training neural implicit field (${options.niEpochs ?? 400} epochs)…`, 'info');
    if (onProgress) onProgress(0.02);

    const trainedModel = await trainGeoImplicit(
      allBoreholes, geoUnits, geoCtx,
      {
        epochs:          options.niEpochs ?? 400,
        lr:              0.01,
        lrMin:           0.001,
        l2:              0.001,
        samplesPerLayer: 8,
        onProgress: (frac, loss) => {
          if (onProgress) onProgress(0.02 + frac * 0.7);
          if (frac < 1) log(`  …epoch ${Math.round(frac * (options.niEpochs ?? 400))} loss=${loss?.toFixed(4) ?? '–'}`, 'info');
        },
      },
    );

    if (!trainedModel) {
      log('Neural implicit training failed (no samples) — falling back to IDW', 'warn');
      // Fall through to IDW below
    } else {
      log('Inferring voxel grid from neural implicit field…', 'info');
      const gridMeta = { nx, ny, nz, cellSize, cellHeight: cellH, origin: { x: ox, y: oz, z: oy } };
      const inferred = inferGeoImplicit(trainedModel, gridMeta, geoUnits);
      unitIds.set(inferred.unitIds);
      certainty.set(inferred.certainty);
      blendUnitIds.set(inferred.blendUnitIds);
      blendRatios.set(inferred.blendRatios);

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

      if (onProgress) onProgress(1);
      return {
        nx, ny, nz,
        cellSize, cellHeight: cellH,
        origin: { x: ox, y: oz, z: oy },
        worldWidth:  nx * cellSize,
        worldHeight: nz * cellH,
        worldDepth:  ny * cellSize,
        unitIds, certainty, blendUnitIds, blendRatios,
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
          const cands = getCandidates(allBoreholes, x, y, z, anisoSinAz, anisoCosAz, anisoRatio);
          cands.sort((a, b) => a.dist - b.dist);
          const nb = cands.slice(0, kNeighbors);

          if (!nb.length) {
            result = nearestFallback(allBoreholes, x, y, unitIndex, unknownId);
          } else if (method === 'kriging') {
            result = krigingVote(nb, x, y, unitIndex, unknownId, range, sill, options.varNugget ?? null)
                  ?? idwVote(nb, idwPower, unitIndex, unknownId, typicalSpacing);
          } else if (method === 'uk') {
            result = ukVote(nb, x, y, unitIndex, unknownId, range, sill, trendOrder)
                  ?? idwVote(nb, idwPower, unitIndex, unknownId, typicalSpacing);
          } else if (method === 'gp') {
            result = gpVote(nb, x, y, unitIndex, unknownId, gpLen)
                  ?? idwVote(nb, idwPower, unitIndex, unknownId, typicalSpacing);
          } else if (method === 'rbf') {
            result = rbfVote(nb, x, y, unitIndex, unknownId, null)
                  ?? idwVote(nb, idwPower, unitIndex, unknownId, typicalSpacing);
          } else {
            result = idwVote(nb, idwPower, unitIndex, unknownId, typicalSpacing);
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

        unitIds[idx]      = result.unitId;
        certainty[idx]    = Math.max(0.05, cert);
        blendUnitIds[idx] = result.blendUnitId;
        blendRatios[idx]  = result.blendRatio;
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
  };
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

  return { ...base, unitIds, certainty, blendUnitIds, blendRatios };
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
