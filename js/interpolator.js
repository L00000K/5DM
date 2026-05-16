import { log } from './app.js';
import { buildStratRankMap, stratigraphicConsistencyPenalty } from './semantic-engine.js';

const MIN_BH_DIST = 0.1;

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
function getCandidates(boreholes, x, y, z, sinAz = 0, cosAz = 1, anisoRatio = 1) {
  const out = [];
  for (const bh of boreholes) {
    // Anisotropic distance: scale perpendicular-to-strike by 1/ratio
    const dx = bh.x - x, dy = bh.y - y;
    const dAlong = dx * sinAz + dy * cosAz;         // along strike
    const dPerp  = dx * cosAz - dy * sinAz;         // across strike
    const dist2d = Math.hypot(dAlong, dPerp / anisoRatio);
    const depth  = (bh.groundLevel ?? 0) - z;
    if (!bh.layers.length || depth < 0) continue;
    let layer;
    if      (depth < bh.layers[0].top)                        layer = bh.layers[0];
    else if (depth > bh.layers[bh.layers.length - 1].base)    layer = bh.layers[bh.layers.length - 1];
    else layer = bh.layers.find(l => depth >= l.top && depth <= l.base);
    if (!layer?.unitCode) continue;
    out.push({ dist: Math.max(dist2d, MIN_BH_DIST), x: bh.x, y: bh.y,
               unitCode: layer.unitCode, layerCert: layer.certainty ?? 0.8 });
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
  return makeResult(bc, sc, bw / totalW, totalW, sw, cert, unitIndex, unknownId);
}

// ── Ordinary Kriging (spherical variogram, indicator approach) ────────────────
//   Augments the covariance system with a Lagrange multiplier to enforce
//   the unbiasedness constraint (weights sum to 1), giving optimal linear
//   unbiased prediction at the query location.
function krigingVote(neighbours, qx, qy, unitIndex, unknownId, range, sill) {
  const n = neighbours.length;
  const nugget = sill * 0.05;
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
  const onProgress  = options.onProgress ?? null;
  const stratRanks  = buildStratRankMap(options.stratOrder ?? []);
  // Anisotropy: anisoAzimuth = strike direction (degrees from North), anisoRatio > 1 = elongated along strike
  const anisoAz    = ((options.anisoAzimuth ?? 0) * Math.PI) / 180;
  const anisoRatio = Math.max(1, options.anisoRatio ?? 1);
  const anisoSinAz = Math.sin(anisoAz);
  const anisoCosAz = Math.cos(anisoAz);

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

  const methodLabel = { idw: 'IDW', kriging: 'Kriging', gp: 'Gauss. Process', nn: 'Neural Net' }[method] ?? method;
  log(`Grid ${nx}×${ny}×${nz} = ${(nx * ny * nz).toLocaleString()} voxels @ ${cellSize} m | ${methodLabel} K=${kNeighbors}`, 'info');

  const total       = nx * ny * nz;
  const unitIds      = new Uint8Array(total);
  const certainty    = new Float32Array(total);
  const blendUnitIds = new Uint8Array(total);
  const blendRatios  = new Float32Array(total);

  const unitIndex = {};
  geoUnits.forEach(u => { unitIndex[u.code] = u.id; });
  const unknownId = geoUnits.find(u => u.code === 'UNKN')?.id ?? 0;

  // Geostatistical parameters
  const range = typicalSpacing * 1.5;   // Kriging variogram range
  const sill  = 1.0;                    // normalised sill
  const gpLen = typicalSpacing * 0.8;   // GP length-scale

  // Neural network: train once before the main voxel loop
  let nnPredict = null;
  if (method === 'nn') {
    log('Training neural network on borehole data…', 'info');
    nnPredict = trainNN(boreholes, geoUnits,
      { minX, maxX, minY, maxY, botZ, topZ });
    if (!nnPredict) log('NN training yielded no samples — falling back to IDW', 'warn');
  }

  // ── 3. Classify every voxel ────────────────────────────────────────────────
  for (let iz = 0; iz < nz; iz++) {
    const z = oz + iz * cellH + cellH * 0.5;
    if (onProgress && iz % 3 === 0) {
      onProgress(iz / nz);
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
          const cands = getCandidates(boreholes, x, y, z, anisoSinAz, anisoCosAz, anisoRatio);
          cands.sort((a, b) => a.dist - b.dist);
          const nb = cands.slice(0, kNeighbors);

          if (!nb.length) {
            result = nearestFallback(boreholes, x, y, unitIndex, unknownId);
          } else if (method === 'kriging') {
            result = krigingVote(nb, x, y, unitIndex, unknownId, range, sill)
                  ?? idwVote(nb, idwPower, unitIndex, unknownId, typicalSpacing);
          } else if (method === 'gp') {
            result = gpVote(nb, x, y, unitIndex, unknownId, gpLen)
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
