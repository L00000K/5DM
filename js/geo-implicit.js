// ── Geo-Implicit: Neural Implicit Geological Field ──────────────────────────
// Architecture:
//   FourierEncoder(x,y,z)    → 39-dim positional features (L=6)
//   GeoKeywordEncoder(text)  → N_vocab-dim binary keyword presence
//   Concat → 4-layer MLP (nIn→64→64→64→nUnits) with skip H0→H2 (×0.1)
// Training: Adam + cosine-annealed LR, cross-entropy + L2
// Oracle:   BFS cluster detection → Claude reasons over uncertain regions

const L_FOURIER = 6; // Fourier frequency bands → 3 + 3×2×6 = 39 features

// ~60-term geological vocabulary: materials, geometry, direction, properties, processes
export const GEO_VOCAB = [
  // Materials
  'clay','silt','sand','gravel','cobble','boulder','rock','chalk','limestone',
  'sandstone','mudstone','shale','granite','basalt','till','peat','organic','fill',
  'made','alluvium','terrace','glacial','fluvial','marine','estuarine',
  // Morphology / geometry
  'layer','lens','channel','paleochannel','valley','ridge','basin','dome','anticline',
  'syncline','fold','fault','wedge','pinch','horizon','seam','vein','dyke',
  'intrusion','contact','unconformity','pocket','infill','deposit',
  // Directions / orientation
  'east','west','north','south','lateral','vertical','horizontal','inclined','dipping',
  'trending','strike','dip',
  // Engineering properties
  'soft','firm','stiff','hard','dense','loose','medium','plastic','brittle',
  'fissured','laminated','interbedded','weathered','fractured','massive','bedded',
  // Depositional / diagenetic processes
  'consolidated','overconsolidated','normally','eroded','deposited','compressed',
];

// ── Fourier Positional Encoder ───────────────────────────────────────────────
export class FourierEncoder {
  constructor(L = L_FOURIER) {
    this.L = L;
    this.outDim = 3 + 3 * 2 * L;
  }

  encode(x, y, z, bounds) {
    const nx = 2 * (x - bounds.minX) / (bounds.maxX - bounds.minX) - 1;
    const ny = 2 * (y - bounds.minY) / (bounds.maxY - bounds.minY) - 1;
    const nz = 2 * (z - bounds.minZ) / (bounds.maxZ - bounds.minZ) - 1;
    const out = new Float32Array(this.outDim);
    out[0] = nx; out[1] = ny; out[2] = nz;
    let i = 3;
    for (let k = 0; k < this.L; k++) {
      const f = Math.PI * Math.pow(2, k);
      out[i++] = Math.sin(f * nx); out[i++] = Math.cos(f * nx);
      out[i++] = Math.sin(f * ny); out[i++] = Math.cos(f * ny);
      out[i++] = Math.sin(f * nz); out[i++] = Math.cos(f * nz);
    }
    return out;
  }
}

// ── Geological Keyword Encoder ───────────────────────────────────────────────
export class GeoKeywordEncoder {
  constructor(vocab = GEO_VOCAB) {
    this.vocab = vocab;
    this.outDim = vocab.length;
    this._idx = {};
    vocab.forEach((w, i) => { this._idx[w] = i; });
  }

  encode(text) {
    const out = new Float32Array(this.outDim);
    if (!text) return out;
    const tokens = text.toLowerCase().replace(/[^a-z\s]/g, ' ').split(/\s+/);
    for (const tok of tokens) {
      const i = this._idx[tok];
      if (i !== undefined) out[i] = 1;
    }
    return out;
  }
}

// ── Adam Optimizer ───────────────────────────────────────────────────────────
class AdamOpt {
  constructor(params, lr = 0.01, beta1 = 0.9, beta2 = 0.999, eps = 1e-8) {
    this.params = params;
    this.lr = lr; this.beta1 = beta1; this.beta2 = beta2; this.eps = eps;
    this.t = 0;
    this.m = params.map(p => new Float32Array(p.length));
    this.v = params.map(p => new Float32Array(p.length));
  }

  step(grads) {
    this.t++;
    const { beta1, beta2, eps, t } = this;
    const lrt = this.lr * Math.sqrt(1 - Math.pow(beta2, t)) / (1 - Math.pow(beta1, t));
    for (let pi = 0; pi < this.params.length; pi++) {
      const p = this.params[pi], g = grads[pi], m = this.m[pi], v = this.v[pi];
      for (let i = 0; i < p.length; i++) {
        m[i] = beta1 * m[i] + (1 - beta1) * g[i];
        v[i] = beta2 * v[i] + (1 - beta2) * g[i] * g[i];
        p[i] -= lrt * m[i] / (Math.sqrt(v[i]) + eps);
      }
    }
  }

  setLr(lr) { this.lr = lr; }
}

// ── 4-Layer MLP with skip connection H0 → H2 ────────────────────────────────
class GeoImplicitNet {
  constructor(nIn, nHidden, nOut) {
    this.nIn = nIn; this.nHidden = nHidden; this.nOut = nOut;
    const k0 = Math.sqrt(2 / nIn), k1 = Math.sqrt(2 / nHidden);
    this.W = [
      this._rand(nHidden, nIn,     k0),
      this._rand(nHidden, nHidden, k1),
      this._rand(nHidden, nHidden, k1),
      this._rand(nOut,    nHidden, k1),
    ];
    this.b = [
      new Float32Array(nHidden), new Float32Array(nHidden),
      new Float32Array(nHidden), new Float32Array(nOut),
    ];
    this._params = [...this.W, ...this.b];
  }

  _rand(rows, cols, scale) {
    const a = new Float32Array(rows * cols);
    for (let i = 0; i < a.length; i++) a[i] = (Math.random() * 2 - 1) * scale;
    return a;
  }

  _linear(W, b, x, rows, cols) {
    const out = new Float32Array(rows);
    for (let r = 0; r < rows; r++) {
      let s = b[r];
      for (let c = 0; c < cols; c++) s += W[r * cols + c] * x[c];
      out[r] = s;
    }
    return out;
  }

  _relu(a) {
    const out = new Float32Array(a.length);
    for (let i = 0; i < a.length; i++) out[i] = a[i] > 0 ? a[i] : 0;
    return out;
  }

  _softmax(a) {
    let mx = -Infinity;
    for (let i = 0; i < a.length; i++) if (a[i] > mx) mx = a[i];
    let s = 0;
    const ex = new Float32Array(a.length);
    for (let i = 0; i < a.length; i++) { ex[i] = Math.exp(a[i] - mx); s += ex[i]; }
    for (let i = 0; i < a.length; i++) ex[i] /= s;
    return ex;
  }

  forward(x) {
    const { nHidden, nIn, nOut, W, b } = this;
    const H0_pre = this._linear(W[0], b[0], x, nHidden, nIn);
    const H0     = this._relu(H0_pre);
    const H1_pre = this._linear(W[1], b[1], H0, nHidden, nHidden);
    const H1     = this._relu(H1_pre);
    const H2_raw = this._linear(W[2], b[2], H1, nHidden, nHidden);
    const H2_pre = new Float32Array(nHidden);
    for (let i = 0; i < nHidden; i++) H2_pre[i] = H2_raw[i] + 0.1 * H0[i]; // skip
    const H2     = this._relu(H2_pre);
    const logits = this._linear(W[3], b[3], H2, nOut, nHidden);
    const probs  = this._softmax(logits);
    return { H0_pre, H0, H1_pre, H1, H2_pre, H2, logits, probs };
  }

  predict(x) { return this.forward(x).probs; }

  // Returns grads in same order as _params = [W0,W1,W2,W3, b0,b1,b2,b3]
  backward(x, act, targetIdx, l2 = 0.001) {
    const { nHidden, nIn, nOut, W } = this;
    const { H0_pre, H0, H1_pre, H1, H2_pre, H2, probs } = act;

    const dLogits = new Float32Array(nOut);
    for (let i = 0; i < nOut; i++) dLogits[i] = probs[i] - (i === targetIdx ? 1 : 0);

    const outerGrad = (dOut, inp, rows, cols, Wmat) => {
      const dW = new Float32Array(rows * cols), db = new Float32Array(rows);
      for (let r = 0; r < rows; r++) {
        db[r] = dOut[r];
        for (let c = 0; c < cols; c++) dW[r * cols + c] = dOut[r] * inp[c] + l2 * Wmat[r * cols + c];
      }
      return { dW, db };
    };
    const matVecT = (Wmat, dOut, rows, cols) => {
      const dIn = new Float32Array(cols);
      for (let c = 0; c < cols; c++)
        for (let r = 0; r < rows; r++) dIn[c] += Wmat[r * cols + c] * dOut[r];
      return dIn;
    };
    const reluBack = (dOut, pre) => {
      const d = new Float32Array(pre.length);
      for (let i = 0; i < pre.length; i++) d[i] = pre[i] > 0 ? dOut[i] : 0;
      return d;
    };

    // Layer 3
    const { dW: dW3, db: db3 } = outerGrad(dLogits, H2, nOut, nHidden, W[3]);
    const dH2      = matVecT(W[3], dLogits, nOut, nHidden);
    const dH2_pre  = reluBack(dH2, H2_pre);

    // Layer 2 + skip gradient back to H0
    const { dW: dW2, db: db2 } = outerGrad(dH2_pre, H1, nHidden, nHidden, W[2]);
    const dH1       = matVecT(W[2], dH2_pre, nHidden, nHidden);
    const dH0_skip  = new Float32Array(nHidden);
    for (let i = 0; i < nHidden; i++) dH0_skip[i] = 0.1 * dH2_pre[i];

    // Layer 1
    const dH1_pre    = reluBack(dH1, H1_pre);
    const { dW: dW1, db: db1 } = outerGrad(dH1_pre, H0, nHidden, nHidden, W[1]);
    const dH0_layer1 = matVecT(W[1], dH1_pre, nHidden, nHidden);

    // Combine H0 gradients (layer1 path + skip path)
    const dH0_sum = new Float32Array(nHidden);
    for (let i = 0; i < nHidden; i++) dH0_sum[i] = dH0_layer1[i] + dH0_skip[i];
    const dH0_pre = reluBack(dH0_sum, H0_pre);

    // Layer 0
    const { dW: dW0, db: db0 } = outerGrad(dH0_pre, x, nHidden, nIn, W[0]);

    return [dW0, dW1, dW2, dW3, db0, db1, db2, db3];
  }
}

// ── Build geological context from unit metadata + free text ─────────────────
export function buildGeoContext(geoUnits, siteHistory, unitDescriptions) {
  const allText = [
    ...geoUnits.map(u => `${u.name ?? ''} ${u.description ?? ''}`),
    siteHistory ?? '',
    ...(unitDescriptions ?? []),
  ].join(' ');
  const kwEnc = new GeoKeywordEncoder();
  const keywords = kwEnc.encode(allText);
  return { text: allText, keywords, kwEncoder: kwEnc };
}

// ── Train the neural implicit geological field ───────────────────────────────
// Returns { net, fourierEnc, kwVec, bounds, nUnits, unitCodes } or null
export async function trainGeoImplicit(boreholes, geoUnits, context, options = {}) {
  const {
    epochs          = 400,
    lr              = 0.01,
    lrMin           = 0.001,
    l2              = 0.001,
    samplesPerLayer = 6,
    onProgress      = null,
    sectionSamples  = [],   // [{pos, target, weight, localKwVec}] from section-interpreter
  } = options;

  // Compute world bounds from borehole data
  const bounds = { minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity, minZ: Infinity, maxZ: -Infinity };
  for (const bh of boreholes) {
    bounds.minX = Math.min(bounds.minX, bh.x);
    bounds.maxX = Math.max(bounds.maxX, bh.x);
    bounds.minY = Math.min(bounds.minY, bh.y);
    bounds.maxY = Math.max(bounds.maxY, bh.y);
    for (const l of (bh.layers ?? [])) {
      bounds.minZ = Math.min(bounds.minZ, bh.groundLevel - l.base);
      bounds.maxZ = Math.max(bounds.maxZ, bh.groundLevel - l.top);
    }
  }
  // Expand bounds to include section sample positions
  for (const s of sectionSamples) {
    if (s.x != null) {
      bounds.minX = Math.min(bounds.minX, s.x); bounds.maxX = Math.max(bounds.maxX, s.x);
      bounds.minY = Math.min(bounds.minY, s.y); bounds.maxY = Math.max(bounds.maxY, s.y);
      bounds.minZ = Math.min(bounds.minZ, s.z); bounds.maxZ = Math.max(bounds.maxZ, s.z);
    }
  }
  // Add margin to avoid normalisation edge issues
  for (const [mn, mx] of [['minX','maxX'],['minY','maxY'],['minZ','maxZ']]) {
    const margin = (bounds[mx] - bounds[mn]) * 0.05 || 1;
    bounds[mn] -= margin; bounds[mx] += margin;
  }

  const fourierEnc = new FourierEncoder(L_FOURIER);
  const kwEnc      = context?.kwEncoder ?? new GeoKeywordEncoder();
  const kwVec      = context?.keywords  ?? new Float32Array(kwEnc.outDim);
  const localDim   = kwEnc.outDim;  // local section context has same vocab size
  const nIn        = fourierEnc.outDim + kwEnc.outDim + localDim;
  const nUnits     = geoUnits.length;
  const unitCodes  = geoUnits.map(u => u.code);
  const unitIdx    = {};
  geoUnits.forEach((u, i) => { unitIdx[u.code] = i; });
  const zeroLocal  = new Float32Array(localDim); // BH samples have no local section context

  // Build training samples from boreholes (local context = zeros)
  const samples = [];
  for (const bh of boreholes) {
    for (const layer of (bh.layers ?? [])) {
      const ti = unitIdx[layer.unitCode];
      if (ti === undefined) continue;
      const zTop  = bh.groundLevel - layer.top;
      const zBase = bh.groundLevel - layer.base;
      const wt    = layer.certainty ?? 0.9;
      for (let s = 0; s < samplesPerLayer; s++) {
        const t   = (s + 0.5) / samplesPerLayer;
        const z   = zBase + t * (zTop - zBase);
        const pos = fourierEnc.encode(bh.x, bh.y, z, bounds);
        const inp = new Float32Array(nIn);
        inp.set(pos);
        inp.set(kwVec,    fourierEnc.outDim);
        inp.set(zeroLocal, fourierEnc.outDim + kwEnc.outDim); // no local context at BH locations
        samples.push({ inp, target: ti, weight: wt });
      }
    }
  }

  // Merge section training samples (carry their local keyword context)
  for (const ss of sectionSamples) {
    const pos = ss.pos ?? fourierEnc.encode(ss.x, ss.y, ss.z, bounds);
    const inp = new Float32Array(nIn);
    inp.set(pos);
    inp.set(kwVec, fourierEnc.outDim);
    inp.set(ss.localKwVec ?? zeroLocal, fourierEnc.outDim + kwEnc.outDim);
    samples.push({ inp, target: ss.target, weight: ss.weight });
  }

  if (samples.length === 0) return null;

  const net = new GeoImplicitNet(nIn, 64, nUnits);
  const opt = new AdamOpt([...net.W, ...net.b], lr);

  for (let ep = 0; ep < epochs; ep++) {
    // Cosine LR annealing
    opt.setLr(lrMin + 0.5 * (lr - lrMin) * (1 + Math.cos(Math.PI * ep / epochs)));

    // Fisher-Yates shuffle
    for (let i = samples.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      const tmp = samples[i]; samples[i] = samples[j]; samples[j] = tmp;
    }

    let totalLoss = 0;
    for (const s of samples) {
      const act  = net.forward(s.inp);
      totalLoss -= s.weight * Math.log(Math.max(act.probs[s.target], 1e-9));
      const grads = net.backward(s.inp, act, s.target, l2);
      opt.step(grads);
    }

    if (onProgress && ep % 20 === 0) {
      onProgress(ep / epochs, totalLoss / samples.length);
      await new Promise(r => setTimeout(r, 0));
    }
  }

  if (onProgress) onProgress(1, 0);
  return { net, fourierEnc, kwVec, localDim, bounds, nUnits, unitCodes };
}

// ── Infer voxel grid from trained model ─────────────────────────────────────
// grid must have { nx, ny, nz, cellSize, cellHeight, origin: {x,y,z} }
// options.sectionPlanes: [{fence, localKwVec}] for spatially-local section context
// options.computeLocalContext: function(x, y, planes) from section-interpreter
export function inferGeoImplicit(trained, grid, geoUnits, options = {}) {
  const sectionPlanes = Array.isArray(options) ? [] : (options.sectionPlanes ?? []);
  const computeLC     = Array.isArray(options) ? null : (options.computeLocalContext ?? null);
  const { net, fourierEnc, kwVec, localDim, bounds, unitCodes } = trained;
  const zeroLocal = new Float32Array(localDim ?? 0);
  const { nx, ny, nz, cellSize, cellHeight, origin } = grid;
  const kwLen  = kwVec?.length ?? 0;
  const locLen = localDim ?? 0;
  const nIn    = fourierEnc.outDim + kwLen + locLen;

  const unitIds      = new Uint8Array(nx * ny * nz);
  const certainty    = new Float32Array(nx * ny * nz);
  const blendUnitIds = new Uint8Array(nx * ny * nz);
  const blendRatios  = new Float32Array(nx * ny * nz);

  const codeToId = {};
  geoUnits.forEach(u => { codeToId[u.code] = u.id; });

  for (let iz = nz - 1; iz >= 0; iz--) {
    const worldZ = origin.y + iz * cellHeight + cellHeight * 0.5;
    for (let iy = 0; iy < ny; iy++) {
      const worldY = origin.z + iy * cellSize + cellSize * 0.5;
      for (let ix = 0; ix < nx; ix++) {
        const worldX = origin.x + ix * cellSize + cellSize * 0.5;
        const idx    = ix + iy * nx + iz * nx * ny;

        const pos = fourierEnc.encode(worldX, worldY, worldZ, bounds);
        const inp = new Float32Array(nIn);
        inp.set(pos);
        if (kwVec) inp.set(kwVec, fourierEnc.outDim);
        // Local context from section planes — spatially blended by proximity
        if (computeLC && sectionPlanes.length && locLen > 0) {
          const local = computeLC(worldX, worldY, sectionPlanes);
          inp.set(local, fourierEnc.outDim + kwLen);
        }

        const probs = net.predict(inp);

        // Find top-1 and top-2 classes
        let b1 = 0, b2 = 1;
        if (probs[1] > probs[0]) { b1 = 1; b2 = 0; }
        for (let u = 2; u < probs.length; u++) {
          if      (probs[u] > probs[b1]) { b2 = b1; b1 = u; }
          else if (probs[u] > probs[b2])  { b2 = u; }
        }

        unitIds[idx]      = codeToId[unitCodes[b1]] ?? 0;
        certainty[idx]    = Math.max(0.05, Math.min(1, 0.5 + probs[b1] - probs[b2]));
        blendUnitIds[idx] = codeToId[unitCodes[b2]] ?? 0;
        blendRatios[idx]  = probs[b2];
      }
    }
  }

  return { unitIds, certainty, blendUnitIds, blendRatios };
}

// ── Patch voxel grid with oracle probability distributions ───────────────────
// oracleResults: [{ voxelIdxs: number[], distribution: { [code]: prob } }]
export function patchWithOracle(unitIds, certainty, blendUnitIds, blendRatios, oracleResults, geoUnits) {
  const codeToId = {};
  geoUnits.forEach(u => { codeToId[u.code] = u.id; });

  for (const res of oracleResults) {
    if (!res?.distribution || !Array.isArray(res.voxelIdxs)) continue;
    // Sort by probability descending
    const sorted = Object.entries(res.distribution).sort((a, b) => b[1] - a[1]);
    if (!sorted.length) continue;
    const [code1, p1] = sorted[0];
    const [code2, p2] = sorted[1] ?? [sorted[0][0], 0];
    const uid1 = codeToId[code1];
    const uid2 = codeToId[code2];
    if (uid1 === undefined) continue;
    for (const vidx of res.voxelIdxs) {
      unitIds[vidx]      = uid1;
      certainty[vidx]    = Math.min(1, p1 + 0.1); // oracle gets slight boost
      blendUnitIds[vidx] = uid2 ?? uid1;
      blendRatios[vidx]  = p2;
    }
  }
}

// ── Find high-entropy clusters for LLM oracle refinement ────────────────────
// Returns top N clusters: { voxels, centroid, entropy, worldPos }
export function findUncertainClusters(certainty, grid, threshold = 0.45, maxClusters = 12) {
  const { nx, ny, nz, cellSize, cellHeight, origin } = grid;
  const total   = nx * ny * nz;
  const visited = new Uint8Array(total);
  const clusters = [];

  for (let iz = 0; iz < nz; iz++) {
    for (let iy = 0; iy < ny; iy++) {
      for (let ix = 0; ix < nx; ix++) {
        const idx = ix + iy * nx + iz * nx * ny;
        if (visited[idx] || certainty[idx] >= threshold) continue;

        // BFS flood-fill
        const queue  = [idx];
        const voxels = [];
        visited[idx] = 1;
        let sumEnt = 0, sx = 0, sy = 0, sz = 0;

        while (queue.length) {
          const cur  = queue.pop();
          voxels.push(cur);
          const ciz  = Math.floor(cur / (nx * ny));
          const rem  = cur - ciz * nx * ny;
          const ciy  = Math.floor(rem / nx);
          const cix  = rem % nx;
          sx += cix; sy += ciy; sz += ciz;
          sumEnt += 1 - certainty[cur];

          const nbrs = [
            [cix+1,ciy,ciz],[cix-1,ciy,ciz],
            [cix,ciy+1,ciz],[cix,ciy-1,ciz],
            [cix,ciy,ciz+1],[cix,ciy,ciz-1],
          ];
          for (const [nx2,ny2,nz2] of nbrs) {
            if (nx2 < 0 || nx2 >= nx || ny2 < 0 || ny2 >= ny || nz2 < 0 || nz2 >= nz) continue;
            const ni = nx2 + ny2 * nx + nz2 * nx * ny;
            if (!visited[ni] && certainty[ni] < threshold) { visited[ni] = 1; queue.push(ni); }
          }
        }

        if (voxels.length < 4) continue;
        const n = voxels.length;
        const cix_c = sx / n, ciy_c = sy / n, ciz_c = sz / n;
        clusters.push({
          voxels,
          centroid: { ix: cix_c, iy: ciy_c, iz: ciz_c },
          worldPos: {
            x: origin.x + cix_c * cellSize   + cellSize   * 0.5,
            y: origin.y + ciz_c * cellHeight + cellHeight * 0.5,
            z: origin.z + ciy_c * cellSize   + cellSize   * 0.5,
          },
          entropy: sumEnt / n,
          score:   n * (sumEnt / n),
        });
      }
    }
  }

  clusters.sort((a, b) => b.score - a.score);
  return clusters.slice(0, maxClusters);
}
