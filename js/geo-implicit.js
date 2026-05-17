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

  step(grads, clipNorm = 5.0) {
    this.t++;
    const { beta1, beta2, eps, t } = this;
    const lrt = this.lr * Math.sqrt(1 - Math.pow(beta2, t)) / (1 - Math.pow(beta1, t));

    // Global gradient clipping: prevents explosive updates when concept boosts
    // create large gradient magnitudes early in training.
    let gnorm2 = 0;
    for (const g of grads) for (let i = 0; i < g.length; i++) gnorm2 += g[i] * g[i];
    const gnorm = Math.sqrt(gnorm2);
    const scale = gnorm > clipNorm ? clipNorm / gnorm : 1.0;

    for (let pi = 0; pi < this.params.length; pi++) {
      const p = this.params[pi], g = grads[pi], m = this.m[pi], v = this.v[pi];
      for (let i = 0; i < p.length; i++) {
        const gi = g[i] * scale;
        m[i] = beta1 * m[i] + (1 - beta1) * gi;
        v[i] = beta2 * v[i] + (1 - beta2) * gi * gi;
        p[i] -= lrt * m[i] / (Math.sqrt(v[i]) + eps);
      }
    }
  }

  setLr(lr) { this.lr = lr; }
}

// ── 4-Layer MLP with FiLM conditioning and skip connection H0f → H2 ─────────
// FiLM = Feature-wise Linear Modulation: the concept context vector (last 32
// elements of the input) generates per-layer scale γ and shift β that modulate
// each hidden layer's pre-activation features before the non-linearity.
class GeoImplicitNet {
  constructor(nIn, nHidden, nOut, fourierDim = 39) {
    this.nIn = nIn; this.nHidden = nHidden; this.nOut = nOut;
    this.fourierDim = fourierDim;
    const CTX_DIM = 32;
    const k0 = Math.sqrt(2 / nIn), k1 = Math.sqrt(2 / nHidden), kg = Math.sqrt(2 / CTX_DIM);
    // Main weight matrices W0..W3
    this.W = [
      this._rand(nHidden, nIn,     k0),
      this._rand(nHidden, nHidden, k1),
      this._rand(nHidden, nHidden, k1),
      this._rand(nOut,    nHidden, k1),
    ];
    // FiLM scale projections Wg0, Wg1  (nHidden × CTX_DIM) — small random init
    this.Wg = [
      this._rand(nHidden, CTX_DIM, kg * 0.1),
      this._rand(nHidden, CTX_DIM, kg * 0.1),
    ];
    // FiLM shift projections Wb0, Wb1  (nHidden × CTX_DIM) — zero init
    this.Wb = [
      new Float32Array(nHidden * CTX_DIM),
      new Float32Array(nHidden * CTX_DIM),
    ];
    this.b = [
      new Float32Array(nHidden), new Float32Array(nHidden),
      new Float32Array(nHidden), new Float32Array(nOut),
    ];
    // _params order: [W0,W1,W2,W3, Wg0,Wg1, Wb0,Wb1, b0,b1,b2,b3]
    this._params = [...this.W, ...this.Wg, ...this.Wb, ...this.b];
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

  // FiLM projection: returns nHidden-dim vector from CTX_DIM-dim ctx
  _filmProj(Wg, ctx, nHidden, CTX_DIM) {
    const out = new Float32Array(nHidden);
    for (let r = 0; r < nHidden; r++) {
      let s = 0;
      for (let c = 0; c < CTX_DIM; c++) s += Wg[r * CTX_DIM + c] * ctx[c];
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

  forward(inp) {
    const { nHidden, nIn, nOut, W, Wg, Wb, b, fourierDim } = this;
    const CTX_DIM = 32;
    // ctx = last 32 elements (concept context)
    const ctx = inp.subarray(fourierDim);

    // Layer 0
    const H0_pre = this._linear(W[0], b[0], inp, nHidden, nIn);
    const H0     = this._relu(H0_pre);
    // FiLM layer 0: γ0 = 1 + Wg0@ctx, β0 = Wb0@ctx
    const gamma0_raw = this._filmProj(Wg[0], ctx, nHidden, CTX_DIM);
    const beta0      = this._filmProj(Wb[0], ctx, nHidden, CTX_DIM);
    const gamma0     = new Float32Array(nHidden);
    const H0f        = new Float32Array(nHidden);
    for (let i = 0; i < nHidden; i++) {
      gamma0[i] = 1 + gamma0_raw[i];
      H0f[i]    = gamma0[i] * H0[i] + beta0[i];
    }

    // Layer 1 (takes modulated H0f)
    const H1_pre = this._linear(W[1], b[1], H0f, nHidden, nHidden);
    const H1     = this._relu(H1_pre);
    // FiLM layer 1: γ1 = 1 + Wg1@ctx, β1 = Wb1@ctx
    const gamma1_raw = this._filmProj(Wg[1], ctx, nHidden, CTX_DIM);
    const beta1      = this._filmProj(Wb[1], ctx, nHidden, CTX_DIM);
    const gamma1     = new Float32Array(nHidden);
    const H1f        = new Float32Array(nHidden);
    for (let i = 0; i < nHidden; i++) {
      gamma1[i] = 1 + gamma1_raw[i];
      H1f[i]    = gamma1[i] * H1[i] + beta1[i];
    }

    // Layer 2 (takes modulated H1f, skip from H0f)
    const H2_raw = this._linear(W[2], b[2], H1f, nHidden, nHidden);
    const H2_pre = new Float32Array(nHidden);
    for (let i = 0; i < nHidden; i++) H2_pre[i] = H2_raw[i] + 0.1 * H0f[i]; // skip uses H0f
    const H2     = this._relu(H2_pre);

    // Output layer
    const logits = this._linear(W[3], b[3], H2, nOut, nHidden);
    const probs  = this._softmax(logits);

    return { H0_pre, H0, gamma0, beta0, H0f, H1_pre, H1, gamma1, beta1, H1f, H2_pre, H2, logits, probs };
  }

  predict(inp) { return this.forward(inp).probs; }

  // Returns grads in same order as _params = [dW0,dW1,dW2,dW3, dWg0,dWg1, dWb0,dWb1, db0,db1,db2,db3]
  // sampleWeight scales the gradient — allows high-confidence samples to drive stronger updates.
  backward(inp, act, targetIdx, l2 = 0.001, sampleWeight = 1.0) {
    const { nHidden, nIn, nOut, W, Wg, Wb, fourierDim } = this;
    const CTX_DIM = 32;
    const ctx = inp.subarray(fourierDim);
    const { H0_pre, H0, gamma0, beta0, H0f, H1_pre, H1, gamma1, beta1, H1f, H2_pre, H2, probs } = act;

    const dLogits = new Float32Array(nOut);
    for (let i = 0; i < nOut; i++) dLogits[i] = (probs[i] - (i === targetIdx ? 1 : 0)) * sampleWeight;

    const outerGrad = (dOut, inp_vec, rows, cols, Wmat) => {
      const dW = new Float32Array(rows * cols), db = new Float32Array(rows);
      for (let r = 0; r < rows; r++) {
        db[r] = dOut[r];
        for (let c = 0; c < cols; c++) dW[r * cols + c] = dOut[r] * inp_vec[c] + l2 * Wmat[r * cols + c];
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
    // Outer product gradient for FiLM projection (nHidden × CTX_DIM)
    const filmOuterGrad = (dRaw, ctx_vec, Wmat) => {
      const dW = new Float32Array(nHidden * CTX_DIM);
      for (let r = 0; r < nHidden; r++)
        for (let c = 0; c < CTX_DIM; c++)
          dW[r * CTX_DIM + c] = dRaw[r] * ctx_vec[c] + l2 * Wmat[r * CTX_DIM + c];
      return dW;
    };

    // ── Layer 3 ──────────────────────────────────────────────────────────────
    const { dW: dW3, db: db3 } = outerGrad(dLogits, H2, nOut, nHidden, W[3]);
    const dH2     = matVecT(W[3], dLogits, nOut, nHidden);
    const dH2_pre = reluBack(dH2, H2_pre);

    // ── Layer 2 (input was H1f; skip from H0f) ────────────────────────────
    const { dW: dW2, db: db2 } = outerGrad(dH2_pre, H1f, nHidden, nHidden, W[2]);
    const dH1f       = matVecT(W[2], dH2_pre, nHidden, nHidden);
    // Skip gradient flows back to H0f
    const dH0f_skip  = new Float32Array(nHidden);
    for (let i = 0; i < nHidden; i++) dH0f_skip[i] = 0.1 * dH2_pre[i];

    // ── FiLM layer 1 (H1f = γ1 ⊙ H1 + β1) ───────────────────────────────
    // dH1  = dH1f ⊙ γ1
    const dH1       = new Float32Array(nHidden);
    const dGamma1   = new Float32Array(nHidden); // d(γ1_raw) = dH1f ⊙ H1
    const dBeta1    = dH1f;                      // d(β1) = dH1f
    for (let i = 0; i < nHidden; i++) {
      dH1[i]     = dH1f[i] * gamma1[i];
      dGamma1[i] = dH1f[i] * H1[i];             // grad into Wg1@ctx
    }
    const dWg1 = filmOuterGrad(dGamma1, ctx, Wg[1]);
    const dWb1 = filmOuterGrad(dBeta1,  ctx, Wb[1]);

    // ── Layer 1 (input was H0f) ───────────────────────────────────────────
    const dH1_pre    = reluBack(dH1, H1_pre);
    const { dW: dW1, db: db1 } = outerGrad(dH1_pre, H0f, nHidden, nHidden, W[1]);
    // Gradient into H0f from layer 1
    const dH0f_layer1 = matVecT(W[1], dH1_pre, nHidden, nHidden);

    // Combine H0f gradients (layer1 path + skip)
    const dH0f = new Float32Array(nHidden);
    for (let i = 0; i < nHidden; i++) dH0f[i] = dH0f_layer1[i] + dH0f_skip[i];

    // ── FiLM layer 0 (H0f = γ0 ⊙ H0 + β0) ───────────────────────────────
    const dH0       = new Float32Array(nHidden);
    const dGamma0   = new Float32Array(nHidden);
    const dBeta0    = dH0f;
    for (let i = 0; i < nHidden; i++) {
      dH0[i]     = dH0f[i] * gamma0[i];
      dGamma0[i] = dH0f[i] * H0[i];
    }
    const dWg0 = filmOuterGrad(dGamma0, ctx, Wg[0]);
    const dWb0 = filmOuterGrad(dBeta0,  ctx, Wb[0]);

    // ── Layer 0 ───────────────────────────────────────────────────────────
    const dH0_pre = reluBack(dH0, H0_pre);
    const { dW: dW0, db: db0 } = outerGrad(dH0_pre, inp, nHidden, nIn, W[0]);

    // Return in same order as _params: [W0,W1,W2,W3, Wg0,Wg1, Wb0,Wb1, b0,b1,b2,b3]
    return [dW0, dW1, dW2, dW3, dWg0, dWg1, dWb0, dWb1, db0, db1, db2, db3];
  }
}

// ── Build geological context (deprecated — use ConceptStore instead) ─────────
export function buildGeoContext(geoUnits, siteHistory, unitDescriptions) {
  console.warn('buildGeoContext is deprecated. Pass a ConceptStore to trainGeoImplicit instead.');
  return null;
}

// ── Train the neural implicit geological field ───────────────────────────────
// conceptStore: ConceptStore instance (or null for neutral/no-concept runs)
// The store's concept embeddings warp the coordinate space so that the output
// geometry of the implicit field directly reflects geological conceptual inputs.
// Returns { net, fourierEnc, conceptStore, warpedBounds, bounds, nUnits, unitCodes } or null
export async function trainGeoImplicit(boreholes, geoUnits, conceptStore, options = {}) {
  const {
    epochs          = 400,
    lr              = 0.01,
    lrMin           = 0.001,
    l2              = 0.001,
    samplesPerLayer = 6,
    onProgress      = null,
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
  // Add margin
  for (const [mn, mx] of [['minX','maxX'],['minY','maxY'],['minZ','maxZ']]) {
    const margin = (bounds[mx] - bounds[mn]) * 0.05 || 1;
    bounds[mn] -= margin; bounds[mx] += margin;
  }

  const fourierEnc = new FourierEncoder(L_FOURIER);
  const CONCEPT_DIM = 32;
  // nIn = Fourier positional (39) + concept context (32) = 71
  const nIn        = fourierEnc.outDim + CONCEPT_DIM;
  const nUnits     = geoUnits.length;
  const unitCodes  = geoUnits.map(u => u.code);
  const unitIdx    = {};
  geoUnits.forEach((u, i) => { unitIdx[u.code] = i; });

  // Compute global anisotropy tensor for the warped bounds used by the Fourier encoder.
  // Using a global average tensor keeps the normalisation bounds consistent while still
  // allowing per-point local warping to vary.
  const gTensor = conceptStore?.globalTensor() ?? { Ax: 1, Ay: 1, Az: 1 };
  const warpedBounds = {
    minX: bounds.minX / gTensor.Ax, maxX: bounds.maxX / gTensor.Ax,
    minY: bounds.minY / gTensor.Ay, maxY: bounds.maxY / gTensor.Ay,
    minZ: bounds.minZ / gTensor.Az, maxZ: bounds.maxZ / gTensor.Az,
  };

  const zeroCtx = new Float32Array(CONCEPT_DIM);

  // Build training samples from boreholes.
  // Each sample's positional features are warped by the local concept tensor at (x,y,z),
  // and the concept context vector is appended so the network learns to associate the
  // semantic geometry signal with the factual borehole observation.
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
        const wz  = zBase + t * (zTop - zBase);
        // Compute local concept context at this point; pass unitCode so concepts with
        // unit affinity restrictions apply only to the relevant geological units.
        const ctx    = conceptStore ? conceptStore.computeAt(bh.x, bh.y, wz, layer.unitCode) : null;
        const ctxVec = ctx?.vec ?? zeroCtx;
        const tensor = ctx?.tensor ?? gTensor;
        // Boost sample weight in concept-active zones: concept relevance provides
        // extra semantic certainty, so we want stronger gradient signal there.
        const conceptBoost = ctx ? Math.min(0.5, ctx.totalWeight * 0.3) : 0;
        const warped = { x: bh.x / tensor.Ax, y: bh.y / tensor.Ay, z: wz / tensor.Az };
        // Apply depth trend: deepening axes tilt the Z coordinate in normalised space.
        // This makes the implicit field naturally learn dipping contacts without extra samples.
        let warpedZ = warped.z;
        const trend = ctx?.trend;
        if (trend && (Math.abs(trend.dz_dxN) > 0.005 || Math.abs(trend.dz_dyN) > 0.005)) {
          const xN = 2 * (warped.x - warpedBounds.minX) / Math.max(1e-6, warpedBounds.maxX - warpedBounds.minX) - 1;
          const yN = 2 * (warped.y - warpedBounds.minY) / Math.max(1e-6, warpedBounds.maxY - warpedBounds.minY) - 1;
          warpedZ += trend.dz_dxN * xN + trend.dz_dyN * yN;
        }
        const pos = fourierEnc.encode(warped.x, warped.y, warpedZ, warpedBounds);
        const inp = new Float32Array(nIn);
        inp.set(pos);
        inp.set(ctxVec, fourierEnc.outDim);
        samples.push({ inp, target: ti, weight: wt + conceptBoost });
      }
    }
  }

  if (samples.length === 0) return null;

  // ── Concept-guided virtual samples ──────────────────────────────────────────
  // When a concept defines strong anisotropy (e.g., palaeochannel E-W), synthesise
  // interpolated training points along the concept's elongation axis between pairs
  // of real borehole observations of the same unit. This directly reinforces lateral
  // continuity in the concept direction without adding hard constraints.
  if (conceptStore && !conceptStore.isEmpty) {
    const gT = conceptStore.globalTensor();
    // Only synthesise if there's meaningful anisotropy (>1.5× in any axis)
    if (Math.max(gT.Ax, gT.Ay) > 1.5) {
      const unitBHs = {};  // { unitIdx → [{x, y, z, unitCode}] }
      for (const bh of boreholes) {
        for (const layer of (bh.layers ?? [])) {
          const ti = unitIdx[layer.unitCode];
          if (ti === undefined) continue;
          const zMid = bh.groundLevel - (layer.top + layer.base) / 2;
          if (!unitBHs[ti]) unitBHs[ti] = [];
          unitBHs[ti].push({ x: bh.x, y: bh.y, z: zMid, unitCode: layer.unitCode });
        }
      }
      const N_SYNTH = 2; // points along the interpolation between each pair
      for (const [ti, pts] of Object.entries(unitBHs)) {
        if (pts.length < 2) continue;
        // Connect nearby pairs (within 3× site span / concept anisotropy)
        const spanX = (bounds.maxX - bounds.minX) / gT.Ax;
        const spanY = (bounds.maxY - bounds.minY) / gT.Ay;
        const maxDist = Math.max(spanX, spanY) * 0.7;
        for (let a = 0; a < pts.length; a++) {
          for (let b = a + 1; b < pts.length; b++) {
            const pa = pts[a], pb = pts[b];
            const dist = Math.hypot(pa.x - pb.x, pa.y - pb.y);
            if (dist > maxDist) continue;
            for (let s = 1; s <= N_SYNTH; s++) {
              const t   = s / (N_SYNTH + 1);
              const sx  = pa.x + t * (pb.x - pa.x);
              const sy  = pa.y + t * (pb.y - pa.y);
              const sz  = pa.z + t * (pb.z - pa.z);
              const ctx = conceptStore.computeAt(sx, sy, sz, pa.unitCode);
              // Only add virtual samples where concept is active (relevance > 0.2)
              if (ctx.totalWeight < 0.2) continue;
              const tensor = ctx.tensor;
              const warpedPt = { x: sx / tensor.Ax, y: sy / tensor.Ay, z: sz / tensor.Az };
              let warpedPtZ = warpedPt.z;
              const vTrend = ctx.trend;
              if (vTrend && (Math.abs(vTrend.dz_dxN) > 0.005 || Math.abs(vTrend.dz_dyN) > 0.005)) {
                const xN = 2 * (warpedPt.x - warpedBounds.minX) / Math.max(1e-6, warpedBounds.maxX - warpedBounds.minX) - 1;
                const yN = 2 * (warpedPt.y - warpedBounds.minY) / Math.max(1e-6, warpedBounds.maxY - warpedBounds.minY) - 1;
                warpedPtZ += vTrend.dz_dxN * xN + vTrend.dz_dyN * yN;
              }
              const pos  = fourierEnc.encode(warpedPt.x, warpedPt.y, warpedPtZ, warpedBounds);
              const inp  = new Float32Array(nIn);
              inp.set(pos);
              inp.set(ctx.vec, fourierEnc.outDim);
              // Virtual samples get reduced weight (0.4) so real BH data dominates
              samples.push({ inp, target: parseInt(ti), weight: 0.4 * ctx.totalWeight });
            }
          }
        }
      }
    }
  }

  const nRealSamples    = samples.filter(s => s.weight >= 0.4).length;
  const nVirtualSamples = samples.length - nRealSamples;
  if (onProgress) onProgress(0, 0, { nSamples: samples.length, nReal: nRealSamples, nVirtual: nVirtualSamples });

  const net = new GeoImplicitNet(nIn, 64, nUnits, fourierEnc.outDim);
  const opt = new AdamOpt(net._params, lr);

  // FiLM warmup: for the first filmWarmup fraction of training, gradually scale up
  // FiLM contributions so the base network learns the borehole distribution first,
  // then FiLM modulation refines the geometry toward the conceptual model.
  const FILM_WARMUP = 0.25; // fraction of epochs for warmup
  const filmParamStart = 4; // params 4-7 are Wg0,Wg1,Wb0,Wb1 (FiLM weights)
  const filmParamEnd   = 8;
  const filmOrigScales = net._params.slice(filmParamStart, filmParamEnd).map(p => p.slice());

  for (let ep = 0; ep < epochs; ep++) {
    opt.setLr(lrMin + 0.5 * (lr - lrMin) * (1 + Math.cos(Math.PI * ep / epochs)));

    // FiLM warmup: scale film weights by ramp factor (0→1 over first 25% of epochs)
    const filmScale = Math.min(1, ep / Math.max(1, FILM_WARMUP * epochs));
    for (let pi = filmParamStart; pi < filmParamEnd; pi++) {
      const orig = filmOrigScales[pi - filmParamStart];
      const cur  = net._params[pi];
      for (let i = 0; i < cur.length; i++) cur[i] = orig[i] * filmScale;
    }

    // Fisher-Yates shuffle
    for (let i = samples.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [samples[i], samples[j]] = [samples[j], samples[i]];
    }
    let totalLoss = 0;
    for (const s of samples) {
      const act  = net.forward(s.inp);
      const loss = -Math.log(Math.max(act.probs[s.target], 1e-9));
      totalLoss += s.weight * loss;
      // Pass sample weight so high-confidence concept regions drive stronger gradient updates
      opt.step(net.backward(s.inp, act, s.target, l2, s.weight));
    }

    // After warmup ends, let Adam update FiLM weights naturally (don't rescale anymore)
    if (ep === Math.floor(FILM_WARMUP * epochs)) {
      for (let pi = filmParamStart; pi < filmParamEnd; pi++) {
        filmOrigScales[pi - filmParamStart].set(net._params[pi]);
      }
    }

    if (onProgress && ep % 20 === 0) {
      onProgress(ep / epochs, totalLoss / samples.length);
      await new Promise(r => setTimeout(r, 0));
    }
  }

  if (onProgress) onProgress(1, 0);
  return { net, fourierEnc, conceptStore, warpedBounds, bounds, nUnits, unitCodes, CONCEPT_DIM, fourierDim: fourierEnc.outDim };
}

// ── Infer voxel grid from trained model ─────────────────────────────────────
// grid must have { nx, ny, nz, cellSize, cellHeight, origin: {x,y,z} }
// options.sectionPlanes: [{fence, localKwVec}] for spatially-local section context
// conceptStore: ConceptStore instance (or null) — same store passed to trainGeoImplicit.
// At each voxel the store's computeAt() returns the concept context that warps coordinates
// exactly as during training, ensuring inference geometry matches the trained distribution.
export function inferGeoImplicit(trained, grid, geoUnits, conceptStore) {
  // Backward compat: if old options object passed, ignore options silently
  if (conceptStore && typeof conceptStore !== 'object') conceptStore = null;
  if (conceptStore && typeof conceptStore.computeAt !== 'function') conceptStore = null;

  const { net, fourierEnc, warpedBounds, bounds, unitCodes, CONCEPT_DIM } = trained;
  const cdim      = CONCEPT_DIM ?? 32;
  const nIn       = fourierEnc.outDim + cdim;
  const zeroCtx   = new Float32Array(cdim);
  const gTensor   = conceptStore?.globalTensor() ?? { Ax: 1, Ay: 1, Az: 1 };
  const useBounds = warpedBounds ?? bounds; // use warped bounds if available

  const { nx, ny, nz, cellSize, cellHeight, origin } = grid;
  const total = nx * ny * nz;

  const unitIds         = new Uint8Array(total);
  const certainty       = new Float32Array(total);
  const blendUnitIds    = new Uint8Array(total);
  const blendRatios     = new Float32Array(total);
  // Per-voxel concept semantic dominance [0..1]: 0 = data-driven, 1 = concept-driven
  // Used for traceability visualisation (heat-map overlay).
  const conceptInfluence = new Float32Array(total);

  const codeToId = {};
  geoUnits.forEach(u => { codeToId[u.code] = u.id; });

  // Column-context cache: concept context only varies with (worldX, worldY) since
  // all current domain types use only horizontal position for relevance.
  // Caching saves nz concept-context computations per column during inference.
  const colCtxCache = new Map(); // key = `${ix},${iy}`

  for (let iz = nz - 1; iz >= 0; iz--) {
    const worldZ = origin.y + iz * cellHeight + cellHeight * 0.5;
    for (let iy = 0; iy < ny; iy++) {
      const worldY = origin.z + iy * cellSize + cellSize * 0.5;
      for (let ix = 0; ix < nx; ix++) {
        const worldX = origin.x + ix * cellSize + cellSize * 0.5;
        const idx    = ix + iy * nx + iz * nx * ny;

        // Fetch or compute concept context — cached per (ix,iy) column
        const colKey = ix * ny + iy;
        let ctx = colCtxCache.get(colKey);
        if (ctx === undefined) {
          ctx = conceptStore ? conceptStore.computeAt(worldX, worldY, worldZ) : null;
          colCtxCache.set(colKey, ctx);
        }
        const ctxVec = ctx?.vec ?? zeroCtx;
        const tensor = ctx?.tensor ?? gTensor;

        // Warp coordinates by concept anisotropy tensor, then Fourier encode
        const wx  = worldX / tensor.Ax;
        const wy  = worldY / tensor.Ay;
        let   wz  = worldZ / tensor.Az;
        // Apply depth trend: tilt Z so the field naturally dips in the predicted direction
        const iTrend = ctx?.trend;
        if (iTrend && (Math.abs(iTrend.dz_dxN) > 0.005 || Math.abs(iTrend.dz_dyN) > 0.005)) {
          const xN = 2 * (wx - useBounds.minX) / Math.max(1e-6, useBounds.maxX - useBounds.minX) - 1;
          const yN = 2 * (wy - useBounds.minY) / Math.max(1e-6, useBounds.maxY - useBounds.minY) - 1;
          wz += iTrend.dz_dxN * xN + iTrend.dz_dyN * yN;
        }
        const pos = fourierEnc.encode(wx, wy, wz, useBounds);

        const inp = new Float32Array(nIn);
        inp.set(pos);
        inp.set(ctxVec, fourierEnc.outDim);

        const probs = net.predict(inp);

        // Top-1 and top-2 classes
        let b1 = 0, b2 = 1;
        if (probs[1] > probs[0]) { b1 = 1; b2 = 0; }
        for (let u = 2; u < probs.length; u++) {
          if      (probs[u] > probs[b1]) { b2 = b1; b1 = u; }
          else if (probs[u] > probs[b2])  { b2 = u; }
        }

        // Certainty: base separation between top-2 probs, boosted slightly
        // where strong concepts are active (they provide extra epistemic confidence).
        const baseCert = 0.5 + probs[b1] - probs[b2];
        const conceptBoost = ctx ? Math.min(0.1, ctx.totalWeight * 0.08) : 0;

        unitIds[idx]          = codeToId[unitCodes[b1]] ?? 0;
        certainty[idx]        = Math.max(0.05, Math.min(1, baseCert + conceptBoost));
        blendUnitIds[idx]     = codeToId[unitCodes[b2]] ?? 0;
        blendRatios[idx]      = probs[b2];
        conceptInfluence[idx] = ctx ? Math.min(1, ctx.totalWeight) : 0;
      }
    }
  }

  return { unitIds, certainty, blendUnitIds, blendRatios, conceptInfluence };
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
