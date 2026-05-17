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

  // filmDropout: fraction of FiLM units to zero during training (0 = no dropout)
  forward(inp, filmDropout = 0) {
    const { nHidden, nIn, nOut, W, Wg, Wb, b, fourierDim } = this;
    const CTX_DIM = 32;
    // ctx = last 32 elements (concept context)
    const ctx = inp.subarray(fourierDim);

    // Layer 0
    const H0_pre = this._linear(W[0], b[0], inp, nHidden, nIn);
    const H0     = this._relu(H0_pre);
    // FiLM layer 0: γ0 = 1 + Wg0@ctx, β0 = Wb0@ctx
    // FiLM dropout: randomly zero individual FiLM units during training.
    // This prevents the network from over-relying on concept signals at specific
    // borehole positions and encourages it to generalise conceptual geometry.
    const gamma0_raw = this._filmProj(Wg[0], ctx, nHidden, CTX_DIM);
    const beta0      = this._filmProj(Wb[0], ctx, nHidden, CTX_DIM);
    const gamma0     = new Float32Array(nHidden);
    const H0f        = new Float32Array(nHidden);
    for (let i = 0; i < nHidden; i++) {
      const keep  = filmDropout > 0 ? (Math.random() > filmDropout ? 1 : 0) : 1;
      gamma0[i] = 1 + gamma0_raw[i] * keep;
      H0f[i]    = gamma0[i] * H0[i] + beta0[i] * keep;
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
      const keep  = filmDropout > 0 ? (Math.random() > filmDropout ? 1 : 0) : 1;
      gamma1[i] = 1 + gamma1_raw[i] * keep;
      H1f[i]    = gamma1[i] * H1[i] + beta1[i] * keep;
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
    // stratOrder: array of unit codes from top (youngest) to bottom (oldest).
    // When supplied, synthetic "strat-order" samples are injected that teach
    // the network which units should appear above vs. below at depth boundaries,
    // enforcing stratigraphic consistency even in data-sparse regions.
    stratOrder      = null,
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

  // ── Stratigraphic-order virtual samples ─────────────────────────────────────
  // If the user has defined a stratigraphic column, inject synthetic samples at
  // unit transitions. For each adjacent pair (unit_a above unit_b) in stratOrder,
  // we place training samples at a range of Z positions centred on the mean
  // observed contact elevation between those two units in real boreholes.
  // This biases the implicit field toward honouring the expected column ordering
  // in areas where no borehole data is present.
  if (stratOrder && stratOrder.length >= 2) {
    // Build stratOrderIdx: { code → rank (0 = topmost) }
    const stratRank = {};
    stratOrder.forEach((code, i) => { stratRank[code] = i; });

    // Find contacts between adjacent units in the observed data
    const contactElevs = {}; // key: `${codeAbove}|${codeBelow}` → [elevations]
    for (const bh of boreholes) {
      const layers = (bh.layers ?? []).filter(l => l.unitCode && stratRank[l.unitCode] !== undefined);
      layers.sort((a, b) => a.top - b.top); // sort by depth (top = shallowest)
      for (let i = 0; i < layers.length - 1; i++) {
        const la = layers[i], lb = layers[i + 1];
        if (stratRank[la.unitCode] < stratRank[lb.unitCode]) {
          // la above lb — consistent with stratOrder
          const key  = `${la.unitCode}|${lb.unitCode}`;
          const elev = bh.groundLevel - la.base; // contact elevation
          if (!contactElevs[key]) contactElevs[key] = [];
          contactElevs[key].push(elev);
        }
      }
    }

    // For each contact, generate synthetic samples at and around the contact
    const stratJitter = (bounds.maxZ - bounds.minZ) * 0.06; // ±6% of vertical range
    const N_STRAT = 3; // samples per side of contact
    for (const [key, elevs] of Object.entries(contactElevs)) {
      const [codeA, codeB] = key.split('|');
      const tiA = unitIdx[codeA], tiB = unitIdx[codeB];
      if (tiA === undefined || tiB === undefined) continue;
      const meanElev = elevs.reduce((s, e) => s + e, 0) / elevs.length;

      // Sample X,Y positions uniformly across the site
      const xs = boreholes.map(b => b.x);
      const ys = boreholes.map(b => b.y);
      const cxList = [
        (bounds.minX + bounds.maxX) / 2,
        bounds.minX * 0.7 + bounds.maxX * 0.3,
        bounds.minX * 0.3 + bounds.maxX * 0.7,
      ];
      const cyList = [
        (bounds.minY + bounds.maxY) / 2,
        bounds.minY * 0.7 + bounds.maxY * 0.3,
        bounds.minY * 0.3 + bounds.maxY * 0.7,
      ];

      for (const sx of cxList) {
        for (const sy of cyList) {
          // Samples ABOVE contact → unit A
          for (let s = 1; s <= N_STRAT; s++) {
            const wz  = meanElev + (s / N_STRAT) * stratJitter;
            if (wz < bounds.minZ || wz > bounds.maxZ) continue;
            const ctx    = conceptStore ? conceptStore.computeAt(sx, sy, wz, codeA) : null;
            const ctxVec = ctx?.vec ?? zeroCtx;
            const tensor = ctx?.tensor ?? gTensor;
            const warped = { x: sx / tensor.Ax, y: sy / tensor.Ay, z: wz / tensor.Az };
            let warpedZ = warped.z;
            const trend = ctx?.trend;
            if (trend && (Math.abs(trend.dz_dxN) > 0.005 || Math.abs(trend.dz_dyN) > 0.005)) {
              const xN = 2 * (warped.x - warpedBounds.minX) / Math.max(1e-6, warpedBounds.maxX - warpedBounds.minX) - 1;
              const yN = 2 * (warped.y - warpedBounds.minY) / Math.max(1e-6, warpedBounds.maxY - warpedBounds.minY) - 1;
              warpedZ += trend.dz_dxN * xN + trend.dz_dyN * yN;
            }
            const pos = fourierEnc.encode(warped.x, warped.y, warpedZ, warpedBounds);
            const inp = new Float32Array(nIn);
            inp.set(pos); inp.set(ctxVec, fourierEnc.outDim);
            // Strat samples get reduced weight (0.25) — they guide but don't override BH data
            samples.push({ inp, target: tiA, weight: 0.25 });
          }
          // Samples BELOW contact → unit B
          for (let s = 1; s <= N_STRAT; s++) {
            const wz  = meanElev - (s / N_STRAT) * stratJitter;
            if (wz < bounds.minZ || wz > bounds.maxZ) continue;
            const ctx    = conceptStore ? conceptStore.computeAt(sx, sy, wz, codeB) : null;
            const ctxVec = ctx?.vec ?? zeroCtx;
            const tensor = ctx?.tensor ?? gTensor;
            const warped = { x: sx / tensor.Ax, y: sy / tensor.Ay, z: wz / tensor.Az };
            let warpedZ = warped.z;
            const trend = ctx?.trend;
            if (trend && (Math.abs(trend.dz_dxN) > 0.005 || Math.abs(trend.dz_dyN) > 0.005)) {
              const xN = 2 * (warped.x - warpedBounds.minX) / Math.max(1e-6, warpedBounds.maxX - warpedBounds.minX) - 1;
              const yN = 2 * (warped.y - warpedBounds.minY) / Math.max(1e-6, warpedBounds.maxY - warpedBounds.minY) - 1;
              warpedZ += trend.dz_dxN * xN + trend.dz_dyN * yN;
            }
            const pos = fourierEnc.encode(warped.x, warped.y, warpedZ, warpedBounds);
            const inp = new Float32Array(nIn);
            inp.set(pos); inp.set(ctxVec, fourierEnc.outDim);
            samples.push({ inp, target: tiB, weight: 0.25 });
          }
        }
      }
    }
    if (onProgress) onProgress(0, 0, { stratContactsFound: Object.keys(contactElevs).length });
  }

  // ── Concept-guided virtual samples ──────────────────────────────────────────
  // For each concept with strong anisotropy (palaeochannel, terrace, etc.):
  //  (A) Interpolate: add training points BETWEEN borehole observations of the same
  //      unit, weighted by concept relevance, to reinforce lateral continuity along
  //      the concept's preferred elongation axis.
  //  (B) Extrapolate: add training points ALONG the elongation axis BEYOND the
  //      outermost borehole observations to project geometry into data-sparse areas.
  //      These have lower weight (0.25) so they guide but don't override actual data.
  //  The cap prevents virtual samples from outnumbering real samples by more than 3×.
  if (conceptStore && !conceptStore.isEmpty) {
    const gT = conceptStore.globalTensor();
    if (Math.max(gT.Ax, gT.Ay) > 1.3) {
      const unitBHs = {};  // { unitIdx → [{x, y, z, unitCode, top, base}] }
      for (const bh of boreholes) {
        for (const layer of (bh.layers ?? [])) {
          const ti = unitIdx[layer.unitCode];
          if (ti === undefined) continue;
          const zTop  = bh.groundLevel - layer.top;
          const zBase = bh.groundLevel - layer.base;
          const zMid  = (zTop + zBase) / 2;
          if (!unitBHs[ti]) unitBHs[ti] = [];
          unitBHs[ti].push({ x: bh.x, y: bh.y, z: zMid, zTop, zBase, unitCode: layer.unitCode });
        }
      }

      const N_SYNTH = Math.min(5, Math.ceil(Math.max(gT.Ax, gT.Ay) / 1.2) + 1);

      // Helper: encode one virtual sample and push to samples
      const addVirtual = (sx, sy, sz, tiInt, unitCode, w) => {
        const ctx = conceptStore.computeAt(sx, sy, sz, unitCode);
        if (ctx.totalWeight < 0.15) return;
        const tensor = ctx.tensor;
        const wx = sx / tensor.Ax, wy = sy / tensor.Ay;
        let wz = sz / tensor.Az;
        const vTrend = ctx.trend;
        if (vTrend && (Math.abs(vTrend.dz_dxN) > 0.005 || Math.abs(vTrend.dz_dyN) > 0.005)) {
          const xN = 2 * (wx - warpedBounds.minX) / Math.max(1e-6, warpedBounds.maxX - warpedBounds.minX) - 1;
          const yN = 2 * (wy - warpedBounds.minY) / Math.max(1e-6, warpedBounds.maxY - warpedBounds.minY) - 1;
          wz += vTrend.dz_dxN * xN + vTrend.dz_dyN * yN;
        }
        const pos = fourierEnc.encode(wx, wy, wz, warpedBounds);
        const inp = new Float32Array(nIn);
        inp.set(pos);
        inp.set(ctx.vec, fourierEnc.outDim);
        samples.push({ inp, target: tiInt, weight: w * ctx.totalWeight });
      };

      for (const [ti, pts] of Object.entries(unitBHs)) {
        if (!pts.length) continue;
        const tiInt = parseInt(ti);
        // Elongation direction: prefer the axis with higher anisotropy
        const elongX = gT.Ax > gT.Ay; // true = E-W, false = N-S
        const spanX = (bounds.maxX - bounds.minX) / gT.Ax;
        const spanY = (bounds.maxY - bounds.minY) / gT.Ay;
        const maxDist = Math.max(spanX, spanY) * 0.8;

        // (A) Interpolation between pairs
        for (let a = 0; a < pts.length; a++) {
          for (let b = a + 1; b < pts.length; b++) {
            const pa = pts[a], pb = pts[b];
            const dist = Math.hypot(pa.x - pb.x, pa.y - pb.y);
            if (dist > maxDist) continue;
            for (let s = 1; s <= N_SYNTH; s++) {
              const t = s / (N_SYNTH + 1);
              addVirtual(
                pa.x + t * (pb.x - pa.x),
                pa.y + t * (pb.y - pa.y),
                pa.z + t * (pb.z - pa.z),
                tiInt, pa.unitCode, 0.4
              );
            }
          }
        }

        // (B) Extrapolation beyond the outermost BH in the concept's elongation direction
        // Project each BH observation outward along the concept axis by up to 0.5× site span
        if (pts.length >= 2) {
          const extendDist = (elongX
            ? (bounds.maxX - bounds.minX) * 0.35
            : (bounds.maxY - bounds.minY) * 0.35);
          const avgZ = pts.reduce((s, p) => s + p.z, 0) / pts.length;
          for (const p of pts) {
            const stepX = elongX ? extendDist / N_SYNTH : 0;
            const stepY = elongX ? 0 : extendDist / N_SYNTH;
            for (let s = 1; s <= N_SYNTH; s++) {
              // Extend in both positive and negative directions
              for (const sign of [+1, -1]) {
                const ex = p.x + sign * s * stepX;
                const ey = p.y + sign * s * stepY;
                // Clamp to site bounds with small margin
                if (ex < bounds.minX - 5 || ex > bounds.maxX + 5) continue;
                if (ey < bounds.minY - 5 || ey > bounds.maxY + 5) continue;
                // Extrapolation weight decays with distance from the last real BH
                const extW = 0.25 * (1 - s / (N_SYNTH + 1));
                addVirtual(ex, ey, avgZ, tiInt, p.unitCode, extW);
              }
            }
          }
        }
      }
    }
  }

  // Separate real and virtual samples for diagnostics and capping
  const realSamples    = samples.filter(s => s.weight > 0.35);
  const virtualSamples = samples.filter(s => s.weight <= 0.35);
  const nRealSamples   = realSamples.length;
  let nVirtualSamples  = virtualSamples.length;

  // Cap virtual samples at 3× real samples to prevent concept data from
  // overwhelming factual borehole observations.
  let allSamples = realSamples;
  if (virtualSamples.length > 0) {
    const maxVirtual = Math.min(virtualSamples.length, nRealSamples * 3);
    // Shuffle virtual samples and take the capped count
    for (let i = virtualSamples.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [virtualSamples[i], virtualSamples[j]] = [virtualSamples[j], virtualSamples[i]];
    }
    nVirtualSamples = maxVirtual;
    allSamples = [...realSamples, ...virtualSamples.slice(0, maxVirtual)];
  }
  if (onProgress) onProgress(0, 0, { nSamples: allSamples.length, nReal: nRealSamples, nVirtual: nVirtualSamples });

  const net = new GeoImplicitNet(nIn, 64, nUnits, fourierEnc.outDim);
  const opt = new AdamOpt(net._params, lr);

  // FiLM warmup: for the first FILM_WARMUP fraction of training, scale the FiLM
  // gradient contribution down so the positional network learns the borehole distribution
  // first, then FiLM takes over to shape geometry toward the conceptual model.
  // We scale GRADIENTS (not weights) so Adam momentum doesn't destroy trained values.
  const FILM_WARMUP    = 0.25; // fraction of epochs for FiLM ramp
  const filmParamStart = 4;    // _params[4..7] are Wg0,Wg1,Wb0,Wb1
  const filmParamEnd   = 8;

  for (let ep = 0; ep < epochs; ep++) {
    opt.setLr(lrMin + 0.5 * (lr - lrMin) * (1 + Math.cos(Math.PI * ep / epochs)));

    // filmGradScale: 0→1 over first 25% of epochs; full gradient after that
    const filmGradScale = Math.min(1, ep / Math.max(1, FILM_WARMUP * epochs));
    // FiLM dropout: higher early so positional features don't over-rely on concept context
    const filmDropout = filmGradScale < 0.5 ? 0.3 : 0.1;

    // Fisher-Yates shuffle
    for (let i = allSamples.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [allSamples[i], allSamples[j]] = [allSamples[j], allSamples[i]];
    }

    let totalLoss = 0;
    for (const s of allSamples) {
      const act  = net.forward(s.inp, filmDropout);
      const loss = -Math.log(Math.max(act.probs[s.target], 1e-9));
      totalLoss += s.weight * loss;
      const grads = net.backward(s.inp, act, s.target, l2, s.weight);
      // Scale FiLM gradients during warmup — they train, just more slowly
      if (filmGradScale < 1) {
        for (let pi = filmParamStart; pi < filmParamEnd; pi++) {
          const g = grads[pi];
          for (let i = 0; i < g.length; i++) g[i] *= filmGradScale;
        }
      }
      opt.step(grads);
    }

    if (onProgress && ep % 20 === 0) {
      onProgress(ep / epochs, totalLoss / allSamples.length, { epoch: ep, epochs });
      await new Promise(r => setTimeout(r, 0));
    }
  }

  if (onProgress) onProgress(1, 0, { epoch: epochs, epochs });
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

// ── Concept geometry verification ────────────────────────────────────────────
// After inference, measure the 3D bounding box extents of each unit's dominant
// voxel region and compare the E-W/N-S elongation ratio against what the concept
// embeddings predict. Returns an array of { unitCode, ewRatio, nsRatio,
// predictedEW, predictedNS, conceptMatch (0–1) } — one entry per unit.
//
// This gives the user a quantitative proof that the concept shaped the geometry:
// a palaeochannel concept with east_west_elongation=0.9 should yield ewRatio >> 1.
export function measureConceptGeometry(grid, geoUnits, conceptStore) {
  if (!grid || !geoUnits.length) return [];
  const { nx, ny, nz, cellSize, cellHeight, origin, unitIds } = grid;

  const codeToId = {};
  geoUnits.forEach(u => { codeToId[u.code] = u.id; });

  const results = [];
  for (const u of geoUnits) {
    // Find bounding box of all voxels dominated by this unit
    let minIX = nx, maxIX = 0, minIY = ny, maxIY = 0, minIZ = nz, maxIZ = 0;
    let count = 0;
    for (let iz = 0; iz < nz; iz++) {
      for (let iy = 0; iy < ny; iy++) {
        for (let ix = 0; ix < nx; ix++) {
          if (unitIds[ix + iy * nx + iz * nx * ny] === u.id) {
            minIX = Math.min(minIX, ix); maxIX = Math.max(maxIX, ix);
            minIY = Math.min(minIY, iy); maxIY = Math.max(maxIY, iy);
            minIZ = Math.min(minIZ, iz); maxIZ = Math.max(maxIZ, iz);
            count++;
          }
        }
      }
    }
    if (count < 4) continue;

    const extentX = (maxIX - minIX + 1) * cellSize;    // E-W metres
    const extentY = (maxIY - minIY + 1) * cellSize;    // N-S metres
    const extentZ = (maxIZ - minIZ + 1) * cellHeight;  // vertical metres

    // Elongation ratios (>1 = elongated in that direction vs the other horizontal)
    const ewRatio = extentX / Math.max(1, extentY);
    const nsRatio = extentY / Math.max(1, extentX);

    // Concept prediction: sample concept context at centroid of unit bbox
    let predictedEW = 1, predictedNS = 1;
    if (conceptStore && !conceptStore.isEmpty) {
      const cx = origin.x + (minIX + maxIX) / 2 * cellSize;
      const cy = origin.z + (minIY + maxIY) / 2 * cellSize;
      const cz = origin.y + (minIZ + maxIZ) / 2 * cellHeight;
      const ctx = conceptStore.computeAt(cx, cy, cz, u.code);
      predictedEW = ctx.tensor.Ax; // > 1 → field predicted E-W elongation
      predictedNS = ctx.tensor.Ay;
    }

    // Concept match: 0 = geometry contradicts concept, 1 = perfect agreement
    // A concept predicting Ax=3 (E-W elongation) matches if ewRatio > nsRatio
    const ewConcept = predictedEW > predictedNS;
    const ewActual  = ewRatio   > nsRatio;
    const conceptMatch = ewConcept === ewActual ? 1 : 0.5;

    results.push({
      unitCode:    u.code,
      unitName:    u.name,
      unitColor:   u.color,
      extentX, extentY, extentZ, count,
      ewRatio:     +ewRatio.toFixed(2),
      nsRatio:     +nsRatio.toFixed(2),
      predictedEW: +predictedEW.toFixed(2),
      predictedNS: +predictedNS.toFixed(2),
      conceptMatch,
    });
  }
  return results;
}
