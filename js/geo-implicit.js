// ── Geo-Implicit: Neural Implicit Geological Field ──────────────────────────
// Architecture:
//   FourierEncoder(x,y,z)    → 39-dim positional features (L=6)
//   GeoKeywordEncoder(text)  → N_vocab-dim binary keyword presence
//   Concat → 4-layer MLP (nIn→64→64→64→nUnits) with skip H0→H2 (×0.1)
// Training: Adam + cosine-annealed LR, cross-entropy + L2
// Oracle:   BFS cluster detection → Claude reasons over uncertain regions

import { prepareShapesForSDF, evaluateAllSDFs } from './geo-shapes.js';

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

// ── 4-Layer MLP with skip connection H0 → H2 ────────────────────────────────
// Simple MLP: Fourier positional features + optional SDF inputs → geo unit probs.
class GeoImplicitNet {
  constructor(nIn, nHidden, nOut, fourierDim = 39) {
    this.nIn = nIn; this.nHidden = nHidden; this.nOut = nOut;
    this.fourierDim = fourierDim;
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
    // _params order: [W0,W1,W2,W3, b0,b1,b2,b3]
    this._params = [...this.W, ...this.b];
  }

  getParams() { return this._params; }

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

  forward(inp) {
    const { nHidden, nIn, nOut, W, b } = this;
    const H0_pre = this._linear(W[0], b[0], inp, nHidden, nIn);
    const H0     = this._relu(H0_pre);
    const H1_pre = this._linear(W[1], b[1], H0,  nHidden, nHidden);
    const H1     = this._relu(H1_pre);
    // Layer 2 with skip from H0 (×0.1)
    const H2_raw = this._linear(W[2], b[2], H1, nHidden, nHidden);
    const H2_pre = new Float32Array(nHidden);
    for (let i = 0; i < nHidden; i++) H2_pre[i] = H2_raw[i] + 0.1 * H0[i];
    const H2     = this._relu(H2_pre);
    const logits = this._linear(W[3], b[3], H2,  nOut,    nHidden);
    const probs  = this._softmax(logits);
    return { H0_pre, H0, H1_pre, H1, H2_pre, H2, logits, probs };
  }

  predict(inp) { return this.forward(inp).probs; }

  // Returns grads in same order as _params = [dW0,dW1,dW2,dW3, db0,db1,db2,db3]
  // sampleWeight scales the gradient — allows high-confidence samples to drive stronger updates.
  backward(inp, act, targetIdx, l2 = 0.001, sampleWeight = 1.0) {
    const { nHidden, nIn, nOut, W, b } = this;
    const { H0_pre, H0, H1_pre, H1, H2_pre, H2, probs } = act;

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

    // Layer 3
    const { dW: dW3, db: db3 } = outerGrad(dLogits, H2, nOut, nHidden, W[3]);
    const dH2 = matVecT(W[3], dLogits, nOut, nHidden);
    // Layer 2
    const dH2_pre = reluBack(dH2, H2_pre);
    const { dW: dW2, db: db2 } = outerGrad(dH2_pre, H1, nHidden, nHidden, W[2]);
    const dH1      = matVecT(W[2], dH2_pre, nHidden, nHidden);
    // Skip connection from H0
    const dH0_skip = new Float32Array(nHidden);
    for (let i = 0; i < nHidden; i++) dH0_skip[i] = 0.1 * dH2_pre[i];
    // Layer 1
    const dH1_pre = reluBack(dH1, H1_pre);
    const { dW: dW1, db: db1 } = outerGrad(dH1_pre, H0, nHidden, nHidden, W[1]);
    const dH0_layer1 = matVecT(W[1], dH1_pre, nHidden, nHidden);
    // Layer 0
    const dH0_combined = new Float32Array(nHidden);
    for (let i = 0; i < nHidden; i++) dH0_combined[i] = dH0_layer1[i] + dH0_skip[i];
    const dH0_pre = reluBack(dH0_combined, H0_pre);
    const { dW: dW0, db: db0 } = outerGrad(dH0_pre, inp, nHidden, nIn, W[0]);

    // Return order matches _params: [W0,W1,W2,W3, b0,b1,b2,b3]
    return [dW0, dW1, dW2, dW3, db0, db1, db2, db3];
  }
}

// ── Build geological context (deprecated — use ConceptStore instead) ─────────
export function buildGeoContext(geoUnits, siteHistory, unitDescriptions) {
  console.warn('buildGeoContext is deprecated. Pass a ConceptStore to trainGeoImplicit instead.');
  return null;
}

// ── TF.js GPU-accelerated training ───────────────────────────────────────────
// Replaces the manual sample-by-sample Adam loop with batched GPU tensor ops.
// Runs full-batch gradient descent (all samples per epoch) via WebGL backend.
// Weights are synced back to the JS net after training for JS inference compat.
async function _trainWithTF(net, allSamples, opts, onProgress) {
  const tf = window.tf;
  const { epochs, lr, lrMin, l2 } = opts;
  const { nHidden, nIn, nOut } = net;
  const N   = allSamples.length;

  // Pack all samples into flat typed arrays (created once, reused every epoch)
  const inpArr = new Float32Array(N * nIn);
  const tgtArr = new Int32Array(N);
  const wtArr  = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    inpArr.set(allSamples[i].inp, i * nIn);
    tgtArr[i] = allSamples[i].target;
    wtArr[i]  = allSamples[i].weight ?? 1;
  }

  // Unique name suffix prevents variable registry collisions on repeated runs
  const uid = Math.random().toString(36).slice(2, 7);
  const nm  = s => `${s}_${uid}`;

  // Create named tf.Variables from the net's initial JS weight arrays
  const V = {
    W0:  tf.variable(tf.tensor2d(net.W[0],  [nHidden, nIn]),     true, nm('W0')),
    W1:  tf.variable(tf.tensor2d(net.W[1],  [nHidden, nHidden]), true, nm('W1')),
    W2:  tf.variable(tf.tensor2d(net.W[2],  [nHidden, nHidden]), true, nm('W2')),
    W3:  tf.variable(tf.tensor2d(net.W[3],  [nOut,    nHidden]), true, nm('W3')),
    b0:  tf.variable(tf.tensor1d(net.b[0]),                      true, nm('b0')),
    b1:  tf.variable(tf.tensor1d(net.b[1]),                      true, nm('b1')),
    b2:  tf.variable(tf.tensor1d(net.b[2]),                      true, nm('b2')),
    b3:  tf.variable(tf.tensor1d(net.b[3]),                      true, nm('b3')),
  };
  const allVars = Object.values(V);

  // Persistent input tensors — created once, reused each epoch (no re-allocation)
  const inpT = tf.tensor2d(inpArr, [N, nIn]);
  const tgtT = tf.tensor1d(tgtArr, 'int32');
  const wtT  = tf.tensor1d(wtArr);

  const opt       = tf.train.adam(lr, 0.9, 0.999, 1e-8);
  const warmupEps = Math.max(1, Math.round(epochs * 0.05));
  const nOut_     = nOut; // local for closure

  for (let ep = 0; ep < epochs; ep++) {
    // LR schedule: linear warmup → cosine decay (mirrors the JS path)
    const wf = ep < warmupEps ? (ep + 1) / warmupEps : 1;
    const cf = ep < warmupEps ? 0 : (ep - warmupEps) / Math.max(1, epochs - warmupEps);
    opt.learningRate = lrMin + 0.5 * (lr * wf - lrMin) * (1 + Math.cos(Math.PI * cf));

    // Full-batch forward pass + auto-diff backward via TF
    const { value: lossT, grads } = tf.variableGrads(() => {
      const H0  = inpT.matMul(V.W0.transpose()).add(V.b0).relu();
      const H1  = H0.matMul(V.W1.transpose()).add(V.b1).relu();
      const H2r = H1.matMul(V.W2.transpose()).add(V.b2);
      const H2  = H2r.add(H0.mul(0.1)).relu();  // skip
      const logits = H2.matMul(V.W3.transpose()).add(V.b3);
      // Output: weighted cross-entropy + L2 on main weights
      const ce  = tf.oneHot(tgtT, nOut_).toFloat()
                    .mul(tf.logSoftmax(logits)).sum(1).neg().mul(wtT).mean();
      const l2L = [V.W0, V.W1, V.W2, V.W3]
                    .reduce((a, w) => a.add(w.square().sum()), tf.scalar(0)).mul(l2 / N);
      return ce.add(l2L);
    }, allVars);

    opt.applyGradients(grads);

    if (onProgress && ep % 20 === 0) {
      const lv = (await lossT.data())[0];
      onProgress(ep / epochs, lv, { epoch: ep, epochs, gpu: true });
    }
    lossT.dispose();
    for (const g of Object.values(grads)) g.dispose();

    // Yield to UI every 5 epochs so the progress bar updates
    if (ep % 5 === 0) await new Promise(r => setTimeout(r, 0));
  }

  // Copy trained weights back to the JS net (for JS inference compatibility)
  net.W[0].set(await V.W0.data()); net.W[1].set(await V.W1.data());
  net.W[2].set(await V.W2.data()); net.W[3].set(await V.W3.data());
  net.b[0].set(await V.b0.data()); net.b[1].set(await V.b1.data());
  net.b[2].set(await V.b2.data()); net.b[3].set(await V.b3.data());

  // Free all GPU memory
  allVars.forEach(v => v.dispose());
  inpT.dispose(); tgtT.dispose(); wtT.dispose();
  opt.dispose();
}

// ── TF.js GPU-accelerated batched inference ───────────────────────────────────
// Runs all voxel forward passes in one GPU call (chunked to avoid OOM).
// Returns Float32Array [total * nOut] of softmax probabilities.
function _inferBatchTF(net, inpFlat, total, nIn, CHUNK = 8192) {
  const tf = window.tf;
  const { nHidden, nOut } = net;

  // Constant tensors for weights — not variables, no gradient tracking
  const W  = net.W.map((w, i) => tf.tensor2d(w, [i === 3 ? nOut : nHidden, i === 0 ? nIn : nHidden]));
  const b  = net.b.map(bv => tf.tensor1d(bv));
  // Pre-transpose all weight matrices once — reused across chunks
  const Wt = W.map(w => w.transpose());

  const outFlat = new Float32Array(total * nOut);

  for (let start = 0; start < total; start += CHUNK) {
    const count = Math.min(CHUNK, total - start);
    const chunk = inpFlat.subarray(start * nIn, (start + count) * nIn);

    // tf.tidy disposes all intermediate tensors except the returned one
    const probT = tf.tidy(() => {
      const inp    = tf.tensor2d(chunk, [count, nIn]);
      const H0     = inp.matMul(Wt[0]).add(b[0]).relu();
      const H1     = H0.matMul(Wt[1]).add(b[1]).relu();
      const H2     = H1.matMul(Wt[2]).add(b[2]).add(H0.mul(0.1)).relu();
      const logits = H2.matMul(Wt[3]).add(b[3]);
      return tf.softmax(logits);
    });

    outFlat.set(probT.dataSync(), start * nOut);
    probT.dispose();
  }

  // Free all GPU memory held by weight tensors
  [...W, ...b, ...Wt].forEach(t => t.dispose());
  return outFlat;
}

// ── Train the neural implicit geological field ───────────────────────────────
// geoShapes: PreparedShape[] from geo-shapes.js prepareShapesForSDF() (can be [])
// Returns { net, fourierEnc, geoShapes, bounds, nUnits, unitCodes, nSDFs } or null
export async function trainGeoImplicit(boreholes, geoUnits, geoShapes = [], options = {}) {
  const {
    epochs          = 600,
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

  // Convert fractional centroid coordinates to world coords using site bounds.
  const sdfBbox       = { minX: bounds.minX, maxX: bounds.maxX, minY: bounds.minY, maxY: bounds.maxY, maxGL: bounds.maxZ };
  const preparedShapes = prepareShapesForSDF(geoShapes, sdfBbox);

  const fourierEnc = new FourierEncoder(L_FOURIER);
  const nSDFs      = preparedShapes.length;
  // nIn = Fourier positional (39) + SDF values (nSDFs, may be 0)
  const nIn        = fourierEnc.outDim + nSDFs;
  const nUnits     = geoUnits.length;
  const unitCodes  = geoUnits.map(u => u.code);
  const unitIdx    = {};
  geoUnits.forEach((u, i) => { unitIdx[u.code] = i; });

  // Build training samples from boreholes.
  const samples = [];
  for (const bh of boreholes) {
    for (const layer of (bh.layers ?? [])) {
      const ti = unitIdx[layer.unitCode];
      if (ti === undefined) continue;
      const zTop  = bh.groundLevel - layer.top;
      const zBase = bh.groundLevel - layer.base;
      const wt    = layer.certainty ?? 0.9;
      for (let s = 0; s < samplesPerLayer; s++) {
        const t  = (s + 0.5) / samplesPerLayer;
        const wz = zBase + t * (zTop - zBase);
        const pos = fourierEnc.encode(bh.x, bh.y, wz, bounds);
        const inp = new Float32Array(nIn);
        inp.set(pos);
        if (nSDFs) inp.set(evaluateAllSDFs(preparedShapes, bh.x, bh.y, wz), fourierEnc.outDim);
        samples.push({ inp, target: ti, weight: wt });
      }
    }
  }

  // ── Boundary-emphasis contact samples ───────────────────────────────────────
  // Unit contacts carry the most geological information. The uniform interior
  // samples above are mostly redundant — the network can interpolate between them.
  // Here we add high-weight samples placed just inside each unit on either side
  // of every observed contact, so the network learns precisely WHERE boundaries sit.
  {
    const CONTACT_OFFSET = 0.04; // fraction of layer thickness to offset inward
    const CONTACT_WEIGHT = 1.25; // elevated sample weight at contacts
    for (const bh of boreholes) {
      const layers = (bh.layers ?? []).filter(l => l.unitCode && unitIdx[l.unitCode] !== undefined);
      for (let li = 0; li < layers.length - 1; li++) {
        const above = layers[li];
        const below = layers[li + 1];
        const tiAbove = unitIdx[above.unitCode];
        const tiBelow = unitIdx[below.unitCode];
        if (tiAbove === undefined || tiBelow === undefined) continue;

        // Contact elevation (shared boundary between above.base and below.top)
        const contactZ = bh.groundLevel - above.base;

        // Layer thicknesses to determine offset distance
        const thickAbove = Math.abs(above.base - above.top);
        const thickBelow = Math.abs(below.base - below.top);
        const offsetAbove = Math.max(0.05, thickAbove * CONTACT_OFFSET);
        const offsetBelow = Math.max(0.05, thickBelow * CONTACT_OFFSET);

        // 2 samples just inside the UPPER unit (slightly above contact)
        for (let k = 0; k < 2; k++) {
          const wz = contactZ + offsetAbove * (k + 1);
          const pos = fourierEnc.encode(bh.x, bh.y, wz, bounds);
          const inp = new Float32Array(nIn);
          inp.set(pos);
          if (nSDFs) inp.set(evaluateAllSDFs(preparedShapes, bh.x, bh.y, wz), fourierEnc.outDim);
          samples.push({ inp, target: tiAbove, weight: (above.certainty ?? 0.9) * CONTACT_WEIGHT });
        }

        // 2 samples just inside the LOWER unit (slightly below contact)
        for (let k = 0; k < 2; k++) {
          const wz = contactZ - offsetBelow * (k + 1);
          const pos = fourierEnc.encode(bh.x, bh.y, wz, bounds);
          const inp = new Float32Array(nIn);
          inp.set(pos);
          if (nSDFs) inp.set(evaluateAllSDFs(preparedShapes, bh.x, bh.y, wz), fourierEnc.outDim);
          samples.push({ inp, target: tiBelow, weight: (below.certainty ?? 0.9) * CONTACT_WEIGHT });
        }
      }
    }
  }

  if (samples.length === 0) return null;

  // ── Inter-borehole contact surface interpolation ─────────────────────────────
  // For each unit, find all borehole contacts (top of that unit in each borehole).
  // For nearby borehole PAIRS that both observe this unit's top, interpolate the
  // contact elevation between them and place training samples along the interpolated
  // surface at N intermediate XY positions. This teaches the network the SHAPE
  // of the contact surface, not just its depth at observation points.
  //
  // This is analogous to how Leapfrog fits surfaces to drillhole contacts:
  // the surface shape between boreholes is constrained by interpolation, not guessed.
  {
    const INTERP_STEPS  = 3;  // positions between BH pair
    const INTERP_WEIGHT = 0.65; // lower than real observations
    const MAX_PAIR_DIST = Math.hypot(
      (bounds.maxX - bounds.minX), (bounds.maxY - bounds.minY)
    ) * 0.6; // only pair BHs within 60% of site diagonal

    // Build per-unit contact list: { bhX, bhY, contactZ, unitCode, aboveCode, belowCode }
    const unitContacts = {}; // { unitCode → [{x, y, topZ}] }
    for (const bh of boreholes) {
      const gl = bh.groundLevel ?? 0;
      for (const layer of (bh.layers ?? [])) {
        const ti = unitIdx[layer.unitCode];
        if (ti === undefined) continue;
        const topZ = gl - layer.top;
        if (!unitContacts[layer.unitCode]) unitContacts[layer.unitCode] = [];
        unitContacts[layer.unitCode].push({ x: bh.x, y: bh.y, topZ, unitCode: layer.unitCode });
      }
    }

    for (const [code, contacts] of Object.entries(unitContacts)) {
      const ti = unitIdx[code];
      if (ti === undefined || contacts.length < 2) continue;

      for (let a = 0; a < contacts.length - 1; a++) {
        for (let b = a + 1; b < contacts.length; b++) {
          const ca = contacts[a], cb = contacts[b];
          const pairDist = Math.hypot(ca.x - cb.x, ca.y - cb.y);
          if (pairDist > MAX_PAIR_DIST || pairDist < 0.5) continue;

          // Interpolate N positions along the straight line between BH pair
          for (let step = 1; step <= INTERP_STEPS; step++) {
            const t  = step / (INTERP_STEPS + 1); // 0..1 exclusive
            const ix = ca.x + t * (cb.x - ca.x);
            const iy = ca.y + t * (cb.y - ca.y);
            // Linearly interpolated contact elevation at this XY
            const iz = ca.topZ + t * (cb.topZ - ca.topZ);
            if (iz < bounds.minZ || iz > bounds.maxZ) continue;

            // topZ is the TOP elevation of the unit; points below it are inside the unit.
            // Place 2 samples just below the interpolated contact (inside this unit).
            const interpThick = Math.max(0.2,
              Math.abs(ca.topZ - cb.topZ) / (INTERP_STEPS + 1) * 0.5 + 0.1);
            for (const dz of [-interpThick, -interpThick * 2.5]) {
              const wz = iz + dz; // negative dz = below contact = inside unit
              if (wz < bounds.minZ || wz > bounds.maxZ) continue;
              const pos = fourierEnc.encode(ix, iy, wz, bounds);
              const inp = new Float32Array(nIn);
              inp.set(pos);
              if (nSDFs) inp.set(evaluateAllSDFs(preparedShapes, ix, iy, wz), fourierEnc.outDim);
              samples.push({ inp, target: ti, weight: INTERP_WEIGHT });
            }
          }
        }
      }
    }
  }

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
            const pos = fourierEnc.encode(sx, sy, wz, bounds);
            const inp = new Float32Array(nIn);
            inp.set(pos);
            if (nSDFs) inp.set(evaluateAllSDFs(preparedShapes, sx, sy, wz), fourierEnc.outDim);
            // Strat samples get reduced weight (0.25) — they guide but don't override BH data
            samples.push({ inp, target: tiA, weight: 0.25 });
          }
          // Samples BELOW contact → unit B
          for (let s = 1; s <= N_STRAT; s++) {
            const wz  = meanElev - (s / N_STRAT) * stratJitter;
            if (wz < bounds.minZ || wz > bounds.maxZ) continue;
            const pos = fourierEnc.encode(sx, sy, wz, bounds);
            const inp = new Float32Array(nIn);
            inp.set(pos);
            if (nSDFs) inp.set(evaluateAllSDFs(preparedShapes, sx, sy, wz), fourierEnc.outDim);
            samples.push({ inp, target: tiB, weight: 0.25 });
          }
        }
      }
    }
    if (onProgress) onProgress(0, 0, { stratContactsFound: Object.keys(contactElevs).length });
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

  const net   = new GeoImplicitNet(nIn, 80, nUnits, fourierEnc.outDim);
  const hasTF = typeof window !== 'undefined' && !!window.tf;

  if (hasTF) {
    // ── GPU path: TF.js WebGL full-batch Adam ──────────────────────────────
    if (onProgress) onProgress(0, 0, { nSamples: allSamples.length, gpu: true, nReal: nRealSamples, nVirtual: nVirtualSamples });
    await _trainWithTF(net, allSamples, { epochs, lr, lrMin, l2 }, onProgress);
  } else {
    // ── CPU fallback: original sample-by-sample Adam ───────────────────────
    const opt = new AdamOpt(net._params, lr);

    for (let ep = 0; ep < epochs; ep++) {
      const warmupEps = Math.max(1, Math.round(epochs * 0.05));
      const lrWarmup  = ep < warmupEps ? (lr * (ep + 1) / warmupEps) : lr;
      const cosPhase  = ep < warmupEps ? 0 : (ep - warmupEps) / (epochs - warmupEps);
      opt.setLr(lrMin + 0.5 * (lrWarmup - lrMin) * (1 + Math.cos(Math.PI * cosPhase)));

      for (let i = allSamples.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [allSamples[i], allSamples[j]] = [allSamples[j], allSamples[i]];
      }

      let totalLoss = 0;
      for (const s of allSamples) {
        const act   = net.forward(s.inp);
        const loss  = -Math.log(Math.max(act.probs[s.target], 1e-9));
        totalLoss  += s.weight * loss;
        const grads = net.backward(s.inp, act, s.target, l2, s.weight);
        opt.step(grads);
      }

      if (onProgress && ep % 20 === 0) {
        onProgress(ep / epochs, totalLoss / allSamples.length, { epoch: ep, epochs });
        await new Promise(r => setTimeout(r, 0));
      }
    }
  }

  if (onProgress) onProgress(1, 0, { epoch: epochs, epochs });
  return { net, fourierEnc, geoShapes: preparedShapes, bounds, nUnits, unitCodes, nSDFs };
}

// ── Concept-driven refinement fine-tune ──────────────────────────────────────
// Takes an already-trained model and a list of refinement samples, then runs
// a short additional training pass (epochs << main training) using those samples
// at reduced learning rate. Modifies `trained.net` in-place.
//
// refinementSamples: [{ x, y, z, unitCode, weight }]  — world-space virtual obs.
// options.epochs (default 100), options.lr (default 0.002)
export async function finetuneGeoImplicit(trained, geoUnits, refinementSamples, options = {}) {
  if (!trained || !refinementSamples?.length) return trained;
  const { epochs = 100, lr = 0.002, l2 = 0.0005, onProgress = null } = options;

  const { net, fourierEnc, bounds, unitCodes, nSDFs } = trained;
  const trainedGeoShapes = trained.geoShapes ?? [];
  const nIn  = fourierEnc.outDim + (nSDFs ?? 0);
  const unitIdx = {};
  geoUnits.forEach((u, i) => { unitIdx[u.code] = i; });

  // Build training samples from refinement hints
  const samples = [];
  for (const { x, y, z, unitCode, weight = 0.15 } of refinementSamples) {
    const ti = unitIdx[unitCode];
    if (ti === undefined) continue;
    const pos = fourierEnc.encode(x, y, z, bounds);
    const inp = new Float32Array(nIn);
    inp.set(pos);
    if (nSDFs) inp.set(evaluateAllSDFs(trainedGeoShapes, x, y, z), fourierEnc.outDim);
    samples.push({ inp, target: ti, weight });
  }
  if (!samples.length) return trained;

  const opt = new AdamOpt(net.getParams(), lr);

  for (let ep = 0; ep < epochs; ep++) {
    // Cosine decay from lr to lr*0.1
    const lrMin = lr * 0.1;
    opt.setLr(lrMin + 0.5 * (lr - lrMin) * (1 + Math.cos(Math.PI * ep / epochs)));

    // Shuffle
    for (let i = samples.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [samples[i], samples[j]] = [samples[j], samples[i]];
    }
    for (const s of samples) {
      const act = net.forward(s.inp);
      opt.step(net.backward(s.inp, act, s.target, l2, s.weight));
    }
    if (onProgress && ep % 25 === 0) {
      onProgress(ep / epochs);
      await new Promise(r => setTimeout(r, 0));
    }
  }
  if (onProgress) onProgress(1);
  return trained;
}

// ── Infer voxel grid from trained model ─────────────────────────────────────
// grid must have { nx, ny, nz, cellSize, cellHeight, origin: {x,y,z} }
// geoShapes: PreparedShape[] from geo-shapes.js (same list used during training, can be [])
export function inferGeoImplicit(trained, grid, geoUnits, geoShapes = [], options = {}) {
  const { net, fourierEnc, bounds, unitCodes, nSDFs } = trained;
  const inferShapes = geoShapes.length ? geoShapes : (trained.geoShapes ?? []);
  const nIn         = fourierEnc.outDim + (nSDFs ?? 0);

  const { nx, ny, nz, cellSize, cellHeight, origin } = grid;
  const total   = nx * ny * nz;
  const nUnitsI = unitCodes.length;

  const unitIds      = new Uint8Array(total);
  const certainty    = new Float32Array(total);
  const blendUnitIds = new Uint8Array(total);
  const blendRatios  = new Float32Array(total);

  // Accumulator for probability averaging — shape [total * nUnitsI]
  const probAcc = new Float32Array(total * nUnitsI);

  const codeToId = {};
  geoUnits.forEach(u => { codeToId[u.code] = u.id; });

  const hasTFInfer = typeof window !== 'undefined' && !!window.tf;

  if (hasTFInfer) {
    // ── GPU-accelerated batched inference ──────────────────────────────────────
    const allInps = new Float32Array(total * nIn);

    for (let iz = nz - 1; iz >= 0; iz--) {
      const worldZ = origin.y + iz * cellHeight + cellHeight * 0.5;
      for (let iy = 0; iy < ny; iy++) {
        const worldY = origin.z + iy * cellSize + cellSize * 0.5;
        for (let ix = 0; ix < nx; ix++) {
          const worldX = origin.x + ix * cellSize + cellSize * 0.5;
          const idx    = ix + iy * nx + iz * nx * ny;
          const pos    = fourierEnc.encode(worldX, worldY, worldZ, bounds);
          allInps.set(pos, idx * nIn);
          if (nSDFs) allInps.set(evaluateAllSDFs(inferShapes, worldX, worldY, worldZ), idx * nIn + fourierEnc.outDim);
        }
      }
    }

    // Single batched GPU forward pass over all voxels
    const allProbs = _inferBatchTF(net, allInps, total, nIn);
    for (let i = 0; i < total * nUnitsI; i++) probAcc[i] = allProbs[i];
  } else {
    // ── JS per-voxel inference ─────────────────────────────────────────────────
    for (let iz = nz - 1; iz >= 0; iz--) {
      const worldZ = origin.y + iz * cellHeight + cellHeight * 0.5;
      for (let iy = 0; iy < ny; iy++) {
        const worldY = origin.z + iy * cellSize + cellSize * 0.5;
        for (let ix = 0; ix < nx; ix++) {
          const worldX = origin.x + ix * cellSize + cellSize * 0.5;
          const idx    = ix + iy * nx + iz * nx * ny;
          const pos    = fourierEnc.encode(worldX, worldY, worldZ, bounds);
          const inp    = new Float32Array(nIn);
          inp.set(pos);
          if (nSDFs) inp.set(evaluateAllSDFs(inferShapes, worldX, worldY, worldZ), fourierEnc.outDim);
          const probs = net.forward(inp).probs;
          const base  = idx * nUnitsI;
          for (let u = 0; u < nUnitsI; u++) probAcc[base + u] = probs[u];
        }
      }
    }
  }

  // Build outputs from probAcc
  const probVolumes = new Map();
  unitCodes.forEach(code => probVolumes.set(code, new Float32Array(total)));

  for (let idx = 0; idx < total; idx++) {
    const base = idx * nUnitsI;

    let b1 = 0, b2 = -1;
    let p1 = probAcc[base], p2 = 0;
    if (nUnitsI > 1) {
      b2 = 1; p2 = probAcc[base + 1];
      if (p2 > p1) { b1 = 1; b2 = 0; const tmp = p1; p1 = p2; p2 = tmp; }
    }
    for (let u = 2; u < nUnitsI; u++) {
      const p = probAcc[base + u];
      if (p > p1) { b2 = b1; p2 = p1; b1 = u; p1 = p; }
      else if (p > p2) { b2 = u; p2 = p; }
    }

    unitIds[idx]      = codeToId[unitCodes[b1]] ?? 0;
    blendUnitIds[idx] = b2 >= 0 ? (codeToId[unitCodes[b2]] ?? 0) : 0;
    blendRatios[idx]  = p2;
    certainty[idx]    = Math.max(0.05, Math.min(1, 0.5 + p1 - p2));

    for (let u = 0; u < nUnitsI; u++) {
      probVolumes.get(unitCodes[u])[idx] = probAcc[base + u];
    }
  }

  return { unitIds, certainty, blendUnitIds, blendRatios, probVolumes };
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
            y: origin.z + ciy_c * cellSize   + cellSize   * 0.5,
            z: origin.y + ciz_c * cellHeight + cellHeight * 0.5,
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

    // Concept prediction: sample concept context at centroid of unit bbox.
    // Use rotated-tensor fields (Amaj, theta) for direction-aware matching.
    let predictedEW = 1, predictedNS = 1, predictedAmaj = 1, predictedAmin = 1;
    let predictedTheta = 0, predictedThetaDeg = 0;
    if (conceptStore && !conceptStore.isEmpty) {
      const cx = origin.x + (minIX + maxIX) / 2 * cellSize;
      const cy = origin.z + (minIY + maxIY) / 2 * cellSize;
      const cz = origin.y + (minIZ + maxIZ) / 2 * cellHeight;
      const ctx = conceptStore.computeAt(cx, cy, cz, u.code);
      predictedEW        = ctx.tensor.Ax;
      predictedNS        = ctx.tensor.Ay;
      predictedAmaj      = ctx.tensor.Amaj ?? Math.max(ctx.tensor.Ax, ctx.tensor.Ay);
      predictedAmin      = ctx.tensor.Amin ?? Math.min(ctx.tensor.Ax, ctx.tensor.Ay);
      predictedTheta     = ctx.tensor.theta ?? 0;
      predictedThetaDeg  = predictedTheta * 180 / Math.PI;
    }

    // Direction-aware concept match: project actual voxel bbox onto the concept's
    // predicted major axis direction (theta) and compare elongation ratios.
    // This handles diagonal concepts (NE-SW, NW-SE) correctly.
    const cosT = Math.cos(predictedTheta);
    const sinT = Math.sin(predictedTheta);
    // Project bbox extent E-W (extentX) and N-S (extentY) onto major/minor axes
    const actualMajorExtent = Math.abs(extentX * cosT) + Math.abs(extentY * sinT);
    const actualMinorExtent = Math.abs(extentX * sinT) + Math.abs(extentY * cosT);
    const actualElongation  = actualMajorExtent / Math.max(1, actualMinorExtent);

    // conceptMatch: 1 = geometry is elongated in concept's predicted direction,
    //              0.5 = no strong elongation detected,
    //              0 = elongated in the WRONG direction
    let conceptMatch;
    if (predictedAmaj < 1.15) {
      // Concept predicts roughly isotropic — any result is acceptable
      conceptMatch = 0.75;
    } else {
      const ratio = actualElongation / Math.max(1, predictedAmaj);
      conceptMatch = Math.min(1, Math.max(0, ratio));
    }

    results.push({
      unitCode:          u.code,
      unitName:          u.name,
      unitColor:         u.color,
      extentX, extentY, extentZ, count,
      ewRatio:           +ewRatio.toFixed(2),
      nsRatio:           +nsRatio.toFixed(2),
      predictedEW:       +predictedEW.toFixed(2),
      predictedNS:       +predictedNS.toFixed(2),
      predictedAmaj:     +predictedAmaj.toFixed(2),
      predictedAmin:     +predictedAmin.toFixed(2),
      predictedThetaDeg: +predictedThetaDeg.toFixed(1),
      actualElongation:  +actualElongation.toFixed(2),
      conceptMatch:      +conceptMatch.toFixed(2),
    });
  }
  return results;
}

// ── Data-derived concept embeddings from borehole geometry ────────────────────
// Analyzes lateral thickness variation, depth trends, and occurrence patterns of
// each geological unit across the borehole network and directly computes 32-dim
// concept embeddings from the geometric statistics — no API call needed.
// Complements suggestConceptsFromBoreholes (in claude-client.js) which asks
// Claude to describe the concepts; this version encodes them directly.
//
// Returns [{unitCode, unitName, description, embedding: Float32Array(32), confidence, reason}]
// The embeddings can be passed directly to ConceptStore.add().
export function analyzeBoreholeGeometry(boreholes, geoUnits) {
  if (!boreholes.length || !geoUnits.length) return [];

  const suggestions = [];
  const unitIdx = {};
  geoUnits.forEach((u, i) => { unitIdx[u.code] = i; });

  // Per-unit borehole observation: {x, y, top, base, thickness, midZ}
  const unitObs = {};
  geoUnits.forEach(u => { unitObs[u.code] = []; });

  for (const bh of boreholes) {
    const gl = bh.groundLevel ?? 0;
    for (const layer of (bh.layers ?? [])) {
      if (!layer.unitCode || !unitObs[layer.unitCode]) continue;
      const thickness = Math.abs(layer.base - layer.top);
      if (thickness < 0.05) continue;
      const midZ = gl - (layer.top + layer.base) / 2;
      unitObs[layer.unitCode].push({ x: bh.x, y: bh.y, top: gl - layer.top, base: gl - layer.base, thickness, midZ });
    }
  }

  for (const u of geoUnits) {
    const obs = unitObs[u.code];
    if (obs.length < 2) continue;

    const emb = new Float32Array(32); // 32-dim concept embedding (legacy analyzeBoreholeGeometry output)
    const reasons = [];

    // ── Compute lateral statistics ────────────────────────────────────────────
    const xs = obs.map(o => o.x);
    const ys = obs.map(o => o.y);
    const ts = obs.map(o => o.thickness);
    const zs = obs.map(o => o.midZ);
    const n  = obs.length;

    const meanX = xs.reduce((a, b) => a + b, 0) / n;
    const meanY = ys.reduce((a, b) => a + b, 0) / n;
    const meanT = ts.reduce((a, b) => a + b, 0) / n;
    const meanZ = zs.reduce((a, b) => a + b, 0) / n;

    // Variance in X, Y, Z directions
    const varX = xs.reduce((s, x) => s + (x - meanX) ** 2, 0) / n;
    const varY = ys.reduce((s, y) => s + (y - meanY) ** 2, 0) / n;

    // ── E-W vs N-S elongation from spatial distribution of observations ───────
    // If the unit appears in BHs that span more E-W than N-S, likely E-W elongated
    const spanX = Math.max(...xs) - Math.min(...xs);
    const spanY = Math.max(...ys) - Math.min(...ys);
    if (spanX > 1 || spanY > 1) {
      const ewScore  = Math.min(1, spanX / Math.max(1, spanX + spanY) * 2 - 0.5);
      const nsScore  = Math.min(1, spanY / Math.max(1, spanX + spanY) * 2 - 0.5);
      if (Math.abs(ewScore - nsScore) > 0.2) {
        emb[3] = Math.max(-0.5, Math.min(0.9, ewScore * 1.2));
        emb[4] = Math.max(-0.5, Math.min(0.9, nsScore * 1.2));
        const dir = spanX > spanY ? 'E-W' : 'N-S';
        reasons.push(`unit found across ${(Math.max(spanX, spanY)).toFixed(0)}m ${dir} extent`);
      }
    }

    // ── Thickness-gradient: does thickness increase E→W or S→N? ─────────────
    // Linear regression: thickness ~ a*x + b*y + c
    if (n >= 3) {
      // Correlate thickness with x (E-W) and y (N-S) separately
      const covTX = obs.reduce((s, o) => s + (o.x - meanX) * (o.thickness - meanT), 0) / n;
      const covTY = obs.reduce((s, o) => s + (o.y - meanY) * (o.thickness - meanT), 0) / n;
      const stdX  = Math.sqrt(varX) || 1;
      const stdY  = Math.sqrt(varY) || 1;
      const stdT  = Math.sqrt(ts.reduce((s, t) => s + (t - meanT) ** 2, 0) / n) || 1;
      const rTX   = covTX / (stdX * stdT);
      const rTY   = covTY / (stdY * stdT);

      // Thinning: negative correlation = thins in that direction
      if (rTX < -0.5) { emb[10] = Math.min(0.9, -rTX); reasons.push('thins eastward'); }
      if (rTX >  0.5) { emb[11] = Math.min(0.9,  rTX); reasons.push('thickens eastward'); }
      if (rTY < -0.5) { emb[12] = Math.min(0.9, -rTY); reasons.push('thins northward'); }
      if (rTY >  0.5) { emb[13] = Math.min(0.9,  rTY); reasons.push('thickens northward'); }
    }

    // ── Depth trend: does the unit deepen in a particular direction? ──────────
    if (n >= 3) {
      const covZX = obs.reduce((s, o) => s + (o.x - meanX) * (o.midZ - meanZ), 0) / n;
      const covZY = obs.reduce((s, o) => s + (o.y - meanY) * (o.midZ - meanZ), 0) / n;
      const stdX  = Math.sqrt(varX) || 1;
      const stdY  = Math.sqrt(varY) || 1;
      const stdZ  = Math.sqrt(zs.reduce((s, z) => s + (z - meanZ) ** 2, 0) / n) || 1;
      const rZX   = covZX / (stdX * stdZ);  // positive = deepens east
      const rZY   = covZY / (stdY * stdZ);  // positive = deepens north

      if (rZX >  0.45) { emb[14] = Math.min(0.9,  rZX); emb[1] = 0.5; reasons.push('deepens eastward'); }
      if (rZX < -0.45) { emb[15] = Math.min(0.9, -rZX); emb[1] = 0.5; reasons.push('deepens westward'); }
      if (rZY >  0.45) { emb[16] = Math.min(0.9,  rZY); emb[1] = 0.5; reasons.push('deepens northward'); }
      if (rZY < -0.45) { emb[17] = Math.min(0.9, -rZY); emb[1] = 0.5; reasons.push('deepens southward'); }
    }

    // ── Lateral continuity: is the unit consistently present? ─────────────────
    const presentFraction = obs.length / boreholes.filter(b => b.layers?.length).length;
    if (presentFraction > 0.6) {
      emb[9]  = 0.7;  // laterally continuous
      emb[27] = 0.5;  // lateral anisotropy
      reasons.push(`present in ${(presentFraction * 100).toFixed(0)}% of BHs`);
    } else if (presentFraction < 0.3) {
      emb[9]  = -0.4; // discontinuous/lenticular
      emb[10] += 0.3; emb[11] += 0.3; // some thinning
    }

    // ── Horizontal layering: near-constant depth → horizontal ─────────────────
    const zCoV = (meanZ !== 0) ? Math.abs(Math.sqrt(zs.reduce((s,z) => s+(z-meanZ)**2,0)/n) / Math.abs(meanZ)) : 1;
    if (zCoV < 0.05 && n >= 3) {
      emb[0] = 0.7;  // horizontal_layering
      reasons.push('consistent depth across BHs');
    }

    // ── Thickness consistency: erosional base → irregular ─────────────────────
    const tCoV = meanT > 0 ? Math.sqrt(ts.reduce((s, t) => s + (t - meanT) ** 2, 0) / n) / meanT : 0;
    if (tCoV > 0.4) {
      emb[19] = Math.min(0.8, tCoV); // irregular_base
      emb[8]  = 0.5;                  // erosional_contact
      reasons.push(`thickness varies (CoV ${(tCoV * 100).toFixed(0)}%)`);
    }

    // ── Incision depth: thin on average but variable → incised channel ────────
    const meanThickM = meanT;
    const siteSpan = Math.max(spanX, spanY, 1);
    const incisionRatio = meanThickM / siteSpan;
    if (incisionRatio < 0.03 && tCoV > 0.3) {
      emb[29] = 0.7;  // incision_depth_ratio
      emb[5]  = 0.6;  // channel_morphology
      reasons.push('thin/incised relative to site span');
    }

    // Clamp all
    for (let i = 0; i < emb.length; i++) emb[i] = Math.max(-1, Math.min(1, emb[i]));

    // Only suggest if we have at least one meaningful observation
    if (reasons.length === 0) continue;

    // Build description from reasons
    const dirTerms   = reasons.filter(r => /E-W|N-S|east|west|north|south/.test(r));
    const shapeTerms = reasons.filter(r => !/E-W|N-S|east|west|north|south/.test(r));
    let description  = `${u.name} (${u.code})`;
    if (dirTerms.length) description += ` — ${dirTerms[0]}`;
    if (shapeTerms.length) description += ` — ${shapeTerms[0]}`;

    const confidence = Math.min(0.85, 0.45 + Math.min(n / 8, 0.25) + Math.min(reasons.length * 0.07, 0.15));

    suggestions.push({
      unitCode:    u.code,
      unitName:    u.name,
      unitColor:   u.color,
      description,
      embedding:   emb,
      confidence:  +confidence.toFixed(2),
      reason:      reasons.join('; '),
      nObservations: n,
    });
  }

  return suggestions;
}
