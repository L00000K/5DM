// ── Concept Store ──────────────────────────────────────────────────────────────
// Holds geological conceptual model entries, each encoded as a dense 32-dim
// embedding on CONCEPT_AXES. At each spatial position (x,y,z) the store returns
// a position-dependent context vector and an anisotropy tensor that warp the
// coordinate space fed to the neural implicit field — so the OUTPUT GEOMETRY of
// the 3D model directly reflects conceptual inputs like "palaeochannel E-W" or
// "stepped rockhead", rather than just adding point observations.

export const CONCEPT_AXES = [
  'horizontal_layering',   //  0 — tends to form flat beds
  'inclined_bedding',      //  1 — dip > ~5°
  'dip_magnitude',         //  2 — 0=flat, 1=near-vertical
  'east_west_elongation',  //  3 — body trends E-W
  'north_south_elongation',//  4 — body trends N-S
  'channel_morphology',    //  5 — concave-up erosional trough
  'dome_anticline',        //  6 — convex-up / upward-arching
  'fault_controlled',      //  7 — boundary is a fault surface
  'erosional_contact',     //  8 — unconformable/erosional base
  'lateral_continuity',    //  9 — unit persists laterally
  'lateral_thinning_east', // 10 — wedges out eastward
  'lateral_thinning_west', // 11 — wedges out westward
  'lateral_thinning_north',// 12 — wedges out northward
  'lateral_thinning_south',// 13 — wedges out southward
  'deepens_east',          // 14 — surface deepens to east
  'deepens_west',          // 15 — surface deepens to west
  'deepens_north',         // 16 — surface deepens to north
  'deepens_south',         // 17 — surface deepens to south
  'stepped_boundary',      // 18 — piecewise/stepped contact
  'irregular_base',        // 19 — dissolution / karstic base
  'nested_channels',       // 20 — multi-storey channel fill
  'coarsening_upward',     // 21 — grain size increases upward
  'fining_upward',         // 22 — grain size decreases upward
  'gravel_basal_lag',      // 23 — coarse lag at base
  'dissolution_features',  // 24 — karst / voids present
  'structural_complexity', // 25 — complex deformation
  'data_confidence',       // 26 — overall confidence in concept
  'lateral_anisotropy',    // 27 — horizontally elongated (any dir)
  'vertical_anisotropy',   // 28 — strong layering / anisotropy in Z
  'incision_depth_ratio',  // 29 — deep channel relative to width
  'overburden_control',    // 30 — geometry controlled by load
  'complexity_gradient',   // 31 — complexity increases in one direction
];

// Indices of axes used to derive the anisotropy tensor for coordinate warping
const AX_EW   = 3;   // east_west_elongation   → major axis if EW dominant
const AX_NS   = 4;   // north_south_elongation  → major axis if NS dominant
const AX_VERT = 29;  // incision_depth_ratio    → scale z
const AX_CHAN = 5;   // channel_morphology      → additional vertical compression

// ── Standalone warp utilities (importable without a ConceptStore instance) ──

/**
 * Warp world coordinates (wx, wy, wz) using an anisotropy tensor.
 * For axis-aligned tensors: divides each axis by Ax/Ay/Az.
 * For rotated tensors: rotates to principal-axis frame first (major/minor), then scales.
 * The warped space is NOT rotated back — it remains in the principal-axis frame.
 * This is intentional: Fourier features are computed consistently in this warped space.
 */
export function warpPoint(wx, wy, wz, tensor) {
  const { theta, cosT, sinT, Amaj = 1, Amin = 1, Az = 1, Ax = 1, Ay = 1 } = tensor;
  if (theta !== undefined && Math.abs(theta) > 0.05 && (Amaj ?? 1) > 1.1) {
    // Rotate world coords onto major/minor axes, then scale
    const major =  wx * cosT + wy * sinT;
    const minor = -wx * sinT + wy * cosT;
    return { x: major / Amaj, y: minor / Amin, z: wz / Az };
  }
  return { x: wx / Ax, y: wy / Ay, z: wz / Az };
}

/**
 * Compute the bounding box of the warped coordinate space for a given site extent.
 * Required because the Fourier encoder normalises positions within these bounds;
 * rotation can shift the warped extents away from axis-aligned extrema.
 */
export function computeWarpedBounds(bounds, tensor) {
  const { theta, cosT, sinT, Amaj = 1, Amin = 1, Az = 1, Ax = 1, Ay = 1 } = tensor;
  const zMin = bounds.minZ / Az, zMax = bounds.maxZ / Az;
  if (theta !== undefined && Math.abs(theta) > 0.05 && (Amaj ?? 1) > 1.1) {
    // Evaluate all four XY corners to find true warped extent
    const wXs = [], wYs = [];
    for (const wx of [bounds.minX, bounds.maxX]) {
      for (const wy of [bounds.minY, bounds.maxY]) {
        wXs.push((wx * cosT + wy * sinT) / Amaj);
        wYs.push((-wx * sinT + wy * cosT) / Amin);
      }
    }
    return {
      minX: Math.min(...wXs), maxX: Math.max(...wXs),
      minY: Math.min(...wYs), maxY: Math.max(...wYs),
      minZ: zMin, maxZ: zMax,
    };
  }
  return {
    minX: bounds.minX / Ax, maxX: bounds.maxX / Ax,
    minY: bounds.minY / Ay, maxY: bounds.maxY / Ay,
    minZ: zMin, maxZ: zMax,
  };
}

export class ConceptStore {
  constructor() {
    this._concepts = [];
    this._nextId   = 1;
  }

  get concepts() { return this._concepts; }
  get isEmpty()  { return this._concepts.length === 0; }

  /**
   * Add a geological concept.
   * @param {object} entry
   * @param {string}        entry.description   - free-text description
   * @param {Float32Array}  entry.embedding     - 32-dim vector on CONCEPT_AXES (−1..+1)
   * @param {number}        [entry.confidence]  - 0–1
   * @param {object}        [entry.domain]      - { type:'global'|'bbox', minX,maxX,minY,maxY, sigma }
   * @param {string[]}      [entry.unitAffinity]- unit codes this concept applies to ([] = all)
   * @returns {string} concept id
   */
  add({ description, embedding, confidence = 0.7, domain = { type: 'global' }, unitAffinity = [], temporalOrder = null, parentId = null }) {
    const id = `c${this._nextId++}`;
    this._concepts.push({
      id, description,
      embedding: embedding instanceof Float32Array ? embedding : new Float32Array(embedding),
      confidence, domain, unitAffinity,
      temporalOrder, // integer rank: lower = older, higher = younger; null = unspecified
      parentId,      // optional parent concept id for inheritance (child blends in 40% of parent)
    });
    return id;
  }

  // Set parent concept (for inheritance). parentId = null removes the relationship.
  setParent(id, parentId) {
    const c = this._concepts.find(c => c.id === id);
    if (c) c.parentId = parentId ?? null;
  }

  // Compute effective embedding for a concept, blending in parent embedding if set.
  // Inheritance weight: each level contributes (INHERIT_W)^depth to the blend.
  _effectiveEmbedding(concept, depth = 0, visited = null) {
    const INHERIT_W = 0.4;
    const MAX_DEPTH = 3;
    const seen = visited ?? new Set();
    seen.add(concept.id);
    if (depth >= MAX_DEPTH || !concept.parentId) return concept.embedding;
    if (seen.has(concept.parentId)) return concept.embedding; // break circular chain
    const parent = this._concepts.find(p => p.id === concept.parentId);
    if (!parent) return concept.embedding;
    const parentEmb = this._effectiveEmbedding(parent, depth + 1, seen);
    const DIM = Math.min(concept.embedding.length, parentEmb.length);
    const result = new Float32Array(32);
    const selfW = 1 - INHERIT_W;
    for (let i = 0; i < DIM; i++) {
      result[i] = selfW * concept.embedding[i] + INHERIT_W * parentEmb[i];
    }
    return result;
  }

  setTemporalOrder(id, rank) {
    const c = this._concepts.find(c => c.id === id);
    if (c) c.temporalOrder = (rank === null || rank === undefined) ? null : Number(rank);
  }

  /**
   * Return all pairs (younger, older) of concepts with explicit temporalOrder set.
   * Pairs are sorted so the younger concept (higher rank) comes first.
   * Only includes pairs whose spatial domains overlap (or at least one is global).
   */
  temporallyOrderedPairs() {
    const withOrder = this._concepts.filter(c => c.temporalOrder !== null && c.temporalOrder !== undefined);
    if (withOrder.length < 2) return [];
    const pairs = [];
    for (let i = 0; i < withOrder.length; i++) {
      for (let j = i + 1; j < withOrder.length; j++) {
        const ca = withOrder[i], cb = withOrder[j];
        if (ca.temporalOrder === cb.temporalOrder) continue; // same rank, skip
        const [younger, older] = ca.temporalOrder > cb.temporalOrder ? [ca, cb] : [cb, ca];
        // Check domain overlap (skip only if both are non-overlapping bbox domains)
        const aGlobal = !ca.domain || ca.domain.type === 'global';
        const bGlobal = !cb.domain || cb.domain.type === 'global';
        if (!aGlobal && !bGlobal) {
          const ad = ca.domain, bd = cb.domain;
          if (ad.maxX < bd.minX || bd.maxX < ad.minX || ad.maxY < bd.minY || bd.maxY < ad.minY) continue;
        }
        pairs.push({ younger, older });
      }
    }
    return pairs;
  }

  remove(id) {
    this._concepts = this._concepts.filter(c => c.id !== id);
  }

  clear() { this._concepts = []; this._nextId = 1; }

  // ── Spatial relevance ───────────────────────────────────────────────────────

  _relevance(c, wx, wy, wz = null) {
    let base = c.confidence;

    // Vertical domain filter: if domain specifies minZ/maxZ (AOD), apply Gaussian decay
    // outside that depth range. sigmaZ defaults to 20% of the specified depth range.
    if (wz !== null && c.domain) {
      const { minZ, maxZ, sigmaZ } = c.domain;
      if (minZ !== undefined && maxZ !== undefined) {
        const sz = sigmaZ ?? Math.max(1, (maxZ - minZ) * 0.2);
        const dz = Math.max(0, wz < minZ ? minZ - wz : wz > maxZ ? wz - maxZ : 0);
        base *= Math.exp(-(dz * dz) / (2 * sz * sz));
        if (base < 0.005) return 0;
      }
    }

    if (!c.domain || c.domain.type === 'global') return base;
    if (c.domain.type === 'bbox') {
      const { minX = 0, maxX = 0, minY = 0, maxY = 0, sigma = 50 } = c.domain;
      const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
      // Distance to box edge (0 inside box)
      const dx = Math.max(0, Math.abs(wx - cx) - (maxX - minX) / 2);
      const dy = Math.max(0, Math.abs(wy - cy) - (maxY - minY) / 2);
      const dist = Math.hypot(dx, dy);
      return base * Math.exp(-(dist * dist) / (2 * sigma * sigma));
    }
    return base;
  }

  // ── Core: position-dependent concept context ────────────────────────────────

  /**
   * Compute the semantic concept context at world position (wx, wy, wz).
   * @param {number}  wx        - world X (easting)
   * @param {number}  wy        - world Y (northing)
   * @param {number}  [wz=0]   - world Z (elevation)
   * @param {string}  [unitCode] - geological unit code at this point; concepts whose
   *                              unitAffinity excludes this unit contribute zero weight.
   * Returns:
   *   vec        Float32Array(32)  — weighted composite embedding (L2-normalised if >0)
   *   weights    [{id, description, weight}]  — per-concept normalised weights (sorted desc)
   *   tensor     {Ax, Ay, Az}  — anisotropy scales for coordinate warping
   *   totalWeight number        — raw sum before normalisation (0 = no concepts active)
   */
  computeAt(wx, wy, wz = 0, unitCode = null) {
    const DIM = 32;
    const vec     = new Float32Array(DIM);
    const weights = [];
    let totalW    = 0;

    for (const c of this._concepts) {
      // Unit affinity filter: skip concept if it doesn't apply to this unit
      if (unitCode && c.unitAffinity?.length > 0 && !c.unitAffinity.includes(unitCode)) continue;
      const w = this._relevance(c, wx, wy, wz);
      if (w < 0.005) continue;
      weights.push({ id: c.id, description: c.description, weight: w });
      totalW += w;
      const emb = this._effectiveEmbedding(c);  // may blend in parent embedding
      for (let i = 0; i < DIM; i++) vec[i] += w * emb[i];
    }

    if (totalW > 0) {
      for (let i = 0; i < DIM; i++) vec[i] /= totalW;
      weights.sort((a, b) => b.weight - a.weight);
      for (const w of weights) w.weight = w.weight / totalW;
    }

    // Compute active axes for traceability (top axes by |value|)
    const activeAxes = Array.from(vec)
      .map((v, i) => ({ name: CONCEPT_AXES[i], value: v, idx: i }))
      .filter(a => Math.abs(a.value) > 0.1)
      .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
      .slice(0, 8);

    // Depth trend: axes 14-17 encode E/W/N/S deepening directions.
    // Net trend vector in normalised coordinate space (per unit of normalised position).
    // Applied as: z_adj = z + trend.dz_dxN * xNorm + trend.dz_dyN * yNorm
    // This biases the Z coordinate so the implicit surface naturally dips in the
    // predicted direction without needing extra training samples.
    const TREND_SCALE = 0.35; // controls how strongly deepening axes tilt the coordinate
    const trend = {
      dz_dxN: (vec[14] - vec[15]) * TREND_SCALE,  // E deepening → dz/dx positive
      dz_dyN: (vec[16] - vec[17]) * TREND_SCALE,  // N deepening → dz/dy positive
    };

    return { vec, weights, tensor: this._embeddingToTensor(vec), totalWeight: totalW, activeAxes, trend };
  }

  /**
   * Compute soft unit-affinity probability boosts at (wx, wy, wz).
   * Returns a Float32Array of length unitCodes.length where values > 1.0 mean
   * the active concepts in this location favour that unit.
   * Used in inferGeoImplicit to give concepts a direct, interpretable effect on unit prediction.
   * @param {number}   wx - world X
   * @param {number}   wy - world Y
   * @param {number}   wz - world Z
   * @param {string[]} unitCodes - ordered array of unit codes (matches network output indices)
   * @returns {Float32Array}
   */
  computeAffinityBoostsAt(wx, wy, wz, unitCodes) {
    const boosts = new Float32Array(unitCodes.length).fill(1.0);
    for (const c of this._concepts) {
      if (!c.unitAffinity?.length) continue; // no affinity → skip
      const rel = this._relevance(c, wx, wy, wz);
      if (rel < 0.01) continue;
      // Gentle boost: max 30% increase to avoid overwhelming the network's learned distribution
      const boost = 1 + Math.min(0.3, rel * c.confidence * 0.4);
      for (let ui = 0; ui < unitCodes.length; ui++) {
        if (c.unitAffinity.includes(unitCodes[ui])) boosts[ui] *= boost;
      }
    }
    return boosts;
  }

  /**
   * Derive anisotropy tensor from a 32-dim composite embedding.
   *
   * Supports both axis-aligned and diagonally-rotated elongation:
   * — Pure E-W (ew=1, ns=0): Amaj along East, theta=0
   * — Pure N-S (ew=0, ns=1): Amaj along North, theta=π/2
   * — NE-SW (ew=0.6, ns=0.6): Amaj at 45°, strong elongation along NE diagonal
   * — NW-SE (ew=−0.3, ns=0.6): handled by signed projections
   *
   * Ax, Ay are kept as the display-facing axis-aligned scales (for traceability UI).
   * Amaj, Amin, theta, cosT, sinT govern the actual coordinate warp used in training/inference.
   */
  _embeddingToTensor(vec) {
    const ew   = Math.max(-1, Math.min(1, vec[AX_EW]   ?? 0));
    const ns   = Math.max(-1, Math.min(1, vec[AX_NS]   ?? 0));
    const vert = vec[AX_VERT] ?? 0;
    const chan = vec[AX_CHAN]  ?? 0;

    // Axis-aligned display scales — kept for backward compat with traceability / geometry reports
    const Ax = Math.max(0.1, Math.min(10, Math.exp(+ew * 1.4)));
    const Ay = Math.max(0.1, Math.min(10, Math.exp(+ns * 1.4)));
    const Az = Math.max(0.1, Math.min(10, Math.exp(-vert * 1.0 - Math.max(0, chan) * 0.5)));

    // Rotated anisotropy: use positive ew/ns components as a 2D elongation vector
    const ewPos = Math.max(0, ew);
    const nsPos = Math.max(0, ns);
    const magnitude = Math.hypot(ewPos, nsPos);

    if (magnitude > 0.05) {
      // Angle of major axis measured from East, counterclockwise toward North
      // ew=1,ns=0 → 0° (E-W)  |  ew=0,ns=1 → 90° (N-S)  |  ew=0.6,ns=0.6 → 45° (NE-SW)
      const theta = Math.atan2(nsPos, ewPos);
      const cosT  = Math.cos(theta);
      const sinT  = Math.sin(theta);
      // Major: elongated in principal direction; Minor: compressed perpendicular to it
      const Amaj = Math.max(0.5, Math.min(10, Math.exp(magnitude * 1.4)));
      const Amin = Math.max(0.1, Math.min(2,  1 / Math.max(0.3, Amaj * 0.25)));
      return { Ax, Ay, Az, Amaj, Amin, theta, cosT, sinT };
    }

    // No significant positive horizontal elongation: axis-aligned neutral warp
    return { Ax, Ay, Az, Amaj: Math.max(Ax, Ay), Amin: Math.min(Ax, Ay), theta: 0, cosT: 1, sinT: 0 };
  }

  /**
   * Warp world coordinates using the anisotropy tensor before Fourier encoding.
   * Delegates to the exported standalone warpPoint() for rotation-aware warping.
   */
  warpCoords(wx, wy, wz, tensor) {
    return warpPoint(wx, wy, wz, tensor);
  }

  /**
   * Compute the "average" tensor from all concepts (for computing warped bounds).
   * Uses globally-weighted average embedding.
   */
  globalTensor() {
    if (this._concepts.length === 0) return { Ax: 1, Ay: 1, Az: 1 };
    const DIM = 32;
    const avg = new Float32Array(DIM);
    let totalW = 0;
    for (const c of this._concepts) {
      totalW += c.confidence;
      for (let i = 0; i < DIM; i++) avg[i] += c.confidence * c.embedding[i];
    }
    if (totalW > 0) for (let i = 0; i < DIM; i++) avg[i] /= totalW;
    return this._embeddingToTensor(avg);
  }

  // ── Similarity & conflict utilities ─────────────────────────────────────────

  /**
   * Cosine similarity between two Float32Array embeddings.
   * Returns value in [−1, +1]; 1 = identical direction, 0 = orthogonal, −1 = opposite.
   */
  static cosineSimilarity(a, b) {
    let dot = 0, magA = 0, magB = 0;
    const n = Math.min(a.length, b.length);
    for (let i = 0; i < n; i++) {
      dot  += a[i] * b[i];
      magA += a[i] * a[i];
      magB += b[i] * b[i];
    }
    const denom = Math.sqrt(magA) * Math.sqrt(magB);
    return denom > 1e-9 ? dot / denom : 0;
  }

  /**
   * Find existing concepts whose embedding is similar to the given one.
   * @param {Float32Array} embedding
   * @param {number} threshold  cosine similarity cutoff (default 0.80)
   * @returns {Array<{concept, similarity}>} sorted descending
   */
  findSimilar(embedding, threshold = 0.80) {
    return this._concepts
      .map(c => ({ concept: c, similarity: ConceptStore.cosineSimilarity(embedding, c.embedding) }))
      .filter(r => r.similarity >= threshold)
      .sort((a, b) => b.similarity - a.similarity);
  }

  /**
   * Check a single embedding for internal axis contradictions, BEFORE adding it
   * to the store. Returns an array of warning strings (empty = no issues).
   *
   * Mirrors the INTRA_PAIRS logic in detectConceptConflicts() but works on a
   * raw embedding rather than stored concepts, so the UI can warn at encode time.
   */
  static detectIntraConflicts(embedding) {
    const INTRA_PAIRS = [
      { a: 0,  b: 5,  msg: 'horizontal_layering (0) and channel_morphology (5) are geometrically incompatible — one should be negative' },
      { a: 6,  b: 0,  msg: 'dome_anticline (6) and horizontal_layering (0) conflict — domed bodies are not flat-bedded' },
      { a: 7,  b: 5,  msg: 'fault_controlled (7) and channel_morphology (5) operate at different scales and styles' },
      { a: 14, b: 15, msg: 'deepens_east (14) and deepens_west (15) are mutually exclusive dip directions' },
      { a: 16, b: 17, msg: 'deepens_north (16) and deepens_south (17) are mutually exclusive dip directions' },
      { a: 10, b: 11, msg: 'lateral_thinning_east (10) and lateral_thinning_west (11) are mutually exclusive' },
      { a: 12, b: 13, msg: 'lateral_thinning_north (12) and lateral_thinning_south (13) are mutually exclusive' },
      { a: 21, b: 22, msg: 'coarsening_upward (21) and fining_upward (22) cannot both be true simultaneously' },
    ];
    const warnings = [];
    for (const { a, b, msg } of INTRA_PAIRS) {
      if ((embedding[a] ?? 0) > 0.45 && (embedding[b] ?? 0) > 0.45) warnings.push(msg);
    }
    if ((embedding[3] ?? 0) > 0.5 && (embedding[4] ?? 0) > 0.5) {
      warnings.push('E-W elongation (3) and N-S elongation (4) are both high — body will be isotropic; choose the dominant direction');
    }
    return warnings;
  }

  // ── Clone with scaled confidences (for ensemble uncertainty analysis) ────────
  // Returns a new ConceptStore with all concept confidences multiplied by scale.
  // scale = 0 → concepts off (pure borehole), scale = 1 → baseline,
  // scale > 1 → amplified semantic influence.
  cloneScaled(scale) {
    const store = new ConceptStore();
    store._nextId = this._nextId;
    store._concepts = this._concepts.map(c => ({
      ...c,
      embedding: (c.embedding instanceof Float32Array && c.embedding.length === 32)
        ? new Float32Array(c.embedding)
        : new Float32Array(32),
      confidence: Math.max(0, Math.min(1, (c.confidence ?? 0.7) * scale)),
    }));
    return store;
  }

  // Create a perturbed clone for stochastic knowledge uncertainty sampling.
  // Each axis is jittered by N(0, sigma_i) where sigma_i = baseNoise × (1 - data_confidence).
  // data_confidence = embedding[26]; lower confidence → larger perturbation.
  clonePerturbed(baseNoise = 0.12) {
    const store = new ConceptStore();
    store._nextId = this._nextId;
    // Box-Muller for standard normal samples
    const randn = () => {
      let u = 0, v = 0;
      while (!u) u = Math.random();
      while (!v) v = Math.random();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    };
    store._concepts = this._concepts.map(c => {
      const emb = c.embedding instanceof Float32Array ? new Float32Array(c.embedding) : new Float32Array(32);
      const dataConf = Math.max(0.1, Math.min(0.9, emb[26] || 0.6));
      const sigma = baseNoise * (1 - dataConf);
      for (let i = 0; i < 32; i++) {
        if (i === 26) continue; // don't perturb the confidence axis itself
        emb[i] = Math.max(-1, Math.min(1, emb[i] + randn() * sigma));
      }
      return { ...c, embedding: emb };
    });
    return store;
  }

  // ── Serialisation ───────────────────────────────────────────────────────────

  serialize() {
    return JSON.stringify({
      nextId:   this._nextId,
      concepts: this._concepts.map(c => ({ ...c, embedding: Array.from(c.embedding) })),
    });
  }

  static deserialize(json) {
    const store = new ConceptStore();
    try {
      const d = JSON.parse(json);
      store._nextId   = d.nextId ?? 1;
      store._concepts = (d.concepts ?? []).map(c => ({
        ...c,
        embedding: c.embedding?.length ? new Float32Array(c.embedding) : new Float32Array(32),
        confidence: c.confidence ?? 0.7,
        domain: c.domain ?? { type: 'global' },
        unitAffinity: c.unitAffinity ?? [],
      }));
    } catch { /* ignore */ }
    return store;
  }
}
