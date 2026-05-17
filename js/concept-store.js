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
const AX_EW   = 3;   // east_west_elongation   → scale x
const AX_NS   = 4;   // north_south_elongation  → scale y
const AX_VERT = 29;  // incision_depth_ratio    → scale z
const AX_CHAN = 5;   // channel_morphology      → additional vertical compression

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
  add({ description, embedding, confidence = 0.7, domain = { type: 'global' }, unitAffinity = [] }) {
    const id = `c${this._nextId++}`;
    this._concepts.push({
      id, description,
      embedding: embedding instanceof Float32Array ? embedding : new Float32Array(embedding),
      confidence, domain, unitAffinity,
    });
    return id;
  }

  remove(id) {
    this._concepts = this._concepts.filter(c => c.id !== id);
  }

  clear() { this._concepts = []; this._nextId = 1; }

  // ── Spatial relevance ───────────────────────────────────────────────────────

  _relevance(c, wx, wy) {
    const base = c.confidence;
    if (c.domain.type === 'global') return base;
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
      const w = this._relevance(c, wx, wy);
      if (w < 0.005) continue;
      weights.push({ id: c.id, description: c.description, weight: w });
      totalW += w;
      for (let i = 0; i < DIM; i++) vec[i] += w * c.embedding[i];
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
   * Derive anisotropy scale factors from a 32-dim composite embedding.
   *
   * Semantics: Ax, Ay, Az > 1 → divide world coordinates → field changes slowly
   * in that direction → bodies are elongated in that direction.
   * Ax = 1 (neutral), Ax = 4 (strongly E-W elongated), Ax = 0.25 (strongly compressed E-W)
   */
  _embeddingToTensor(vec) {
    const ew   = vec[AX_EW]   ?? 0;   // east_west_elongation
    const ns   = vec[AX_NS]   ?? 0;   // north_south_elongation
    const vert = vec[AX_VERT] ?? 0;   // incision_depth_ratio
    const chan = vec[AX_CHAN]  ?? 0;   // channel_morphology — adds vertical compression

    // exp scale: 0→1.0 (neutral), +1→e^1.4≈4.1 (elongated), −1→e^−1.4≈0.25 (compressed)
    // Clamped to [0.1, 10] to prevent numerical instability
    const Ax = Math.max(0.1, Math.min(10, Math.exp(+ew   * 1.4)));
    const Ay = Math.max(0.1, Math.min(10, Math.exp(+ns   * 1.4)));
    // Channel morphology sharpens vertical boundaries (compresses z → fast variation → sharp)
    const Az = Math.max(0.1, Math.min(10, Math.exp(-vert * 1.0 - Math.max(0, chan) * 0.5)));

    return { Ax, Ay, Az };
  }

  /**
   * Warp world coordinates using the anisotropy tensor before Fourier encoding.
   * Compressing a coordinate (dividing by Ax) makes the Fourier features change
   * more slowly in that direction → implicit field changes slowly → unit bodies
   * are elongated in that direction.
   */
  warpCoords(wx, wy, wz, tensor) {
    return {
      x: wx / tensor.Ax,
      y: wy / tensor.Ay,
      z: wz / tensor.Az,
    };
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
        ...c, embedding: new Float32Array(c.embedding),
      }));
    } catch { /* ignore */ }
    return store;
  }
}
