// Semantic engine: computes per-voxel certainty adjustments based on
// spatial data density and cross-borehole consistency.

// ── Compute data density score for a location ─────────────────────────────────
// Returns 0–1: how much borehole data constrains a given (x,y) position.
export function dataDensity(x, y, boreholes, searchRadius) {
  let score = 0;
  for (const bh of boreholes) {
    const d = Math.hypot(bh.x - x, bh.y - y);
    if (d < searchRadius) {
      score += Math.exp(-d / searchRadius); // Gaussian decay
    }
  }
  return Math.min(score, 1.0);
}

// ── Compute unit consistency score at a depth slice ───────────────────────────
// Returns 0–1: 1 if all nearby BHs agree on the unit at this depth.
export function unitConsistency(x, y, z, boreholes, searchRadius, dominantUnit) {
  let total = 0, agree = 0;
  for (const bh of boreholes) {
    const d = Math.hypot(bh.x - x, bh.y - y);
    if (d > searchRadius) continue;
    const depth = bh.groundLevel - z;
    if (depth < 0) continue;
    const layer = bh.layers.find(l => depth >= l.top && depth <= l.base);
    if (!layer) continue;
    total++;
    if (layer.unitCode === dominantUnit) agree++;
  }
  return total ? agree / total : 0.5;
}

// ── Final certainty: combines IDW vote confidence + data density + consistency ─
export function computeCertainty(idwConfidence, density, consistency) {
  return Math.min(1.0,
    idwConfidence * 0.5 +
    density       * 0.25 +
    consistency   * 0.25
  );
}
