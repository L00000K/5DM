// Semantic engine: computes per-voxel certainty adjustments based on
// spatial data density and cross-borehole consistency.
// Also provides geological description similarity and transition matrix functions.

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

// ── Geological description similarity (Jaccard on keyword tokens) ─────────────
export function descriptionJaccard(a, b) {
  // geological stop words to filter
  const STOP = new Set(['the','and','with','of','in','to','a','an','at','by','for','is','it','some','trace','occasional','frequent','becoming','grading']);
  const tok = s => new Set(s.toLowerCase().split(/\W+/).filter(w => w.length > 2 && !STOP.has(w)));
  const A = tok(a ?? ''), B = tok(b ?? '');
  if (!A.size || !B.size) return 0;
  let inter = 0;
  for (const w of A) if (B.has(w)) inter++;
  return inter / new Set([...A, ...B]).size;
}

// ── Mean pairwise similarity for a set of descriptions ────────────────────────
export function meanDescriptionSimilarity(descs) {
  const v = descs.filter(Boolean);
  if (v.length < 2) return 0.5;
  let sum = 0, cnt = 0;
  for (let i = 0; i < v.length; i++)
    for (let j = i + 1; j < v.length; j++) { sum += descriptionJaccard(v[i], v[j]); cnt++; }
  return cnt > 0 ? sum / cnt : 0.5;
}

// ── Unit transition probability matrix from observed BH sequences ──────────────
export function buildTransitionMatrix(classifiedBH, geoUnits) {
  // Matrix[i][j] = P(unit j directly below unit i), with Laplace smoothing
  const n = geoUnits.length;
  const idx = {}; geoUnits.forEach((u, i) => { idx[u.code] = i; });
  const counts = Array.from({ length: n }, () => new Array(n).fill(0));
  for (const bh of classifiedBH) {
    const layers = [...bh.layers].sort((a, b) => a.top - b.top);
    for (let i = 0; i < layers.length - 1; i++) {
      const ai = idx[layers[i].unitCode], bi = idx[layers[i + 1].unitCode];
      if (ai !== undefined && bi !== undefined) counts[ai][bi]++;
    }
  }
  return counts.map(row => {
    const sum = row.reduce((a, b) => a + b, 0) + n;
    return row.map(v => (v + 1) / sum); // Laplace-smoothed
  });
}

// ── Build stratigraphic rank map from inferred order ──────────────────────────
// stratOrder: array of unit codes from top to bottom (index 0 = shallowest)
// Returns Map<unitCode, rank> where lower rank = shallower
export function buildStratRankMap(stratOrder) {
  const map = new Map();
  stratOrder.forEach((code, i) => map.set(code, i));
  return map;
}

// ── Stratigraphic consistency penalty ─────────────────────────────────────────
// Returns a multiplier 0.2–1.0 applied to certainty.
// Penalises a voxel whose unit appears ABOVE its expected stratigraphic position
// relative to a neighbouring unit observed in the same borehole column.
//
// unitCode:   the predicted unit for this voxel
// depth:      depth below ground level (positive downward, metres)
// x, y:       voxel centre world coords
// classifiedBH: array of classified borehole objects
// stratRanks: Map<unitCode, rank> from buildStratRankMap()
// searchRadius: horizontal search radius in metres
export function stratigraphicConsistencyPenalty(
  unitCode, depth, x, y, classifiedBH, stratRanks, searchRadius,
) {
  if (!stratRanks.size) return 1.0;
  const predictedRank = stratRanks.get(unitCode);
  if (predictedRank == null) return 1.0; // unknown unit — no penalty

  let violations = 0, comparisons = 0;

  for (const bh of classifiedBH) {
    const dist = Math.hypot(bh.x - x, bh.y - y);
    if (dist > searchRadius) continue;

    // Find what unit is observed in this BH at the same depth
    const obsLayer = bh.layers.find(l => depth >= l.top && depth <= l.base);
    if (!obsLayer) continue;
    const obsRank = stratRanks.get(obsLayer.unitCode);
    if (obsRank == null) continue;

    comparisons++;
    // If our predicted unit has a LOWER rank (= shallower) but we're deeper
    // than expected, that's a stratigraphic inversion → violation
    if (predictedRank < obsRank && depth > obsLayer.base) violations++;
    else if (predictedRank > obsRank && depth < obsLayer.top) violations++;
  }

  if (comparisons === 0) return 1.0;
  const violationRate = violations / comparisons;
  // Penalty: 20% minimum, linear from 0% to 100% violations
  return Math.max(0.2, 1.0 - violationRate * 0.8);
}
