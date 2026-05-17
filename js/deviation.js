/**
 * Borehole Deviation Survey — Minimum Curvature Method
 *
 * Converts a downhole deviation survey {depth, incl_deg, azim_deg} into
 * actual 3D world coordinates for each sample depth.
 *
 * Inclination convention: 0° = vertical, 90° = horizontal.
 * Azimuth convention: degrees clockwise from north.
 */

const DEG = Math.PI / 180;

/**
 * Integrate deviation survey using minimum curvature.
 * Returns cumulative 3D path: array of {depth, x, y, elev}.
 * @param {number} collarX  Collar easting (m)
 * @param {number} collarY  Collar northing (m)
 * @param {number} collarGL Collar ground level (m AOD)
 * @param {Array}  survey   [{depth, incl, azim}] sorted ascending by depth
 */
export function buildDeviationPath(collarX, collarY, collarGL, survey) {
  if (!survey || survey.length === 0) return null;

  // Ensure survey starts at depth 0
  const pts = [{ depth: 0, incl: 0, azim: 0 }];
  for (const s of survey) {
    if (isFinite(s.depth) && s.depth >= 0) {
      pts.push({ depth: s.depth, incl: s.incl ?? 0, azim: s.azim ?? 0 });
    }
  }
  pts.sort((a, b) => a.depth - b.depth);

  const path = [{ depth: 0, x: collarX, y: collarY, elev: collarGL }];
  let x = collarX, y = collarY, elev = collarGL;

  for (let i = 1; i < pts.length; i++) {
    const p1 = pts[i - 1], p2 = pts[i];
    const dd = p2.depth - p1.depth;
    if (dd <= 0) continue;

    const a1 = p1.incl * DEG, b1 = p1.azim * DEG;
    const a2 = p2.incl * DEG, b2 = p2.azim * DEG;

    // Dog-leg angle
    const cosAcos = Math.cos(a2 - a1) - Math.sin(a1) * Math.sin(a2) * (1 - Math.cos(b2 - b1));
    const dl = Math.acos(Math.max(-1, Math.min(1, cosAcos)));

    // Ratio factor (minimum curvature)
    const rf = dl < 1e-6 ? 1.0 : (2.0 / dl) * Math.tan(dl / 2);

    x    += (dd / 2) * (Math.sin(a1) * Math.sin(b1) + Math.sin(a2) * Math.sin(b2)) * rf;
    y    += (dd / 2) * (Math.sin(a1) * Math.cos(b1) + Math.sin(a2) * Math.cos(b2)) * rf;
    elev -= (dd / 2) * (Math.cos(a1) + Math.cos(a2)) * rf;

    path.push({ depth: p2.depth, x, y, elev });
  }

  return path;
}

/**
 * Interpolate 3D position at a given downhole depth along the deviation path.
 * Uses linear interpolation between survey stations.
 * @returns {x, y, elev} world coordinates
 */
export function interpolateAtDepth(path, depth) {
  if (!path || path.length === 0) return null;
  if (depth <= path[0].depth) return { x: path[0].x, y: path[0].y, elev: path[0].elev };

  for (let i = 1; i < path.length; i++) {
    if (depth <= path[i].depth) {
      const t = (depth - path[i - 1].depth) / (path[i].depth - path[i - 1].depth);
      return {
        x:    path[i - 1].x    + t * (path[i].x    - path[i - 1].x),
        y:    path[i - 1].y    + t * (path[i].y    - path[i - 1].y),
        elev: path[i - 1].elev + t * (path[i].elev - path[i - 1].elev),
      };
    }
  }

  // Extrapolate beyond last survey station using final segment direction
  const last = path[path.length - 1];
  const prev = path[path.length - 2] ?? last;
  const segLen = last.depth - prev.depth;
  if (segLen <= 0) return { x: last.x, y: last.y, elev: last.elev };
  const extra = depth - last.depth;
  const frac = extra / segLen;
  return {
    x:    last.x    + frac * (last.x    - prev.x),
    y:    last.y    + frac * (last.y    - prev.y),
    elev: last.elev + frac * (last.elev - prev.elev),
  };
}

/**
 * Add deviation-corrected 3D positions to each layer's midpoint.
 * Mutates bh.deviationPath and sets layer.midX, layer.midY, layer.midElev.
 * Boreholes without a deviation survey fall back to vertical projection.
 */
export function applyDeviationSurveys(boreholes) {
  for (const bh of boreholes) {
    const path = bh.deviation?.length >= 2
      ? buildDeviationPath(bh.x, bh.y, bh.groundLevel, bh.deviation)
      : null;

    bh.deviationPath = path;

    for (const layer of (bh.layers ?? [])) {
      const mid = (layer.top + layer.base) / 2;
      if (path) {
        const pos = interpolateAtDepth(path, mid);
        layer.midX    = pos.x;
        layer.midY    = pos.y;
        layer.midElev = pos.elev;
      } else {
        layer.midX    = bh.x;
        layer.midY    = bh.y;
        layer.midElev = bh.groundLevel - mid;
      }
    }
  }
}

/**
 * Parse a simple 3-column deviation CSV: depth, inclination, azimuth.
 * Header row is auto-detected.
 */
export function parseDeviationCSV(text) {
  const lines = text.split('\n').map(l => l.trim()).filter(Boolean);
  if (!lines.length) return [];

  const COL_DEPTH = ['depth', 'dept', 'dep', 'md', 'measured_depth'];
  const COL_INCL  = ['incl', 'inclination', 'dip', 'inc'];
  const COL_AZIM  = ['azim', 'azimuth', 'bearing', 'azi', 'az'];

  function colIdx(headers, candidates) {
    for (const c of candidates) {
      const i = headers.findIndex(h => h.toLowerCase().replace(/[^a-z]/g,'').startsWith(c));
      if (i >= 0) return i;
    }
    return -1;
  }

  const sep = lines[0].includes('\t') ? '\t' : ',';
  const hdr = lines[0].split(sep).map(s => s.trim().toLowerCase());

  let depIdx = colIdx(hdr, COL_DEPTH);
  let inclIdx = colIdx(hdr, COL_INCL);
  let azimIdx = colIdx(hdr, COL_AZIM);
  let startRow = 0;

  if (depIdx < 0) {
    // No header — assume columns: depth, incl, azim
    depIdx = 0; inclIdx = 1; azimIdx = 2;
  } else {
    startRow = 1;
  }

  const result = [];
  for (let i = startRow; i < lines.length; i++) {
    const cols = lines[i].split(sep);
    const depth = parseFloat(cols[depIdx]);
    const incl  = parseFloat(cols[inclIdx] ?? '0');
    const azim  = parseFloat(cols[azimIdx] ?? '0');
    if (isFinite(depth)) result.push({ depth, incl: isFinite(incl) ? incl : 0, azim: isFinite(azim) ? azim : 0 });
  }
  return result;
}
