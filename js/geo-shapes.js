// js/geo-shapes.js — Geological shape primitive library
// Natural-language geological feature descriptions → virtual observation points
// that participate in the IDW/Kriging interpolation like boreholes.
//
// Shape types:
//   palaeochannel  — parabolic trough; linear feature
//   lens / pod     — 3D ellipsoidal isolated body
//   buried_hill    — Gaussian dome (e.g. buried hill, raised basement)
//   pinch_out      — unit that thins to zero at a line
//   unconformity   — erosional surface that truncates units below it
//   fold           — sinusoidally deformed stratigraphy
//   intrusion      — now handled directly in constraints.js (kept for parity)
//
// Each shape produces virtual boreholes that are added to allBoreholes before
// interpolation. Their certainty is weighted by the shape's prior confidence
// and the semantic weight setting.

// ─── Shape function: Palaeochannel ────────────────────────────────────────────
// A parabolic trough running at orientationDeg (degrees from North, clockwise).
// centreX, centreY — trough centreline in grid coords
// widthM — full width of channel in metres
// maxDepthM — maximum incision depth (at centreline)
// lengthM — along-strike extent
// unitCode — unit filling the channel
// Returns array of virtual sample points {x, y, z, unitCode, certainty}
function palaeochannelPoints(
  centreX, centreY, surfaceElev, orientationDeg, widthM, maxDepthM, lengthM, unitCode, confidence
) {
  const points = [];
  const azRad = orientationDeg * Math.PI / 180;
  const sinA  = Math.sin(azRad), cosA = Math.cos(azRad);

  // Sample along the centreline (-halfLen to +halfLen) and across (-halfW to +halfW)
  const nAlong = Math.max(5, Math.ceil(lengthM / 15));
  const nAcross = Math.max(5, Math.ceil(widthM / 10));
  const halfLen = lengthM / 2, halfW = widthM / 2;

  for (let ia = 0; ia <= nAlong; ia++) {
    const tAlong = -halfLen + (ia / nAlong) * lengthM;
    const cx = centreX + tAlong * sinA;
    const cy = centreY + tAlong * cosA;

    for (let iw = 0; iw <= nAcross; iw++) {
      const tAcross = -halfW + (iw / nAcross) * widthM;
      // Perpendicular direction
      const px = cx + tAcross * cosA;
      const py = cy - tAcross * sinA;

      // Parabolic depth profile across width
      const normAcross = tAcross / halfW; // -1 to +1
      const depth = maxDepthM * (1 - normAcross * normAcross);

      if (depth > 0.1) {
        // Two points: top of channel fill and bottom
        const topElev = surfaceElev - (maxDepthM - depth);
        const botElev = surfaceElev - maxDepthM;
        points.push({ x: px, y: py, elev: topElev - 0.5, unitCode, certainty: confidence * 0.7 });
        points.push({ x: px, y: py, elev: botElev + 0.3,  unitCode, certainty: confidence * 0.5 });
      }
    }
  }
  return points;
}

// ─── Shape function: Lens / Pod ────────────────────────────────────────────────
// An isolated ellipsoidal body.
function lensPoints(cx, cy, cElev, rxM, ryM, rzM, orientationDeg, unitCode, confidence) {
  const points = [];
  const azRad  = orientationDeg * Math.PI / 180;
  const sinA   = Math.sin(azRad), cosA = Math.cos(azRad);
  const nSamp  = 20;

  // Sample on a grid of the bounding box and accept inside ellipsoid
  for (let ix = -2; ix <= 2; ix++) {
    for (let iy = -2; iy <= 2; iy++) {
      for (let iz = -2; iz <= 2; iz++) {
        // Rotated local coords
        const lx = ix * rxM * 0.4, ly = iy * ryM * 0.4, lz = iz * rzM * 0.4;
        if ((lx/rxM)**2 + (ly/ryM)**2 + (lz/rzM)**2 > 0.9) continue;
        // Rotate in plan by orientationDeg
        const wx = cx + lx * cosA - ly * sinA;
        const wy = cy + lx * sinA + ly * cosA;
        points.push({ x: wx, y: wy, elev: cElev + lz, unitCode, certainty: confidence * 0.8 });
      }
    }
  }
  return points;
}

// ─── Shape function: Buried Hill / Dome ──────────────────────────────────────
// Gaussian uplift of the basement unit; creates a dome rising above surrounding level.
function buriedHillPoints(cx, cy, crestElev, amplitudeM, halfWidthM, unitCode, confidence) {
  const points = [];
  const nR = 5, nAz = 8;
  for (let ir = 0; ir <= nR; ir++) {
    const r = (ir / nR) * halfWidthM * 2;
    const elev = crestElev + amplitudeM * Math.exp(-(r * r) / (2 * halfWidthM * halfWidthM));
    for (let iaz = 0; iaz < nAz; iaz++) {
      const az = (iaz / nAz) * 2 * Math.PI;
      const px = cx + r * Math.cos(az);
      const py = cy + r * Math.sin(az);
      // Points just above and below crest surface
      points.push({ x: px, y: py, elev: elev - 0.5, unitCode, certainty: confidence * 0.75 });
      points.push({ x: px, y: py, elev: elev + 0.5, unitCode: null, certainty: confidence * 0.5 }); // absence above
    }
  }
  return points;
}

// ─── Convert virtual points → synthetic boreholes ─────────────────────────────
// Compatible format with AppState.classifiedBH entries so interpolator can use them.
export function shapePointsToBoreholes(shapePoints, groundLevel, semanticWeight) {
  const bhMap = {}; // key = `${x.toFixed(1)}_${y.toFixed(1)}` → borehole

  for (const pt of shapePoints) {
    if (!pt.unitCode && !pt.absence) continue;
    const key = `${pt.x.toFixed(1)}_${pt.y.toFixed(1)}`;
    if (!bhMap[key]) {
      bhMap[key] = {
        id: `SHAPE-${key}`,
        x: pt.x, y: pt.y,
        groundLevel: groundLevel,
        depth: groundLevel - (pt.elev - 1),
        synthetic: true,
        _trajectory: null,
        layers: [],
      };
    }
    const bh = bhMap[key];
    const depth = groundLevel - pt.elev;
    if (depth < 0) continue;
    if (pt.unitCode) {
      bh.layers.push({
        top: Math.max(0, depth - 0.5),
        base: depth + 0.5,
        unitCode: pt.unitCode,
        certainty: pt.certainty * (1 - semanticWeight * 0.3),
        description: '',
      });
      bh.depth = Math.max(bh.depth, depth + 0.5);
    }
  }
  return Object.values(bhMap).filter(bh => bh.layers.length > 0);
}

// ─── Parse geological features from natural language ──────────────────────────
// Called with Claude response (structured JSON) or falls back to regex parsing.
// Returns array of shape definitions.
export function parseShapesFromClaude(claudeOutput, geoUnits) {
  const shapes = [];
  const unitByName = {};
  const unitByCode = {};
  geoUnits.forEach(u => {
    unitByName[u.name?.toLowerCase() ?? ''] = u;
    unitByCode[u.code?.toLowerCase() ?? ''] = u;
  });

  const findUnit = (text) => {
    if (!text) return null;
    const lower = text.toLowerCase();
    for (const u of geoUnits) {
      if (lower === u.code?.toLowerCase()) return u;
    }
    for (const u of geoUnits) {
      if (u.name && lower.includes(u.name.toLowerCase())) return u;
      if (u.code && lower.includes(u.code.toLowerCase())) return u;
    }
    return null;
  };

  // Claude is expected to return an array of shape objects
  const items = Array.isArray(claudeOutput) ? claudeOutput : [claudeOutput];
  for (const item of items) {
    if (!item?.feature_type) continue;
    const unit = findUnit(item.unit_code ?? item.unit ?? '');
    shapes.push({ ...item, unit });
  }
  return shapes;
}

// ─── Generate virtual borehole points from parsed shapes ──────────────────────
// bbox = { minX, maxX, minY, maxY, maxGL, maxDepth }
export function generateShapeBoreholes(shapes, bbox, semanticWeight) {
  const { minX, maxX, minY, maxY, maxGL } = bbox;
  const midX = (minX + maxX) / 2, midY = (minY + maxY) / 2;
  const siteW = maxX - minX || 100, siteH = maxY - minY || 100;
  const allPoints = [];

  for (const shape of shapes) {
    const unit = shape.unit;
    const conf = shape.confidence ?? 0.5;
    const cx = minX + (shape.centroid_x_frac ?? 0.5) * siteW;
    const cy = minY + (shape.centroid_y_frac ?? 0.5) * siteH;
    const elev = shape.elevation_mAOD ?? (maxGL - (shape.depth_m ?? 5));
    const ft = shape.feature_type?.toLowerCase() ?? '';

    if (ft.includes('palaeochannel') || ft.includes('channel')) {
      const pts = palaeochannelPoints(
        cx, cy, maxGL,
        shape.orientation_deg ?? 90,
        shape.width_m ?? 40,
        shape.max_depth_m ?? 6,
        shape.length_m ?? (siteW * 0.7),
        unit?.code,
        conf
      );
      allPoints.push(...pts);
    } else if (ft.includes('lens') || ft.includes('pod') || ft.includes('lenticle')) {
      const rx = shape.rx_m ?? shape.radius_m ?? 20;
      const ry = shape.ry_m ?? rx;
      const rz = shape.rz_m ?? rx * 0.3;
      const pts = lensPoints(cx, cy, elev, rx, ry, rz, shape.orientation_deg ?? 0, unit?.code, conf);
      allPoints.push(...pts);
    } else if (ft.includes('hill') || ft.includes('dome') || ft.includes('mound') || ft.includes('ridge')) {
      const pts = buriedHillPoints(cx, cy, elev, shape.amplitude_m ?? 5, shape.half_width_m ?? 30, unit?.code, conf);
      allPoints.push(...pts);
    }
  }

  return shapePointsToBoreholes(allPoints, maxGL, semanticWeight);
}

// ─── Prompt template for Claude ────────────────────────────────────────────────
// Returns the system+user prompt to send to Claude to parse feature descriptions.
export function buildShapeParsePrompt(featureText, geoUnits, bbox) {
  const unitList = geoUnits.map(u => `${u.code} (${u.name ?? ''})`).join(', ');
  const { minX, maxX, minY, maxY } = bbox;

  return {
    system: `You are a geotechnical expert. Parse geological feature descriptions into structured shape primitives for a 3D ground model. Respond ONLY with valid JSON — an array of shape objects.

Available unit codes: ${unitList}
Site bounds: Easting ${minX.toFixed(0)}–${maxX.toFixed(0)} m, Northing ${minY.toFixed(0)}–${maxY.toFixed(0)} m

Each shape object must have:
- feature_type: "palaeochannel" | "lens" | "buried_hill" | "pinch_out" | "fold"
- unit_code: string (from the list above, or null if feature is absence/void)
- confidence: 0.0–1.0
- centroid_x_frac: 0–1 (fractional position in site E-W)
- centroid_y_frac: 0–1 (fractional position in site N-S)
- orientation_deg: degrees from North (clockwise) for elongated features
- For palaeochannel: width_m, max_depth_m, length_m
- For lens: rx_m, ry_m, rz_m (semi-axes)
- For buried_hill: amplitude_m, half_width_m`,
    user: featureText,
  };
}
