// ── Section Interpreter ────────────────────────────────────────────────────────
// Converts text descriptions or sketch strokes of geological cross-sections
// into two outputs that influence interpolation:
//   1. Virtual boreholes — dense synthetic observations along the section fence
//      for use with ALL interpolation methods (IDW, kriging, GP, etc.)
//   2. Section training samples — high-confidence training points with a
//      LOCAL semantic keyword context for the neural implicit field.

import { GeoKeywordEncoder } from './geo-implicit.js';

const _kwEnc = new GeoKeywordEncoder();

// ── Fence geometry helpers ────────────────────────────────────────────────────

// Project world point (x, y) onto the fence line. Returns {dist, perp, t}.
// dist = distance along fence from start (m), perp = signed perpendicular offset (m)
export function projectToFence(x, y, fence) {
  const dx = fence.endX - fence.startX, dy = fence.endY - fence.startY;
  const len = Math.hypot(dx, dy) || 1;
  const ux = dx / len, uy = dy / len;         // unit along-fence
  const nx = -uy,  ny = ux;                   // unit normal
  const rx = x - fence.startX, ry = y - fence.startY;
  return {
    dist: rx * ux + ry * uy,                  // along-fence distance (m)
    perp: rx * nx + ry * ny,                  // perpendicular offset (m)
    t:    (rx * ux + ry * uy) / len,          // 0..1 normalised
  };
}

// World position for a given along-fence distance
export function fenceWorldPos(dist, fence) {
  const dx = fence.endX - fence.startX, dy = fence.endY - fence.startY;
  const len = Math.hypot(dx, dy) || 1;
  return {
    x: fence.startX + (dist / len) * dx,
    y: fence.startY + (dist / len) * dy,
  };
}

export function fenceLength(fence) {
  return Math.hypot(fence.endX - fence.startX, fence.endY - fence.startY);
}

// ── Claude prompt ─────────────────────────────────────────────────────────────
export function buildSectionPrompt(text, geoUnits, fence, groundLevel = 0) {
  const len = fenceLength(fence).toFixed(0);
  const unitList = geoUnits
    .filter(u => u.code !== 'UNKN')
    .map(u => `  "${u.code}": ${u.name}`)
    .join('\n');

  const system = `You are a geotechnical AI. Convert geological cross-section descriptions into structured JSON observations for a 3D ground model. Respond with JSON only — no explanation.`;

  const user =
`Section fence: ${len} m long (dist=0 = west/south end, dist=${len} = east/north end).
Ground surface approximately ${groundLevel.toFixed(1)} mAOD.
Depths are positive downward from ground surface.

Available unit codes:
${unitList}

Geological section description:
"""
${text}
"""

Return this JSON structure only:
{
  "virtual_boreholes": [
    {
      "dist_m": <along-fence distance 0..${len}>,
      "layers": [
        {"unit_code": "<code>", "top_m": <depth>, "base_m": <depth or null>, "confidence": <0.5-0.99>}
      ]
    }
  ],
  "contacts": [
    {
      "unit_above": "<code>",
      "unit_below": "<code>",
      "dip_deg": <0-90 positive = deepening toward east/north end>,
      "points": [{"dist_m": <d>, "depth_m": <z>}]
    }
  ],
  "semantic_keywords": ["<geological keyword>"],
  "conceptual_statements": [
    "brief 1-sentence geological concept describing morphology or geometry",
    "e.g. The palaeochannel trends E-W with an erosional concave-up base",
    "e.g. Rockhead deepens steeply to the north across a fault"
  ]
}

Rules:
- Place virtual_boreholes at evenly spaced distances capturing the described stratigraphy (min 3, max 20).
- contacts.points must have at least 2 points spanning the section.
- semantic_keywords: pick 3-8 terms from: dipping thinning thickening pinching lateral inclined wedging deepening shallowing eroded unconformity lens channel fold faulted truncated.
- Only use unit_codes from the list above.
- conceptual_statements: 2-5 statements, each describing a geometric or morphological aspect of the geology. Focus on geometry: orientation, shape (channel/lens/wedge/dome/fault), continuity, depth trends. These will be encoded as concept embeddings to shape the 3D neural field geometry. NOT descriptions of individual layers — geometric/conceptual observations only.`;

  return { system, user };
}

// ── Demo mode fallback ────────────────────────────────────────────────────────
function _demoSection(text, geoUnits, fenceLen) {
  const codes = geoUnits.filter(u => u.code !== 'UNKN').map(u => u.code);
  const c0 = codes[0] ?? 'MG', c1 = codes[1] ?? 'LC', c2 = codes[2] ?? 'SA';
  const dip = /dip|deep|thin|wedg/i.test(text);
  const lens = /lens|pod|lenticle|sand/i.test(text);
  const vbs = [0, fenceLen * 0.25, fenceLen * 0.5, fenceLen * 0.75, fenceLen].map((d, i) => ({
    dist_m: d,
    layers: [
      { unit_code: c0, top_m: 0,              base_m: 1.5 + (dip ? i * 0.3 : 0), confidence: 0.9 },
      { unit_code: c1, top_m: 1.5 + (dip ? i * 0.3 : 0), base_m: null,           confidence: 0.8 },
    ],
  }));
  if (lens && c2) {
    [1, 2, 3].forEach(i => {
      vbs[i].layers.splice(1, 0, { unit_code: c2, top_m: 6, base_m: 9, confidence: 0.7 });
      // keep base of lower unit null
    });
  }
  const fault   = /fault|step|displace/i.test(text);
  const channel = /channel|palaeochannel|fluvial/i.test(text);
  const karst   = /karst|dissolution|void/i.test(text);
  const conceptual_statements = [];
  if (channel) {
    conceptual_statements.push('A palaeochannel feature is present with an erosional concave-up base.');
    conceptual_statements.push('Channel fill exhibits lateral thinning towards the margins.');
  } else if (fault) {
    conceptual_statements.push('A fault-controlled stepped boundary offsets the stratigraphy.');
    conceptual_statements.push('Rockhead deepens abruptly across the fault trend.');
  } else if (karst) {
    conceptual_statements.push('Dissolution features produce an irregular base to the soluble unit.');
    conceptual_statements.push('Structural complexity increases toward dissolution zones.');
  } else if (dip) {
    conceptual_statements.push('Stratigraphy dips and deepens progressively along the section.');
    conceptual_statements.push('Inclined bedding with lateral continuity along the section fence.');
  } else if (lens) {
    conceptual_statements.push('A lenticular sand body is present with lateral thinning to both margins.');
  } else {
    conceptual_statements.push('Stratigraphy is broadly horizontal and laterally continuous across the section.');
    conceptual_statements.push('Flat-lying beds with consistent thickness and no significant dip.');
  }
  return {
    virtual_boreholes: vbs,
    contacts: [{
      unit_above: c0, unit_below: c1, dip_deg: dip ? 3 : 0,
      points: [{ dist_m: 0, depth_m: 1.5 }, { dist_m: fenceLen, depth_m: dip ? 1.5 + 5 * 0.3 : 1.5 }],
    }],
    semantic_keywords: [dip ? 'dipping' : 'lateral', lens ? 'lens' : 'bedded', 'inclined'],
    conceptual_statements,
  };
}

// ── Parse section description via Claude ──────────────────────────────────────
export async function parseSectionFromText(text, geoUnits, fence, apiKey, demoMode = false) {
  const len = fenceLength(fence);
  if (demoMode || !apiKey) {
    await new Promise(r => setTimeout(r, 300));
    return _demoSection(text, geoUnits, len);
  }

  const { system, user } = buildSectionPrompt(text, geoUnits, fence);

  const resp = await fetch('https://api.anthropic.com/v1/messages', {
    method:  'POST',
    headers: {
      'Content-Type':      'application/json',
      'x-api-key':         apiKey,
      'anthropic-version': '2023-06-01',
      'anthropic-dangerous-direct-browser-access': 'true',
    },
    body: JSON.stringify({
      model:      'claude-opus-4-7',
      max_tokens: 2000,
      system,
      messages:   [{ role: 'user', content: user }],
    }),
  });

  if (!resp.ok) throw new Error(`Claude API ${resp.status}: ${await resp.text()}`);
  const data  = await resp.json();
  const raw   = data.content?.[0]?.text ?? '';
  const match = raw.match(/\{[\s\S]*\}/);
  if (!match) throw new Error('No JSON in Claude response');
  return JSON.parse(match[0]);
}

// ── Section → virtual boreholes ───────────────────────────────────────────────
// Converts parsed section JSON into an array of synthetic borehole objects
// compatible with the interpolator (same format as classifiedBH).
export function sectionToVirtualBoreholes(parsed, fence, geoUnits, groundLevel = 0, weight = 0.9) {
  const unitSet = new Set(geoUnits.map(u => u.code));
  const bhs     = [];

  // From explicitly described virtual boreholes
  for (const vb of (parsed.virtual_boreholes ?? [])) {
    const { x, y } = fenceWorldPos(vb.dist_m, fence);
    const layers = (vb.layers ?? [])
      .filter(l => unitSet.has(l.unit_code))
      .map((l, i, arr) => ({
        top:      l.top_m  ?? 0,
        base:     l.base_m ?? (arr[i + 1]?.top_m ?? (l.top_m ?? 0) + 20),
        unitCode: l.unit_code,
        certainty: Math.min(0.98, (l.confidence ?? 0.8) * weight),
        description: `[section] ${l.unit_code}`,
      }))
      .filter(l => l.base > l.top);

    if (layers.length) {
      bhs.push({
        id: `SEC-${vb.dist_m.toFixed(0)}`,
        x, y,
        groundLevel: groundLevel,
        depth: Math.max(...layers.map(l => l.base)),
        synthetic: true,
        _trajectory: null,
        layers,
      });
    }
  }

  // From contact traces — sample along each contact at spacing = ~10% of fence length
  const step = Math.max(5, fenceLength(fence) / 15);
  for (const contact of (parsed.contacts ?? [])) {
    const pts = (contact.points ?? []).sort((a, b) => a.dist_m - b.dist_m);
    if (pts.length < 2) continue;
    if (!unitSet.has(contact.unit_above) && !unitSet.has(contact.unit_below)) continue;

    const minD = pts[0].dist_m, maxD = pts[pts.length - 1].dist_m;
    for (let d = minD; d <= maxD + 0.1; d += step) {
      const cd = Math.min(d, maxD);
      // Linear interpolation along contact points
      let depth = 0;
      for (let i = 1; i < pts.length; i++) {
        if (pts[i].dist_m >= cd) {
          const t = (cd - pts[i - 1].dist_m) / (pts[i].dist_m - pts[i - 1].dist_m + 1e-9);
          depth = pts[i - 1].depth_m + t * (pts[i].depth_m - pts[i - 1].depth_m);
          break;
        }
        depth = pts[pts.length - 1].depth_m;
      }

      const { x, y } = fenceWorldPos(cd, fence);
      const layers = [];
      if (unitSet.has(contact.unit_above) && depth > 0) {
        layers.push({ top: 0, base: depth, unitCode: contact.unit_above,
                      certainty: 0.85 * weight, description: `[section-contact] ${contact.unit_above}` });
      }
      if (unitSet.has(contact.unit_below)) {
        layers.push({ top: depth, base: depth + 20, unitCode: contact.unit_below,
                      certainty: 0.85 * weight, description: `[section-contact] ${contact.unit_below}` });
      }
      if (layers.length) {
        bhs.push({
          id: `SEC-CTR-${cd.toFixed(0)}`,
          x, y, groundLevel, depth: depth + 20,
          synthetic: true, _trajectory: null, layers,
        });
      }
    }
  }

  return bhs;
}

// ── Section → neural implicit training samples ────────────────────────────────
// Returns an array of { inp: Float32Array, target: unitIndex, weight, localKwVec }
// These are merged into trainGeoImplicit's training set.
// The localKwVec encodes section-specific semantics: keywords from the description
// + parsed semantic_keywords, encoded by GeoKeywordEncoder.
export function sectionToTrainingSamples(parsed, fence, geoUnits, fourierEnc,
                                          bounds, sectionText = '', groundLevel = 0) {
  // Local keyword vector for this section
  const sectionKwText = [
    sectionText,
    ...(parsed.semantic_keywords ?? []),
    ...((parsed.contacts ?? []).map(c => `${c.unit_above} ${c.unit_below}`)),
  ].join(' ');
  const localKwVec = _kwEnc.encode(sectionKwText);

  // Build global kwVec from unit descriptions (all zeros here — will be merged by trainGeoImplicit)
  const samples = [];
  const unitIdx = {};
  geoUnits.forEach((u, i) => { unitIdx[u.code] = i; });

  // Samples from virtual boreholes
  for (const vb of (parsed.virtual_boreholes ?? [])) {
    const { x, y } = fenceWorldPos(vb.dist_m, fence);
    for (const l of (vb.layers ?? [])) {
      const ti = unitIdx[l.unit_code];
      if (ti === undefined) continue;
      const top_z  = groundLevel - (l.top_m  ?? 0);
      const base_z = groundLevel - (l.base_m ?? l.top_m + 10);
      const n      = 6;
      for (let s = 0; s < n; s++) {
        const t = (s + 0.5) / n;
        const z = base_z + t * (top_z - base_z);
        const pos = fourierEnc.encode(x, y, z, bounds);
        samples.push({ pos, target: ti, weight: (l.confidence ?? 0.8) * 0.95, localKwVec, x, y, z });
      }
    }
  }

  // Samples along contact traces
  const step = Math.max(5, fenceLength(fence) / 12);
  for (const contact of (parsed.contacts ?? [])) {
    const pts = (contact.points ?? []).sort((a, b) => a.dist_m - b.dist_m);
    if (pts.length < 2) continue;
    const aboveIdx = unitIdx[contact.unit_above], belowIdx = unitIdx[contact.unit_below];

    const minD = pts[0].dist_m, maxD = pts[pts.length - 1].dist_m;
    for (let d = minD; d <= maxD + 0.1; d += step) {
      const cd = Math.min(d, maxD);
      let depth = 0;
      for (let i = 1; i < pts.length; i++) {
        if (pts[i].dist_m >= cd) {
          const t = (cd - pts[i - 1].dist_m) / (pts[i].dist_m - pts[i - 1].dist_m + 1e-9);
          depth = pts[i - 1].depth_m + t * (pts[i].depth_m - pts[i - 1].depth_m);
          break;
        }
        depth = pts[pts.length - 1].depth_m;
      }
      const { x, y } = fenceWorldPos(cd, fence);
      // Above contact: 1m above
      if (aboveIdx !== undefined && depth > 0.5) {
        const z = groundLevel - (depth - 0.5);
        const pos = fourierEnc.encode(x, y, z, bounds);
        samples.push({ pos, target: aboveIdx, weight: 0.88, localKwVec, x, y, z });
      }
      // Below contact: 1m below
      if (belowIdx !== undefined) {
        const z = groundLevel - (depth + 0.5);
        const pos = fourierEnc.encode(x, y, z, bounds);
        samples.push({ pos, target: belowIdx, weight: 0.88, localKwVec, x, y, z });
      }
    }
  }

  return samples;
}

// ── Local context computation at inference (for inferGeoImplicit) ─────────────
// Given a set of section planes, compute the local keyword vector for a world point.
// Blends section keyword vectors by proximity (Gaussian with σ = blendSigmaM).
export function computeLocalContext(x, y, sectionPlanes, blendSigmaM = 30) {
  const out = new Float32Array(_kwEnc.outDim);
  let totalW = 0;

  for (const plane of sectionPlanes) {
    const { dist, perp } = projectToFence(x, y, plane.fence);
    const fLen = fenceLength(plane.fence);
    // Only apply if within fence extent (with margin) and within blendSigma perpendicular
    if (dist < -blendSigmaM || dist > fLen + blendSigmaM) continue;
    const perpW = Math.exp(-(perp * perp) / (2 * blendSigmaM * blendSigmaM));
    if (perpW < 0.01) continue;
    const kw = plane.localKwVec;
    for (let i = 0; i < out.length; i++) out[i] += perpW * kw[i];
    totalW += perpW;
  }

  if (totalW > 0.01) {
    for (let i = 0; i < out.length; i++) out[i] = Math.min(1, out[i] / totalW);
  }
  return out;
}

// ── Sketch strokes → virtual boreholes ───────────────────────────────────────
// strokes: [{unitCode, points: [{distM, depthM}]}]
// Each stroke represents the TOP of a unit; strokes are sorted by depth at each dist.
export function sketchToVirtualBoreholes(strokes, fence, geoUnits, groundLevel = 0, weight = 0.92) {
  if (!strokes.length) return [];
  const unitSet = new Set(geoUnits.map(u => u.code));
  const step    = Math.max(3, fenceLength(fence) / 20);
  const len     = fenceLength(fence);
  const bhs     = [];

  // Sample at regular intervals
  for (let d = 0; d <= len; d += step) {
    // Evaluate each stroke's depth at this distance (linear interpolation)
    const contacts = [];
    for (const stroke of strokes) {
      if (!unitSet.has(stroke.unitCode)) continue;
      const pts = [...stroke.points].sort((a, b) => a.distM - b.distM);
      if (pts.length < 1) continue;
      // Clamp to stroke extent
      const minD = pts[0].distM, maxD = pts[pts.length - 1].distM;
      if (d < minD - step || d > maxD + step) continue;
      const dc = Math.max(minD, Math.min(maxD, d));
      let depth = pts[0].depthM;
      for (let i = 1; i < pts.length; i++) {
        if (pts[i].distM >= dc) {
          const t = (dc - pts[i-1].distM) / (pts[i].distM - pts[i-1].distM + 1e-9);
          depth = pts[i-1].depthM + t * (pts[i].depthM - pts[i-1].depthM);
          break;
        }
        depth = pts[pts.length - 1].depthM;
      }
      contacts.push({ depth, unitCode: stroke.unitCode });
    }

    if (!contacts.length) continue;
    contacts.sort((a, b) => a.depth - b.depth); // sort by depth

    const { x, y } = fenceWorldPos(d, fence);
    const layers   = contacts.map((c, i) => ({
      top:      c.depth,
      base:     contacts[i + 1]?.depth ?? c.depth + 20,
      unitCode: c.unitCode,
      certainty: weight,
      description: `[sketch] ${c.unitCode}`,
    })).filter(l => l.base > l.top);

    if (layers.length) {
      bhs.push({
        id: `SKETCH-${d.toFixed(0)}`,
        x, y, groundLevel,
        depth: Math.max(...layers.map(l => l.base)),
        synthetic: true, _trajectory: null, layers,
      });
    }
  }
  return bhs;
}
