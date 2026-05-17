import { analysisLog, log, AppState } from './app.js';
import { CONCEPT_AXES } from './concept-store.js';

const API_URL    = 'https://api.anthropic.com/v1/messages';
const MODEL      = 'claude-opus-4-5';
const MAX_TOKENS = 1024;

const SYSTEM_PROMPT = `You are an expert geotechnical and engineering geologist with 30 years of experience classifying ground investigation data in the UK. Your task is to classify borehole log descriptions into geological units with precise engineering parameters. Always respond with valid JSON only — no markdown, no explanation, no preamble.`;

// ── Claude API call ────────────────────────────────────────────────────────────
async function callClaude(messages, apiKey) {
  const resp = await fetch(API_URL, {
    method: 'POST',
    headers: {
      'Content-Type':      'application/json',
      'x-api-key':         apiKey,
      'anthropic-version': '2023-06-01',
    },
    body: JSON.stringify({
      model: MODEL,
      max_tokens: MAX_TOKENS,
      system: SYSTEM_PROMPT,
      messages,
    }),
  });

  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    throw new Error(err?.error?.message || `API error ${resp.status}`);
  }

  const data = await resp.json();
  const text = data.content?.[0]?.text || '';

  // Strip any accidental markdown fences
  const cleaned = text.replace(/^```[a-z]*\n?/i, '').replace(/\n?```$/i, '').trim();
  try {
    return JSON.parse(cleaned);
  } catch {
    throw new Error(`Claude returned non-JSON: ${text.slice(0, 120)}`);
  }
}

// ── Classify a single borehole layer description ───────────────────────────────
async function classifyLayer(description, knownUnits, apiKey) {
  const unitsHint = knownUnits.length
    ? `Known units on this site: ${knownUnits.map(u => `${u.code} (${u.name})`).join(', ')}. Prefer these codes if appropriate.`
    : '';

  const messages = [{
    role: 'user',
    content: `Classify this borehole layer description into a geological unit.
${unitsHint}

Description: "${description}"

Respond with JSON in exactly this format:
{
  "unit_code": "short code e.g. MG/AC/CH/RTD/LC",
  "unit_name": "full unit name",
  "material": "primary material e.g. CLAY/SAND/GRAVEL/CHALK",
  "consistency": "e.g. soft/firm/stiff/dense/medium dense",
  "colour": "e.g. grey/brown/white",
  "plasticity": "low/medium/high/unknown",
  "undrained_strength_kPa": null or number,
  "spt_n": null or number,
  "color_hex": "hex colour for visualisation e.g. #4A7C59",
  "confidence": 0.0 to 1.0,
  "engineering_notes": "one sentence"
}`
  }];

  return callClaude(messages, apiKey);
}

// ── Discover geological units from a batch of descriptions ─────────────────────
async function discoverUnits(descriptions, apiKey) {
  const sample = descriptions.slice(0, 40).join('\n');
  const messages = [{
    role: 'user',
    content: `From these borehole layer descriptions, identify the distinct geological units present on this site.

Descriptions:
${sample}

Respond with JSON in exactly this format:
{
  "units": [
    {
      "code": "short code e.g. MG",
      "name": "full name e.g. Made Ground",
      "color": "#hex",
      "description": "one line engineering description",
      "age": "e.g. Holocene / Cretaceous",
      "engineering_notes": "one line"
    }
  ]
}`
  }];

  return callClaude(messages, apiKey);
}

// ── Demo mode mock AI ──────────────────────────────────────────────────────────
function demoClassify(layer, geoUnits) {
  const desc = (layer.description || '').toLowerCase();

  // Use pre-attached unit code if available (from demo JSON)
  if (layer.unitCode && geoUnits.find(u => u.code === layer.unitCode)) {
    return {
      unit_code:    layer.unitCode,
      confidence:   layer.certainty ?? 0.9,
    };
  }

  // Heuristic fallback
  if (desc.includes('fill') || desc.includes('made ground') || desc.includes('rubble')) {
    return { unit_code: 'MG', confidence: 0.88 };
  }
  if (desc.includes('chalk')) return { unit_code: 'CH', confidence: 0.94 };
  if (desc.includes('clay'))  return { unit_code: 'AC', confidence: 0.82 };
  if (desc.includes('gravel') || desc.includes('sand')) return { unit_code: 'RTD', confidence: 0.87 };
  return { unit_code: 'UNKN', confidence: 0.3 };
}

// ── Geological interpretation: semantic knowledge → structured constraints ────
export async function interpretGeology(siteHistory, unitDescriptions, geoUnits, apiKey, demoMode) {
  const unitContext = geoUnits.length
    ? geoUnits.map(u => `${u.code}: ${u.name}${u.description ? ' — ' + u.description : ''}`).join('\n')
    : '(no units defined yet)';

  if (demoMode || !apiKey) {
    return _demoInterpretation(geoUnits);
  }

  const messages = [{
    role: 'user',
    content: `You are a senior engineering geologist in the UK. Analyse the following site information and provide a rigorous geological interpretation that can be used to constrain a 3D ground model.

GEOLOGICAL UNITS IN MODEL:
${unitContext}

UNIT DESCRIPTIONS:
${unitDescriptions || '(none provided)'}

SITE HISTORY / CONTEXT:
${siteHistory || '(none provided)'}

Produce a structured geological interpretation. Respond with JSON ONLY (no markdown):
{
  "interpretation_summary": "2–4 sentence geological narrative for the site",
  "stratigraphic_order": ["youngest_unit_code", "...", "oldest_unit_code"],
  "stratigraphic_notes": {"UNIT_CODE": "one sentence on origin and character"},
  "constraints": [
    "natural language constraint rules e.g. 'Made Ground not deeper than 4m'",
    "e.g. 'Chalk is above 30mAOD'",
    "e.g. 'Alluvial Clay exists only in the south half'"
  ],
  "hazards": [
    {"type": "settlement|liquefaction|contamination|collapse|groundwater|slope", "description": "brief description", "unit_code": "code or null"}
  ],
  "colour_suggestions": {"UNIT_CODE": "#hexcolour"},
  "interpolation_advice": "one sentence on recommended cell size, search radius or method",
  "confidence": 0.0
}
`
  }];

  try {
    return await callClaude(messages, apiKey);
  } catch (err) {
    throw new Error(`Geological interpretation failed: ${err.message}`);
  }
}

function _demoInterpretation(geoUnits) {
  const codes = geoUnits.map(u => u.code);
  const order = codes.length ? [...codes] : ['MG', 'RTD', 'ACL', 'CH'];
  return {
    interpretation_summary:
      'The site comprises superficial deposits overlying bedrock in a typical UK lowland sequence. ' +
      'Made Ground is present across the site reflecting historical development. ' +
      'Alluvial deposits infill a palaeochannel feature in the northern third of the site. ' +
      'Chalk bedrock is encountered at depth with variable head deposits in between.',
    stratigraphic_order: order,
    stratigraphic_notes: Object.fromEntries(geoUnits.map(u => [u.code, `${u.name}: typical UK engineering unit.`])),
    constraints: [
      geoUnits[0] ? `${geoUnits[0].name} not deeper than 5m` : 'Made Ground not deeper than 5m',
      geoUnits[geoUnits.length - 1]
        ? `${geoUnits[geoUnits.length - 1].name} is above 20mAOD`
        : 'Chalk is above 20mAOD',
    ],
    hazards: [
      { type: 'settlement', description: 'Soft alluvial deposits may produce differential settlement.', unit_code: geoUnits[1]?.code ?? null },
      { type: 'groundwater', description: 'Shallow groundwater likely in alluvial sequence.', unit_code: null },
    ],
    colour_suggestions: {},
    interpolation_advice: 'Use IDW with 5 neighbours and 1 m horizontal cell size for this site scale.',
    confidence: 0.7,
  };
}

// ── Infer stratigraphic order from borehole data (no AI needed) ───────────────
export function inferStratOrderFromData(classifiedBH, geoUnits) {
  // Build directed graph: A above B ↔ edge A → B
  const unitCodes = new Set(geoUnits.map(u => u.code));
  const pairCount = {}; // `${above}→${below}`: count

  for (const bh of classifiedBH) {
    const layers = [...bh.layers].sort((a, b) => a.top - b.top); // top-most first
    for (let i = 0; i < layers.length - 1; i++) {
      const a = layers[i].unitCode;
      const b = layers[i + 1].unitCode;
      if (!a || !b || a === b || !unitCodes.has(a) || !unitCodes.has(b)) continue;
      const key = `${a}→${b}`;
      pairCount[key] = (pairCount[key] ?? 0) + 1;
    }
  }

  // Simple topological sort using pair counts
  const inDegree = {};
  const adj = {};
  geoUnits.forEach(u => { inDegree[u.code] = 0; adj[u.code] = []; });

  for (const [key, cnt] of Object.entries(pairCount)) {
    const [a, b] = key.split('→');
    if (!adj[a]) continue;
    if (!adj[a].includes(b)) { adj[a].push(b); inDegree[b] = (inDegree[b] ?? 0) + 1; }
  }

  // Kahn's algorithm
  const queue  = geoUnits.map(u => u.code).filter(c => (inDegree[c] ?? 0) === 0);
  const order  = [];
  while (queue.length) {
    const node = queue.shift();
    order.push(node);
    (adj[node] ?? []).forEach(nb => {
      inDegree[nb]--;
      if (inDegree[nb] === 0) queue.push(nb);
    });
  }
  // Append any remaining (cycles)
  geoUnits.forEach(u => { if (!order.includes(u.code)) order.push(u.code); });

  const pairs = Object.entries(pairCount)
    .map(([key, count]) => { const [above, below] = key.split('→'); return { above, below, count }; })
    .sort((a, b) => b.count - a.count);

  return { order, pairs };
}

// ── Main entry: run full analysis pipeline ─────────────────────────────────────
export async function runAIAnalysis(boreholes, apiKey, demoMode) {
  // Step 1: discover / confirm units
  let geoUnits;

  if (demoMode || !apiKey) {
    analysisLog('Demo Mode', 'Using pre-loaded geological units — no API key required.', 'ai');
    geoUnits = AppState.geoUnits.length ? AppState.geoUnits : defaultUnits();
  } else {
    log('Discovering geological units…', 'info');
    analysisLog('Unit Discovery', 'Analysing all layer descriptions to identify geological units…', 'ai');
    const allDesc = boreholes.flatMap(bh => bh.layers.map(l => l.description)).filter(Boolean);
    try {
      const result = await discoverUnits(allDesc, apiKey);
      geoUnits = result.units.map((u, i) => ({ id: i + 1, ...u }));
      analysisLog('Units Found', `${geoUnits.length} geological units identified:\n${geoUnits.map(u => `• ${u.code}: ${u.name}`).join('\n')}`, 'ok');
    } catch (err) {
      analysisLog('Unit Discovery Failed', err.message + '\nFalling back to heuristic classification.', 'warn');
      geoUnits = defaultUnits();
    }
  }

  // Ensure 'Unknown' unit exists
  if (!geoUnits.find(u => u.code === 'UNKN')) {
    geoUnits.push({ id: 0, code: 'UNKN', name: 'Unknown', color: '#888888', description: 'Unclassified' });
  }

  // Step 2: classify each layer
  const classified = [];
  let total  = boreholes.reduce((s, bh) => s + bh.layers.length, 0);
  let done   = 0;
  let errors = 0;

  for (const bh of boreholes) {
    const bhCopy = { ...bh, layers: [] };

    for (const layer of bh.layers) {
      done++;
      try {
        let result;

        if (demoMode || !apiKey || layer.classified) {
          result = demoClassify(layer, geoUnits);
        } else {
          result = await classifyLayer(layer.description, geoUnits, apiKey);
          // Rate limit: small delay between calls
          await new Promise(r => setTimeout(r, 150));
        }

        // Ensure the unit code exists
        let unit = geoUnits.find(u => u.code === result.unit_code);
        if (!unit) {
          // Add as new unit
          unit = {
            id:          geoUnits.length + 1,
            code:        result.unit_code,
            name:        result.unit_name || result.unit_code,
            color:       result.color_hex || '#888888',
            description: result.engineering_notes || '',
          };
          geoUnits.push(unit);
        }

        bhCopy.layers.push({
          ...layer,
          unitCode:   result.unit_code,
          certainty:  result.confidence ?? 0.8,
          classified: true,
        });

        if (!demoMode && apiKey) {
          analysisLog(
            `${bh.id} · ${layer.top.toFixed(1)}–${layer.base.toFixed(1)}m`,
            `→ ${result.unit_code} (${unit.name}) — confidence ${(result.confidence * 100).toFixed(0)}%\n${result.engineering_notes || ''}`,
            'ai'
          );
        }
      } catch (err) {
        errors++;
        bhCopy.layers.push({ ...layer, unitCode: 'UNKN', certainty: 0.2 });
        if (errors <= 3) log(`Layer classify error: ${err.message}`, 'warn');
      }

      log(`Classifying layers… ${done}/${total}`, done < total ? 'info' : 'ok');
    }
    classified.push(bhCopy);
  }

  if (demoMode || !apiKey) {
    analysisLog('Classification Complete',
      `${total} layers classified across ${boreholes.length} boreholes.\nDemo mode: using pre-classified data.`, 'ok');
  } else {
    analysisLog('Classification Complete',
      `${total - errors} layers classified successfully.\n${errors} errors (marked as Unknown).`, errors ? 'warn' : 'ok');
  }

  return { units: geoUnits, classified };
}

// ── Infer geotechnical parameters from unit description ────────────────────────
export async function inferUnitParameters(unit, apiKey, demoMode) {
  if (demoMode || !apiKey) {
    return _demoParams(unit);
  }
  const messages = [{
    role: 'user',
    content: `Based on this geological unit, infer typical UK geotechnical engineering parameters for preliminary design.

Unit: ${unit.code} — ${unit.name}
Description: ${unit.description ?? '(none)'}

Respond with JSON only (no explanation):
{
  "gamma_kNm3":  number or null,
  "cu_kPa":      number or null,
  "phi_deg":     number or null,
  "cprime_kPa":  number or null,
  "E_MPa":       number or null,
  "Cc":          number or null,
  "e0":          number or null,
  "N_spt":       number or null,
  "notes":       "one sentence justification"
}
`
  }];
  return callClaude(messages, apiKey);
}

function _demoParams(unit) {
  const n = (unit.name ?? unit.code).toLowerCase();
  const d = (unit.description ?? '').toLowerCase();
  const all = n + ' ' + d;
  // Material heuristics for UK geology
  if (/made ground|fill|rubble/.test(all))
    return { gamma_kNm3: 18, cu_kPa: null, phi_deg: 28, cprime_kPa: 0, E_MPa: 5,  Cc: null, e0: null, N_spt: 8,  notes: 'Typical UK made ground parameters' };
  if (/chalk/.test(all))
    return { gamma_kNm3: 20, cu_kPa: null, phi_deg: 35, cprime_kPa: 5, E_MPa: 80, Cc: null, e0: null, N_spt: 30, notes: 'Chalk — highly variable; SPT N from in-situ' };
  if (/gravel|terrace|rtd|sand and gravel/.test(all))
    return { gamma_kNm3: 20, cu_kPa: null, phi_deg: 35, cprime_kPa: 0, E_MPa: 40, Cc: null, e0: null, N_spt: 30, notes: 'Dense river terrace gravel' };
  if (/sand/.test(all))
    return { gamma_kNm3: 19, cu_kPa: null, phi_deg: 32, cprime_kPa: 0, E_MPa: 25, Cc: null, e0: null, N_spt: 20, notes: 'Loose to medium dense sand' };
  if (/london clay|lc/.test(all))
    return { gamma_kNm3: 20, cu_kPa: 120, phi_deg: 25, cprime_kPa: 5, E_MPa: 60,  Cc: 0.15, e0: 0.7, N_spt: 30, notes: 'London Clay — stiff fissured clay' };
  if (/soft|alluvial|estuarine/.test(all))
    return { gamma_kNm3: 16, cu_kPa: 20,  phi_deg: 24, cprime_kPa: 0, E_MPa: 3,   Cc: 0.5,  e0: 1.2, N_spt: 4,  notes: 'Soft alluvial clay — settlement-critical' };
  if (/peat/.test(all))
    return { gamma_kNm3: 11, cu_kPa: 8,   phi_deg: 20, cprime_kPa: 0, E_MPa: 0.5, Cc: 2.0,  e0: 3.0, N_spt: 2,  notes: 'Peat — highly compressible' };
  if (/clay/.test(all))
    return { gamma_kNm3: 19, cu_kPa: 60,  phi_deg: 24, cprime_kPa: 0, E_MPa: 15,  Cc: 0.3,  e0: 0.9, N_spt: 15, notes: 'Generic firm clay' };
  // Generic
  return { gamma_kNm3: 19, cu_kPa: null, phi_deg: 30, cprime_kPa: 0, E_MPa: 20, Cc: null, e0: null, N_spt: null, notes: 'Generic parameters — update from test data' };
}

// ── Semantic knowledge model: AI analysis of classified dataset ───────────────
export async function generateSemanticModel(geoUnits, classifiedBH, siteContext, apiKey, demoMode) {
  if (demoMode || !apiKey) return _demoSemanticModel(geoUnits);

  const unitSummary = geoUnits.filter(u => u.code !== 'UNKN')
    .map(u => `${u.code} (${u.name}${u.description ? ': ' + u.description : ''})`).join('\n');

  // Summarise per-unit statistics
  const statsByCode = {};
  geoUnits.forEach(u => { statsByCode[u.code] = { depths: [], thicks: [] }; });
  for (const bh of classifiedBH) {
    for (const l of bh.layers) {
      if (statsByCode[l.unitCode]) {
        statsByCode[l.unitCode].depths.push((l.top + l.base) / 2);
        statsByCode[l.unitCode].thicks.push(l.base - l.top);
      }
    }
  }
  const mean = arr => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : null;
  const statsSummary = Object.entries(statsByCode).map(([code, s]) => {
    const md = mean(s.depths), mt = mean(s.thicks);
    return `${code}: mean depth ${md?.toFixed(1) ?? '?'}m bgl, mean thickness ${mt?.toFixed(1) ?? '?'}m (n=${s.depths.length} layers)`;
  }).join('\n');

  // Observed transition pairs
  const transObs = {};
  for (const bh of classifiedBH) {
    const layers = [...bh.layers].sort((a, b) => a.top - b.top);
    for (let i = 0; i < layers.length - 1; i++) {
      const key = `${layers[i].unitCode}→${layers[i+1].unitCode}`;
      transObs[key] = (transObs[key] ?? 0) + 1;
    }
  }
  const topTransitions = Object.entries(transObs).sort((a,b) => b[1]-a[1]).slice(0, 10)
    .map(([k, v]) => `${k}: ${v}×`).join(', ');

  const messages = [{
    role: 'user',
    content: `You are a senior engineering geologist. Analyse this UK ground investigation dataset and produce a semantic knowledge model to guide 3D geological interpolation.

GEOLOGICAL UNITS:
${unitSummary}

BOREHOLE STATISTICS (${classifiedBH.length} boreholes):
${statsSummary}

OBSERVED LAYER TRANSITIONS (above→below):
${topTransitions || 'none yet'}

SITE CONTEXT:
${siteContext || '(not provided)'}

Produce a semantic geological knowledge model. Use your geological expertise to fill in gaps beyond what the data alone shows. Respond with JSON ONLY:
{
  "unit_depth_profiles": {
    "UNIT_CODE": {
      "typical_top_depth_m": number,
      "typical_base_depth_m": number,
      "typical_thickness_m": number,
      "max_realistic_depth_m": number,
      "depth_confidence": 0.0-1.0,
      "geological_notes": "one sentence"
    }
  },
  "transition_priors": {
    "FROM→TO": 0.0-1.0
  },
  "lateral_continuity": {
    "UNIT_CODE": "high|medium|low"
  },
  "characteristic_keywords": {
    "UNIT_CODE": ["keyword1","keyword2","keyword3"]
  },
  "depth_exclusions": [
    {"unit_code": "CODE", "exclude_above_m": number, "exclude_below_m": number, "confidence": 0.0-1.0}
  ],
  "conceptual_descriptions": [
    {
      "statement": "1-sentence geometric/morphological concept (orientation, shape, continuity, depth trend)",
      "unit_codes": ["CODE"],
      "confidence": 0.0-1.0
    }
  ],
  "synthetic_anchors": [
    {
      "label": "inferred point label",
      "x_frac": 0.0-1.0,
      "y_frac": 0.0-1.0,
      "layers": [{"top": 0, "base": 2, "unit_code": "MG", "certainty": 0.5}],
      "rationale": "why inferred"
    }
  ],
  "model_narrative": "2-3 sentence geological summary"
}

Rules:
- conceptual_descriptions: 2-5 objects, each a geometry/morphology concept that will be encoded as a 32-dim neural field embedding to shape 3D model geometry. Focus on: orientation (E-W channel), shape (concave-up trough, flat terrace), depth trends (deepens north), structural controls (fault-stepped rockhead). Unit_codes = which units this applies to (empty = all). Confidence 0.5-0.95.`
  }];

  try {
    return await callClaude(messages, apiKey);
  } catch (err) {
    throw new Error(`Semantic model generation failed: ${err.message}`);
  }
}

function _demoSemanticModel(geoUnits) {
  const codes = geoUnits.filter(u => u.code !== 'UNKN').map(u => u.code);
  const profiles = {};
  codes.forEach((code, i) => {
    profiles[code] = {
      typical_top_depth_m: i * 2.5,
      typical_base_depth_m: (i + 1) * 2.5 + 1,
      typical_thickness_m: 2.5 + i * 0.5,
      max_realistic_depth_m: (i + 2) * 5,
      depth_confidence: 0.65,
      geological_notes: `Typical UK ${code} unit characteristics.`,
    };
  });
  return {
    unit_depth_profiles: profiles,
    transition_priors: {},
    lateral_continuity: Object.fromEntries(codes.map(c => [c, 'medium'])),
    characteristic_keywords: {},
    depth_exclusions: [],
    conceptual_descriptions: [
      { statement: 'Stratigraphy is broadly horizontal and laterally continuous across the site.', unit_codes: [], confidence: 0.7 },
      { statement: 'Shallow superficial deposits show lateral thinning away from river channel axes.', unit_codes: codes.slice(0, 1), confidence: 0.65 },
    ],
    synthetic_anchors: [],
    model_narrative: 'Demo semantic model — provide an API key for site-specific geological intelligence.',
  };
}

// ── Oracle Refinement: reason about uncertain voxel clusters ─────────────────
// clusters: output of findUncertainClusters()
// nearbyBH: boreholes near each cluster (caller filters)
// Returns array of { voxelIdxs, distribution: { code: prob } }
export async function oracleRefinement(clusters, geoUnits, apiKey, demoMode) {
  if (!clusters.length) return [];
  if (demoMode || !apiKey) return _demoOracleResults(clusters, geoUnits);

  const unitList = geoUnits.filter(u => u.code !== 'UNKN')
    .map(u => `${u.code}: ${u.name} — ${u.description ?? ''}`).join('\n');

  // Send up to 8 largest clusters to Claude in one call to minimise tokens
  const clusterSummaries = clusters.slice(0, 8).map((cl, i) => ({
    id: i,
    world_x: Math.round(cl.worldPos.x),
    world_y: Math.round(cl.worldPos.z),
    world_z: Math.round(cl.worldPos.y),
    voxel_count: cl.voxels.length,
    mean_entropy: parseFloat(cl.entropy.toFixed(3)),
  }));

  const messages = [{
    role: 'user',
    content: `You are reviewing a 3D geological model that has regions of high uncertainty — places where the neural network cannot determine the geological unit confidently.

GEOLOGICAL UNITS ON SITE:
${unitList}

UNCERTAIN CLUSTERS (world coordinates in metres, Z = elevation):
${JSON.stringify(clusterSummaries, null, 2)}

For each cluster, provide a probability distribution over the geological units based on your understanding of typical stratigraphy, the cluster's 3D position, and geological plausibility (e.g. made ground is unlikely at great depth; palaeochannels occur at valley base elevations).

Respond ONLY with JSON:
{
  "oracle_patches": [
    {
      "cluster_id": 0,
      "distribution": { "UNIT_CODE": 0.0, ... },
      "reasoning": "one sentence"
    }
  ]
}`,
  }];

  try {
    const result = await callClaude(messages, apiKey);
    return (result.oracle_patches ?? []).map(patch => ({
      voxelIdxs: clusters[patch.cluster_id]?.voxels ?? [],
      distribution: patch.distribution ?? {},
      reasoning: patch.reasoning ?? '',
    }));
  } catch {
    return _demoOracleResults(clusters, geoUnits);
  }
}

function _demoOracleResults(clusters, geoUnits) {
  const codes = geoUnits.filter(u => u.code !== 'UNKN').map(u => u.code);
  if (!codes.length) return [];
  return clusters.slice(0, 8).map(cl => {
    const dist = {};
    // Simple heuristic: deepest clusters lean toward last unit, shallowest toward first
    const depthBias = cl.centroid.iz / 20; // normalised
    codes.forEach((code, i) => {
      dist[code] = Math.max(0.05, (i / codes.length < depthBias ? 0.5 : 0.2));
    });
    const sum = Object.values(dist).reduce((a, b) => a + b, 0);
    for (const k of Object.keys(dist)) dist[k] /= sum;
    return { voxelIdxs: cl.voxels, distribution: dist };
  });
}

// ── Generate AI geotechnical narrative for report ─────────────────────────────
// Returns { narrative, key_findings, geotechnical_risks, recommendations } or null
export async function generateReportNarrative(geoUnits, classifiedBH, voxelGrid, siteContext, apiKey, demoMode, conceptStore = null) {
  if (demoMode || !apiKey) return _demoNarrative(geoUnits, classifiedBH);

  const { nx, ny, nz, cellSize: cs, cellHeight: ch, unitIds, certainty } = voxelGrid;
  const counts = {}, certSums = {};
  geoUnits.forEach(u => { counts[u.id] = 0; certSums[u.id] = 0; });
  for (let i = 0; i < unitIds.length; i++) {
    const uid = unitIds[i];
    if (uid && counts[uid] !== undefined) { counts[uid]++; certSums[uid] += certainty[i]; }
  }
  const total    = Object.values(counts).reduce((a, b) => a + b, 0) || 1;
  const cellVol  = cs * cs * ch;
  const bhCount  = classifiedBH.filter(b => !b.synthetic).length;
  const maxDepth = Math.max(...classifiedBH.filter(b=>!b.synthetic)
    .map(b => b.layers.length ? Math.max(...b.layers.map(l=>l.base)) : 0), 0);

  const unitSummary = geoUnits.map(u => {
    const n    = counts[u.id] ?? 0;
    const vol  = Math.round(n * cellVol);
    const pct  = (n / total * 100).toFixed(1);
    const cert = n > 0 ? ((certSums[u.id] / n) * 100).toFixed(0) : '0';
    const p    = u.params ?? {};
    return `${u.code} (${u.name}): ${pct}% of model volume (${vol.toLocaleString()} m³), avg certainty ${cert}%, Cu=${p.cu ?? '—'}kPa, SPT_N=${p.N_spt ?? '—'}, φ=${p.phi ?? '—'}°`;
  }).join('\n');

  // Include semantic conceptual model if active
  const conceptSummary = conceptStore && !conceptStore.isEmpty
    ? '\n\nSEMANTIC CONCEPTUAL MODEL (encoded as 32-dim geometry embeddings):\n' +
      conceptStore.concepts.map(c => {
        const topAxes = Array.from(c.embedding)
          .map((v, i) => ({ name: CONCEPT_AXES[i], v }))
          .filter(a => Math.abs(a.v) > 0.4)
          .sort((a, b) => Math.abs(b.v) - Math.abs(a.v))
          .slice(0, 5)
          .map(a => `${a.name}=${a.v >= 0 ? '+' : ''}${a.v.toFixed(2)}`).join(', ');
        return `- "${c.description}" (conf=${(c.confidence * 100).toFixed(0)}%, domain=${c.domain?.type ?? 'global'}): ${topAxes}`;
      }).join('\n')
    : '';

  const siteInfo = [
    `Boreholes: ${bhCount}`,
    `Max depth: ${maxDepth.toFixed(1)} m`,
    `Model grid: ${nx}×${ny}×${nz} voxels at ${cs}m horizontal × ${ch.toFixed(2)}m vertical`,
    `Site context: ${siteContext || 'Not provided'}`,
  ].join('\n');

  const messages = [{
    role: 'user',
    content: `You are a senior geotechnical engineer. Write a professional geotechnical interpretation for this ground investigation. Be concise but technically rigorous.

SITE INFORMATION:
${siteInfo}

GEOLOGICAL UNITS AND MODEL STATISTICS:
${unitSummary}${conceptSummary}

${conceptSummary ? 'The conceptual model embeddings show the geological interpretation framework used to shape the 3D model geometry. Reference these where relevant in the narrative.' : ''}

Respond ONLY with JSON:
{
  "narrative": "2-3 paragraph professional geotechnical description of the ground conditions",
  "key_findings": ["bullet 1", "bullet 2", ...],
  "geotechnical_risks": ["risk 1 with severity", ...],
  "recommendations": ["recommendation 1", ...]
}`,
  }];

  try {
    const result = await callClaude(messages, apiKey);
    if (!result?.narrative) throw new Error('No narrative in response');
    return result;
  } catch {
    return _demoNarrative(geoUnits, classifiedBH);
  }
}

function _demoNarrative(geoUnits, classifiedBH) {
  const codes = geoUnits.filter(u => u.code !== 'UNKN').map(u => u.name).join(', ');
  const bhCount = classifiedBH.filter(b => !b.synthetic).length;
  return {
    narrative: `The ground investigation comprised ${bhCount} boreholes revealing a sequence of ${codes}. The geological model was constructed using spatial interpolation of classified borehole logs. [Provide an API key for a site-specific AI-generated geotechnical narrative.]`,
    key_findings: [
      'Ground conditions characterised from borehole logs',
      'Model built using spatial interpolation',
      'Unit certainty varies with borehole density',
    ],
    geotechnical_risks: [
      'Variable ground conditions — review cross-sections at structure locations',
      'Uncertainty increases between boreholes',
    ],
    recommendations: [
      'Verify model against as-built records',
      'Consider additional investigation in zones of low certainty',
    ],
  };
}

// ── Parse geological feature descriptions into shape primitives ───────────────
// Returns array of shape objects compatible with geo-shapes.js.
export async function parseGeologicalFeatures(featureText, geoUnits, bbox, apiKey, demoMode) {
  if (!featureText?.trim()) return [];

  if (demoMode || !apiKey) {
    return _demoShapes(featureText, geoUnits, bbox);
  }

  const unitList = geoUnits.map(u => `${u.code} (${u.name ?? ''})`).join(', ');
  const { minX = 0, maxX = 100, minY = 0, maxY = 100 } = bbox ?? {};

  const system = `You are a geotechnical expert. Parse geological feature descriptions into structured shape primitives for a 3D ground model. Respond ONLY with valid JSON — an array of shape objects, no markdown.

Available unit codes: ${unitList}
Site bounds: E ${minX.toFixed(0)}–${maxX.toFixed(0)} m, N ${minY.toFixed(0)}–${maxY.toFixed(0)} m

Each shape object must have:
{
  "feature_type": "palaeochannel"|"lens"|"buried_hill"|"fold"|"pinch_out",
  "unit_code": "CODE" or null,
  "confidence": 0.0–1.0,
  "centroid_x_frac": 0–1,
  "centroid_y_frac": 0–1,
  "orientation_deg": 0–360,
  // palaeochannel: width_m, max_depth_m, length_m
  // lens: rx_m, ry_m, rz_m
  // buried_hill: amplitude_m, half_width_m
  "description": "brief"
}`;

  const resp = await fetch(API_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': apiKey,
      'anthropic-version': '2023-06-01',
    },
    body: JSON.stringify({
      model: MODEL,
      max_tokens: 1024,
      system,
      messages: [{ role: 'user', content: featureText }],
    }),
  });

  if (!resp.ok) {
    const e = await resp.json().catch(() => ({}));
    throw new Error(e?.error?.message || `API error ${resp.status}`);
  }

  const data = await resp.json();
  const text = (data.content?.[0]?.text ?? '').replace(/^```[a-z]*\n?/i,'').replace(/\n?```$/i,'').trim();
  try {
    const parsed = JSON.parse(text);
    return Array.isArray(parsed) ? parsed : [parsed];
  } catch {
    throw new Error(`Shape parse returned non-JSON: ${text.slice(0,120)}`);
  }
}

function _demoShapes(text, geoUnits, bbox) {
  const lower = text.toLowerCase();
  const shapes = [];
  const { minX=0, maxX=100, minY=0, maxY=100 } = bbox ?? {};
  const firstUnit = geoUnits.find(u => u.code !== 'UNKN');

  if (/palaeochannel|channel|trough/.test(lower)) {
    shapes.push({
      feature_type: 'palaeochannel',
      unit_code: firstUnit?.code ?? null,
      confidence: 0.6,
      centroid_x_frac: 0.5,
      centroid_y_frac: 0.67,
      orientation_deg: 90,
      width_m: 40,
      max_depth_m: 6,
      length_m: (maxX - minX) * 0.6,
      description: 'Demo palaeochannel (east–west)',
    });
  }
  if (/lens|pod|lenticle/.test(lower)) {
    shapes.push({
      feature_type: 'lens',
      unit_code: firstUnit?.code ?? null,
      confidence: 0.55,
      centroid_x_frac: 0.4,
      centroid_y_frac: 0.5,
      orientation_deg: 45,
      rx_m: 25, ry_m: 15, rz_m: 4,
      description: 'Demo sand lens',
    });
  }
  if (/hill|dome|mound|ridge/.test(lower)) {
    shapes.push({
      feature_type: 'buried_hill',
      unit_code: geoUnits.find(u => /chalk|rock|bedrock/i.test(u.name ?? ''))?.code ?? firstUnit?.code,
      confidence: 0.7,
      centroid_x_frac: 0.5,
      centroid_y_frac: 0.5,
      amplitude_m: 8,
      half_width_m: 35,
      description: 'Demo buried bedrock hill',
    });
  }
  if (!shapes.length) {
    shapes.push({
      feature_type: 'lens',
      unit_code: firstUnit?.code ?? null,
      confidence: 0.45,
      centroid_x_frac: 0.5,
      centroid_y_frac: 0.5,
      rx_m: 20, ry_m: 20, rz_m: 5,
      description: 'Demo geological body (generic)',
    });
  }
  return shapes;
}

// ── Default units (fallback) ───────────────────────────────────────────────────
function defaultUnits() {
  return [
    { id: 1, code: 'MG',  name: 'Made Ground',           color: '#8B6914', description: 'Variable fill material' },
    { id: 2, code: 'RTD', name: 'River Terrace Deposits', color: '#D4A843', description: 'Gravel with sand' },
    { id: 3, code: 'AC',  name: 'Alluvial Clay',          color: '#4A7C59', description: 'Soft to firm silty clay' },
    { id: 4, code: 'CH',  name: 'Chalk',                  color: '#EDE8D8', description: 'Soft to medium hard chalk' },
  ];
}

// ── Encode a geological concept as a 32-dim geometry embedding ────────────────
// Claude rates the concept on 32 geological geometry axes (−1 to +1).
// Demo mode uses keyword heuristics when no API key is provided.
// Returns Float32Array(32).
export async function encodeGeologicalConcept(description, apiKey, demoMode) {
  if (demoMode || !apiKey) {
    return _demoConceptEmbedding(description);
  }

  const AXIS_HINTS = [
    'flat horizontal beds (+) vs structureless/massive (−)',
    'inclined/dipping beds present (+)',
    'dip magnitude: 0=flat, +1=near-vertical',
    'body elongated E-W (+) vs compressed E-W (−)',
    'body elongated N-S (+) vs compressed N-S (−)',
    'concave-up trough / channel geometry (+)',
    'convex-up dome / anticline (+)',
    'contact is a fault surface (+)',
    'erosional/unconformable base (+) vs gradational (−)',
    'laterally continuous (+) vs discontinuous/lenticular (−)',
    'wedges/thins eastward (+)',
    'wedges/thins westward (+)',
    'wedges/thins northward (+)',
    'wedges/thins southward (+)',
    'surface deepens / dips toward east (+)',
    'surface deepens / dips toward west (+)',
    'surface deepens / dips toward north (+)',
    'surface deepens / dips toward south (+)',
    'stepped/piecewise boundary (+)',
    'irregular karstic/dissolution base (+)',
    'multi-storey nested channels (+)',
    'coarsening-upward sequence (+)',
    'fining-upward sequence (+)',
    'coarse gravel lag at base (+)',
    'dissolution/karst voids present (+)',
    'structurally complex / deformed (+)',
    'data confidence / certainty of this concept (0=low, +1=high)',
    'horizontally elongated in any direction (+)',
    'strong vertical anisotropy / layer-parallel fabric (+)',
    'deep incision relative to body width (+)',
    'geometry controlled by overburden load (+)',
    'complexity increases in one direction (+)',
  ];
  const axisLines = CONCEPT_AXES.map((a, i) => `[${i}] ${a}: ${AXIS_HINTS[i]}`).join('\n');

  const resp = await fetch(API_URL, {
    method: 'POST',
    headers: {
      'Content-Type':      'application/json',
      'x-api-key':         apiKey,
      'anthropic-version': '2023-06-01',
    },
    body: JSON.stringify({
      model: MODEL,
      max_tokens: 350,
      system: 'You are an expert structural geologist and sedimentologist encoding geological concepts as geometric embeddings. Return ONLY a JSON array of 32 numbers — no explanation, no markdown, no prose.',
      messages: [{
        role: 'user',
        content: `Rate the following geological concept on each of the 32 morphological geometry axes.
Use values: −1.0 = strongly absent or opposite sense, 0.0 = neutral/not applicable, +1.0 = strongly present/dominant.
Be generous with non-zero values when the concept implies a clear geometric tendency.

Concept: "${description}"

Axes (index: name: geometric meaning):
${axisLines}

Respond with ONLY a JSON array of exactly 32 numbers, e.g.: [0.8, -0.2, 0.5, ...]`,
      }],
    }),
  });

  if (!resp.ok) {
    const e = await resp.json().catch(() => ({}));
    throw new Error(e?.error?.message || `API error ${resp.status}`);
  }

  const data = await resp.json();
  const text = (data.content?.[0]?.text ?? '')
    .replace(/^```[a-z]*\n?/i, '').replace(/\n?```$/i, '').trim();

  let arr;
  try { arr = JSON.parse(text); } catch {
    throw new Error(`Concept encoding returned non-JSON: ${text.slice(0, 100)}`);
  }
  if (!Array.isArray(arr) || arr.length < 32) {
    throw new Error(`Expected 32 values, got ${arr?.length ?? 0}`);
  }

  const emb = new Float32Array(32);
  for (let i = 0; i < 32; i++) emb[i] = Math.max(-1, Math.min(1, +arr[i] || 0));
  return emb;
}

function _demoConceptEmbedding(description) {
  const d   = description.toLowerCase();
  const emb = new Float32Array(32);

  // Helper: accumulate (don't overwrite) so multiple keywords combine
  const acc = (idx, val) => { emb[idx] = Math.max(-1, Math.min(1, emb[idx] + val)); };

  // ── Morphological type ────────────────────────────────────────────────────
  if (/palaeochannel|paleochannel|buried\s+channel|incised\s+channel/.test(d)) {
    acc(0, -0.8); acc(5, 1.0); acc(8, 0.9); acc(19, 0.6);
    acc(22, 0.5); acc(23, 0.8); acc(27, 0.9); acc(29, 0.8);
  }
  if (/trough|valley|incision/.test(d)) { acc(5, 0.7); acc(29, 0.7); acc(8, 0.7); }
  if (/river\s+terrace|terrace\s+deposit|rtd/.test(d)) {
    acc(0, 0.7); acc(8, 0.6); acc(9, 0.8); acc(22, 0.4); acc(23, 0.7);
  }
  if (/floodplain|alluvial\s+plain|overbank/.test(d)) {
    acc(0, 0.7); acc(9, 0.7); acc(22, 0.5); acc(0, 0.1);
  }
  if (/alluvial|alluvium|alluviated/.test(d))  { acc(0, 0.5); acc(9, 0.6); acc(22, 0.3); }
  if (/glacial|glaciofluvial|glaciogenic/.test(d)) {
    acc(23, 0.6); acc(19, 0.4); acc(25, 0.3);
  }
  if (/esker/.test(d)) { acc(3, 0.5); acc(4, 0.5); acc(27, 0.8); acc(5, 0.4); }
  if (/drumlin/.test(d)) { acc(6, 0.5); acc(27, 0.7); acc(9, 0.5); }
  if (/moraine/.test(d)) { acc(19, 0.6); acc(25, 0.5); acc(27, 0.6); }
  if (/dome|anticline|diapir|mound/.test(d)) { acc(6, 0.9); acc(9, 0.6); acc(0, -0.5); }
  if (/lens|lenticle|pod|blob/.test(d)) {
    acc(9, -0.5); acc(10, 0.6); acc(11, 0.6); acc(12, 0.6); acc(13, 0.6);
  }
  if (/nested\s+channel|multistory|multi.?storey/.test(d)) { acc(5, 0.8); acc(20, 0.9); }
  if (/karst|dissolution|sinkhole|doline/.test(d)) { acc(19, 0.8); acc(24, 1.0); acc(25, 0.5); }

  // ── Structural controls ───────────────────────────────────────────────────
  if (/fault(?!less)/.test(d))   { acc(7, 1.0); acc(18, 0.8); acc(25, 0.7); acc(28, 0.6); }
  if (/step|stepped|offset/.test(d)) { acc(18, 0.9); acc(7, 0.3); }
  if (/rockhead|bedrock|top\s+of\s+rock|chalk\s+surface|limestone\s+surface/.test(d)) {
    acc(0, 0.2); acc(18, 0.4); acc(19, 0.5); acc(8, 0.6);
  }
  if (/dip|inclined|dipping/.test(d))  { acc(1, 0.8); acc(2, 0.7); }
  if (/steep|near.?vertical|vertical/.test(d)) { acc(2, 0.9); acc(28, 0.7); }
  if (/gentle|shallow\s+dip/.test(d))  { acc(2, 0.3); acc(0, 0.4); }
  if (/fold|synclinal|anticlinal/.test(d)) { acc(6, 0.7); acc(2, 0.5); }
  if (/horizontal|flat.?lying|tabular|layer/.test(d)) { acc(0, 0.9); acc(9, 0.8); }

  // ── Directional orientation (elongation) ──────────────────────────────────
  if (/east.?west|e.?w\b|ew\b|along\s+strike\s+e/i.test(d)) {
    acc(3, 0.9); acc(27, 0.8);
    if (emb[4] >= 0) acc(4, -0.6); // suppress N-S
  }
  if (/north.?south|n.?s\b|ns\b/i.test(d)) {
    acc(4, 0.9); acc(27, 0.8);
    if (emb[3] >= 0) acc(3, -0.6); // suppress E-W
  }
  if (/northeast|ne.?sw|ne\b/i.test(d))  { acc(3, 0.6); acc(4, 0.6); acc(27, 0.7); }
  if (/northwest|nw.?se|nw\b/i.test(d))  { acc(3, 0.6); acc(4, 0.6); acc(27, 0.7); }

  // ── Directional deepening ─────────────────────────────────────────────────
  if (/deepen.*(east|right)|tilts?\s+east/.test(d)) { acc(14, 0.8); acc(1, 0.4); acc(2, 0.4); }
  if (/deepen.*(west|left)|tilts?\s+west/.test(d))  { acc(15, 0.8); acc(1, 0.4); acc(2, 0.4); }
  if (/deepen.*(north)|tilts?\s+north/.test(d))     { acc(16, 0.8); acc(1, 0.4); acc(2, 0.4); }
  if (/deepen.*(south)|tilts?\s+south/.test(d))     { acc(17, 0.8); acc(1, 0.4); acc(2, 0.4); }

  // ── Lateral continuity & thinning ────────────────────────────────────────
  if (/continuous|persistent|widespread|extensive/.test(d)) { acc(9, 0.8); acc(27, 0.6); }
  if (/thin|pinch|wedge/.test(d)) {
    if (/east/.test(d))  acc(10, 0.8);
    if (/west/.test(d))  acc(11, 0.8);
    if (/north/.test(d)) acc(12, 0.8);
    if (/south/.test(d)) acc(13, 0.8);
    if (!/east|west|north|south/.test(d)) { acc(10, 0.5); acc(11, 0.5); }
  }

  // ── Grain size / sequence ────────────────────────────────────────────────
  if (/coarsen.*(up|upward|wards)|upward\s+coarsen/.test(d)) { acc(21, 0.9); }
  if (/fining.*(up|upward)|upward\s+fin/.test(d))             { acc(22, 0.9); }
  if (/gravel.*base|basal\s+gravel|lag\s+gravel|coarse\s+lag/.test(d)) { acc(23, 1.0); }
  if (/gravel|cobble|pebble|graveliferous/.test(d)) { acc(23, 0.5); }

  // ── Contact character ────────────────────────────────────────────────────
  if (/erosional|unconformity|erosive|scoured/.test(d)) { acc(8, 0.9); }
  if (/gradational|transitional|gradual/.test(d))        { acc(8, -0.4); }
  if (/irregular|wavy|undulating/.test(d))               { acc(19, 0.7); }
  if (/complex|variable|heterogeneous/.test(d))          { acc(25, 0.7); acc(31, 0.5); }

  // ── Sedimentary body geometry ─────────────────────────────────────────────
  if (/lacustrine|lake\s+deposit|lacustral/.test(d))  { acc(0, 0.8); acc(9, 0.8); acc(0, 0.1); }
  if (/fluvial|river\s+deposit|braided/.test(d))      { acc(5, 0.4); acc(27, 0.6); acc(22, 0.3); }
  if (/aeolian|dune|wind.?blown/.test(d))             { acc(1, 0.5); acc(2, 0.4); acc(27, 0.7); }
  if (/delta|deltaic|prograding/.test(d))             { acc(1, 0.5); acc(8, 0.4); acc(9, 0.5); acc(27, 0.6); }
  if (/sand\s+body|sandstone\s+body/.test(d))         { acc(27, 0.5); acc(9, 0.3); }
  if (/bedded|stratified|laminated/.test(d))          { acc(0, 0.7); acc(9, 0.6); acc(28, 0.5); }
  if (/massive|structureless|unbedded/.test(d))       { acc(0, -0.7); acc(28, -0.4); }
  if (/thick|wide|extensive\s+body/.test(d))          { acc(9, 0.5); acc(27, 0.4); }
  if (/thin\s+bed|lamina|varve/.test(d))              { acc(0, 0.8); acc(28, 0.7); }

  // ── Overburden and load controls ─────────────────────────────────────────
  if (/overburden|load|preconsolid|ice.?load|surcharge/.test(d)) { acc(30, 0.8); }
  if (/confined|loaded|buried\s+deep/.test(d))                   { acc(30, 0.5); }
  if (/near.?surface|shallow|outcrop/.test(d))                   { acc(30, -0.3); }

  // ── Complexity gradient ───────────────────────────────────────────────────
  if (/complexity\s+increas|more\s+complex|highly\s+variable/.test(d)) { acc(31, 0.8); }
  if (/increases\s+(?:towards?|toward)\s+(east|west|north|south)/.test(d)) { acc(31, 0.7); }
  if (/uniform|consistent|homogeneous/.test(d))                           { acc(31, -0.5); }

  // ── Vertical sequence ─────────────────────────────────────────────────────
  if (/upward.?fining|fining.?up/.test(d))           { acc(22, 0.9); }
  if (/upward.?coarsen|coarsen.?up/.test(d))         { acc(21, 0.9); }
  if (/graded\s+bed|normal\s+grading/.test(d))       { acc(22, 0.7); }
  if (/reverse\s+grad/.test(d))                      { acc(21, 0.7); }
  if (/basal\s+sand|sand\s+base/.test(d))            { acc(8, 0.6); acc(23, 0.3); }

  // ── Confidence qualifiers ─────────────────────────────────────────────────
  if (/certain|confident|definite|clear/.test(d))    { acc(26, 0.4); }
  if (/uncertain|possible|probable|inferred/.test(d)) { acc(26, -0.3); }
  if (/very\s+likely|high\s+confidence/.test(d))     { acc(26, 0.3); }
  if (/low\s+confidence|speculative/.test(d))         { acc(26, -0.5); }

  // Data confidence default
  if (emb[26] === 0) acc(26, 0.6);

  // Clamp all axes to [-1, +1]
  for (let i = 0; i < 32; i++) emb[i] = Math.max(-1, Math.min(1, emb[i]));
  return emb;
}

// ── Automated concept suggestion from borehole data ────────────────────────────
// Analyses the observed borehole patterns and suggests relevant geological
// concepts that should be encoded in the ConceptStore. Returns an array of
// { description, axes, confidence, reason } objects.
export async function suggestConceptsFromBoreholes(classifiedBH, geoUnits, apiKey, demoMode) {
  if (demoMode || !apiKey) return _demoConceptSuggestions(classifiedBH, geoUnits);

  const bhCount = classifiedBH.filter(b => !b.synthetic).length;
  if (bhCount < 2) return [];

  // Compute spatial statistics from borehole data
  const unitByCode = {};
  geoUnits.forEach(u => { unitByCode[u.code] = u; });

  // Per-unit: mean top and base elevation, spatial extent
  const unitStats = {};
  for (const bh of classifiedBH.filter(b => !b.synthetic)) {
    for (const layer of (bh.layers ?? [])) {
      if (!unitStats[layer.unitCode]) {
        unitStats[layer.unitCode] = { tops: [], bases: [], xs: [], ys: [] };
      }
      const st = unitStats[layer.unitCode];
      st.tops.push(bh.groundLevel - layer.top);
      st.bases.push(bh.groundLevel - layer.base);
      st.xs.push(bh.x);
      st.ys.push(bh.y);
    }
  }

  const unitStatsStr = Object.entries(unitStats).map(([code, st]) => {
    const meanTop  = st.tops.reduce((a, b) => a + b, 0) / st.tops.length;
    const meanBase = st.bases.reduce((a, b) => a + b, 0) / st.bases.length;
    const spanX    = Math.max(...st.xs) - Math.min(...st.xs);
    const spanY    = Math.max(...st.ys) - Math.min(...st.ys);
    const thick    = meanTop - meanBase;
    const unit     = unitByCode[code];
    return `${code} (${unit?.name ?? code}): top=${meanTop.toFixed(1)}m, base=${meanBase.toFixed(1)}m, thickness=${thick.toFixed(1)}m, E-W span=${spanX.toFixed(0)}m, N-S span=${spanY.toFixed(0)}m, n=${st.tops.length} observations`;
  }).join('\n');

  const bhPositions = classifiedBH.filter(b => !b.synthetic)
    .map(b => `${b.id}: (${b.x.toFixed(0)}, ${b.y.toFixed(0)}) GL=${b.groundLevel?.toFixed(1)}m`)
    .join('\n');

  const messages = [{
    role: 'user',
    content: `You are an expert geological modeller. Based on the borehole data statistics below, suggest 3-5 geological concepts that should be encoded in the semantic conceptual model to improve 3D model geometry.

BOREHOLE POSITIONS:
${bhPositions}

UNIT STATISTICS (from ${bhCount} boreholes):
${unitStatsStr}

For each suggested concept, respond with JSON:
[
  {
    "description": "1-sentence concept description (e.g. 'Palaeochannel trending E-W incised into chalk')",
    "reason": "why this concept is suggested by the data pattern",
    "confidence": 0.0-1.0,
    "unit_codes": ["CODE1"] or []
  },
  ...
]

Focus on: directional elongation (E-W or N-S span differences), channel-like geometry (thick in some BHs, absent in others), depth trends (unit gets deeper in one direction), structural controls (abrupt thickness changes suggesting faults), karst (irregular bases). Return ONLY the JSON array.`,
  }];

  try {
    const result = await callClaude(messages, apiKey);
    if (!Array.isArray(result)) return _demoConceptSuggestions(classifiedBH, geoUnits);
    return result.filter(s => s.description && s.confidence);
  } catch {
    return _demoConceptSuggestions(classifiedBH, geoUnits);
  }
}

function _demoConceptSuggestions(classifiedBH, geoUnits) {
  const bhCount = classifiedBH.filter(b => !b.synthetic).length;
  if (bhCount < 2) return [];
  // Analyse lateral unit presence to suggest elongation concepts
  const unitBHPresence = {};
  for (const bh of classifiedBH.filter(b => !b.synthetic)) {
    for (const layer of (bh.layers ?? [])) {
      if (!unitBHPresence[layer.unitCode]) unitBHPresence[layer.unitCode] = { xs: [], ys: [] };
      unitBHPresence[layer.unitCode].xs.push(bh.x);
      unitBHPresence[layer.unitCode].ys.push(bh.y);
    }
  }
  const suggestions = [];
  for (const [code, data] of Object.entries(unitBHPresence)) {
    if (data.xs.length < 2) continue;
    const spanX = Math.max(...data.xs) - Math.min(...data.xs);
    const spanY = Math.max(...data.ys) - Math.min(...data.ys);
    const unit  = geoUnits.find(u => u.code === code);
    if (!unit) continue;
    if (spanX > spanY * 1.5 && spanX > 30) {
      suggestions.push({
        description: `${unit.name} appears to trend E-W — elongated along the east-west axis`,
        reason: `${code} observed across ${spanX.toFixed(0)}m E-W vs ${spanY.toFixed(0)}m N-S`,
        confidence: Math.min(0.85, 0.5 + (spanX / spanY - 1) * 0.2),
        unit_codes: [code],
      });
    } else if (spanY > spanX * 1.5 && spanY > 30) {
      suggestions.push({
        description: `${unit.name} appears to trend N-S — elongated along the north-south axis`,
        reason: `${code} observed across ${spanY.toFixed(0)}m N-S vs ${spanX.toFixed(0)}m E-W`,
        confidence: Math.min(0.85, 0.5 + (spanY / spanX - 1) * 0.2),
        unit_codes: [code],
      });
    }
  }
  if (!suggestions.length) {
    suggestions.push({
      description: 'Lateral continuity — units appear laterally persistent across the site',
      reason: 'Units observed in multiple boreholes with similar depths',
      confidence: 0.65,
      unit_codes: [],
    });
  }
  return suggestions;
}
