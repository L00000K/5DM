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

// ── Generic Claude request that returns raw response (no JSON parse) ──────────
async function _claudeRequest(messages, apiKey, model = MODEL, maxTokens = MAX_TOKENS) {
  const resp = await fetch(API_URL, {
    method: 'POST',
    headers: {
      'Content-Type':      'application/json',
      'x-api-key':         apiKey,
      'anthropic-version': '2023-06-01',
    },
    body: JSON.stringify({ model, max_tokens: maxTokens, messages }),
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    throw new Error(err?.error?.message || `API error ${resp.status}`);
  }
  return resp.json();
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
// options.siteContext: { units: [{code, name}], description: string }
export async function encodeGeologicalConcept(description, apiKey, demoMode, options = {}) {
  if (demoMode || !apiKey) {
    return _demoConceptEmbedding(description);
  }
  const { siteContext } = options;

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

  const withRationale = options.withRationale ?? false;

  const resp = await fetch(API_URL, {
    method: 'POST',
    headers: {
      'Content-Type':      'application/json',
      'x-api-key':         apiKey,
      'anthropic-version': '2023-06-01',
    },
    body: JSON.stringify({
      model: MODEL,
      max_tokens: withRationale ? 600 : 350,
      system: 'You are an expert structural geologist and sedimentologist encoding geological concepts as geometric embeddings.',
      messages: [{
        role: 'user',
        content: `Rate the following geological concept on each of the 32 morphological geometry axes.
Use values: −1.0 = strongly absent or opposite sense, 0.0 = neutral/not applicable, +1.0 = strongly present/dominant.
Be generous with non-zero values when the concept implies a clear geometric tendency.
${siteContext?.units?.length ? `\nSite geological units (for context): ${siteContext.units.map(u => `${u.code} (${u.name})`).join(', ')}.` : ''}
${siteContext?.description ? `\nSite description: ${siteContext.description.slice(0, 300)}` : ''}
Concept: "${description}"

Axes (index: name: geometric meaning):
${axisLines}

${withRationale
  ? `Respond with a JSON object: {"embedding": [32 numbers], "rationale": "1–2 sentence plain-language explanation of the key geometric axes chosen and why"}\nExample: {"embedding": [0.8, -0.2, ...], "rationale": "This concept implies E-W elongation (axis 3=+0.9) and a concave-up channel morphology (axis 5=+1.0) because..."}`
  : `Respond with ONLY a JSON array of exactly 32 numbers, e.g.: [0.8, -0.2, 0.5, ...]`
}`,
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

  let arr, rationale = null;
  if (withRationale) {
    let parsed;
    try { parsed = JSON.parse(text); } catch {
      // Fallback: try extracting array from within the text
      const m = text.match(/\[([^\]]+)\]/);
      if (m) parsed = { embedding: JSON.parse(`[${m[1]}]`) };
      else throw new Error(`Concept encoding returned non-JSON: ${text.slice(0, 100)}`);
    }
    arr       = parsed.embedding ?? parsed;
    rationale = parsed.rationale ?? null;
  } else {
    try { arr = JSON.parse(text); } catch {
      throw new Error(`Concept encoding returned non-JSON: ${text.slice(0, 100)}`);
    }
  }

  if (!Array.isArray(arr) || arr.length < 32) {
    throw new Error(`Expected 32 values, got ${arr?.length ?? 0}`);
  }

  const emb = new Float32Array(32);
  for (let i = 0; i < 32; i++) emb[i] = Math.max(-1, Math.min(1, +arr[i] || 0));
  if (withRationale) return { embedding: emb, rationale };
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

// ── Auto-correlate: merge BH descriptions → recommended unit codes ────────────
// Returns [{description, currentCode, recommendedCode, reason, confidence}]
export async function autoCorrelateUnits(classifiedBH, geoUnits, apiKey, demoMode) {
  const bhs = classifiedBH.filter(b => !b.synthetic);
  if (!bhs.length || !geoUnits.length) return [];

  // Collect unique descriptions (clipped to first 80 chars for context)
  const descMap = new Map(); // description → currentCode
  bhs.forEach(bh => {
    (bh.layers ?? []).forEach(l => {
      const d = (l.description ?? '').trim().toLowerCase().slice(0, 80);
      if (d && !descMap.has(d)) descMap.set(d, l.unitCode ?? '?');
    });
  });

  if (!apiKey || demoMode) {
    // Keyword-based demo heuristic
    const KEYWORDS = [
      { re: /clay|silty clay|firm clay|soft clay|stiff clay/i, code: geoUnits.find(u => /clay/i.test(u.name))?.code },
      { re: /sand|loose sand|medium sand|fine sand/i,          code: geoUnits.find(u => /sand/i.test(u.name))?.code },
      { re: /gravel|pebbles|cobbles|granular/i,                code: geoUnits.find(u => /gravel/i.test(u.name))?.code },
      { re: /chalk|limestone|flint|calcareous/i,               code: geoUnits.find(u => /chalk|limestone/i.test(u.name))?.code },
      { re: /made ground|fill|topsoil|made-ground/i,           code: geoUnits.find(u => /fill|made|topsoil/i.test(u.name))?.code },
      { re: /mudstone|siltstone|shale/i,                       code: geoUnits.find(u => /mudstone|silt|shale/i.test(u.name))?.code },
    ].filter(k => k.code);

    return Array.from(descMap.entries()).slice(0, 30).map(([desc, current]) => {
      const match = KEYWORDS.find(k => k.re.test(desc));
      if (!match || match.code === current) return null;
      return {
        description: desc,
        currentCode: current,
        recommendedCode: match.code,
        reason: `Description matches ${match.re.source} pattern`,
        confidence: 0.6,
      };
    }).filter(Boolean);
  }

  const unitList = geoUnits.map(u => `${u.code}: ${u.name} — ${u.description ?? ''}`).join('\n');
  const descList = Array.from(descMap.entries()).slice(0, 60)
    .map(([d, c]) => `  "${d}" → currently coded: ${c}`)
    .join('\n');

  const prompt = `You are a geotechnical engineer checking unit classifications.

AVAILABLE GEOLOGICAL UNITS:
${unitList}

BOREHOLE LAYER DESCRIPTIONS (with current unit code):
${descList}

For each description, decide if the current unit code is correct or if it should be remapped.
Return a JSON array of corrections ONLY (omit correctly-classified items):
[{"description":"...","currentCode":"...","recommendedCode":"...","reason":"...","confidence":0.0–1.0}]
Return ONLY the JSON array, nothing else.`;

  try {
    const resp = await _claudeRequest([{ role: 'user', content: prompt }], apiKey, 'claude-haiku-4-5-20251001', 1024);
    const text = resp.content?.[0]?.text ?? '';
    const arr  = JSON.parse(text.match(/\[[\s\S]*\]/)?.[0] ?? '[]');
    return Array.isArray(arr) ? arr.filter(x => x.recommendedCode && x.currentCode !== x.recommendedCode) : [];
  } catch (e) {
    console.warn('autoCorrelateUnits error:', e.message);
    return [];
  }
}

// ── Extract geological concepts from free-form text ──────────────────────────
// Parses a site investigation report, field notes, or any geological text and
// returns an array of encodable concept descriptions.
//
// Returns [{description, confidence, unitAffinity: string[]}]
export async function extractConceptsFromText(text, geoUnits, apiKey, demoMode) {
  if (!text?.trim()) return [];
  if (demoMode || !apiKey) return _demoExtractConcepts(text, geoUnits);

  const unitList = geoUnits.map(u => `${u.code}: ${u.name ?? u.code}`).join('\n');
  const axisNames = CONCEPT_AXES.join(', ');

  const prompt = `You are an expert geotechnical interpreter. Extract all distinct geological conceptual statements from the text below that would inform the 3D geometry of subsurface units — things like depositional environment, structural controls, directional trends, erosional surfaces, and unit morphology.

AVAILABLE UNIT CODES:
${unitList}

TEXT TO ANALYSE:
${text.slice(0, 4000)}

For each concept you identify, produce one entry. A "concept" here means a single statement about the 3D geometry, shape, or spatial character of one or more units (not a factual observation like "BH01 encountered clay at 2m").

Examples of valid concepts:
- "Alluvial gravel forms a laterally continuous sheet dipping gently to the north-east"
- "Chalk rockhead is irregular and stepped, controlled by E-W joint sets"
- "River terrace deposits thin westward and pinch out near the valley margin"
- "Palaeochannel incised into Till — trending approximately north-south"

Return ONLY a JSON array (no markdown, no prose):
[
  {
    "description": "concise 1-sentence concept statement (≤120 chars)",
    "confidence": 0.0-1.0,
    "unit_codes": ["CODE"] or []
  }
]

Rules:
- confidence: 0.9 if stated as fact/certain, 0.7 if inferred, 0.5 if speculative/possible
- unit_codes: only codes from the list above; empty array if multiple or unknown
- Omit pure factual depth observations — only geometry/morphology concepts
- 3–8 concepts maximum; omit duplicates or near-duplicates
- Return ONLY the JSON array`;

  try {
    const resp = await _claudeRequest([{ role: 'user', content: prompt }], apiKey, 'claude-haiku-4-5-20251001', 600);
    const raw  = resp.content?.[0]?.text ?? '';
    const arr  = JSON.parse(raw.match(/\[[\s\S]*\]/)?.[0] ?? '[]');
    if (!Array.isArray(arr)) return [];
    return arr
      .filter(c => c?.description?.trim())
      .map(c => ({
        description:  c.description.trim(),
        confidence:   Math.max(0.1, Math.min(1.0, parseFloat(c.confidence) || 0.7)),
        unitAffinity: Array.isArray(c.unit_codes) ? c.unit_codes.filter(Boolean) : [],
      }))
      .slice(0, 10);
  } catch (e) {
    console.warn('extractConceptsFromText error:', e.message);
    return _demoExtractConcepts(text, geoUnits);
  }
}

function _demoExtractConcepts(text, geoUnits) {
  const lower = text.toLowerCase();
  const results = [];

  const PATTERNS = [
    { re: /palaeochannel|palaeo.?channel|buried.?channel|incised.?channel/,
      fn: () => 'Palaeochannel — incised erosional feature, concave-up base', conf: 0.8, axisHint: [5, 8, 29] },
    { re: /e.?w.*(channel|trend|orient)|east.*west.*(elongat|trend)|trending.*e.?w/,
      fn: () => 'E-W trending elongation — east-west directional anisotropy', conf: 0.75 },
    { re: /n.?s.*(channel|trend|orient)|north.*south.*(elongat|trend)|trending.*n.?s/,
      fn: () => 'N-S trending elongation — north-south directional anisotropy', conf: 0.75 },
    { re: /terrace|river.?terrace|fluvial.?terrace/,
      fn: () => 'River terrace deposits — laterally continuous, dipping gently toward valley', conf: 0.75 },
    { re: /fault|faulted|fault.?controlled|downthrow/,
      fn: () => 'Fault-controlled geometry — abrupt lateral unit changes, stepped boundaries', conf: 0.7 },
    { re: /stepped|step.?like|abrupt.?(change|boundary)|irregular.?rockhead/,
      fn: () => 'Stepped rockhead — irregular erosional surface with abrupt level changes', conf: 0.7 },
    { re: /chalk|limestone|karst|dissolution/,
      fn: () => 'Irregular dissolution features in bedrock — localised depressions and pinnacles', conf: 0.65 },
    { re: /dip.*north|northward.?dip|gentle.?dip/,
      fn: () => 'Gently dipping stratigraphy — units deepen toward north', conf: 0.65 },
    { re: /dip.*south|southward.?dip/,
      fn: () => 'Units dipping southward — deepening in south direction', conf: 0.65 },
    { re: /lateral.?continu|laterally.?persist|sheet.?deposit/,
      fn: () => 'Laterally continuous sheet deposits — high lateral continuity across site', conf: 0.7 },
    { re: /pinch.?out|thin.?westward|wedge.?out|thickens/,
      fn: () => 'Unit lateral pinch-out — thinning toward site margin', conf: 0.7 },
    { re: /gravel.?lag|basal.?gravel|coarsen.?down|lag.?deposit/,
      fn: () => 'Coarse basal lag gravels — gravel concentration at base of channel', conf: 0.75 },
    { re: /till|glacial|boulder.?clay|drumlin/,
      fn: () => 'Glacial till — poorly sorted, structurally complex, variable thickness', conf: 0.7 },
    { re: /alluvial|alluvium|floodplain|holocene/,
      fn: () => 'Recent alluvial deposits — sub-horizontal, variable thickness over irregular base', conf: 0.65 },
  ];

  for (const p of PATTERNS) {
    if (p.re.test(lower)) {
      const desc = p.fn();
      if (!results.some(r => r.description === desc)) {
        const unitMatch = geoUnits.find(u => lower.includes(u.name?.toLowerCase() ?? ''));
        results.push({
          description:  desc,
          confidence:   p.conf,
          unitAffinity: unitMatch ? [unitMatch.code] : [],
        });
      }
    }
    if (results.length >= 6) break;
  }

  if (!results.length) {
    results.push({
      description:  'Sub-horizontal stratified deposits — broadly layered geology with lateral continuity',
      confidence:   0.55,
      unitAffinity: [],
    });
  }
  return results;
}

// ── Concept Refinement Loop ───────────────────────────────────────────────────
// After building the neural implicit model, if concept-geometry match is poor,
// ask Claude to analyse the mismatch and suggest refined concept descriptions or
// updated embedding axis values. Returns an array of refinement suggestions.
//
// geoCheck: output of measureConceptGeometry()
// concepts: AppState.conceptStore.concepts
// demoMode: boolean — use heuristic feedback if no API key
//
// Returns [{
//   conceptId: string,         — id of concept to update (or null for new concept)
//   description: string,       — new or existing description
//   reason: string,            — why this change is suggested
//   adjustments: [{axis, delta}],  — axis index + recommended delta (−1..+1)
//   newEmbedding: number[]|null,   — full 32-dim replacement (only when API available)
// }]
export async function refineConceptsWithClaude(geoCheck, concepts, apiKey, demoMode) {
  // Demo mode: simple rule-based suggestions without API
  if (demoMode || !apiKey) {
    const suggestions = [];
    for (const r of geoCheck) {
      if (r.conceptMatch >= 0.9) continue;
      const ewExpected = r.predictedEW > r.predictedNS;
      const ewActual   = r.ewRatio > r.nsRatio;
      if (ewExpected && !ewActual) {
        suggestions.push({
          conceptId:   null,
          description: `${r.unitName} — elongated E-W with stronger lateral continuity`,
          reason:      `${r.unitCode}: concept predicted E-W elongation ×${r.predictedEW} but model only achieved ×${r.ewRatio}. Increasing east_west_elongation and lateral_continuity axes may help.`,
          adjustments: [{ axis: 3, delta: +0.3 }, { axis: 9, delta: +0.2 }],
          newEmbedding: null,
        });
      } else if (!ewExpected && ewActual) {
        suggestions.push({
          conceptId:   null,
          description: `${r.unitName} — elongated N-S`,
          reason:      `${r.unitCode}: concept predicted N-S elongation but model output is E-W elongated (×${r.ewRatio}). Add a concept emphasising N-S continuity.`,
          adjustments: [{ axis: 4, delta: +0.3 }, { axis: 3, delta: -0.2 }],
          newEmbedding: null,
        });
      } else {
        suggestions.push({
          conceptId:   null,
          description: `Add more borehole data or increase concept confidence for ${r.unitName}`,
          reason:      `${r.unitCode} geometry does not match concept prediction — may need more data or a stronger concept.`,
          adjustments: [{ axis: 26, delta: +0.2 }],
          newEmbedding: null,
        });
      }
    }
    return suggestions;
  }

  // Build context for Claude
  const conceptSummary = concepts.map(c => {
    const axes = CONCEPT_AXES.map((a, i) => ({ a, v: c.embedding[i] }))
      .filter(x => Math.abs(x.v) > 0.2)
      .sort((a, b) => Math.abs(b.v) - Math.abs(a.v))
      .slice(0, 6)
      .map(x => `${x.a}: ${x.v > 0 ? '+' : ''}${x.v.toFixed(2)}`);
    return `  [${c.id}] "${c.description}" (confidence ${c.confidence?.toFixed(2) ?? '?'})\n    Active axes: ${axes.join(', ')}`;
  }).join('\n');

  const geoSummary = geoCheck.map(r =>
    `  ${r.unitCode} (${r.unitName}): actual E-W ×${r.ewRatio} N-S ×${r.nsRatio} · concept predicted E-W ×${r.predictedEW} N-S ×${r.predictedNS} · ${r.count} voxels · match: ${r.conceptMatch >= 0.9 ? 'GOOD' : r.conceptMatch >= 0.5 ? 'PARTIAL' : 'POOR'}`
  ).join('\n');

  const axisLines = CONCEPT_AXES.map((a, i) => `[${i}] ${a}`).join(', ');

  const prompt = `You are a geological modelling expert. A neural implicit geological model has been built from borehole data conditioned by semantic concept embeddings. Analyse the mismatch between predicted and actual geometry and suggest concept refinements.

CURRENT CONCEPTS:
${conceptSummary}

GEOMETRY VERIFICATION (actual model output vs concept prediction):
${geoSummary}

AVAILABLE AXES (32-dim embedding, values −1 to +1):
${axisLines}

For each unit with PARTIAL or POOR match, suggest:
1. Which existing concept to adjust (by its id), OR suggest a new concept description
2. Which axes to modify and by how much (delta values −0.5 to +0.5)
3. A brief reason

Respond ONLY with a JSON array:
[{
  "conceptId": "c1" or null,
  "description": "refined or new concept text",
  "reason": "brief explanation of what is causing the mismatch and how this fixes it",
  "adjustments": [{"axis": 3, "delta": 0.3}, ...]
}]
No prose, no markdown, JSON only.`;

  try {
    const resp = await _claudeRequest([{ role: 'user', content: prompt }], apiKey, 'claude-haiku-4-5-20251001', 800);
    const text = resp.content?.[0]?.text ?? '';
    const arr  = JSON.parse(text.match(/\[[\s\S]*\]/)?.[0] ?? '[]');
    if (!Array.isArray(arr)) return [];

    // For each suggestion with axis adjustments, compute a full updated embedding
    // by applying deltas to the most relevant existing concept's embedding.
    return arr.map(s => {
      const base = s.conceptId ? concepts.find(c => c.id === s.conceptId) : null;
      if (base && s.adjustments?.length) {
        const emb = Array.from(base.embedding);
        for (const { axis, delta } of s.adjustments) {
          if (axis >= 0 && axis < 32) emb[axis] = Math.max(-1, Math.min(1, emb[axis] + (delta ?? 0)));
        }
        s.newEmbedding = emb;
      }
      return s;
    }).filter(s => s.reason || s.description);
  } catch (e) {
    console.warn('refineConceptsWithClaude error:', e.message);
    return [];
  }
}

// ── AI Borehole Gap Analysis ──────────────────────────────────────────────────
// Analyses the built 3D model and suggests optimal new borehole locations for
// maximum information gain. Considers concept influence, coverage density,
// model certainty, and concept-geometry match.
//
// grid:         voxelGrid (nx, ny, cellSize, origin, certainty, conceptInfluence,
//               coverageDensity, attributionGrid)
// classifiedBH: array of borehole objects with x, y, groundLevel
// geoUnits:     array of geological unit objects
// conceptStore: ConceptStore with .concepts
// geoCheck:     output of measureConceptGeometry (unit geometry match results)
// apiKey, demoMode
//
// Returns [{x, y, reason, priority: 'high'|'medium', score}]
export async function analyseBoreholeGaps(grid, classifiedBH, geoUnits, conceptStore, geoCheck, apiKey, demoMode) {
  if (!grid) return [];
  const { nx, ny, nz, cellSize: cs = 1, cellHeight: ch = 1, origin: O,
          certainty, conceptInfluence, coverageDensity } = grid;

  // ── Compute spatial statistics across the grid ────────────────────────────
  // Downsample to a coarse horizontal grid (target ~8×8 cells) for analysis
  const GRID_N = 8;
  const stepX = Math.max(1, Math.floor(nx / GRID_N));
  const stepY = Math.max(1, Math.floor(ny / GRID_N));
  const cells = [];

  for (let iy = 0; iy < ny; iy += stepY) {
    for (let ix = 0; ix < nx; ix += stepX) {
      // Average certainty + concept influence across the column
      let sumCert = 0, sumCI = 0, sumCov = 0, cnt = 0;
      for (let iz = 0; iz < nz; iz++) {
        const flat = ix + iy * nx + iz * nx * ny;
        if (certainty) sumCert += certainty[flat] ?? 0;
        if (conceptInfluence) sumCI   += conceptInfluence[flat] ?? 0;
        if (coverageDensity)  sumCov  += coverageDensity[flat] ?? 0;
        cnt++;
      }
      const meanCert = cnt ? sumCert / cnt : 0.5;
      const meanCI   = cnt ? sumCI / cnt   : 0;
      const meanCov  = cnt ? sumCov / cnt  : 0.5;
      const wx = O.x + (ix + 0.5) * cs;
      const wy = O.z + (iy + 0.5) * cs;
      // Score = high concept influence + low certainty + low coverage
      const score = (meanCI * 0.4 + (1 - meanCert) * 0.35 + (1 - meanCov) * 0.25);
      cells.push({ ix, iy, wx, wy, score, meanCert, meanCI, meanCov });
    }
  }

  // Pick top cells, ensuring minimum spatial separation
  cells.sort((a, b) => b.score - a.score);
  const MIN_SEP_M = Math.max(10, Math.min(cs * stepX * 2, 30));
  const topCells = [];
  for (const c of cells) {
    if (topCells.length >= 6) break;
    if (topCells.every(t => Math.hypot(t.wx - c.wx, t.wy - c.wy) > MIN_SEP_M)) {
      topCells.push(c);
    }
  }

  if (!topCells.length) return [];

  // ── Demo mode: return rule-based suggestions ──────────────────────────────
  if (demoMode || !apiKey) {
    return _demoBoreholeGaps(topCells, geoCheck, classifiedBH);
  }

  // ── Build context for Claude ──────────────────────────────────────────────
  const bhList = classifiedBH.filter(b => !b.synthetic)
    .map(b => `${b.id}: (${b.x.toFixed(0)}, ${b.y.toFixed(0)}) GL=${b.groundLevel?.toFixed(1) ?? '?'}m`)
    .join('\n');

  const conceptSummary = conceptStore?.concepts.length
    ? conceptStore.concepts.map(c =>
        `  "${c.description}" (conf=${(c.confidence ?? 0.7).toFixed(2)}, domain=${c.domain?.type ?? 'global'})`
      ).join('\n')
    : '  None';

  const geoCheckSummary = geoCheck?.length
    ? geoCheck.filter(r => r.conceptMatch < 0.9).map(r =>
        `  ${r.unitCode}: concept-geometry match ${(r.conceptMatch * 100).toFixed(0)}% (E-W actual ×${r.ewRatio} vs predicted ×${r.predictedEW})`
      ).join('\n')
    : '  All units match concepts well';

  const candidateStr = topCells.slice(0, 6).map((c, i) =>
    `  [${i+1}] x=${c.wx.toFixed(0)}m, y=${c.wy.toFixed(0)}m — certainty=${(c.meanCert*100).toFixed(0)}%, concept_influence=${(c.meanCI*100).toFixed(0)}%, bh_coverage=${(c.meanCov*100).toFixed(0)}%`
  ).join('\n');

  const unitList = geoUnits.map(u => `${u.code} (${u.name ?? u.code})`).join(', ');

  const prompt = `You are a geotechnical ground investigation planning expert. Based on the analysis below, recommend the best 3–5 new borehole locations to maximise information gain for the 3D geological model.

EXISTING BOREHOLES (${classifiedBH.filter(b => !b.synthetic).length} total):
${bhList}

GEOLOGICAL UNITS: ${unitList}

ACTIVE GEOLOGICAL CONCEPTS:
${conceptSummary}

CONCEPT-GEOMETRY MISMATCHES (units where model geometry doesn't match concept prediction):
${geoCheckSummary}

CANDIDATE NEW BOREHOLE LOCATIONS (ranked by info-gain score):
${candidateStr}

For each recommended borehole, explain:
1. Which candidate location [number] to use (or adjust slightly)
2. Why this location is most valuable (which concept, unit, or geometry it would resolve)
3. What depth to investigate to (justify from stratigraphy)
4. Priority: "high" if it resolves a concept mismatch or covers a data gap in a critical unit, "medium" otherwise

Respond ONLY with JSON array:
[{
  "location_idx": 1,
  "x": number,
  "y": number,
  "depth_m": number,
  "priority": "high"|"medium",
  "reason": "1-2 sentence explanation"
}]
No prose, no markdown, JSON only.`;

  try {
    const resp = await _claudeRequest([{ role: 'user', content: prompt }], apiKey, 'claude-haiku-4-5-20251001', 700);
    const text = resp.content?.[0]?.text ?? '';
    const arr  = JSON.parse(text.match(/\[[\s\S]*\]/)?.[0] ?? '[]');
    if (!Array.isArray(arr)) return _demoBoreholeGaps(topCells, geoCheck, classifiedBH);
    return arr
      .filter(s => s.reason && s.x != null && s.y != null)
      .map(s => ({
        x:       parseFloat(s.x),
        y:       parseFloat(s.y),
        depth_m: parseFloat(s.depth_m) || 15,
        priority: s.priority ?? 'medium',
        reason:  s.reason,
        score:   s.priority === 'high' ? 1 : 0.6,
      }))
      .slice(0, 6);
  } catch (e) {
    console.warn('analyseBoreholeGaps error:', e.message);
    return _demoBoreholeGaps(topCells, geoCheck, classifiedBH);
  }
}

function _demoBoreholeGaps(topCells, geoCheck, classifiedBH) {
  const hasMismatch = geoCheck?.some(r => r.conceptMatch < 0.5);
  return topCells.slice(0, 4).map((c, i) => ({
    x:       c.wx,
    y:       c.wy,
    depth_m: 15,
    priority: (i === 0 || (hasMismatch && i < 2)) ? 'high' : 'medium',
    reason:  i === 0
      ? `High concept influence (${(c.meanCI*100).toFixed(0)}%) with low model certainty (${(c.meanCert*100).toFixed(0)}%) — concept geometry predictions are unconstrained here.`
      : `Low borehole coverage (${(c.meanCov*100).toFixed(0)}%) in a concept-active zone — new data would reduce extrapolation uncertainty.`,
    score:   c.score,
  }));
}

// ── One-shot site concept setup from a site description paragraph ─────────────
// Takes a free-form site description and returns a structured setup object:
//   concepts: [{description, confidence, unitAffinity}]
//   events:   [{type, name, unitCodes}] — geological event timeline entries, oldest first
//   stratOrder: string[]  — unit codes oldest→youngest
//
// This allows the user to paste a brief geological description and have Claude
// configure the entire conceptual model automatically.
export async function setupConceptsFromSiteDescription(description, geoUnits, apiKey, demoMode) {
  if (!description?.trim()) return { concepts: [], events: [], stratOrder: [] };
  if (demoMode || !apiKey) return _demoSiteSetup(description, geoUnits);

  const unitList = geoUnits.map(u => `${u.code}: ${u.name ?? u.code}`).join('\n');
  const eventTypes = 'deposition | erosion | fault | intrusion | folding | karst | fill | terrace';

  const prompt = `You are an expert geotechnical modeller. Given the site description below, set up a complete geological conceptual model configuration.

AVAILABLE UNITS:
${unitList}

SITE DESCRIPTION:
"""
${description.slice(0, 3000)}
"""

Return ONLY a JSON object (no markdown, no prose):
{
  "concepts": [
    {
      "description": "concise 1-sentence geometric/morphological concept (≤120 chars)",
      "confidence": 0.0-1.0,
      "unit_codes": ["CODE"] or []
    }
  ],
  "events": [
    {
      "type": "deposition"|"erosion"|"fault"|"intrusion"|"folding"|"karst"|"fill"|"terrace",
      "name": "brief event name (≤30 chars)",
      "unit_codes": ["CODE"] or []
    }
  ],
  "strat_order": ["OLDEST_CODE", ..., "YOUNGEST_CODE"]
}

Rules:
- concepts: 3-6 distinct geometric/morphological concept statements. Focus on shape, orientation, directional trend, erosional surfaces, structural controls.
- events: list 2-6 geological events in STRATIGRAPHIC ORDER (oldest first = index 0).
  Use only the event types listed. Link each event to 0-2 unit codes.
- strat_order: list unit codes from oldest (deepest) to youngest (shallowest). Only include codes from the unit list.
- confidence: 0.9 if described as certain, 0.75 if inferred, 0.6 if speculative.
- unit_codes: only from the list above.`;

  try {
    const resp = await _claudeRequest([{ role: 'user', content: prompt }], apiKey, 'claude-haiku-4-5-20251001', 900);
    const text = resp.content?.[0]?.text ?? '';
    const obj  = JSON.parse(text.match(/\{[\s\S]*\}/)?.[0] ?? '{}');

    const concepts = (obj.concepts ?? [])
      .filter(c => c?.description?.trim())
      .map(c => ({
        description:  c.description.trim(),
        confidence:   Math.max(0.1, Math.min(1, parseFloat(c.confidence) || 0.7)),
        unitAffinity: Array.isArray(c.unit_codes) ? c.unit_codes.filter(Boolean) : [],
      }))
      .slice(0, 8);

    const events = (obj.events ?? [])
      .filter(e => e?.type)
      .map(e => ({
        type:      e.type,
        name:      e.name ?? '',
        unitCodes: Array.isArray(e.unit_codes) ? e.unit_codes.filter(Boolean) : [],
      }))
      .slice(0, 10);

    const stratOrder = Array.isArray(obj.strat_order)
      ? obj.strat_order.filter(c => geoUnits.some(u => u.code === c))
      : [];

    return { concepts, events, stratOrder };
  } catch (e) {
    console.warn('setupConceptsFromSiteDescription error:', e.message);
    return _demoSiteSetup(description, geoUnits);
  }
}

function _demoSiteSetup(text, geoUnits) {
  const lower = text.toLowerCase();
  const codes  = geoUnits.filter(u => u.code !== 'UNKN').map(u => u.code);
  const concepts = [];
  const events   = [];

  if (/channel|palaeochannel/i.test(lower)) {
    concepts.push({ description: 'Palaeochannel — incised erosional feature with concave-up base, E-W trend', confidence: 0.8, unitAffinity: [] });
    events.push({ type: 'erosion',     name: 'Channel incision', unitCodes: [] });
    events.push({ type: 'fill',        name: 'Channel fill',     unitCodes: codes.slice(0, 1) });
  } else if (/terrace|river.?terrace/i.test(lower)) {
    concepts.push({ description: 'River terrace deposits — laterally continuous, gently dipping toward valley', confidence: 0.8, unitAffinity: [] });
    events.push({ type: 'deposition', name: 'Terrace formation', unitCodes: codes.slice(0, 1) });
  } else if (/fault|faulted/i.test(lower)) {
    concepts.push({ description: 'Fault-controlled geometry — stepped boundaries and abrupt lateral unit changes', confidence: 0.75, unitAffinity: [] });
    events.push({ type: 'fault', name: 'Faulting event', unitCodes: [] });
  } else {
    concepts.push({ description: 'Sub-horizontal stratified sequence — broadly layered, laterally continuous', confidence: 0.65, unitAffinity: [] });
    events.push({ type: 'deposition', name: 'Primary deposition', unitCodes: codes.slice(0, 1) });
  }

  if (/chalk|limestone|karst/i.test(lower)) {
    concepts.push({ description: 'Irregular dissolution features in bedrock — localised hollows and pinnacles', confidence: 0.7, unitAffinity: [] });
    events.push({ type: 'karst', name: 'Dissolution', unitCodes: codes.slice(-1) });
  }
  if (/alluvial|alluvium|fill|made.?ground/i.test(lower)) {
    events.unshift({ type: 'deposition', name: 'Made ground / fill', unitCodes: [] });
  }

  // Build strat order: older codes last, younger first
  const stratOrder = [...codes].reverse();
  return { concepts, events, stratOrder };
}

// ── Unit semantic similarity analysis ─────────────────────────────────────────
// Compares all pairs of geological units and flags likely duplicates or near-
// duplicates using keyword cosine similarity (demo) or Claude (API mode).
// Returns { pairs: [{codeA, nameA, codeB, nameB, similarity, sharedTokens, suggestion}] }
export async function analyseUnitSimilarity(geoUnits, apiKey, demoMode) {
  if (geoUnits.length < 2) return { pairs: [] };
  if (!demoMode && apiKey) return _claudeUnitSimilarity(geoUnits, apiKey);
  return _heuristicUnitSimilarity(geoUnits);
}

function _heuristicUnitSimilarity(geoUnits) {
  // Weighted token list: [token, weight]
  const TOKENS = [
    ['clay', 2], ['sand', 2], ['gravel', 2], ['chalk', 2], ['limestone', 2],
    ['mudstone', 2], ['sandstone', 2], ['made ground', 2], ['made', 1.5],
    ['fill', 2], ['peat', 2], ['silt', 2], ['alluvium', 2], ['cobble', 2],
    ['boulder', 2], ['flint', 1.5], ['rock', 1.5], ['bedrock', 2],
    ['soft', 1], ['firm', 1], ['stiff', 1], ['hard', 1],
    ['loose', 1], ['medium dense', 1], ['dense', 1], ['very dense', 1],
    ['brown', 0.5], ['grey', 0.5], ['gray', 0.5], ['red', 0.5],
    ['orange', 0.5], ['yellow', 0.5], ['blue', 0.5], ['black', 0.5],
    ['fissured', 1], ['laminated', 1], ['weathered', 1], ['organic', 1],
    ['calcareous', 1], ['sandy', 1], ['silty', 1], ['gravelly', 1],
    ['plastic', 1], ['brittle', 1], ['fractured', 1],
  ];

  function vecFor(u) {
    const text = `${u.name ?? ''} ${u.code ?? ''}`.toLowerCase();
    return TOKENS.map(([t, w]) => (text.includes(t) ? w : 0));
  }

  function cosineSim(a, b) {
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] ** 2; nb += b[i] ** 2; }
    const denom = Math.sqrt(na * nb);
    return denom > 0 ? dot / denom : 0;
  }

  const vecs = geoUnits.map(u => ({ u, v: vecFor(u) }));
  const pairs = [];

  for (let i = 0; i < vecs.length; i++) {
    for (let j = i + 1; j < vecs.length; j++) {
      const sim = cosineSim(vecs[i].v, vecs[j].v);
      if (sim < 0.60) continue;
      const sharedTokens = TOKENS
        .filter((_tok, k) => vecs[i].v[k] > 0 && vecs[j].v[k] > 0)
        .map(([t]) => t);
      pairs.push({
        codeA: vecs[i].u.code, nameA: vecs[i].u.name ?? vecs[i].u.code,
        codeB: vecs[j].u.code, nameB: vecs[j].u.name ?? vecs[j].u.code,
        similarity: sim,
        sharedTokens,
        suggestion: sim > 0.88 ? 'Consider merging — nearly identical descriptions'
          : sim > 0.72 ? 'Possible subdivision of same material — verify in logs'
          : 'Similar material type — check depth / lateral boundaries',
      });
    }
  }

  pairs.sort((a, b) => b.similarity - a.similarity);
  return { pairs };
}

async function _claudeUnitSimilarity(geoUnits, apiKey) {
  const unitList = geoUnits.map(u => `${u.code}: ${u.name ?? u.code}`).join('\n');

  const prompt = `You are an expert geotechnical geologist reviewing geological unit definitions for a ground model.

UNITS:
${unitList}

Identify all pairs of units that appear to be duplicates, near-duplicates, or subdivisions of the same geological material. For each pair, estimate a similarity score (0.0–1.0).

Only include pairs with similarity > 0.60.

Return ONLY a JSON array (no markdown, no prose):
[
  {
    "code_a": "CODE",
    "code_b": "CODE",
    "similarity": 0.0–1.0,
    "reason": "brief explanation ≤80 chars",
    "suggestion": "Consider merging" | "Verify subdivision" | "Check boundary"
  }
]`;

  try {
    const resp = await _claudeRequest([{ role: 'user', content: prompt }], apiKey, 'claude-haiku-4-5-20251001', 512);
    const text  = resp.content?.[0]?.text ?? '';
    let raw = [];
    try { raw = JSON.parse(text.match(/\[[\s\S]*\]/)?.[0] ?? '[]'); } catch { raw = []; }
    const unitMap = new Map(geoUnits.map(u => [u.code, u]));

    const pairs = raw
      .filter(p => p?.code_a && p?.code_b && unitMap.has(p.code_a) && unitMap.has(p.code_b))
      .map(p => {
        const sim = parseFloat(p.similarity);
        return {
          codeA: p.code_a, nameA: unitMap.get(p.code_a)?.name ?? p.code_a,
          codeB: p.code_b, nameB: unitMap.get(p.code_b)?.name ?? p.code_b,
          similarity: isNaN(sim) ? 0 : Math.max(0, Math.min(1, sim)),
          sharedTokens: [],
          suggestion: p.suggestion ?? p.reason ?? '',
          reason: p.reason ?? '',
        };
      })
      .filter(p => p.similarity > 0)
      .sort((a, b) => b.similarity - a.similarity);

    return { pairs };
  } catch (e) {
    console.warn('analyseUnitSimilarity Claude error:', e.message);
    return _heuristicUnitSimilarity(geoUnits);
  }
}

// ── Concept–BH Coherence Scoring ─────────────────────────────────────────────
// Scores how well each active concept's embedding is supported by the borehole
// evidence. Purely heuristic — no API call needed. Returns per-concept
// { score: 0–1, details: string, suggestions: string[] }.
//
// Axes checked:
//   east_west_elongation (3)     → BH units should vary more N-S than E-W
//   north_south_elongation (4)   → BH units should vary more E-W than N-S
//   channel_morphology (5)       → mid-site BHs show deeper unit top than edge BHs
//   horizontal_layering (0)      → unit elevations are consistent across BHs
//   incision_depth_ratio (29)    → unit base shows concave profile
//   fault_controlled (7)         → unit occurrence shows abrupt lateral change
//   lateral_continuity (9)       → same unit appears in ≥ 60% of BHs
export function scoreConceptCoherence(concept, classifiedBH, geoUnits) {
  const emb  = concept.embedding;
  const bhOk = classifiedBH?.filter(b => !b.synthetic && b.layers?.length >= 1 && isFinite(b.x) && isFinite(b.y));
  if (!bhOk?.length || !emb) return null;

  // Build per-unit occurrence stats across BHs
  const unitTopZ  = {}; // unitCode → [topZ, ...]
  const unitBHPos = {}; // unitCode → [{x, y, topZ}]
  for (const bh of bhOk) {
    const gl = bh.groundLevel ?? 0;
    for (const layer of bh.layers) {
      if (!layer.unitCode) continue;
      const topZ = gl - (layer.top ?? 0);
      if (!unitTopZ[layer.unitCode])  unitTopZ[layer.unitCode]  = [];
      if (!unitBHPos[layer.unitCode]) unitBHPos[layer.unitCode] = [];
      unitTopZ[layer.unitCode].push(topZ);
      unitBHPos[layer.unitCode].push({ x: bh.x, y: bh.y, topZ });
    }
  }

  const scores = [];
  const details = [];
  const suggestions = [];

  // ── E-W vs N-S elongation check ──────────────────────────────────────────
  const ewElong = emb[3], nsElong = emb[4];
  if (Math.abs(ewElong) > 0.3 || Math.abs(nsElong) > 0.3) {
    // Compute mean E-W and N-S unit-top variance across BHs
    let ewVar = 0, nsVar = 0, nPairs = 0;
    for (const [, pts] of Object.entries(unitBHPos)) {
      for (let a = 0; a < pts.length; a++) {
        for (let b = a + 1; b < pts.length; b++) {
          const dx = Math.abs(pts[a].x - pts[b].x), dy = Math.abs(pts[a].y - pts[b].y);
          const dz = Math.abs(pts[a].topZ - pts[b].topZ);
          if (dx < 1 && dy < 1) continue;
          if (dx > dy * 2) { ewVar += dz / (dx + 1); nPairs++; }
          else if (dy > dx * 2) { nsVar += dz / (dy + 1); nPairs++; }
        }
      }
    }
    if (nPairs > 0) {
      ewVar /= nPairs; nsVar /= nPairs;
      // E-W elongation concept: expect ewVar < nsVar (flat E-W, steep N-S)
      if (ewElong > 0.3) {
        const ratio = nsVar > 0.001 ? ewVar / nsVar : 1;
        const coherence = Math.max(0, 1 - ratio); // 0 = inconsistent, 1 = consistent
        scores.push(coherence * ewElong);
        if (coherence > 0.5) details.push(`BH data shows E-W unit continuity (score: ${(coherence*100).toFixed(0)}%)`);
        else {
          details.push(`BH data shows weak E-W elongation support (E-W/N-S variance ratio: ${ratio.toFixed(2)})`);
          suggestions.push('Consider reducing east_west_elongation or adding more E-W boreholes');
        }
      }
      if (nsElong > 0.3) {
        const ratio = ewVar > 0.001 ? nsVar / ewVar : 1;
        const coherence = Math.max(0, 1 - ratio);
        scores.push(coherence * nsElong);
        if (coherence < 0.4) suggestions.push('Consider reducing north_south_elongation — BH data does not strongly support N-S continuity');
      }
    }
  }

  // ── Lateral continuity check ──────────────────────────────────────────────
  const latCont = emb[9];
  if (latCont > 0.2) {
    const unitCodes = Object.keys(unitBHPos);
    const totalBH   = bhOk.length;
    const bestCoverage = unitCodes.reduce((best, code) => {
      return Math.max(best, unitBHPos[code].length / totalBH);
    }, 0);
    const latScore = Math.min(1, bestCoverage / Math.max(0.3, latCont));
    scores.push(latScore);
    if (latScore > 0.7) details.push(`Dominant unit appears in ${(bestCoverage*100).toFixed(0)}% of BHs — supports lateral continuity`);
    else if (latCont > 0.6) suggestions.push(`Lateral continuity is high (${latCont.toFixed(2)}) but dominant unit only in ${(bestCoverage*100).toFixed(0)}% of BHs — consider reducing confidence`);
  }

  // ── Channel morphology check ──────────────────────────────────────────────
  const chanMorph = emb[5];
  if (chanMorph > 0.4) {
    // Channel: depth to unit base should be deeper at site centre vs edges
    const bhsSorted = [...bhOk].sort((a, b) => a.x - b.x);
    if (bhsSorted.length >= 3) {
      const midX = (bhsSorted[0].x + bhsSorted[bhsSorted.length - 1].x) / 2;
      let edgeDepth = 0, midDepth = 0, ne = 0, nm = 0;
      for (const bh of bhsSorted) {
        const baseZ = (bh.layers[0] ? (bh.groundLevel - bh.layers[bh.layers.length - 1].base) : null);
        if (baseZ == null) continue;
        if (Math.abs(bh.x - midX) > (bhsSorted[bhsSorted.length-1].x - bhsSorted[0].x) * 0.3) {
          edgeDepth += baseZ; ne++;
        } else { midDepth += baseZ; nm++; }
      }
      if (ne > 0 && nm > 0) {
        edgeDepth /= ne; midDepth /= nm;
        const channelScore = edgeDepth > midDepth ? Math.min(1, (edgeDepth - midDepth) / 2) : 0;
        scores.push(channelScore);
        if (channelScore < 0.2 && chanMorph > 0.6) {
          suggestions.push('Channel morphology concept applied but BH data shows no clear depth deepening at edges — verify channel orientation');
        } else if (channelScore > 0.3) {
          details.push(`BH base depths support channel geometry (edge avg ${edgeDepth.toFixed(1)}m, centre ${midDepth.toFixed(1)}m)`);
        }
      }
    }
  }

  // ── Horizontal layering check ─────────────────────────────────────────────
  const hLayer = emb[0];
  if (hLayer > 0.4) {
    // Units should appear at similar elevations in all BHs
    let zStdSum = 0, nUnits = 0;
    for (const [, zArr] of Object.entries(unitTopZ)) {
      if (zArr.length < 2) continue;
      const mean = zArr.reduce((s, z) => s + z, 0) / zArr.length;
      const std  = Math.sqrt(zArr.reduce((s, z) => s + (z - mean) ** 2, 0) / zArr.length);
      const range = Math.max(...zArr) - Math.min(...zArr);
      zStdSum += range > 0 ? std / range : 0;
      nUnits++;
    }
    if (nUnits > 0) {
      const meanCV = zStdSum / nUnits;
      const layerScore = Math.max(0, 1 - meanCV * 2);
      scores.push(layerScore * hLayer);
      if (layerScore < 0.4 && hLayer > 0.5) {
        suggestions.push('Horizontal layering concept applied but unit elevations vary significantly between BHs — consider reducing horizontal_layering or adding tilt/dip concept');
      }
    }
  }

  const overallScore = scores.length > 0 ? scores.reduce((s, v) => s + v, 0) / scores.length : 0.5;
  return {
    score: overallScore,
    grade: overallScore > 0.7 ? 'strong' : overallScore > 0.45 ? 'moderate' : 'weak',
    details: details.length ? details : ['Insufficient BH data for detailed coherence analysis'],
    suggestions,
  };
}

// ── Geological Laws Compiler ────────────────────────────────────────────────────
// Parses geological rule statements into structured concept+domain definitions.
// Each rule becomes one or more concept embeddings ready for ConceptStore.add().
//
// Rule types recognised:
//   superposition: "A always overlies B" → temporal ordering constraint
//   morphological:  "channels trend E-W, 5-20m wide" → embedding + optional bbox
//   facies:         "sand only below 5m AOD" → depth-domain concept
//   contact:        "chalk contact is irregular/dissolution" → unit-affinity concept
//   regional:       "till covers western half" → bbox-domain concept
//
// Returns [{description, embedding: Float32Array(32), domain, unitAffinity,
//           confidence, temporalOrder, ruleText, ruleType}]
export async function compileGeologicalRules(rulesText, geoUnits, bounds, apiKey, demoMode) {
  if (!rulesText?.trim()) return [];
  if (demoMode || !apiKey) return _demoCompileRules(rulesText, geoUnits, bounds);

  const unitList  = geoUnits.map(u => `${u.code} (${u.name})`).join(', ');
  const siteW     = bounds ? Math.round(bounds.maxX - bounds.minX) : 200;
  const siteH     = bounds ? Math.round(bounds.maxY - bounds.minY) : 200;
  const axisNames = CONCEPT_AXES.join(', ');

  const prompt = `You are a geological modelling expert compiling geological rule statements into semantic concept embeddings for a neural implicit geological model.

SITE UNITS: ${unitList || 'not specified'}
SITE SIZE: approximately ${siteW}m E-W × ${siteH}m N-S
AVAILABLE EMBEDDING AXES (index 0–31): [${axisNames}]

GEOLOGICAL RULES TO COMPILE:
"""
${rulesText}
"""

For each distinct geological rule, output one JSON object. If a rule implies multiple geometric aspects, split into separate concept objects. Output a JSON array:
[{
  "ruleText": "the original rule sentence",
  "ruleType": "superposition" | "morphological" | "facies" | "contact" | "regional",
  "description": "concise geological concept description (max 80 chars)",
  "embedding": [32 floats, −1 to +1, index matching the axes list above],
  "domain": one of:
    {"type":"global"} — applies everywhere
    {"type":"global","minZ":N,"maxZ":N,"sigmaZ":N} — depth-constrained (metres AOD)
    {"type":"bbox","minX":N,"maxX":N,"minY":N,"maxY":N,"sigma":N} — spatial sub-domain
  "unitAffinity": [] or [unit codes from site units that this rule specifically governs],
  "confidence": 0.5–0.95,
  "temporalOrder": null or integer (0=oldest; higher=younger; use only if the rule explicitly states age/time relationships)
}]
Rules with no clear spatial constraint → type global.
Superposition rules: set temporalOrder (older unit gets lower number).
Facies/depth rules: use minZ/maxZ domain.
Regional rules: estimate bbox fractions from direction words (e.g. "western half" → minX=site_minX, maxX=site_minX+siteW/2).
Respond with ONLY the JSON array. No prose, no markdown fences.`;

  try {
    const resp = await _claudeRequest([{ role: 'user', content: prompt }], apiKey, MODEL, 1600);
    const text = resp.content?.[0]?.text ?? '';
    const raw  = text.match(/\[[\s\S]*\]/)?.[0] ?? '[]';
    const arr  = JSON.parse(raw);
    if (!Array.isArray(arr)) return [];
    return arr.map(r => _normaliseCompiledRule(r));
  } catch (e) {
    console.warn('compileGeologicalRules error:', e.message);
    return _demoCompileRules(rulesText, geoUnits, bounds);
  }
}

function _normaliseCompiledRule(r) {
  const emb = new Float32Array(32);
  if (Array.isArray(r.embedding)) {
    for (let i = 0; i < 32; i++) emb[i] = Math.max(-1, Math.min(1, +r.embedding[i] || 0));
  }
  emb[26] = emb[26] || 0.7; // ensure data_confidence has a value
  return {
    ruleText:      r.ruleText      ?? '',
    ruleType:      r.ruleType      ?? 'morphological',
    description:   (r.description ?? '').slice(0, 120),
    embedding:     emb,
    domain:        r.domain        ?? { type: 'global' },
    unitAffinity:  Array.isArray(r.unitAffinity) ? r.unitAffinity : [],
    confidence:    Math.max(0.3, Math.min(1, +(r.confidence ?? 0.75))),
    temporalOrder: r.temporalOrder !== null && r.temporalOrder !== undefined
                     ? Number(r.temporalOrder) : null,
  };
}

function _demoCompileRules(rulesText, geoUnits, bounds) {
  // Pattern-based rule compilation when no API key available
  const lines  = rulesText.split(/\n+/).map(l => l.trim()).filter(Boolean);
  const rules  = [];
  const unitsByCode = {};
  geoUnits.forEach(u => { unitsByCode[u.code.toLowerCase()] = u.code; });
  const unitCodes = Object.values(unitsByCode);

  const siteW = bounds ? bounds.maxX - bounds.minX : 200;
  const siteH = bounds ? bounds.maxY - bounds.minY : 200;
  const ox    = bounds?.minX ?? 0;
  const oy    = bounds?.minY ?? 0;

  for (const line of lines) {
    const d = line.toLowerCase();
    const emb = new Float32Array(32);
    emb[26] = 0.7;
    let domain        = { type: 'global' };
    let unitAffinity  = [];
    let temporalOrder = null;
    let ruleType      = 'morphological';
    let description   = line.slice(0, 80);

    // Superposition / temporal rules
    const overMatch = d.match(/(\w+)\s+(always|typically)\s+overlies?\s+(\w+)/);
    const underMatch = d.match(/(\w+)\s+(always|typically)\s+underlies?\s+(\w+)/);
    if (overMatch || underMatch) {
      ruleType = 'superposition';
      const [, young, , old] = overMatch ?? [, underMatch[3], , underMatch[1]];
      temporalOrder = overMatch ? 1 : 0; // relative
      const matchedY = unitCodes.find(c => d.includes(c.toLowerCase()));
      if (matchedY) { unitAffinity = [matchedY]; temporalOrder = 1; }
      emb[0] = 0.5; emb[9] = 0.5; // horizontal continuity
      description = line.slice(0, 80);
    }

    // Channel/morphological rules
    if (/channel|palaeochannel/.test(d)) {
      ruleType = 'morphological';
      emb[5] = 1.0; emb[8] = 0.9; emb[29] = 0.7; emb[27] = 0.9; emb[0] = -0.7;
      if (/e.?w|east.?west/.test(d)) { emb[3] = 0.9; emb[4] = -0.5; }
      if (/n.?s|north.?south/.test(d)) { emb[4] = 0.9; emb[3] = -0.5; }
      if (/ne.?sw|northeast/.test(d)) { emb[3] = 0.65; emb[4] = 0.65; }
    }

    // Dissolution / karst
    if (/dissolution|karst|irregular/.test(d) && /chalk|limestone|rockhead/.test(d)) {
      ruleType = 'contact';
      emb[19] = 0.8; emb[24] = 0.9; emb[8] = 0.7; emb[25] = 0.5;
    }

    // Fault / stepped
    if (/fault|step|offset/.test(d)) {
      ruleType = 'contact';
      emb[7] = 1.0; emb[18] = 0.9; emb[25] = 0.7;
    }

    // Horizontal / bedded / continuous
    if (/horizontal|bedded|continuous|laterally\s+continuous/.test(d)) {
      emb[0] = 0.9; emb[9] = 0.9; emb[28] = 0.6;
    }

    // Depth rules
    const depthMatch = d.match(/(?:below|above|between|from)\s+([-\d.]+)\s*(?:m|metre)\s*(?:aod|od)?/);
    if (depthMatch) {
      ruleType = 'facies';
      const z = parseFloat(depthMatch[1]);
      if (/below/.test(d))  domain = { type: 'global', maxZ: z, sigmaZ: 2 };
      if (/above/.test(d))  domain = { type: 'global', minZ: z, sigmaZ: 2 };
    }

    // Regional rules
    if (/western|west\s+half|west\s+side/.test(d)) {
      ruleType = 'regional';
      domain = { type: 'bbox', minX: ox, maxX: ox + siteW * 0.5, minY: oy, maxY: oy + siteH, sigma: siteW * 0.15 };
    } else if (/eastern|east\s+half|east\s+side/.test(d)) {
      ruleType = 'regional';
      domain = { type: 'bbox', minX: ox + siteW * 0.5, maxX: ox + siteW, minY: oy, maxY: oy + siteH, sigma: siteW * 0.15 };
    } else if (/northern|north\s+half/.test(d)) {
      ruleType = 'regional';
      domain = { type: 'bbox', minX: ox, maxX: ox + siteW, minY: oy + siteH * 0.5, maxY: oy + siteH, sigma: siteH * 0.15 };
    } else if (/southern|south\s+half/.test(d)) {
      ruleType = 'regional';
      domain = { type: 'bbox', minX: ox, maxX: ox + siteW, minY: oy, maxY: oy + siteH * 0.5, sigma: siteH * 0.15 };
    }

    // Unit affinity detection from code mentions
    for (const code of unitCodes) {
      if (d.includes(code.toLowerCase()) && !unitAffinity.includes(code)) {
        unitAffinity.push(code);
      }
    }

    // Clamp
    for (let i = 0; i < 32; i++) emb[i] = Math.max(-1, Math.min(1, emb[i]));

    rules.push({ ruleText: line, ruleType, description, embedding: emb, domain, unitAffinity, confidence: 0.75, temporalOrder });
  }
  return rules;
}

// ── Optimal Drilling Recommendation ────────────────────────────────────────────
// Identifies the most informative borehole locations given the current model
// uncertainty map, active concept geometry predictions, and existing BH coverage.
//
// Algorithm:
//   1. Build a 2D uncertainty surface (max certainty across depth column, inverted)
//   2. Apply concept-geometry weighting: prefer high-uncertainty zones that concepts
//      predict are geometrically significant (channels, fault zones, pinch-outs)
//   3. Apply BH coverage penalty: avoid zones already well-sampled
//   4. Pick the N highest-scoring grid cells, enforce minimum spacing
//
// Returns [{x, y, score, reason, conceptContext}]
export function recommendDrillingLocations(grid, geoUnits, conceptStore, nLocations = 5) {
  if (!grid || !grid.certainty || !grid.nx) return [];
  const { nx, ny, nz, cellSize, cellHeight, origin, certainty, unitIds } = grid;

  // 2D uncertainty: per (ix, iy) — minimum certainty in the column
  const colUncert  = new Float32Array(nx * ny);
  const colUnitDiv = new Float32Array(nx * ny); // unit diversity (# distinct units in column)

  for (let ix = 0; ix < nx; ix++) {
    for (let iy = 0; iy < ny; iy++) {
      let minC = 1, prevUnit = -1, divCount = 0;
      for (let iz = 0; iz < nz; iz++) {
        const idx = ix + iy * nx + iz * nx * ny;
        minC = Math.min(minC, certainty[idx]);
        if (unitIds[idx] !== prevUnit) { divCount++; prevUnit = unitIds[idx]; }
      }
      const flat2 = ix + iy * nx;
      colUncert[flat2]  = 1 - minC;
      colUnitDiv[flat2] = Math.min(1, divCount / Math.max(1, nz * 0.3));
    }
  }

  // BH coverage penalty: voxels within one cell radius of a real BH get suppressed
  const bhPenalty = new Float32Array(nx * ny).fill(1.0);
  if (grid._boreholes?.length) {
    for (const bh of grid._boreholes) {
      if (bh.synthetic) continue;
      const bix = Math.round((bh.x - origin.x) / cellSize - 0.5);
      const biy = Math.round((bh.y - origin.z) / cellSize - 0.5);
      for (let dx = -2; dx <= 2; dx++) {
        for (let dy = -2; dy <= 2; dy++) {
          const px = bix + dx, py = biy + dy;
          if (px < 0 || px >= nx || py < 0 || py >= ny) continue;
          const dist = Math.hypot(dx, dy);
          bhPenalty[px + py * nx] *= Math.min(1, dist * 0.3);
        }
      }
    }
  }

  // Concept geometry score: favour regions where concepts predict geometric interest
  const conceptScore = new Float32Array(nx * ny).fill(0.5);
  if (conceptStore && !conceptStore.isEmpty) {
    for (let ix = 0; ix < nx; ix++) {
      for (let iy = 0; iy < ny; iy++) {
        const wx = origin.x + ix * cellSize + cellSize * 0.5;
        const wy = origin.z + iy * cellSize + cellSize * 0.5;
        const ctx = conceptStore.computeAt(wx, wy, 0);
        if (!ctx || ctx.totalWeight < 0.05) { conceptScore[ix + iy * nx] = 0.4; continue; }
        const v = ctx.vec;
        // High score for zones with channels, faults, karst, contact uncertainty
        const chanScore  = Math.max(0, v[5] ?? 0);     // channel_morphology
        const faultScore = Math.max(0, v[7] ?? 0);     // fault_controlled
        const karsScore  = Math.max(0, v[24] ?? 0);    // dissolution_features
        const compScore  = Math.max(0, v[25] ?? 0);    // structural_complexity
        const irregScore = Math.max(0, v[19] ?? 0);    // irregular_base
        conceptScore[ix + iy * nx] = 0.3 + 0.7 * Math.min(1, chanScore + faultScore * 0.8 + karsScore * 0.9 + compScore * 0.5 + irregScore * 0.4);
      }
    }
  }

  // Combined score: uncertainty × diversity × concept × coverage
  const scores = new Float32Array(nx * ny);
  for (let i = 0; i < nx * ny; i++) {
    scores[i] = colUncert[i] * 0.5 + colUnitDiv[i] * 0.2 + conceptScore[i] * 0.3;
    scores[i] *= bhPenalty[i];
  }

  // Pick top N with minimum spacing (4 cells = 4 × cellSize apart)
  const MIN_SPACING = 4;
  const picked = [];
  const used   = new Uint8Array(nx * ny);

  for (let iter = 0; iter < 200 && picked.length < nLocations; iter++) {
    let best = -1, bestScore = -1;
    for (let i = 0; i < nx * ny; i++) {
      if (!used[i] && scores[i] > bestScore) { bestScore = scores[i]; best = i; }
    }
    if (best < 0) break;

    const bix = best % nx, biy = Math.floor(best / nx);
    picked.push(best);
    // Suppress neighbours within MIN_SPACING
    for (let dx = -MIN_SPACING; dx <= MIN_SPACING; dx++) {
      for (let dy = -MIN_SPACING; dy <= MIN_SPACING; dy++) {
        const px = bix + dx, py = biy + dy;
        if (px >= 0 && px < nx && py >= 0 && py < ny) used[px + py * nx] = 1;
      }
    }
  }

  return picked.map(flat => {
    const ix = flat % nx, iy = Math.floor(flat / nx);
    const wx = origin.x + ix * cellSize + cellSize * 0.5;
    const wy = origin.z + iy * cellSize + cellSize * 0.5;
    const unc  = colUncert[flat];
    const div  = colUnitDiv[flat];
    const cs   = conceptScore[flat];

    // Gather concept context for reason text
    let reasonParts = [];
    if (unc > 0.5) reasonParts.push(`high model uncertainty (${(unc * 100).toFixed(0)}%)`);
    if (div > 0.4) reasonParts.push('multiple unit transitions in column');
    if (cs > 0.65) {
      if (conceptStore && !conceptStore.isEmpty) {
        const ctx = conceptStore.computeAt(wx, wy, 0);
        if (ctx?.vec) {
          const v = ctx.vec;
          if ((v[5] ?? 0) > 0.5) reasonParts.push('concept predicts channel zone');
          if ((v[7] ?? 0) > 0.5) reasonParts.push('concept predicts fault proximity');
          if ((v[24] ?? 0) > 0.5) reasonParts.push('concept predicts dissolution features');
          if ((v[25] ?? 0) > 0.5) reasonParts.push('complex structural context');
        }
      }
    }
    const reason = reasonParts.length ? reasonParts.join('; ') : 'uncertain zone with sparse BH coverage';

    return {
      x:     +wx.toFixed(1),
      y:     +wy.toFixed(1),
      score: +scores[flat].toFixed(3),
      uncert: +unc.toFixed(2),
      reason,
    };
  });
}
