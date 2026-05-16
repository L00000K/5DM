import { analysisLog, log, AppState } from './app.js';

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

// ── Default units (fallback) ───────────────────────────────────────────────────
function defaultUnits() {
  return [
    { id: 1, code: 'MG',  name: 'Made Ground',           color: '#8B6914', description: 'Variable fill material' },
    { id: 2, code: 'RTD', name: 'River Terrace Deposits', color: '#D4A843', description: 'Gravel with sand' },
    { id: 3, code: 'AC',  name: 'Alluvial Clay',          color: '#4A7C59', description: 'Soft to firm silty clay' },
    { id: 4, code: 'CH',  name: 'Chalk',                  color: '#EDE8D8', description: 'Soft to medium hard chalk' },
  ];
}
