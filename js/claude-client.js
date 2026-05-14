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

// ── Default units (fallback) ───────────────────────────────────────────────────
function defaultUnits() {
  return [
    { id: 1, code: 'MG',  name: 'Made Ground',           color: '#8B6914', description: 'Variable fill material' },
    { id: 2, code: 'RTD', name: 'River Terrace Deposits', color: '#D4A843', description: 'Gravel with sand' },
    { id: 3, code: 'AC',  name: 'Alluvial Clay',          color: '#4A7C59', description: 'Soft to firm silty clay' },
    { id: 4, code: 'CH',  name: 'Chalk',                  color: '#EDE8D8', description: 'Soft to medium hard chalk' },
  ];
}
