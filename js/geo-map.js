export function parseGeoMap(csvText, geoUnits, realBoreholes) {
  const lines = csvText.split(/\r?\n/).filter(l => l.trim());

  // Build lookup: unitCode (lower) → canonical code
  const codeMap = {};
  for (const u of geoUnits) {
    codeMap[u.code.toLowerCase()] = u.code;
    const firstWord = u.name?.split(/\s+/)[0]?.toLowerCase();
    if (firstWord) codeMap[firstWord] = u.code;
  }

  const boreholes = [];
  let skipped = 0;

  for (const line of lines) {
    const sep  = line.includes('\t') ? /\t/ : /,/;
    const cols = line.split(sep).map(c => c.trim());
    if (cols.length < 3) { skipped++; continue; }

    const xRaw = parseFloat(cols[0]);
    const yRaw = parseFloat(cols[1]);
    if (isNaN(xRaw) || isNaN(yRaw)) { skipped++; continue; }

    const rawCode    = cols[2].trim();
    const matched    = codeMap[rawCode.toLowerCase()];
    if (!matched) { skipped++; continue; }

    const groundLevel = idwGroundLevel(xRaw, yRaw, realBoreholes);
    const i = boreholes.length;

    boreholes.push({
      id: `MAP-${i}`,
      x: xRaw,
      y: yRaw,
      groundLevel,
      depth: 0.5,
      layers: [{
        top: 0,
        base: 0.5,
        unitCode: matched,
        certainty: 0.88,
        description: 'Geological map observation',
      }],
      classified: true,
      synthetic: true,
    });
  }

  return { boreholes, count: boreholes.length, skipped };
}

function idwGroundLevel(x, y, realBoreholes) {
  if (!realBoreholes?.length) return 0;

  let sumW = 0, sumWV = 0;
  for (const bh of realBoreholes) {
    const d2 = (bh.x - x) ** 2 + (bh.y - y) ** 2;
    if (d2 < 1e-10) return bh.groundLevel ?? 0;
    const w = 1 / d2;
    sumW  += w;
    sumWV += w * (bh.groundLevel ?? 0);
  }
  return sumW > 0 ? sumWV / sumW : 0;
}
