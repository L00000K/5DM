// ── Project Configuration — export / import ────────────────────────────────────
// Serialises the full project state to a .geomodel JSON file and
// restores it. The voxel grid is not saved (recomputed on load).

export const CONFIG_VERSION = '1.1';

export function exportConfig(state) {
  const constraints = document.getElementById('constraints-text')?.value ?? '';
  const gwt         = parseFloat(document.getElementById('gwt-elevation')?.value ?? '') || null;
  const siteHistory = document.getElementById('input-site-history')?.value ?? '';
  const conceptsJson = state.conceptStore?.isEmpty === false
    ? JSON.parse(state.conceptStore.serialize())
    : null;

  // Collect unit descriptions from desc-list
  const unitDescs = Array.from(document.querySelectorAll('#desc-list .desc-item'))
    .map(el => el.querySelector('.desc-text')?.textContent?.trim() ?? el.textContent.trim())
    .filter(Boolean);

  return {
    version:  CONFIG_VERSION,
    meta: {
      project:     'GeoModel AI',
      created:     new Date().toISOString(),
      description: '',
    },
    geoUnits: state.geoUnits.map(u => ({
      id:          u.id,
      code:        u.code,
      name:        u.name,
      color:       u.color,
      description: u.description ?? '',
      params:      u.params ?? {},
    })),
    classifiedBH: state.classifiedBH.filter(b => !b.synthetic).map(bh => ({
      id:           bh.id,
      x:            bh.x,
      y:            bh.y,
      groundLevel:  bh.groundLevel,
      depth:        bh.depth,
      classified:   true,
      layers: bh.layers.map(l => ({
        top:         l.top,
        base:        l.base,
        unitCode:    l.unitCode,
        certainty:   l.certainty ?? 0.9,
        description: l.description ?? '',
        sptN:        l.sptN ?? null,
      })),
    })),
    settings: {
      cellSizeH:    state.cellSizeH,
      cellSizeZ:    state.cellSizeZ,
      kNeighbors:   state.kNeighbors,
      idwPower:     state.idwPower,
      interpMethod:  state.interpMethod,
      anisoAzimuth:  state.anisoAzimuth ?? 0,
      anisoRatio:    state.anisoRatio   ?? 1,
      trendOrder:    state.trendOrder   ?? 1,
    },
    constraints,
    gwtElevation:     gwt,
    siteHistory,
    unitDescriptions: unitDescs,
    concepts:         conceptsJson,
  };
}

export function importConfig(data) {
  if (!data?.version || !data?.geoUnits) {
    throw new Error('Invalid project config: missing version or geoUnits');
  }
  return {
    geoUnits:        data.geoUnits ?? [],
    classifiedBH:    data.classifiedBH ?? [],
    cellSizeH:       data.settings?.cellSizeH   ?? 1,
    cellSizeZ:       data.settings?.cellSizeZ   ?? 0.25,
    kNeighbors:      data.settings?.kNeighbors  ?? 5,
    idwPower:        data.settings?.idwPower    ?? 2,
    interpMethod:    data.settings?.interpMethod ?? 'idw',
    anisoAzimuth:    data.settings?.anisoAzimuth ?? 0,
    anisoRatio:      data.settings?.anisoRatio   ?? 1,
    trendOrder:      data.settings?.trendOrder   ?? 1,
    constraints:      data.constraints    ?? '',
    gwtElevation:     data.gwtElevation   ?? null,
    siteHistory:      data.siteHistory    ?? '',
    unitDescriptions: data.unitDescriptions ?? [],
    concepts:         data.concepts        ?? null,
  };
}
