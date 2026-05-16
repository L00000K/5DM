// ── Session save / restore ─────────────────────────────────────────────────────
// Saves geoUnits (with params), classifiedBH, grid parameters, and constraints
// to sessionStorage (not localStorage — never persist across browser restarts).
// The voxel grid itself is NOT saved (too large); caller must rebuild after load.

const SESSION_KEY = 'geomodel_session_v1';

export function saveSession(state) {
  const payload = {
    ts:              Date.now(),
    geoUnits:        state.geoUnits.map(u => ({ ...u })),
    classifiedBH:    state.classifiedBH.map(bh => ({
      id: bh.id, x: bh.x, y: bh.y,
      groundLevel: bh.groundLevel,
      depth: bh.depth,
      classified: bh.classified,
      synthetic: bh.synthetic,
      layers: bh.layers?.map(l => ({ ...l })) ?? [],
    })),
    cellSizeH:       state.cellSizeH,
    cellSizeZ:       state.cellSizeZ,
    kNeighbors:      state.kNeighbors,
    idwPower:        state.idwPower,
    interpMethod:    state.interpMethod,
    anisoAzimuth:    state.anisoAzimuth ?? 0,
    anisoRatio:      state.anisoRatio   ?? 1,
    trendOrder:      state.trendOrder   ?? 1,
    constraintsText: document.getElementById('constraints-text')?.value ?? '',
  };
  try {
    sessionStorage.setItem(SESSION_KEY, JSON.stringify(payload));
    return true;
  } catch (e) {
    return false;
  }
}

export function loadSession() {
  try {
    const raw = sessionStorage.getItem(SESSION_KEY);
    if (!raw) return null;
    return JSON.parse(raw);
  } catch (_) {
    return null;
  }
}

export function hasSavedSession() {
  return !!sessionStorage.getItem(SESSION_KEY);
}

export function clearSession() {
  sessionStorage.removeItem(SESSION_KEY);
}
