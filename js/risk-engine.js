// ── Geotechnical Risk Engine ──────────────────────────────────────────────────
// Identifies and scores geotechnical hazard zones from the voxel model.
// Risk categories:
//   settlement  — high-compressibility units (Cc, e0) near surface
//   bearing     — soft units (low Cu) at shallow depth
//   liquefaction — low SPT N sand units below GWT
//   slope       — low φ' units with moderate slope angle
//   uncertainty — model certainty below threshold
//   contamination — made ground / fill indicator

const RISK_LEVELS = {
  low:    { label: 'Low',    color: '#4fba6f', bg: 'rgba(79,186,111,0.15)' },
  medium: { label: 'Medium', color: '#d4a843', bg: 'rgba(212,168,67,0.15)' },
  high:   { label: 'High',   color: '#e06040', bg: 'rgba(224,96,64,0.15)'  },
};

// ── Main risk assessment ───────────────────────────────────────────────────────
// grid:      voxel grid object
// geoUnits:  array of unit objects with params
// classifiedBH: array of classified borehole objects
// gwtElevation: groundwater table elevation (mAOD) or null
// Returns: RiskReport { zones: RiskZone[], summary: string }
export function assessRisk(grid, geoUnits, classifiedBH, gwtElevation) {
  if (!grid) return null;

  const { nx, ny, nz, cellSize: cs, cellHeight: ch, origin: O, unitIds, certainty } = grid;
  const unitById = {}, unitByCode = {};
  geoUnits.forEach(u => { unitById[u.id] = u; unitByCode[u.code] = u; });

  const zones = [];

  // ── Per-column analysis ──────────────────────────────────────────────────────
  let settleCount = 0, bearingCount = 0, liquefactionCount = 0, uncertCount = 0;

  for (let iy = 0; iy < ny; iy++) {
    for (let ix = 0; ix < nx; ix++) {
      const wx = O.x + (ix + 0.5) * cs;
      const wz = O.z + (iy + 0.5) * cs;

      // Find surface voxel (topmost non-empty)
      let surfaceElev = null;
      for (let iz = nz - 1; iz >= 0; iz--) {
        if (unitIds[ix + iy * nx + iz * nx * ny]) {
          surfaceElev = O.y + iz * ch + ch;
          break;
        }
      }
      if (surfaceElev == null) continue;

      // Scan downward from surface
      let compressibleDepth = 0; // m of compressible material in top 10m
      let minCu = Infinity;      // minimum Cu in top 5m
      let hasSandBelowGWT = false;

      for (let iz = nz - 1; iz >= 0; iz--) {
        const flat  = ix + iy * nx + iz * nx * ny;
        const uid   = unitIds[flat];
        if (!uid) continue;
        const unit  = unitById[uid];
        const elev  = O.y + iz * ch + ch * 0.5;
        const depth = surfaceElev - elev;

        // Settlement risk: compressible soils in top 10m
        if (depth <= 10 && unit?.params?.Cc != null && unit.params.Cc > 0.15) {
          compressibleDepth += ch;
        }

        // Bearing capacity risk: soft soils in top 5m
        if (depth <= 5 && unit?.params?.cu != null) {
          minCu = Math.min(minCu, unit.params.cu);
        }

        // Liquefaction: sand with low N below GWT
        if (gwtElevation != null && elev < gwtElevation) {
          const isSand = /sand|gravel|rtd|rtg/i.test((unit?.name ?? '') + (unit?.description ?? ''));
          if (isSand && unit?.params?.N_spt != null && unit.params.N_spt < 15) {
            hasSandBelowGWT = true;
          }
        }
      }

      // Flag columns
      if (compressibleDepth > 2) {
        settleCount++;
      }
      if (minCu < 25 && minCu !== Infinity) {
        bearingCount++;
      }
      if (hasSandBelowGWT) {
        liquefactionCount++;
      }
    }
  }

  // Low certainty voxels
  let totalVox = 0, lowCertVox = 0;
  for (let i = 0; i < unitIds.length; i++) {
    if (unitIds[i]) { totalVox++; if (certainty[i] < 0.4) lowCertVox++; }
  }
  if (totalVox > 0) uncertCount = Math.round(lowCertVox / totalVox * 100);

  const totalCols = nx * ny;

  // ── Build zone summary ────────────────────────────────────────────────────────
  const pct = n => (n / totalCols * 100).toFixed(0);

  if (settleCount > 0) {
    zones.push({
      id: 'settlement',
      name: 'Settlement Risk',
      icon: '⇣',
      level: settleCount / totalCols > 0.3 ? 'high' : settleCount / totalCols > 0.1 ? 'medium' : 'low',
      description: `${pct(settleCount)}% of model columns contain >2m of compressible material (Cc>0.15) in the top 10m. Monitor for differential settlement.`,
      affectedCols: settleCount,
      pct: pct(settleCount),
    });
  }

  if (bearingCount > 0) {
    zones.push({
      id: 'bearing',
      name: 'Bearing Capacity',
      icon: '⊟',
      level: bearingCount / totalCols > 0.2 ? 'high' : bearingCount / totalCols > 0.05 ? 'medium' : 'low',
      description: `${pct(bearingCount)}% of columns have Cu<25 kPa within 5m of surface. Shallow foundation design requires verification.`,
      affectedCols: bearingCount,
      pct: pct(bearingCount),
    });
  }

  if (liquefactionCount > 0) {
    zones.push({
      id: 'liquefaction',
      name: 'Liquefaction Potential',
      icon: '〰',
      level: liquefactionCount / totalCols > 0.15 ? 'high' : 'medium',
      description: `${pct(liquefactionCount)}% of columns have loose sand (N<15) below the inferred GWT at ${gwtElevation?.toFixed(1) ?? '?'} mAOD. Seismic assessment recommended.`,
      affectedCols: liquefactionCount,
      pct: pct(liquefactionCount),
    });
  }

  // Made ground / contamination check from unit names
  const mgUnit = geoUnits.find(u => /made.?ground|fill|mg/i.test(u.code + ' ' + u.name));
  if (mgUnit) {
    let mgCols = 0;
    for (let iy = 0; iy < ny; iy++) {
      for (let ix = 0; ix < nx; ix++) {
        for (let iz = nz - 1; iz >= 0; iz--) {
          const uid = unitIds[ix + iy * nx + iz * nx * ny];
          if (uid === mgUnit.id) { mgCols++; break; }
        }
      }
    }
    if (mgCols > 0) {
      zones.push({
        id: 'contamination',
        name: 'Made Ground / Fill',
        icon: '⚠',
        level: mgCols / totalCols > 0.3 ? 'medium' : 'low',
        description: `${pct(mgCols)}% of site area contains made ground (${mgUnit.code}). Phase 1 desk study and gas monitoring recommended. Potential contamination source.`,
        affectedCols: mgCols,
        pct: pct(mgCols),
      });
    }
  }

  if (uncertCount > 20) {
    zones.push({
      id: 'uncertainty',
      name: 'Model Uncertainty',
      icon: '?',
      level: uncertCount > 40 ? 'high' : 'medium',
      description: `${uncertCount}% of voxels have certainty <40%. Additional SI boreholes recommended in data-sparse areas to improve model confidence.`,
      affectedCols: lowCertVox,
      pct: uncertCount.toString(),
    });
  }

  // Overall risk grade
  const highCount = zones.filter(z => z.level === 'high').length;
  const medCount  = zones.filter(z => z.level === 'medium').length;
  const overallLevel = highCount > 0 ? 'high' : medCount > 1 ? 'medium' : 'low';

  return {
    zones,
    overallLevel,
    summary: zones.length === 0
      ? 'No significant geotechnical hazards identified from model data.'
      : `${zones.length} hazard zone(s) identified. ${highCount > 0 ? 'HIGH risk — immediate engineering attention required.' : medCount > 0 ? 'MEDIUM risk — detailed geotechnical assessment recommended.' : 'LOW risk — standard precautions apply.'}`,
  };
}

// ── Render risk report to a container element ─────────────────────────────────
export function renderRiskReport(report, container) {
  if (!container) return;
  if (!report) {
    container.innerHTML = '<p class="hint">Build 3D model and click Assess Risk to evaluate hazard zones.</p>';
    return;
  }

  const levelDot = lvl => `<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${RISK_LEVELS[lvl]?.color};margin-right:5px;vertical-align:middle"></span>`;

  let html = `
    <div style="padding:6px 0 10px;border-bottom:1px solid var(--border);margin-bottom:8px">
      <span style="font-size:11px;color:var(--text-mid)">Overall risk:</span>
      <span style="margin-left:6px;font-weight:700;color:${RISK_LEVELS[report.overallLevel]?.color}">
        ${levelDot(report.overallLevel)}${RISK_LEVELS[report.overallLevel]?.label ?? report.overallLevel}
      </span>
    </div>`;

  if (!report.zones.length) {
    html += `<p class="hint">${report.summary}</p>`;
  } else {
    report.zones.forEach(z => {
      const style = RISK_LEVELS[z.level] ?? RISK_LEVELS.low;
      html += `
        <div style="border-left:3px solid ${style.color};background:${style.bg};padding:6px 8px;margin-bottom:8px;border-radius:0 4px 4px 0">
          <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:3px">
            <span style="font-size:11px;font-weight:700;color:${style.color}">${z.icon} ${z.name}</span>
            <span style="font-size:10px;background:${style.color};color:#fff;padding:1px 5px;border-radius:3px">${style.label} · ${z.pct}%</span>
          </div>
          <p style="font-size:10px;color:var(--text-mid);margin:0;line-height:1.4">${z.description}</p>
        </div>`;
    });
  }

  container.innerHTML = html;
}
