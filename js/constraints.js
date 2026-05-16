// ── Geological constraint parser and applicator ───────────────────────────────
// Parses natural-language rules and applies them to a built voxel grid.
// Rules are matched per-line against known unit codes / names.

export function parseConstraints(text, geoUnits) {
  const rules = [];
  const lines = text.split('\n').map(l => l.trim()).filter(Boolean);

  function findUnit(line) {
    const sl = line.toLowerCase();
    for (const u of geoUnits) {
      if (sl.includes(u.code.toLowerCase())) return u;
    }
    for (const u of geoUnits) {
      const words = u.name.toLowerCase().split(/\s+/);
      if (words.some(w => w.length > 3 && sl.includes(w))) return u;
    }
    return null;
  }

  for (const line of lines) {
    const lower = line.toLowerCase();
    let matched = false;

    // "X not deeper than Ym" | "X depth < Ym"
    let m = lower.match(/not\s+deeper\s+than\s+([\d.]+)\s*m/);
    if (!m) m = lower.match(/depth\s*[<≤]\s*([\d.]+)\s*m/);
    if (m) {
      const unit = findUnit(line);
      if (unit) {
        rules.push({ type: 'maxDepth', unitId: unit.id, unitCode: unit.code, maxDepth: parseFloat(m[1]), raw: line });
        matched = true;
      }
    }

    if (!matched) {
      // "X is below YmAOD" | "X below Y m elevation"
      m = lower.match(/(?:is\s+)?below\s+([-\d.]+)\s*m(?:\s*aod)?/);
      if (m) {
        const unit = findUnit(line);
        if (unit) {
          rules.push({ type: 'maxElevation', unitId: unit.id, unitCode: unit.code, maxElev: parseFloat(m[1]), raw: line });
          matched = true;
        }
      }
    }

    if (!matched) {
      // "X is above YmAOD"
      m = lower.match(/(?:is\s+)?above\s+([-\d.]+)\s*m(?:\s*aod)?/);
      if (m) {
        const unit = findUnit(line);
        if (unit) {
          rules.push({ type: 'minElevation', unitId: unit.id, unitCode: unit.code, minElev: parseFloat(m[1]), raw: line });
          matched = true;
        }
      }
    }

    if (!matched) {
      // "X between YmAOD and ZmAOD"
      m = lower.match(/between\s+([-\d.]+)\s*m\s+and\s+([-\d.]+)\s*m/);
      if (m) {
        const unit = findUnit(line);
        if (unit) {
          rules.push({ type: 'elevRange', unitId: unit.id, unitCode: unit.code,
            minElev: Math.min(parseFloat(m[1]), parseFloat(m[2])),
            maxElev: Math.max(parseFloat(m[1]), parseFloat(m[2])), raw: line });
          matched = true;
        }
      }
    }

    if (!matched) {
      // "X exists only in [N/S/E/W] half/third/quarter"
      m = lower.match(/only\s+in\s+(?:the\s+)?(north|south|east|west)(?:ern)?\s+(half|third|quarter)/);
      if (m) {
        const unit = findUnit(line);
        if (unit) {
          rules.push({ type: 'spatialZone', unitId: unit.id, unitCode: unit.code,
            direction: m[1], fraction: { half: 0.5, third: 0.333, quarter: 0.25 }[m[2]], raw: line });
          matched = true;
        }
      }
    }

    if (!matched) {
      // "X dips N degrees [to the] north/south/east/west"
      m = lower.match(/dips?\s+([\d.]+)\s*(?:degrees?|°)?\s*(?:to\s+(?:the\s+)?)?(north|south|east|west)(?:ern)?/);
      if (!m) m = lower.match(/dips?\s+(north|south|east|west)(?:ern)?\s*(?:at|@)?\s*([\d.]+)\s*(?:degrees?|°)?/) &&
                 [null, lower.match(/dips?\s+(north|south|east|west)(?:ern)?/)[1],
                  lower.match(/([\d.]+)\s*(?:degrees?|°)/)?.[1]];
      if (m && m[1] && m[2]) {
        const angleStr = parseFloat(m[1]) ? m[1] : m[2];
        const dirStr   = isNaN(parseFloat(m[1])) ? m[1] : m[2];
        const angle    = parseFloat(angleStr);
        if (!isNaN(angle) && angle >= 0 && angle <= 89) {
          const unit = findUnit(line);
          if (unit) {
            rules.push({ type: 'dip', unitId: unit.id, unitCode: unit.code,
              dipAngle: angle, dipDir: dirStr.toLowerCase().replace(/ern$/, ''), raw: line });
            matched = true;
          }
        }
      }
    }

    if (!matched) {
      // "Fault/fault zone at X=Nm / northing Nm / easting Nm"  → fault plane rule
      m = lower.match(/fault(?:\s+zone)?\s+(?:at\s+)?(?:x\s*=?\s*|easting\s+)([-\d.]+)\s*m?/);
      if (!m) m = lower.match(/fault(?:\s+zone)?\s+(?:at\s+)?(?:y\s*=?\s*|northing\s+)([-\d.]+)\s*m?/);
      if (m) {
        const coord = parseFloat(m[1]);
        const axis  = lower.includes('north') || lower.match(/\by\b/) ? 'y' : 'x';
        rules.push({ type: 'fault', axis, coord, raw: line });
        matched = true;
      }
    }

    if (!matched) {
      // Semantic note — store but no auto-action
      rules.push({ type: 'note', raw: line });
    }
  }

  return rules;
}

// Apply parsed constraints to a voxel grid (modifies unitIds in-place).
// Returns count of reassigned voxels.
export function applyConstraints(grid, rules, geoUnits) {
  const { nx, ny, nz, cellHeight: ch, cellSize: cs, origin, unitIds, certainty, blendUnitIds } = grid;

  const unitById = {};
  geoUnits.forEach(u => { unitById[u.id] = u; });

  // Pre-compute ground surface elevation per XY column (topmost non-zero voxel)
  const surfaceY = new Float32Array(nx * ny).fill(origin.y);
  for (let iy = 0; iy < ny; iy++) {
    for (let ix = 0; ix < nx; ix++) {
      for (let iz = nz - 1; iz >= 0; iz--) {
        const flat = ix + iy * nx + iz * nx * ny;
        if (unitIds[flat] !== 0) {
          surfaceY[ix + iy * nx] = origin.y + iz * ch + ch;
          break;
        }
      }
    }
  }

  // Site bounding box for spatial zone rules
  const siteMinX = origin.x, siteMaxX = origin.x + nx * cs;
  const siteMinZ = origin.z, siteMaxZ = origin.z + ny * cs;

  let applied = 0;

  const actionRules = rules.filter(r => r.type !== 'note');

  // Pre-compute reference elevation (mean top of unit) per dip rule at model centre
  const centreX = origin.x + (nx / 2) * cs;
  const centreZ = origin.z + (ny / 2) * cs;
  const dipRefElev = {};
  for (const rule of actionRules) {
    if (rule.type !== 'dip') continue;
    let maxElevForUnit = -Infinity;
    const cx = Math.floor(nx / 2), cy = Math.floor(ny / 2);
    for (let iz = nz - 1; iz >= 0; iz--) {
      if (unitIds[cx + cy * nx + iz * nx * ny] === rule.unitId) {
        maxElevForUnit = origin.y + iz * ch + ch;
        break;
      }
    }
    if (maxElevForUnit === -Infinity) {
      // Fallback: scan whole grid for mean top of unit
      let sum = 0, count = 0;
      for (let iy = 0; iy < ny; iy++) {
        for (let ix = 0; ix < nx; ix++) {
          for (let iz = nz - 1; iz >= 0; iz--) {
            if (unitIds[ix + iy * nx + iz * nx * ny] === rule.unitId) {
              sum += origin.y + iz * ch + ch;
              count++;
              break;
            }
          }
        }
      }
      maxElevForUnit = count > 0 ? sum / count : 0;
    }
    dipRefElev[`${rule.unitId}`] = maxElevForUnit;
  }

  for (let iz = 0; iz < nz; iz++) {
    const voxelElev = origin.y + iz * ch + ch * 0.5;

    for (let iy = 0; iy < ny; iy++) {
      const voxelZ = origin.z + iy * cs + cs * 0.5;

      for (let ix = 0; ix < nx; ix++) {
        const flat = ix + iy * nx + iz * nx * ny;
        const uid  = unitIds[flat];
        if (uid === 0) continue;
        const unit = unitById[uid];
        if (!unit) continue;

        const voxelX = origin.x + ix * cs + cs * 0.5;

        for (const rule of actionRules) {
          let violated = false;

          if (rule.type === 'fault') {
            // Fault rule applies to ALL units — reassign across fault plane
            const pos = rule.axis === 'x' ? voxelX : voxelZ;
            // Voxels within 1 cell of fault are always blanked; otherwise skip
            violated = Math.abs(pos - rule.coord) < cs;
          } else {
            if (rule.unitId !== uid) continue;

            if (rule.type === 'maxDepth') {
              const surf = surfaceY[ix + iy * nx];
              violated = (surf - voxelElev) > rule.maxDepth;
            } else if (rule.type === 'maxElevation') {
              violated = voxelElev > rule.maxElev;
            } else if (rule.type === 'minElevation') {
              violated = voxelElev < rule.minElev;
            } else if (rule.type === 'elevRange') {
              violated = voxelElev < rule.minElev || voxelElev > rule.maxElev;
            } else if (rule.type === 'spatialZone') {
              const f = rule.fraction;
              const dir = rule.direction;
              if (dir === 'north') violated = voxelZ < siteMaxZ - (siteMaxZ - siteMinZ) * f;
              if (dir === 'south') violated = voxelZ > siteMinZ + (siteMaxZ - siteMinZ) * f;
              if (dir === 'east')  violated = voxelX < siteMaxX - (siteMaxX - siteMinX) * f;
              if (dir === 'west')  violated = voxelX > siteMinX + (siteMaxX - siteMinX) * f;
            } else if (rule.type === 'dip') {
              // Dip rule: unit's top contact shifts linearly with position in dip direction
              const refElev = dipRefElev[`${rule.unitId}`] ?? 0;
              const tanDip  = Math.tan(rule.dipAngle * Math.PI / 180);
              let dipShift = 0;
              if (rule.dipDir === 'north') dipShift = -(voxelZ - centreZ) * tanDip;
              if (rule.dipDir === 'south') dipShift =  (voxelZ - centreZ) * tanDip;
              if (rule.dipDir === 'east')  dipShift = -(voxelX - centreX) * tanDip;
              if (rule.dipDir === 'west')  dipShift =  (voxelX - centreX) * tanDip;
              // Unit is violated if above the dipping top surface
              violated = voxelElev > refElev + dipShift + ch * 0.5;
            }
          }

          if (violated) {
            const bid = blendUnitIds[flat];
            unitIds[flat]   = (bid && bid !== uid) ? bid : 0;
            certainty[flat] = Math.max(0, certainty[flat] - 0.35);
            applied++;
            break;
          }
        }
      }
    }
  }

  return applied;
}

export function constraintSummary(rules) {
  return rules.map(r => {
    if (r.type === 'note')  return { label: '📝 Note', text: r.raw, active: false };
    if (r.type === 'fault') return { label: `✓ Fault@${r.coord}m`, text: r.raw, active: true };
    if (r.type === 'dip')   return { label: `✓ ${r.unitCode} dip`, text: r.raw, active: true };
    return { label: `✓ ${r.unitCode}`, text: r.raw, active: true };
  });
}
