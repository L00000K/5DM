import { log } from './app.js';

const MIN_BH_DIST = 0.1;

// ── Build the voxel grid using K-nearest-neighbour IDW ─────────────────────────
//   options: { kNeighbors=5, idwPower=2 }
export function buildVoxelGrid(boreholes, geoUnits, cellSizeParam, options = {}) {
  if (!boreholes.length) throw new Error('No borehole data to interpolate');

  const kNeighbors = Math.max(1, options.kNeighbors ?? 5);
  const idwPower   = Math.max(0.5, options.idwPower ?? 2);

  // ── 1. Bounding box ────────────────────────────────────────────────────────
  const xs  = boreholes.map(b => b.x);
  const ys  = boreholes.map(b => b.y);
  const gls = boreholes.map(b => b.groundLevel ?? 0);
  const maxDepths = boreholes.map(b =>
    b.depth ?? (b.layers.length ? Math.max(...b.layers.map(l => l.base)) : 10));

  const minX = Math.min(...xs),  maxX = Math.max(...xs);
  const minY = Math.min(...ys),  maxY = Math.max(...ys);
  const maxGL  = Math.max(...gls);
  const maxDep = Math.max(...maxDepths);
  const topZ   = maxGL;
  const botZ   = maxGL - maxDep;

  // Typical inter-borehole spacing — drives certainty decay with distance
  const siteDiag      = Math.hypot(maxX - minX + 1, maxY - minY + 1);
  const typicalSpacing = siteDiag / Math.sqrt(boreholes.length);

  // 20 % margin
  const marginX = Math.max((maxX - minX) * 0.15, cellSizeParam * 2);
  const marginY = Math.max((maxY - minY) * 0.15, cellSizeParam * 2);

  const ox = minX - marginX;
  const oy = minY - marginY;
  const oz = botZ;

  // ── 2. Grid dimensions ─────────────────────────────────────────────────────
  const MAX_VOXELS = 500_000;
  let cellSize = cellSizeParam;
  let cellH    = cellSize / 5;
  let nx = Math.ceil((maxX + marginX - ox) / cellSize);
  let ny = Math.ceil((maxY + marginY - oy) / cellSize);
  let nz = Math.ceil((topZ - botZ)         / cellH);

  while (nx * ny * nz > MAX_VOXELS) {
    cellSize += 1;
    cellH = cellSize / 5;
    nx = Math.ceil((maxX + marginX - ox) / cellSize);
    ny = Math.ceil((maxY + marginY - oy) / cellSize);
    nz = Math.ceil((topZ - botZ)         / cellH);
    log(`Cell size auto-increased to ${cellSize} m to stay under 500 K voxels`, 'warn');
  }

  log(`Grid ${nx}×${ny}×${nz} = ${(nx*ny*nz).toLocaleString()} voxels @ ${cellSize} m cells | K=${kNeighbors} p=${idwPower}`, 'info');

  const total = nx * ny * nz;
  const unitIds     = new Uint8Array(total);    // winning unit id
  const certainty   = new Float32Array(total);  // 0–1
  const blendUnitIds = new Uint8Array(total);   // second-best unit id (for colour mixing)
  const blendRatios  = new Float32Array(total); // fraction of weight going to second-best

  // ── 3. Lookups ─────────────────────────────────────────────────────────────
  const unitIndex = {};
  geoUnits.forEach(u => { unitIndex[u.code] = u.id; });
  const unknownId = geoUnits.find(u => u.code === 'UNKN')?.id ?? 0;

  // ── 4. Interpolate ─────────────────────────────────────────────────────────
  for (let iz = 0; iz < nz; iz++) {
    const z = oz + iz * cellH + cellH * 0.5;

    for (let iy = 0; iy < ny; iy++) {
      const y = oy + iy * cellSize + cellSize * 0.5;

      for (let ix = 0; ix < nx; ix++) {
        const x   = ox + ix * cellSize + cellSize * 0.5;
        const idx = ix + iy * nx + iz * nx * ny;

        // Build candidate list: all BHs sorted by 2-D distance
        const candidates = [];
        for (const bh of boreholes) {
          const dist2d = Math.hypot(bh.x - x, bh.y - y);
          const bhGL   = bh.groundLevel ?? maxGL;
          const depth  = bhGL - z;        // depth of this voxel relative to this BH

          // Find which layer covers this depth; extrapolate beyond top/base if needed
          let layer = null;
          if (bh.layers.length) {
            if (depth < 0) {
              // Voxel is above this BH's ground surface — skip this BH
              continue;
            } else if (depth < bh.layers[0].top) {
              // Between ground surface and first logged layer — use shallowest unit
              layer = bh.layers[0];
            } else if (depth > bh.layers[bh.layers.length - 1].base) {
              // Below last logged layer — extrapolate using deepest unit
              layer = bh.layers[bh.layers.length - 1];
            } else {
              layer = bh.layers.find(l => depth >= l.top && depth <= l.base);
            }
          }
          if (!layer?.unitCode) continue;

          candidates.push({
            dist: Math.max(dist2d, MIN_BH_DIST),
            unitCode: layer.unitCode,
            layerCert: layer.certainty ?? 0.8,
          });
        }

        // Sort by distance, take K nearest
        candidates.sort((a, b) => a.dist - b.dist);
        const neighbours = candidates.slice(0, kNeighbors);

        if (!neighbours.length) {
          // Fall back to nearest BH's shallowest layer so every voxel gets a unit
          const fallback = boreholes
            .filter(b => b.layers.length)
            .map(b => ({ dist: Math.hypot(b.x - x, b.y - y), b }))
            .sort((a, b) => a.dist - b.dist)[0];
          const fbCode = fallback?.b.layers[0].unitCode;
          unitIds[idx]      = fbCode ? (unitIndex[fbCode] ?? unknownId) : unknownId;
          certainty[idx]    = 0.05;
          blendUnitIds[idx] = unitIds[idx];
          blendRatios[idx]  = 0;
          continue;
        }

        // IDW vote
        const votes = {};
        let totalW  = 0;
        for (const n of neighbours) {
          const w = (1 / Math.pow(n.dist, idwPower)) * n.layerCert;
          votes[n.unitCode] = (votes[n.unitCode] ?? 0) + w;
          totalW += w;
        }

        // Sort votes descending
        const sorted = Object.entries(votes).sort((a, b) => b[1] - a[1]);
        const [bestCode, bestW]         = sorted[0];
        const [secondCode, secondW = 0] = sorted[1] ?? [];

        const bestShare   = bestW / totalW;
        const blendRatio  = secondW / totalW;  // fraction going to second-best unit

        // Certainty: agreement × distance decay × mean layer confidence
        const nearestDist   = neighbours[0].dist;
        const distDecay     = Math.exp(-nearestDist / typicalSpacing);
        const meanLayerCert = neighbours.reduce((s, n) => s + n.layerCert, 0) / neighbours.length;
        const cert = Math.min(1,
          bestShare   * 0.55 +
          distDecay   * 0.30 +
          meanLayerCert * 0.15
        );

        unitIds[idx]      = unitIndex[bestCode]   ?? unknownId;
        certainty[idx]    = cert;
        blendUnitIds[idx] = secondCode ? (unitIndex[secondCode] ?? unknownId) : (unitIndex[bestCode] ?? unknownId);
        blendRatios[idx]  = blendRatio;
      }
    }
  }

  return {
    nx, ny, nz,
    cellSize, cellHeight: cellH,
    origin: { x: ox, y: oz, z: oy },
    worldWidth:  nx * cellSize,
    worldHeight: nz * cellH,
    worldDepth:  ny * cellSize,
    unitIds, certainty,
    blendUnitIds, blendRatios,
  };
}

export function voxelIndex(ix, iy, iz, grid) {
  return ix + iy * grid.nx + iz * grid.nx * grid.ny;
}

export function voxelWorldPos(ix, iy, iz, grid) {
  return {
    x: grid.origin.x + ix * grid.cellSize   + grid.cellSize   * 0.5,
    y: grid.origin.y + iz * grid.cellHeight + grid.cellHeight * 0.5,
    z: grid.origin.z + iy * grid.cellSize   + grid.cellSize   * 0.5,
  };
}
