import { computeCertainty, dataDensity, unitConsistency } from './semantic-engine.js';
import { log } from './app.js';

const MAX_VOXELS   = 500_000;
const IDW_POWER    = 2;
const MIN_BH_DIST  = 0.1; // avoid division by zero

// ── Build the voxel grid from classified boreholes ─────────────────────────────
export function buildVoxelGrid(boreholes, geoUnits, cellSizeParam) {
  if (!boreholes.length) throw new Error('No borehole data to interpolate');

  // ── 1. Bounding box ────────────────────────────────────────────────────────
  const xs = boreholes.map(b => b.x);
  const ys = boreholes.map(b => b.y);
  const gls = boreholes.map(b => b.groundLevel ?? 0);
  const maxDepths = boreholes.map(b => b.depth ?? Math.max(...b.layers.map(l => l.base)));

  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);

  const maxGL   = Math.max(...gls);
  const maxDep  = Math.max(...maxDepths);
  const topZ    = maxGL;
  const botZ    = maxGL - maxDep;

  // Add 20% margin around BH extents
  const marginX = Math.max((maxX - minX) * 0.15, cellSizeParam * 2);
  const marginY = Math.max((maxY - minY) * 0.15, cellSizeParam * 2);

  const ox = minX - marginX;
  const oy = minY - marginY;
  const oz = botZ;

  // ── 2. Grid dimensions ─────────────────────────────────────────────────────
  let cellSize = cellSizeParam;
  const cellH  = cellSize / 5;   // vertical cell is 1/5 of horizontal

  let nx = Math.ceil((maxX + marginX - ox) / cellSize);
  let ny = Math.ceil((maxY + marginY - oy) / cellSize);
  let nz = Math.ceil((topZ - botZ)         / cellH);

  // Cap voxel count, auto-increase cell size if needed
  while (nx * ny * nz > MAX_VOXELS) {
    cellSize += 1;
    nx = Math.ceil((maxX + marginX - ox) / cellSize);
    ny = Math.ceil((maxY + marginY - oy) / cellSize);
    nz = Math.ceil((topZ - botZ)         / (cellSize / 5));
    log(`Cell size increased to ${cellSize}m to stay under 500K voxels`, 'warn');
  }

  log(`Grid: ${nx}×${ny}×${nz} = ${(nx*ny*nz).toLocaleString()} voxels at ${cellSize}m cells`, 'info');

  const total = nx * ny * nz;
  const unitIds   = new Uint8Array(total);     // unit index (0 = unknown)
  const certainty = new Float32Array(total);   // 0–1

  // ── 3. Build unit code → id lookup ────────────────────────────────────────
  const unitIndex = {};
  geoUnits.forEach(u => { unitIndex[u.code] = u.id; });
  const unknownId = geoUnits.find(u => u.code === 'UNKN')?.id ?? 0;

  const searchR = cellSize * 3.5;

  // ── 4. Interpolate each voxel ─────────────────────────────────────────────
  for (let iz = 0; iz < nz; iz++) {
    const z = oz + iz * (cellSize / 5) + (cellSize / 5) * 0.5;

    for (let iy = 0; iy < ny; iy++) {
      const y = oy + iy * cellSize + cellSize * 0.5;

      for (let ix = 0; ix < nx; ix++) {
        const x = ox + ix * cellSize + cellSize * 0.5;
        const idx = ix + iy * nx + iz * nx * ny;

        // Find boreholes within search radius
        const neighbours = [];
        for (const bh of boreholes) {
          const dist2d = Math.hypot(bh.x - x, bh.y - y);
          if (dist2d > searchR) continue;
          const depth = (bh.groundLevel ?? maxGL) - z;
          if (depth < 0) continue; // above ground surface
          const layer = bh.layers.find(l => depth >= l.top && depth <= l.base);
          if (!layer || !layer.unitCode) continue;
          neighbours.push({
            bh, layer, dist: Math.max(dist2d, MIN_BH_DIST),
            unitCode: layer.unitCode,
            certainty: layer.certainty ?? 0.8,
          });
        }

        if (!neighbours.length) {
          unitIds[idx]   = unknownId;
          certainty[idx] = 0;
          continue;
        }

        // IDW vote: accumulate weight per unit code
        const votes = {};
        let totalW = 0;

        for (const n of neighbours) {
          const w = (1 / Math.pow(n.dist, IDW_POWER)) * n.certainty;
          votes[n.unitCode] = (votes[n.unitCode] || 0) + w;
          totalW += w;
        }

        // Winning unit
        let bestCode = 'UNKN', bestW = 0;
        for (const [code, w] of Object.entries(votes)) {
          if (w > bestW) { bestW = w; bestCode = code; }
        }

        const idwConf  = bestW / totalW;
        const density  = dataDensity(x, y, boreholes, searchR);
        const consist  = unitConsistency(x, y, z, boreholes, searchR, bestCode);

        unitIds[idx]   = unitIndex[bestCode] ?? unknownId;
        certainty[idx] = computeCertainty(idwConf, density, consist);
      }
    }
  }

  return {
    nx, ny, nz,
    cellSize,
    cellHeight: cellSize / 5,
    origin: { x: ox, y: oz, z: oy },  // Three.js: x=East, y=Up(elevation), z=North
    worldWidth:  nx * cellSize,
    worldHeight: nz * (cellSize / 5),
    worldDepth:  ny * cellSize,
    unitIds,
    certainty,
  };
}

// ── Voxel index helper ─────────────────────────────────────────────────────────
export function voxelIndex(ix, iy, iz, grid) {
  return ix + iy * grid.nx + iz * grid.nx * grid.ny;
}

// ── World position of voxel centre ────────────────────────────────────────────
export function voxelWorldPos(ix, iy, iz, grid) {
  return {
    x: grid.origin.x + ix * grid.cellSize    + grid.cellSize    * 0.5,
    y: grid.origin.y + iz * grid.cellHeight  + grid.cellHeight  * 0.5,  // y=elevation
    z: grid.origin.z + iy * grid.cellSize    + grid.cellSize    * 0.5,
  };
}
