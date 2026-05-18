import { GLTFExporter } from 'three/addons/exporters/GLTFExporter.js';
import { OBJExporter  } from 'three/addons/exporters/OBJExporter.js';
import { AppState, log } from './app.js';
import { exportPropertiesCSV } from './properties.js';

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a   = document.createElement('a');
  a.href     = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
}

export function initExporter() {
  document.getElementById('btn-export-gltf').addEventListener('click', () => {
    if (!AppState.scene) return;
    log('Exporting GLTF…', 'info');
    const exporter = new GLTFExporter();
    exporter.parse(
      AppState.scene.voxelGroup,
      gltf => {
        const blob = new Blob([JSON.stringify(gltf)], { type: 'model/gltf+json' });
        downloadBlob(blob, 'geomodel.gltf');
        log('GLTF exported.', 'ok');
      },
      err => log(`GLTF export error: ${err}`, 'error'),
      { binary: false }
    );
  });

  document.getElementById('btn-export-obj').addEventListener('click', () => {
    if (!AppState.scene) return;
    log('Exporting OBJ…', 'info');
    try {
      const exporter = new OBJExporter();
      const obj  = exporter.parse(AppState.scene.voxelGroup);
      const blob = new Blob([obj], { type: 'text/plain' });
      downloadBlob(blob, 'geomodel.obj');
      log('OBJ exported.', 'ok');
    } catch (err) {
      log(`OBJ export error: ${err.message}`, 'error');
    }
  });

  document.getElementById('btn-export-json').addEventListener('click', () => {
    if (!AppState.voxelGrid) { log('No voxel grid to export.', 'warn'); return; }
    log('Exporting JSON voxel data…', 'info');
    const { nx, ny, nz, cellSize, cellHeight, origin } = AppState.voxelGrid;
    const payload = {
      meta: { nx, ny, nz, cellSize, cellHeight, origin, units: AppState.geoUnits },
      unitIds:   Array.from(AppState.voxelGrid.unitIds),
      certainty: Array.from(AppState.voxelGrid.certainty),
    };
    downloadBlob(new Blob([JSON.stringify(payload)], { type: 'application/json' }), 'geomodel-voxels.json');
    log('JSON exported.', 'ok');
  });

  // ── Export VTK rectilinear grid (Paraview) ────────────────────────────────
  document.getElementById('btn-export-vtk')?.addEventListener('click', () => {
    const grid = AppState.voxelGrid;
    if (!grid) { log('No voxel grid to export.', 'warn'); return; }
    const { nx, ny, nz, cellSize: cs, cellHeight: ch, origin: O, unitIds, certainty } = grid;

    const xc = Array.from({length: nx + 1}, (_, i) => (O.x + i * cs).toFixed(2)).join(' ');
    const yc = Array.from({length: ny + 1}, (_, i) => (O.z + i * cs).toFixed(2)).join(' ');
    const zc = Array.from({length: nz + 1}, (_, i) => (O.y + i * ch).toFixed(2)).join(' ');

    const uIds = [], certs = [];
    for (let iz = 0; iz < nz; iz++) {
      for (let iy = 0; iy < ny; iy++) {
        for (let ix = 0; ix < nx; ix++) {
          const flat = ix + iy * nx + iz * nx * ny;
          uIds.push(unitIds[flat]);
          certs.push(certainty[flat].toFixed(3));
        }
      }
    }

    const vtk = [
      '# vtk DataFile Version 3.0',
      'GeoModel AI',
      'ASCII',
      'DATASET RECTILINEAR_GRID',
      `DIMENSIONS ${nx + 1} ${ny + 1} ${nz + 1}`,
      `X_COORDINATES ${nx + 1} float`, xc,
      `Y_COORDINATES ${ny + 1} float`, yc,
      `Z_COORDINATES ${nz + 1} float`, zc,
      `CELL_DATA ${nx * ny * nz}`,
      'SCALARS unit_id int 1',
      'LOOKUP_TABLE default',
      uIds.join(' '),
      'SCALARS certainty float 1',
      'LOOKUP_TABLE default',
      certs.join(' '),
    ].join('\n');

    downloadBlob(new Blob([vtk], { type: 'text/plain' }), 'geomodel.vtk');
    log(`VTK exported — ${(nx*ny*nz).toLocaleString()} cells.`, 'ok');
  });

  // ── Export formation contacts as CSV ─────────────────────────────────────
  // Includes P10/P50/P90 depth uncertainty when MC probability volumes exist.
  document.getElementById('btn-export-contacts')?.addEventListener('click', () => {
    const grid = AppState.voxelGrid;
    if (!grid) { log('No voxel grid to export.', 'warn'); return; }
    const { nx, ny, nz, cellSize: cs, cellHeight: ch, origin: O, unitIds, probVolumes } = grid;
    const unitById = {};
    AppState.geoUnits.forEach(u => { unitById[u.id] = u; });

    const hasMC = probVolumes?.size > 0;
    const header = hasMC
      ? 'Unit_Code,Unit_Name,IX,IY,X_m,Y_m,Top_Z_P50_mAOD,Top_Z_P10_mAOD,Top_Z_P90_mAOD,Depth_P50_m,Depth_P10_m,Depth_P90_m'
      : 'Unit_Code,Unit_Name,IX,IY,X_m,Y_m,Top_Z_mAOD,Depth_bGL_m';
    const rows = [header];

    for (let iy = 0; iy < ny; iy++) {
      for (let ix = 0; ix < nx; ix++) {
        const x = O.x + (ix + 0.5) * cs;
        const y = O.z + (iy + 0.5) * cs;

        if (hasMC) {
          // For each unit with a probability volume, find P10/P50/P90 top contact
          for (const [code, probVol] of probVolumes) {
            const unit = AppState.geoUnits.find(u => u.code === code);
            if (!unit) continue;

            let izP50 = -1, izP10 = -1, izP90 = -1;
            for (let iz = nz - 1; iz >= 0; iz--) {
              const p = probVol[ix + iy * nx + iz * nx * ny];
              if (izP10 < 0 && p >= 0.10) izP10 = iz;
              if (izP50 < 0 && p >= 0.50) izP50 = iz;
              if (izP90 < 0 && p >= 0.90) izP90 = iz;
              if (izP10 >= 0 && izP50 >= 0 && izP90 >= 0) break;
            }
            if (izP50 < 0) continue; // unit not present in this column

            const zP50 = O.y + (izP50 + 1) * ch;
            const zP10 = izP10 >= 0 ? O.y + (izP10 + 1) * ch : zP50 + ch;
            const zP90 = izP90 >= 0 ? O.y + (izP90 + 1) * ch : zP50 - ch;

            // Estimate ground surface from topmost voxel in column
            let surfZ = O.y + nz * ch;
            for (let iz = nz - 1; iz >= 0; iz--) {
              if (unitIds[ix + iy * nx + iz * nx * ny]) { surfZ = O.y + (iz + 1) * ch; break; }
            }

            rows.push([
              code,
              `"${(unit.name ?? '').replace(/"/g, '""')}"`,
              ix, iy,
              x.toFixed(2), y.toFixed(2),
              zP50.toFixed(2), zP10.toFixed(2), zP90.toFixed(2),
              (surfZ - zP50).toFixed(2), (surfZ - zP10).toFixed(2), (surfZ - zP90).toFixed(2),
            ].join(','));
          }
        } else {
          // Without probVolumes: deterministic top per unit
          const topByUnit = {};
          for (let iz = nz - 1; iz >= 0; iz--) {
            const uid = unitIds[ix + iy * nx + iz * nx * ny];
            if (uid && topByUnit[uid] === undefined) topByUnit[uid] = iz;
          }
          // Ground surface elevation (topmost occupied voxel)
          const surfIz = Math.max(...Object.values(topByUnit));
          const surfZ  = surfIz >= 0 ? O.y + (surfIz + 1) * ch : O.y + nz * ch;

          for (const [uid, iz] of Object.entries(topByUnit)) {
            const unit = unitById[uid];
            if (!unit) continue;
            const top = O.y + (parseInt(iz) + 1) * ch;
            rows.push([
              unit.code,
              `"${(unit.name ?? '').replace(/"/g, '""')}"`,
              ix, iy, x.toFixed(2), y.toFixed(2),
              top.toFixed(2), (surfZ - top).toFixed(2),
            ].join(','));
          }
        }
      }
    }

    downloadBlob(new Blob([rows.join('\n')], { type: 'text/csv' }), 'formation-contacts.csv');
    const suffix = hasMC ? ' (with P10/P50/P90 uncertainty)' : '';
    log(`Formation contacts exported — ${rows.length - 1} contact points${suffix}.`, 'ok');
  });

  // ── Export point cloud CSV (every voxel centre) ──────────────────────────────
  document.getElementById('btn-export-pointcloud')?.addEventListener('click', () => {
    const grid = AppState.voxelGrid;
    if (!grid) { log('No voxel grid to export.', 'warn'); return; }
    const { nx, ny, nz, cellSize: cs, cellHeight: ch, origin: O, unitIds, certainty } = grid;
    const unitById = {};
    AppState.geoUnits.forEach(u => { unitById[u.id] = u; });
    log('Generating point cloud CSV…', 'info');
    const rows = ['X,Y,Z,Unit_Code,Unit_Name,Certainty'];
    for (let iz = 0; iz < nz; iz++) {
      for (let iy = 0; iy < ny; iy++) {
        for (let ix = 0; ix < nx; ix++) {
          const flat = ix + iy * nx + iz * nx * ny;
          const uid  = unitIds[flat];
          if (!uid) continue;
          const unit = unitById[uid];
          if (!unit) continue;
          const x = (O.x + (ix + 0.5) * cs).toFixed(2);
          const y = (O.z + (iy + 0.5) * cs).toFixed(2);
          const z = (O.y + (iz + 0.5) * ch).toFixed(2);
          rows.push([x, y, z, unit.code,
            `"${(unit.name ?? '').replace(/"/g,'""')}"`,
            certainty[flat].toFixed(3)].join(','));
        }
      }
    }
    downloadBlob(new Blob([rows.join('\n')], { type: 'text/csv' }), 'geomodel-pointcloud.csv');
    log(`Point cloud exported — ${(rows.length - 1).toLocaleString()} points.`, 'ok');
  });

  // ── Export model statistics CSV ───────────────────────────────────────────
  document.getElementById('btn-export-stats')?.addEventListener('click', () => {
    const grid = AppState.voxelGrid;
    if (!grid) { log('No voxel grid to export.', 'warn'); return; }
    const { nx, ny, nz, cellSize: cs, cellHeight: ch, origin: O, unitIds, certainty } = grid;
    const cellVol = cs * cs * ch;
    const cellArea = cs * cs;

    // Per-unit accumulators
    const stats = {};
    AppState.geoUnits.forEach(u => {
      stats[u.id] = { count: 0, certSum: 0,
        minElev: Infinity, maxElev: -Infinity, topCols: 0, topElevSum: 0 };
    });

    // Scan top-of-unit per column
    const topElev = {}; // uid → {sum, count}
    AppState.geoUnits.forEach(u => { topElev[u.id] = { sum: 0, count: 0 }; });

    for (let iy = 0; iy < ny; iy++) {
      for (let ix = 0; ix < nx; ix++) {
        const colTop = {}; // first (topmost) occurrence of each unit per column
        for (let iz = nz - 1; iz >= 0; iz--) {
          const flat = ix + iy * nx + iz * nx * ny;
          const uid  = unitIds[flat];
          if (!uid || !stats[uid]) continue;
          const elev = O.y + iz * ch + ch * 0.5;
          const s    = stats[uid];
          s.count++;
          s.certSum += certainty[flat];
          s.minElev = Math.min(s.minElev, elev - ch * 0.5);
          s.maxElev = Math.max(s.maxElev, elev + ch * 0.5);
          if (colTop[uid] === undefined) colTop[uid] = elev + ch * 0.5;
        }
        for (const [uid, e] of Object.entries(colTop)) {
          topElev[uid].sum  += e;
          topElev[uid].count++;
        }
      }
    }

    const totalCount = AppState.geoUnits.reduce((a, u) => a + (stats[u.id]?.count ?? 0), 0);
    const cols = [
      'Unit_Code','Unit_Name',
      'Volume_m3','Area_m2','Pct_Model',
      'Mean_Top_mAOD','Mean_Base_mAOD','Mean_Thickness_m',
      'Mean_Certainty_pct',
    ];
    const rows = [cols.join(',')];
    for (const u of AppState.geoUnits) {
      const s  = stats[u.id];
      if (!s || !s.count) continue;
      const vol   = (s.count * cellVol).toFixed(1);
      const area  = (topElev[u.id].count * cellArea).toFixed(1);
      const pct   = totalCount > 0 ? (s.count / totalCount * 100).toFixed(1) : '0';
      const meanTop = topElev[u.id].count > 0
        ? (topElev[u.id].sum / topElev[u.id].count).toFixed(2) : '';
      const meanBase = s.minElev !== Infinity ? s.minElev.toFixed(2) : '';
      const thick    = (s.maxElev !== -Infinity && s.minElev !== Infinity)
        ? (s.maxElev - s.minElev).toFixed(2) : '';
      const cert = (s.certSum / s.count * 100).toFixed(1);
      rows.push([
        u.code, `"${(u.name ?? '').replace(/"/g, '""')}"`,
        vol, area, pct, meanTop, meanBase, thick, cert,
      ].join(','));
    }
    downloadBlob(new Blob([rows.join('\n')], { type: 'text/csv' }), 'geomodel-stats.csv');
    log(`Model statistics exported — ${rows.length - 1} units.`, 'ok');
  });

  // ── Material Quantity Takeoff — HTML printable report ──────────────────────
  document.getElementById('btn-export-qty-report')?.addEventListener('click', () => {
    const grid = AppState.voxelGrid;
    if (!grid) { log('Build the 3D model first.', 'warn'); return; }
    const { nx, ny, nz, cellSize: cs, cellHeight: ch, origin: O, unitIds, certainty } = grid;
    const cellVol = cs * cs * ch;

    const stats = {};
    AppState.geoUnits.forEach(u => { stats[u.id] = { count: 0, certSum: 0, minElev: Infinity, maxElev: -Infinity }; });
    const topElev = {};
    AppState.geoUnits.forEach(u => { topElev[u.id] = { sum: 0, count: 0 }; });

    for (let iy = 0; iy < ny; iy++) {
      for (let ix = 0; ix < nx; ix++) {
        const colTop = {};
        for (let iz = nz - 1; iz >= 0; iz--) {
          const flat = ix + iy * nx + iz * nx * ny;
          const uid  = unitIds[flat];
          if (!uid || !stats[uid]) continue;
          const elev = O.y + iz * ch + ch * 0.5;
          const s = stats[uid];
          s.count++;
          s.certSum += certainty[flat];
          s.minElev = Math.min(s.minElev, elev - ch * 0.5);
          s.maxElev = Math.max(s.maxElev, elev + ch * 0.5);
          if (colTop[uid] === undefined) colTop[uid] = elev + ch * 0.5;
        }
        for (const [uid, e] of Object.entries(colTop)) {
          topElev[uid].sum += e; topElev[uid].count++;
        }
      }
    }

    const totalVol = AppState.geoUnits.reduce((a, u) => a + (stats[u.id]?.count ?? 0) * cellVol, 0);
    const dateStr = new Date().toLocaleDateString('en-GB', { day:'2-digit', month:'short', year:'numeric' });

    const rows = AppState.geoUnits.map(u => {
      const s = stats[u.id];
      if (!s?.count) return '';
      const vol   = s.count * cellVol;
      const gamma = u.params?.gamma;
      const mass  = gamma ? (vol * gamma * 0.1).toFixed(0) : '—'; // γ kN/m³ → t/m³ ≈ γ/9.81, mass = vol × density
      const massT = gamma ? `${(vol * gamma / 9.81).toFixed(0)} t` : '—';
      const area  = topElev[u.id].count > 0 ? (topElev[u.id].count * cs * cs).toFixed(0) : '—';
      const thick = (s.maxElev !== -Infinity && s.minElev !== Infinity)
        ? (s.maxElev - s.minElev).toFixed(1) : '—';
      const cert  = (s.certSum / s.count * 100).toFixed(0);
      const pct   = totalVol > 0 ? (vol / totalVol * 100).toFixed(1) : '0';
      return `<tr>
        <td><span style="display:inline-block;width:12px;height:12px;border-radius:2px;background:${u.color};margin-right:5px;vertical-align:middle"></span>${u.code}</td>
        <td>${u.name ?? ''}</td>
        <td style="text-align:right;font-family:monospace">${vol >= 1e6 ? (vol/1e6).toFixed(2)+' Mm³' : vol >= 1000 ? (vol/1000).toFixed(0)+' k m³' : vol.toFixed(0)+' m³'}</td>
        <td style="text-align:right;font-family:monospace">${massT}</td>
        <td style="text-align:right;font-family:monospace">${area} m²</td>
        <td style="text-align:right;font-family:monospace">${thick} m</td>
        <td style="text-align:right">${cert}%</td>
        <td style="text-align:right">${pct}%</td>
      </tr>`;
    }).filter(Boolean).join('');

    const html = `<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Material Quantity Takeoff</title>
<style>
  body { font-family: Arial,sans-serif; font-size: 12px; color: #222; margin: 20px; }
  h1 { font-size: 18px; margin-bottom: 4px; }
  .meta { font-size: 11px; color: #666; margin-bottom: 16px; }
  table { border-collapse: collapse; width: 100%; margin-top: 8px; }
  th, td { border: 1px solid #ccc; padding: 5px 8px; font-size: 11px; }
  th { background: #f0f4f8; font-weight: 600; text-align: center; }
  tr:nth-child(even) { background: #f8fafb; }
  .footer { margin-top: 16px; font-size: 10px; color: #888; }
  @media print { @page { margin: 1cm; } }
</style>
</head><body>
<h1>Material Quantity Takeoff</h1>
<div class="meta">Generated: ${dateStr} &nbsp;|&nbsp; Grid: ${nx}×${ny}×${nz} cells &nbsp;|&nbsp; Cell: ${cs}m × ${cs}m × ${ch}m &nbsp;|&nbsp; Total model volume: ${(totalVol/1000).toFixed(0)} k m³</div>
<table>
  <thead><tr>
    <th>Code</th><th>Unit Name</th><th>Volume</th><th>Mass*</th>
    <th>Plan Area</th><th>Thickness Range</th><th>Mean Certainty</th><th>% Model</th>
  </tr></thead>
  <tbody>${rows}</tbody>
</table>
<div class="footer">* Mass calculated from unit weight γ (kN/m³) where assigned; — where not assigned.<br>
Volume from voxel model. Certainty from classification confidence. Plan area = footprint of unit at model top surface.</div>
</body></html>`;

    downloadBlob(new Blob([html], { type: 'text/html' }),
      `quantity-takeoff-${new Date().toISOString().slice(0,10)}.html`);
    log(`Material quantity takeoff exported — ${AppState.geoUnits.filter(u => stats[u.id]?.count).length} units.`, 'ok');
  });

  // ── Export unit properties as CSV ─────────────────────────────────────────
  document.getElementById('btn-export-props')?.addEventListener('click', () => {
    if (!AppState.geoUnits.length) { log('No units to export.', 'warn'); return; }
    exportPropertiesCSV(AppState.geoUnits);
    log('Unit properties CSV exported.', 'ok');
  });

  // ── Export formation surfaces as OBJ (one surface per unit) ─────────────
  document.getElementById('btn-export-surfaces')?.addEventListener('click', () => {
    const grid = AppState.voxelGrid;
    if (!grid) { log('No voxel grid to export.', 'warn'); return; }
    const { nx, ny, nz, cellSize: cs, cellHeight: ch, origin: O, unitIds } = grid;
    const unitById = {};
    AppState.geoUnits.forEach(u => { unitById[u.id] = u; });

    // For each unit build a grid of top elevations (NaN = absent)
    const unitList = AppState.geoUnits.filter(u => u.id);
    const objParts = [];
    let vtxOffset = 0;

    for (const unit of unitList) {
      // topZ[iy][ix] = elevation of topmost voxel of this unit
      const topZ = Array.from({ length: ny }, () => new Float32Array(nx).fill(NaN));
      for (let iy = 0; iy < ny; iy++) {
        for (let ix = 0; ix < nx; ix++) {
          for (let iz = nz - 1; iz >= 0; iz--) {
            if (unitIds[ix + iy * nx + iz * nx * ny] === unit.id) {
              topZ[iy][ix] = O.y + (iz + 1) * ch; // top of voxel
              break;
            }
          }
        }
      }

      // Triangulate the surface using a simple grid mesh
      // Skip cells where either corner is absent
      const verts = [];
      const faces = [];

      for (let iy = 0; iy < ny; iy++) {
        for (let ix = 0; ix < nx; ix++) {
          const z  = topZ[iy][ix];
          const wx = O.x + ix * cs + cs * 0.5;
          const wy = O.z + iy * cs + cs * 0.5;
          verts.push([wx, wy, isNaN(z) ? O.y : z, isNaN(z)]);
        }
      }

      // Build quad faces only where all 4 corners have data
      for (let iy = 0; iy < ny - 1; iy++) {
        for (let ix = 0; ix < nx - 1; ix++) {
          const i00 = iy * nx + ix;
          const i10 = iy * nx + ix + 1;
          const i01 = (iy + 1) * nx + ix;
          const i11 = (iy + 1) * nx + ix + 1;
          if (verts[i00][3] || verts[i10][3] || verts[i01][3] || verts[i11][3]) continue;
          faces.push([i00, i10, i11]);
          faces.push([i00, i11, i01]);
        }
      }

      if (!faces.length) continue;

      const lines = [];
      lines.push(`# GeoModel AI — Formation surface: ${unit.code} ${unit.name}`);
      lines.push(`g ${unit.code}`);
      for (const [x, y, z] of verts) {
        lines.push(`v ${x.toFixed(3)} ${z.toFixed(3)} ${(-y).toFixed(3)}`); // OBJ: Y-up
      }
      for (const [a, b, c] of faces) {
        lines.push(`f ${a + 1 + vtxOffset} ${b + 1 + vtxOffset} ${c + 1 + vtxOffset}`);
      }
      vtxOffset += verts.length;
      objParts.push(lines.join('\n'));
    }

    if (!objParts.length) { log('No formation surfaces to export.', 'warn'); return; }
    const blob = new Blob([objParts.join('\n\n')], { type: 'text/plain' });
    downloadBlob(blob, 'formation-surfaces.obj');
    log(`Formation surfaces exported — ${objParts.length} surface(s).`, 'ok');
  });

  // ── Export formation surfaces as binary STL ──────────────────────────────────
  document.getElementById('btn-export-stl')?.addEventListener('click', () => {
    const grid = AppState.voxelGrid;
    if (!grid) { log('Build the 3D model first.', 'warn'); return; }
    const { nx, ny, nz, cellSize: cs, cellHeight: ch, origin: O, unitIds } = grid;
    log('Building binary STL triangles…', 'info');

    const triangles = []; // {n:[nx,ny,nz], v0, v1, v2}

    const smooth2D = (src) => {
      const dst = new Float32Array(nx * ny);
      for (let iy = 0; iy < ny; iy++) {
        for (let ix = 0; ix < nx; ix++) {
          if (isNaN(src[ix + iy * nx])) { dst[ix + iy * nx] = NaN; continue; }
          let sum = 0, w = 0;
          for (let dy = -1; dy <= 1; dy++) {
            for (let dx = -1; dx <= 1; dx++) {
              const jx = ix+dx, jy = iy+dy;
              if (jx < 0 || jx >= nx || jy < 0 || jy >= ny) continue;
              const v = src[jx + jy * nx];
              if (isNaN(v)) continue;
              const wt = (dx===0&&dy===0)?4:(dx!==0&&dy!==0)?1:2;
              sum += v * wt; w += wt;
            }
          }
          dst[ix + iy * nx] = w > 0 ? sum / w : NaN;
        }
      }
      return dst;
    };

    const triNorm = (v0,v1,v2) => {
      const ax=v1[0]-v0[0], ay=v1[1]-v0[1], az=v1[2]-v0[2];
      const bx=v2[0]-v0[0], by=v2[1]-v0[1], bz=v2[2]-v0[2];
      const cx=ay*bz-az*by, cy=az*bx-ax*bz, cz=ax*by-ay*bx;
      const l=Math.sqrt(cx*cx+cy*cy+cz*cz)||1;
      return [cx/l, cy/l, cz/l];
    };

    for (const unit of AppState.geoUnits) {
      if (!unit.id) continue;
      const topZ = new Float32Array(nx * ny).fill(NaN);
      for (let iy = 0; iy < ny; iy++)
        for (let ix = 0; ix < nx; ix++)
          for (let iz = nz - 1; iz >= 0; iz--)
            if (unitIds[ix + iy * nx + iz * nx * ny] === unit.id) {
              topZ[ix + iy * nx] = O.y + (iz + 1) * ch; break;
            }

      const smoothed = smooth2D(smooth2D(topZ));

      for (let iy = 0; iy < ny - 1; iy++) {
        for (let ix = 0; ix < nx - 1; ix++) {
          const z00=smoothed[ix   + iy   *nx], z10=smoothed[(ix+1)+ iy   *nx];
          const z01=smoothed[ix   +(iy+1)*nx], z11=smoothed[(ix+1)+(iy+1)*nx];
          if ([z00,z10,z01,z11].some(isNaN)) continue;
          // STL: X=Easting, Y=elevation mAOD, Z=Northing
          const x0=O.x+ix*cs+cs/2, x1=O.x+(ix+1)*cs+cs/2;
          const n0=O.z+iy*cs+cs/2, n1=O.z+(iy+1)*cs+cs/2;
          const v00=[x0,z00,n0], v10=[x1,z10,n0], v01=[x0,z01,n1], v11=[x1,z11,n1];
          triangles.push({ n:triNorm(v00,v10,v01), v0:v00, v1:v10, v2:v01 });
          triangles.push({ n:triNorm(v10,v11,v01), v0:v10, v1:v11, v2:v01 });
        }
      }
    }

    if (!triangles.length) { log('No surface triangles to export.', 'warn'); return; }
    const buf  = new ArrayBuffer(80 + 4 + triangles.length * 50);
    const view = new DataView(buf);
    const u8   = new Uint8Array(buf);
    const hdr  = `GeoModel AI formation surfaces — ${AppState.geoUnits.length} units`;
    for (let i = 0; i < 80; i++) u8[i] = i < hdr.length ? hdr.charCodeAt(i) : 0x20;
    view.setUint32(80, triangles.length, true);
    let off = 84;
    for (const { n, v0, v1, v2 } of triangles) {
      const f = (v) => { view.setFloat32(off,v,true); off+=4; };
      n.forEach(f); v0.forEach(f); v1.forEach(f); v2.forEach(f);
      view.setUint16(off, 0, true); off += 2;
    }
    downloadBlob(new Blob([buf], { type: 'model/stl' }), 'formation-surfaces.stl');
    log(`Binary STL exported — ${triangles.length.toLocaleString()} triangles.`, 'ok');
  });

  // ── Export classified BH data as AGS 4.x ────────────────────────────────
  document.getElementById('btn-export-ags')?.addEventListener('click', () => {
    const bhs = AppState.classifiedBH?.filter(b => !b.synthetic);
    if (!bhs?.length) { log('No borehole data to export.', 'warn'); return; }
    const lines = [];
    const q = v => `"${String(v ?? '').replace(/"/g, '""')}"`;
    const d = v => v != null ? `"${parseFloat(v).toFixed(3)}"` : '""';

    // AGS 4.x file header
    lines.push('"GROUP","PROJ"');
    lines.push('"HEADING","PROJ_ID","PROJ_NAME","PROJ_LOC","PROJ_CLNT","PROJ_CONT","PROJ_ENG","PROJ_DATE"');
    lines.push('"UNIT","","","","","","","yyyy-mm-dd"');
    const today = new Date().toISOString().slice(0, 10);
    lines.push(`"DATA","GeoModel AI Export","","","","","",${q(today)}`);
    lines.push('');

    // TRAN group (file transmission)
    lines.push('"GROUP","TRAN"');
    lines.push('"HEADING","TRAN_RTYP","TRAN_PROD","TRAN_VER","TRAN_STAT","TRAN_DESC","TRAN_AGS","TRAN_RECV","TRAN_DTTM"');
    lines.push('"UNIT","","","","","","","",""');
    lines.push(`"DATA","DATA","GeoModel AI","1.0","FINAL","Classified ground investigation data","4.0","","${today}T00:00:00"`);
    lines.push('');

    // LOCA group — borehole locations
    lines.push('"GROUP","LOCA"');
    lines.push('"HEADING","LOCA_ID","LOCA_TYPE","LOCA_STAT","LOCA_NATE","LOCA_NATN","LOCA_GREF","LOCA_GL","LOCA_FDEP","LOCA_STAR","LOCA_PURP","LOCA_TERM","LOCA_ENDD"');
    lines.push('"UNIT","","","","m","m","","m","m","yyyy-mm-dd","","",""');
    bhs.forEach(bh => {
      const dep = bh.depth ?? (bh.layers.length ? Math.max(...bh.layers.map(l => l.base)) : 0);
      lines.push([
        '"DATA"', q(bh.id), '"BH"', '"S"',
        d(bh.x), d(bh.y), '"OSGB36"', d(bh.groundLevel),
        d(dep), q(today), '"Geotechnical investigation"', '"GND"', '""',
      ].join(','));
    });
    lines.push('');

    // GEOL group — geology layers (classified)
    lines.push('"GROUP","GEOL"');
    lines.push('"HEADING","LOCA_ID","GEOL_TOP","GEOL_BASE","GEOL_DESC","GEOL_LEG","GEOL_UNIT","GEOL_CERT"');
    lines.push('"UNIT","","m","m","","","",""');
    bhs.forEach(bh => {
      bh.layers.forEach(l => {
        const unit = AppState.geoUnits.find(u => u.code === l.unitCode);
        lines.push([
          '"DATA"', q(bh.id), d(l.top), d(l.base),
          q(l.description ?? ''), q(l.unitCode ?? ''),
          q(unit?.name ?? ''), d(l.certainty ?? ''),
        ].join(','));
      });
    });
    lines.push('');

    // ISPT group — SPT data where present
    const hasN = bhs.some(bh => bh.layers.some(l => l.sptN != null));
    if (hasN) {
      lines.push('"GROUP","ISPT"');
      lines.push('"HEADING","LOCA_ID","ISPT_TOP","ISPT_NVAL","ISPT_REP","ISPT_COR"');
      lines.push('"UNIT","","m","","",""');
      bhs.forEach(bh => {
        bh.layers.forEach(l => {
          if (l.sptN == null) return;
          const mid = ((l.top ?? 0) + (l.base ?? 0)) / 2;
          lines.push(['"DATA"', q(bh.id), d(mid), d(l.sptN), '"Y"', '""'].join(','));
        });
      });
      lines.push('');
    }

    const ags = lines.join('\r\n');
    downloadBlob(new Blob([ags], { type: 'text/plain;charset=utf-8' }),
      `geomodel-export-${today}.ags`);
    log(`AGS 4.x exported — ${bhs.length} borehole(s).`, 'ok');
  });

  // ── Export borehole logs as CSV ──────────────────────────────────────────
  document.getElementById('btn-export-bh-csv')?.addEventListener('click', () => {
    const bhs = AppState.classifiedBH?.filter(b => !b.synthetic);
    if (!bhs?.length) { log('No borehole data to export.', 'warn'); return; }
    const unitByCode = {};
    AppState.geoUnits.forEach(u => { unitByCode[u.code] = u; });
    const rows = ['BH_ID,X,Y,Ground_Level_mAOD,Depth_From_m,Depth_To_m,Unit_Code,Unit_Name,Description,Certainty'];
    bhs.forEach(bh => {
      bh.layers.forEach(l => {
        const u = unitByCode[l.unitCode] ?? {};
        rows.push([
          bh.id, bh.x?.toFixed(2), bh.y?.toFixed(2),
          bh.groundLevel?.toFixed(2),
          l.top?.toFixed(2), l.base?.toFixed(2),
          l.unitCode ?? '',
          `"${(u.name ?? '').replace(/"/g, '""')}"`,
          `"${(l.description ?? '').replace(/"/g, '""')}"`,
          (l.certainty ?? '').toString(),
        ].join(','));
      });
    });
    downloadBlob(new Blob([rows.join('\n')], { type: 'text/csv' }), 'borehole-logs.csv');
    log(`BH CSV exported — ${bhs.length} boreholes.`, 'ok');
  });

  // ── Export each borehole as LAS 2.0 (combined into a single text archive) ──────
  document.getElementById('btn-export-las')?.addEventListener('click', () => {
    const bhs = AppState.classifiedBH?.filter(b => !b.synthetic);
    if (!bhs?.length) { log('No borehole data to export.', 'warn'); return; }
    const unitByCode = {};
    AppState.geoUnits.forEach(u => { unitByCode[u.code] = u; });

    const today = new Date().toISOString().slice(0, 10);
    const allParts = [];

    bhs.forEach(bh => {
      if (!bh.layers?.length) return;
      const gl     = bh.groundLevel ?? 0;
      const maxD   = Math.max(...bh.layers.map(l => l.base ?? 0));
      const STEP   = 0.5; // 0.5m depth samples
      const nRows  = Math.ceil(maxD / STEP) + 1;

      // Interpolate unit code (as integer index), certainty, and SPT N at each sample
      const unitIdx = (depth) => {
        const layer = bh.layers.find(l => depth >= l.top && depth < l.base);
        if (!layer) return -999.25;
        const idx = AppState.geoUnits.findIndex(u => u.code === layer.unitCode);
        return idx >= 0 ? idx + 1 : -999.25;
      };
      const certAt = (depth) => {
        const layer = bh.layers.find(l => depth >= l.top && depth < l.base);
        return layer?.certainty != null ? layer.certainty.toFixed(3) : '-999.25';
      };
      const sptAt = (depth) => {
        const layer = bh.layers.find(l => depth >= l.top && depth < l.base);
        return layer?.sptN != null ? layer.sptN.toFixed(1) : '-999.25';
      };

      let las = `~VERSION INFORMATION\n`;
      las += ` VERS.                2.0   : LAS Version 2.0\n`;
      las += ` WRAP.                NO    : One line per depth step\n`;
      las += `~WELL INFORMATION\n`;
      las += ` STRT.M               ${(0).toFixed(2)} : Start depth\n`;
      las += ` STOP.M               ${maxD.toFixed(2)} : Stop depth\n`;
      las += ` STEP.M               ${STEP.toFixed(2)} : Depth step\n`;
      las += ` NULL.               -999.25 : Null value\n`;
      las += ` COMP.                GeoModel AI\n`;
      las += ` WELL.                ${bh.id}\n`;
      las += ` FLD.                 SITE\n`;
      las += ` LOC.                 X:${bh.x?.toFixed(2) ?? 0} Y:${bh.y?.toFixed(2) ?? 0}\n`;
      las += ` SRVC.                GeoModel AI\n`;
      las += ` DATE.                ${today}\n`;
      las += ` API.                 00-000-00000-00\n`;
      las += `~CURVE INFORMATION\n`;
      las += ` DEPT.M               : Depth (m below GL)\n`;
      las += ` ELEV.M               : Elevation (mAOD)\n`;
      las += ` UNIT.                : Geological unit index (1-based; 0=no data)\n`;
      las += ` CERT.                : Classification certainty (0–1)\n`;
      las += ` SPTN.                : SPT N value (blows/300mm)\n`;
      las += `~PARAMETER INFORMATION\n`;
      las += ` KB.M                 ${gl.toFixed(2)} : Ground level (mAOD)\n`;
      las += `# Unit index map:\n`;
      AppState.geoUnits.forEach((u, i) => {
        las += `# ${i + 1} = ${u.code} (${u.name})\n`;
      });
      las += `~A  DEPT       ELEV       UNIT  CERT     SPTN\n`;

      for (let i = 0; i < nRows; i++) {
        const depth = i * STEP;
        const elev  = gl - depth;
        las += ` ${depth.toFixed(2).padStart(8)}  ${elev.toFixed(2).padStart(8)}  ${String(unitIdx(depth)).padStart(4)}  ${certAt(depth).padStart(6)}  ${sptAt(depth).padStart(8)}\n`;
      }

      allParts.push({ name: `${bh.id}.las`, content: las });
    });

    if (!allParts.length) { log('No borehole layers to export.', 'warn'); return; }

    // Combine all LAS files into a single text file with clear separators
    const combined = allParts.map(p =>
      `${'='.repeat(60)}\n# FILE: ${p.name}\n${'='.repeat(60)}\n${p.content}`
    ).join('\n\n');

    downloadBlob(new Blob([combined], { type: 'text/plain' }), `boreholes-las2.txt`);
    log(`LAS 2.0 exported — ${allParts.length} borehole(s) in combined text file.`, 'ok');
  });

  // ── Export full block model as CSV (Leapfrog / Vulcan / Datamine compatible) ──
  document.getElementById('btn-export-blockmodel')?.addEventListener('click', () => {
    const grid = AppState.voxelGrid;
    if (!grid) { log('Build the 3D model first.', 'warn'); return; }
    const { nx, ny, nz, cellSize: cs, cellHeight: ch, origin: O, unitIds, certainty } = grid;
    const unitById = {};
    AppState.geoUnits.forEach(u => { unitById[u.id] = u; });

    log('Generating block model CSV…', 'info');
    const rows = ['CENTROID_X,CENTROID_Y,CENTROID_Z_mAOD,UNIT_CODE,UNIT_NAME,CERTAINTY,Cu_kPa,PHI_deg,Cc,E_MPa,GAMMA_kNm3,N_SPT'];

    for (let iz = 0; iz < nz; iz++) {
      const wz = O.y + iz * ch + ch * 0.5;   // elevation (mAOD)
      for (let iy = 0; iy < ny; iy++) {
        const wy = O.z + iy * cs + cs * 0.5;
        for (let ix = 0; ix < nx; ix++) {
          const wx   = O.x + ix * cs + cs * 0.5;
          const flat = ix + iy * nx + iz * nx * ny;
          const uid  = unitIds[flat];
          const unit = unitById[uid];
          if (!uid) continue;
          const cert = certainty[flat].toFixed(3);
          const p    = unit?.params ?? {};
          rows.push([
            wx.toFixed(2), wy.toFixed(2), wz.toFixed(2),
            unit?.code ?? '',
            `"${(unit?.name ?? '').replace(/"/g, '""')}"`,
            cert,
            p.cu    != null ? p.cu.toFixed(1)    : '',
            p.phi   != null ? p.phi.toFixed(1)   : '',
            p.Cc    != null ? p.Cc.toFixed(4)    : '',
            p.E     != null ? p.E.toFixed(1)     : '',
            p.gamma != null ? p.gamma.toFixed(2) : '',
            p.N_spt != null ? p.N_spt.toFixed(0) : '',
          ].join(','));
        }
      }
    }

    const total = nx * ny * nz;
    downloadBlob(new Blob([rows.join('\n')], { type: 'text/csv' }),
      `block-model-${new Date().toISOString().slice(0,10)}.csv`);
    log(`Block model CSV exported — ${total.toLocaleString()} cells.`, 'ok');
  });

  // ── GeoJSON plan-slice export ──────────────────────────────────────────────
  document.getElementById('btn-export-geojson')?.addEventListener('click', () => {
    const grid = AppState.voxelGrid;
    if (!grid) { log('Build the 3D model first.', 'warn'); return; }
    const sliceDepth = parseFloat(document.getElementById('geojson-depth')?.value ?? '5');
    const { nx, ny, nz, unitIds, certainty, cellSize: cs, cellHeight: ch, origin: O } = grid;

    const unitById = {};
    AppState.geoUnits.forEach(u => { unitById[u.id] = u; });

    // Find the Z slice corresponding to the requested depth
    // Depth from ground: need the top ground elevation
    const maxElev = AppState.classifiedBH.reduce((m, b) => Math.max(m, b.groundLevel ?? 0), O.y + nz * ch);
    const targetElev = maxElev - sliceDepth;
    const iz = Math.max(0, Math.min(nz - 1, Math.round((targetElev - O.y) / ch)));
    const actualDepth = (maxElev - (O.y + iz * ch + ch * 0.5)).toFixed(1);

    const features = [];
    for (let iy = 0; iy < ny; iy++) {
      for (let ix = 0; ix < nx; ix++) {
        const flat = ix + iy * nx + iz * nx * ny;
        const uid  = unitIds[flat];
        if (!uid) continue;
        const unit = unitById[uid];
        const cert = certainty[flat];
        // Cell corners (local project coordinates)
        const x0 = O.x + ix * cs, x1 = x0 + cs;
        const y0 = O.z + iy * cs, y1 = y0 + cs;
        // GeoJSON uses [longitude, latitude] → here we use [easting, northing]
        features.push({
          type: 'Feature',
          geometry: {
            type: 'Polygon',
            coordinates: [[[x0,y0],[x1,y0],[x1,y1],[x0,y1],[x0,y0]]],
          },
          properties: {
            unit_code: unit?.code ?? '',
            unit_name: unit?.name ?? '',
            unit_color: unit?.color ?? '#888888',
            certainty: Math.round(cert * 100),
            depth_m: parseFloat(actualDepth),
            elev_mAOD: parseFloat((O.y + iz * ch + ch * 0.5).toFixed(2)),
          },
        });
      }
    }

    const geojson = { type: 'FeatureCollection', features };
    downloadBlob(
      new Blob([JSON.stringify(geojson, null, 0)], { type: 'application/json' }),
      `geomodel-slice-${actualDepth}m-${new Date().toISOString().slice(0,10)}.geojson`
    );
    log(`GeoJSON exported — ${features.length} cells at ${actualDepth} m bgl.`, 'ok');
  });

  // ── GeoJSON formation tops export ─────────────────────────────────────────
  document.getElementById('btn-export-geojson-tops')?.addEventListener('click', () => {
    const bhs = AppState.classifiedBH.filter(b => !b.synthetic && b.layers?.length);
    if (!bhs.length) { log('Load borehole data first.', 'warn'); return; }

    const unitById = {};
    AppState.geoUnits.forEach(u => { unitById[u.code] = u; });

    const features = [];
    for (const bh of bhs) {
      for (const layer of bh.layers) {
        const unit = unitById[layer.unitCode];
        const topElev = (bh.groundLevel ?? 0) - layer.top;
        const baseElev = (bh.groundLevel ?? 0) - layer.base;
        features.push({
          type: 'Feature',
          geometry: {
            type: 'Point',
            coordinates: [bh.x, bh.y, topElev],
          },
          properties: {
            borehole_id: bh.id,
            unit_code: layer.unitCode ?? '',
            unit_name: unit?.name ?? '',
            unit_color: unit?.color ?? '#888888',
            depth_to_top_m: parseFloat(layer.top.toFixed(2)),
            depth_to_base_m: parseFloat(layer.base.toFixed(2)),
            thickness_m: parseFloat((layer.base - layer.top).toFixed(2)),
            top_elev_mAOD: parseFloat(topElev.toFixed(2)),
            base_elev_mAOD: parseFloat(baseElev.toFixed(2)),
            certainty: layer.certainty ?? null,
          },
        });
      }
    }

    const geojson = { type: 'FeatureCollection', features };
    downloadBlob(
      new Blob([JSON.stringify(geojson, null, 0)], { type: 'application/json' }),
      `formation-tops-${new Date().toISOString().slice(0,10)}.geojson`
    );
    log(`Formation tops GeoJSON exported — ${features.length} layer intervals from ${bhs.length} boreholes.`, 'ok');
  });
}
