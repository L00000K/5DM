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
  document.getElementById('btn-export-contacts')?.addEventListener('click', () => {
    const grid = AppState.voxelGrid;
    if (!grid) { log('No voxel grid to export.', 'warn'); return; }
    const { nx, ny, nz, cellSize: cs, cellHeight: ch, origin: O, unitIds } = grid;
    const unitById = {};
    AppState.geoUnits.forEach(u => { unitById[u.id] = u; });

    const rows = ['Unit_Code,Unit_Name,IX,IY,X,Y,Top_Z_mAOD'];
    for (let iy = 0; iy < ny; iy++) {
      for (let ix = 0; ix < nx; ix++) {
        // Per unit: find topmost voxel (highest iz)
        const topByUnit = {};
        for (let iz = nz - 1; iz >= 0; iz--) {
          const uid = unitIds[ix + iy * nx + iz * nx * ny];
          if (uid && topByUnit[uid] === undefined) topByUnit[uid] = iz;
        }
        for (const [uid, iz] of Object.entries(topByUnit)) {
          const unit = unitById[uid];
          if (!unit) continue;
          const x   = (O.x + (ix + 0.5) * cs).toFixed(2);
          const y   = (O.z + (iy + 0.5) * cs).toFixed(2);
          const top = (O.y + (parseInt(iz) + 1) * ch).toFixed(2);
          rows.push([
            unit.code,
            `"${(unit.name ?? '').replace(/"/g, '""')}"`,
            ix, iy, x, y, top,
          ].join(','));
        }
      }
    }
    downloadBlob(new Blob([rows.join('\n')], { type: 'text/csv' }), 'formation-contacts.csv');
    log(`Formation contacts exported — ${rows.length - 1} contact points.`, 'ok');
  });

  // ── Export unit properties as CSV ─────────────────────────────────────────
  document.getElementById('btn-export-props')?.addEventListener('click', () => {
    if (!AppState.geoUnits.length) { log('No units to export.', 'warn'); return; }
    exportPropertiesCSV(AppState.geoUnits);
    log('Unit properties CSV exported.', 'ok');
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
}
