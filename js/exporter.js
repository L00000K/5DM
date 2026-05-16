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
}
