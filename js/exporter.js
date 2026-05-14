import { GLTFExporter } from 'three/addons/exporters/GLTFExporter.js';
import { OBJExporter  } from 'three/addons/exporters/OBJExporter.js';
import { AppState, log } from './app.js';

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
    if (!AppState.voxelGrid) {
      log('No voxel grid to export.', 'warn');
      return;
    }
    log('Exporting JSON voxel data…', 'info');
    const { nx, ny, nz, cellSize, cellHeight, origin } = AppState.voxelGrid;
    const payload = {
      meta: {
        nx, ny, nz, cellSize, cellHeight, origin,
        units: AppState.geoUnits,
      },
      unitIds:   Array.from(AppState.voxelGrid.unitIds),
      certainty: Array.from(AppState.voxelGrid.certainty),
    };
    const blob = new Blob([JSON.stringify(payload)], { type: 'application/json' });
    downloadBlob(blob, 'geomodel-voxels.json');
    log('JSON exported.', 'ok');
  });
}
