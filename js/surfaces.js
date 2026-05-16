import * as THREE from 'three';
import { marchingCubes, smooth3D } from './marching-cubes.js';

export class SurfaceManager {
  constructor(scene) {
    this._scene   = scene;
    this._meshes  = {};       // top-surface meshes
    this._mcMeshes = {};      // marching-cubes isosurface meshes
    this._visible = false;
    this._mcVisible = false;
  }

  build(grid, geoUnits) {
    this.clear();

    const { nx, ny, nz, cellSize: cs, cellHeight: ch, origin, unitIds } = grid;
    const n2 = nx * ny;

    for (const unit of geoUnits) {
      const elev = new Float32Array(n2).fill(NaN);

      for (let iy = 0; iy < ny; iy++) {
        for (let ix = 0; ix < nx; ix++) {
          for (let iz = nz - 1; iz >= 0; iz--) {
            if (unitIds[ix + iy * nx + iz * n2] === unit.id) {
              elev[ix + iy * nx] = origin.y + iz * ch + ch;
              break;
            }
          }
        }
      }

      if (elev.every(v => isNaN(v))) continue;

      // 2-pass weighted (3×3 box) smoothing
      const smooth = (src) => {
        const dst = new Float32Array(n2);
        for (let iy = 0; iy < ny; iy++) {
          for (let ix = 0; ix < nx; ix++) {
            if (isNaN(src[ix + iy * nx])) { dst[ix + iy * nx] = NaN; continue; }
            let sum = 0, w = 0;
            for (let dy = -1; dy <= 1; dy++) {
              for (let dx = -1; dx <= 1; dx++) {
                const jx = ix + dx, jy = iy + dy;
                if (jx < 0 || jx >= nx || jy < 0 || jy >= ny) continue;
                const v = src[jx + jy * nx];
                if (isNaN(v)) continue;
                const wt = (dx === 0 && dy === 0) ? 4 : (dx !== 0 && dy !== 0) ? 1 : 2;
                sum += v * wt; w += wt;
              }
            }
            dst[ix + iy * nx] = w > 0 ? sum / w : NaN;
          }
        }
        return dst;
      };

      const smoothed = smooth(smooth(elev));

      // Build vertex index map (only valid cells)
      const vertIdx = new Int32Array(n2).fill(-1);
      let vCount = 0;
      for (let i = 0; i < n2; i++) {
        if (!isNaN(smoothed[i])) vertIdx[i] = vCount++;
      }
      if (vCount === 0) continue;

      const positions = new Float32Array(vCount * 3);
      for (let iy = 0; iy < ny; iy++) {
        for (let ix = 0; ix < nx; ix++) {
          const vi = vertIdx[ix + iy * nx];
          if (vi < 0) continue;
          positions[vi * 3]     = origin.x + ix * cs + cs * 0.5;
          positions[vi * 3 + 1] = smoothed[ix + iy * nx];
          positions[vi * 3 + 2] = origin.z + iy * cs + cs * 0.5;
        }
      }

      const indices = [];
      for (let iy = 0; iy < ny - 1; iy++) {
        for (let ix = 0; ix < nx - 1; ix++) {
          const a = vertIdx[ix       + iy       * nx];
          const b = vertIdx[(ix + 1) + iy       * nx];
          const c = vertIdx[ix       + (iy + 1) * nx];
          const d = vertIdx[(ix + 1) + (iy + 1) * nx];
          if (a < 0 || b < 0 || c < 0 || d < 0) continue;
          indices.push(a, c, b, b, c, d);
        }
      }
      if (!indices.length) continue;

      const geom = new THREE.BufferGeometry();
      geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      geom.setIndex(indices);
      geom.computeVertexNormals();

      const color = new THREE.Color(unit.color).convertSRGBToLinear();
      const mat = new THREE.MeshLambertMaterial({
        color,
        transparent: true,
        opacity: 0.55,
        side: THREE.DoubleSide,
        depthWrite: false,
      });

      const mesh = new THREE.Mesh(geom, mat);
      mesh.userData.unitCode = unit.code;
      mesh.visible = false;
      this._meshes[unit.code] = mesh;
      this._scene.add(mesh);
    }
  }

  // ── Marching-cubes isosurface build ─────────────────────────────────────────
  // Called asynchronously — pass onProgress(0..1) for progress feedback.
  buildIsosurfaces(grid, geoUnits, opacity = 0.6, onProgress = null) {
    this._clearMC();
    const { nx, ny, nz, cellSize: cs, cellHeight: ch, origin, unitIds } = grid;
    const total = nx * ny * nz;

    let done = 0;
    for (const unit of geoUnits) {
      // Binary scalar field: 1.0 inside this unit, 0.0 outside
      const field = new Float32Array(total);
      for (let i = 0; i < total; i++) field[i] = unitIds[i] === unit.id ? 1.0 : 0.0;

      // 2-pass smoothing to get a gradient transition at boundaries
      const smoothed = smooth3D(smooth3D(field, nx, ny, nz, 1), nx, ny, nz, 1);

      const pos = marchingCubes(smoothed, nx, ny, nz, 0.5, origin, cs, ch);
      if (!pos.length) { done++; onProgress?.(done / geoUnits.length); continue; }

      const geom = new THREE.BufferGeometry();
      geom.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      geom.computeVertexNormals();

      const color = new THREE.Color(unit.color).convertSRGBToLinear();
      const mat = new THREE.MeshLambertMaterial({
        color,
        transparent: true,
        opacity,
        side: THREE.DoubleSide,
        depthWrite: false,
      });

      const mesh = new THREE.Mesh(geom, mat);
      mesh.userData.unitCode = unit.code;
      mesh.visible = this._mcVisible;
      this._mcMeshes[unit.code] = mesh;
      this._scene.add(mesh);

      done++;
      onProgress?.(done / geoUnits.length);
    }
  }

  setMCVisible(v) {
    this._mcVisible = v;
    for (const mesh of Object.values(this._mcMeshes)) mesh.visible = v;
  }

  setMCUnitVisible(code, v) {
    const mesh = this._mcMeshes[code];
    if (mesh) mesh.visible = this._mcVisible && v;
  }

  setMCOpacity(op) {
    for (const mesh of Object.values(this._mcMeshes)) mesh.material.opacity = op;
  }

  _clearMC() {
    for (const mesh of Object.values(this._mcMeshes)) {
      this._scene.remove(mesh);
      mesh.geometry.dispose();
      mesh.material.dispose();
    }
    this._mcMeshes = {};
  }

  setVisible(v) {
    this._visible = v;
    for (const mesh of Object.values(this._meshes)) {
      mesh.visible = v;
    }
  }

  setUnitVisible(code, v) {
    const mesh = this._meshes[code];
    if (mesh) mesh.visible = this._visible && v;
  }

  setOpacity(op) {
    for (const mesh of Object.values(this._meshes)) {
      mesh.material.opacity = op;
    }
  }

  getMeshes() {
    return [...Object.values(this._meshes), ...Object.values(this._mcMeshes)];
  }

  clear() {
    for (const mesh of Object.values(this._meshes)) {
      this._scene.remove(mesh);
      mesh.geometry.dispose();
      mesh.material.dispose();
    }
    this._meshes = {};
    this._clearMC();
  }
}
