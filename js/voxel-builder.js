import * as THREE from 'three';
import { voxelIndex } from './interpolator.js';

// One InstancedMesh per geological unit for performance.
// Certainty is encoded as colour brightness (full colour = 1.0, greyed = 0.0).

export class VoxelBuilder {
  constructor(scene) {
    this.scene        = scene;
    this.meshes       = {};    // unitCode → THREE.InstancedMesh
    this.unitMeta     = {};    // unitCode → { color, visible }
    this.grid         = null;
    this.units        = [];
    this.certThresh   = 0;
    this.hiddenUnits  = new Set();
    this._dummy       = new THREE.Object3D();
    this._group       = new THREE.Group();
    scene.add(this._group);
  }

  // ── Build all voxel meshes from the grid ──────────────────────────────────
  build(grid, geoUnits) {
    this.clear();
    this.grid  = grid;
    this.units = geoUnits;

    const { nx, ny, nz, cellSize, cellHeight, origin, unitIds, certainty } = grid;

    // Map unit id → unit object
    const unitById = {};
    geoUnits.forEach(u => { unitById[u.id] = u; });

    // Count instances per unit
    const counts = {};
    for (let i = 0; i < unitIds.length; i++) {
      const uid = unitIds[i];
      if (!unitById[uid]) continue;
      counts[uid] = (counts[uid] || 0) + 1;
    }

    // Create one InstancedMesh per unit
    const geom = new THREE.BoxGeometry(cellSize * 0.98, cellHeight * 0.98, cellSize * 0.98);

    for (const uid of Object.keys(counts)) {
      const unit = unitById[uid];
      if (!unit) continue;
      const mat = new THREE.MeshLambertMaterial({
        vertexColors: true,
      });
      const mesh = new THREE.InstancedMesh(geom, mat, counts[uid]);
      mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      mesh.instanceColor = new THREE.InstancedBufferAttribute(
        new Float32Array(counts[uid] * 3), 3
      );
      mesh.userData = { unitCode: unit.code, unitId: parseInt(uid) };
      this.meshes[unit.code] = mesh;
      this._group.add(mesh);
    }

    // Populate instances
    const iCounters = {};
    const baseColor = new THREE.Color();
    const greyColor = new THREE.Color(0.55, 0.55, 0.58);
    const blendColor = new THREE.Color();

    for (let iz = 0; iz < nz; iz++) {
      for (let iy = 0; iy < ny; iy++) {
        for (let ix = 0; ix < nx; ix++) {
          const idx = voxelIndex(ix, iy, iz, grid);
          const uid = unitIds[idx];
          const unit = unitById[uid];
          if (!unit) continue;

          const cert = certainty[idx];
          const code = unit.code;
          if (!this.meshes[code]) continue;

          const i = iCounters[code] = (iCounters[code] ?? 0);
          iCounters[code]++;

          // Position
          const wx = origin.x + ix * cellSize    + cellSize    * 0.5;
          const wy = origin.y + iz * cellHeight  + cellHeight  * 0.5;
          const wz = origin.z + iy * cellSize    + cellSize    * 0.5;

          this._dummy.position.set(wx, wy, wz);
          this._dummy.updateMatrix();
          this.meshes[code].setMatrixAt(i, this._dummy.matrix);

          // Colour blended by certainty: full unit colour at cert=1, grey at cert=0
          baseColor.set(unit.color);
          blendColor.r = baseColor.r * cert + greyColor.r * (1 - cert);
          blendColor.g = baseColor.g * cert + greyColor.g * (1 - cert);
          blendColor.b = baseColor.b * cert + greyColor.b * (1 - cert);
          this.meshes[code].setColorAt(i, blendColor);
        }
      }
    }

    // Mark buffers dirty
    for (const mesh of Object.values(this.meshes)) {
      mesh.instanceMatrix.needsUpdate = true;
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
      mesh.computeBoundingSphere();
    }

    return this._group;
  }

  // ── Show/hide a unit ──────────────────────────────────────────────────────
  setUnitVisibility(code, visible) {
    if (visible) this.hiddenUnits.delete(code);
    else this.hiddenUnits.add(code);
    const mesh = this.meshes[code];
    if (mesh) mesh.visible = visible;
  }

  // ── Filter by certainty threshold ─────────────────────────────────────────
  // Rebuilds instance matrices showing only voxels above threshold.
  setCertaintyThreshold(threshold) {
    this.certThresh = threshold;
    if (!this.grid) return;

    const { nx, ny, nz, cellSize, cellHeight, origin, unitIds, certainty } = this.grid;
    const unitById = {};
    this.units.forEach(u => { unitById[u.id] = u; });

    // Count new instance counts
    const counts = {};
    for (let i = 0; i < unitIds.length; i++) {
      if (certainty[i] < threshold) continue;
      const uid = unitIds[i];
      const unit = unitById[uid];
      if (!unit) continue;
      counts[unit.code] = (counts[unit.code] || 0) + 1;
    }

    // Reset per-unit counter
    const iCounters = {};

    const baseColor  = new THREE.Color();
    const greyColor  = new THREE.Color(0.55, 0.55, 0.58);
    const blendColor = new THREE.Color();

    for (let iz = 0; iz < nz; iz++) {
      for (let iy = 0; iy < ny; iy++) {
        for (let ix = 0; ix < nx; ix++) {
          const idx  = voxelIndex(ix, iy, iz, this.grid);
          const cert = certainty[idx];
          if (cert < threshold) continue;

          const uid  = unitIds[idx];
          const unit = unitById[uid];
          if (!unit || !this.meshes[unit.code]) continue;
          const code = unit.code;

          const i = iCounters[code] = (iCounters[code] ?? 0);
          iCounters[code]++;

          if (i >= this.meshes[code].count) continue;

          const wx = origin.x + ix * cellSize   + cellSize   * 0.5;
          const wy = origin.y + iz * cellHeight + cellHeight * 0.5;
          const wz = origin.z + iy * cellSize   + cellSize   * 0.5;

          this._dummy.position.set(wx, wy, wz);
          this._dummy.updateMatrix();
          this.meshes[code].setMatrixAt(i, this._dummy.matrix);

          baseColor.set(unit.color);
          blendColor.r = baseColor.r * cert + greyColor.r * (1 - cert);
          blendColor.g = baseColor.g * cert + greyColor.g * (1 - cert);
          blendColor.b = baseColor.b * cert + greyColor.b * (1 - cert);
          this.meshes[code].setColorAt(i, blendColor);
        }
      }
    }

    for (const [code, mesh] of Object.entries(this.meshes)) {
      mesh.count = iCounters[code] ?? 0;
      mesh.instanceMatrix.needsUpdate = true;
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    }
  }

  // ── Clear all meshes ──────────────────────────────────────────────────────
  clear() {
    for (const mesh of Object.values(this.meshes)) {
      this._group.remove(mesh);
      mesh.geometry.dispose();
      mesh.material.dispose();
    }
    this.meshes = {};
    this.grid   = null;
  }

  get group() { return this._group; }
}
