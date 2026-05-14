import * as THREE from 'three';
import { voxelIndex } from './interpolator.js';

// ── Shader injection: adds per-instance alpha to MeshLambertMaterial ──────────
function makeAlphaMaterial(color) {
  const mat = new THREE.MeshLambertMaterial({
    color,
    vertexColors: true,
    transparent: true,
    depthWrite: false,
    side: THREE.FrontSide,
  });

  mat.onBeforeCompile = shader => {
    shader.vertexShader = shader.vertexShader
      .replace(
        '#include <color_pars_vertex>',
        `#include <color_pars_vertex>
         attribute float instanceAlpha;
         varying float vInstanceAlpha;`
      )
      .replace(
        '#include <color_vertex>',
        `#include <color_vertex>
         vInstanceAlpha = instanceAlpha;`
      );
    shader.fragmentShader = shader.fragmentShader
      .replace(
        '#include <color_pars_fragment>',
        `#include <color_pars_fragment>
         varying float vInstanceAlpha;`
      )
      .replace(
        'gl_FragColor = vec4( outgoingLight, diffuseColor.a );',
        'gl_FragColor = vec4( outgoingLight, diffuseColor.a * vInstanceAlpha );'
      );
  };
  return mat;
}

// ── VoxelBuilder ──────────────────────────────────────────────────────────────
export class VoxelBuilder {
  constructor(scene) {
    this.scene   = scene;
    this.meshes  = {};        // unitCode → THREE.InstancedMesh
    this.grid    = null;
    this.units   = [];
    this._dummy  = new THREE.Object3D();
    this._group  = new THREE.Group();
    scene.add(this._group);
  }

  // ── Build all voxel meshes ─────────────────────────────────────────────────
  build(grid, geoUnits) {
    this.clear();
    this.grid  = grid;
    this.units = geoUnits;
    this._buildMeshes(grid, geoUnits, 0);
    return this._group;
  }

  _buildMeshes(grid, geoUnits, certThreshold) {
    const { nx, ny, nz, cellSize: cs, cellHeight: ch,
            origin, unitIds, certainty, blendUnitIds, blendRatios } = grid;

    const unitById = {};
    geoUnits.forEach(u => { unitById[u.id] = u; });

    // Count instances per unit above threshold
    const counts = {};
    for (let i = 0; i < unitIds.length; i++) {
      if (certainty[i] < certThreshold) continue;
      const u = unitById[unitIds[i]];
      if (!u) continue;
      counts[u.code] = (counts[u.code] ?? 0) + 1;
    }

    // One InstancedMesh + instanceAlpha buffer per unit
    const geom = new THREE.BoxGeometry(cs * 0.88, ch * 0.88, cs * 0.88);

    for (const [code, count] of Object.entries(counts)) {
      const unit = geoUnits.find(u => u.code === code);
      if (!unit) continue;
      const mat  = makeAlphaMaterial(unit.color);
      const mesh = new THREE.InstancedMesh(geom, mat, count);
      mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);

      // instanceColor (RGB) and instanceAlpha (scalar) per instance
      mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(count * 3), 3);
      const alphaAttr    = new THREE.InstancedBufferAttribute(new Float32Array(count).fill(1), 1);
      mesh.geometry.setAttribute('instanceAlpha', alphaAttr);

      mesh.userData = { unitCode: code, unitId: unit.id };
      mesh.renderOrder = 1; // ensure transparent voxels sort correctly
      this.meshes[code] = mesh;
      this._group.add(mesh);
    }

    // Populate instances
    const iCounters = {};
    const c1 = new THREE.Color();
    const c2 = new THREE.Color();
    const cx = new THREE.Color();

    for (let iz = 0; iz < nz; iz++) {
      for (let iy = 0; iy < ny; iy++) {
        for (let ix = 0; ix < nx; ix++) {
          const flat = voxelIndex(ix, iy, iz, grid);
          const cert = certainty[flat];
          if (cert < certThreshold) continue;

          const uid  = unitIds[flat];
          const unit = unitById[uid];
          if (!unit || !this.meshes[unit.code]) continue;

          const i = iCounters[unit.code] = (iCounters[unit.code] ?? 0);
          iCounters[unit.code]++;

          // World position
          const wx = origin.x + ix * cs + cs * 0.5;
          const wy = origin.y + iz * ch + ch * 0.5;
          const wz = origin.z + iy * cs + cs * 0.5;
          this._dummy.position.set(wx, wy, wz);
          this._dummy.updateMatrix();
          this.meshes[unit.code].setMatrixAt(i, this._dummy.matrix);

          // Colour: blend winning unit colour toward second-best at contacts
          c1.set(unit.color);
          const blendUnit = unitById[blendUnitIds[flat]];
          const blend     = blendRatios[flat] ?? 0;
          if (blendUnit && blendUnit.code !== unit.code && blend > 0.05) {
            c2.set(blendUnit.color);
            // max 50 % blend so winning unit stays dominant
            const t = Math.min(blend * 0.7, 0.5);
            cx.r = c1.r + (c2.r - c1.r) * t;
            cx.g = c1.g + (c2.g - c1.g) * t;
            cx.b = c1.b + (c2.b - c1.b) * t;
          } else {
            cx.copy(c1);
          }
          this.meshes[unit.code].setColorAt(i, cx);

          // Alpha: certainty → transparency (clamp to 0.08 floor so low-cert voxels remain barely visible)
          const alpha = Math.max(0.08, cert);
          this.meshes[unit.code].geometry.getAttribute('instanceAlpha').setX(i, alpha);
        }
      }
    }

    for (const mesh of Object.values(this.meshes)) {
      mesh.instanceMatrix.needsUpdate = true;
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
      const a = mesh.geometry.getAttribute('instanceAlpha');
      if (a) a.needsUpdate = true;
      mesh.computeBoundingSphere();
    }
  }

  // ── Show/hide unit ────────────────────────────────────────────────────────
  setUnitVisibility(code, visible) {
    const mesh = this.meshes[code];
    if (mesh) mesh.visible = visible;
  }

  // ── Certainty threshold: rebuild with new floor ───────────────────────────
  setCertaintyThreshold(threshold) {
    if (!this.grid) return;
    this.clear();
    this._buildMeshes(this.grid, this.units, threshold);
  }

  // ── Clear ─────────────────────────────────────────────────────────────────
  clear() {
    for (const mesh of Object.values(this.meshes)) {
      this._group.remove(mesh);
      mesh.geometry.dispose();
      mesh.material.dispose();
    }
    this.meshes = {};
  }

  get group() { return this._group; }
}
