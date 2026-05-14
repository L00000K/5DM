import * as THREE from 'three';
import { voxelIndex } from './interpolator.js';

// ── Shared ShaderMaterial: per-instance colour + alpha, Lambert lighting ──────
// Uses custom voxelColor + voxelAlpha attributes to avoid Three.js colour-
// multiplication bug (MeshLambertMaterial multiplies diffuse × instanceColor,
// squaring the RGB values and producing near-black output).
const VERT = `
  attribute vec3  voxelColor;
  attribute float voxelAlpha;
  varying   vec3  vCol;
  varying   float vAlph;
  varying   vec3  vNorm;

  void main() {
    vCol  = voxelColor;
    vAlph = voxelAlpha;

    #ifdef USE_INSTANCING
      // instanceMatrix is injected by Three.js renderer prefix
      gl_Position = projectionMatrix * modelViewMatrix * instanceMatrix * vec4(position, 1.0);
      // Voxels are pure translations — rotation part of instanceMatrix is identity
      vNorm = normalMatrix * normal;
    #else
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      vNorm = normalMatrix * normal;
    #endif
  }
`;

const FRAG = `
  varying vec3  vCol;
  varying float vAlph;
  varying vec3  vNorm;

  void main() {
    if (vAlph < 0.01) discard;
    vec3 n = normalize(vNorm);

    // Match the two directional lights defined in scene.js
    vec3 l1 = normalize(vec3( 300.0,  400.0,  200.0)); // key (warm)
    vec3 l2 = normalize(vec3(-200.0,  100.0, -300.0)); // fill (cool)
    float d1 = max(dot(n, l1), 0.0) * 0.80;
    float d2 = max(dot(n, l2), 0.0) * 0.28;
    float ambient = 0.45;

    vec3 lit = vCol * (ambient + d1 + d2);
    gl_FragColor = vec4(lit, vAlph);
  }
`;

function makeMaterial() {
  return new THREE.ShaderMaterial({
    vertexShader:   VERT,
    fragmentShader: FRAG,
    transparent:    false,
    depthWrite:     true,
    side:           THREE.FrontSide,
  });
}

// ── VoxelBuilder ──────────────────────────────────────────────────────────────
export class VoxelBuilder {
  constructor(scene) {
    this.scene   = scene;
    this.meshes  = {};    // unitCode → InstancedMesh
    this.grid    = null;
    this.units   = [];
    this._dummy  = new THREE.Object3D();
    this._group  = new THREE.Group();
    scene.add(this._group);

    // Per-instance certainty buffers — used for fast alpha updates without rebuild
    this._certBuffers = {};   // unitCode → Float32Array

    this.certThreshold       = 0;
    this.transparencyEnabled = false;
    this.transparencyAmount  = 0.8;
  }

  // ── Build all voxel meshes ─────────────────────────────────────────────────
  build(grid, geoUnits) {
    this.clear();
    this.grid  = grid;
    this.units = geoUnits;

    const { nx, ny, nz, cellSize: cs, cellHeight: ch,
            origin, unitIds, certainty, blendUnitIds, blendRatios } = grid;

    const unitById = {};
    geoUnits.forEach(u => { unitById[u.id] = u; });

    // Count instances per unit (all voxels, including unknown)
    const counts = {};
    for (let i = 0; i < unitIds.length; i++) {
      const u = unitById[unitIds[i]];
      if (!u) continue;
      counts[u.code] = (counts[u.code] ?? 0) + 1;
    }

    // Create one InstancedMesh + custom attribute buffers per unit.
    // Each unit needs its OWN geometry instance — if a single BoxGeometry
    // were shared, every setAttribute call would overwrite the previous unit's
    // voxelColor/voxelAlpha buffers on the shared object (they all point to
    // the same BufferGeometry), so only the last unit's colours would survive.
    for (const [code, count] of Object.entries(counts)) {
      const unit = geoUnits.find(u => u.code === code);
      if (!unit) continue;

      const geom = new THREE.BoxGeometry(cs, ch, cs); // own geometry per unit
      const mesh = new THREE.InstancedMesh(geom, makeMaterial(), count);
      mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      mesh.userData = { unitCode: code, unitId: unit.id };

      // Custom per-instance colour + alpha (avoids Three.js colour-multiply bug)
      mesh.geometry.setAttribute('voxelColor',
        new THREE.InstancedBufferAttribute(new Float32Array(count * 3), 3));
      mesh.geometry.setAttribute('voxelAlpha',
        new THREE.InstancedBufferAttribute(new Float32Array(count).fill(1), 1));

      this._certBuffers[code] = new Float32Array(count);
      this.meshes[code] = mesh;
      this._group.add(mesh);
    }

    // Populate instance data
    const counters = {};
    const c1 = new THREE.Color();
    const c2 = new THREE.Color();
    const cx = new THREE.Color();

    for (let iz = 0; iz < nz; iz++) {
      for (let iy = 0; iy < ny; iy++) {
        for (let ix = 0; ix < nx; ix++) {
          const flat  = voxelIndex(ix, iy, iz, grid);
          const uid   = unitIds[flat];
          const unit  = unitById[uid];
          if (!unit) continue;

          const code  = unit.code;
          const mesh  = this.meshes[code];
          if (!mesh) continue;

          const i = counters[code] = (counters[code] ?? 0);
          counters[code]++;

          // World position
          const wx = origin.x + ix * cs + cs * 0.5;
          const wy = origin.y + iz * ch + ch * 0.5;
          const wz = origin.z + iy * cs + cs * 0.5;
          this._dummy.position.set(wx, wy, wz);
          this._dummy.updateMatrix();
          mesh.setMatrixAt(i, this._dummy.matrix);

          // Colour — blend toward second-best unit at geological contacts.
          // convertSRGBToLinear: Three.js Color.set(hexString) stores sRGB values;
          // the custom ShaderMaterial outputs to a linear framebuffer, so we must
          // supply linear-space values or colours will appear too dark / desaturated.
          const cert  = certainty[flat];
          c1.set(unit.color).convertSRGBToLinear();
          const bu    = unitById[blendUnitIds[flat]];
          const blend = blendRatios[flat] ?? 0;
          if (bu && bu.code !== code && blend > 0.05) {
            c2.set(bu.color).convertSRGBToLinear();
            const t = Math.min(blend * 0.7, 0.5);
            cx.r = c1.r + (c2.r - c1.r) * t;
            cx.g = c1.g + (c2.g - c1.g) * t;
            cx.b = c1.b + (c2.b - c1.b) * t;
          } else {
            cx.copy(c1);
          }
          const colAttr = mesh.geometry.getAttribute('voxelColor');
          colAttr.setXYZ(i, cx.r, cx.g, cx.b);

          // Store certainty for later alpha updates
          this._certBuffers[code][i] = cert;
        }
      }
    }

    // Mark buffers dirty and compute bounding spheres
    for (const mesh of Object.values(this.meshes)) {
      mesh.instanceMatrix.needsUpdate = true;
      mesh.geometry.getAttribute('voxelColor').needsUpdate = true;
      mesh.computeBoundingSphere();
    }

    // Apply initial alphas
    this._updateAlphas();
    return this._group;
  }

  // ── Alpha update — called on threshold or transparency changes (no rebuild) ─
  _updateAlphas() {
    const { certThreshold, transparencyEnabled, transparencyAmount } = this;

    for (const [code, mesh] of Object.entries(this.meshes)) {
      const certs    = this._certBuffers[code];
      const alphaAttr = mesh.geometry.getAttribute('voxelAlpha');
      if (!certs || !alphaAttr) continue;

      for (let i = 0; i < certs.length; i++) {
        const cert = certs[i];
        if (cert < certThreshold) {
          alphaAttr.setX(i, 0);
        } else if (transparencyEnabled) {
          // alpha = 1 - amount × (1 - certainty)
          const alpha = Math.max(0.04, 1.0 - transparencyAmount * (1.0 - cert));
          alphaAttr.setX(i, alpha);
        } else {
          alphaAttr.setX(i, 1.0);
        }
      }
      alphaAttr.needsUpdate = true;
    }
  }

  // ── Public controls ───────────────────────────────────────────────────────
  setUnitVisibility(code, visible) {
    const mesh = this.meshes[code];
    if (mesh) mesh.visible = visible;
  }

  setCertaintyThreshold(threshold) {
    this.certThreshold = threshold;
    this._updateAlphas();
  }

  setTransparencyMode(enabled, amount) {
    this.transparencyEnabled = enabled;
    this.transparencyAmount  = amount;
    for (const mesh of Object.values(this.meshes)) {
      mesh.material.transparent = enabled;
      mesh.material.depthWrite  = !enabled;
      mesh.material.needsUpdate = true;
    }
    this._updateAlphas();
  }

  clear() {
    for (const mesh of Object.values(this.meshes)) {
      this._group.remove(mesh);
      mesh.geometry.dispose();
      mesh.material.dispose();
    }
    this.meshes        = {};
    this._certBuffers  = {};
    this.grid          = null;
  }

  get group() { return this._group; }
}
