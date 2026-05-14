import * as THREE from 'three';
import { voxelIndex } from './interpolator.js';

// ── Shared ShaderMaterial: per-instance colour + alpha, Lambert lighting ──────
// Three.js does NOT auto-inject clipping plane chunks into raw ShaderMaterial;
// they must be included explicitly. mvPosition is required by the vertex chunk.
const VERT = `
  attribute vec3  voxelColor;
  attribute float voxelAlpha;
  attribute float voxelCert;
  varying   vec3  vCol;
  varying   float vAlph;
  varying   float vCert;
  varying   vec3  vNorm;

  #include <clipping_planes_pars_vertex>

  void main() {
    vCol  = voxelColor;
    vAlph = voxelAlpha;
    vCert = voxelCert;

    #ifdef USE_INSTANCING
      vec4 mvPosition = modelViewMatrix * instanceMatrix * vec4(position, 1.0);
      vNorm = normalMatrix * normal;
    #else
      vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
      vNorm = normalMatrix * normal;
    #endif
    gl_Position = projectionMatrix * mvPosition;

    #include <clipping_planes_vertex>
  }
`;

const FRAG = `
  uniform float uColorFade;
  uniform float uGlobalAlpha;

  varying vec3  vCol;
  varying float vAlph;
  varying float vCert;
  varying vec3  vNorm;

  #include <clipping_planes_pars_fragment>

  void main() {
    #include <clipping_planes_fragment>

    if (vAlph < 0.01) discard;
    vec3 n = normalize(vNorm);

    // Match the two directional lights defined in scene.js
    vec3 l1 = normalize(vec3( 300.0,  400.0,  200.0));
    vec3 l2 = normalize(vec3(-200.0,  100.0, -300.0));
    float d1 = max(dot(n, l1), 0.0) * 0.80;
    float d2 = max(dot(n, l2), 0.0) * 0.28;
    float ambient = 0.45;

    vec3 lit = vCol * (ambient + d1 + d2);

    // Certainty colour fade: blend lit colour toward white as certainty drops
    lit = mix(lit, vec3(1.0), (1.0 - vCert) * uColorFade);

    float alpha = vAlph * uGlobalAlpha;
    if (alpha < 0.01) discard;
    gl_FragColor = vec4(lit, alpha);
  }
`;

function makeMaterial() {
  return new THREE.ShaderMaterial({
    vertexShader:   VERT,
    fragmentShader: FRAG,
    transparent:    false,
    depthWrite:     true,
    side:           THREE.FrontSide,
    uniforms: {
      uColorFade:   { value: 0.0 },
      uGlobalAlpha: { value: 1.0 },
    },
  });
}

// ── VoxelBuilder ──────────────────────────────────────────────────────────────
export class VoxelBuilder {
  constructor(scene) {
    this.scene   = scene;
    this.meshes  = {};
    this.grid    = null;
    this.units   = [];
    this._dummy  = new THREE.Object3D();
    this._group  = new THREE.Group();
    scene.add(this._group);

    this._certBuffers = {};

    this.certThreshold       = 0;
    this.transparencyEnabled = false;
    this.transparencyAmount  = 0.8;
    this._globalAlpha        = 1.0;
    this._colorFadeEnabled   = false;
    this._colorFadeAmount    = 0.8;
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

    // Count instances per unit
    const counts = {};
    for (let i = 0; i < unitIds.length; i++) {
      const u = unitById[unitIds[i]];
      if (!u) continue;
      counts[u.code] = (counts[u.code] ?? 0) + 1;
    }

    // One InstancedMesh per unit — each needs its own geometry instance so
    // setAttribute calls don't overwrite each other's buffers.
    for (const [code, count] of Object.entries(counts)) {
      const unit = geoUnits.find(u => u.code === code);
      if (!unit) continue;

      const geom = new THREE.BoxGeometry(cs, ch, cs);
      const mesh = new THREE.InstancedMesh(geom, makeMaterial(), count);
      mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      mesh.userData = { unitCode: code, unitId: unit.id };

      mesh.geometry.setAttribute('voxelColor',
        new THREE.InstancedBufferAttribute(new Float32Array(count * 3), 3));
      mesh.geometry.setAttribute('voxelAlpha',
        new THREE.InstancedBufferAttribute(new Float32Array(count).fill(1), 1));
      mesh.geometry.setAttribute('voxelCert',
        new THREE.InstancedBufferAttribute(new Float32Array(count).fill(1), 1));

      this._certBuffers[code] = new Float32Array(count);
      this.meshes[code] = mesh;
      this._group.add(mesh);
    }

    // Populate instance transforms, colours, certainty
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

          const code = unit.code;
          const mesh = this.meshes[code];
          if (!mesh) continue;

          const i = counters[code] = (counters[code] ?? 0);
          counters[code]++;

          // World position
          this._dummy.position.set(
            origin.x + ix * cs + cs * 0.5,
            origin.y + iz * ch + ch * 0.5,
            origin.z + iy * cs + cs * 0.5,
          );
          this._dummy.updateMatrix();
          mesh.setMatrixAt(i, this._dummy.matrix);

          // Colour with contact blending.
          // convertSRGBToLinear because Color.set(hex) stores sRGB but the
          // shader writes to a linear WebGL2 framebuffer.
          const cert = certainty[flat];
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
          mesh.geometry.getAttribute('voxelColor').setXYZ(i, cx.r, cx.g, cx.b);

          // Certainty — stored on GPU for shader colour-fade and on CPU for
          // alpha threshold updates.
          mesh.geometry.getAttribute('voxelCert').setX(i, cert);
          this._certBuffers[code][i] = cert;
        }
      }
    }

    // Mark all buffers dirty
    for (const mesh of Object.values(this.meshes)) {
      mesh.instanceMatrix.needsUpdate = true;
      mesh.geometry.getAttribute('voxelColor').needsUpdate = true;
      mesh.geometry.getAttribute('voxelCert').needsUpdate  = true;
      mesh.computeBoundingSphere();
    }

    // Re-apply current display settings to newly built meshes
    this._applyMaterialState();
    this._applyColorFadeUniforms();
    this._applyGlobalAlphaUniforms();
    this._updateAlphas();
    return this._group;
  }

  // ── Alpha update — called on threshold / transparency changes (no rebuild) ──
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
          alphaAttr.setX(i, Math.max(0.04, 1.0 - transparencyAmount * (1.0 - cert)));
        } else {
          alphaAttr.setX(i, 1.0);
        }
      }
      alphaAttr.needsUpdate = true;
    }
  }

  // ── Material state: transparent + depthWrite depend on multiple controls ───
  _applyMaterialState() {
    const needsTransp = this.transparencyEnabled || this._globalAlpha < 0.999;
    for (const mesh of Object.values(this.meshes)) {
      mesh.material.transparent = needsTransp;
      mesh.material.depthWrite  = !needsTransp;
      mesh.material.needsUpdate = true;
    }
  }

  _applyColorFadeUniforms() {
    const v = this._colorFadeEnabled ? this._colorFadeAmount : 0.0;
    for (const mesh of Object.values(this.meshes)) {
      mesh.material.uniforms.uColorFade.value = v;
    }
  }

  _applyGlobalAlphaUniforms() {
    for (const mesh of Object.values(this.meshes)) {
      mesh.material.uniforms.uGlobalAlpha.value = this._globalAlpha;
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
    this._applyMaterialState();
    this._updateAlphas();
  }

  setColorFadeMode(enabled, amount) {
    this._colorFadeEnabled = enabled;
    this._colorFadeAmount  = amount;
    this._applyColorFadeUniforms();
  }

  setGlobalAlpha(alpha) {
    this._globalAlpha = alpha;
    this._applyGlobalAlphaUniforms();
    this._applyMaterialState();
  }

  clear() {
    for (const mesh of Object.values(this.meshes)) {
      this._group.remove(mesh);
      mesh.geometry.dispose();
      mesh.material.dispose();
    }
    this.meshes       = {};
    this._certBuffers = {};
    this.grid         = null;
  }

  get group() { return this._group; }
}
