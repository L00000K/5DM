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

    // Flat-ish shading: top faces get full colour, sides/bottom 80%.
    // Strong directional lights cause visible colour jumps at each voxel
    // step; this keeps the unit colour consistent across all face orientations.
    vec3 n = normalize(vNorm);
    float shade = 0.80 + 0.20 * max(n.y, 0.0);
    vec3 col = vCol * shade;

    // Certainty colour fade: blend toward white as certainty drops
    col = mix(col, vec3(1.0), (1.0 - vCert) * uColorFade);

    float alpha = vAlph * uGlobalAlpha;
    if (alpha < 0.01) discard;
    gl_FragColor = vec4(col, alpha);
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

    // Track flat index per instance so colorByParameter can update without rebuild
    this._unitFlatIdx  = {};
    this._unitColorBuf = {};
    for (const code of Object.keys(counts)) {
      this._unitFlatIdx[code]  = new Int32Array(counts[code]);
      this._unitColorBuf[code] = new Float32Array(counts[code] * 3);
    }

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

          // Store original color for reset
          this._unitColorBuf[code][i * 3]     = cx.r;
          this._unitColorBuf[code][i * 3 + 1] = cx.g;
          this._unitColorBuf[code][i * 3 + 2] = cx.b;

          // Store flat index for parameter coloring
          this._unitFlatIdx[code][i] = flat;

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

  // ── Parameter coloring ────────────────────────────────────────────────────
  // Jet-like colormap: blue → cyan → green → yellow → red
  static _paramColor(t, c) {
    const r = t < 0.5 ? t * 2 : 1;
    const g = t < 0.25 ? t * 4 : t < 0.75 ? 1 : (1 - t) * 4;
    const b = t < 0.5 ? 1 : (1 - t) * 2;
    c.setRGB(r, g, b);
  }

  // Color voxels by an engineering parameter.
  // geoUnits: unit array with .params; paramName: 'cu'|'phi'|'N_spt'|'Cc'|'E'|'gamma'
  // paramGrid (optional): Float32Array[nx*ny*nz] from spatial interpolation
  // Returns { min, max } of the value range used.
  colorByParameter(paramName, geoUnits, paramGrid = null) {
    if (!this.grid) return null;
    const { unitIds } = this.grid;
    const unitById = {};
    geoUnits.forEach(u => { unitById[u.id] = u; });

    // Collect values to determine range
    const vals = [];
    if (paramGrid) {
      for (let i = 0; i < paramGrid.length; i++) {
        if (unitIds[i] && isFinite(paramGrid[i])) vals.push(paramGrid[i]);
      }
    } else {
      for (const unit of geoUnits) {
        const v = unit.params?.[paramName];
        if (v != null && isFinite(v)) vals.push(v);
      }
    }
    if (!vals.length) return null;
    const vMin = Math.min(...vals), vMax = Math.max(...vals);
    const range = vMax - vMin || 1;

    const col = new THREE.Color();
    for (const [code, mesh] of Object.entries(this.meshes)) {
      const flatIdx = this._unitFlatIdx?.[code];
      if (!flatIdx) continue;
      const colorAttr = mesh.geometry.getAttribute('voxelColor');
      for (let i = 0; i < flatIdx.length; i++) {
        let v;
        if (paramGrid) {
          v = paramGrid[flatIdx[i]];
        } else {
          const uid = unitIds[flatIdx[i]];
          v = unitById[uid]?.params?.[paramName];
        }
        if (v == null || !isFinite(v)) {
          col.setRGB(0.5, 0.5, 0.5);
        } else {
          VoxelBuilder._paramColor((v - vMin) / range, col);
        }
        colorAttr.setXYZ(i, col.r, col.g, col.b);
      }
      colorAttr.needsUpdate = true;
    }
    return { min: vMin, max: vMax };
  }

  // Color voxels by boundary uncertainty (blendRatio from interpolation).
  // High blend ratio = voxel sits near a unit contact = warm-red highlight.
  colorByBoundaryUncertainty() {
    if (!this.grid) return;
    const { blendRatios } = this.grid;
    const col = new THREE.Color();
    for (const [code, mesh] of Object.entries(this.meshes)) {
      const flatIdx = this._unitFlatIdx?.[code];
      if (!flatIdx) continue;
      const colorAttr = mesh.geometry.getAttribute('voxelColor');
      for (let i = 0; i < flatIdx.length; i++) {
        VoxelBuilder._paramColor(blendRatios[flatIdx[i]] ?? 0, col);
        colorAttr.setXYZ(i, col.r, col.g, col.b);
      }
      colorAttr.needsUpdate = true;
    }
  }

  // Restore original unit colours after parameter coloring.
  resetUnitColors() {
    for (const [code, mesh] of Object.entries(this.meshes)) {
      const buf = this._unitColorBuf?.[code];
      if (!buf) continue;
      const colorAttr = mesh.geometry.getAttribute('voxelColor');
      for (let i = 0; i < buf.length / 3; i++) {
        colorAttr.setXYZ(i, buf[i * 3], buf[i * 3 + 1], buf[i * 3 + 2]);
      }
      colorAttr.needsUpdate = true;
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

  setClippingPlanes(planes) {
    for (const mesh of Object.values(this.meshes)) {
      mesh.material.clippingPlanes = planes.length ? planes : null;
      mesh.material.needsUpdate = true;
    }
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
