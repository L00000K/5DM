import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { VoxelBuilder } from './voxel-builder.js';
import { log } from './app.js';

export async function initScene(canvasId) {
  return new SceneManager(canvasId);
}

class SceneManager {
  constructor(canvasId) {
    this._canvas   = document.getElementById(canvasId);
    this._scene    = new THREE.Scene();
    this._scene.background = new THREE.Color(0xf0f2f5);

    this._renderer = new THREE.WebGLRenderer({
      canvas: this._canvas,
      antialias: true,
      localClippingEnabled: true,
    });
    this._renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    this._camera = new THREE.PerspectiveCamera(55, 1, 0.5, 10000);
    this._camera.position.set(200, 80, 280);

    this._controls = new OrbitControls(this._camera, this._canvas);
    this._controls.enableDamping  = true;
    this._controls.dampingFactor  = 0.07;
    this._controls.screenSpacePanning = false;

    this._lights();
    this._grid   = this._addGrid();
    this._axes   = this._addAxes();
    this._builder = new VoxelBuilder(this._scene);

    // Clipping planes (X, Y, Z) — disabled by default
    this._clipX = new THREE.Plane(new THREE.Vector3(-1, 0,  0), Infinity);
    this._clipY = new THREE.Plane(new THREE.Vector3( 0, 0, -1), Infinity);
    this._clipZ = new THREE.Plane(new THREE.Vector3( 0,-1,  0), Infinity);
    this._renderer.clippingPlanes = [];

    this._resizeObserver = new ResizeObserver(() => this._onResize());
    this._resizeObserver.observe(this._canvas.parentElement);
    this._onResize();

    this._animate();
    this._initSliceControls();
    this._initTooltip();
  }

  // ── Lights ────────────────────────────────────────────────────────────────
  _lights() {
    const ambient = new THREE.AmbientLight(0xffffff, 0.55);
    this._scene.add(ambient);

    const sun = new THREE.DirectionalLight(0xffeedd, 0.9);
    sun.position.set(300, 400, 200);
    this._scene.add(sun);

    const fill = new THREE.DirectionalLight(0xd0e8ff, 0.35);
    fill.position.set(-200, 100, -300);
    this._scene.add(fill);
  }

  // ── Ground grid ───────────────────────────────────────────────────────────
  _addGrid() {
    const grid = new THREE.GridHelper(500, 50, 0x243548, 0x1a2840);
    grid.position.y = -0.5;
    this._scene.add(grid);
    return grid;
  }

  // ── Compass axes ──────────────────────────────────────────────────────────
  _addAxes() {
    const axes = new THREE.AxesHelper(30);
    axes.position.set(-20, 0, -20);
    this._scene.add(axes);
    return axes;
  }

  // ── Resize ────────────────────────────────────────────────────────────────
  _onResize() {
    const el = this._canvas.parentElement;
    if (!el) return;
    const w = el.clientWidth, h = el.clientHeight;
    if (!w || !h) return;
    this._renderer.setSize(w, h);
    this._camera.aspect = w / h;
    this._camera.updateProjectionMatrix();
  }

  // ── Render loop ───────────────────────────────────────────────────────────
  _animate() {
    requestAnimationFrame(() => this._animate());
    this._controls.update();
    this._renderer.render(this._scene, this._camera);
  }

  // ── Build voxel model ─────────────────────────────────────────────────────
  buildVoxels(grid, geoUnits) {
    this._grid.position.y   = grid.origin.y - 1;
    this._grid.scale.setScalar(Math.max(grid.worldWidth, grid.worldDepth) / 500);
    this._axes.position.set(grid.origin.x - 20, grid.origin.y, grid.origin.z - 20);

    this._builder.build(grid, geoUnits);
    this._applyClipping();

    // Frame camera on model
    const cx = grid.origin.x + grid.worldWidth  * 0.5;
    const cy = grid.origin.y + grid.worldHeight * 0.5;
    const cz = grid.origin.z + grid.worldDepth  * 0.5;
    const size = Math.max(grid.worldWidth, grid.worldDepth, grid.worldHeight);

    this._controls.target.set(cx, cy, cz);
    this._camera.position.set(cx + size * 0.8, cy + size * 0.5, cz + size * 0.8);
    this._camera.lookAt(cx, cy, cz);
    this._controls.update();

    // Update clipping plane constants to model bounds
    this._modelBounds = { cx, cy, cz, size, grid };
    this._syncClipPlanes();

    log('3D scene updated', 'ok');
  }

  // ── Unit visibility ───────────────────────────────────────────────────────
  setUnitVisibility(code, visible) {
    this._builder.setUnitVisibility(code, visible);
  }

  // ── Certainty filter ──────────────────────────────────────────────────────
  setCertaintyThreshold(t) {
    this._builder.setCertaintyThreshold(t);
  }

  // ── Clear scene ───────────────────────────────────────────────────────────
  clear() { this._builder.clear(); }

  // ── Clipping plane sync ───────────────────────────────────────────────────
  _syncClipPlanes() {
    if (!this._modelBounds) return;
    const { grid } = this._modelBounds;
    const maxX = grid.origin.x + grid.worldWidth;
    const maxY = grid.origin.y + grid.worldHeight;
    const maxZ = grid.origin.z + grid.worldDepth;

    this._clipX.constant = maxX * (this._clipXPct ?? 1.0);
    this._clipY.constant = maxZ * (this._clipYPct ?? 1.0);
    this._clipZ.constant = maxY * (this._clipZPct ?? 1.0);
  }

  _applyClipping() {
    const planes = [];
    if (this._clipXEnabled) planes.push(this._clipX);
    if (this._clipYEnabled) planes.push(this._clipY);
    if (this._clipZEnabled) planes.push(this._clipZ);
    this._renderer.clippingPlanes = planes;
  }

  // ── Slice controls wiring ─────────────────────────────────────────────────
  _initSliceControls() {
    const wire = (id, enableId, pctProp, enableProp, planeProp) => {
      const slider = document.getElementById(id);
      const chk    = document.getElementById(enableId);
      if (!slider || !chk) return;

      slider.addEventListener('input', () => {
        this[pctProp] = parseInt(slider.value) / 100;
        this._syncClipPlanes();
      });

      chk.addEventListener('change', () => {
        this[enableProp] = chk.checked;
        this._applyClipping();
      });
    };

    wire('slice-x', 'slice-x-en', '_clipXPct', '_clipXEnabled', '_clipX');
    wire('slice-y', 'slice-y-en', '_clipYPct', '_clipYEnabled', '_clipY');
    wire('slice-z', 'slice-z-en', '_clipZPct', '_clipZEnabled', '_clipZ');
  }

  // ── Voxel hover tooltip ───────────────────────────────────────────────────
  _initTooltip() {
    const tooltip  = document.getElementById('voxel-tooltip');
    const raycaster = new THREE.Raycaster();
    const mouse     = new THREE.Vector2();

    this._canvas.addEventListener('mousemove', evt => {
      if (!this._modelBounds || !tooltip) return;
      const rect = this._canvas.getBoundingClientRect();
      mouse.x = ((evt.clientX - rect.left)  / rect.width)  * 2 - 1;
      mouse.y = -((evt.clientY - rect.top) / rect.height) * 2 + 1;

      raycaster.setFromCamera(mouse, this._camera);
      const meshes = Object.values(this._builder.meshes);
      const hits = raycaster.intersectObjects(meshes);

      if (hits.length) {
        const hit  = hits[0];
        const unit = this._builder.units.find(u => u.code === hit.object.userData.unitCode);
        const iIdx = hit.instanceId;
        let cert = '—';

        if (this._modelBounds?.grid && iIdx !== undefined) {
          // Best-effort: get certainty for this instance
          const grid = this._modelBounds.grid;
          const { nx, ny, nz, cellSize, cellHeight, origin, unitIds, certainty: certs } = grid;
          const unitById = {};
          this._builder.units.forEach(u => { unitById[u.id] = u; });

          let cnt = 0;
          for (let idx = 0; idx < unitIds.length; idx++) {
            const u = unitById[unitIds[idx]];
            if (u?.code === hit.object.userData.unitCode) {
              if (cnt === iIdx) {
                cert = `${(certs[idx] * 100).toFixed(0)}%`;
                break;
              }
              cnt++;
            }
          }
        }

        tooltip.hidden = false;
        tooltip.style.left = `${evt.clientX - this._canvas.getBoundingClientRect().left + 14}px`;
        tooltip.style.top  = `${evt.clientY - this._canvas.getBoundingClientRect().top  -  8}px`;
        tooltip.innerHTML = `
          <div class="tooltip-title">${unit?.name ?? hit.object.userData.unitCode}</div>
          <div class="tooltip-row"><span>Code</span><span class="tooltip-val">${hit.object.userData.unitCode}</span></div>
          <div class="tooltip-row"><span>Certainty</span><span class="tooltip-val">${cert}</span></div>
          <div class="tooltip-row"><span>Pos</span><span class="tooltip-val">(${hit.point.x.toFixed(0)}, ${hit.point.y.toFixed(0)}, ${hit.point.z.toFixed(0)})</span></div>`;
      } else {
        tooltip.hidden = true;
      }
    });

    this._canvas.addEventListener('mouseleave', () => {
      if (tooltip) tooltip.hidden = true;
    });
  }

  // ── Getters for exporter ──────────────────────────────────────────────────
  get threeScene()  { return this._scene; }
  get threeCamera() { return this._camera; }
  get voxelGroup()  { return this._builder.group; }
}
