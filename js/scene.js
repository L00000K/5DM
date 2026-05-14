import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { VoxelBuilder } from './voxel-builder.js';
import { SlicerTool } from './slicer.js';
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
    this._slicer  = new SlicerTool(
      this._scene, this._camera, this._controls, this._renderer,
      planes => this._builder.setClippingPlanes(planes)
    );

    this._resizeObserver = new ResizeObserver(() => this._onResize());
    this._resizeObserver.observe(this._canvas.parentElement);
    this._onResize();

    this._animate();
    this._initTooltip();
    this._initMiddleMouse();
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
    const grid = new THREE.GridHelper(500, 50, 0x9aaabb, 0xc4d0d8);
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
  buildVoxels(grid, geoUnits, classifiedBH) {
    this._grid.position.y   = grid.origin.y - 1;
    this._grid.scale.setScalar(Math.max(grid.worldWidth, grid.worldDepth) / 500);
    this._axes.position.set(grid.origin.x - 20, grid.origin.y, grid.origin.z - 20);

    this._builder.build(grid, geoUnits);
    if (classifiedBH?.length) this.addBoreholeSticks(classifiedBH, geoUnits);

    // Frame camera on model
    const cx = grid.origin.x + grid.worldWidth  * 0.5;
    const cy = grid.origin.y + grid.worldHeight * 0.5;
    const cz = grid.origin.z + grid.worldDepth  * 0.5;
    const size = Math.max(grid.worldWidth, grid.worldDepth, grid.worldHeight);

    this._controls.target.set(cx, cy, cz);
    this._camera.position.set(cx + size * 0.8, cy + size * 0.5, cz + size * 0.8);
    this._camera.lookAt(cx, cy, cz);
    this._controls.update();

    this._modelBounds = { cx, cy, cz, size, grid };
    this._slicer.setModelBounds(grid);

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

  // ── Transparency mode ─────────────────────────────────────────────────────
  setTransparencyMode(enabled, amount) {
    this._builder.setTransparencyMode(enabled, amount);
  }

  setColorFadeMode(enabled, amount) {
    this._builder.setColorFadeMode(enabled, amount);
  }

  setGlobalAlpha(alpha) {
    this._builder.setGlobalAlpha(alpha);
  }

  // ── Vertical exaggeration ─────────────────────────────────────────────────
  setVerticalExaggeration(ve) {
    this._builder.group.scale.y = ve;
    if (this._bhSticks) this._bhSticks.scale.y = ve;
  }

  // ── Centre view on model (middle-mouse or button) ─────────────────────────
  centreView() {
    if (!this._modelBounds) return;
    const { cx, cy, cz, size } = this._modelBounds;
    this._controls.target.set(cx, cy, cz);
    this._camera.position.set(cx + size * 0.8, cy + size * 0.5, cz + size * 0.8);
    this._camera.lookAt(cx, cy, cz);
    this._controls.update();
  }

  // ── Borehole sticks ───────────────────────────────────────────────────────
  addBoreholeSticks(classifiedBH, geoUnits) {
    if (this._bhSticks) {
      this._scene.remove(this._bhSticks);
      this._bhSticks.traverse(obj => {
        obj.geometry?.dispose();
        obj.material?.dispose();
      });
    }

    const group = new THREE.Group();
    const unitByCode = {};
    geoUnits.forEach(u => { unitByCode[u.code] = u; });

    const radius = 0.6;
    classifiedBH.forEach(bh => {
      if (!bh.layers?.length) return;
      const gl = bh.groundLevel ?? 0;
      bh.layers.forEach(layer => {
        const unit = unitByCode[layer.unitCode];
        if (!unit) return;
        const height = Math.max(layer.base - layer.top, 0.01);
        const midElev = gl - (layer.top + layer.base) * 0.5;
        const geom = new THREE.CylinderGeometry(radius, radius, height, 8);
        const mat  = new THREE.MeshLambertMaterial({ color: unit.color });
        const mesh = new THREE.Mesh(geom, mat);
        mesh.position.set(bh.x, midElev, bh.y);
        group.add(mesh);
      });
    });

    this._bhSticks = group;
    this._scene.add(group);
  }

  toggleBoreholeSticks(visible) {
    if (this._bhSticks) this._bhSticks.visible = visible;
  }

  // ── Topography surface ────────────────────────────────────────────────────
  showTopography(points) {
    this._clearTopo();
    if (!points?.length) return;

    let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
    points.forEach(p => {
      minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x);
      minZ = Math.min(minZ, p.y); maxZ = Math.max(maxZ, p.y);
    });

    const GRID = 48;
    const gx = GRID, gz = GRID;
    const vCount = (gx + 1) * (gz + 1);
    const positions = new Float32Array(vCount * 3);
    const indices   = [];

    for (let iz = 0; iz <= gz; iz++) {
      for (let ix = 0; ix <= gx; ix++) {
        const wx = minX + (maxX - minX) * (ix / gx);
        const wz = minZ + (maxZ - minZ) * (iz / gz);
        let bestDist = Infinity, bestY = 0;
        points.forEach(p => {
          const d = (p.x - wx) ** 2 + (p.y - wz) ** 2;
          if (d < bestDist) { bestDist = d; bestY = p.z; }
        });
        const i = (iz * (gx + 1) + ix) * 3;
        positions[i] = wx; positions[i + 1] = bestY; positions[i + 2] = wz;
      }
    }
    for (let iz = 0; iz < gz; iz++) {
      for (let ix = 0; ix < gx; ix++) {
        const a = iz * (gx + 1) + ix;
        const b = a + 1, c = a + gx + 1, d = c + 1;
        indices.push(a, c, b, b, c, d);
      }
    }

    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geom.setIndex(indices);
    geom.computeVertexNormals();

    const mat = new THREE.MeshLambertMaterial({
      color: 0x7aaa88, transparent: true, opacity: 0.32,
      side: THREE.DoubleSide, depthWrite: false,
    });
    this._topoMesh = new THREE.Mesh(geom, mat);
    this._scene.add(this._topoMesh);
  }

  _clearTopo() {
    if (this._topoMesh) {
      this._scene.remove(this._topoMesh);
      this._topoMesh.geometry.dispose();
      this._topoMesh.material.dispose();
      this._topoMesh = null;
    }
  }

  toggleTopography(visible) {
    if (this._topoMesh) this._topoMesh.visible = visible;
  }

  // ── Clear scene ───────────────────────────────────────────────────────────
  clear() {
    this._builder.clear();
    this._clearTopo();
    if (this._bhSticks) {
      this._scene.remove(this._bhSticks);
      this._bhSticks.traverse(obj => {
        obj.geometry?.dispose();
        obj.material?.dispose();
      });
      this._bhSticks = null;
    }
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

  // ── Middle-mouse centres view on model ────────────────────────────────────
  _initMiddleMouse() {
    this._canvas.addEventListener('mousedown', e => {
      if (e.button === 1) { e.preventDefault(); this.centreView(); }
    });
    // Suppress the default scroll-on-middle-drag browser behaviour
    this._canvas.addEventListener('auxclick', e => {
      if (e.button === 1) e.preventDefault();
    });
  }

  // ── Getters ───────────────────────────────────────────────────────────────
  get threeScene()  { return this._scene; }
  get threeCamera() { return this._camera; }
  get voxelGroup()  { return this._builder.group; }
  get slicer()      { return this._slicer; }
}
