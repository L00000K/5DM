import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { VoxelBuilder } from './voxel-builder.js';
import { SlicerTool } from './slicer.js';
import { SurfaceManager } from './surfaces.js';
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

    // Surface group — separate so VE can be applied independently
    this._surfaceGroup = new THREE.Group();
    this._scene.add(this._surfaceGroup);
    this._surfaces = new SurfaceManager(this._surfaceGroup);

    this._slicer = new SlicerTool(
      this._scene, this._camera, this._controls, this._renderer,
      planes => {
        this._builder.setClippingPlanes(planes);
        // Apply clipping to surface meshes too
        this._surfaces.getMeshes().forEach(m => {
          m.material.clippingPlanes = planes.length ? planes : null;
          m.material.needsUpdate = true;
        });
      }
    );

    this._vbhMode    = false;
    this._viewMode   = 'voxels';
    this._bhData     = [];

    this._resizeObserver = new ResizeObserver(() => this._onResize());
    this._resizeObserver.observe(this._canvas.parentElement);
    this._onResize();

    this._animate();
    this._initTooltip();
    this._initMiddleMouse();
    this._initVirtualBH();
    this._initBHClick();
    this._initKeyboard();
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

  _addGrid() {
    const grid = new THREE.GridHelper(500, 50, 0x9aaabb, 0xc4d0d8);
    grid.position.y = -0.5;
    this._scene.add(grid);
    return grid;
  }

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
    this._surfaces.build(grid, geoUnits);
    this._applyViewMode();

    if (classifiedBH?.length) {
      this._bhData = classifiedBH.filter(b => !b.synthetic);
      this.addBoreholeSticks(classifiedBH, geoUnits);
    }

    const cx = grid.origin.x + grid.worldWidth  * 0.5;
    const cy = grid.origin.y + grid.worldHeight * 0.5;
    const cz = grid.origin.z + grid.worldDepth  * 0.5;
    const size = Math.max(grid.worldWidth, grid.worldDepth, grid.worldHeight);

    this._controls.target.set(cx, cy, cz);
    this._camera.position.set(cx + size * 0.8, cy + size * 0.5, cz + size * 0.8);
    this._camera.lookAt(cx, cy, cz);
    this._controls.update();

    this._modelBounds = { cx, cy, cz, size, grid, geoUnits };
    this._slicer.setModelBounds(grid);

    log('3D scene updated', 'ok');
  }

  // ── View mode: 'voxels' | 'surfaces' | 'both' ────────────────────────────
  setViewMode(mode) {
    this._viewMode = mode;
    this._applyViewMode();
  }

  _applyViewMode() {
    const showVox  = this._viewMode !== 'surfaces';
    const showSurf = this._viewMode !== 'voxels';
    this._builder.group.visible = showVox;
    this._surfaces.setVisible(showSurf);
    this._syncViewModeButtons();
  }

  _syncViewModeButtons() {
    document.querySelectorAll('.view-mode-btn').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.mode === this._viewMode);
    });
  }

  // ── Unit visibility ───────────────────────────────────────────────────────
  setUnitVisibility(code, visible) {
    this._builder.setUnitVisibility(code, visible);
    this._surfaces.setUnitVisible(code, visible && this._viewMode !== 'voxels');
  }

  // ── Certainty filter ──────────────────────────────────────────────────────
  setCertaintyThreshold(t) {
    this._builder.setCertaintyThreshold(t);
  }

  // ── Transparency ──────────────────────────────────────────────────────────
  setTransparencyMode(enabled, amount) {
    this._builder.setTransparencyMode(enabled, amount);
  }

  setColorFadeMode(enabled, amount) {
    this._builder.setColorFadeMode(enabled, amount);
  }

  setGlobalAlpha(alpha) {
    this._builder.setGlobalAlpha(alpha);
  }

  setSurfaceOpacity(op) {
    this._surfaces.setOpacity(op);
  }

  // ── Vertical exaggeration ─────────────────────────────────────────────────
  setVerticalExaggeration(ve) {
    this._builder.group.scale.y = ve;
    this._surfaceGroup.scale.y  = ve;
    if (this._bhSticks) this._bhSticks.scale.y = ve;
  }

  // ── Centre view ───────────────────────────────────────────────────────────
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
      this._bhSticks.traverse(obj => { obj.geometry?.dispose(); obj.material?.dispose(); });
    }
    const group = new THREE.Group();
    const unitByCode = {};
    geoUnits.forEach(u => { unitByCode[u.code] = u; });
    const radius = 0.6;
    classifiedBH.filter(b => !b.synthetic).forEach(bh => {
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
        mesh.userData.bhId = bh.id;
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
    const positions = new Float32Array((GRID + 1) ** 2 * 3);
    const indices   = [];
    for (let iz = 0; iz <= GRID; iz++) {
      for (let ix = 0; ix <= GRID; ix++) {
        const wx = minX + (maxX - minX) * (ix / GRID);
        const wz = minZ + (maxZ - minZ) * (iz / GRID);
        let bestDist = Infinity, bestY = 0;
        points.forEach(p => {
          const d = (p.x - wx) ** 2 + (p.y - wz) ** 2;
          if (d < bestDist) { bestDist = d; bestY = p.z; }
        });
        const i = (iz * (GRID + 1) + ix) * 3;
        positions[i] = wx; positions[i + 1] = bestY; positions[i + 2] = wz;
      }
    }
    for (let iz = 0; iz < GRID; iz++) {
      for (let ix = 0; ix < GRID; ix++) {
        const a = iz * (GRID + 1) + ix;
        const b = a + 1, c = a + GRID + 1, d = c + 1;
        indices.push(a, c, b, b, c, d);
      }
    }
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geom.setIndex(indices);
    geom.computeVertexNormals();
    this._topoMesh = new THREE.Mesh(geom, new THREE.MeshLambertMaterial({
      color: 0x7aaa88, transparent: true, opacity: 0.32,
      side: THREE.DoubleSide, depthWrite: false,
    }));
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

  // ── Virtual BH mode ───────────────────────────────────────────────────────
  setVBHMode(active) {
    this._vbhMode = active;
    this._canvas.style.cursor = active ? 'crosshair' : '';
    document.getElementById('btn-vbh')?.classList.toggle('active', active);
    if (!active) this._hideLogPopup();
  }

  _initVirtualBH() {
    this._canvas.addEventListener('click', e => {
      if (!this._vbhMode || !this._modelBounds?.grid) return;
      if (e.shiftKey) return; // let shift+click do something else later
      e.stopPropagation();
      const rect = this._canvas.getBoundingClientRect();
      const pt = this._canvasToWorld(e.clientX - rect.left, e.clientY - rect.top);
      if (pt) this._showLogPopup('vbh', pt.x, pt.z, e.clientX - rect.left, e.clientY - rect.top);
    });
  }

  _canvasToWorld(px, py) {
    const rect = this._canvas.getBoundingClientRect();
    const mx = (px / rect.width)  *  2 - 1;
    const my = (py / rect.height) * -2 + 1;
    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera({ x: mx, y: my }, this._camera);

    const allMeshes = [
      ...Object.values(this._builder.meshes ?? {}),
      ...this._surfaces.getMeshes(),
    ];
    const hits = raycaster.intersectObjects(allMeshes);
    if (hits.length) return hits[0].point;

    const grid = this._modelBounds.grid;
    const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -grid.origin.y);
    const out = new THREE.Vector3();
    return raycaster.ray.intersectPlane(plane, out) ? out : null;
  }

  // ── BH stick click → show actual log ─────────────────────────────────────
  _initBHClick() {
    this._canvas.addEventListener('click', e => {
      if (this._vbhMode) return;
      if (!this._bhSticks?.children.length) return;
      const rect = this._canvas.getBoundingClientRect();
      const mx = ((e.clientX - rect.left) / rect.width)  *  2 - 1;
      const my = ((e.clientY - rect.top)  / rect.height) * -2 + 1;
      const raycaster = new THREE.Raycaster();
      raycaster.setFromCamera({ x: mx, y: my }, this._camera);
      const hits = raycaster.intersectObjects(this._bhSticks.children);
      if (!hits.length) return;
      const bhId = hits[0].object.userData.bhId;
      const bh   = this._bhData.find(b => b.id === bhId);
      if (!bh) return;
      this._showLogPopup('bh', bh.x, bh.y, e.clientX - rect.left, e.clientY - rect.top, bh);
    });
  }

  // ── Log popup (shared VBH + BH) ───────────────────────────────────────────
  _showLogPopup(mode, wx, wz, px, py, bhData) {
    const popup = document.getElementById('log-popup');
    if (!popup) return;
    const grid = this._modelBounds?.grid;

    let layers, title, subtitle;

    if (mode === 'bh' && bhData) {
      title    = bhData.id;
      subtitle = `(${wx.toFixed(0)}, ${wz.toFixed(0)}) · GL ${bhData.groundLevel?.toFixed(1)} mAOD`;
      const unitByCode = {};
      (this._modelBounds.geoUnits ?? []).forEach(u => { unitByCode[u.code] = u; });
      layers = bhData.layers.map(l => ({
        unit:  unitByCode[l.unitCode] ?? { code: l.unitCode, name: l.unitCode, color: '#888' },
        from:  bhData.groundLevel - l.base,
        to:    bhData.groundLevel - l.top,
        thick: l.base - l.top,
        cert:  l.certainty ?? 1,
      }));
    } else if (grid) {
      title    = 'Virtual BH';
      subtitle = `(${wx.toFixed(0)}, ${wz.toFixed(0)})`;
      layers   = this._sampleGridColumn(wx, wz, grid);
    }

    if (!layers?.length) return;

    const unitByCode = {};
    (this._modelBounds.geoUnits ?? []).forEach(u => { unitByCode[u.code] = u; });

    popup.querySelector('#log-popup-title').textContent    = title;
    popup.querySelector('#log-popup-subtitle').textContent = subtitle;

    const totalThick = layers.reduce((s, l) => s + l.thick, 0);
    const LOG_H = 260;

    const strips = popup.querySelector('#log-strips');
    const labels = popup.querySelector('#log-labels');
    strips.innerHTML = '';
    labels.innerHTML = '';

    layers.forEach(layer => {
      const pxH = Math.max(4, (layer.thick / totalThick) * LOG_H);
      const strip = document.createElement('div');
      strip.className = 'log-strip';
      strip.style.cssText = `height:${pxH}px; background:${layer.unit.color};`;
      strip.title = `${layer.unit.name}  ${layer.from?.toFixed(1)}–${layer.to?.toFixed(1)} mAOD  cert:${((layer.cert??1)*100).toFixed(0)}%`;
      strips.appendChild(strip);

      const lbl = document.createElement('div');
      lbl.className = 'log-label';
      lbl.style.height = `${pxH}px`;
      lbl.innerHTML = `<span class="log-lbl-code">${layer.unit.code}</span>` +
                      `<span class="log-lbl-depth">${layer.from?.toFixed(1)}–${layer.to?.toFixed(1)}</span>`;
      labels.appendChild(lbl);
    });

    const rect = this._canvas.getBoundingClientRect();
    popup.style.left = `${Math.min(px + 16, rect.width  - 230)}px`;
    popup.style.top  = `${Math.min(py - 10, rect.height - 320)}px`;
    popup.hidden = false;
  }

  _sampleGridColumn(wx, wz, grid) {
    const { nx, ny, nz, unitIds, certainty, cellSize: cs, cellHeight: ch, origin } = grid;
    const ix = Math.floor((wx - origin.x) / cs);
    const iy = Math.floor((wz - origin.z) / cs);
    if (ix < 0 || ix >= nx || iy < 0 || iy >= ny) return [];

    const unitById = {};
    (this._modelBounds.geoUnits ?? []).forEach(u => { unitById[u.id] = u; });

    const layers = [];
    let curUnit = null, curTop = null, certAcc = [];

    for (let iz = nz - 1; iz >= 0; iz--) {
      const flat = ix + iy * nx + iz * nx * ny;
      const uid  = unitIds[flat];
      const unit = uid ? unitById[uid] : null;
      const elev = origin.y + iz * ch + ch;

      if (unit !== curUnit) {
        if (curUnit && curTop !== null) {
          const from = origin.y + iz * ch + ch;
          layers.push({ unit: curUnit, from, to: curTop, thick: curTop - from, cert: certAcc.reduce((a,b)=>a+b,0)/certAcc.length });
        }
        curUnit = unit;
        curTop  = elev;
        certAcc = unit ? [certainty[flat]] : [];
      } else if (unit) {
        certAcc.push(certainty[flat]);
      }
    }
    if (curUnit && curTop !== null) {
      layers.push({ unit: curUnit, from: origin.y, to: curTop, thick: curTop - origin.y, cert: certAcc.reduce((a,b)=>a+b,0)/certAcc.length });
    }
    return layers;
  }

  _hideLogPopup() {
    const p = document.getElementById('log-popup');
    if (p) p.hidden = true;
  }

  // ── Keyboard shortcuts ────────────────────────────────────────────────────
  _initKeyboard() {
    window.addEventListener('keydown', e => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
      switch (e.key) {
        case 'v': case 'V': this.setViewMode('voxels');   break;
        case 's': case 'S': this.setViewMode('surfaces'); break;
        case 'b': case 'B': this.setViewMode('both');     break;
        case 'k': case 'K': window.dispatchEvent(new CustomEvent('geomodel:toggle-vbh')); break;
        case 'Escape': this.setVBHMode(false); this._hideLogPopup(); break;
      }
    });
  }

  // ── Clear scene ───────────────────────────────────────────────────────────
  clear() {
    this._builder.clear();
    this._surfaces.clear();
    this._clearTopo();
    if (this._bhSticks) {
      this._scene.remove(this._bhSticks);
      this._bhSticks.traverse(obj => { obj.geometry?.dispose(); obj.material?.dispose(); });
      this._bhSticks = null;
    }
    this._bhData = [];
    this._hideLogPopup();
  }

  // ── Voxel hover tooltip ───────────────────────────────────────────────────
  _initTooltip() {
    const tooltip  = document.getElementById('voxel-tooltip');
    const raycaster = new THREE.Raycaster();
    const mouse     = new THREE.Vector2();

    this._canvas.addEventListener('mousemove', evt => {
      if (!this._modelBounds || !tooltip || this._vbhMode) return;
      const rect = this._canvas.getBoundingClientRect();
      mouse.x = ((evt.clientX - rect.left)  / rect.width)  *  2 - 1;
      mouse.y = -((evt.clientY - rect.top) / rect.height) *  2 + 1;

      raycaster.setFromCamera(mouse, this._camera);
      const meshes = Object.values(this._builder.meshes ?? {});
      const hits = raycaster.intersectObjects(meshes);

      if (hits.length) {
        const hit  = hits[0];
        const unit = this._builder.units.find(u => u.code === hit.object.userData.unitCode);
        const iIdx = hit.instanceId;
        let cert = '—';
        if (this._modelBounds?.grid && iIdx !== undefined) {
          const { nx, ny, nz, unitIds, certainty: certs } = this._modelBounds.grid;
          const unitById = {};
          this._builder.units.forEach(u => { unitById[u.id] = u; });
          let cnt = 0;
          for (let idx = 0; idx < unitIds.length; idx++) {
            const u = unitById[unitIds[idx]];
            if (u?.code === hit.object.userData.unitCode) {
              if (cnt === iIdx) { cert = `${(certs[idx] * 100).toFixed(0)}%`; break; }
              cnt++;
            }
          }
        }
        tooltip.hidden = false;
        tooltip.style.left = `${evt.clientX - rect.left + 14}px`;
        tooltip.style.top  = `${evt.clientY - rect.top  -  8}px`;
        tooltip.innerHTML = `
          <div class="tooltip-title">${unit?.name ?? hit.object.userData.unitCode}</div>
          <div class="tooltip-row"><span>Code</span><span class="tooltip-val">${hit.object.userData.unitCode}</span></div>
          <div class="tooltip-row"><span>Certainty</span><span class="tooltip-val">${cert}</span></div>
          <div class="tooltip-row"><span>Pos</span><span class="tooltip-val">(${hit.point.x.toFixed(0)}, ${hit.point.y.toFixed(0)}, ${hit.point.z.toFixed(0)})</span></div>`;
      } else {
        tooltip.hidden = true;
      }
    });
    this._canvas.addEventListener('mouseleave', () => { if (tooltip) tooltip.hidden = true; });
  }

  // ── Middle-mouse centres view ─────────────────────────────────────────────
  _initMiddleMouse() {
    this._canvas.addEventListener('mousedown', e => {
      if (e.button === 1) { e.preventDefault(); this.centreView(); }
    });
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
