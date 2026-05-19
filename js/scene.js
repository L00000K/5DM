import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { VoxelBuilder } from './voxel-builder.js';
import { SlicerTool } from './slicer.js';
import { SurfaceManager } from './surfaces.js';
import { log } from './app.js';
import { buildDeviationPath, interpolateAtDepth } from './deviation.js';

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
      preserveDrawingBuffer: true,
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
        // renderer.clippingPlanes clips standard materials (surfaces, BH sticks).
        // Voxel ShaderMaterial uses custom world-space uniforms (uClipPlanes/
        // uNumClipPlanes) to avoid compile-time NUM_CLIPPING_PLANES = 0 baking.
        this._renderer.clippingPlanes = planes;
        this._builder.setClippingPlanes(planes);
      }
    );

    this._vbhMode        = false;
    this._viewMode       = 'voxels';
    this._bhData         = [];
    this._measureMode    = false;
    this._measurePts     = [];
    this._measureLines   = [];
    this._annotationMode = false;
    this._annotations    = [];

    this._resizeObserver = new ResizeObserver(() => this._onResize());
    this._resizeObserver.observe(this._canvas.parentElement);
    this._onResize();

    this._northArrow = document.getElementById('north-arrow');
    this._northCtx   = this._northArrow?.getContext('2d');

    this._animate();
    this._initTooltip();
    this._initMiddleMouse();
    this._initVirtualBH();
    this._initBHClick();
    this._initKeyboard();
    this._initMeasure();
    this._initAnnotationMode();
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
    this._updateScaleBar();
    this._updateNorthArrow();
    this._updateAnnotations();
  }

  // ── Scale bar (updates every frame) ──────────────────────────────────────
  _updateScaleBar() {
    const bar   = document.getElementById('scale-bar');
    const line  = document.getElementById('scale-bar-line');
    const label = document.getElementById('scale-bar-label');
    if (!bar || !line || !label) return;

    const w = this._canvas.clientWidth;
    if (!w) return;

    // Project a world-space horizontal segment at the model centre
    const centre = this._controls.target;
    const cam    = this._camera;

    const p1 = new THREE.Vector3(centre.x - 0.5, centre.y, centre.z).project(cam);
    const p2 = new THREE.Vector3(centre.x + 0.5, centre.y, centre.z).project(cam);
    const pxPerM = Math.abs(p2.x - p1.x) * w * 0.5;

    if (pxPerM < 0.001) { bar.hidden = true; return; }

    // Choose a nice round scale
    const NICE = [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 5000];
    const targetPx = 80;
    const rawM  = targetPx / pxPerM;
    const scaleM = NICE.find(n => n >= rawM) ?? NICE[NICE.length - 1];
    const barPx  = Math.round(scaleM * pxPerM);

    if (barPx < 20 || barPx > 300) { bar.hidden = true; return; }
    bar.hidden = false;
    line.style.width  = `${barPx}px`;
    label.textContent = scaleM >= 1000 ? `${scaleM / 1000} km` : `${scaleM} m`;
  }

  // ── Build voxel model ─────────────────────────────────────────────────────
  buildVoxels(grid, geoUnits, classifiedBH) {
    this._grid.position.y   = grid.origin.y - 1;
    this._grid.scale.setScalar(Math.max(grid.worldWidth, grid.worldDepth) / 500);
    this._axes.position.set(grid.origin.x - 20, grid.origin.y, grid.origin.z - 20);

    this._builder.build(grid, geoUnits);
    this._surfaces.build(grid, geoUnits);
    this._applyViewMode();

    // Re-apply any active slicer planes to the freshly-built meshes
    this._slicer.reapply();

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

  // ── Concept domain 3D visualisation ──────────────────────────────────────
  // Draws semi-transparent coloured bounding boxes for spatially-constrained
  // concepts (domain.type === 'bbox'). Global concepts show no 3D marker.
  drawConceptDomains(conceptStore) {
    if (this._conceptGroup) {
      this._scene.remove(this._conceptGroup);
      this._conceptGroup.traverse(obj => {
        obj.geometry?.dispose();
        if (Array.isArray(obj.material)) obj.material.forEach(m => m.dispose());
        else obj.material?.dispose();
      });
      this._conceptGroup = null;
    }
    if (!conceptStore || conceptStore.isEmpty) return;

    const group = new THREE.Group();
    const bounds = this._modelBounds?.grid;
    if (!bounds) return;

    const modelMinY = bounds.origin.y;
    const modelMaxY = bounds.origin.y + bounds.worldHeight;

    conceptStore.concepts.forEach((c, ci) => {
      if (c.domain?.type !== 'bbox') return;
      const { minX = 0, maxX = 0, minY = 0, maxY = 0 } = c.domain;
      const w = Math.max(0.5, maxX - minX);
      const d = Math.max(0.5, maxY - minY);
      const h = modelMaxY - modelMinY;
      const cx = (minX + maxX) / 2;
      const cy = (modelMinY + modelMaxY) / 2;
      const cz = (minY + maxY) / 2;

      // Hue cycling per concept index (golden angle)
      const hue = ((ci * 137.508) % 360) / 360;
      const col = new THREE.Color().setHSL(hue, 0.8, 0.5);

      // Wireframe box edges
      const boxGeo = new THREE.BoxGeometry(w, h, d);
      const edges  = new THREE.EdgesGeometry(boxGeo);
      const lineMat = new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: 0.85 });
      const wire = new THREE.LineSegments(edges, lineMat);
      wire.position.set(cx, cy, cz);
      group.add(wire);
      boxGeo.dispose();

      // Translucent fill
      const fillGeo = new THREE.BoxGeometry(w, h, d);
      const fillMat = new THREE.MeshBasicMaterial({
        color: col, transparent: true, opacity: 0.04, depthWrite: false, side: THREE.FrontSide,
      });
      const fill = new THREE.Mesh(fillGeo, fillMat);
      fill.position.set(cx, cy, cz);
      group.add(fill);

      // Label sprite at top edge of bbox domain
      const label = this._makeLabelSprite(c.description.slice(0, 28), 28, 'rgba(0,0,0,0.65)', col.getStyle());
      label.position.set(cx, modelMaxY + 3, cz);
      label.scale.set(20, 4, 1);
      group.add(label);
    });

    this._conceptGroup = group;
    this._scene.add(group);
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

  // ── Geology on/off toggle ─────────────────────────────────────────────────
  setGeologyVisible(visible) {
    this._builder.group.visible = visible;
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

  colorByParameter(paramName, geoUnits, paramGrid = null) {
    return this._builder.colorByParameter(paramName, geoUnits, paramGrid);
  }

  resetUnitColors() {
    this._builder.resetUnitColors();
  }

  colorByBoundaryUncertainty() {
    this._builder?.colorByBoundaryUncertainty();
  }

  colorByCoverage() {
    return this._builder?.colorByCoverage() ?? false;
  }

  colorByEntropy(nUnits) {
    return this._builder?.colorByEntropy(nUnits) ?? false;
  }

  colorByDominantConcept(conceptStore) {
    return this._builder?.colorByDominantConcept(conceptStore) ?? false;
  }

  colorBySingleConcept(conceptId, conceptStore) {
    return this._builder?.colorBySingleConcept(conceptId, conceptStore) ?? false;
  }

  colorByConceptStability(runUnitIds) {
    return this._builder?.colorByConceptStability(runUnitIds) ?? false;
  }

  colorByConceptInfluence() {
    return this._builder?.colorByConceptInfluence() ?? false;
  }

  colorByGeologicalAge(geoUnits, periodColorMap) {
    this._builder?.colorByGeologicalAge(geoUnits, periodColorMap);
  }

  colorByGradeShell(paramName, minVal, maxVal, mode, highlightHex, dimOthers, geoUnits) {
    return this._builder?.colorByGradeShell(paramName, minVal, maxVal, mode, highlightHex, dimOthers, geoUnits) ?? null;
  }

  // ── 3D Orientation / dip symbols ──────────────────────────────────────────
  // orientations: [{x, y, elev, dip, dipDir, color}] — dip in degrees from horizontal,
  //               dipDir in degrees clockwise from north (same as dip direction convention).
  showOrientationSymbols(orientations) {
    if (this._orientGroup) {
      this._scene.remove(this._orientGroup);
      this._orientGroup.traverse(o => { o.geometry?.dispose(); o.material?.dispose(); });
    }
    if (!orientations?.length) { this._orientGroup = null; return; }

    const group = new THREE.Group();
    const DEG = Math.PI / 180;
    const RADIUS = 2.5;

    for (const o of orientations) {
      const { x, y, elev, dip, dipDir, color = '#ff8800' } = o;
      const dipRad = (dip ?? 0) * DEG;
      const azRad  = (dipDir ?? 0) * DEG; // clockwise from north

      // Create a disc (circle) with a tick mark showing dip direction
      const discGeom = new THREE.CircleGeometry(RADIUS, 16);
      const mat = new THREE.MeshBasicMaterial({
        color, side: THREE.DoubleSide, transparent: true, opacity: 0.75,
      });
      const disc = new THREE.Mesh(discGeom, mat);

      // Strike direction is dipDir - 90° (right-hand rule)
      const strikeRad = azRad - Math.PI / 2;

      // Rotate disc: start flat (XZ plane), rotate around strike axis by dip angle
      disc.rotation.x = Math.PI / 2;   // lie in XZ plane
      disc.rotation.y = -strikeRad;    // align strike
      disc.rotateOnWorldAxis(new THREE.Vector3(Math.cos(strikeRad), 0, Math.sin(strikeRad)), dipRad);

      disc.position.set(x, elev, y);
      group.add(disc);

      // Dip tick: a short line in the dip direction from centre
      const tickEnd = new THREE.Vector3(
        x + RADIUS * 0.8 * Math.sin(azRad) * Math.cos(dipRad),
        elev - RADIUS * 0.8 * Math.sin(dipRad),
        y + RADIUS * 0.8 * Math.cos(azRad) * Math.cos(dipRad),
      );
      const tickGeom = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(x, elev, y), tickEnd,
      ]);
      group.add(new THREE.Line(tickGeom, new THREE.LineBasicMaterial({ color: 0x222222 })));
    }

    this._orientGroup = group;
    this._scene.add(group);
    return orientations.length;
  }

  toggleOrientationSymbols(visible) {
    if (this._orientGroup) this._orientGroup.visible = visible;
  }

  setSurfaceOpacity(op) {
    this._surfaces.setOpacity(op);
    this._surfaces.setMCOpacity(op);
  }

  buildIsosurfaces(grid, geoUnits, opacity, onProgress) {
    this._surfaces.buildIsosurfaces(grid, geoUnits, opacity, onProgress);
  }

  setIsosurfacesVisible(v) {
    this._surfaces.setMCVisible(v);
  }

  setIsosurfaceUnitVisible(code, v) {
    this._surfaces.setMCUnitVisible(code, v);
  }

  buildUncertaintySurface(grid, threshold, opacity) {
    this._surfaces.buildUncertaintySurface(grid, threshold, opacity);
  }

  setUncertaintySurfaceVisible(v) {
    this._surfaces.setUncertaintyVisible(v);
  }

  setUncertaintySurfaceOpacity(op) {
    this._surfaces.setUncertaintyOpacity(op);
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

  // ── Camera bookmarks ────────────────────────────────────────────────────────
  getCameraState() {
    const c = this._camera, t = this._controls.target;
    return {
      position: { x: c.position.x, y: c.position.y, z: c.position.z },
      target:   { x: t.x, y: t.y, z: t.z },
      fov: c.fov,
    };
  }

  setCameraState(state) {
    if (!state) return;
    const { position: p, target: t, fov } = state;
    this._camera.position.set(p.x, p.y, p.z);
    if (fov) { this._camera.fov = fov; this._camera.updateProjectionMatrix(); }
    this._controls.target.set(t.x, t.y, t.z);
    this._controls.update();
  }

  // ── BH label sprite helper ────────────────────────────────────────────────
  _makeLabelSprite(text, fontSize = 48, bgColor = 'rgba(0,0,0,0.72)', fgColor = '#fff') {
    const canvas = document.createElement('canvas');
    const ctx    = canvas.getContext('2d');
    ctx.font     = `bold ${fontSize}px Arial`;
    const w = ctx.measureText(text).width + fontSize;
    canvas.width  = Math.pow(2, Math.ceil(Math.log2(w + 8)));
    canvas.height = fontSize * 2;
    ctx.font      = `bold ${fontSize}px Arial`;
    ctx.fillStyle = bgColor;
    const pad = fontSize * 0.25;
    const tw  = ctx.measureText(text).width;
    ctx.beginPath();
    ctx.roundRect(canvas.width/2 - tw/2 - pad, pad, tw + pad*2, fontSize + pad, fontSize*0.3);
    ctx.fill();
    ctx.fillStyle = fgColor;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, canvas.width / 2, canvas.height / 2);
    const tex = new THREE.CanvasTexture(canvas);
    const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false });
    const spr = new THREE.Sprite(mat);
    spr.scale.set(canvas.width / canvas.height * 4, 4, 1);
    return spr;
  }

  // ── Borehole sticks ───────────────────────────────────────────────────────
  addBoreholeSticks(classifiedBH, geoUnits) {
    if (this._bhSticks) {
      this._scene.remove(this._bhSticks);
      this._bhSticks.traverse(obj => {
        obj.geometry?.dispose();
        if (obj.material) {
          obj.material.map?.dispose();
          obj.material.dispose();
        }
      });
    }
    const group = new THREE.Group();
    const unitByCode = {};
    geoUnits.forEach(u => { unitByCode[u.code] = u; });
    const radius   = 0.6;
    const maxSPT   = 60;    // N value that fills the full bar
    const sptScale = 3.5;   // bar length at N=maxSPT in metres

    classifiedBH.filter(b => !b.synthetic).forEach(bh => {
      if (!bh.layers?.length) return;
      const gl = bh.groundLevel ?? 0;

      // Build deviation path (null if no survey data → vertical)
      const devPath = (bh.deviation?.length >= 2)
        ? buildDeviationPath(bh.x, bh.y, gl, bh.deviation)
        : null;

      bh.layers.forEach((layer, li) => {
        const unit = unitByCode[layer.unitCode];
        if (!unit) return;

        const topDepth  = layer.top;
        const baseDepth = layer.base;
        const midDepth  = (topDepth + baseDepth) * 0.5;

        let topX, topY, topElev3d, baseX, baseY, baseElev3d;
        if (devPath) {
          const tp = interpolateAtDepth(devPath, topDepth);
          const bp = interpolateAtDepth(devPath, baseDepth);
          topX = tp.x;  topY = tp.y;  topElev3d  = tp.elev;
          baseX = bp.x; baseY = bp.y; baseElev3d = bp.elev;
        } else {
          topX = baseX = bh.x; topY = baseY = bh.y;
          topElev3d  = gl - topDepth;
          baseElev3d = gl - baseDepth;
        }

        const midX    = (topX + baseX) / 2;
        const midY    = (topY + baseY) / 2;
        const midElev = (topElev3d + baseElev3d) / 2;

        // Direction vector for inclined cylinder
        const dir = new THREE.Vector3(
          baseX - topX, baseElev3d - topElev3d, baseY - topY
        );
        const segLen = Math.max(dir.length(), 0.01);
        dir.normalize();

        const geom = new THREE.CylinderGeometry(radius, radius, segLen, 8);
        const mat  = new THREE.MeshLambertMaterial({ color: unit.color });
        const mesh = new THREE.Mesh(geom, mat);
        mesh.position.set(midX, midElev, midY);
        if (devPath) {
          const yAxis = new THREE.Vector3(0, 1, 0);
          mesh.quaternion.setFromUnitVectors(yAxis, dir);
        }
        mesh.userData.bhId = bh.id;
        group.add(mesh);

        // Formation contact ring at top of this layer (not first)
        if (li > 0) {
          const ringGeom = new THREE.CylinderGeometry(radius * 2.2, radius * 2.2, 0.15, 12, 1, true);
          const ringMat  = new THREE.MeshBasicMaterial({ color: '#ffffff', side: THREE.DoubleSide });
          const ring = new THREE.Mesh(ringGeom, ringMat);
          ring.position.set(topX, topElev3d, topY);
          group.add(ring);
        }

        // SPT N horizontal bar extending in +X direction from mid position
        if (layer.sptN != null && layer.sptN > 0) {
          const barLen = Math.min(layer.sptN / maxSPT, 1) * sptScale;
          const barH   = Math.min(segLen * 0.6, 0.8);
          const barGeom = new THREE.BoxGeometry(barLen, barH, 0.3);
          const barMat  = new THREE.MeshLambertMaterial({ color: 0x2a4a6a });
          const bar = new THREE.Mesh(barGeom, barMat);
          bar.position.set(midX + radius + barLen / 2, midElev, midY);
          group.add(bar);
        }
      });

      // Deviation trace polyline (only for deviated boreholes)
      if (devPath && devPath.length >= 2) {
        const pts = devPath.map(p => new THREE.Vector3(p.x, p.elev, p.y));
        const traceGeom = new THREE.BufferGeometry().setFromPoints(pts);
        const traceMat  = new THREE.LineBasicMaterial({ color: 0xffaa00, linewidth: 2 });
        group.add(new THREE.Line(traceGeom, traceMat));
      }

      // BH ID label sprite at collar + 2m
      const labelElev = gl + 2;
      const lbl = this._makeLabelSprite(bh.id, 36);
      lbl.position.set(bh.x, labelElev, bh.y);
      group.add(lbl);

      // Leader line from collar to label
      const lineGeom = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(bh.x, gl, bh.y),
        new THREE.Vector3(bh.x, labelElev - 2, bh.y),
      ]);
      const line = new THREE.Line(lineGeom, new THREE.LineBasicMaterial({ color: 0x888888 }));
      group.add(line);
    });

    this._bhSticks = group;
    this._scene.add(group);
  }

  toggleBoreholeSticks(visible) {
    if (this._bhSticks) this._bhSticks.visible = visible;
  }

  // Show boreholes in 3D immediately on data load, before a model is built
  showBoreholes(classifiedBH, geoUnits) {
    if (!classifiedBH?.length) return;
    this.addBoreholeSticks(classifiedBH, geoUnits);

    // Fit camera to borehole extents
    const bhs = classifiedBH.filter(b => !b.synthetic);
    if (!bhs.length) return;
    const xs = bhs.map(b => b.x);
    const ys = bhs.map(b => b.y);
    const tops = bhs.map(b => b.groundLevel ?? 0);
    const bots = bhs.map(b => (b.groundLevel ?? 0) - (b.depth ?? 10));
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const minE = Math.min(...bots), maxE = Math.max(...tops);

    const cx = (minX + maxX) / 2;
    const cy = (minE + maxE) / 2;
    const cz = (minY + maxY) / 2;
    const size = Math.max(maxX - minX, maxY - minY, maxE - minE, 20);

    this._grid.position.y = minE - 1;
    this._grid.scale.setScalar(Math.max(maxX - minX, maxY - minY, 100) / 500);
    this._axes.position.set(minX - 20, minE, minY - 20);

    this._controls.target.set(cx, cy, cz);
    this._camera.position.set(cx + size * 0.9, cy + size * 0.55, cz + size * 0.9);
    this._camera.lookAt(cx, cy, cz);
    this._controls.update();
  }

  // ── Interpolated GWT surface ─────────────────────────────────────────────
  // boreholes: array with .gwtDepth (m) and .groundLevel (mAOD), .x, .y (m)
  // grid: voxel grid for bounding box / cell size
  showInterpolatedGWT(boreholes, grid) {
    this._clearInterpGWT();
    const bhs = boreholes.filter(b => b.gwtDepth != null && !b.synthetic);
    if (!bhs.length || !grid) return;

    const { nx, ny, cellSize: cs, origin } = grid;
    const GRID = Math.min(nx, ny, 48);
    const pts   = bhs.map(b => ({ x: b.x, y: b.y, elev: (b.groundLevel ?? 0) - b.gwtDepth }));

    const vW = cs * nx, vD = cs * ny;
    const positions = new Float32Array((GRID + 1) ** 2 * 3);
    const indices   = [];

    for (let iz = 0; iz <= GRID; iz++) {
      for (let ix = 0; ix <= GRID; ix++) {
        const wx = origin.x + (ix / GRID) * vW;
        const wz = origin.z + (iz / GRID) * vD;
        // IDW from BH GWT elevations
        let wSum = 0, vSum = 0;
        for (const pt of pts) {
          const d = Math.max(Math.hypot(pt.x - wx, pt.y - wz), 0.01);
          const w = 1 / (d * d);
          vSum += pt.elev * w;
          wSum += w;
        }
        const elev = wSum > 0 ? vSum / wSum : pts[0].elev;
        const i = (iz * (GRID + 1) + ix) * 3;
        positions[i] = wx; positions[i+1] = elev; positions[i+2] = wz;
      }
    }
    for (let iz = 0; iz < GRID; iz++) {
      for (let ix = 0; ix < GRID; ix++) {
        const a = iz*(GRID+1)+ix, b = a+1, c = a+GRID+1, d = c+1;
        indices.push(a, c, b, b, c, d);
      }
    }

    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geom.setIndex(indices);
    geom.computeVertexNormals();
    this._interpGWTMesh = new THREE.Mesh(geom, new THREE.MeshLambertMaterial({
      color: 0x3080d0, transparent: true, opacity: 0.28,
      side: THREE.DoubleSide, depthWrite: false,
    }));
    this._scene.add(this._interpGWTMesh);
    return this._interpGWTMesh;
  }

  _clearInterpGWT() {
    if (this._interpGWTMesh) {
      this._scene.remove(this._interpGWTMesh);
      this._interpGWTMesh.geometry.dispose();
      this._interpGWTMesh.material.dispose();
      this._interpGWTMesh = null;
    }
  }

  toggleInterpolatedGWT(visible) {
    if (this._interpGWTMesh) this._interpGWTMesh.visible = visible;
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
        sptN:  l.sptN ?? null,
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
    const sptDiv = popup.querySelector('#log-spt');
    strips.innerHTML = '';
    labels.innerHTML = '';
    if (sptDiv) sptDiv.innerHTML = '';

    const hasSPT = layers.some(l => l.sptN !== null && l.sptN !== undefined);
    if (sptDiv) sptDiv.hidden = !hasSPT;
    const sptLegend = popup.querySelector('#log-spt-legend');
    if (sptLegend) sptLegend.hidden = !hasSPT;
    const maxN = hasSPT ? Math.max(...layers.map(l => l.sptN ?? 0), 60) : 60;

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

      if (hasSPT && sptDiv) {
        const sptBar = document.createElement('div');
        sptBar.className = 'log-spt-bar';
        sptBar.style.height = `${pxH}px`;
        if (layer.sptN !== null && layer.sptN !== undefined) {
          const pct = Math.min(100, (layer.sptN / maxN) * 100);
          sptBar.innerHTML = `<div class="log-spt-fill" style="width:${pct}%"></div>` +
                             `<span class="log-spt-val">${layer.sptN}</span>`;
        }
        sptDiv.appendChild(sptBar);
      }
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
        case 'm': case 'M': window.dispatchEvent(new CustomEvent('geomodel:toggle-measure')); break;
        case 'a': case 'A': window.dispatchEvent(new CustomEvent('geomodel:toggle-annotate')); break;
        case 'Escape':
          this.setVBHMode(false);
          this.setMeasureMode(false);
          this.setAnnotationMode(false);
          this._hideLogPopup();
          break;
      }
    });
  }

  // ── Clear scene ───────────────────────────────────────────────────────────
  clear() {
    this._builder.clear();
    this._surfaces.clear();
    this._clearTopo();
    this._clearMeasure();
    this.clearAnnotations();
    this.setGroundwaterTable(null);
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
        // Find top elevation of this unit at this XY column
        let unitTopStr = '—';
        if (this._modelBounds?.grid && hit.object.userData.unitCode) {
          const g    = this._modelBounds.grid;
          const code = hit.object.userData.unitCode;
          const ub   = {};
          this._builder.units.forEach(u => { ub[u.id] = u; });
          const ix = Math.floor((hit.point.x - g.origin.x) / g.cellSize);
          const iy = Math.floor((hit.point.z - g.origin.z) / g.cellSize);
          if (ix >= 0 && ix < g.nx && iy >= 0 && iy < g.ny) {
            let topIz = -1;
            for (let iz = g.nz - 1; iz >= 0; iz--) {
              const u = ub[g.unitIds[ix + iy * g.nx + iz * g.nx * g.ny]];
              if (u?.code === code) { topIz = iz; break; }
            }
            if (topIz >= 0) unitTopStr = `${(g.origin.y + (topIz + 1) * g.cellHeight).toFixed(1)} mAOD`;
          }
        }
        tooltip.innerHTML = `
          <div class="tooltip-title">${unit?.name ?? hit.object.userData.unitCode}</div>
          <div class="tooltip-row"><span>Code</span><span class="tooltip-val">${hit.object.userData.unitCode}</span></div>
          <div class="tooltip-row"><span>Top of unit</span><span class="tooltip-val">${unitTopStr}</span></div>
          <div class="tooltip-row"><span>Certainty</span><span class="tooltip-val">${cert}</span></div>
          <div class="tooltip-row"><span>Pos</span><span class="tooltip-val">(${hit.point.x.toFixed(0)}, ${hit.point.y.toFixed(0)}, ${hit.point.z.toFixed(0)})</span></div>`;

        // Emit hover event for traceability panel
        window.dispatchEvent(new CustomEvent('geomodel:voxel-hover', {
          detail: { worldX: hit.point.x, worldY: hit.point.z, worldZ: hit.point.y, unitCode: hit.object.userData.unitCode },
        }));
      } else {
        tooltip.hidden = true;
        window.dispatchEvent(new CustomEvent('geomodel:voxel-hover', { detail: null }));
      }
    });
    this._canvas.addEventListener('mouseleave', () => {
      if (tooltip) tooltip.hidden = true;
      window.dispatchEvent(new CustomEvent('geomodel:voxel-hover', { detail: null }));
    });
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

  // ── North arrow ──────────────────────────────────────────────────────────
  _updateNorthArrow() {
    const cvs = this._northArrow;
    const ctx = this._northCtx;
    if (!cvs || !ctx) return;

    cvs.hidden = false;
    const S = 48, cx = S / 2, cy = S / 2, R = 19;

    // Get camera direction projected onto XZ plane
    const dir = new THREE.Vector3();
    this._camera.getWorldDirection(dir);
    dir.y = 0; dir.normalize();
    // North is -Z in world space (convention). Angle of North in screen space:
    const angle = Math.atan2(-dir.x, -dir.z); // angle to rotate arrow to point North

    ctx.clearRect(0, 0, S, S);

    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(angle);

    // North (red) half
    ctx.beginPath();
    ctx.moveTo(0, -R); ctx.lineTo(-6, 4); ctx.lineTo(0, 0); ctx.closePath();
    ctx.fillStyle = '#d03030';
    ctx.fill();
    ctx.beginPath();
    ctx.moveTo(0, -R); ctx.lineTo(6, 4); ctx.lineTo(0, 0); ctx.closePath();
    ctx.fillStyle = '#b02020';
    ctx.fill();

    // South (grey) half
    ctx.beginPath();
    ctx.moveTo(0, R); ctx.lineTo(-6, -4); ctx.lineTo(0, 0); ctx.closePath();
    ctx.fillStyle = '#9aaabb';
    ctx.fill();
    ctx.beginPath();
    ctx.moveTo(0, R); ctx.lineTo(6, -4); ctx.lineTo(0, 0); ctx.closePath();
    ctx.fillStyle = '#8898a8';
    ctx.fill();

    ctx.restore();

    // N label
    ctx.fillStyle = '#1c2a38';
    ctx.font = 'bold 9px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('N', cx, cy);
  }

  // ── Measurement tool ─────────────────────────────────────────────────────
  setMeasureMode(active) {
    this._measureMode = active;
    this._canvas.style.cursor = active ? 'crosshair' : (this._vbhMode ? 'crosshair' : '');
    document.getElementById('btn-measure')?.classList.toggle('active', active);
    if (!active) this._clearMeasure();
  }

  _initMeasure() {
    this._canvas.addEventListener('click', e => {
      if (!this._measureMode) return;
      if (this._vbhMode) return;
      e.stopPropagation();
      const rect = this._canvas.getBoundingClientRect();
      const pt = this._canvasToWorld(e.clientX - rect.left, e.clientY - rect.top);
      if (!pt) return;

      this._measurePts.push(pt.clone());
      if (this._measurePts.length === 1) {
        this._addMarker(pt, '#ff6633');
        this._showMeasureTooltip('Click second point to measure distance', e.clientX - rect.left, e.clientY - rect.top);
      } else if (this._measurePts.length === 2) {
        this._addMarker(pt, '#ff6633');
        this._drawMeasureLine(this._measurePts[0], this._measurePts[1]);
        const dx = pt.x - this._measurePts[0].x;
        const dy = pt.y - this._measurePts[0].y;
        const dz = pt.z - this._measurePts[0].z;
        const horiz = Math.hypot(dx, dz);
        const total = Math.hypot(dx, dy, dz);
        const bearing = ((Math.atan2(dx, dz) * 180 / Math.PI) + 360) % 360;
        const msg = `Distance: ${total.toFixed(1)} m  Horiz: ${horiz.toFixed(1)} m  ΔZ: ${dy.toFixed(1)} m  Bearing: ${bearing.toFixed(0)}°`;
        this._showMeasureTooltip(msg, e.clientX - rect.left, e.clientY - rect.top, true);
        this._measurePts = [];  // reset for next pair
      }
    });
  }

  _addMarker(pt, color) {
    const geo = new THREE.SphereGeometry(0.5, 8, 8);
    const mat = new THREE.MeshBasicMaterial({ color });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.copy(pt);
    this._scene.add(mesh);
    this._measureLines.push(mesh);
  }

  _drawMeasureLine(p1, p2) {
    const pts = [p1, p2];
    const geo = new THREE.BufferGeometry().setFromPoints(pts);
    const mat = new THREE.LineBasicMaterial({ color: '#ff6633', linewidth: 2 });
    const line = new THREE.Line(geo, mat);
    this._scene.add(line);
    this._measureLines.push(line);
  }

  _showMeasureTooltip(msg, px, py, persist = false) {
    const tt = document.getElementById('measure-tooltip');
    if (!tt) return;
    tt.textContent = msg;
    tt.hidden = false;
    const rect = this._canvas.getBoundingClientRect();
    tt.style.left = `${Math.min(px + 14, rect.width  - 310)}px`;
    tt.style.top  = `${Math.min(py -  8, rect.height -  60)}px`;
    if (persist) setTimeout(() => { if (tt) tt.hidden = true; }, 6000);
  }

  _clearMeasure() {
    this._measureLines.forEach(obj => {
      this._scene.remove(obj);
      obj.geometry?.dispose();
      obj.material?.dispose();
    });
    this._measureLines = [];
    this._measurePts   = [];
    const tt = document.getElementById('measure-tooltip');
    if (tt) tt.hidden = true;
  }

  // ── Annotation labels ─────────────────────────────────────────────────────
  setAnnotationMode(active) {
    this._annotationMode = active;
    const cur = active ? 'text' : (this._vbhMode || this._measureMode ? 'crosshair' : '');
    this._canvas.style.cursor = cur;
    document.getElementById('btn-annotate')?.classList.toggle('active', active);
  }

  _initAnnotationMode() {
    this._canvas.addEventListener('click', e => {
      if (!this._annotationMode) return;
      if (this._measureMode || this._vbhMode) return;
      e.stopPropagation();
      const rect = this._canvas.getBoundingClientRect();
      const pt = this._canvasToWorld(e.clientX - rect.left, e.clientY - rect.top);
      if (!pt) return;
      const text = prompt('Enter annotation label:', '');
      if (!text?.trim()) return;
      this._addAnnotation(pt, text.trim());
      this.setAnnotationMode(false);
    });
  }

  _addAnnotation(pos, text) {
    const viewer = document.getElementById('viewer-panel');
    if (!viewer) return;
    const div = document.createElement('div');
    div.className = 'annotation-label';
    div.innerHTML = `<span class="annotation-text">${_escHtml(text)}</span>` +
                    `<button class="annotation-close" title="Remove">×</button>`;
    viewer.appendChild(div);
    div.querySelector('.annotation-close')?.addEventListener('click', () => {
      try { viewer.removeChild(div); } catch (_) {}
      this._annotations = this._annotations.filter(a => a.div !== div);
    });
    this._annotations.push({ pos: pos.clone(), div });
  }

  _updateAnnotations() {
    if (!this._annotations.length) return;
    const rect = this._canvas.getBoundingClientRect();
    const w = rect.width, h = rect.height;
    for (const ann of this._annotations) {
      const p = ann.pos.clone().project(this._camera);
      if (p.z > 1) { ann.div.style.display = 'none'; continue; }
      ann.div.style.display = 'block';
      ann.div.style.left = `${Math.round((p.x + 1) / 2 * w)}px`;
      ann.div.style.top  = `${Math.round((-p.y + 1) / 2 * h)}px`;
    }
  }

  clearAnnotations() {
    const viewer = document.getElementById('viewer-panel');
    for (const ann of this._annotations) {
      try { viewer?.removeChild(ann.div); } catch (_) {}
    }
    this._annotations = [];
  }

  // ── Screenshot ───────────────────────────────────────────────────────────
  takeScreenshot(filename = 'geomodel.png') {
    this._renderer.render(this._scene, this._camera);
    const url = this._canvas.toDataURL('image/png');
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
  }

  // ── Groundwater table plane ───────────────────────────────────────────────
  setGroundwaterTable(elevM) {
    if (this._gwtMesh) {
      this._scene.remove(this._gwtMesh);
      this._gwtMesh.geometry.dispose();
      this._gwtMesh.material.dispose();
      this._gwtMesh = null;
    }
    if (elevM === null || elevM === undefined || !this._modelBounds?.grid) return;
    const g  = this._modelBounds.grid;
    const cx = g.origin.x + g.worldWidth * 0.5;
    const cz = g.origin.z + g.worldDepth * 0.5;
    const extra = g.cellSize * 2;
    const geom = new THREE.PlaneGeometry(g.worldWidth + extra, g.worldDepth + extra);
    const mat  = new THREE.MeshBasicMaterial({
      color: 0x4499dd, transparent: true, opacity: 0.22,
      side: THREE.DoubleSide, depthWrite: false,
    });
    this._gwtMesh = new THREE.Mesh(geom, mat);
    this._gwtMesh.rotation.x = -Math.PI / 2;
    this._gwtMesh.position.set(cx, elevM, cz);
    this._scene.add(this._gwtMesh);
  }

  // ── Preset camera views ───────────────────────────────────────────────────
  setCameraView(preset) {
    const b = this._modelBounds;
    if (!b) return;
    const { cx, cy, cz, size } = b;
    let pos, up;
    switch (preset) {
      case 'plan':
        pos = [cx, cy + size * 2.2, cz + 0.001];
        up  = [0, 0, -1];
        break;
      case 'ns':
        pos = [cx, cy + size * 0.25, cz - size * 1.6];
        up  = [0, 1, 0];
        break;
      case 'ew':
        pos = [cx + size * 1.6, cy + size * 0.25, cz];
        up  = [0, 1, 0];
        break;
      default: // '3d'
        pos = [cx + size * 0.8, cy + size * 0.5, cz + size * 0.8];
        up  = [0, 1, 0];
        break;
    }
    this._camera.position.set(...pos);
    this._camera.up.set(...up);
    this._camera.lookAt(cx, cy, cz);
    this._controls.target.set(cx, cy, cz);
    this._controls.update();
  }

  // ── Background ────────────────────────────────────────────────────────────
  setBackground(dark) {
    this._scene.background = new THREE.Color(dark ? 0x141820 : 0xf0f2f5);
    document.getElementById('viewer-panel').style.background = dark ? '#141820' : '';
    document.getElementById('three-canvas').style.background = dark ? '#141820' : '';
  }

  // ── Getters ───────────────────────────────────────────────────────────────
  get threeScene()  { return this._scene; }
  get threeCamera() { return this._camera; }
  get voxelGroup()  { return this._builder.group; }
  get slicer()      { return this._slicer; }
}

function _escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
