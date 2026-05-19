import * as THREE from 'three';

// ── Leapfrog-style interactive section slicer ─────────────────────────────────
// Controls:
//   D              — toggle top-down (plan) view / restore
//   Ctrl + scroll  — slide slab forward / backward
//   "Draw Line"    — click-drag on the canvas to define orientation
export class SlicerTool {
  constructor(scene, camera, controls, renderer, onPlanesChange) {
    this._scene    = scene;
    this._camera   = camera;
    this._controls = controls;
    this._renderer = renderer;
    this._canvas   = renderer.domElement;
    this._overlay  = document.getElementById('slicer-overlay');
    // Per-material clipping is more reliable than renderer.clippingPlanes for
    // custom ShaderMaterials (avoids compile-time NUM_CLIPPING_PLANES=0 trap)
    this._onPlanesChange = onPlanesChange ?? (p => { renderer.clippingPlanes = p; });

    this._mode      = 'slab';       // 'slab' | 'removeFront' | 'removeBack'
    this._thickness = 20;
    this._normal    = new THREE.Vector3(0, 0, 1);
    this._centerD   = 0;
    this._baseCenterD = 0;
    this._hasSlice  = false;
    this._grid      = null;
    this._modelCenter = null;

    this._pA = new THREE.Plane();
    this._pB = new THREE.Plane();
    this._visual = null;

    this._topDownMode = false;
    this._savedCamPos = null;
    this._savedCamUp  = null;
    this._savedTarget = null;

    // Keep overlay in sync with canvas size
    if (this._overlay) {
      new ResizeObserver(() => {
        const r = this._canvas.getBoundingClientRect();
        if (r.width) { this._overlay.width = r.width; this._overlay.height = r.height; }
      }).observe(this._canvas);
    }

    this._initKeyboard();
    this._initPanel();
  }

  // ── Called after a model is built ────────────────────────────────────────
  setModelBounds(grid) {
    this._grid = grid;
    this._modelCenter = new THREE.Vector3(
      grid.origin.x + grid.worldWidth  * 0.5,
      grid.origin.y + grid.worldHeight * 0.5,
      grid.origin.z + grid.worldDepth  * 0.5
    );
    this._thickness = Math.max(5, Math.round(grid.worldWidth * 0.08));

    const maxSlide = Math.round(Math.max(grid.worldWidth, grid.worldDepth) * 0.6);
    const maxThick = Math.round(Math.max(grid.worldWidth, grid.worldDepth));

    const tSlider = document.getElementById('slicer-thickness');
    if (tSlider) {
      tSlider.max = maxThick;
      tSlider.value = this._thickness;
      const lbl = document.getElementById('slicer-thickness-val');
      if (lbl) lbl.textContent = `${this._thickness} m`;
    }
    const pSlider = document.getElementById('slicer-position');
    if (pSlider) { pSlider.min = -maxSlide; pSlider.max = maxSlide; pSlider.value = 0; }
  }

  // ── Keyboard: D = top-down toggle, Ctrl+scroll = slide ───────────────────
  _initKeyboard() {
    window.addEventListener('keydown', e => {
      const tag = e.target?.tagName?.toLowerCase();
      if (tag === 'input' || tag === 'textarea') return;
      if (e.key === 'd' || e.key === 'D') this._toggleTopDown();
    });

    this._canvas.addEventListener('wheel', e => {
      if (!e.ctrlKey || !this._hasSlice) return;
      e.preventDefault();
      this._centerD += e.deltaY > 0 ? 2 : -2;
      this._updatePlanes();
      this._updateVisual();
      const ps = document.getElementById('slicer-position');
      if (ps) ps.value = Math.round(this._centerD - this._baseCenterD);
    }, { passive: false });
  }

  _toggleTopDown() {
    if (!this._modelCenter) return;
    if (this._topDownMode) {
      this._camera.position.copy(this._savedCamPos);
      this._camera.up.copy(this._savedCamUp);
      this._controls.target.copy(this._savedTarget);
      this._controls.update();
      this._topDownMode = false;
    } else {
      this._savedCamPos = this._camera.position.clone();
      this._savedCamUp  = this._camera.up.clone();
      this._savedTarget = this._controls.target.clone();
      const t    = this._controls.target;
      const dist = this._grid
        ? Math.max(this._grid.worldWidth, this._grid.worldDepth) * 1.6 : 400;
      this._camera.position.set(t.x, t.y + dist, t.z);
      this._camera.up.set(0, 0, -1);
      this._controls.target.copy(t);
      this._controls.update();
      this._topDownMode = true;
    }
  }

  // ── Draw-line mode ────────────────────────────────────────────────────────
  startDraw() {
    if (!this._overlay) return;
    const rect = this._canvas.getBoundingClientRect();
    this._overlay.width  = rect.width;
    this._overlay.height = rect.height;
    this._overlay.hidden = false;
    this._overlay.style.pointerEvents = 'all';
    this._overlay.style.cursor = 'crosshair';
    this._controls.enabled = false;

    let startScr = null, startWorld = null;
    const ctx = this._overlay.getContext('2d');
    ctx.clearRect(0, 0, this._overlay.width, this._overlay.height);

    // hint text
    ctx.font = '13px Inter, sans-serif';
    ctx.fillStyle = 'rgba(26,111,165,0.8)';
    ctx.fillText('Click and drag to draw a section line', 12, 24);

    const onDown = e => {
      startScr   = [e.offsetX, e.offsetY];
      startWorld = this._toGround(e.clientX, e.clientY);
    };
    const onMove = e => {
      if (!startScr) return;
      ctx.clearRect(0, 0, this._overlay.width, this._overlay.height);
      ctx.setLineDash([10, 5]);
      ctx.strokeStyle = '#1a6fa5';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(startScr[0], startScr[1]);
      ctx.lineTo(e.offsetX, e.offsetY);
      ctx.stroke();
      ctx.setLineDash([]);
      [[startScr[0], startScr[1]], [e.offsetX, e.offsetY]].forEach(([px, py]) => {
        ctx.beginPath(); ctx.arc(px, py, 6, 0, Math.PI * 2);
        ctx.fillStyle = '#1a6fa5'; ctx.fill();
      });
    };
    const onUp = e => {
      cleanup();
      if (!startWorld) return;
      const end = this._toGround(e.clientX, e.clientY);
      if (end && startWorld.distanceTo(end) > 1) this._finalize(startWorld, end);
    };
    const cleanup = () => {
      this._overlay.removeEventListener('mousedown', onDown);
      this._overlay.removeEventListener('mousemove', onMove);
      this._overlay.removeEventListener('mouseup',   onUp);
      this._overlay.style.pointerEvents = 'none';
      this._overlay.style.cursor = 'default';
      this._controls.enabled = true;
      setTimeout(() => {
        ctx.clearRect(0, 0, this._overlay.width, this._overlay.height);
        this._overlay.hidden = true;
      }, 700);
    };
    this._overlay.addEventListener('mousedown', onDown);
    this._overlay.addEventListener('mousemove', onMove);
    this._overlay.addEventListener('mouseup',   onUp);
  }

  _toGround(cx, cy) {
    const rect = this._canvas.getBoundingClientRect();
    const ndc  = new THREE.Vector2(
      ((cx - rect.left) / rect.width)  * 2 - 1,
      -((cy - rect.top) / rect.height) * 2 + 1
    );
    const rc = new THREE.Raycaster();
    rc.setFromCamera(ndc, this._camera);
    const groundY = this._modelCenter?.y ?? 0;
    const gp = new THREE.Plane(new THREE.Vector3(0, 1, 0), -groundY);
    const pt = new THREE.Vector3();
    return rc.ray.intersectPlane(gp, pt) ? pt.clone() : null;
  }

  _finalize(wA, wB) {
    const dir = new THREE.Vector3().subVectors(wB, wA);
    dir.y = 0;
    if (dir.length() < 1) return;
    dir.normalize();
    // Normal is perpendicular to drawn line (90° rotation in horizontal plane)
    this._normal.set(-dir.z, 0, dir.x);
    const mid = new THREE.Vector3().addVectors(wA, wB).multiplyScalar(0.5);
    this._centerD     = this._normal.dot(mid);
    this._baseCenterD = this._centerD;
    this._hasSlice    = true;
    this._updatePlanes();
    this._updateVisual();
    const hint = document.getElementById('slicer-hint');
    if (hint) hint.textContent = 'Ctrl+scroll to move  ·  D = plan view  ·  Draw again to redefine';
    const ps = document.getElementById('slicer-position');
    if (ps) ps.value = 0;
  }

  // Programmatically set section by two world XZ points.
  // x1/z1 and x2/z2 are in geological coordinates (X=Easting, Z=Northing).
  setByWorldPoints(x1, z1, x2, z2) {
    const wA = new THREE.Vector3(x1, 0, z1);
    const wB = new THREE.Vector3(x2, 0, z2);
    this._finalize(wA, wB);
  }

  // ── Clipping planes ───────────────────────────────────────────────────────
  // Three.js clips fragments where: plane.normal · point + plane.constant < 0
  _updatePlanes() {
    if (!this._hasSlice) return;
    const n = this._normal, d = this._centerD, t = this._thickness / 2;
    if (this._mode === 'slab') {
      // Keep: d - t  ≤  n·x  ≤  d + t
      this._pA.set(n.clone().negate(), d + t); // clips n·x > d+t
      this._pB.set(n.clone(), t - d);           // clips n·x < d-t
      this._onPlanesChange([this._pA, this._pB]);
    } else if (this._mode === 'removeFront') {
      // Keep: n·x ≤ d  (remove viewer-side)
      this._pA.set(n.clone().negate(), d);
      this._onPlanesChange([this._pA]);
    } else {
      // Keep: n·x ≥ d  (remove back)
      this._pA.set(n.clone(), -d);
      this._onPlanesChange([this._pA]);
    }
  }

  // ── 3D slab wireframe visual ──────────────────────────────────────────────
  _updateVisual() {
    if (this._visual) {
      this._scene.remove(this._visual);
      this._visual.traverse(o => { o.geometry?.dispose(); o.material?.dispose(); });
      this._visual = null;
    }
    if (!this._hasSlice || !this._grid) return;

    const g   = this._grid;
    const len = Math.max(g.worldWidth, g.worldDepth) * 1.4;
    const ht  = g.worldHeight * 1.1;
    const n   = this._normal;
    const mc  = this._modelCenter;
    const pos = mc.clone().addScaledVector(n, this._centerD - n.dot(mc));
    const ang = Math.atan2(n.x, n.z); // rotate box so depth(Z) aligns with normal

    const group = new THREE.Group();

    // Wireframe slab box
    const boxGeo  = new THREE.BoxGeometry(len, ht, this._thickness);
    const edgeMat = new THREE.LineBasicMaterial({ color: 0x1a6fa5, opacity: 0.75, transparent: true });
    group.add(new THREE.LineSegments(new THREE.EdgesGeometry(boxGeo), edgeMat));
    boxGeo.dispose();

    // Subtle fill for the cut face
    const faceMat = new THREE.MeshBasicMaterial({
      color: 0x4488cc, transparent: true, opacity: 0.06,
      side: THREE.DoubleSide, depthWrite: false,
    });
    group.add(new THREE.Mesh(new THREE.PlaneGeometry(len, ht), faceMat));

    group.rotation.y = ang;
    group.position.copy(pos);
    this._visual = group;
    this._scene.add(group);
  }

  // Re-emit planes to freshly-built meshes (call after model rebuild)
  reapply() {
    if (this._hasSlice) this._updatePlanes();
  }

  // ── Clear all clipping ────────────────────────────────────────────────────
  clear() {
    this._hasSlice = false;
    this._onPlanesChange([]);
    if (this._visual) {
      this._scene.remove(this._visual);
      this._visual.traverse(o => { o.geometry?.dispose(); o.material?.dispose(); });
      this._visual = null;
    }
  }

  // ── Panel wiring ──────────────────────────────────────────────────────────
  _initPanel() {
    document.getElementById('btn-open-slicer')
      ?.addEventListener('click', () => {
        document.getElementById('slicer-panel').hidden = false;
      });
    document.getElementById('slicer-close')
      ?.addEventListener('click', () => {
        document.getElementById('slicer-panel').hidden = true;
      });
    document.getElementById('slicer-draw-btn')
      ?.addEventListener('click', () => this.startDraw());
    document.getElementById('slicer-clear-btn')
      ?.addEventListener('click', () => {
        this.clear();
        const hint = document.getElementById('slicer-hint');
        if (hint) hint.textContent = 'Press D for plan view, then click Draw Line';
        const ps = document.getElementById('slicer-position');
        if (ps) ps.value = 0;
      });
    document.querySelectorAll('.slicer-mode-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.slicer-mode-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        this._mode = btn.dataset.mode;
        if (this._hasSlice) this._updatePlanes();
      });
    });
    document.getElementById('slicer-thickness')?.addEventListener('input', e => {
      this._thickness = parseInt(e.target.value);
      const lbl = document.getElementById('slicer-thickness-val');
      if (lbl) lbl.textContent = `${this._thickness} m`;
      if (this._hasSlice) { this._updatePlanes(); this._updateVisual(); }
    });
    document.getElementById('slicer-position')?.addEventListener('input', e => {
      if (!this._hasSlice) return;
      this._centerD = this._baseCenterD + parseInt(e.target.value);
      this._updatePlanes();
      this._updateVisual();
    });
  }
}
