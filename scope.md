# GeoModel AI — Project Scope

> Living document. Updated as features are designed, discussed, and built.
> Last updated: 2026-05-25 (batch 28 — SPT profile bin size RangeError fix: clamp binSize to ≥0.1 to prevent Array.from({length:Infinity}) crash when user enters 0)

---

## 1. Vision

A browser-based, AI-augmented 3D ground modelling tool that approaches the capability of commercial software (Leapfrog, RockWorks, GeoModeller) while remaining:

- **Fully static** — no server, no database, deployable on GitHub Pages
- **AI-native** — Claude is embedded in the modelling pipeline, not bolted on as a chat assistant
- **Geologically intelligent** — the model encodes geological rules, unit geometry, and site-specific knowledge, not just spatial statistics
- **Professional-grade output** — reports, exports, and visualisations suitable for geotechnical practice

---

## 2. Architecture

### Deployment
- Pure static HTML/CSS/ES-module JavaScript
- GitHub Pages (also works locally via `npx serve .`)
- No build step, no bundler

### Key Dependencies
| Library | Purpose |
|---|---|
| `three@0.165.0` | 3D rendering — InstancedMesh, ShaderMaterial, OrbitControls, clipping planes |
| Anthropic API (`claude-opus-4-5`) | AI classification, interpretation, semantic modelling |

### File Structure
```
5DM/
├── index.html                    App shell, layout, all UI controls
├── css/styles.css                Dark geotechnical theme
├── DEVELOPMENT_LOG.md            Full development history, architecture notes, roadmap
└── js/
    ├── app.js                    AppState, event bus, step orchestration
    ├── api-key.js                API key modal — sessionStorage only, never logged
    ├── data-parser.js            AGS 4.x (LOCA/GEOL/ISPT/CPTG/WSTB/TRAN) + CSV
    ├── claude-client.js          All Claude API calls + demo-mode mock data
    ├── concept-store.js          ConceptStore — 32-axis geological embeddings, spatial relevance, anisotropy tensor
    ├── geo-implicit.js           Neural Implicit Geological Field — 4-layer MLP, Fourier encoding, FiLM conditioning
    ├── section-interpreter.js    Section text parser — virtual BHs + concept extraction
    ├── section-sketch.js         Interactive 2D section sketch canvas
    ├── semantic-engine.js        Description similarity, transition matrix, certainty
    ├── interpolator.js           Voxel grid builder — IDW/Kriging/GP/NN/UK + semantic + predicted BH log + cross-section
    ├── voxel-builder.js          Three.js InstancedMesh per unit, ShaderMaterial
    ├── scene.js                  Three.js scene, camera, clipping, BH sticks, GWT
    ├── marching-cubes.js         Full 256-case MC isosurface extraction
    ├── surfaces.js               Unit contact surfaces + MC mesh manager
    ├── cross-section.js          X/Y/Z orthogonal slice controls
    ├── slicer.js                 Leapfrog-style interactive slicer panel
    ├── fence-section.js          Fence diagram along arbitrary polyline
    ├── plan-view.js              2D plan slice with geotechnical zonation modes
    ├── isopach.js                Thickness / depth / settlement / certainty maps
    ├── bh-log-view.js            SVG borehole log strip viewer
    ├── cpt-log-view.js           CPT qc/fs profile viewer
    ├── strat-correlation.js      Stratigraphic correlation panel
    ├── layer-controls.js         Unit legend, visibility, certainty threshold
    ├── report.js                 HTML report — site plan, log strips, tables, risk
    ├── risk-engine.js            Geotechnical hazard assessment
    ├── exporter.js               GLTF / OBJ / JSON voxel export
    ├── settlement.js             Consolidation settlement calculator
    ├── bearing.js                Bearing capacity calculator
    ├── pile.js                   Pile capacity estimator
    ├── properties.js             Engineering properties panel
    ├── constraints.js            Natural language constraint parser
    ├── geo-map.js                Geological map import (surface outcrop pinning)
    ├── geo-shapes.js             Geometric feature primitives (palaeochannels, lenses, faults)
    ├── lab-import.js             Lab results CSV import → unit parameters
    ├── liquefaction.js           CPT seismic liquefaction (Robertson & Wride)
    ├── mohr-circle.js            Mohr circle stress visualisation
    ├── crs.js                    Coordinate reference system utilities
    ├── deviation.js              Drillhole deviation minimum-curvature correction
    ├── project-config.js         .geomodel project file export/import
    ├── session.js                Session save/restore (sessionStorage only)
    ├── stereonet.js              Surface orientation analysis — Schmidt net + rose diagram
    └── slope-stability.js        Bishop simplified circular slip stability (Fs)
```

---

## 3. Data Inputs

| Input | Format | Parser | Notes |
|---|---|---|---|
| Borehole logs | AGS 4.x | `data-parser.js` | LOCA, GEOL, ISPT, CPTG, WSTB, TRAN groups |
| Borehole logs | CSV | `data-parser.js` | Flexible column aliases |
| CPT logs | AGS / CSV | `data-parser.js` | qc, fs → SBT Ic classification |
| Lab results | CSV | `lab-import.js` | Unit_Code, Test_Type, Value → unit.params |
| Topography | CSV (X,Y,Z) | app.js | Rendered as surface, clips voxel display |
| Topography | GeoTIFF (.tif/.tiff) | app.js | Parsed via minimal built-in GeoTIFF reader; Float32/Int16/UInt16; Deflate or uncompressed; ModelTiepoint georeferencing |
| Open Ground tables | CSV (LOCA / GEOL / ISPT / WSTB) | data-parser.js | Drop multiple files together; detected by AGS column names; merged into buildBoreholes |
| Geological map | CSV (X,Y,UnitCode) | `geo-map.js` | Surface outcrop pinned as interpolation constraints |
| Drillhole deviation | AGS TRAN group | `data-parser.js` | TRAN_DPTH, TRAN_INCL, TRAN_AZMH |
| Unit descriptions | Plain text | `text-input.js` | Fed to Claude for parameter inference |
| Site history | Plain text | `text-input.js` | Fed to Claude for hazard/constraint inference |
| Orientation data | Plain text | app.js | Strike/dip measurements → circular-mean azimuth |

---

## 4. Implemented Features

### 4.1 Data Ingestion
- [x] AGS 4.x parser — full LOCA/GEOL/ISPT/CPTG/WSTB/TRAN group support
- [x] CSV borehole import with flexible column aliases
- [x] CPT log import and qc/fs profile viewer
- [x] Lab data CSV import → auto-populates unit engineering parameters
- [x] Topography CSV import → rendered as mesh, used to clip voxel display
- [x] Geological map CSV import → surface outcrop pins interpolation
- [x] Drillhole deviation — minimum-curvature trajectory correction (AGS TRAN)
- [x] Structural orientation import (strike/dip text → circular-mean azimuth)
- [x] **Open Ground Studio table CSV import** — drop LOCA.csv + GEOL.csv + ISPT.csv etc. together; auto-detected by AGS column names (LOCA_NATE, GEOL_TOP etc.); merged and passed to shared buildBoreholes(); flat combined exports also supported via enhanced parseCSV aliases
- [x] **GeoTIFF topography import** — pure-JS parser; Float32/Int16/UInt16/Int32 DEMs; uncompressed and Deflate (zlib) compressed strips; ModelTiepoint + ModelPixelScale georeferencing; auto-subsampled to ≤ 50 k points; clear error messages for LZW/tiled TIFFs with QGIS re-export instructions

### 4.2 AI / Claude Integration
- [x] Unit discovery — Claude identifies distinct geological units from all layer descriptions
- [x] Layer classification — each BH layer classified to a unit code with confidence score
- [x] Parameter inference — Claude infers γ, Cu, φ', Cc, E, N_spt from unit description
- [x] Geological interpretation — stratigraphic order, constraints, hazards from site context
- [x] Demo mode — full offline operation with heuristic mock AI
- [x] **Semantic knowledge model** — Claude analyses classified dataset, returns:
  - Unit depth profiles (typical top/base/thickness per unit)
  - Transition priors (P(unit B below unit A) as AI prior)
  - Lateral continuity ratings (high/medium/low)
  - Characteristic keywords per unit
  - Depth exclusion rules (unit X unlikely above/below Y metres)
  - Synthetic anchor boreholes for data-sparse areas
- [x] **Concept encoding** — `encodeGeologicalConcept(description)` → Claude rates concept on 32 geological geometry axes (−1 to +1); demo fallback uses keyword heuristics
- [x] **Concept gap suggestions** — Claude analyses active concept embeddings and suggests missing geological themes
- [x] **Concept refinement** — Claude reviews all concepts for internal consistency, flags conflicts
- [x] **Unit similarity analysis** — Claude compares unit descriptions pairwise, returns similarity matrix
- [x] **Auto-correlate units** — Claude reviews BH layer descriptions and flags likely mis-classifications
- [x] **Geological rules compilation** — Claude derives site-specific stratigraphic rules from BH data
- [x] **Drilling location recommendations** — Claude identifies highest-uncertainty zones and recommends next BH positions
- [x] API key — sessionStorage only, never logged, only sent to api.anthropic.com

### 4.3 Interpolation Engine
Seven methods, all operating on the same voxel grid:

| Method | Description |
|---|---|
| IDW | Inverse-distance weighting, power configurable |
| Ordinary Kriging | Spherical variogram, Lagrange multiplier unbiasedness |
| Gaussian Process | RBF kernel, variance-based uncertainty |
| Neural Network | 2-layer MLP (3→32→nUnits), cosine-annealed SGD |
| Universal Kriging | Polynomial drift removal (order 0/1/2) |
| RBF (Multiquadric) | Smooth implicit surface fitting — φ(r)=√(1+(r/ε)²) per unit, solves Gram system |
| Neural Implicit Field | F(x,y,z,SDF₁…SDFₙ)→P(unit₁…unitₙ); 4-layer MLP, Fourier encoding, **SDF geometric priors** (signed distance fields for channel/lens/hill/fault/pinch shapes), LLM Oracle. **GPU-accelerated** via TF.js WebGL: full-batch training + batched voxel inference (20–50× faster than JS). |

**Advanced uncertainty:**
- [x] **Monte Carlo boundary perturbation** — N IDW realisations with Gaussian-perturbed layer boundaries (σ=0.5m); certainty = fraction agreeing with majority vote
- [x] **Variogram auto-fitting** — grid search over (C0, C1, range) to minimise SSR against empirical indicator variogram; orange dashed fitted curve + range marker drawn on variogram canvas
- [x] **Drillhole compositing** — regularises BH logs to uniform depth intervals; dominant unit by accumulated length; certainty = source × purity
- [x] **Fault plane interpolation boundary** — boreholes on the far side of a fault plane are excluded from search; parsed from constraints text (`Fault at easting X`, `Fault at northing Y`)

**Semantic layers embedded in interpolation:**
- [x] Description similarity weighting — Jaccard similarity on geological keywords; neighbours with similar descriptions get 0.8–1.2× certainty multiplier
- [x] Unit transition matrix — Laplace-smoothed P(B below A) from observed BH sequences; loop runs top-down so each voxel conditions on unit above (Markov chain)
- [x] Depth exclusion priors — AI-specified depth ranges where a unit is unlikely reduce certainty of that assignment
- [x] Synthetic anchor boreholes — AI-generated observations in data gaps, weighted by semantic weight slider

**Spatial controls:**
- [x] Anisotropic search ellipse (global) — strike azimuth + ratio (1–10×)
- [x] Semantic weight slider (0–100%) — blends spatial vs semantic guidance
- [x] K-neighbours, IDW power, cell size H/V controls
- [x] Stratigraphic consistency penalty — penalises stratigraphically reversed units
- [x] Drillhole deviation correction — observations placed at true deviated position

### 4.4 Semantic Concept Embedding Architecture

#### ConceptStore (`js/concept-store.js`) — Semantic Analysis
The ConceptStore is retained for semantic analysis functions (cross-sections, concept coherence, back-projection, predicted borehole logs). Each concept is encoded on 32 named geological axes (−1 to +1) via `encodeGeologicalConcept()`. ConceptStore is **not** passed to the neural network training/inference path.

#### SDF Geometric Priors (`js/geo-shapes.js`) — Neural Network Input

The neural implicit field receives direct spatial geometry via **Signed Distance Fields** for each parsed geological feature. This replaces the previous FiLM concept conditioning approach.

Feature types and their SDF functions:
- **Palaeochannel** — parabolic trough `_channelSDF`: orientation-aligned, width, depth, concave-up
- **Lens / lenticle** — triaxial ellipsoid `_lensSDF`: semi-axes from width_m/depth_m/length_m
- **Buried hill / dome** — Gaussian dome `_hillSDF`: amplitude, σ from width_m
- **Fault / shear** — signed plane distance `_faultSDF`: orientation, damage zone width
- **Pinch-out / taper** — linear taper `_pinchSDF`: along-axis thickness gradient

```js
// Network input (39 + nSDFs dims):
inp = [Fourier(pos)(39) | SDF₁(pos) … SDFₙ(pos)]
```

`prepareShapesForSDF(shapes, bbox)` converts fractional centroid coordinates to world space (called inside `trainGeoImplicit` once bounds are known). `evaluateAllSDFs(preparedShapes, wx, wy, wz)` returns one value per shape in [−1, +1] (scaled by feature confidence).

**Why SDF over FiLM**: FiLM conditioning requires the network to learn how abstract axis values (e.g. east_west_elongation=+0.7) correlate with spatial unit distributions — too indirect a signal for 10–20 boreholes. SDF inputs explicitly encode WHERE the geological body is expected; the network only needs to learn HOW MUCH to trust the prior vs borehole observations.

#### Concept-Driven Analysis (`js/app.js` window functions)
- [x] **Knowledge uncertainty** — K perturbed ConceptStore copies via `clonePerturbed()`; re-infer grid each time (no retrain); Shannon entropy H = −Σ p_k log₂ p_k per voxel; 3-component uncertainty decomposition (data / model / knowledge)
- [x] **Concept back-projection** — unit concept signatures = mean concept context at dominant voxels − global mean; reveals what the neural field learned about each unit's geological character; "Add as concept" button
- [x] **Predicted borehole log** — `predictBoreholeLog(x,y)` extracts vertical column from voxel grid; SVG strip with unit blocks, depth ticks, certainty shading, concept sparklines at sampled depths, nearest real BH comparison
- [x] **Cross-section generator** — `generateCrossSection(azimuth, length, midpoint)` samples grid along arbitrary azimuth; canvas render with unit pixels, contact lines, BH sticks; concept annotations (concave-up arc for channel morphology, stepped lines for stepped_boundary)
- [x] **Geomechanical inference** — 38-rule table maps concept axes to parameter deltas (SPT N, Cu, φ', γ); no test data required; per-unit cards with min/typical/max range bars
- [x] **Concept completeness scanner** — 12 geological theme groups; coverage = max(|emb[i]| × confidence) per theme; gap suggestions; overall completeness score
- [x] **Concept coherence check** — scores how well built model geometry matches each active concept
- [x] **Concept sensitivity** — shows how many voxels each concept influences; highlights dominant vs marginal concepts
- [x] **Concept hazard map** — projects concept axes (dissolution, stepped_boundary, irregular_base) to risk zones on plan view
- [x] **Concept ensemble** — runs multiple scaled concept variants; shows spread in unit volume predictions
- [x] **Concept contribution report** — per-voxel breakdown of concept vs borehole influence across the grid

#### Concepts UI
- [x] **Concepts tab** — textarea + confidence slider + domain selector (global / bbox) + unit affinity multiselect
- [x] **Live preview** — 32-bar sparkline updates as user types (demo encoding, instant)
- [x] **[Encode Concept] button** — calls `encodeGeologicalConcept()` → Claude or demo heuristic → adds to ConceptStore
- [x] **Concept list** — each entry shows description, 32-bar sparkline (green/red by sign), confidence, [×] remove
- [x] **Export** — concept store as JSON; concept domains as GeoJSON for QGIS/ArcGIS
- [x] **Import** — load concept JSON from file
- [x] Pre-encoded demo concepts in sample datasets (Palaeochannel E-W, River Terrace deposits, etc.)

### 4.5 3D Visualisation
- [x] InstancedMesh per geological unit — one draw call per unit
- [x] Custom ShaderMaterial — per-instance colour, alpha, certainty attributes
- [x] Contact colour blending — voxels near unit boundaries blend colours
- [x] Certainty threshold slider — hides voxels below confidence level
- [x] Transparency mode — overall and per-unit opacity
- [x] Colour-fade mode — uncertain voxels fade toward grey
- [x] X/Y/Z orthogonal clipping planes (Leapfrog-style slicer) — voxels use custom world-space `uClipPlanes[4]`/`uNumClipPlanes` uniforms (bypasses compile-time `NUM_CLIPPING_PLANES` baking); standard materials (surfaces, BH sticks) use `renderer.clippingPlanes`
- [x] Vertical exaggeration control
- [x] Interactive OrbitControls (rotate/pan/zoom)
- [x] Draggable resize handles for left and right panels — updates `--left-w`/`--right-w` CSS variables; touch-drag supported; hidden on mobile
- [x] **Geology on/off toggle** — hides/shows voxel InstancedMesh group without clearing the model; button text + opacity feedback
- [x] **Fence section pan/zoom** — wheel zoom centred on cursor; alt+drag or middle-drag pans; double-click resets to 1:1; CSS transform on canvas element
- [x] **Formation surface lines on fence section** — coloured lines per unit tracing the topmost voxel of each unit column-by-column; labelled with unit code; toggled by Surfaces checkbox
- [x] Camera presets (plan, section, isometric)
- [x] Marching cubes isosurfaces — smooth 3D unit solids from binary scalar fields
- [x] 3D borehole sticks with label sprites and SPT N bar charts — shown immediately on data load, before model build
- [x] Groundwater table surface — IDW-interpolated from BH water strike depths
- [x] Topography mesh overlay

### 4.6 Parameter / Geotechnical Views
- [x] Parameter block model — colour voxels by engineering parameter:
  - Cu, φ', Cc, E, γ, N_spt (unit-level)
  - Boundary uncertainty (blend ratio — highlights unit contacts)
- [x] Plan view — horizontal slice with 6 modes:
  - Geology (unit colours)
  - Certainty
  - Cu (undrained shear strength)
  - SPT N (blow count)
  - Settlement risk (Cc)
  - Bearing capacity risk (Cu)
- [x] Isopach / thickness map — per-column unit thickness, top/base elevation, settlement, certainty
- [x] Fence section — vertical cross-section along arbitrary polyline path
- [x] SVG borehole log strip viewer
- [x] CPT qc/fs profile viewer

### 4.7 Geotechnical Calculations
- [x] Foundation design grid export — Skempton bearing capacity (5.14·Cu) + consolidation settlement at user-specified depth, CSV output
- [x] Settlement calculator — 1D consolidation (Cc method) per layer
- [x] Bearing capacity — Terzaghi/Hansen for strip/circular/rectangular footings
- [x] Pile capacity — α-method (cohesive) and β-method (granular)
- [x] Risk engine — flags liquefaction, settlement, contamination, slope, groundwater hazard zones

### 4.8 Stratigraphic Column
- [x] Proportional-thickness column rendered in right panel
- [x] Mean thickness per unit from voxel grid
- [x] Updates live as model is rebuilt or units are renamed/recoloured

### 4.9 Export
- [x] GLTF 3D model export
- [x] OBJ mesh export (contact surfaces)
- [x] JSON voxel grid export
- [x] **Binary STL export** — formation top surfaces as binary STL (80-byte header + 50-byte/triangle); compatible with Plaxis, Abaqus, FreeCAD, Blender, SolidWorks
- [x] VTK rectilinear grid export — Paraview/Visit compatible, unit_id + certainty fields
- [x] Formation contacts CSV — per-unit topmost voxel per column
- [x] Block model CSV — full voxel grid with centroids, unit codes, engineering parameters (Leapfrog/Vulcan/Datamine compatible)
- [x] Point cloud CSV
- [x] Model statistics CSV — volume, area, mean depth, mean certainty per unit
- [x] Unit properties CSV
- [x] BH logs CSV
- [x] AGS 4.x export — PROJ, TRAN, LOCA, GEOL, ISPT groups
- [x] SVG borehole log strip export
- [x] HTML geotechnical report — site plan SVG, BH log strips, unit table, risk section
- [x] Foundation design grid CSV
- [x] PNG export — plan view, isopach map

### 4.10 Advanced Analysis
- [x] **Surface orientation stereonet** — Schmidt equal-area lower hemisphere stereonet + bidirectional strike rose diagram; orientation computed from elevation gradients of unit top surfaces; Fisher mean pole plotted in red; unit-level or all-units mode
- [x] **Bishop slope stability** — simplified circular slip (Bishop 1955); grid search over slip circle centres and radii; iterative Fs solution; SVG cross-section rendering; uses dominant unit properties from model or user override; colour-coded Fs (STABLE/MARGINAL/UNSAFE)
- [x] **Intrusion / anomaly body constraints** — ellipsoidal volume overrides interpolated with unit certainty 0.98; parsed from constraints text (`Intrusion at easting X, northing Y, elevation ZmAOD, radius Rm, unit CODE`); Void bodies blank voxels
- [x] **AI Geotechnical Narrative** — Claude generates professional narrative: key_findings, geotechnical_risks, recommendations; demo fallback
- [x] **Method comparison tool** — runs all 5 interpolation methods, computes LOO accuracy for each, ranks results
- [x] **LLM Oracle** (neural implicit method) — finds high-entropy voxel clusters via BFS flood-fill, sends to Claude for geological reasoning, patches model
- [x] **Depth histogram** — SVG bar histogram showing voxel count per elevation level per unit in Unit Statistics panel

### 4.11 Project Management
- [x] Geological scenarios — save/restore up to 5 named interpretations in sessionStorage
- [x] Concept scenarios — save/restore named concept sets; multi-scenario comparison
- [x] Project config — `.geomodel` JSON file export/import (full round-trip except voxel grid)
- [x] Session save/restore — sessionStorage (not localStorage)
- [x] Cross-validation — leave-one-out BH test, logs confusion matrix and mean accuracy
- [x] Empirical variogram — computed from BH data, displayed when Kriging/UK active
- [x] 5 sample datasets — Riverside Development, A99 Highway Widening, Former Gasworks (urban), Tidal Gateway Wharf (coastal), Channel Concept Test (2-borehole palaeochannel E-W concept demo)

---

## 5. Feature Discussions

### 5.1 Semantic Knowledge Embedding (implemented)
**Discussion:** How to embed geological meaning in the interpolation rather than treating it as pure spatial statistics.

**Implemented approach:**
- Description similarity Jaccard scoring biases IDW certainty
- Markov-chain transition matrix (top-down voxel loop) conditions each voxel on the unit above
- AI semantic model provides depth profiles, exclusion rules, synthetic anchors
- Semantic weight slider (0–100%) controls blend between spatial and semantic guidance

### 5.2 Semantic Concept Embedding (implemented — see §4.4)
**Discussion:** Binary keyword vectors tell the network "the word 'channel' appeared" but don't make sand/gravel bodies elongated E-W. Virtual boreholes constrain WHERE a unit is, not what SHAPE it takes. The geometry of an implicit field is controlled by its coordinate correlation structure.

**Implemented approach:** Two mechanisms working together:
1. **Anisotropic coordinate warp** — concept axes 3/4/5/18/29 → diagonal tensor → warped Fourier features → E-W BH observations "see" each other as closer → natural E-W elongated geology
2. **FiLM conditioning** — concept_ctx → (γ, β) per hidden layer → unit identity modulation without retraining

**Key result:** The output geometry directly reflects conceptual input as a structural constraint, not a classification hint.

---

### 5.3 Knowledge Uncertainty vs Model Uncertainty (implemented — see §4.4)
**Discussion:** Standard uncertainty (MC dropout, Kriging variance) only captures model uncertainty given fixed geological interpretation. But interpretation itself is uncertain — "is this a palaeochannel or a lens?" has its own epistemic uncertainty.

**Implemented approach:** Three-component decomposition:
- **Data uncertainty** — IDW/Kriging variance from borehole spacing
- **Model uncertainty** — MC dropout entropy across forward passes
- **Knowledge uncertainty** — Shannon entropy across K perturbed ConceptStore realisations (Box-Muller noise scaled by `1 − data_confidence` axis)

This is the first tool to separate these three sources of uncertainty in a ground model.

---

### 5.4 Per-Unit Geometry Characteristics (discussed, partially implemented)
**Discussion:** How to let descriptions of typical unit geometry modify the 3D interpretation.

**Key insight:** Different geological units have fundamentally different geometries — a marine clay is a laterally continuous blanket; a fluvial channel sand is a narrow ribbon; a lens is an isolated pod. Current interpolation treats all units identically in search geometry.

**Approaches identified:**
1. **Per-unit correlation length / variogram range** — marine clay searches 500m, channel sand searches 50m
2. **Per-unit anisotropy ellipsoid** — each unit has its own strike and aspect ratio, not just a global setting
3. **Thickness prior** — if London Clay is typically 15–30m thick, certainty decays beyond that range
4. **Contact roughness → nugget effect** — "irregular erosion surface" = high nugget; "conformable contact" = near-zero nugget
5. **Claude-parsed geometry descriptors** → `unit.geometry` object: `{correlationLength, anisoRatio, anisoAzimuth, typicalThickness, contactNugget}`

### 5.5 LLM-Driven Geometric Shape Constraint Interpolation (implemented)
**Discussion:** Can natural language descriptions of geological features (palaeochannels, faults, lenses) directly generate interpolation constraints that produce geometrically correct bodies?

**Core concept:** This is the most transformative feature gap. No commercial ground modelling software currently converts natural language geometry descriptions into interpolation constraints.

**The pipeline:**
1. User writes: *"A palaeochannel filled with soft alluvial clay runs roughly east–west through the northern third of the site, ~50m wide and up to 8m deep"*
2. Claude parses to geometric primitives:
   ```json
   {
     "feature_type": "palaeochannel",
     "unit_code": "ACL",
     "orientation_deg": 90,
     "centroid_y_frac": 0.67,
     "width_m": 50,
     "max_depth_m": 8,
     "cross_section": "parabolic",
     "confidence": 0.55
   }
   ```
3. Shape function generates virtual observation points tracing the geometry (channel centreline, margins, floor)
4. BH data **conditionally activates and refines** the prior — one confirming BH snaps the channel to higher confidence and locks on to EW geometry; a contradicting BH deflates the primitive

**Shape primitive library (to be built):**

| Feature type | Shape function | Key parameters |
|---|---|---|
| Palaeochannel | Parabolic trough | orientation°, width_m, max_depth_m, centreline position (x/y fraction) |
| Lens / pod | 3D ellipsoid | centre xyz, semi-axes a/b/c, orientation |
| Fault / shear zone | Planar + damage aureole | strike°, dip°, throw_m, damage zone width |
| Dipping layer | Tilted plane | dip direction°, dip°, depth at reference |
| Pinch-out | Linear thickness taper | thin direction, zero-thickness line position |
| Buried hill | Gaussian dome | crest position, amplitude_m, half-width_m |
| Dissolution void | Irregular void | approximate centre, radius range |

**Conditional activation from BH data:**
- Each primitive starts at prior confidence (e.g. 0.5)
- Confirming BH within predicted body → weight increases (Bayesian update)
- Contradicting BH → primitive deflates or repositions
- Certainty field around shape function integrates into voxel loop alongside IDW/Kriging

**Implemented in:** `js/geo-shapes.js`, `js/claude-client.js` (`parseGeologicalFeatures`), `js/interpolator.js`, `js/app.js`

---

## 6. Planned / Backlog Features

### 6.1 Near-term
- [x] **LLM geometric shape constraints** — palaeochannels, lenses, buried hills (§5.5)
- [x] **Per-unit geometry descriptors** — corrLength, anisoRatio, anisoAzimuth per unit; table in Properties tab; IDW-weighted effective range fed to kriging/GP/RBF vote functions
- [x] **DXF export** — fence cross-section as AutoCAD R12 DXF with LAYER table, SOLID fills, BH sticks
- [x] **Seismic liquefaction (CPT)** — Robertson & Wride (1998) CSR/CRR framework; Ic soil classification; iterative qc1N normalisation; fines correction; LPI (Iwasaki); per-CPT SVG FS profile; summary table
- [x] **Semantic concept embedding** — 32-axis geological embeddings, FiLM conditioning, anisotropic warp (§4.4)
- [x] **Knowledge uncertainty** — 3-component decomposition: data / model / knowledge (§4.4)
- [x] **Concept back-projection** — neural field learns and exposes per-unit geological character (§4.4)
- [x] **Geomechanical parameter inference** — 38-rule concept-axis → parameter table (§4.4)
- [x] **Predicted borehole log** — synthetic log at any x,y from built model (§4.4)
- [x] **Cross-section generator** — arbitrary azimuth section with concept annotations (§4.4)
- [x] **Concept completeness scanner** — 12 theme groups, gap detection, suggestions (§4.4)
- [ ] **Contact surface orientation statistics** — dip / strike / azimuth statistics per unit contact; stereonet already implemented, add summary table
- [ ] **Dipping fault throw** — fault modelling with vertical throw applied to unit contacts (not just interpolation boundary)
- [ ] **Variogram from concept axes** — derive anisotropy parameters (range, sill, nugget) directly from concept embedding without fitting to sparse BH data
- [ ] **UK formation library** — pre-encoded embeddings for BGS named formations (London Clay, Chalk, River Terrace Deposits, Mercia Mudstone, etc.)
- [ ] **Concept gradient field visualisation** — 3D arrows showing the direction in which each concept's influence increases most steeply

### 6.1.1 Batch 19 — UI Redesign (2026-05-19)
**Slicer exit bug fix:** The Draw Line overlay canvas was set to `pointer-events: all` which intercepted clicks on the panel's ✕ close button. Fixed by:
- Adding `SlicerTool._cancelDraw()` class method — resets overlay pointer-events, cursor, hidden state, clears canvas, re-enables OrbitControls
- Refactored `startDraw()` local `cleanup` closure to call `this._cancelDraw()` rather than inline
- Added ESC key handler in `_initKeyboard()` that calls `_cancelDraw()` then hides the slicer panel

**UI wholesale redesign:**
Left panel reduced from 7 flat tabs (Data, Text, Section, Analysis, Concepts, Constraints, Properties) to **3 structured tabs**:

| Tab | Contents |
|---|---|
| **Data** | Sub-tabs: Uploads (AGS/CSV), Table (BH data grid), Logs (BH/CPT log viewer), Props (engineering properties) |
| **Knowledge** | Structural orientation, section input (draw + text), geological features, semantic AI, conceptual model (concepts), geological rules |
| **Build** | Grid settings, interpolation method + parameters, anisotropy, action buttons (AI classify / Build), MC settings, analysis (charts, volume, log) |

Sub-tab system: distinct CSS classes (`dsub-btn`, `dsub-content`) separate from the main `tab-btn`/`tab-content` system — avoids the global tab-switcher interfering with nested tabs.

Right panel converted from settings-only to **layer-based Table of Contents**:
- Nested TOC groups: Geological Model (Voxels / Surfaces / Uncertainty), Analysis (Plan View / Cross-Section / Isopach / Stratigraphy), Data (Boreholes / Topography / Groundwater)
- Each layer row: eye toggle, label, launch button (→ opens the corresponding analysis panel)
- Compact certainty threshold + global opacity sliders below TOC
- Unit legend beneath sliders
- Analysis panels (plan view, fence, isopach, strat correlation) remain as right-panel overlays launched from TOC

JS wiring added:
- `switchDsub(name)` — mirrors `switchTab()` for the sub-tab system
- `_initTocButtons()` — wires eye toggles (proxy-click existing buttons), launch buttons, colour mode sync, surface opacity sync, uncertainty threshold sync, `geomodel:built` event enables analysis launch buttons
- Updated `switchTab()` call-sites: `'analysis'` → `'build'`, `'concepts'` → `'knowledge'`
- Updated `.tab-btn` selectors to `.dsub-btn` for log/props sub-tabs

### 6.2 Medium-term
- [ ] **Conditional simulation** — stochastic realisations of the model respecting variogram statistics
- [ ] **Unit pinch-out modelling** — units that thin to zero at defined boundaries
- [ ] **Fold axis modelling** — periodic sinusoidal dip variation to model folded stratigraphy
- [ ] **Graben / half-graben templates** — pre-configured fault block geometry
- [ ] **WebGL2 compute shaders** — move interpolation to GPU for real-time rebuilds
- [ ] **Drillhole planning** — given current model uncertainty, suggest optimal next BH locations to maximally reduce uncertainty
- [ ] **Trajectory replay** — animate model evolution as BHs are added one by one; shows how interpretation changes with data
- [ ] **Real-time concept update** — rebuild only affected voxel columns when a concept is edited (incremental inference)

### 6.3 Longer-term
- [ ] **Multiple scenario comparison** — side-by-side or diff view of two interpretations
- [ ] **Time-lapse / construction sequence** — model changes as excavation or construction proceeds
- [ ] **3D slope stability** — full 3D sliding surface rather than 2D Bishop
- [ ] **Concept rule extraction** — identify which concept axes the neural field relied on most heavily; expose as human-readable geological rules
- [ ] **Trend surface from concepts** — generate smooth geological trend surface (dip/strike) directly from anisotropy tensors without BH fitting
- [ ] **Multi-site transfer learning** — pre-train neural field on one site, fine-tune on a second with sparse data; concept embeddings transfer the geological character

---

## 7. Key Design Decisions

### Security
- API key stored **only** in `sessionStorage` — never `localStorage`, never logged, never sent anywhere except `api.anthropic.com`
- No server-side component — all computation in browser

### Interpolation loop direction
- Voxel loop runs **top-down** (shallow → deep) to enable Markov-chain conditioning: each voxel can see the unit already assigned above it

### Synthetic boreholes
- AI-generated anchor points and shape-primitive virtual points participate in the interpolation **as if they were boreholes**, weighted by their confidence × semantic weight factor

### Unit IDs
- Units have stable integer IDs; `code` (e.g. "LC") is the human-readable key used in descriptions, constraints, and exports
- UNKN (id=0) is the fallback for unclassified voxels

### Grid coordinates
- Origin stored as `{x, y, z}` where y = minimum elevation (mAOD), z = minimum northing
- World position: `wx = origin.x + ix*cellSize + cellSize/2`; `wy = origin.y + iz*cellHeight + cellHeight/2`; `wz = origin.z + iy*cellSize + cellSize/2`
- Three.js Y-up convention: three-world Y = elevation

### Blend ratios
- Every voxel stores `blendUnitId` + `blendRatio` (0–1) — the second-ranked unit and its relative vote weight
- Used for contact colour blending in shader and for boundary uncertainty visualisation

---

## 8. Glossary

| Term | Meaning |
|---|---|
| mAOD | Metres Above Ordnance Datum (UK elevation datum) |
| BGL | Below Ground Level (depth) |
| IDW | Inverse Distance Weighting |
| UK | Universal Kriging |
| GP | Gaussian Process |
| Cu | Undrained shear strength (kPa) |
| φ' | Effective friction angle (degrees) |
| Cc | Compression index (dimensionless) |
| E | Young's stiffness modulus (MPa) |
| γ | Unit weight (kN/m³) |
| SPT N | Standard Penetration Test blow count |
| AGS | Association of Geotechnical and Geoenvironmental Specialists (data format) |
| Isopach | Map of unit thickness across the site |
| Variogram | γ(h) — measure of spatial dissimilarity as a function of separation distance h |
| Nugget | Short-range variability in a variogram (at zero lag) |
| Sill | Asymptotic variogram value at large lag |
| Marching Cubes | Algorithm to extract an isosurface mesh from a 3D scalar field |
| Thalweg | The deepest line along a channel |
| Palaeochannel | Ancient buried river channel, often infilled with soft sediments |
| RBF | Radial Basis Function — used for implicit surface interpolation |
