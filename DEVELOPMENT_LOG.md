# GeoModel AI — Development Log
> Created: 2026-05-19 | Branch: `claude/ai-geological-model-poc-8xeSa` → merged to `main`
> All dates UTC. Commit SHAs abbreviated to 7 chars.

---

## PART 1: SESSION TIMELINE & DISCUSSION SUMMARY

### Thursday 15 May — Redesign & Sample Sites

**Context:** Project was a working but raw proof-of-concept. 3D viewer was hidden behind build steps. No sample data, no welcome flow.

**User requests / discussion:**
- Make the 3D viewer always-visible ("always-on")
- Add sample datasets so the tool is immediately usable without uploading data
- Fix scrolling and layout issues in the left panel

**What was built:**
- Major redesign: always-on Three.js viewer, welcome overlay with quick-start options
- Three sample sites: Riverside Development (complex E-W fault + palaeochannel lens), A99 Highway, Coastal Site
- Topography auto-loaded with each demo site
- Layout: min-height flex fix so left panel scrolls correctly

---

### Friday 16 May (morning/afternoon) — Platform Foundation

**Context:** Tool needed to be a credible geotechnical platform before AI features were worth building on top of.

**User requests / discussion:**
- "Build out the full feature set — exports, calculators, analysis tools"
- Comparison to Leapfrog and RockWorks was implicit driver for feature breadth

**What was built (50+ commits):**
- Formation surfaces, isopach maps (thickness / top / base / certainty modes)
- Fence sections along arbitrary polylines, plan view with 6 geotechnical modes
- Geotechnical calculators: settlement (1D consolidation), bearing capacity (Terzaghi/Hansen), pile capacity (α/β methods)
- Exports: VTK, STL (binary), AGS 4.x, block model CSV, GeoJSON, point cloud, report HTML
- AI: geological interpretation narrative, auto parameter inference, stratigraphic consistency scoring
- Marching cubes isosurfaces for smooth unit solids
- SVG borehole log strip viewer, CPT log viewer
- Monte Carlo uncertainty (N IDW realisations with perturbed contacts)
- Universal Kriging with polynomial trend removal, RBF (multiquadric) interpolation
- Cross-validation (LOO), variogram auto-fitting
- Bishop simplified circular slip stability (Fs), stereonet (Schmidt net + rose diagram)
- LLM geometric shape injection: palaeochannel (parabolic trough), lens/pod (ellipsoid), buried hill (Gaussian dome)
- Per-unit geometry descriptors (correlationLength, anisoRatio, anisoAzimuth)
- CPT seismic liquefaction assessment (Robertson & Wride 1998)
- `scope.md` created — living project document

---

### Friday 16 May (evening) — Neural Implicit Field

**Context:** Discussion about how to make the 3D model respond to *conceptual* geological knowledge, not just borehole observations. The gap: conventional interpolation (IDW, Kriging) treats geology as pure spatial statistics — it doesn't know that a palaeochannel is concave-up, or that a fault creates a stepped contact.

**Key insight from discussion:**
> Binary keyword encoding tells the network "the word 'channel' appeared." It does NOT make the output geometry elongated or concave-up. The geometry of an implicit field is controlled by its coordinate correlation structure — how quickly the function changes as you move in each direction.

**What was built:**
- `js/geo-implicit.js`: Neural Implicit Geological Field
  - 4-layer MLP with Fourier positional encoding (sinusoidal basis functions, 13-dim → 39-dim)
  - Softmax output → P(unit₁…unitₙ) per position
  - Training: SGD with cosine-annealing LR schedule, gradient clipping
  - LLM Oracle: finds high-entropy clusters via BFS flood-fill, patches with Claude's geological reasoning
- Initial keyword-based `GeoKeywordEncoder` (60-dim bag-of-words) as first attempt

---

### Saturday 17 May (morning) — Semantic Concept Embedding Architecture

**Context:** The keyword encoder was identified as insufficient. Binary keywords tell the network *what words appeared* but not *what shape the geology takes*. A fundamental redesign was planned.

**User direction (explicit):**
> "The key focus is on new and innovative application of AI to modelling. Especially functionality in embedding semantic knowledge and geological rules into 3D models. Do not spend any more time recreating leapfrog issues."

**The architectural breakthrough — two mechanisms:**

#### Mechanism 1: Concept-Conditioned Anisotropic Coordinate Warping
Before Fourier-encoding position, warp coordinates using an anisotropy tensor derived from the concept embedding:
```
concept_embedding(32-dim) → anisotropy_tensor A (diagonal 3×3)
warped = A @ [x, y, z]
Fourier_encode(warped) → positional features
```
Effect: For "palaeochannel E-W" → A stretches x by 4×, compresses y by 0.5×, z by 0.3×. Fourier features now encode an ellipsoidal neighbourhood. The implicit field **naturally forms E-W elongated bodies** because E-W boreholes "see" each other as closer. The output geometry directly reflects the conceptual input — not as a hint, but as a structural constraint on the function's shape.

#### Mechanism 2: FiLM Concept Conditioning
The 32-dim concept context vector → two linear projections → (γ, β) applied to each hidden layer via Feature-wise Linear Modulation:
```
h'_layer = γ(concept_ctx) ⊙ h_layer + β(concept_ctx)
```
This is the standard mechanism by which ControlNet and HyperNetworks condition generation on semantic descriptions. The network learns: "when palaeochannel axes are active, activate THIS pattern of units at the channel base."

#### The 32 Geological Geometry Axes
Claude rates each concept on these axes (−1 to +1), each describing a specific geometric property:

| Idx | Axis | Role |
|-----|------|------|
| 0 | horizontal_layering | Flat beds (+) vs structureless (−) |
| 1 | inclined_bedding | Dipping beds present (+) |
| 2 | dip_magnitude | Dip steepness |
| 3 | east_west_elongation | Body elongated E-W |
| 4 | north_south_elongation | Body elongated N-S |
| 5 | channel_morphology | Concave-up trough geometry |
| 6 | dome_anticline | Convex-up dome |
| 7 | fault_controlled | Fault surface contact |
| 8 | erosional_contact | Erosional/unconformable base |
| 9 | lateral_continuity | Laterally continuous |
| 10–13 | lateral_thinning_[EWNS] | Wedging in each direction |
| 14–17 | deepens_[EWNS] | Surface deepens in each direction |
| 18 | stepped_boundary | Piecewise/stepped contact |
| 19 | irregular_base | Karstic/dissolution base |
| 20 | nested_channels | Multi-storey channels |
| 21 | coarsening_upward | Coarsening-upward sequence |
| 22 | fining_upward | Fining-upward sequence |
| 23 | gravel_basal_lag | Coarse gravel at base |
| 24 | dissolution_features | Karst voids |
| 25 | structural_complexity | Deformed/complex |
| 26 | data_confidence | Certainty of interpretation |
| 27 | lateral_anisotropy | Horizontally elongated |
| 28 | vertical_anisotropy | Layer-parallel fabric |
| 29 | incision_depth_ratio | Deep incision vs width |
| 30 | overburden_control | Geometry from load history |
| 31 | complexity_gradient | Complexity increases in one direction |

**What was built (May 17):**
- `js/concept-store.js`: `ConceptStore` class
  - `add({ description, embedding, confidence, domain, unitAffinity })` → id
  - `computeAt(x, y, z)` → `{ vec: Float32Array(32), weights, tensor: [Ax, Ay, Az] }`
  - Spatial relevance: global (uniform) or bbox/radius (Gaussian decay, σ configurable)
  - `_embeddingToTensor(embedding)` → anisotropy tensor from axes 3,4,5,18,29
  - `cloneScaled(factor)` for ensemble analysis
  - `globalTensor()` → site-wide average anisotropy (used in plan view)
  - `serialize() / deserialize()` for project file persistence
- `encodeGeologicalConcept(description, apiKey, demoMode)` in `claude-client.js`
  - Sends 32-axis prompt to Claude, returns `Float32Array(32)` clamped ±1
  - Demo fallback: keyword→axis heuristic regex map (100+ patterns)
  - `withRationale` option: Claude explains each axis rating
- `GeoImplicitNet` updated: 3-layer FiLM (filmW_γ, filmW_β per hidden layer)
- `trainGeoImplicit` updated: `nIn = fourierEnc.outDim + 32 = 71`
- `inferGeoImplicit` updated: per-voxel warp + FiLM, stores `attributionGrid`
- Concept UI in left panel: textarea, confidence slider, domain selector (global/bbox/draw)
- Real-time axis preview bar as user types (debounced 250ms, demo encoding)
- Concept list with 32-bar sparkline per concept (green/red by sign)
- `ConceptStore` persisted in `sessionStorage` and `.geomodel` project files

---

### Saturday 17 May (afternoon/evening) — Concept Feature Expansion

**What was built:**
- **Concept temporal ordering** — assign integer rank to concepts; younger-over-older pairs inject training samples enforcing stratigraphic sequence in data-sparse zones
- **Concept conflict detection** — intra-concept contradictions (e.g. horizontal_layering + channel_morphology) + inter-concept domain overlaps with opposing axes
- **Concept coherence scoring** — after build, measures whether each unit's actual voxel geometry matches its concept's predicted geometry (orientation, elongation, contact roughness)
- **Concept scenario management** — save/restore up to 5 named concept sets; scenario comparison panel shows voxel agreement % and unit-level diffs
- **Concept inheritance** — parent-child concept blending (child inherits 40% of parent embedding); models geological specialisation ("Terrace gravel" inherits from "Fluvial gravel")
- **Concept sensitivity analysis** — which concepts influence the most voxels; % coverage per concept
- **Geological event timeline** (Leapfrog-style) — drag-reorder events (Deposition, Erosion, Faulting, Intrusion, Folding, Karst, Fill, Terrace); auto-encodes each event as concept embedding; applies stratigraphic order to model build
- **Fence section concept overlays** — concept ribbon at bottom of fence section, anisotropy arrows showing warp direction
- **Concept territory map** on plan view — 2D Voronoi-style coloring by dominant concept
- **Concept similarity map** (2D PCA manifold) — 32-dim concept embeddings projected to 2D; interactive canvas with hover tooltips
- **Concept axis correlation matrix** — 32×32 heatmap of how axes co-vary across active concepts
- **Investigation planning** — drill location suggestion driven by concept geometry + model entropy; concept-reliance factor in drill recommendation
- **P10/P50/P90 probabilistic bounds** from MC inference — contact surfaces with uncertainty ribbons in fence sections
- **Stratigraphic inversion detection and correction** — scans columns top→bottom, flags younger-over-older violations, replaces with expected older unit
- **SPT profile**, formation tops matrix, concept radar chart per concept, certainty histogram
- **3D dip symbols** on formation surfaces
- **Indicator kriging** as additional interpolation method
- **BNG/WGS84 coordinate tools**, GeoJSON export of formation tops
- **Material quantity takeoff** — volumes and masses by unit from block model

---

### Sunday 18 May (morning) — Neural Field Maturation

**What was built:**
- **3-layer FiLM network** with hidden size 80 (upgraded from 2-layer/64)
- FiLM warmup schedule — FiLM projection weights enabled gradually during training to prevent early saturation
- **MC dropout uncertainty inference** — N=8 forward passes with dropout active; per-voxel entropy
- **Smooth isosurfaces from MC probability volumes** — P50 isosurface + P10/P90 uncertainty shells
- **Boundary-emphasis training samples** — extra samples near BH-derived contacts; prevents neural field from blurring contacts
- **Inter-borehole contact surface interpolation** — contact elevations from adjacent BHs used as additional training constraints
- **Structural orientation → concept embedding** — "→ Concept" button converts measured dip/strike to axis values
- **Diagonal palaeochannel templates** — NE-SW and NW-SE orientations added to concept library
- **Neural model prediction column in BH log strip viewer** — shows AI prediction alongside actual log
- **Confusion matrix** in model validation
- **Formation contacts export** with MC P10/P50/P90 uncertainty bounds per column
- **Concept cosine similarity** pre-check before encoding — warns if description is very similar to existing concept
- **Concept-driven contact sharpening** — voxels near stepped_boundary/fault_controlled concept zones get sharper (steeper probability gradient) training targets
- **Vertical depth domain** for concepts — restrict concept to minZ/maxZ AOD with Gaussian decay; enables "channel exists only between -5m and +2m AOD"
- **Concept narrative generator** — auto-writes professional report paragraph from concept store + model statistics
- **Concept-aware certainty calibration** — deflates certainty in data-sparse zones where concepts dominate (prevents overconfidence)
- **Improved concept encoding quality** — site context injected into Claude prompt; data_confidence axis given extra weight

---

### Sunday 18 May (afternoon) — Concept Intelligence Layer

**Context:** User reiterated: focus on innovative AI modelling, not Leapfrog feature parity. This triggered a shift toward features that use the concept embedding in genuinely novel ways — analysis, interrogation, and back-projection.

**What was built:**

#### Geological Laws Compiler
User writes geological rules in plain English (one per line). Claude compiles each into a concept embedding with rule type classification (superposition / morphological / facies / contact / regional) and spatial domain. Demo fallback uses regex pattern matching.
- "Sand/gravel channels trend E-W and are 5-20m wide" → morphological rule, E-W elongation axes
- "Chalk always underlies quaternary deposits" → superposition rule, temporal ordering applied

#### Neural Axis Sensitivity Scan
For any voxel: perturb each of the 32 concept axes by Δ=0.2, measure ΔP(dominantUnit)/Δ. Returns `Float32Array(32)` of partial derivatives. Shown in traceability panel as bar chart. Reveals which concept axes the neural network has learned to associate with each location's geology.

*Theory:* This is numerical differentiation of the neural implicit field with respect to the concept embedding space — a form of gradient-based attribution that identifies which geological dimensions most influence each prediction.

#### Sequence Stratigraphic Surface Identification
Scans voxel columns top→bottom for unit transitions. Uses concept temporal ordering (if present) or geoUnit index as proxy stratigraphy. Classifies transitions as:
- **Boundary** (younger over older) — normal stratigraphic contact
- **Reversal** (older over younger) — unconformity or inversion anomaly

Reports surfaces sorted by voxel count, with elevation range and centroid.

#### Bayesian Concept Confidence Calibration
For each concept, scans all BH locations within the concept's spatial domain. At each location, checks whether the model's prediction agrees with the actual observed unit. Computes:
```
logit(posteriorConf) = logit(priorConf) + α × (posSignal − negSignal) / totalSignal
```
where α=1.2, posSignal = Σ(relevance × certainty) where correct, negSignal = where incorrect.
Concepts that improved BH predictions get higher confidence; concepts that hurt predictions get lower confidence.

*Theory:* Logit-space Bayesian update is equivalent to multiplicative update in odds space: the posterior odds = prior odds × likelihood ratio. The relevance weight ensures only BH locations within the concept's spatial domain contribute signal.

#### Concept-Driven Engineering Hazard Map
Derives 2D engineering hazard zones directly from concept embeddings — no model build required. For each cell on a 32×24 raster, `conceptStore.computeAt()` returns the concept context vector. Five hazard types scored from axis values:
- Karst/Dissolution: axes 24 (dissolution), 19 (irregular_base)
- Fault/Structure: axes 7 (fault), 25 (complex), 18 (stepped)
- Ground Instability: axes 5 (channel), 29 (incision_depth)
- Settlement Risk: axes 0 (h_layering), suppressed by 9 (continuity)
- Data Uncertainty: inverse of axis 26 (data_confidence)

*Theory:* This is a semantic knowledge map — the concept embeddings encode expert geological knowledge about where hazards occur, and the spatial relevance function distributes that knowledge across the site. No inference needed — purely from expert encoding.

#### Geological Implication Gap Detection
During concept encoding, checks for implied-but-missing axes:
```
channel_morphology > 0.7  →  expects gravel_basal_lag > 0.3
channel_morphology > 0.7  →  expects erosional_contact > 0.5
fault_controlled > 0.7    →  expects stepped_boundary > 0.5
dissolution_features > 0.6 → expects irregular_base > 0.4
... (9 rules total)
```
If the implied axis is not covered by any existing concept, an info-level warning is shown: "Geological implication: channel_morphology implies a basal gravel lag..."

---

### Sunday 18 May (evening) — Closing the Loop

**Context:** The final session focused on completing the feedback loop: data → model → knowledge extraction → concept store. Each feature takes a different slice of this loop.

**What was built:**

#### Geological Knowledge Uncertainty (`_runKnowledgeUncertainty`)
Completes the 3-component uncertainty decomposition:
1. **Data uncertainty** — model uncertainty from sparse BH coverage (IDW/kriging variance)
2. **Model uncertainty** — neural network weight uncertainty (MC dropout, N passes)
3. **Knowledge uncertainty** — concept embedding space uncertainty (this feature)

Algorithm: Run K=6 inference passes, each with `clonePerturbed(baseNoise=0.12)` — Box-Muller Gaussian noise on each concept axis, scaled by `(1 − data_confidence_axis_value)`. High-confidence concepts perturb less; uncertain concepts perturb more.

Per-voxel Shannon entropy: H = −Σ p_k log₂ p_k across K unit-assignment distributions.
Model coloured green (entropy ≈ 0, stable) → red (high entropy, concept-sensitive).

*Theory:* This samples the epistemic uncertainty in the geological interpretation space — as opposed to aleatoric uncertainty (data noise) or model uncertainty (network weights). It answers: "If I had described my geology slightly differently, how much would the model change?"

#### Predictive Borehole Log (`predictBoreholeLog`)
Extracts a vertical column from the voxel grid at any world (x, y). Builds unit runs (top → base) with certainty weighting. Samples concept context from ConceptStore at 5 depth intervals. Renders as SVG borehole log with:
- Coloured unit blocks, opacity = certainty
- Depth axis (mAOD) with auto-scaled tick marks
- 32-axis concept sparklines at sampled depths (green/red bars)
- Nearest real borehole column for comparison

*Innovation:* No commercial software predicts a synthetic borehole log from an AI-conditioned neural implicit field and shows the conceptual context at each depth.

#### Concept-Annotated Cross-Section Generator (`generateCrossSection`)
Samples the voxel grid along any azimuth section plane. Renders to canvas as coloured pixel grid (opacity = certainty). Concept annotations overlaid:
- Concave-up arc when `channel_morphology > 0.5` (yellow dashed)
- Stepped fault lines when `stepped_boundary > 0.6` (red dashed + downthrow tick)
- Section-orientation text: "E-W body cut along trend (long axis)" vs "cut across trend"
- Real BH intersection sticks labelled by ID

*Theory:* The annotation algorithm projects concept axis values onto the section plane. `sectionDotEW = |cos(azimuth_rad)|` measures how much the section cuts along vs across the E-W concept geometry. This drives the "along/across trend" annotation.

#### Unit Concept Signature Analysis (`_analyseUnitConceptSignatures`)
The most theoretically novel feature: back-projection of the neural implicit field into the concept embedding space.

Algorithm:
1. Sample every STEP-th voxel with certainty > 0.52 (high-confidence predictions only)
2. At each sample: `conceptStore.computeAt(wx, wy, wz)` → concept context vector
3. Accumulate weighted mean per unit (weight = certainty − 0.5)
4. Compute site-wide global mean
5. Signature = unit mean − global mean (signed deviation from background)
6. Top discriminating axes: |sig| > 0.02, sorted by magnitude

Output: For each unit, a 32-bar chart showing how the concept context differs at its dominant voxels vs the rest of the site. "Add" button encodes the unit's mean concept vec as a new concept with unitAffinity — enabling iterative conceptual refinement from model output.

*Theory:* This inverts the standard workflow. Instead of concept → model geometry, this computes model geometry → inferred concept context. It reveals what geological interpretation the neural field learned to associate with each unit, independently of what the user explicitly specified.

#### Concept-Driven Geomechanical Parameter Inference (`_inferGeomechanicalParameters`)
Maps 32-axis concept embeddings to geomechanical parameter ranges (SPT N, Cu, φ′, γ) using a 38-rule table. No borehole test data required.

Rule structure: `{ axis, sign, param, delta, reason }`. Example rules:
```
gravel_basal_lag > 0.35  →  N +20  (dense granular)
horizontal_layering > 0.35  →  N −10, Cu +20  (laminated clay)
fault_controlled > 0.35  →  N −12, Cu −20, φ′ −4°  (shear zone)
dissolution_features > 0.35  →  N −15, γ −2.0 kN/m³  (void risk)
overburden_control > 0.35  →  N +8, Cu +25  (preconsolidated)
```
Base ranges from material keywords in unit description. Concept affinity filter: only apply rules to units with matching `unitAffinity`.

*Theory:* This is a form of geological knowledge compilation — encoding decades of geotechnical experience as a mapping from semantic axis values to engineering parameter ranges. The concept embedding acts as a structured intermediate representation between geological description and engineering design.

#### Concept Completeness Scanner (`_scanConceptCompleteness`)
Analyses coverage of all 32 axes grouped into 12 geological themes. Coverage = max(|emb[axis]| × confidence) across all concepts per theme. Shows:
- Overall completeness score (0–100%)
- Top 5 under-specified themes
- Specific, copy-pasteable concept description suggestions for each gap

*Purpose:* Diagnostic tool — tells users what geological dimension they haven't specified and provides the exact language to fill the gap.

---

## PART 2: ARCHITECTURE REFERENCE

### Coordinate Convention (IMPORTANT — source of bugs)
```
grid.origin = { x: min_Easting, y: bot_Elevation_AOD, z: min_Northing }

World position from grid index:
  wx = origin.x + (ix + 0.5) * cellSize       [Easting]
  wy = origin.y + (iz + 0.5) * cellHeight      [Elevation, AOD]
  wz_north = origin.z + (iy + 0.5) * cellSize  [Northing]

Three.js Y-up: three_Y = elevation
Grid flat index: flat = ix + iy * nx + iz * nx * ny
```

### Neural Field Input Composition (71-dim)
```
inp(71) = [Fourier(warped_pos)(39) | concept_ctx(32)]

warped_pos = applyTensor([wx, wy, wz], anisotropy_tensor)
concept_ctx = normalise( Σ_c relevance(c, wx, wy, wz) × c.confidence × c.embedding )

Fourier encoding: sin/cos at 7 frequency bands → 13 dims × 3 coords → 39 dims
```

### FiLM Conditioning (per hidden layer)
```
h_raw = ReLU(W_layer @ prev_h)
γ = filmW_γ[layer] @ concept_ctx   (nHidden-dim)
β = filmW_β[layer] @ concept_ctx   (nHidden-dim)
h = γ ⊙ h_raw + β
```
Three hidden layers (size 80), FiLM applied after each ReLU. FiLM warmup: weight multiplier ramps 0→1 over first 30% of training epochs to prevent early saturation.

### Anisotropy Tensor Derivation
```
Ax = exp(+east_west_elongation × 1.4)    [axis 3]
Ay = exp(+north_south_elongation × 1.4)  [axis 4]
Az = exp(−incision_depth_ratio × 1.0)   [axis 29]
+ channel_morphology (axis 5) adds curvature correction to Z
+ stepped_boundary (axis 18) adds piecewise Z discontinuity
```
The tensor warps the coordinate space before Fourier encoding. E-W palaeochannel (axis 3 = +0.9) → Ax ≈ 3.5× → E-W boreholes "see" each other as much closer → network naturally interpolates E-W → elongated body geometry.

### Concept Spatial Relevance
```
domain.type = 'global' : relevance = c.confidence (uniform)
domain.type = 'bbox'   : relevance = c.confidence × Gaussian_2D(cx,cy, sigma=domain.sigma)
domain.type = 'depth'  : relevance × Gaussian_1D(wz, minZ, maxZ, sigmaZ=domain.sigmaZ)
```
Per-voxel concept context: `concept_ctx = normalise( Σ_c relevance(c,x,y,z) × c.embedding )`

### Security Constraint (MUST NOT CHANGE)
```
API key → sessionStorage ONLY (never localStorage)
API calls → api.anthropic.com ONLY (never logged, never anywhere else)
```

---

## PART 3: FEATURE INVENTORY

### Concept Store Features
| Feature | Function | Notes |
|---------|----------|-------|
| Encode concept | `encodeGeologicalConcept()` | Claude API or demo heuristic |
| Real-time preview | debounced textarea input | 250ms, demo encoding only |
| Concept library | 22+ geological presets | 5 categories |
| Concept inheritance | parent-child blending | child = 0.6×own + 0.4×parent |
| Temporal ordering | `temporalOrder` integer | injects training samples |
| Spatial domain | global / bbox / depth band | Gaussian decay |
| Conflict detection | intra + inter concept | cosine, spatial overlap |
| Implication gaps | 9 rules | channel→gravel, fault→stepped, etc. |
| Scenario management | 5 named saves | comparison panel |
| GeoJSON export | concept polygons | GIS integration |
| Axis correlation matrix | 32×32 canvas | semantic coherence |
| Manifold explorer | 2D PCA canvas | concept space visualization |
| Concept narrative | auto-generate paragraph | report-ready text |

### Neural Field Analysis Features
| Feature | Function | Notes |
|---------|----------|-------|
| Traceability panel | hover voxel → attribution | concept weights, BH weights, tensor |
| Neural sensitivity scan | ∂P(unit)/∂axis | 33 forward passes, 32 bars |
| Knowledge uncertainty | K=6 perturbed inferences | Shannon entropy, model recolouring |
| Unit concept signatures | back-projection | model → concept space |
| Concept completeness | 12 theme groups | gap suggestions |

### Concept-Driven Analysis Features
| Feature | Function | Notes |
|---------|----------|-------|
| Geological Laws Compiler | NL rules → concepts | Claude API + demo fallback |
| Drilling recommendation | concept × uncertainty scoring | 5 locations, min-spacing enforced |
| Sequence stratigraphy | unit contact classification | boundary / reversal types |
| Bayesian calibration | BH accuracy → confidence update | logit-space Bayesian update |
| Hazard map | concept axes → 2D raster | no build required |
| Contribution report | ablation study | % voxels shifted per concept |
| Ensemble uncertainty | 3× inference (off/base/amplified) | green=data, red=concept-driven |
| Geomechanical inference | axes → SPT N, Cu, φ′, γ | 38 rules, no test data needed |

### Visualization Features
| Feature | Location | Notes |
|---------|----------|-------|
| Predictive borehole log | Analysis tab | SVG, concept sparklines at depth |
| Concept-annotated cross-section | Analysis tab | Canvas, any azimuth, AI overlays |
| Concept territory map | Plan view | dominant concept per cell |
| Anisotropy arrows | Fence sections | warp direction overlaid |
| P10/P90 uncertainty ribbons | Fence sections | from MC probability volumes |
| Probability volume viewer | Build output | per-unit P10/P50/P90 isosurfaces |
| 3D concept effect map | 3D viewer | colour by concept influence |
| Per-concept 3D highlight | Concepts panel | isolate one concept's geometry |

---

## PART 4: ROADMAP & NEXT FEATURES

### Immediate Priority (high value, contained scope)

#### 1. Concept-Driven Stratigraphic Column Generation
Given concept temporal ordering + unit affinities, generate a generalized stratigraphic column showing expected vertical sequence, typical thicknesses derived from concept embeddings, and characteristic features per unit. No model build required — purely from concept data. The column would be exportable as SVG for inclusion in reports.

*Theory:* The temporal ordering already exists; the thickness can be estimated from `incision_depth_ratio` (axis 29) relative to concept domain size. The contact character can be read from `erosional_contact` (axis 8) and `stepped_boundary` (axis 18).

#### 2. AI Concept Gap Suggestion (API mode)
After running the Completeness Scanner, send the gap list + existing concepts + BH unit list to Claude: "Given these concepts and these unit types, what geological concept is most likely missing?" Returns a suggested concept description that the user can immediately encode.
- File: `claude-client.js` — new `suggestMissingConcepts(store, units, gaps, apiKey)` function
- ~50 lines

#### 3. Concept-Driven Variogram Parameters
For non-neural interpolation methods (Kriging, UK), use the concept embedding to set variogram range parameters automatically:
- `east_west_elongation > 0.6` → set anisotropy ratio 3:1, azimuth 90°
- `lateral_continuity > 0.7` → extend search range by 1.5×
- `channel_morphology > 0.5` → tighten nugget effect (erosional contacts are sharp)
This would make Kriging aware of the conceptual model geometry without rebuilding as neural-implicit.

#### 4. Named UK Geological Formation Embeddings
A curated library of ~20 BGS-named formations with pre-computed 32-axis embeddings:
- London Clay, Lambeth Group, Thames Gravel, Woolwich Formation, Chalk, Made Ground, River Terrace Deposits, etc.
- Each encoding sourced from published BGS descriptions
- Adds to the concept library panel under a "UK Formations" category
- Practically valuable for UK practitioners; demonstrates that the embedding encodes published geological knowledge

#### 5. Concept Space Gradient Field Visualization
For the plan view: for each selected axis, compute the spatial gradient ∂vec[axis]/∂x and ∂vec[axis]/∂y across the site (from `computeAt()` samples on a grid). Visualise as a 2D arrow field overlaid on the plan view. Shows WHERE each concept is most active and how it transitions spatially.
- Useful for checking whether concept domains are set up correctly
- Zero computation — just samples the ConceptStore

### Medium Priority (significant innovation)

#### 6. Geological Concept Trajectory Replay
A temporal animation showing how the model looks as concepts are added one by one (in encoding order). Each frame: re-infer with concept store containing only the first N concepts. Shows how each concept shifts the 3D geometry. Makes the embedding architecture tangible and interactive.
- Requires fast re-inference (already works: `inferGeoImplicit` without retraining ~0.5s)
- File: `app.js` — `window._replayConceptTrajectory()`
- UI: play/pause/step animation on a canvas or by cycling the 3D model colour

#### 7. Cross-Site Concept Transfer & Library
Allow saving/loading concept stores as named `.concepts.json` files (separate from `.geomodel`). A shared concept library would let practitioners reuse interpretations across projects. The "Named UK Formations" library above would be the seed.
- Files: export/import buttons already exist but concept transfer between projects needs UX work

#### 8. Real-Time Model Update on Confidence Slider
When user adjusts a concept's confidence slider, re-infer the full grid immediately (without retraining). The anisotropy tensor and FiLM conditioning update instantly because re-inference uses the frozen trained network with updated concept context.
- Currently: sliders update the store but user must manually rebuild
- Change: debounced (500ms) call to `inferGeoImplicit(trainedModel, gridMeta, geoUnits, store)` on any slider change
- Visual: 3D model recolours smoothly

#### 9. Probabilistic Contact Surface Export (GeoTIFF/CSV)
Export the P10/P50/P90 formation top elevations as raster grids (one file per unit per percentile). Compatible with ArcGIS, QGIS, and Surfer. Currently only CSV is exported (point cloud format); proper raster grids would be more useful.

#### 10. Monte Carlo Sensitivity Tornado Chart
For each concept axis, run two inference passes (axis +σ and axis −σ, holding everything else constant). Plot the resulting P90−P10 volume change as a horizontal bar chart (tornado). Shows: "If my interpretation of east_west_elongation is wrong by ±0.2, the sand volume changes by ±8,000 m³." Genuinely novel for ground modelling practice.

### Longer-Term (research-level innovation)

#### 11. Learned Geological Rule Extraction
After training, extract interpretable geological rules from the neural implicit field by:
1. Generating systematic grid of (z_rel, concept_axes) inputs
2. Running predictions on each
3. Finding the combination of axes that most strongly predicts each unit at each depth
4. Presenting as: "IF east_west_elongation > 0.6 AND z < midZ THEN P(SAND) > 0.8"
This inverts the encoding: concept → model → symbolic rules → human-readable geology.

#### 12. Structural Trend Surface Extraction
Use the trained neural field to extract structural trend surfaces (bedding planes, unconformities) as implicit isosurfaces of the gradient field `∇F`. Where the implicit field changes rapidly vertically (high ∂F/∂z) and slowly horizontally, this identifies bedding planes. Extractable as mesh surfaces.

#### 13. Multi-Site Concept Transfer Learning
Train a meta-concept network on multiple site models, then fine-tune for a new site with minimal BH data. The meta-network learns general geological patterns (channel geometry, fault styles) and transfers them as concept priors for new sites. This is the geological equivalent of few-shot learning.

#### 14. Collaborative Geological Interpretation
Multiple geologists encode different conceptual interpretations of the same site as separate concept stores. The system computes the disagreement map (voxels where stores predict different units) and the ensemble median model. Presents as: "Two interpretations disagree in the NW corner — which BH location would most resolve this?"

---

## PART 5: KEY FILES REFERENCE

| File | Role | Last major change |
|------|------|-------------------|
| `js/geo-implicit.js` | Neural implicit field: MLP, Fourier encoding, FiLM layers, train/infer | May 18 (3-layer FiLM, size 80, MC dropout) |
| `js/concept-store.js` | ConceptStore: 32-axis embeddings, spatial relevance, tensor, clonePerturbed | May 18 (clonePerturbed Box-Muller) |
| `js/claude-client.js` | All Claude API calls + demo fallbacks; Geological Laws Compiler | May 18 (Laws Compiler, drilling recommendation) |
| `js/interpolator.js` | Voxel grid builders (IDW/Kriging/GP/IK/Neural); sequence surfaces; predictive log; cross-section | May 18 (predictBoreholeLog, generateCrossSection) |
| `js/app.js` | AppState, event bus, all UI wiring; all `window._*` analysis functions | May 19 (knowledge uncertainty, signatures, geomechanics, completeness) |
| `js/scene.js` | Three.js scene, OrbitControls, voxel hover → attribution events | May 18 |
| `js/plan-view.js` | 2D plan slice: concept territory, entropy, drill-target modes | May 18 |
| `index.html` | All UI: tabs, panels, concept forms, analysis sections | May 19 |
| `css/styles.css` | Dark geotechnical theme, concept bars, traceability panel | May 18 |
| `assets/demo-site.json` | Pre-computed demo concepts (Palaeochannel, River Terrace) | May 18 |
| `scope.md` | Living project document (superseded by this file for AI features) | May 16 |
