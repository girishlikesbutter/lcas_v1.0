# Prompt: Functional Block Diagram for the LCAS Light Curve Generation Pipeline

Create a functional block diagram for a satellite light curve simulation pipeline called **LCAS** (Light Curve Analysis Suite). The pipeline takes a 3D satellite model, orbital/attitude data, and material properties as inputs, and produces a synthetic light curve (apparent brightness vs. time) as output. The diagram should show every major processing block, the data flowing between them, and the final outputs.

---

## Pipeline Overview

The pipeline answers: **"How bright does a satellite appear to a ground-based observer over time?"** It does this by combining orbital mechanics, 3D geometry, ray-traced shadows, and physically-based reflectance (BRDF) into a per-epoch flux calculation.

---

## INPUTS (Starting Materials)

These are the raw inputs that enter the pipeline from disk or user specification. Group them visually at the top or left of the diagram.

### A. Satellite Configuration (YAML file)
- **Satellite name & ID** (e.g., "Intelsat 901", SPICE ID -125544)
- **Component list** — names of physical parts (Bus, SP_North, SP_South, AD_East, AD_West)
- **STL file paths** — one `.stl` mesh file per component
- **Component positions & orientations** — [x, y, z] position and [w, x, y, z] quaternion in the body frame
- **BRDF material mappings** — per-component reflectance parameters: r_d (diffuse), r_s (specular), n_phong (shininess)
- **Articulation capabilities** — per-component rotation axis, rotation center, and angular limits (defines what CAN move, not how)
- **SPICE metakernel path** — pointer to the orbital/attitude kernel collection
- **Simulation defaults** — start time (UTC), end time (UTC), output directory

### B. SPICE Kernels (Binary files on disk)
A collection of NASA SPICE kernel files providing:
- **Ephemeris kernels (SPK)** — orbital positions of Sun, Earth, satellite over time
- **Attitude kernels (CK)** — satellite orientation (quaternions) over time
- **Frame kernels (FK)** — definition of satellite body-fixed reference frame
- **Leap second & planetary constants kernels** — time system and physical constants

### C. User-Specified Parameters (set at runtime)
- **Number of time points** (e.g., 100 or 500 epochs)
- **Observer/ground station SPICE ID** (e.g., 399999)
- **Articulation behaviors** — user assigns control laws to articulable components:
  - Sun-tracking behavior (solar panels track the sun)
  - Fixed angle (antenna dishes held at constant angle)
  - Interpolated angle profiles (time-varying offsets via keyframes)
- **Custom attitudes** (optional) — user-defined quaternion keyframes as an alternative to SPICE attitude kernels

---

## FUNCTIONAL BLOCKS (Processing Stages)

Arrange these as sequential blocks flowing top-to-bottom or left-to-right. Data flows are described between each block.

---

### Block 1: Configuration Loading

**Label:** `Load Configuration`

**Function:** Reads the YAML configuration file and resolves all relative paths to absolute paths.

| | |
|---|---|
| **Inputs** | YAML config file path, project root directory |
| **Outputs** | Parsed config object containing: component definitions, BRDF mappings, articulation capabilities, SPICE paths, simulation time window |
| **Module** | `RSO_ConfigManager.load_config()` |

---

### Block 2: STL Model Loading

**Label:** `Load 3D Satellite Model`

**Function:** Reads STL mesh files for each component. Each STL file contains a triangulated surface mesh. Extracts vertices, face normals, and face areas for every triangular facet. Assembles all components into a single Satellite object.

| | |
|---|---|
| **Inputs** | Parsed config (component list, STL file paths, positions, orientations) |
| **Outputs** | `Satellite` object containing a list of `Component` objects, each containing a list of `Facet` objects. Each Facet has: 3 vertices, unit normal vector, area, and material properties slot. |
| **Key data** | Typically thousands of triangular facets (e.g., ~10,000 total for Intelsat 901) |
| **Module** | `STLLoader.create_satellite_from_stl_config()` |

**Data produced per facet:**
- `vertices`: 3 points in 3D body-frame coordinates
- `normal`: outward-facing unit normal vector (3,)
- `area`: triangle area in m^2

---

### Block 3: BRDF Parameter Assignment

**Label:** `Assign Material Properties`

**Function:** Reads BRDF material mappings from config and assigns reflectance parameters to every facet on every component. Uses name-matching (exact then substring) to map config entries to components.

| | |
|---|---|
| **Inputs** | Satellite object, BRDF mappings from config |
| **Outputs** | Satellite object with every facet now carrying BRDF parameters: r_d (diffuse reflectivity, 0-1), r_s (specular reflectivity, 0-1), n_phong (Phong exponent / shininess) |
| **Module** | `BRDFManager` + `BRDFCalculator.update_satellite_brdf_with_manager()` |

---

### Block 4: Articulation Behavior Assignment

**Label:** `Assign Articulation Behaviors`

**Function:** User programmatically assigns control behaviors to articulable components. The config only defines physical capabilities (axis, limits); this block assigns the control logic.

| | |
|---|---|
| **Inputs** | Articulation capabilities from config, user-specified behaviors |
| **Outputs** | `ArticulationEngine` with registered behaviors per component (e.g., SP_North → SunTracking, AD_East → FixedAngle) |
| **Module** | `ArticulationEngine.register_component_behavior()` |

**Behavior types:**
- **SunTrackingBehavior** — computes angle to align a reference normal toward the sun direction; clamped to joint limits
- **Fixed angle** — constant angle for the entire simulation
- **Interpolated profile** — angle varies over time via keyframes with linear/step/constant transitions

---

### Block 5: SPICE Initialization

**Label:** `Initialize Orbital Mechanics (SPICE)`

**Function:** Loads NASA SPICE kernels into memory. These provide the ability to query positions and orientations of celestial bodies and spacecraft at any time.

| | |
|---|---|
| **Inputs** | Metakernel file path (points to all individual kernel files) |
| **Outputs** | Initialized `SpiceHandler` ready to answer position/attitude queries |
| **Module** | `SpiceHandler.load_metakernel_programmatically()` |

---

### Block 6: Time Grid Generation

**Label:** `Generate Epoch Array`

**Function:** Converts UTC start/end times to ephemeris time (ET) and creates a uniformly-spaced array of observation epochs.

| | |
|---|---|
| **Inputs** | Start time (UTC string), end time (UTC string), number of points, SpiceHandler |
| **Outputs** | `epochs` array (N,) in ephemeris time (seconds past J2000), `utc_times` list of UTC strings, `time_hours` array for plotting |
| **Module** | `SpiceHandler.utc_to_et()`, `np.linspace()` |

---

### Block 7: Observation Geometry Computation

**Label:** `Compute Observation Geometry`

**Function:** For each epoch, queries SPICE to determine the positions of the Sun, satellite, and ground observer. Computes the satellite's attitude (orientation). Transforms Sun and observer direction vectors from the inertial frame (J2000) into the satellite body frame. These body-frame vectors are the fundamental inputs to all downstream physics.

| | |
|---|---|
| **Inputs** | Epoch array, satellite SPICE ID, observer SPICE ID, SpiceHandler, config, (optional) custom attitude keyframes |
| **Outputs** | Geometry data dictionary containing: |
| | `k1_vectors` (N, 3) — Sun direction in body frame (normalized) |
| | `k2_vectors` (N, 3) — Observer direction in body frame (normalized) |
| | `observer_distances` (N,) — Observer-to-satellite distance in km |
| | `sat_att_matrices` (N, 3, 3) — Attitude rotation matrices (J2000 ↔ body) |
| | Sun, satellite, observer positions in J2000 |
| **Module** | `compute_observation_geometry()` |

**Internal steps:**
1. Query SPICE for Sun position, satellite position, observer position (all in J2000)
2. Compute relative vectors: sun_direction = (sun_pos - sat_pos) / |sun_pos - sat_pos|
3. Get attitude matrix R from SPICE (or SLERP-interpolate custom quaternions)
4. Transform to body frame: k1 = R^T @ sun_direction_J2000, k2 = R^T @ obs_direction_J2000
5. Normalize k1 and k2

**Attitude source (two paths, show as a decision/switch):**
- **Path A (default):** Query SPICE attitude kernels → rotation matrix per epoch
- **Path B (custom):** User-provided quaternion keyframes → SLERP interpolation → rotation matrix per epoch

---

### Block 8: Articulation Angle Calculation

**Label:** `Compute Articulation Angles`

**Function:** For each articulable component at each epoch, computes the rotation angle using the assigned behavior and the current sun direction.

| | |
|---|---|
| **Inputs** | Satellite, k1_vectors (N, 3), ArticulationEngine with behaviors, (optional) angle interpolation keyframes |
| **Outputs** | `component_angles` dictionary: component_name → angle array (N,) in degrees |
| **Module** | `calculate_angles_from_behaviors()`, `create_angle_interpolator()` |

**Composability:** Angles from behaviors can be added to interpolated offsets:
`final_angle = sun_tracking_angle + time_varying_offset`

---

### Block 9: Rotation Matrix Computation

**Label:** `Convert Angles to Rotation Matrices`

**Function:** Converts per-component angle arrays into 4x4 rotation matrices. Each matrix describes how a component is rotated about its defined rotation axis at each epoch.

| | |
|---|---|
| **Inputs** | `component_angles` dict, Satellite (for rotation axes and centers) |
| **Outputs** | `component_matrices` dictionary: component_name → (N, 4, 4) rotation matrices |
| **Module** | `compute_rotation_matrices_from_angles()` |

Each 4x4 matrix encodes: translate to rotation center → rotate by angle about axis → translate back.

---

### Block 10: Shadow Computation (Ray Tracing)

**Label:** `Ray-Traced Shadow Computation`

**Function:** Determines which facets are illuminated and which are in shadow at each epoch. Uses ray-mesh intersection: for each facet, casts a ray from the facet center toward the Sun and checks if any other part of the satellite blocks it.

| | |
|---|---|
| **Inputs** | Satellite (geometry), k1_vectors (N, 3), component_angles, component_matrices |
| **Outputs** | `lit_status_dict`: component_name → boolean array (N, num_facets). True = lit by sun, False = in shadow. |
| **Module** | `compute_shadows()` |

**Internal steps:**
1. **Separate static vs. articulated components** — static mesh is built once, articulated meshes are rebuilt per epoch
2. **For each epoch:**
   a. Apply rotation matrices to articulated component vertices
   b. Combine all components into a single scene mesh
   c. For each facet: check if normal faces the Sun (back-face culling: n·k1 > 0)
   d. For lit-facing facets: cast shadow ray from facet center toward Sun
   e. If ray hits another mesh face before reaching the Sun → facet is shadowed
3. **Batched ray casting** — all facet rays per epoch are batched into a single trimesh intersection call for performance

**This is the most expensive step** (~60s for 500 epochs with full model, vs ~0.2s without shadows).

---

### Block 11: Light Curve Generation (BRDF + Flux Summation)

**Label:** `BRDF Calculation & Flux Summation`

**Function:** The core physics engine. For each epoch, computes the reflected flux from every visible, illuminated facet using the Ashikhmin-Shirley BRDF model, then sums all contributions to get total brightness.

| | |
|---|---|
| **Inputs** | lit_status_dict, k1_vectors, k2_vectors, observer_distances, Satellite (with BRDF params), epochs, BRDFCalculator, component_matrices |
| **Outputs** | `magnitudes` (N,) — apparent magnitude at each epoch |
| | `total_flux` (N,) — total reflected flux at each epoch |
| | `animation_data` (optional) — per-facet data for 3D visualization |
| **Module** | `generate_lightcurves()` |

**Internal steps per epoch:**

1. **Transform facet normals** — apply articulation rotation matrices to get current facet normals and positions
2. **Visibility check** — a facet contributes flux only if ALL three conditions are met:
   - `n·k1 > 0` (facet faces the Sun)
   - `n·k2 > 0` (facet faces the observer)
   - `lit_status = True` (not shadowed by another component)
3. **Per-facet BRDF evaluation** — for each active facet, compute:
   - Halfway vector: `h = normalize(k1 + k2)`
   - Dot products: `n·k1`, `n·k2`, `n·h`, `h·k1`
   - Ashikhmin-Shirley BRDF value `rho(n·k1, n·k2, n·h, h·k1, r_d, r_s, n_phong)`
4. **Per-facet flux:** `flux_facet = rho * area * (n·k1) * (n·k2)`
5. **Sum all active facets:** `total_flux = sum(flux_facet for all active facets)`
6. **Convert to magnitude:** `m = m_sun + 5*log10(distance_m) - 2.5*log10(total_flux)`
   where `m_sun = -26.74` (Sun's apparent visual magnitude)

**The Ashikhmin-Shirley BRDF model (show as a sub-block or inset):**

```
rho = rho_diffuse + rho_specular

rho_diffuse = (28 * r_d) / (23 * pi) * (1 - r_s) * (1 - (1 - n·k1/2)^5) * (1 - (1 - n·k2/2)^5)

rho_specular = ((n_phong + 1) / (8*pi)) * (n·h)^n_phong / (h·k1 * max(n·k1, n·k2)) * F

F = r_s + (1 - r_s) * (1 - h·k1)^5    [Fresnel term]
```

Where:
- `r_d` = diffuse reflectivity (0-1)
- `r_s` = specular reflectivity (0-1)
- `n_phong` = Phong shininess exponent
- `n` = facet surface normal
- `k1` = sun direction, `k2` = observer direction
- `h` = halfway vector between k1 and k2

---

## OUTPUTS (Final Products)

Show these at the bottom or right of the diagram.

### Primary Output
- **Light Curve**: Apparent magnitude vs. time — the main product. A 1D signal showing how bright the satellite appears to the observer at each epoch. Plotted as magnitude (inverted y-axis: brighter = lower number) vs. time in hours.

### Secondary Outputs
- **Unshadowed Light Curve**: Same calculation but with all facets treated as lit (no ray tracing). Used for comparison to quantify shadow effects.
- **Phase Angles**: Sun-observer angle at each epoch (derived from k1·k2).
- **CSV Data File**: Time, magnitude, flux, phase angle, observer distance exported to CSV.

### Optional Outputs
- **3D Animation Data**: Per-facet flux/lit-status at each frame for interactive Plotly visualization.
- **Interactive 3D Animation (HTML)**: Plotly-based 3D visualization showing the satellite with facet-level coloring (flux intensity or lit/shadowed/back-culled status) at each time step, with Sun and observer vectors overlaid.

---

## DATA FLOW SUMMARY

Show these as labeled arrows between blocks:

```
[YAML Config] ──→ Block 1 ──→ Parsed Config
                                  │
                    ┌─────────────┼─────────────────┐
                    ▼             ▼                  ▼
              Block 2        Block 3            Block 4
           (Load STL)    (Assign BRDF)    (Assign Behaviors)
                │             │                     │
                ▼             ▼                     ▼
           Satellite ←── (BRDF applied) ──→ ArticulationEngine
                │
[SPICE Kernels] ──→ Block 5 ──→ SpiceHandler
                                     │
                    [User params] ──→ Block 6 ──→ Epoch Array
                                                      │
                                     ┌────────────────┘
                                     ▼
                                Block 7
                         (Observation Geometry)
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                 ▼
               k1_vectors      k2_vectors     observer_distances
                    │                │
                    ▼                │
               Block 8              │
         (Articulation Angles)      │
                    │                │
                    ▼                │
               Block 9              │
          (Rotation Matrices)       │
                    │                │
           ┌───────┴───────┐        │
           ▼               ▼        │
      Block 10        Block 11 ◄────┘
   (Ray-Traced      (BRDF + Flux
    Shadows)         Summation)
        │               │
        └──► lit_status ─┘
                         │
                         ▼
                   LIGHT CURVE
              (magnitude vs. time)
```

---

## VISUAL STYLE NOTES

- Use **rounded rectangles** for processing blocks
- Use **parallelograms** or distinct shapes for input data (files on disk)
- Use **arrows with data labels** showing what flows between blocks
- Color-code by domain:
  - **Blue**: Data loading / configuration
  - **Green**: Orbital mechanics / SPICE
  - **Orange**: Articulation / mechanical motion
  - **Red**: Ray tracing / shadows
  - **Purple**: Physics / BRDF / flux calculation
  - **Gold**: Final outputs
- The **shadow computation (Block 10)** and **BRDF/flux summation (Block 11)** are the computational core — make them visually prominent
- Show the **BRDF formula** as a sub-block or callout within Block 11
- Show the **attitude source switch** (SPICE vs. Custom SLERP) as a decision diamond feeding into Block 7
- Indicate the **per-epoch loop** that wraps Blocks 7-11 (all these execute N times, once per time point)
