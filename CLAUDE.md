# LCAS - Light Curve Analysis Suite

This repository provides a modular pipeline for generating synthetic satellite light curves using STL-based 3D models, SPICE orbital mechanics, ray tracing, and Ashikhmin-Shirley BRDF calculations.

## Project Overview

LCAS simulates the apparent brightness of satellites as observed from ground-based telescopes by:
1. Loading satellite geometry from STL mesh files
2. Computing observation geometry (sun/observer vectors) via NASA SPICE
3. Applying component articulation (solar panel tracking, antenna pointing)
4. Performing shadow ray tracing for self-occlusion
5. Calculating reflected light using physical BRDF models
6. Outputting light curves (magnitude vs time) and interactive 3D animations

## Directory Structure

```
lcas_trimmed_test/
├── src/                    # Core Python library modules
│   ├── config/             # YAML configuration loading and schemas
│   ├── io/                 # STL loading, data structures, CSV export
│   ├── spice/              # SPICE orbital mechanics wrapper
│   ├── articulation/       # Component movement and behaviors
│   ├── computation/        # Shadow engine, BRDF, light curve generation
│   ├── interpolation/      # Quaternion SLERP and angle interpolation
│   ├── utils/              # Geometry utilities
│   └── visualization/      # Plotting and Plotly animations
├── notebooks/              # Jupytext-linked workflow notebooks
├── data/                   # Models, SPICE kernels, and results
│   ├── models/             # Satellite STL files and YAML configs
│   ├── spice_kernels/      # SPICE kernel files (generic + mission)
│   └── results/            # Generated light curves and animations
└── CLAUDE.md               # This file
```

## Quick Start

### Typical Pipeline Workflow (from notebooks)

```python
# 1. Load configuration
from src.config.rso_config_manager import RSO_ConfigManager
config_manager = RSO_ConfigManager(PROJECT_ROOT)
config = config_manager.load_config("intelsat_901/intelsat_901_config.yaml")

# 2. Load satellite from STL files
from src.io.stl_loader import STLLoader
satellite = STLLoader.create_satellite_from_stl_config(config, config_manager)

# 3. Initialize SPICE
from src.spice.spice_handler import SpiceHandler
spice_handler = SpiceHandler()
spice_handler.load_metakernel_programmatically(str(metakernel_path))

# 4. Compute observation geometry
from src.computation.observation_geometry import compute_observation_geometry
geometry_data = compute_observation_geometry(epochs, satellite_id, spice_handler, config)

# 5. Set up articulation (sun tracking for solar panels)
from src.articulation import ArticulationEngine, SunTrackingBehavior
articulation_engine = ArticulationEngine()
sun_tracking = SunTrackingBehavior({...})
articulation_engine.register_component_behavior('SP_North', sun_tracking)

# 6. Compute shadows using ray tracing
from src.computation.shadow_engine import compute_shadows
lit_status_dict = compute_shadows(satellite, k1_vectors,
                                  explicit_component_matrices=component_matrices)

# 7. Generate light curves
from src.computation.lightcurve_generator import generate_lightcurves
magnitudes, flux, *_ = generate_lightcurves(...)
```

## Key Concepts

### Coordinate Frames
- **J2000**: Inertial reference frame for SPICE calculations
- **Body Frame**: Satellite-fixed frame where geometry is defined
- **k1_vectors**: Sun direction in body frame (normalized)
- **k2_vectors**: Observer direction in body frame (normalized)

### Articulation
Components can articulate (rotate) based on behaviors:
- **SunTrackingBehavior**: Rotates solar panels to track the sun
- Angles can be computed from behaviors or interpolated from keyframes
- Rotation matrices are pre-computed for efficiency

### Shadow Computation
- Uses batched ray-mesh intersection via trimesh
- Separates static and articulated components for optimized mesh creation
- Returns per-facet lit status (boolean arrays)

### BRDF Model
- Implements Ashikhmin-Shirley BRDF
- Parameters: r_d (diffuse), r_s (specular), n_phong (shininess)
- Configured per-component in YAML files

## Experiment Discipline

- **Analytical before computational.** Before running any pipeline batch, exhaust what you can learn from existing data: re-score result.json files, check if the change only affects ranking/selection (no pipeline re-run needed), run population statistics on existing checkpoints. If a 30-second re-scoring answers the question, don't run a 2-hour batch.
- **State predictions before running.** For every experiment, write: (a) the hypothesis, (b) which seeds you expect to change, (c) what outcome would confirm/refute. If you can't name a seed with an uncertain outcome, you don't need to run the experiment.
- **Run only uncertain seeds.** If re-scoring proves 8/10 seeds will be identical, run the other 2. "Test on all trajectories" means don't bias to one seed — not "mechanically run every seed when you already know the answer."
- **Kill at 2× expected time.** If a seed normally takes 12 min and hits 25 min, diagnose immediately. Don't sleep-wait. Use idle compute time for analysis.
- **Never sleep-wait.** If your only plan is `sleep && check`, you should be doing lightweight analysis instead. If you genuinely have nothing to do while the batch runs, the batch was probably unnecessary.
- **Cite the file for every number.** Quantitative claims (seed counts, ρ-values, basin widths, timing estimates, conservation residuals, anything numeric) must be followed by `(source: path/to/file[:line])` or explicitly marked `unverified`. Memory recall is not a source. If you can't cite, run the Read or Bash call to ground it before stating the value. Applies equally to `result.json`, `batch_summary.json`, git log timings, and prior-session figures.
- **Classify anomalies before alarming.** Before calling something a bug, classify it as (A) convention/sign/coordinate choice, (B) implementation bug, or (C) unknown. State the classification first, then the evidence. Do not use words like "incorrect", "wrong", "broken", or "physically wrong" for category A. The ω-sign episode and the L_J2000 framing both started as category A presented as category B.

## Output Rules

- **Plots:** When saving a plot (PNG, HTML, etc.), always print the saved file path for the user. Example: `Saved: data/results/inversion_diagnostics/micro92_twin_axis_analysis.png`

## Development Guidelines

### Adding a New Satellite Model
1. Create folder under `data/models/<satellite_name>/`
2. Add STL files for each component
3. Create `<satellite_name>_config.yaml` with component definitions, BRDF mappings, and articulation capabilities
4. Add SPICE kernels under `data/spice_kernels/missions/`

### Adding a New Articulation Behavior
1. Create new class in `src/articulation/behaviors/`
2. Inherit from `ArticulationBehavior`
3. Implement `calculate_rotation_angle()` method
4. Export in `src/articulation/behaviors/__init__.py`

### Running Notebooks
Notebooks use Jupytext for version control. The `.py` files are the source:
```bash
# Sync .py to .ipynb
jupytext --sync notebooks/*.py

# Or open .py directly in Jupyter (with jupytext installed)
jupyter notebook notebooks/lcas_stl_pipeline-is901-3.py
```

## Dependencies

Key dependencies include:
- `numpy`, `numpy-quaternion` - Numerical computing
- `trimesh` - STL mesh loading and ray tracing
- `spiceypy` - NASA SPICE toolkit wrapper
- `matplotlib`, `plotly` - Visualization
- `pyyaml` - Configuration loading

## Cross-References

- **Source modules**: See `src/CLAUDE.md` for module-specific patterns
- **Data configuration**: See `data/CLAUDE.md` for YAML config patterns
- **Workflows**: See `notebooks/CLAUDE.md` for notebook usage
