# src/computation - Core Computation Modules

This module contains the core computational components for light curve generation: shadow ray tracing, BRDF calculations, observation geometry, and light curve generation.

## Module Overview

| File | Purpose |
|------|---------|
| `shadow_engine.py` | Ray tracing for shadow computation |
| `brdf.py` | Ashikhmin-Shirley BRDF calculations and material management |
| `observation_geometry.py` | Sun/observer vector computation via SPICE |
| `lightcurve_generator.py` | Light curve generation with BRDF integration |

## shadow_engine.py

### High-Level Functions

```python
from src.computation.shadow_engine import (
    compute_shadows,
    create_no_shadow_lit_status,
    get_shadow_engine
)

# Ray tracing with pre-computed matrices
lit_status_dict = compute_shadows(
    satellite=satellite,
    k1_vectors=k1_vectors,                    # (N, 3) sun vectors
    explicit_component_matrices=component_matrices  # Dict[name, (N, 4, 4)]
)
# Returns: Dict[component_name, np.ndarray(N, num_facets)] boolean lit status

# No-shadow mode (all facets lit)
lit_status_no_shadow = create_no_shadow_lit_status(satellite, num_epochs)

# Access engine directly
engine = get_shadow_engine()
print(engine.get_performance_summary())
```

### ShadowEngine Class
- Creates shadow meshes with articulation applied
- Performs ray-mesh intersection via trimesh

### Shadow Computation Flow
1. Separate static and articulated components
2. Create static mesh once (reused across epochs)
3. Create articulated mesh per epoch
4. Combine meshes and perform ray tracing
5. Return per-facet boolean lit status

## brdf.py

### BRDF Calculator

```python
from src.computation.brdf import BRDFCalculator, BRDFManager

# Initialize and configure
brdf_manager = BRDFManager(config)
brdf_calc = BRDFCalculator()
brdf_calc.update_satellite_brdf_with_manager(satellite, brdf_manager)

# Calculate flux for a facet
flux_numerator = brdf_calc.calculate_facet_flux_numerator(
    facet=facet,
    sun_direction=k1_vector,
    observer_direction=k2_vector,
    lit_fraction=1.0
)
```

### Ashikhmin-Shirley BRDF
```python
# rho = rho_diffuse + rho_specular
rho = brdf_calc.calculate_brdf(
    n_dot_k1,   # normal · sun direction
    n_dot_k2,   # normal · observer direction
    h_dot_k1,   # halfway · sun direction
    n_dot_h,    # normal · halfway vector
    material    # BRDFMaterialProperties
)
```

### Magnitude Conversion
```python
from src.computation.brdf import convert_flux_to_magnitude

magnitude = convert_flux_to_magnitude(
    flux_value=total_flux,
    observer_distance_km=distance,
    mode='normal'  # or 'surrogate' for log flux
)
```

## observation_geometry.py

```python
from src.computation.observation_geometry import compute_observation_geometry

geometry_data = compute_observation_geometry(
    epochs=epochs,                    # ET time array
    satellite_id=satellite_id,        # SPICE ID
    spice_handler=spice_handler,
    config=config,
    attitude_keyframes=None           # Optional for custom SLERP
)

# Returns dict with:
# - 'k1_vectors': (N, 3) sun direction in body frame
# - 'k2_vectors': (N, 3) observer direction in body frame
# - 'observer_distances': (N,) distances in km
# - 'sun_positions': (N, 3) in J2000
# - 'sat_positions': (N, 3) in J2000
# - 'obs_positions': (N, 3) in J2000
# - 'sat_att_matrices': (N, 3, 3) attitude matrices
```

### Custom Attitude Support
```python
attitude_keyframes = {
    'times': ['2020-02-05T10:00:00', '2020-02-05T16:00:00'],
    'time_format': 'utc',
    'attitudes': [quat1, quat2],
    'format': 'quaternion'
}

geometry_data = compute_observation_geometry(
    ...,
    attitude_keyframes=attitude_keyframes  # Triggers SLERP interpolation
)
```

## lightcurve_generator.py

```python
from src.computation.lightcurve_generator import generate_lightcurves

magnitudes, flux, mag_no_shadow, flux_no_shadow, distances, animation_data = \
    generate_lightcurves(
        facet_lit_status_dict=lit_status_dict,
        k1_vectors_array=k1_vectors,
        k2_vectors_array=k2_vectors,
        observer_distances=observer_distances,
        satellite=satellite,
        epochs=epochs,
        pre_computed_matrices=component_matrices,
        generate_no_shadow=True,    # Also compute unshadowed curve
        animate=True                # Collect animation frame data
    )
```

### Output Arrays
- `magnitudes`: (N,) apparent magnitudes (shadowed)
- `flux`: (N,) total flux values
- `mag_no_shadow`: (N,) magnitudes without shadows (if requested)
- `animation_data`: List of frame dicts for 3D visualization

## Pipeline Integration

```python
# 1. Compute geometry
geometry_data = compute_observation_geometry(epochs, sat_id, spice_handler, config)
k1_vectors = geometry_data['k1_vectors']
k2_vectors = geometry_data['k2_vectors']

# 2. Compute articulation
component_matrices = compute_rotation_matrices_from_angles(angles, satellite)

# 3. Compute shadows
lit_status = compute_shadows(satellite, k1_vectors,
                             explicit_component_matrices=component_matrices)

# 4. Setup BRDF
brdf_manager = BRDFManager(config)
brdf_calc = BRDFCalculator()
brdf_calc.update_satellite_brdf_with_manager(satellite, brdf_manager)

# 5. Generate light curves
magnitudes, *_ = generate_lightcurves(
    lit_status, k1_vectors, k2_vectors, distances,
    satellite, epochs, pre_computed_matrices=component_matrices
)
```

## Cross-References

- **Articulation**: Pre-computed matrices from `src/articulation/`
- **SPICE**: Position/attitude from `src/spice/spice_handler.py`
- **Visualization**: Animation data used by `src/visualization/plotly_animation_generator.py`
