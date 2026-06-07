# PRD: Config-Based Inertia Calculation

## Introduction

Update the inertia calculation workflow to load component positions from satellite configuration files rather than requiring manual definition. The existing YAML config files already contain component positions and orientations in the body frame - this feature enables the inertia calculator to leverage that data directly.

Additionally, support computing inertia at specific articulation angles since movable components (solar panels, antenna dishes) affect the total inertia tensor based on their orientation.

## Goals

- Eliminate manual position entry for inertia calculations by reading from existing config files
- Support articulation-aware inertia computation (inertia varies with component orientation)
- Provide sensible defaults (0° for most components, 15° for antenna dishes)
- Maintain backward compatibility with existing manual STLComponent workflow
- Update the inertia calculation notebook to demonstrate the config-based workflow

## User Stories

### US-001: Create config-to-components loader function
**Description:** As a developer, I want a helper function that loads STL meshes and their positions from a satellite config file so that I don't have to manually specify component positions.

**Acceptance Criteria:**
- [ ] Create `load_components_from_config(config, config_manager, masses)` function in `src/computation/inertia_calculator.py`
- [ ] Function accepts `RSO_Config`, `RSO_ConfigManager`, and a `masses` dict mapping component names to masses (kg)
- [ ] Returns list of `STLComponent` objects with mesh, position from config, and mass from dict
- [ ] Raises `ValueError` if a component in masses dict is not found in config
- [ ] Typecheck passes

### US-002: Add articulation angle support to inertia calculation
**Description:** As an engineer, I want to compute the inertia tensor at specific articulation angles so that I can account for movable components in different orientations.

**Acceptance Criteria:**
- [ ] Add `articulation_angles` optional parameter to `compute_inertia_from_stl()` function
- [ ] Parameter is a dict mapping component names to angles in degrees (e.g., `{'SP_North': 45.0}`)
- [ ] If not provided, use default angles (0° for most, 15° for antenna dishes)
- [ ] Validate angles are within config-specified limits (`articulation_capabilities[name].limits`)
- [ ] Raise `ValueError` if angle is outside min/max limits
- [ ] Log warning if component has articulation capability but no angle specified (using default)
- [ ] Components without articulation capability are unaffected by angles
- [ ] Typecheck passes

### US-003: Apply articulation rotation to component geometry
**Description:** As a developer, I need the articulation rotation applied to component meshes before computing their inertia contribution.

**Acceptance Criteria:**
- [ ] Create helper function `apply_articulation_to_mesh(mesh, position, angle, rotation_center, rotation_axis)`
- [ ] Returns rotated mesh and updated position (component CoM moves when rotated about off-center axis)
- [ ] Rotation is applied about the component's `rotation_center` along its `rotation_axis`
- [ ] Uses existing rotation matrix utilities from `src/utils/geometry_utils.py`
- [ ] Typecheck passes

### US-004: Create config-aware inertia function
**Description:** As a user, I want a single function that takes a config and masses and returns the satellite inertia tensor.

**Acceptance Criteria:**
- [ ] Create `compute_inertia_from_config(config, config_manager, masses, articulation_angles=None)` function
- [ ] Internally calls `load_components_from_config()` then `compute_inertia_from_stl()`
- [ ] Passes articulation capabilities from config to enable angle-based computation
- [ ] Default articulation angles: 0° for solar panels, 15° for antenna dishes (components with "AD" prefix)
- [ ] Returns `InertiaResult` dataclass
- [ ] Typecheck passes

### US-005: Update inertia notebook to use config-based workflow
**Description:** As an engineer, I want the inertia calculation notebook to demonstrate loading component positions from config files.

**Acceptance Criteria:**
- [ ] Update `notebooks/inversion/01_inertia_calculation.py` to use config-based workflow
- [ ] Load Intelsat 901 config using `RSO_ConfigManager`
- [ ] Define masses dict in notebook (not in config file)
- [ ] Demonstrate default articulation angles and custom angles
- [ ] Show comparison of inertia at different articulation states
- [ ] Keep analytical validation section (unit cube, asymmetric box)
- [ ] Typecheck passes

### US-006: Export new functions from computation module
**Description:** As a developer, I need the new functions exported from the computation module's `__init__.py`.

**Acceptance Criteria:**
- [ ] Add `load_components_from_config` to `src/computation/__init__.py` exports
- [ ] Add `compute_inertia_from_config` to `src/computation/__init__.py` exports
- [ ] Add `apply_articulation_to_mesh` to exports (if made public)
- [ ] Typecheck passes

## Functional Requirements

- FR-1: `load_components_from_config()` must load STL files from paths resolved by `RSO_ConfigManager.get_component_path()`
- FR-2: Component positions must be read from `config.components[name].position`
- FR-3: Articulation rotation must use `config.articulation_capabilities[name].rotation_center` and `rotation_axis`
- FR-4: Default articulation angles must be 0° for all components except those with "AD" prefix (15°)
- FR-5: Articulation angles are specified in degrees, converted to radians internally
- FR-6: Components without articulation capability in config are treated as static (angle ignored)
- FR-7: The rotated component position is computed as: rotate the vector from rotation_center to original_position
- FR-8: Articulation angles must be validated against config limits; raise `ValueError` if out of range
- FR-9: Log a warning when using default angle for a component with articulation capability

## Non-Goals

- No changes to the satellite config YAML schema (masses stay in notebook)
- No time-varying inertia computation (single snapshot at given angles)
- No automatic mass estimation from mesh volume
- No changes to the existing `STLComponent` or `InertiaResult` dataclasses
- No GUI or interactive angle selection

## Technical Considerations

- Reuse `trimesh` mesh transformation capabilities for rotating meshes
- Reuse `src/utils/geometry_utils.py` for rotation matrix computation (axis-angle to matrix)
- The `RSO_ConfigManager` already stores `config_directory` for resolving STL paths
- Articulation capabilities in config already define `rotation_center` and `rotation_axis`
- Consider caching loaded meshes if the same STL file is used multiple times (e.g., `sp.stl` for both solar panels)

## Success Metrics

- Inertia calculation from config produces same results as manual specification (when using same positions)
- Notebook demonstrates complete workflow with fewer than 10 lines of setup code
- Articulation angle changes produce physically reasonable inertia variations

## Open Questions

None - all questions resolved.
