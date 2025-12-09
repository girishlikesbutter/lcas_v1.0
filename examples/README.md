# LCAS Examples

Standalone example scripts demonstrating LCAS capabilities.

## Running Examples

All examples should be run from the project root. Activate the virtual environment first:

**Windows:**
```cmd
.venv\Scripts\activate.bat
python examples/01_simple_lightcurve.py
```

**Linux/Mac:**
```bash
source .venv/bin/activate
python examples/01_simple_lightcurve.py
```

## Available Examples

| Script | Description | Time |
|--------|-------------|------|
| `01_simple_lightcurve.py` | Minimal pipeline with torus_plate model | ~30s |
| `02_sun_tracking_panels.py` | SunTrackingBehavior articulation | ~1min |
| `03_custom_angle_profiles.py` | Angle interpolation (constant, linear, step) | ~1min |
| `04_custom_attitudes.py` | Custom quaternion attitudes with SLERP | ~2min |
| `05_3d_animation.py` | Interactive 3D visualization | ~2min |

## Example Descriptions

### 01_simple_lightcurve.py
The minimal pipeline showing all essential steps:
- Load configuration
- Load STL model
- Initialize SPICE
- Compute geometry
- Compute shadows
- Generate light curve

Start here to understand the basic workflow.

### 02_sun_tracking_panels.py
Demonstrates automatic sun-tracking for solar panels:
- Uses Intelsat 901 model
- Sets up SunTrackingBehavior
- Shows angle evolution over time
- Computes shadows with articulated panels

### 03_custom_angle_profiles.py
Shows all angle interpolation methods:
- `constant`: Fixed angle
- `linear`: Smooth interpolation
- `step`: Discrete jumps
- Multi-segment profiles
- Composing angles (base + offset)

### 04_custom_attitudes.py
Custom satellite orientations using quaternions:
- Fixed attitude (identity)
- Rotating attitude (SLERP)
- Multi-keyframe sequences
- Comparison with SPICE attitudes

### 05_3d_animation.py
Interactive 3D visualization:
- `lit_status` mode: Discrete shadow coloring
- `flux` mode: Continuous brightness
- Reference frame visualization
- Sun/observer vectors

## Output

Most examples save results to:
```
data/results/<satellite>_results/<YYMMDD>/
```

Animation files are HTML and can be opened in any web browser.
