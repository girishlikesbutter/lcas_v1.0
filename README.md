# LCAS - Light Curve Analysis Suite

A modular pipeline for generating synthetic satellite light curves using STL-based 3D models (moving parts supported), NASA SPICE orbital mechanics, ray tracing, and Ashikhmin-Shirley BRDF calculations.

## What is LCAS?

LCAS simulates how bright a satellite appears when observed primarily from ground-based telescopes, though the tool supports any observer whose position in the ICRF (J2000 in SPICE) frame can be obtained for each epoch of interest. It computes reflected light by combining:

- **3D Geometry**: STL mesh models with component-level articulation (solar panels, antennas, etc.)
- **Orbital Mechanics**: NASA SPICE toolkit for precise sun/observer positioning
- **Shadow Computation**: Ray tracing for self-occlusion
- **Physical Optics**: Ashikhmin-Shirley BRDF for realistic surface reflectance

The output is a light curve showing apparent magnitude over time, plus interactive 3D animations.

## Features

- Component articulation with sun-tracking and custom behaviors
- Quaternion-based attitude interpolation (SLERP)
- Interactive 3D Plotly animations
- Cross-platform support (Linux, Windows)

## Quick Start

### 1. Clone and Install

```bash
git clone <repository-url> lcas
cd lcas
python install_dependencies.py
```

The installer will:
- Create a virtual environment
- Download SPICE utilities and planetary ephemeris data (~120 MB)
- Install all Python dependencies

### 2. Verify Installation

Activate the virtual environment first:

**Windows:**
```cmd
.venv\Scripts\activate.bat
python quickstart.py
```

**Linux/Mac:**
```bash
source .venv/bin/activate
python quickstart.py
```

### 3. Start Jupyter

```bash
# Linux/Mac
./start-jupyter.sh

# Windows
start-jupyter.bat
```

### 4. Run Your First Light Curve

Open `notebooks/lcas_stl_pipeline-is901-3.py` in Jupyter and run all cells.

## System Requirements

### Minimum
- **OS**: Windows 10/11, Linux (Ubuntu 20.04+, Fedora 38+)
- **Python**: 3.11 or 3.12
- **RAM**: 8 GB
- **Storage**: 2 GB (including SPICE kernels)

### Recommended
- **RAM**: 16 GB+

## Project Structure

```
lcas/
├── src/                    # Core Python library
│   ├── config/             # YAML configuration management
│   ├── io/                 # STL loading, data export
│   ├── spice/              # NASA SPICE orbital mechanics
│   ├── articulation/       # Component movement system
│   ├── computation/        # Shadow engine, BRDF, light curves
│   ├── interpolation/      # Quaternion/angle interpolation
│   └── visualization/      # Plotting and 3D animations
├── notebooks/              # Jupyter workflow notebooks
├── data/
│   ├── models/             # Satellite STL files and configs
│   ├── spice_kernels/      # SPICE kernel files
│   └── results/            # Generated output
└── docs/                   # Documentation
```

## Documentation

- [Installation Guide](docs/installation_guide.md) - Detailed setup instructions
- [User Guide](docs/user_guide.md) - Comprehensive usage reference

## Command Line Options

```bash
# Skip SPICE downloads (faster install, limited features)
python install_dependencies.py --skip-spice

# Custom virtual environment name
python install_dependencies.py --venv-name myenv
```

## Troubleshooting

**Virtual environment not found**: Run `python install_dependencies.py` first.

**SPICE kernel errors**: Re-run `python install_dependencies.py` to download missing kernels.

**Import errors**: Activate the virtual environment before running scripts.

## License

MIT License - see [LICENSE](LICENSE) file for details.
