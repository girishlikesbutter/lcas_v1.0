# LCAS Results Directory

This directory contains generated light curves, plots, and interactive 3D animations from LCAS simulations.

## Directory Structure

```
results/
├── intelsat_901_results/    # Intelsat 901 satellite outputs
│   ├── 251126/              # Results from November 26, 2025
│   │   ├── 1430_lc_comparison_100pts.png   # Light curve plot
│   │   ├── 1430_lc_comparison_100pts.csv   # Light curve data
│   │   └── 1430_animation_100pts.html      # 3D animation
│   └── 251127/              # Results from November 27, 2025
├── torus_plate_results/     # Torus-plate test model outputs
└── README.md                # This file
```

## Output Files

Each simulation run produces three files with matching timestamps:

| File Type | Extension | Description |
|-----------|-----------|-------------|
| **Light Curve Plot** | `.png` | Visual comparison of shadowed vs. unshadowed magnitudes |
| **Light Curve Data** | `.csv` | Raw data for further analysis |
| **3D Animation** | `.html` | Interactive visualization (open in web browser) |

## File Naming Convention

Files are named with the pattern: `HHMM_description_Npts.ext`

- **HHMM**: Time the simulation was run (24-hour format)
- **description**: Type of output (`lc_comparison`, `animation`)
- **Npts**: Number of time points simulated
- **ext**: File extension (`.png`, `.csv`, `.html`)

**Example**: `1430_lc_comparison_100pts.png`
- Generated at 14:30
- Comparison light curve
- 100 time points
- PNG image

## CSV File Columns

The CSV files contain the following columns:

| Column | Description |
|--------|-------------|
| `UTC_Time` | Time in UTC format |
| `ET_Seconds` | Ephemeris Time (seconds past J2000) |
| `Hours_Since_Start` | Hours elapsed since simulation start |
| `Magnitude_Shadowed` | Apparent magnitude with self-shadowing |
| `Magnitude_No_Shadow` | Apparent magnitude without self-shadowing |
| `Phase_Angle_Deg` | Sun-satellite-observer angle (degrees) |
| `Observer_Distance_1000km` | Distance to observer (thousands of km) |

## Viewing 3D Animations

The `.html` animation files are interactive Plotly visualizations:

1. Open the file in any modern web browser (Chrome, Firefox, Edge)
2. Use mouse to rotate, zoom, and pan the 3D view
3. Play/pause the animation with the controls
4. Hover over elements to see component names and values

## Organizing Results

Results are automatically organized by:
- **Satellite model**: Each model has its own subfolder
- **Date**: YYMMDD format (e.g., `251126` = Nov 26, 2025)
- **Time**: HHMM prefix on each file

This makes it easy to find and compare results from different simulation runs.
