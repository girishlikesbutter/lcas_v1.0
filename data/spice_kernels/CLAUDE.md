# data/spice_kernels - SPICE Kernel Files

This folder contains NASA SPICE kernel files for orbital mechanics computations.

## Folder Structure

```
spice_kernels/
├── generic/                     # Standard SPICE kernels (shared)
│   ├── lsk/                     # Leap second kernels
│   │   └── naif0012.tls         # Latest leap second data
│   ├── pck/                     # Planetary constants
│   │   └── pck00011.tpc         # Planet shapes, sizes, orientation
│   └── spk/                     # Solar system ephemeris
│       └── de440.bsp            # JPL planetary ephemeris
├── missions/                    # Mission-specific kernels
│   ├── dst-is901/               # Intelsat 901 @ Deep Space Telescope
│   │   ├── INTELSAT_901-metakernel.tm
│   │   └── kernels/
│   │       ├── DST_OBS.bsp      # Observer (ground station) position
│   │       ├── IS901.bsp        # Satellite orbit ephemeris
│   │       ├── IS901-clock.tsc  # Spacecraft clock correlation
│   │       ├── IS901_BUS-frame.tf     # Reference frame definitions
│   │       └── IS901_BUS-orientation.bc  # Attitude/orientation data
│   └── mmt-galaxy4r/            # Galaxy 4R (placeholder)
└── CLAUDE.md                    # This file
```

## Kernel Types

| Extension | Type | Description |
|-----------|------|-------------|
| `.tls` | LSK | Leap Second Kernel - UTC/TDB conversion |
| `.tpc` | PCK | Planetary Constants Kernel - body shapes |
| `.bsp` | SPK | Spacecraft/Planetary Kernel - ephemeris |
| `.tsc` | SCLK | Spacecraft Clock Kernel |
| `.tf` | FK | Frame Kernel - reference frame definitions |
| `.bc` | CK | C-Kernel - attitude/orientation data |
| `.tm` | MK | Metakernel - kernel loading list |

## Metakernel Format

```
KPL/MK

\begindata

   PATH_VALUES = ( 'data/spice_kernels' )
   PATH_SYMBOLS = ( 'KERNELS' )

   KERNELS_TO_LOAD = (
      '$KERNELS/generic/lsk/naif0012.tls',
      '$KERNELS/generic/pck/pck00011.tpc',
      '$KERNELS/generic/spk/de440.bsp',
      '$KERNELS/missions/dst-is901/kernels/DST_OBS.bsp',
      '$KERNELS/missions/dst-is901/kernels/IS901.bsp',
      '$KERNELS/missions/dst-is901/kernels/IS901_BUS-frame.tf',
      '$KERNELS/missions/dst-is901/kernels/IS901_BUS-orientation.bc'
   )

\begintext
```

## Loading Kernels

```python
from src.spice.spice_handler import SpiceHandler

spice_handler = SpiceHandler()
metakernel_path = PROJECT_ROOT / "data/spice_kernels/missions/dst-is901/INTELSAT_901-metakernel.tm"
spice_handler.load_metakernel_programmatically(str(metakernel_path))
```

## Adding a New Mission

1. Create folder: `missions/<mission_name>/`
2. Add mission-specific kernel files
3. Create metakernel `.tm` file listing all kernels
4. Reference in satellite config's `spice_config.metakernel_path`

## Cross-References

- **SPICE handler**: `src/spice/spice_handler.py`
- **Config reference**: `spice_config` section in model YAML
- **General data patterns**: See `/data/CLAUDE.md`
