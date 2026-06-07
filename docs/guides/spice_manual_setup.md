# SPICE Manual Setup Guide

This guide explains how to manually download and install SPICE utilities and kernels if the automatic installation fails.

## When Is This Needed?

The `install_dependencies.py` script automatically downloads SPICE files. Manual setup is only needed if:

- Network issues prevented automatic downloads
- You're behind a firewall blocking NAIF servers
- You want to use different SPICE versions
- You need additional utilities not included by default

## Automatic Installation Status

Check if SPICE files were downloaded:

```bash
python quickstart.py
```

Look for these checks:
- `[OK] de440.bsp` - Planetary ephemeris
- `[OK] mkspk` - SPK generator utility
- `[OK] pinpoint` - Pointing utility
- `[OK] prediCkt` - Predicted attitude utility

---

## Manual Download: SPICE Utilities

SPICE utilities are platform-specific executables.

### Download URLs

**Linux 64-bit:**
- mkspk: https://naif.jpl.nasa.gov/pub/naif/utilities/PC_Linux_64bit/mkspk
- pinpoint: https://naif.jpl.nasa.gov/pub/naif/utilities/PC_Linux_64bit/pinpoint
- prediCkt: https://naif.jpl.nasa.gov/pub/naif/utilities/PC_Linux_64bit/prediCkt

**Windows 64-bit:**
- mkspk: https://naif.jpl.nasa.gov/pub/naif/utilities/PC_Windows_64bit/mkspk.exe
- pinpoint: https://naif.jpl.nasa.gov/pub/naif/utilities/PC_Windows_64bit/pinpoint.exe
- prediCkt: https://naif.jpl.nasa.gov/pub/naif/utilities/PC_Windows_64bit/prediCkt.exe

### Installation Steps

1. Download the files for your platform
2. Place them in: `data/spice_utilities/`
3. (Linux only) Make them executable:
   ```bash
   chmod +x data/spice_utilities/mkspk
   chmod +x data/spice_utilities/pinpoint
   chmod +x data/spice_utilities/prediCkt
   ```

### Verification

```bash
# Linux
./data/spice_utilities/mkspk -h

# Windows
data\spice_utilities\mkspk.exe -h
```

Should display help text for the utility.

---

## Manual Download: de440.bsp Kernel

The de440.bsp file contains planetary ephemeris data (positions of Sun, Earth, Moon, etc.).

### Download URL

https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440.bsp

**Size:** ~114 MB

### Installation

Place the downloaded file in:
```
data/spice_kernels/generic/spk/de440.bsp
```

### Verification

```python
import spiceypy
spiceypy.furnsh("data/spice_kernels/generic/spk/de440.bsp")
print("de440.bsp loaded successfully")
```

---

## Additional Kernels

The metakernel also requires these files (usually already included):

### Leap Second Kernel (LSK)
- File: `naif0012.tls.pc`
- Location: `data/spice_kernels/generic/lsk/`
- Download: https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/

### Planetary Constants (PCK)
- File: `pck00011.tpc` or similar
- Location: `data/spice_kernels/generic/pck/`
- Download: https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/

---

## Directory Structure

After manual setup, your directory should look like:

```
data/
├── spice_kernels/
│   ├── generic/
│   │   ├── lsk/
│   │   │   └── naif0012.tls.pc
│   │   ├── pck/
│   │   │   └── pck00011.tpc
│   │   └── spk/
│   │       └── de440.bsp          ← ~114 MB
│   └── missions/
│       └── dst-is901/
│           ├── INTELSAT_901-metakernel.tm
│           └── kernels/
│               └── ...
└── spice_utilities/
    ├── mkspk                       ← Linux executable
    ├── pinpoint
    └── prediCkt
```

---

## Re-running Automatic Download

If you fix network issues, you can re-run the automatic download:

```bash
python install_dependencies.py
```

The script will skip files that already exist and only download missing ones.

---

## NAIF Resources

- **NAIF Homepage**: https://naif.jpl.nasa.gov/
- **Utilities**: https://naif.jpl.nasa.gov/naif/utilities.html
- **Generic Kernels**: https://naif.jpl.nasa.gov/pub/naif/generic_kernels/
- **SPICE Toolkit**: https://naif.jpl.nasa.gov/naif/toolkit.html
- **Tutorials**: https://naif.jpl.nasa.gov/naif/tutorials.html

---

## Troubleshooting

### "SPICE kernel file was not found"

The metakernel references a file that doesn't exist. Check:
1. de440.bsp is in `data/spice_kernels/generic/spk/`
2. Leap second kernel is in `data/spice_kernels/generic/lsk/`
3. Metakernel paths are correct (relative to project root)

### "Permission denied" (Linux)

Make binaries executable:
```bash
chmod +x data/spice_utilities/*
```

### "Cannot execute binary file"

You downloaded the wrong platform's binaries. Use:
- Linux: Files without `.exe` extension
- Windows: Files with `.exe` extension

### Download interrupted

Delete the partial file and re-download:
```bash
rm data/spice_kernels/generic/spk/de440.bsp
python install_dependencies.py
```
