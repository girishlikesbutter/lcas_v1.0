# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Create Orientation (CK) and Clock (SCLK) Kernels
#
# This notebook creates satellite attitude/orientation data using the `prediCkt` utility.
# prediCkt computes orientation based on geometric rules (e.g., "point X-axis toward Earth").
#
# **What you'll create:**
# - A `.bc` file (CK) containing satellite orientation over time
# - A `.tsc` file (SCLK) containing spacecraft clock correlation (generated automatically)
#
# Configure your satellite parameters below and run all cells.

# %% [markdown]
# ---
# ## User Inputs
#
# Edit these values for your satellite:

# %%
# =============================================================================
# USER INPUTS - Edit these values
# =============================================================================

# Name for your output files (without extension)
OUTPUT_NAME = "intelsat_901_orientation"

# Satellite SPICE ID (negative number, must match your ephemeris)
SPICE_SATELLITE_ID = -126824

# Clock ID (positive version of satellite ID)
CLOCK_ID = 126824

# Frame ID for the body frame (negative number)
FRAME_ID = -999824

# Frame name (must match your frame kernel if you have one)
FRAME_NAME = "IS901_BUS_FRAME"

# Time range for orientation data
START_TIME = "2020-JAN-01-00:00:00.000"
STOP_TIME = "2020-MAR-01-00:00:00.000"

# Orientation type: "nadir" (X toward Earth, Z toward celestial north)
#                   or "sun" (X toward Sun, Z toward celestial north)
ORIENTATION_TYPE = "nadir"

# Path to metakernel that loads your satellite ephemeris + planetary data
# This must include: leapseconds, planetary ephemeris (de440), and your satellite SPK
METAKERNEL = "data/spice_kernels/missions/dst-is901/INTELSAT_901-metakernel.tm"

# Fit tolerance in degrees (1.0 is usually fine)
TOLERANCE_DEGREES = 1.0

# %% [markdown]
# ---
# ## Setup (Run this cell to initialize)

# %%
import sys
from pathlib import Path
from datetime import datetime

# Project Root
if '__file__' in globals():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
else:
    PROJECT_ROOT = Path.cwd().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.spice.spice_kernel_generator import check_binaries_installed, create_orientation_kernel

# Create timestamped output directories
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
setup_dir = PROJECT_ROOT / "data" / "spice_utilities" / "setup_files" / timestamp
output_dir = PROJECT_ROOT / "data" / "spice_utilities" / "output_kernels" / timestamp
setup_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)

# Define file paths
setup_file = setup_dir / f"{OUTPUT_NAME}.setup"
output_ck = output_dir / f"{OUTPUT_NAME}.bc"
output_sclk = output_dir / f"{OUTPUT_NAME}.tsc"

print(f"Session timestamp: {timestamp}")
print(f"Setup file will be: {setup_file}")
print(f"Output CK will be: {output_ck}")
print(f"Output SCLK will be: {output_sclk}")

# Check binary installation
print("\nChecking SPICE binaries...")
status = check_binaries_installed()
predickt_ok = status.get("prediCkt") is not None
if predickt_ok:
    print("  prediCkt: found")
else:
    print("  prediCkt: NOT FOUND - please install to data/spice_utilities/")

# Verify metakernel exists
metakernel_path = PROJECT_ROOT / METAKERNEL
if metakernel_path.exists():
    print(f"  Metakernel: found at {metakernel_path}")
else:
    print(f"  Metakernel: NOT FOUND at {metakernel_path}")

# %% [markdown]
# ---
# ## Generate Setup File and Create CK + SCLK

# %%
# Build direction and orientation specs based on type
if ORIENTATION_TYPE == "nadir":
    # Nadir-pointing: X toward Earth, Z toward celestial north
    direction_specs = f'''DIRECTION_SPECS += ( 'SCToEarth = POSITION OF EARTH -' )
DIRECTION_SPECS += (              'FROM {SPICE_SATELLITE_ID}     -' )
DIRECTION_SPECS += (              'CORRECTION NONE'    )

DIRECTION_SPECS += ( 'CelestialNorth = FIXED J2000 XYZ 0 0 1')'''

    orientation_def = '''ORIENTATION_NAME += 'NadirPointing'
PRIMARY          += '+X = SCToEarth'
SECONDARY        += '+Z = CelestialNorth'
BASE_FRAME       += 'J2000' '''
    orientation_name = "NadirPointing"

elif ORIENTATION_TYPE == "sun":
    # Sun-pointing: X toward Sun, Z toward celestial north
    direction_specs = f'''DIRECTION_SPECS += ( 'SCToSun = POSITION OF SUN -' )
DIRECTION_SPECS += (              'FROM {SPICE_SATELLITE_ID}     -' )
DIRECTION_SPECS += (              'CORRECTION NONE'    )

DIRECTION_SPECS += ( 'CelestialNorth = FIXED J2000 XYZ 0 0 1')'''

    orientation_def = '''ORIENTATION_NAME += 'SunPointing'
PRIMARY          += '+X = SCToSun'
SECONDARY        += '+Z = CelestialNorth'
BASE_FRAME       += 'J2000' '''
    orientation_name = "SunPointing"
else:
    raise ValueError(f"Unknown ORIENTATION_TYPE: {ORIENTATION_TYPE}")

# Create the setup file
setup_content = f'''\\begintext
prediCkt specification for {OUTPUT_NAME}
Generated: {timestamp}
Orientation type: {ORIENTATION_TYPE}

\\begindata

CK-SCLK = {CLOCK_ID}
CK-SPK  = {SPICE_SATELLITE_ID}

\\begintext
Direction definitions
\\begindata

{direction_specs}

\\begintext
Orientation definition
\\begindata

{orientation_def}

\\begintext
Frame assignment and time window
\\begindata

CK-FRAMES += {FRAME_ID}

CK{FRAME_ID}ORIENTATION += 'SOLUTION TO {FRAME_NAME} = {orientation_name}'
CK{FRAME_ID}START       += @{START_TIME}
CK{FRAME_ID}STOP        += @{STOP_TIME}

\\begintext
'''

with open(setup_file, 'w') as f:
    f.write(setup_content)
print(f"Created setup file: {setup_file}")
print(f"\nOrientation configuration:")
print(f"  Type: {ORIENTATION_TYPE}")
print(f"  Satellite ID: {SPICE_SATELLITE_ID}")
print(f"  Frame: {FRAME_NAME} (ID: {FRAME_ID})")
print(f"  Time range: {START_TIME} to {STOP_TIME}")

# Generate the CK and SCLK
print("\nRunning prediCkt...")
result_ck, result_sclk = create_orientation_kernel(
    setup_file,
    metakernel_path,
    output_ck,
    output_sclk,
    TOLERANCE_DEGREES
)
print(f"\nSUCCESS - Created kernels:")
print(f"  CK (orientation): {result_ck}")
print(f"  SCLK (clock): {result_sclk}")

# %% [markdown]
# ---
# ## Verify the Result (Optional)
#
# Test that the new kernels work. Note: You need a frame kernel (.tf) that defines
# your body frame for this to work.

# %%
from src.spice.spice_handler import SpiceHandler

spice = SpiceHandler()

# Load the metakernel (which includes ephemeris, leapseconds, etc.)
spice.load_metakernel_programmatically(str(metakernel_path))

# Load our new kernels
spice.load_kernel(str(result_sclk))
spice.load_kernel(str(result_ck))

# Test orientation
test_time = "2020-02-01T12:00:00"
et = spice.utc_to_et(test_time)

try:
    rot = spice.get_target_orientation("J2000", FRAME_NAME, et)
    print(f"Rotation matrix J2000 -> {FRAME_NAME} at {test_time}:")
    print(f"  [{rot[0,0]:8.5f} {rot[0,1]:8.5f} {rot[0,2]:8.5f}]")
    print(f"  [{rot[1,0]:8.5f} {rot[1,1]:8.5f} {rot[1,2]:8.5f}]")
    print(f"  [{rot[2,0]:8.5f} {rot[2,1]:8.5f} {rot[2,2]:8.5f}]")
    print("\nOrientation kernel verified successfully.")
except Exception as e:
    print(f"Note: Could not verify orientation. This may be normal if the frame")
    print(f"kernel (.tf) isn't loaded or doesn't define {FRAME_NAME}.")
    print(f"Error: {e}")

spice.unload_all_kernels()

# %% [markdown]
# ---
# ## Notes on Frame Kernels
#
# The CK kernel stores orientation data, but you also need a **Frame Kernel** (.tf)
# that tells SPICE about your frame. If you don't have one, here's a template:
#
# ```
# KPL/FK
#
# \begindata
#
# FRAME_IS901_BUS_FRAME       = -999824
# FRAME_-999824_NAME          = 'IS901_BUS_FRAME'
# FRAME_-999824_CLASS         = 3
# FRAME_-999824_CLASS_ID      = -999824
# FRAME_-999824_CENTER        = -126824
# CK_-999824_SCLK             = 126824
# CK_-999824_SPK              = -126824
#
# \begintext
# ```
#
# The CLASS=3 indicates this is a CK-based frame (orientation from C-kernel).
