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
# # Create Satellite Ephemeris from TLE Data
#
# This notebook creates a SPICE SPK (ephemeris) file from Two-Line Element (TLE) data.
#
# **What you'll create:** A `.bsp` file containing satellite position over time.
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

# Name for your output SPK file (without .bsp extension)
OUTPUT_NAME = "intelsat_901_test"

# Path to your TLE file (relative to project root)
TLE_FILE = "data/tle/intelsat_901/2001-024A.tle"

# NORAD catalog number from your TLE (the 5-digit number on line 1)
TLE_NORAD_ID = 26824

# SPICE ID to assign to this satellite (use negative number)
SPICE_SATELLITE_ID = -126824

# Time range for ephemeris generation
START_TIME = "2020 JAN 01"
STOP_TIME = "2020 MAR 01"

# Your name (optional, appears in kernel metadata)
PRODUCER = "Girish Narayanan"

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

from src.spice.spice_kernel_generator import check_binaries_installed, create_ephemeris_from_tle

# Create timestamped output directories
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
setup_dir = PROJECT_ROOT / "data" / "spice_utilities" / "setup_files" / timestamp
output_dir = PROJECT_ROOT / "data" / "spice_utilities" / "output_kernels" / timestamp
setup_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)

# Define file paths
setup_file = setup_dir / f"{OUTPUT_NAME}.setup"
output_spk = output_dir / f"{OUTPUT_NAME}.bsp"

print(f"Session timestamp: {timestamp}")
print(f"Setup file will be: {setup_file}")
print(f"Output SPK will be: {output_spk}")

# Check binary installation
print("\nChecking SPICE binaries...")
status = check_binaries_installed()
mkspk_ok = status.get("mkspk") is not None
if mkspk_ok:
    print("  mkspk: found")
else:
    print("  mkspk: NOT FOUND - please install to data/spice_utilities/")

# %% [markdown]
# ---
# ## Generate Setup File and Create SPK

# %%
# Verify TLE file exists
tle_path = PROJECT_ROOT / TLE_FILE
if not tle_path.exists():
    raise FileNotFoundError(f"TLE file not found: {tle_path}")
print(f"TLE file found: {tle_path}")

# Create the setup file
setup_content = f'''\\begindata
INPUT_DATA_TYPE   = 'TL_ELEMENTS'
OUTPUT_SPK_TYPE   = 10
TLE_INPUT_OBJ_ID  = {TLE_NORAD_ID}
TLE_SPK_OBJ_ID    = {SPICE_SATELLITE_ID}
CENTER_ID         = 399
REF_FRAME_NAME    = 'J2000'
LEAPSECONDS_FILE  = 'data/spice_kernels/generic/lsk/naif0012.tls.pc'
INPUT_DATA_FILE   = '{TLE_FILE}'
OUTPUT_SPK_FILE   = '{output_spk.relative_to(PROJECT_ROOT)}'
PCK_FILE          = 'data/spice_kernels/generic/pck/geophysical.ker'
START_TIME        = '{START_TIME}'
STOP_TIME         = '{STOP_TIME}'
PRODUCER_ID       = '{PRODUCER}'
APPEND_TO_OUTPUT  = 'NO'
\\begintext
'''

with open(setup_file, 'w') as f:
    f.write(setup_content)
print(f"Created setup file: {setup_file}")

# Generate the SPK
print("\nRunning mkspk...")
result_spk = create_ephemeris_from_tle(setup_file)
print(f"\nSUCCESS - Created ephemeris file:")
print(f"  {result_spk}")

# %% [markdown]
# ---
# ## Verify the Result (Optional)
#
# Test that the new kernel works by querying satellite position.

# %%
from src.spice.spice_handler import SpiceHandler

spice = SpiceHandler()
spice.load_kernel(str(PROJECT_ROOT / "data/spice_kernels/generic/lsk/naif0012.tls.pc"))
spice.load_kernel(str(result_spk))

# Test at a time within the ephemeris range
test_time = "2024-01-01T12:00:00"
et = spice.utc_to_et(test_time)
pos, lt = spice.get_body_position(str(SPICE_SATELLITE_ID), et, "J2000", "399")

print(f"Satellite position at {test_time} (J2000, relative to Earth):")
print(f"  X: {pos[0]:,.1f} km")
print(f"  Y: {pos[1]:,.1f} km")
print(f"  Z: {pos[2]:,.1f} km")
print(f"  Distance from Earth center: {(pos[0]**2 + pos[1]**2 + pos[2]**2)**0.5:,.1f} km")

spice.unload_all_kernels()
print("\nKernel verified successfully.")
