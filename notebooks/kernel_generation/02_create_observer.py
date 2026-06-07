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
# # Create Observer/Ground Station Kernel
#
# This notebook creates a SPICE SPK file for a ground-based observer
# (telescope, tracking station, etc.) using the `pinpoint` utility.
#
# **What you'll create:** A `.bsp` file containing the observer's position on Earth.
#
# Configure your ground station parameters below and run all cells.

# %% [markdown]
# ---
# ## User Inputs
#
# Edit these values for your ground station:

# %%
# =============================================================================
# USER INPUTS - Edit these values
# =============================================================================

# Name for your output SPK file (without .bsp extension)
OUTPUT_NAME = "eifel_tower"

# Station identifier (no spaces, use underscores)
STATION_NAME = "EIFLTWR"

# SPICE ID for this station (use 399XXX for Earth-based stations)
STATION_ID = 399998

# Geodetic coordinates
LATITUDE = 48.8584    # Degrees North (negative for South)
LONGITUDE = 2.2945   # Degrees East (negative for West)
ALTITUDE_KM = 0.365       # Kilometers above reference ellipsoid

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

from src.spice.spice_kernel_generator import check_binaries_installed, create_observer_kernel

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
pinpoint_ok = status.get("pinpoint") is not None
if pinpoint_ok:
    print("  pinpoint: found")
else:
    print("  pinpoint: NOT FOUND - please install to data/spice_utilities/")

# %% [markdown]
# ---
# ## Generate Setup File and Create SPK

# %%
# Create the setup file (pinpoint definition format)
setup_content = f'''\\begindata
SITES             = '{STATION_NAME}'
{STATION_NAME}_CENTER = 399
{STATION_NAME}_FRAME  = 'ITRF93'
{STATION_NAME}_IDCODE = {STATION_ID}
{STATION_NAME}_LATLON = ({LATITUDE}, {LONGITUDE}, {ALTITUDE_KM})
\\begintext
'''

with open(setup_file, 'w') as f:
    f.write(setup_content)
print(f"Created setup file: {setup_file}")
print(f"\nStation definition:")
print(f"  Name: {STATION_NAME}")
print(f"  ID: {STATION_ID}")
print(f"  Location: {LATITUDE}°N, {LONGITUDE}°E, {ALTITUDE_KM} km")

# Generate the SPK
print("\nRunning pinpoint...")
result_spk = create_observer_kernel(setup_file, output_spk)
print(f"\nSUCCESS - Created observer kernel:")
print(f"  {result_spk}")

# %% [markdown]
# ---
# ## Verify the Result (Optional)
#
# Test that the new kernel works by querying station position.

# %%
from src.spice.spice_handler import SpiceHandler

spice = SpiceHandler()
spice.load_kernel(str(PROJECT_ROOT / "data/spice_kernels/generic/lsk/naif0012.tls.pc"))
spice.load_kernel(str(PROJECT_ROOT / "data/spice_kernels/generic/pck/pck00011.tpc"))
spice.load_kernel(str(PROJECT_ROOT / "data/spice_kernels/generic/pck/earth_latest_high_prec.bpc"))
spice.load_kernel(str(result_spk))

# Test position
test_time = "2024-01-01T12:00:00"
et = spice.utc_to_et(test_time)
pos, lt = spice.get_body_position(str(STATION_ID), et, "J2000", "399")

print(f"Observer position at {test_time} (J2000, relative to Earth center):")
print(f"  X: {pos[0]:,.1f} km")
print(f"  Y: {pos[1]:,.1f} km")
print(f"  Z: {pos[2]:,.1f} km")
distance = (pos[0]**2 + pos[1]**2 + pos[2]**2)**0.5
print(f"  Distance from Earth center: {distance:,.1f} km")

# Calculate altitude using WGS84 ellipsoid (more accurate than mean radius)
import numpy as np
WGS84_A = 6378.137  # Equatorial radius (km)
WGS84_B = 6356.752  # Polar radius (km)
lat_rad = np.radians(LATITUDE)

# Ellipsoid radius at this latitude
ellipsoid_radius = np.sqrt(
    ((WGS84_A**2 * np.cos(lat_rad))**2 + (WGS84_B**2 * np.sin(lat_rad))**2) /
    ((WGS84_A * np.cos(lat_rad))**2 + (WGS84_B * np.sin(lat_rad))**2)
)
computed_altitude = distance - ellipsoid_radius
print(f"  Ellipsoid radius at {LATITUDE}°N: {ellipsoid_radius:,.3f} km")
print(f"  Computed altitude: {computed_altitude:,.3f} km (input was {ALTITUDE_KM} km)")

spice.unload_all_kernels()
print("\nKernel verified successfully.")
