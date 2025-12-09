"""
SPICE module for orbital mechanics and kernel generation.

Provides:
- SpiceHandler: Load kernels and compute positions/orientations
- Kernel generation utilities: Create SPK, CK, and SCLK files
"""

from .spice_handler import SpiceHandler
from .spice_kernel_generator import (
    create_ephemeris_from_tle,
    create_observer_kernel,
    create_orientation_kernel,
    check_binaries_installed
)

__all__ = [
    'SpiceHandler',
    'create_ephemeris_from_tle',
    'create_observer_kernel',
    'create_orientation_kernel',
    'check_binaries_installed'
]
