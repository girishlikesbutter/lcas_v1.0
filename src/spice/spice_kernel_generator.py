"""
SPICE Kernel Generator

Simple wrapper functions for creating SPICE kernel files using NASA's
mkspk, pinpoint, and prediCkt utilities.

Usage:
    from src.spice.spice_kernel_generator import (
        create_ephemeris_from_tle,
        create_observer_kernel,
        create_orientation_kernel
    )

    # Create satellite ephemeris from TLE
    spk_path = create_ephemeris_from_tle("my_setup.setup", output_dir)

    # Create ground station kernel
    spk_path = create_observer_kernel("my_station.setup", output_dir)

    # Create orientation + clock kernels
    ck_path, sclk_path = create_orientation_kernel("my_orient.setup", metakernel, output_dir)
"""

import subprocess
import logging
from pathlib import Path
from typing import Tuple, Optional
import os

logger = logging.getLogger(__name__)


def _get_project_root() -> Path:
    """Get the project root directory."""
    # This file is at src/spice/spice_kernel_generator.py
    return Path(__file__).parent.parent.parent


def _get_binary_path(binary_name: str) -> Path:
    """
    Get the path to a SPICE binary.

    Looks in data/spice_utilities/ by default.

    Args:
        binary_name: Name of the binary (mkspk, pinpoint, prediCkt)

    Returns:
        Path to the binary

    Raises:
        FileNotFoundError: If binary is not found
    """
    project_root = _get_project_root()
    binary_path = project_root / "data" / "spice_utilities" / binary_name

    if not binary_path.exists():
        raise FileNotFoundError(
            f"SPICE binary '{binary_name}' not found at {binary_path}. "
            f"Please copy it from the NAIF toolkit to data/spice_utilities/"
        )

    return binary_path


def create_ephemeris_from_tle(
    setup_file: Path,
    output_dir: Optional[Path] = None
) -> Path:
    """
    Create a satellite ephemeris (SPK) file from TLE data using mkspk.

    The setup file should specify INPUT_DATA_FILE, OUTPUT_SPK_FILE, and other
    required parameters. See templates/kernel_generation/mkspk_template.setup
    for an example.

    Args:
        setup_file: Path to the mkspk setup file
        output_dir: Optional output directory. If provided, overrides the
                   OUTPUT_SPK_FILE path in the setup file.

    Returns:
        Path to the created SPK file

    Raises:
        FileNotFoundError: If setup file or binary not found
        RuntimeError: If mkspk fails
    """
    setup_file = Path(setup_file)
    if not setup_file.exists():
        raise FileNotFoundError(f"Setup file not found: {setup_file}")

    mkspk = _get_binary_path("mkspk")

    # Parse OUTPUT_SPK_FILE from setup to know where output goes
    output_spk = _parse_setup_value(setup_file, "OUTPUT_SPK_FILE")
    if output_spk:
        output_spk = Path(output_spk.strip("'\""))
    else:
        raise ValueError("OUTPUT_SPK_FILE not found in setup file")

    # Run mkspk
    logger.info(f"Running mkspk with setup file: {setup_file}")

    # mkspk uses paths relative to current directory, so we run from project root
    project_root = _get_project_root()

    result = subprocess.run(
        [str(mkspk), "-setup", str(setup_file)],
        cwd=str(project_root),
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        logger.error(f"mkspk failed: {result.stdout}\n{result.stderr}")
        raise RuntimeError(f"mkspk failed with return code {result.returncode}:\n{result.stdout}\n{result.stderr}")

    # Resolve output path
    if output_spk.is_absolute():
        final_output = output_spk
    else:
        final_output = project_root / output_spk

    if not final_output.exists():
        raise RuntimeError(f"Expected output file not created: {final_output}")

    logger.info(f"Created ephemeris file: {final_output}")
    return final_output


def create_observer_kernel(
    setup_file: Path,
    output_spk: Path,
    pck_file: Optional[Path] = None
) -> Path:
    """
    Create an observer/ground station SPK file using pinpoint.

    The setup file defines the station location (lat/lon/altitude).
    See templates/kernel_generation/pinpoint_template.setup for an example.

    Args:
        setup_file: Path to the pinpoint definition file
        output_spk: Path where the output SPK file should be created
        pck_file: Optional path to PCK file with Earth radii. If not provided,
                 uses the default generic PCK.

    Returns:
        Path to the created SPK file

    Raises:
        FileNotFoundError: If setup file or binary not found
        RuntimeError: If pinpoint fails
    """
    setup_file = Path(setup_file)
    output_spk = Path(output_spk)

    if not setup_file.exists():
        raise FileNotFoundError(f"Setup file not found: {setup_file}")

    pinpoint = _get_binary_path("pinpoint")
    project_root = _get_project_root()

    # Use default PCK if not provided
    if pck_file is None:
        pck_file = project_root / "data" / "spice_kernels" / "generic" / "pck" / "pck00011.tpc"
    pck_file = Path(pck_file)

    if not pck_file.exists():
        raise FileNotFoundError(f"PCK file not found: {pck_file}")

    # Ensure output directory exists
    output_spk.parent.mkdir(parents=True, exist_ok=True)

    # Build command
    cmd = [
        str(pinpoint),
        "-def", str(setup_file),
        "-spk", str(output_spk),
        "-pck", str(pck_file)
    ]

    logger.info(f"Running pinpoint with definition file: {setup_file}")

    result = subprocess.run(
        cmd,
        cwd=str(project_root),
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        logger.error(f"pinpoint failed: {result.stderr}")
        raise RuntimeError(f"pinpoint failed with return code {result.returncode}:\n{result.stderr}")

    if not output_spk.exists():
        raise RuntimeError(f"Expected output file not created: {output_spk}")

    logger.info(f"Created observer kernel: {output_spk}")
    return output_spk


def create_orientation_kernel(
    setup_file: Path,
    metakernel: Path,
    output_ck: Path,
    output_sclk: Optional[Path] = None,
    tolerance_degrees: float = 1.0
) -> Tuple[Path, Path]:
    """
    Create orientation (CK) and clock (SCLK) kernels using prediCkt.

    prediCkt computes satellite attitude based on geometric rules defined
    in the setup file (e.g., point X-axis toward Earth, Z-axis to celestial north).

    The clock file is generated automatically via the -newsclk flag.

    See templates/kernel_generation/predickt_template.setup for an example.

    Args:
        setup_file: Path to the prediCkt specification file
        metakernel: Path to a metakernel that loads required kernels
                   (satellite ephemeris, planetary ephemeris, leapseconds)
        output_ck: Path where the output CK file should be created
        output_sclk: Path for the SCLK file. If not provided, uses same name
                    as CK with .tsc extension.
        tolerance_degrees: Fit tolerance in degrees (default 1.0)

    Returns:
        Tuple of (ck_path, sclk_path)

    Raises:
        FileNotFoundError: If setup file, metakernel, or binary not found
        RuntimeError: If prediCkt fails
    """
    setup_file = Path(setup_file)
    metakernel = Path(metakernel)
    output_ck = Path(output_ck)

    if not setup_file.exists():
        raise FileNotFoundError(f"Setup file not found: {setup_file}")
    if not metakernel.exists():
        raise FileNotFoundError(f"Metakernel not found: {metakernel}")

    predickt = _get_binary_path("prediCkt")
    project_root = _get_project_root()

    # Default SCLK path: same name as CK with .tsc extension
    if output_sclk is None:
        output_sclk = output_ck.with_suffix(".tsc")
    output_sclk = Path(output_sclk)

    # Ensure output directory exists
    output_ck.parent.mkdir(parents=True, exist_ok=True)
    output_sclk.parent.mkdir(parents=True, exist_ok=True)

    # Build command
    cmd = [
        str(predickt),
        "-furnish", str(metakernel),
        "-spec", str(setup_file),
        "-ck", str(output_ck),
        "-tol", str(tolerance_degrees),
        "-newsclk", str(output_sclk)
    ]

    logger.info(f"Running prediCkt with spec file: {setup_file}")

    result = subprocess.run(
        cmd,
        cwd=str(project_root),
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        logger.error(f"prediCkt failed: {result.stderr}")
        logger.error(f"prediCkt stdout: {result.stdout}")
        raise RuntimeError(f"prediCkt failed with return code {result.returncode}:\n{result.stderr}\n{result.stdout}")

    if not output_ck.exists():
        raise RuntimeError(f"Expected CK file not created: {output_ck}")
    if not output_sclk.exists():
        raise RuntimeError(f"Expected SCLK file not created: {output_sclk}")

    logger.info(f"Created orientation kernel: {output_ck}")
    logger.info(f"Created clock kernel: {output_sclk}")

    return output_ck, output_sclk


def _parse_setup_value(setup_file: Path, key: str) -> Optional[str]:
    """
    Parse a value from a SPICE setup file.

    Handles the \begindata / \begintext format.

    Args:
        setup_file: Path to the setup file
        key: The key to look for (e.g., "OUTPUT_SPK_FILE")

    Returns:
        The value as a string, or None if not found
    """
    in_data_section = False

    with open(setup_file, 'r') as f:
        for line in f:
            line = line.strip()

            if line.startswith('\\begindata'):
                in_data_section = True
                continue
            elif line.startswith('\\begintext'):
                in_data_section = False
                continue

            if in_data_section and '=' in line:
                parts = line.split('=', 1)
                if parts[0].strip() == key:
                    return parts[1].strip()

    return None


def check_binaries_installed() -> dict:
    """
    Check which SPICE binaries are installed.

    Returns:
        Dict mapping binary names to their paths (or None if not found)
    """
    binaries = ["mkspk", "pinpoint", "prediCkt"]
    result = {}

    for binary in binaries:
        try:
            path = _get_binary_path(binary)
            result[binary] = str(path)
        except FileNotFoundError:
            result[binary] = None

    return result


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Checking SPICE binary installation...")
    status = check_binaries_installed()

    for binary, path in status.items():
        if path:
            print(f"  {binary}: {path}")
        else:
            print(f"  {binary}: NOT FOUND")
