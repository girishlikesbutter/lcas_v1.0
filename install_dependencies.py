#!/usr/bin/env python3
"""
LCAS Dependency Installer

Creates a virtual environment and installs all dependencies.
Also downloads SPICE utilities and planetary ephemeris kernels.

Usage:
    python install_dependencies.py [--skip-spice] [--venv-name NAME]

Options:
    --skip-spice    Skip downloading SPICE utilities and kernels
    --venv-name     Name of virtual environment folder (default: .venv)
"""
import subprocess
import platform
import shutil
import sys
import venv
import urllib.request
import stat
from pathlib import Path


# =============================================================================
# SPICE Download Configuration
# =============================================================================

SPICE_UTILITIES = {
    'linux': {
        'base_url': 'https://naif.jpl.nasa.gov/pub/naif/utilities/PC_Linux_64bit/',
        'binaries': ['mkspk', 'pinpoint', 'prediCkt']
    },
    'windows': {
        'base_url': 'https://naif.jpl.nasa.gov/pub/naif/utilities/PC_Windows_64bit/',
        'binaries': ['mkspk.exe', 'pinpoint.exe', 'prediCkt.exe']
    }
}

DE440_URL = 'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440.bsp'
DE440_SIZE_MB = 114  # Approximate size for progress display


def create_venv(venv_path):
    """
    Create a virtual environment.

    Args:
        venv_path: Path where to create the venv
    """
    print(f"Creating virtual environment at: {venv_path}")
    venv.create(venv_path, with_pip=True)
    print("Virtual environment created successfully.")


def get_venv_python(venv_path):
    """
    Get the path to the Python executable in the venv.

    Args:
        venv_path: Path to the virtual environment

    Returns:
        Path: Path to the Python executable
    """
    system = platform.system()
    if system == 'Windows':
        return venv_path / 'Scripts' / 'python.exe'
    else:
        return venv_path / 'bin' / 'python'


def run_pip_install(python_exe, args, description):
    """
    Run pip install with given arguments using the venv Python.

    Args:
        python_exe: Path to Python executable
        args: List of arguments to pass to pip install
        description: Description of what's being installed (for logging)
    """
    cmd = [str(python_exe), '-m', 'pip', 'install'] + args
    print(f"Running: {' '.join(cmd)}\n")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\nError: Failed to install {description}")
        sys.exit(1)


# =============================================================================
# SPICE Download Functions
# =============================================================================

def download_file_with_progress(url, dest_path, description=None):
    """
    Download a file with progress indicator.

    Args:
        url: URL to download from
        dest_path: Path where to save the file
        description: Optional description for progress display

    Returns:
        bool: True if successful, False otherwise
    """
    desc = description or dest_path.name

    def progress_hook(block_num, block_size, total_size):
        if total_size > 0:
            downloaded = block_num * block_size
            percent = min(100, (downloaded / total_size) * 100)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\r  {desc}: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='', flush=True)
        else:
            downloaded = block_num * block_size
            mb_downloaded = downloaded / (1024 * 1024)
            print(f"\r  {desc}: {mb_downloaded:.1f} MB downloaded", end='', flush=True)

    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest_path, reporthook=progress_hook)
        print()  # New line after progress
        return True
    except Exception as e:
        print(f"\n  Error downloading {desc}: {e}")
        return False


def download_spice_utilities(project_root):
    """
    Download SPICE utilities for the current OS.

    Args:
        project_root: Path to project root directory

    Returns:
        bool: True if all downloads successful
    """
    system = platform.system().lower()

    if system == 'linux':
        os_key = 'linux'
    elif system == 'windows':
        os_key = 'windows'
    else:
        print(f"  Warning: SPICE utilities not available for {system}")
        print("  You can download them manually from: https://naif.jpl.nasa.gov/naif/utilities.html")
        return False

    config = SPICE_UTILITIES[os_key]
    utilities_dir = project_root / 'data' / 'spice_utilities'
    utilities_dir.mkdir(parents=True, exist_ok=True)

    all_success = True
    for binary in config['binaries']:
        dest_path = utilities_dir / binary

        # Skip if already exists
        if dest_path.exists() and dest_path.stat().st_size > 0:
            print(f"  {binary}: Already exists, skipping")
            continue

        url = config['base_url'] + binary
        success = download_file_with_progress(url, dest_path, binary)

        if success:
            # Make executable on Linux
            if system == 'linux':
                dest_path.chmod(dest_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        else:
            all_success = False

    return all_success


def download_de440_kernel(project_root):
    """
    Download the de440.bsp planetary ephemeris kernel.

    Args:
        project_root: Path to project root directory

    Returns:
        bool: True if download successful
    """
    kernel_dir = project_root / 'data' / 'spice_kernels' / 'generic' / 'spk'
    dest_path = kernel_dir / 'de440.bsp'

    # Skip if already exists and has reasonable size (> 100 MB)
    if dest_path.exists():
        size_mb = dest_path.stat().st_size / (1024 * 1024)
        if size_mb > 100:
            print(f"  de440.bsp: Already exists ({size_mb:.1f} MB), skipping")
            return True

    print(f"  Note: de440.bsp is ~{DE440_SIZE_MB} MB, this may take a few minutes...")
    return download_file_with_progress(DE440_URL, dest_path, "de440.bsp")


def setup_spice_infrastructure(project_root):
    """
    Download SPICE utilities and kernels.

    Args:
        project_root: Path to project root directory

    Returns:
        bool: True if setup was successful
    """
    print("\nDownloading SPICE utilities...")
    utilities_ok = download_spice_utilities(project_root)

    print("\nDownloading planetary ephemeris kernel...")
    kernel_ok = download_de440_kernel(project_root)

    return utilities_ok and kernel_ok


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    args = sys.argv[1:]
    skip_spice = '--skip-spice' in args

    venv_name = '.venv'
    if '--venv-name' in args:
        idx = args.index('--venv-name')
        if idx + 1 < len(args):
            venv_name = args[idx + 1]

    return venv_name, skip_spice


def main():
    print("=" * 60)
    print("LCAS Dependency Installer")
    print("=" * 60)
    print()

    # Parse arguments
    venv_name, skip_spice = parse_args()

    # Paths
    project_root = Path(__file__).parent.resolve()
    venv_path = project_root / venv_name
    requirements_file = project_root / 'requirements.txt'

    # Check requirements.txt exists
    if not requirements_file.exists():
        print(f"Error: requirements.txt not found at {requirements_file}")
        sys.exit(1)

    # Step 1: Create virtual environment
    print("-" * 60)
    print("Step 1: Setting up virtual environment")
    print("-" * 60)

    if venv_path.exists():
        print(f"Virtual environment already exists at: {venv_path}")
        response = input("Recreate it? [y/N]: ").strip().lower()
        if response == 'y':
            print("Removing existing virtual environment...")
            shutil.rmtree(venv_path)
            create_venv(venv_path)
        else:
            print("Using existing virtual environment.")
    else:
        create_venv(venv_path)

    python_exe = get_venv_python(venv_path)
    if not python_exe.exists():
        print(f"Error: Python executable not found at {python_exe}")
        sys.exit(1)

    print()

    # Step 2: Upgrade pip
    print("-" * 60)
    print("Step 2: Upgrading pip")
    print("-" * 60)
    run_pip_install(python_exe, ['--upgrade', 'pip'], "pip upgrade")
    print()

    # Step 3: Download SPICE infrastructure
    print("-" * 60)
    print("Step 3: Setting up SPICE infrastructure")
    print("-" * 60)
    if skip_spice:
        print("Skipped (--skip-spice flag specified)")
        print("You can download SPICE files later by re-running without --skip-spice")
    else:
        spice_ok = setup_spice_infrastructure(project_root)
        if not spice_ok:
            print("\nWarning: Some SPICE files could not be downloaded.")
            print("LCAS will still work but some features may be limited.")
            print("You can try again later by re-running this script.")
    print()

    # Step 4: Install dependencies
    print("-" * 60)
    print("Step 4: Installing dependencies")
    print("-" * 60)
    run_pip_install(python_exe, ['-r', str(requirements_file)], "dependencies")
    print()

    # Success message with activation instructions
    print("=" * 60)
    print("Installation Complete!")
    print("=" * 60)
    print()

    system = platform.system()
    if system == 'Windows':
        activate_cmd = f"{venv_name}\\Scripts\\activate"
    else:
        activate_cmd = f"source {venv_name}/bin/activate"

    print("To activate the virtual environment:")
    print(f"  {activate_cmd}")
    print()
    print("Then you can run the LCAS notebooks:")
    print("  jupyter notebook notebooks/")
    print()
    print("Or import LCAS modules directly:")
    print("  from src.io.stl_loader import STLLoader")
    print()


if __name__ == '__main__':
    main()
