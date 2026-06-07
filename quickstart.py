#!/usr/bin/env python3
"""
LCAS Quick Start - Installation Verification
=============================================

Run: python quickstart.py

This script verifies your LCAS installation by:
1. Checking all required Python packages are installed
2. Verifying SPICE kernels and utilities exist
3. Loading a sample satellite configuration
4. Printing next steps

If all checks pass, your LCAS installation is ready to use!
"""

import sys
from pathlib import Path

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def print_header(title):
    """Print a formatted section header."""
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_check(name, passed, details=None):
    """Print a check result."""
    status = "[OK]" if passed else "[FAIL]"
    print(f"  {status} {name}")
    if details:
        print(f"       {details}")


def check_imports():
    """Check all required Python packages are installed."""
    print_header("Checking Python Dependencies")

    packages = [
        ("numpy", "numpy"),
        ("numpy-quaternion", "quaternion"),
        ("scipy", "scipy"),
        ("trimesh", "trimesh"),
        ("spiceypy", "spiceypy"),
        ("PyYAML", "yaml"),
        ("matplotlib", "matplotlib"),
        ("plotly", "plotly"),
        ("pandas", "pandas"),
        ("tqdm", "tqdm"),
    ]

    all_ok = True
    for name, import_name in packages:
        try:
            module = __import__(import_name)
            version = getattr(module, "__version__", "unknown")
            print_check(name, True, f"version {version}")
        except ImportError as e:
            print_check(name, False, str(e))
            all_ok = False

    return all_ok


def check_spice_kernels():
    """Check SPICE kernel files exist."""
    print_header("Checking SPICE Kernels")

    kernel_dir = PROJECT_ROOT / "data" / "spice_kernels"

    required_files = [
        ("de440.bsp (planetary ephemeris)", "generic/spk/de440.bsp"),
        ("naif0012.tls.pc (leap seconds)", "generic/lsk/naif0012.tls.pc"),
        ("Intelsat 901 metakernel", "missions/dst-is901/INTELSAT_901-metakernel.tm"),
    ]

    all_ok = True
    for name, rel_path in required_files:
        full_path = kernel_dir / rel_path
        if full_path.exists():
            size_mb = full_path.stat().st_size / (1024 * 1024)
            print_check(name, True, f"{size_mb:.1f} MB")
        else:
            print_check(name, False, f"Not found: {rel_path}")
            all_ok = False

    return all_ok


def check_spice_utilities():
    """Check SPICE utility binaries exist."""
    print_header("Checking SPICE Utilities")

    import platform
    system = platform.system().lower()

    utilities_dir = PROJECT_ROOT / "data" / "spice_utilities"

    if system == "windows":
        binaries = ["mkspk.exe", "pinpoint.exe", "prediCkt.exe"]
    else:
        binaries = ["mkspk", "pinpoint", "prediCkt"]

    all_ok = True
    for binary in binaries:
        binary_path = utilities_dir / binary
        if binary_path.exists():
            size_kb = binary_path.stat().st_size / 1024
            print_check(binary, True, f"{size_kb:.0f} KB")
        else:
            print_check(binary, False, "Not found")
            all_ok = False

    if not all_ok:
        print()
        print("  Note: SPICE utilities are optional. They are only needed")
        print("  for generating custom orbital kernels.")

    return all_ok


def check_lcas_modules():
    """Check LCAS modules can be imported."""
    print_header("Checking LCAS Modules")

    modules = [
        ("Config Manager", "src.config.rso_config_manager", "RSO_ConfigManager"),
        ("STL Loader", "src.io.stl_loader", "STLLoader"),
        ("SPICE Handler", "src.spice.spice_handler", "SpiceHandler"),
        ("Shadow Engine", "src.computation.shadow_engine", "compute_shadows"),
        ("Light Curve Generator", "src.computation.lightcurve_generator", "generate_lightcurves"),
        ("Articulation Engine", "src.articulation", "ArticulationEngine"),
    ]

    all_ok = True
    for name, module_path, attr_name in modules:
        try:
            module = __import__(module_path, fromlist=[attr_name])
            obj = getattr(module, attr_name)
            print_check(name, True)
        except Exception as e:
            print_check(name, False, str(e))
            all_ok = False

    return all_ok


def load_sample_config():
    """Try to load the sample Intelsat 901 configuration."""
    print_header("Loading Sample Configuration")

    try:
        from src.config.rso_config_manager import RSO_ConfigManager

        config_manager = RSO_ConfigManager(PROJECT_ROOT)
        config = config_manager.load_config("intelsat_901/intelsat_901_config.yaml")

        satellite_name = config.get("satellite", {}).get("name", "Unknown")
        num_components = len(config.get("satellite", {}).get("components", []))

        print_check("Configuration loaded", True)
        print(f"       Satellite: {satellite_name}")
        print(f"       Components: {num_components}")

        return True
    except Exception as e:
        print_check("Configuration loading", False, str(e))
        return False


def print_summary(results):
    """Print final summary and next steps."""
    print_header("Summary")

    all_passed = all(results.values())
    critical_passed = results.get("imports", False) and results.get("lcas_modules", False)

    if all_passed:
        print("  All checks passed! Your LCAS installation is ready.")
    elif critical_passed:
        print("  Core installation OK. Some optional components may be missing.")
    else:
        print("  Some critical checks failed. Please review the errors above.")
        print()
        print("  Try reinstalling:")
        print("    python install_dependencies.py")
        return

    print()
    print("-" * 60)
    print("  Next Steps:")
    print("-" * 60)
    print()
    print("  1. Start Jupyter Lab:")
    print("     Linux/Mac: ./start-jupyter.sh")
    print("     Windows:   start-jupyter.bat")
    print()
    print("  2. Open the example notebook:")
    print("     notebooks/lcas_stl_pipeline-is901-3.py")
    print()
    print("  3. Run all cells to generate your first light curve!")
    print()


def main():
    """Run all verification checks."""
    print()
    print("=" * 60)
    print("  LCAS Quick Start - Installation Verification")
    print("=" * 60)

    results = {}

    # Run checks
    results["imports"] = check_imports()
    results["spice_kernels"] = check_spice_kernels()
    results["spice_utilities"] = check_spice_utilities()
    results["lcas_modules"] = check_lcas_modules()
    results["sample_config"] = load_sample_config()

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
