# LCAS Installation Guide

This guide provides detailed installation instructions for the Light Curve Analysis Suite (LCAS) on Windows and Linux systems.

## Prerequisites

Before installing LCAS, ensure you have:

- **Python 3.11 or 3.12** installed and accessible from the command line
- **Git** (optional, for cloning the repository)
- **Internet connection** (for downloading dependencies and SPICE data)

### Checking Python Version

```bash
python --version
# Should output: Python 3.11.x or Python 3.12.x
```

If you have multiple Python versions, you may need to use `python3` or `python3.11` instead.

---

## Quick Installation

For most users, this single command handles everything:

```bash
python install_dependencies.py
```

This will:
1. Create a virtual environment (`.venv/`)
2. Download SPICE utilities for your operating system
3. Download the de440.bsp planetary ephemeris (~114 MB)
4. Install all Python dependencies

---

## Detailed Installation: Windows

### Step 1: Download LCAS

Download and extract LCAS to a location of your choice, for example `C:\Users\YourName\lcas`.

Or clone with Git:
```cmd
git clone <repository-url> C:\Users\YourName\lcas
cd C:\Users\YourName\lcas
```

### Step 2: Run the Installer

Open Command Prompt or PowerShell and navigate to the LCAS directory:

```cmd
cd C:\Users\YourName\lcas
python install_dependencies.py
```

The installer will display progress for each step:
- Step 1: Virtual environment creation
- Step 2: Pip upgrade
- Step 3: SPICE infrastructure download
- Step 4: Remaining dependencies

### Step 3: Verify Installation

```cmd
.venv\Scripts\activate
python quickstart.py
```

You should see all checks passing:
```
[OK] numpy
[OK] spiceypy
...
All checks passed! Your LCAS installation is ready.
```

### Step 4: Start Using LCAS

```cmd
start-jupyter.bat
```

This opens Jupyter Lab. Navigate to `notebooks/` and open `lcas_stl_pipeline-is901-3.py`.

---

## Detailed Installation: Linux

### Step 1: Download LCAS

```bash
git clone <repository-url> ~/lcas
cd ~/lcas
```

### Step 2: Run the Installer

```bash
python install_dependencies.py
```

Or with Python 3 explicitly:
```bash
python3 install_dependencies.py
```

### Step 3: Verify Installation

```bash
source .venv/bin/activate
python quickstart.py
```

### Step 4: Start Using LCAS

```bash
./start-jupyter.sh
```

---

## Skipping SPICE Downloads

If you already have SPICE data or want to install it later:

```bash
python install_dependencies.py --skip-spice
```

You can download SPICE files later by running the installer again without `--skip-spice`.

---

## Verifying the Installation

Run the quickstart script to verify everything is working:

```bash
# Activate virtual environment first
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows

# Run verification
python quickstart.py
```

The script checks:
- All Python packages are installed
- SPICE kernels exist
- SPICE utilities are available
- LCAS modules can be imported
- Sample configuration loads correctly

---

## Troubleshooting

### "Python not found"

- Windows: Install Python from [python.org](https://python.org) and ensure "Add to PATH" is checked
- Linux: Install via package manager (e.g., `sudo apt install python3.11`)

### "Virtual environment already exists"

If prompted, choose `y` to recreate it, or use an existing one.

### "Failed to download SPICE files"

This usually indicates a network issue. Try:
1. Check your internet connection
2. Re-run `python install_dependencies.py` (it skips existing files)
3. Download files manually from [NAIF](https://naif.jpl.nasa.gov/)

### Import errors when running notebooks

Make sure the virtual environment is activated:
```bash
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
```

### Jupyter not starting

Install Jupyter manually:
```bash
source .venv/bin/activate
pip install jupyterlab
```

---

## Updating LCAS

To update to a new version:

1. Pull the latest code (if using Git):
   ```bash
   git pull
   ```

2. Re-run the installer to update dependencies:
   ```bash
   python install_dependencies.py
   ```

The installer will skip re-downloading SPICE files that already exist.

---

## Uninstalling

To remove LCAS:

1. Delete the virtual environment:
   ```bash
   rm -rf .venv      # Linux/Mac
   rmdir /s .venv    # Windows
   ```

2. Delete the LCAS directory

No system-wide changes are made by LCAS - everything is contained in the project folder.
