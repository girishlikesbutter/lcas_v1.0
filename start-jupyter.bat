@echo off
REM LCAS Jupyter Launcher for Windows
REM Start Jupyter Lab server for the Light Curve Analysis Suite

REM Change to script directory
cd /d "%~dp0"

REM Check if virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo ========================================
    echo   Error: Virtual environment not found
    echo ========================================
    echo.
    echo Please run the installer first:
    echo   python install_dependencies.py
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment
call .venv\Scripts\activate.bat

echo ========================================
echo   LCAS - Light Curve Analysis Suite
echo   Starting Jupyter Lab Server
echo ========================================
echo.
echo Access the notebook server at:
echo   http://localhost:8888
echo.
echo Press Ctrl+C to stop the server
echo.

REM Start Jupyter Lab
jupyter lab --port=8888 --no-browser

pause
