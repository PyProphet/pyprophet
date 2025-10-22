@echo off
REM Build script for PyProphet Windows executable using PyInstaller

echo ============================================
echo Building PyProphet for Windows
echo ============================================

REM Check Python version
for /f "tokens=*" %%i in ('python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do set PYTHON_VERSION=%%i
echo Using Python version: %PYTHON_VERSION%

REM Check if version is 3.11 or higher
python -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)"
if errorlevel 1 (
    echo ERROR: Python 3.11+ is required for building.
    echo Your Python version: %PYTHON_VERSION%
    echo.
    echo Please install Python 3.11 or later from https://www.python.org/downloads/
    exit /b 1
)

REM Install UPX for compression
echo Installing UPX...
choco install upx -y >nul 2>&1 || echo UPX install skipped (may already be installed)

REM Clean previous builds
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist pyprophet.spec del /q pyprophet.spec
if exist build_entry.py del /q build_entry.py

REM Install/upgrade build dependencies
python -m pip install --upgrade pip setuptools wheel cython numpy pyinstaller

REM Parse and install runtime dependencies from pyproject.toml
echo Installing runtime dependencies...
python -c "import tomllib, subprocess, sys; config = tomllib.load(open('pyproject.toml', 'rb')); deps = config['project']['dependencies']; [subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', dep.strip()]) for dep in deps if dep.strip() and not dep.strip().startswith('#')]"

REM Build Cython extensions in-place
echo Building C extensions...
python setup.py build_ext --inplace

REM Create a minimal entry point script
echo Creating build entry point...
(
echo #!/usr/bin/env python3
echo """
echo PyProphet build entry point for PyInstaller.
echo This script imports pyprophet as an installed package to avoid path conflicts.
echo """
echo import sys
echo import os
echo.
echo # Ensure we don't import from the source directory
echo source_dir = os.path.dirname^(os.path.abspath^(__file__^)^)
echo if source_dir in sys.path:
echo     sys.path.remove^(source_dir^)
echo.
echo # Import and run pyprophet main
echo from pyprophet.main import cli
echo.
echo if __name__ == '__main__':
echo     cli^(^)
) > build_entry.py

REM Run PyInstaller in onefile mode (single executable)
echo Running PyInstaller (onefile mode)...
python -m PyInstaller ^
    --clean ^
    --noconfirm ^
    --onefile ^
    --name pyprophet ^
    --log-level INFO ^
    --additional-hooks-dir packaging/pyinstaller/hooks ^
    --paths . ^
    --exclude-module sphinx ^
    --exclude-module sphinx_rtd_theme ^
    --exclude-module pydata_sphinx_theme ^
    --exclude-module sphinx_copybutton ^
    --exclude-module sphinx.ext ^
    --exclude-module alabaster ^
    --exclude-module babel ^
    --exclude-module docutils ^
    --exclude-module mypy ^
    --exclude-module pytest ^
    --exclude-module pytest-regtest ^
    --exclude-module pytest-xdist ^
    --exclude-module black ^
    --exclude-module ruff ^
    --collect-submodules pyprophet ^
    --collect-all numpy ^
    --collect-all pandas ^
    --collect-all scipy ^
    --collect-all sklearn ^
    --collect-all pyopenms ^
    --copy-metadata duckdb ^
    --copy-metadata duckdb-extensions ^
    --copy-metadata duckdb-extension-sqlite-scanner ^
    --copy-metadata pyopenms ^
    build_entry.py

if errorlevel 1 (
    echo Build failed!
    del /q build_entry.py 2>nul
    exit /b 1
)

REM Clean up temporary entry script
del /q build_entry.py 2>nul

REM Post-process: UPX compress the final binary
where upx >nul 2>&1
if %errorlevel% equ 0 (
    echo Post-processing: compressing with UPX...
    upx --best --lzma dist\pyprophet.exe 2>nul || echo UPX compression skipped
)

echo ============================================
echo Build complete! Single executable at: dist\pyprophet.exe
dir dist\pyprophet.exe
echo ============================================

REM Create ZIP archive
echo Creating ZIP archive...
powershell -Command "Compress-Archive -Path dist\pyprophet.exe -DestinationPath pyprophet-windows-x86_64.zip -Force"

if errorlevel 1 (
    echo Archive creation failed!
    exit /b 1
)

echo Archive created: pyprophet-windows-x86_64.zip

REM Generate SHA256 checksum
powershell -Command "(Get-FileHash pyprophet-windows-x86_64.zip -Algorithm SHA256).Hash | Out-File -FilePath pyprophet-windows-x86_64.zip.sha256 -Encoding ASCII"
echo Checksum created: pyprophet-windows-x86_64.zip.sha256

echo.
echo To test locally:
echo   dist\pyprophet.exe --help