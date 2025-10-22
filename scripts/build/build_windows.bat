@echo off
REM Build script for PyProphet Windows executable using PyInstaller

echo ============================================
echo Building PyProphet for Windows
echo ============================================

REM Install UPX for compression
echo Installing UPX...
choco install upx -y >nul 2>&1 || echo UPX install skipped (may already be installed)

REM Clean previous builds
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist pyprophet.spec del /q pyprophet.spec

REM Install/upgrade build dependencies
python -m pip install --upgrade pip setuptools wheel cython numpy pyinstaller

REM Install ONLY runtime dependencies (no dev/docs/testing extras)
echo Installing runtime dependencies only...
python -m pip install -e . --no-deps

REM Parse and install runtime dependencies from pyproject.toml
python -c "import tomllib, subprocess, sys; config = tomllib.load(open('pyproject.toml', 'rb')); deps = config['project']['dependencies']; [subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', dep.strip()]) for dep in deps if dep.strip() and not dep.strip().startswith('#')]"

REM Build Cython extensions in-place
python setup.py build_ext --inplace

REM Run PyInstaller in onefile mode (single executable)
echo Running PyInstaller (onefile mode)...
python -m PyInstaller ^
    --clean ^
    --noconfirm ^
    --onefile ^
    --name pyprophet ^
    --upx ^
    --log-level INFO ^
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
    --exclude-module tomli ^
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
    packaging/pyinstaller/run_pyprophet.py

if errorlevel 1 (
    echo Build failed!
    exit /b 1
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