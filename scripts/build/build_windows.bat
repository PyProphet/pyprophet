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
if exist *.egg-info rmdir /s /q *.egg-info
if exist .eggs rmdir /s /q .eggs

REM Clean Python cache
for /d /r %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
del /s /q *.pyc 2>nul
del /s /q *.pyo 2>nul

REM Install/upgrade build dependencies
python -m pip install --upgrade pip setuptools wheel cython numpy pyinstaller

REM Parse and install runtime dependencies from pyproject.toml
echo Installing runtime dependencies...
python -c "import tomllib, subprocess, sys; config = tomllib.load(open('pyproject.toml', 'rb')); deps = config['project']['dependencies']; [subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', dep.strip()]) for dep in deps if dep.strip() and not dep.strip().startswith('#')]"

REM Install the package in editable mode
echo Installing pyprophet in editable mode...
python -m pip install -e .

REM Build C extensions in-place
echo Building C extensions...
python setup.py build_ext --inplace

REM Clean previous builds again
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if not exist dist mkdir dist

REM Run PyInstaller in onefile mode (using --collect-all for problematic packages)
echo Running PyInstaller (onefile mode)...
python -m PyInstaller ^
    --clean ^
    --noconfirm ^
    --onefile ^
    --console ^
    --name pyprophet ^
    --log-level INFO ^
    --additional-hooks-dir packaging/pyinstaller/hooks ^
    --hidden-import=pyprophet ^
    --hidden-import=pyprophet.main ^
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