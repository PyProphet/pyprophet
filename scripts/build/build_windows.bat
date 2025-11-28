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

REM Clean previous builds
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist pyprophet.spec del /q pyprophet.spec
if exist *.egg-info rmdir /s /q *.egg-info 2>nul
if exist .eggs rmdir /s /q .eggs 2>nul

REM Clean Python cache
for /d /r %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d" 2>nul
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

REM Try onefile with collect-submodules instead of collect-all (less aggressive)
echo Running PyInstaller (onefile mode with optimized collection)...
python -m PyInstaller ^
    --clean ^
    --noconfirm ^
    --onefile ^
    --console ^
    --name pyprophet ^
    --log-level INFO ^
    --additional-hooks-dir packaging/pyinstaller/hooks ^
    --exclude-module pyarrow ^
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
    --hidden-import=pyprophet ^
    --hidden-import=pyprophet.main ^
    --collect-submodules pyprophet ^
    --collect-submodules numpy ^
    --collect-submodules pandas ^
    --collect-submodules scipy ^
    --collect-submodules sklearn ^
    --collect-submodules pyopenms ^
    --copy-metadata numpy ^
    --copy-metadata pandas ^
    --copy-metadata scipy ^
    --copy-metadata scikit-learn ^
    --copy-metadata pyopenms ^
    --copy-metadata duckdb ^
    --copy-metadata duckdb-extensions ^
    --copy-metadata duckdb-extension-sqlite-scanner ^
    packaging/pyinstaller/run_pyprophet.py

if errorlevel 1 (
    echo Onefile build failed, trying alternative approach...
    
    REM If onefile fails, fall back to onedir but wrap it in a self-extracting archive
    echo.
    echo Falling back to onedir with SFX wrapper...
    
    python -m PyInstaller ^
        --clean ^
        --noconfirm ^
        --onedir ^
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
        echo Both builds failed!
        exit /b 1
    )
    
    REM Create a self-extracting archive using 7-Zip if available
    where 7z >nul 2>&1
    if %errorlevel% equ 0 (
        echo Creating self-extracting archive with 7-Zip...
        7z a -sfx7z.sfx dist\pyprophet-sfx.exe dist\pyprophet\*
        if errorlevel 1 (
            echo SFX creation failed, keeping directory build
        ) else (
            move /y dist\pyprophet-sfx.exe dist\pyprophet.exe
            echo Created self-extracting executable: dist\pyprophet.exe
        )
    ) else (
        echo 7-Zip not found, keeping onedir build
        echo Note: Install 7-Zip for single-file executable support
    )
    
    set BUILD_TYPE=onedir
) else (
    set BUILD_TYPE=onefile
    echo Onefile build succeeded!
)

echo ============================================
if "%BUILD_TYPE%"=="onefile" (
    echo Build complete! Single executable at: dist\pyprophet.exe
    dir dist\pyprophet.exe
) else (
    echo Build complete! Executable at: dist\pyprophet\pyprophet.exe
    dir dist\pyprophet\pyprophet.exe
)
echo ============================================

REM Create ZIP archive
echo Creating ZIP archive...
if "%BUILD_TYPE%"=="onefile" (
    powershell -Command "Compress-Archive -Path dist\pyprophet.exe -DestinationPath pyprophet-windows-x86_64.zip -Force"
) else (
    powershell -Command "Compress-Archive -Path dist\pyprophet -DestinationPath pyprophet-windows-x86_64.zip -Force"
)

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
if "%BUILD_TYPE%"=="onefile" (
    echo   dist\pyprophet.exe --help
) else (
    echo   dist\pyprophet\pyprophet.exe --help
)