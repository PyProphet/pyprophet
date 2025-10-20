@echo off
REM Build script for PyProphet Windows executable using PyInstaller

echo ============================================
echo Building PyProphet for Windows
echo ============================================

REM Clean previous builds
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

REM Install/upgrade build dependencies
python -m pip install --upgrade pip setuptools wheel cython numpy pyinstaller

REM Build Cython extensions in-place
python setup.py build_ext --inplace

REM Run PyInstaller with metadata collection
pyinstaller ^
    --name pyprophet ^
    --onedir ^
    --console ^
    --clean ^
    --noconfirm ^
    --log-level INFO ^
    --additional-hooks-dir=packaging/pyinstaller-hooks ^
    --collect-submodules pyprophet ^
    --collect-all numpy ^
    --collect-all pandas ^
    --collect-all scipy ^
    --collect-all sklearn ^
    --copy-metadata duckdb ^
    --copy-metadata duckdb-extensions ^
    --copy-metadata duckdb-extension-sqlite-scanner ^
    run_pyprophet.py

echo ============================================
echo Build complete! Executable at: dist\pyprophet\pyprophet.exe
echo ============================================

REM Create archive
cd dist
tar -czf ..\pyprophet-windows-x86_64.tar.gz pyprophet\
cd ..

echo Archive created: pyprophet-windows-x86_64.tar.gz