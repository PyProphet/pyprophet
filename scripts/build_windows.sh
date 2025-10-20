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

REM Run PyInstaller
pyinstaller ^
    --name pyprophet ^
    --onedir ^
    --console ^
    --clean ^
    --noconfirm ^
    --additional-hooks-dir=packaging/pyinstaller-hooks ^
    run_pyprophet.py

echo ============================================
echo Build complete! Executable at: dist\pyprophet\pyprophet.exe
echo ============================================
