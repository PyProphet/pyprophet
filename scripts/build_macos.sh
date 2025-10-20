#!/bin/bash
# Build script for PyProphet macOS executable using PyInstaller

set -e

echo "============================================"
echo "Building PyProphet for macOS"
echo "============================================"

# Clean previous builds
rm -rf build dist

# Install/upgrade build dependencies
python3 -m pip install --upgrade pip setuptools wheel cython numpy pyinstaller

# Build Cython extensions in-place
python3 setup.py build_ext --inplace

# Run PyInstaller
pyinstaller \
    --name pyprophet \
    --onedir \
    --console \
    --clean \
    --noconfirm \
    --additional-hooks-dir=packaging/pyinstaller-hooks \
    run_pyprophet.py

echo "============================================"
echo "Build complete! Executable at: dist/pyprophet/pyprophet"
echo "============================================"

# Make executable
chmod +x dist/pyprophet/pyprophet

# Optional: Create a tarball for distribution
cd dist
tar -czf pyprophet-macos-$(uname -m).tar.gz pyprophet/
cd ..

echo "Archive created: dist/pyprophet-macos-$(uname -m).tar.gz"
