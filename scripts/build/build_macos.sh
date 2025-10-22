#!/bin/bash
# Build script for PyProphet macOS executable using PyInstaller

set -e

echo "============================================"
echo "Building PyProphet for macOS"
echo "============================================"

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Using Python version: $PYTHON_VERSION"

REQUIRED_VERSION="3.11"
if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "ERROR: Python 3.11+ is required for building."
    echo "Your Python version: $PYTHON_VERSION"
    echo ""
    echo "Please install Python 3.11 or later using Homebrew:"
    echo "  brew install python@3.11"
    exit 1
fi

# Install/upgrade build dependencies
python3 -m pip install --upgrade pip setuptools wheel cython numpy pyinstaller

# Parse and install runtime dependencies from pyproject.toml
echo "Installing runtime dependencies..."
python3 << 'PYEOF'
import tomllib
import subprocess
import sys

with open('pyproject.toml', 'rb') as f:
    config = tomllib.load(f)

deps = config['project']['dependencies']

# Filter out any malformed entries and install one by one
for dep in deps:
    dep = dep.strip()
    if dep and not dep.startswith('#'):
        try:
            print(f"Installing: {dep}")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', dep])
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to install {dep}: {e}")
            continue
PYEOF

# Build C extensions in-place
echo "Building C extensions..."
python3 setup.py build_ext --inplace

# Collect compiled extension binaries for pyprophet/scoring
ADD_BINARY_ARGS=()
for so in pyprophet/scoring/_optimized*.so pyprophet/scoring/_optimized*.dylib; do
  if [ -f "$so" ]; then
    ADD_BINARY_ARGS+=(--add-binary "$so:pyprophet/scoring")
    echo "Adding binary: $so"
  fi
done

# Clean previous builds
rm -rf build dist pyprophet.spec

# Create a minimal entry point script that imports from the package
cat > build_entry.py << 'EOF'
#!/usr/bin/env python3
"""
PyProphet build entry point for PyInstaller.
This script imports pyprophet as an installed package to avoid path conflicts.
"""
import sys
import os

# Ensure we don't import from the source directory
source_dir = os.path.dirname(os.path.abspath(__file__))
if source_dir in sys.path:
    sys.path.remove(source_dir)

# Import and run pyprophet main
from pyprophet.main import cli

if __name__ == '__main__':
    cli()
EOF

# Run PyInstaller in onefile mode (single executable)
echo "Running PyInstaller (onefile mode)..."
python3 -m PyInstaller \
  --clean \
  --noconfirm \
  --onefile \
  --name pyprophet \
  --strip \
  --log-level INFO \
  --additional-hooks-dir packaging/pyinstaller/hooks \
  --paths . \
  --exclude-module sphinx \
  --exclude-module sphinx_rtd_theme \
  --exclude-module pydata_sphinx_theme \
  --exclude-module sphinx_copybutton \
  --exclude-module sphinx.ext \
  --exclude-module alabaster \
  --exclude-module babel \
  --exclude-module docutils \
  --exclude-module mypy \
  --exclude-module pytest \
  --exclude-module pytest-regtest \
  --exclude-module pytest-xdist \
  --exclude-module black \
  --exclude-module ruff \
  --collect-submodules pyprophet \
  --collect-all numpy \
  --collect-all pandas \
  --collect-all scipy \
  --collect-all sklearn \
  --collect-all pyopenms \
  --copy-metadata duckdb \
  --copy-metadata duckdb-extensions \
  --copy-metadata duckdb-extension-sqlite-scanner \
  --copy-metadata pyopenms \
  "${ADD_BINARY_ARGS[@]}" \
  build_entry.py

# Clean up temporary entry script
rm -f build_entry.py

echo "============================================"
echo "Build complete! Single executable at: dist/pyprophet"
ls -lh dist/pyprophet
echo "============================================"

# Make executable
chmod +x dist/pyprophet

# Create tarball for distribution
cd dist
ARCH=$(uname -m)
ARCHIVE_NAME="pyprophet-macos-${ARCH}.tar.gz"
tar -czf "../${ARCHIVE_NAME}" pyprophet
cd ..

echo "============================================"
echo "Archive created: ${ARCHIVE_NAME}"
echo "============================================"

# Generate SHA256 checksum
if command -v shasum &> /dev/null; then
    shasum -a 256 "${ARCHIVE_NAME}" > "${ARCHIVE_NAME}.sha256"
    echo "Checksum: ${ARCHIVE_NAME}.sha256"
    cat "${ARCHIVE_NAME}.sha256"
fi

echo ""
echo "To test locally:"
echo "  ./dist/pyprophet --help"