#!/bin/bash
# Build script for PyProphet macOS executable using PyInstaller

set -e

echo "============================================"
echo "Building PyProphet for macOS"
echo "============================================"

# Install UPX for compression (strip is built-in on macOS)
echo "Installing UPX..."
if ! command -v upx &> /dev/null; then
    if command -v brew &> /dev/null; then
        brew install upx
    else
        echo "Warning: UPX not found and Homebrew not available. Skipping UPX compression."
    fi
fi

# Install/upgrade build dependencies
python3 -m pip install --upgrade pip setuptools wheel cython numpy pyinstaller

# Install ONLY runtime dependencies (no dev/docs/testing extras)
echo "Installing runtime dependencies only..."
python3 -m pip install -e . --no-deps

# Parse and install runtime dependencies from pyproject.toml
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
python3 setup.py build_ext --inplace

# Collect compiled extension binaries for pyprophet/scoring
ADD_BINARY_ARGS=()
for so in pyprophet/scoring/_optimized*.so pyprophet/scoring/_optimized*.dylib; do
  if [ -f "$so" ]; then
    ADD_BINARY_ARGS+=(--add-binary "$so:pyprophet/scoring")
    echo "Adding binary: $so"
  fi
done

# Locate xgboost native library (macOS uses .dylib)
DYLIB_PATH=$(python3 - <<'PY'
import importlib, os
try:
    m = importlib.import_module("xgboost")
    p = os.path.join(os.path.dirname(m.__file__), "lib", "libxgboost.dylib")
    print(p if os.path.exists(p) else "")
except Exception:
    print("")
PY
)
if [ -n "$DYLIB_PATH" ]; then
  echo "Including xgboost native lib: $DYLIB_PATH"
  ADD_BINARY_ARGS+=(--add-binary "$DYLIB_PATH:xgboost/lib")
fi

# Clean previous builds
rm -rf build dist pyprophet.spec

# Run PyInstaller in onefile mode (single executable)
echo "Running PyInstaller (onefile mode)..."
python3 -m PyInstaller \
  --clean \
  --noconfirm \
  --onefile \
  --name pyprophet \
  --strip \
  --log-level INFO \
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
  --exclude-module tomli \
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
  packaging/pyinstaller/run_pyprophet.py

# Post-process: UPX compress the final binary
if command -v upx &> /dev/null; then
    echo "Post-processing: compressing with UPX..."
    upx --best --lzma dist/pyprophet 2>/dev/null || echo "UPX compression skipped (may have failed)"
fi

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