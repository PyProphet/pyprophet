#!/usr/bin/env bash
# Build script for PyProphet Linux executable using PyInstaller

set -euo pipefail

PYTHON=${PYTHON:-python3}

echo "============================================"
echo "Building PyProphet for Linux"
echo "============================================"

# Check Python version
PYTHON_VERSION=$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Using Python version: $PYTHON_VERSION"

if [[ $(echo "$PYTHON_VERSION < 3.11" | bc -l) -eq 1 ]]; then
    echo "ERROR: Python 3.11+ is required for building."
    echo "Your Python version: $PYTHON_VERSION"
    exit 1
fi

# Install system tools
echo "Installing build tools..."
sudo apt-get update -qq
sudo apt-get install -y binutils

# Save the original directory
ORIGINAL_DIR="$(pwd)"

# Clean build artifacts
echo "Cleaning build artifacts..."
rm -rf build dist pyprophet.spec *.egg-info .eggs
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true

# Install/upgrade build dependencies
$PYTHON -m pip install --upgrade pip setuptools wheel cython numpy pyinstaller

# Parse and install runtime dependencies
echo "Installing runtime dependencies..."
$PYTHON << 'PYEOF'
import tomllib
import subprocess
import sys

with open('pyproject.toml', 'rb') as f:
    config = tomllib.load(f)

for dep in config['project']['dependencies']:
    dep = dep.strip()
    if dep and not dep.startswith('#'):
        try:
            print(f"Installing: {dep}")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', dep])
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to install {dep}: {e}")
PYEOF

# Install the package in editable mode
echo "Installing pyprophet in editable mode..."
$PYTHON -m pip install -e .

# Build C extensions in-place
echo "Building C extensions..."
$PYTHON setup.py build_ext --inplace

# Collect compiled extension binaries
ADD_BINARY_ARGS=()
for so in pyprophet/scoring/_optimized*.so pyprophet/scoring/_optimized*.cpython-*.so; do
  if [ -f "$so" ]; then
    ADD_BINARY_ARGS+=(--add-binary "$so:pyprophet/scoring")
    echo "Adding binary: $so"
  fi
done

# Locate xgboost native library
SO_PATH=$($PYTHON - <<'PY'
import importlib, os
try:
    m = importlib.import_module("xgboost")
    p = os.path.join(os.path.dirname(m.__file__), "lib", "libxgboost.so")
    print(p if os.path.exists(p) else "")
except Exception:
    print("")
PY
)
if [ -n "$SO_PATH" ]; then
  echo "Including xgboost native lib: $SO_PATH"
  ADD_BINARY_ARGS+=(--add-binary "$SO_PATH:xgboost/lib")
fi

# Clean previous builds
rm -rf build dist
mkdir -p dist

# Run PyInstaller with --onefile (using --collect-all for problematic packages)
echo "Running PyInstaller (onefile mode)..."
$PYTHON -m PyInstaller \
  --name pyprophet \
  --onefile \
  --console \
  --clean \
  --noconfirm \
  --log-level INFO \
  --additional-hooks-dir packaging/pyinstaller/hooks \
  --exclude-module pyarrow \
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
  --hidden-import=pyprophet \
  --hidden-import=pyprophet.main \
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

echo "============================================"
echo "Build complete! Executable at: dist/pyprophet"
ls -lh dist/pyprophet
echo "============================================"

# Create compressed archive for distribution
echo "Creating distribution archive..."
cd dist
ARCH=$(uname -m)
ARCHIVE_NAME="pyprophet-linux-${ARCH}.tar.gz"
tar -czf "../${ARCHIVE_NAME}" pyprophet
cd ..

echo "============================================"
echo "Archive created: ${ARCHIVE_NAME}"
echo "============================================"

# Generate SHA256 checksum
if command -v sha256sum &> /dev/null; then
    sha256sum "${ARCHIVE_NAME}" > "${ARCHIVE_NAME}.sha256"
    echo "Checksum: ${ARCHIVE_NAME}.sha256"
    cat "${ARCHIVE_NAME}.sha256"
fi

echo ""
echo "To test locally:"
echo "  ./dist/pyprophet --help"