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
    echo ""
    echo "Please install Python 3.11 or later:"
    echo "  sudo apt update"
    echo "  sudo apt install python3.11 python3.11-venv python3.11-dev"
    echo ""
    echo "Then run the build with:"
    echo "  PYTHON=python3.11 bash scripts/build/build_linux.sh"
    exit 1
fi

# Install system tools for stripping (skip upx - it breaks PyInstaller on Linux)
echo "Installing build tools..."
sudo apt-get update -qq
sudo apt-get install -y binutils

# Install/upgrade build dependencies
$PYTHON -m pip install --upgrade pip setuptools wheel cython numpy pyinstaller

# Parse and install runtime dependencies from pyproject.toml
echo "Installing runtime dependencies..."
$PYTHON << 'PYEOF'
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

# Build and install pyprophet as a wheel (cleanest approach)
echo "Building and installing pyprophet package..."
$PYTHON -m pip wheel --no-deps --wheel-dir /tmp/pyprophet_wheels .
$PYTHON -m pip install --force-reinstall --no-deps /tmp/pyprophet_wheels/pyprophet-*.whl

# Verify installation
echo "Verifying pyprophet installation..."
$PYTHON -c "import pyprophet; print(f'PyProphet installed at: {pyprophet.__file__}')"

# Get the site-packages location where pyprophet was installed
SITE_PACKAGES=$($PYTHON -c "import pyprophet, os; print(os.path.dirname(pyprophet.__file__))")
echo "PyProphet package location: ${SITE_PACKAGES}"

# Collect compiled extension binaries from the installed package
ADD_BINARY_ARGS=()
for so in "${SITE_PACKAGES}"/pyprophet/scoring/_optimized*.so; do
  if [ -f "$so" ]; then
    ADD_BINARY_ARGS+=(--add-binary "$so:pyprophet/scoring")
    echo "Adding binary: $so"
  fi
done

# Clean previous PyInstaller builds
rm -rf build dist pyprophet.spec

# Change to a temporary directory to avoid picking up source files
# This ensures PyInstaller only sees installed packages
BUILD_DIR=$(mktemp -d)
echo "Using temporary build directory: ${BUILD_DIR}"
cd "${BUILD_DIR}"

# Copy only the necessary files
cp "${OLDPWD}/packaging/pyinstaller/run_pyprophet.py" .
cp -r "${OLDPWD}/packaging/pyinstaller/hooks" .

# Run PyInstaller in onefile mode (single executable)
echo "Running PyInstaller (onefile mode)..."
$PYTHON -m PyInstaller \
  --clean \
  --noconfirm \
  --onefile \
  --name pyprophet \
  --strip \
  --log-level INFO \
  --additional-hooks-dir hooks \
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
  run_pyprophet.py

# Move the built executable back to the original directory
mv dist/pyprophet "${OLDPWD}/dist/"

# Return to original directory
cd "${OLDPWD}"

# Clean up temporary build directory and wheel directory
rm -rf "${BUILD_DIR}" /tmp/pyprophet_wheels

# NOTE: UPX compression is NOT applied on Linux because it breaks PyInstaller executables

echo "============================================"
echo "Build complete! Single executable at: dist/pyprophet"
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