#!/usr/bin/env bash
# filepath: /workspaces/pyprophet/scripts/build/build_linux.sh
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

# Save the original directory
ORIGINAL_DIR="$(pwd)"

# Clean ALL build artifacts that might confuse PyInstaller
echo "Cleaning build artifacts..."
rm -rf build dist pyprophet.spec *.egg-info
rm -rf .eggs
# Clean any Python cache that might have source references
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true

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

# Uninstall any existing pyprophet installation
echo "Uninstalling any existing pyprophet..."
$PYTHON -m pip uninstall -y pyprophet 2>/dev/null || true

# Move pyprophet source directory OUT of the way completely
echo "Temporarily hiding pyprophet source directory..."
TEMP_BACKUP_DIR=$(mktemp -d)
if [ -d "pyprophet" ]; then
    mv pyprophet "${TEMP_BACKUP_DIR}/pyprophet_src"
fi
if [ -d "packaging" ]; then
    mv packaging "${TEMP_BACKUP_DIR}/packaging_src"
fi

# Build pyprophet wheel in a completely isolated temp directory
echo "Building pyprophet wheel in isolated directory..."
WHEEL_BUILD_DIR=$(mktemp -d)

# Copy ONLY the files needed for building (not the entire tree)
cp "${ORIGINAL_DIR}/pyproject.toml" "${WHEEL_BUILD_DIR}/"
cp "${ORIGINAL_DIR}/setup.py" "${WHEEL_BUILD_DIR}/"
cp "${ORIGINAL_DIR}/README.md" "${WHEEL_BUILD_DIR}/" 2>/dev/null || true
cp "${ORIGINAL_DIR}/LICENSE" "${WHEEL_BUILD_DIR}/" 2>/dev/null || true

# Copy source files
cp -r "${TEMP_BACKUP_DIR}/pyprophet_src" "${WHEEL_BUILD_DIR}/pyprophet"

cd "${WHEEL_BUILD_DIR}"

# Build wheel
$PYTHON -m pip wheel --no-deps --wheel-dir /tmp/pyprophet_wheels .

# Return to original directory
cd "${ORIGINAL_DIR}"

# Clean up wheel build directory
rm -rf "${WHEEL_BUILD_DIR}"

# Install pyprophet from wheel (now pyprophet source is hidden, so it MUST go to site-packages)
echo "Installing pyprophet from wheel to site-packages..."
$PYTHON -m pip install --no-deps --no-cache-dir /tmp/pyprophet_wheels/pyprophet-*.whl

# Restore the source directories
echo "Restoring source directories..."
if [ -d "${TEMP_BACKUP_DIR}/pyprophet_src" ]; then
    mv "${TEMP_BACKUP_DIR}/pyprophet_src" pyprophet
fi
if [ -d "${TEMP_BACKUP_DIR}/packaging_src" ]; then
    mv "${TEMP_BACKUP_DIR}/packaging_src" packaging
fi
rm -rf "${TEMP_BACKUP_DIR}"

# Verify installation is in site-packages, NOT source directory
echo "Verifying pyprophet installation..."
PYPROPHET_LOCATION=$($PYTHON -c "import pyprophet; print(pyprophet.__file__)")
echo "PyProphet installed at: ${PYPROPHET_LOCATION}"

# Ensure it's in site-packages
if [[ "$PYPROPHET_LOCATION" == *"/site-packages/"* ]]; then
    echo "✓ PyProphet correctly installed in site-packages"
else
    echo "✗ ERROR: PyProphet is NOT in site-packages!"
    echo "  Location: ${PYPROPHET_LOCATION}"
    exit 1
fi

# Get the site-packages location where pyprophet was installed
SITE_PACKAGES=$($PYTHON -c "import pyprophet, os; print(os.path.dirname(pyprophet.__file__))")
echo "PyProphet package location: ${SITE_PACKAGES}"

# Verify numpy is installed correctly (not from source)
echo "Verifying numpy installation..."
NUMPY_LOCATION=$($PYTHON -c "import numpy; print(numpy.__file__)")
echo "NumPy installed at: ${NUMPY_LOCATION}"
if [[ "$NUMPY_LOCATION" == *"/site-packages/"* ]]; then
    echo "✓ NumPy correctly installed in site-packages"
else
    echo "✗ WARNING: NumPy may not be in site-packages!"
fi

$PYTHON -c "import pandas; print('✓ Pandas imports successfully')"

# Collect compiled extension binaries from the installed package
ADD_BINARY_ARGS=()
for so in "${SITE_PACKAGES}"/pyprophet/scoring/_optimized*.so; do
  if [ -f "$so" ]; then
    ADD_BINARY_ARGS+=(--add-binary "$so:pyprophet/scoring")
    echo "Adding binary: $so"
  fi
done

# Create dist directory in original location
mkdir -p "${ORIGINAL_DIR}/dist"

# Change to a temporary directory to avoid picking up ANY source files
BUILD_DIR=$(mktemp -d)
echo "Using temporary build directory: ${BUILD_DIR}"
cd "${BUILD_DIR}"

# Copy only the necessary files (NOT the entire source tree)
cp "${ORIGINAL_DIR}/packaging/pyinstaller/run_pyprophet.py" .
mkdir -p hooks
cp "${ORIGINAL_DIR}/packaging/pyinstaller/hooks"/*.py hooks/ 2>/dev/null || true

# Verify we're in a clean directory with no source pollution
echo "Build directory contents:"
ls -la

# Run PyInstaller in onefile mode
echo "Running PyInstaller (onefile mode)..."
echo "Current directory: $(pwd)"

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
  --copy-metadata numpy \
  --copy-metadata pandas \
  --copy-metadata scipy \
  --copy-metadata scikit-learn \
  --copy-metadata pyopenms \
  --copy-metadata duckdb \
  --copy-metadata duckdb-extensions \
  --copy-metadata duckdb-extension-sqlite-scanner \
  "${ADD_BINARY_ARGS[@]}" \
  run_pyprophet.py

# Move the built executable back to the original directory
echo "Moving executable to ${ORIGINAL_DIR}/dist/"
mv dist/pyprophet "${ORIGINAL_DIR}/dist/pyprophet"

# Return to original directory
cd "${ORIGINAL_DIR}"

# Clean up temporary directories
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