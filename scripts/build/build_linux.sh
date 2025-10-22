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

# Install system tools for stripping
echo "Installing build tools..."
sudo apt-get update -qq
sudo apt-get install -y binutils

# Save the original directory
ORIGINAL_DIR="$(pwd)"

# Clean ALL build artifacts
echo "Cleaning build artifacts..."
rm -rf build dist pyprophet.spec *.egg-info .eggs
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true

# Create a clean virtual environment
echo "Creating clean virtual environment..."
VENV_DIR=$(mktemp -d)/build_venv
$PYTHON -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Upgrade pip and install build tools
pip install --upgrade pip setuptools wheel build

# Build wheel
echo "Building pyprophet wheel..."
python -m build --wheel --outdir /tmp/pyprophet_wheels

# Install dependencies
echo "Installing dependencies..."
pip install cython numpy pyinstaller

# Parse and install runtime dependencies
python << 'PYEOF'
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
        except subprocess.CalledProcessError:
            pass
PYEOF

# Install pyprophet from wheel (NOT editable)
echo "Installing pyprophet from wheel..."
pip install --force-reinstall --no-deps /tmp/pyprophet_wheels/pyprophet-*.whl

# Verify installation
echo "Verifying installation..."
python -c "import pyprophet; print(f'PyProphet: {pyprophet.__file__}')"
python -c "import numpy; print(f'NumPy: {numpy.__file__}')"
python -c "import pandas; print('✓ All imports successful')"

# Get site-packages
SITE_PACKAGES=$(python -c "import pyprophet, os; print(os.path.dirname(pyprophet.__file__))")
echo "Site-packages: ${SITE_PACKAGES}"

# Collect binaries
ADD_BINARY_ARGS=()
for so in "${SITE_PACKAGES}"/pyprophet/scoring/_optimized*.so; do
  if [ -f "$so" ]; then
    ADD_BINARY_ARGS+=(--add-binary "$so:pyprophet/scoring")
    echo "Found binary: $so"
  fi
done

# Create build directory
mkdir -p "${ORIGINAL_DIR}/dist"
BUILD_DIR=$(mktemp -d)
cd "${BUILD_DIR}"

# Copy PyInstaller files
cp "${ORIGINAL_DIR}/packaging/pyinstaller/run_pyprophet.py" .
mkdir -p hooks
cp "${ORIGINAL_DIR}/packaging/pyinstaller/hooks"/*.py hooks/ 2>/dev/null || true

# Run PyInstaller
echo "Running PyInstaller..."
python -m PyInstaller \
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

# Move executable
mv dist/pyprophet "${ORIGINAL_DIR}/dist/pyprophet"
cd "${ORIGINAL_DIR}"

# Cleanup
deactivate
rm -rf "$(dirname "$VENV_DIR")" "${BUILD_DIR}" /tmp/pyprophet_wheels

echo "============================================"
echo "Build complete! Single executable at: dist/pyprophet"
ls -lh dist/pyprophet
echo "============================================"

# Create archive
cd dist
ARCH=$(uname -m)
ARCHIVE_NAME="pyprophet-linux-${ARCH}.tar.gz"
tar -czf "../${ARCHIVE_NAME}" pyprophet
cd ..

echo "Archive created: ${ARCHIVE_NAME}"

# Generate checksum
if command -v sha256sum &> /dev/null; then
    sha256sum "${ARCHIVE_NAME}" > "${ARCHIVE_NAME}.sha256"
    cat "${ARCHIVE_NAME}.sha256"
fi

echo ""
echo "Test with: ./dist/pyprophet --help"