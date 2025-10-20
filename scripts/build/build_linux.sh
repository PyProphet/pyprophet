#!/usr/bin/env bash
# Build script for PyProphet Linux executable using PyInstaller

set -euo pipefail

PYTHON=${PYTHON:-python3}

echo "============================================"
echo "Building PyProphet for Linux"
echo "============================================"

# Install/upgrade build dependencies
$PYTHON -m pip install --upgrade pip setuptools wheel cython numpy
$PYTHON -m pip install -r requirements.txt pyinstaller

# Build C extensions in-place
$PYTHON setup.py build_ext --inplace

# Collect compiled extension binaries for pyprophet/scoring
ADD_BINARY_ARGS=()
for so in pyprophet/scoring/_optimized*.so pyprophet/scoring/_optimized*.cpython-*.so; do
  if [ -f "$so" ]; then
    ADD_BINARY_ARGS+=(--add-binary "$so:pyprophet/scoring")
    echo "Adding binary: $so"
  fi
done

# Locate xgboost native library and add it to the bundle if present
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
else
  echo "Warning: xgboost native lib not found; hook should still work if packaging/pyinstaller-hooks/hook-xgboost.py is present."
fi

# Clean previous builds
rm -rf build dist
mkdir -p build-dist

# Run PyInstaller
$PYTHON -m PyInstaller \
  --name pyprophet \
  --onedir \
  --console \
  --clean \
  --noconfirm \
  --log-level INFO \
  --additional-hooks-dir packaging/pyinstaller/hooks \
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

# Copy to build-dist for local use
cp -r dist/pyprophet build-dist/

echo "============================================"
echo "Build complete! Executable at: build-dist/pyprophet/pyprophet"
echo "============================================"

# Create compressed archive for distribution
echo "Creating distribution archive..."
cd dist
ARCH=$(uname -m)
ARCHIVE_NAME="pyprophet-linux-${ARCH}.tar.gz"
tar -czf "../${ARCHIVE_NAME}" pyprophet/
cd ..

echo "============================================"
echo "Archive created: ${ARCHIVE_NAME}"
echo "============================================"

# Optional: Generate SHA256 checksum
if command -v sha256sum &> /dev/null; then
    sha256sum "${ARCHIVE_NAME}" > "${ARCHIVE_NAME}.sha256"
    echo "Checksum: ${ARCHIVE_NAME}.sha256"
    cat "${ARCHIVE_NAME}.sha256"
fi

echo ""
echo "To test locally:"
echo "  cd build-dist/pyprophet && ./pyprophet --help"
echo ""
echo "To distribute:"
echo "  Upload ${ARCHIVE_NAME} to GitHub releases"
