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

# Run PyInstaller with metadata collection and explicit library collection
python3 -m PyInstaller \
  --name pyprophet \
  --onedir \
  --console \
  --clean \
  --noconfirm \
  --log-level INFO \
  --additional-hooks-dir packaging/pyinstaller-hooks \
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

echo "============================================"
echo "Build complete! Executable at: dist/pyprophet/pyprophet"
echo "============================================"

# Make executable
chmod +x dist/pyprophet/pyprophet

# Create tarball for distribution
cd dist
ARCH=$(uname -m)
ARCHIVE_NAME="pyprophet-macos-${ARCH}.tar.gz"
tar -czf "../${ARCHIVE_NAME}" pyprophet/
cd ..

echo "============================================"
echo "Archive created: ${ARCHIVE_NAME}"
echo "============================================"

# Optional: Generate SHA256 checksum
if command -v shasum &> /dev/null; then
    shasum -a 256 "${ARCHIVE_NAME}" > "${ARCHIVE_NAME}.sha256"
    echo "Checksum: ${ARCHIVE_NAME}.sha256"
    cat "${ARCHIVE_NAME}.sha256"
fi

echo ""
echo "To test locally:"
echo "  cd dist/pyprophet && ./pyprophet --help"
echo ""
echo "To distribute:"
echo "  Upload ${ARCHIVE_NAME} to GitHub releases"