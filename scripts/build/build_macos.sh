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
# Extract and install only [project.dependencies] from pyproject.toml
python3 -c "
import tomllib
with open('pyproject.toml', 'rb') as f:
    deps = tomllib.load(f)['project']['dependencies']
print(' '.join(deps))
" | xargs python3 -m pip install --no-cache-dir

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

# Create PyInstaller spec with dev package excludes
cat > pyprophet.spec << 'SPEC_EOF'
# -*- mode: python ; coding: utf-8 -*-
block_cipher = None

a = Analysis(
    ['packaging/pyinstaller/run_pyprophet.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=['pyprophet', 'pyprophet.main'],
    hookspath=['packaging/pyinstaller/hooks'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude dev/docs/testing packages
        'sphinx', 'sphinx_rtd_theme', 'pydata_sphinx_theme', 'sphinx_copybutton',
        'sphinx.ext', 'alabaster', 'babel', 'docutils',
        'mypy', 'pytest', 'pytest-regtest', 'pytest-xdist',
        'black', 'ruff',
        'tomli',  # build-time only
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Remove pyopenms data files (not needed at runtime)
a.datas = [
    d for d in a.datas
    if not any(x in d[0] for x in ['pyopenms/share/OpenMS/CHEMISTRY/', 'pyopenms/share/OpenMS/CV/'])
]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='pyprophet',
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,  # strip symbols from exe
    upx=True,    # compress exe with UPX
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=True,  # strip symbols from all binaries
    upx=True,    # compress all binaries with UPX
    upx_exclude=[],
    name='pyprophet',
)
SPEC_EOF

# Run PyInstaller with the spec
echo "Running PyInstaller..."
python3 -m PyInstaller \
  --clean \
  --noconfirm \
  --log-level INFO \
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
  pyprophet.spec

# Post-build: aggressive strip + UPX on all .dylib/.so files
echo "Post-processing: stripping and compressing native libraries..."
find dist/pyprophet/_internal -type f \( -name '*.dylib' -o -name '*.so' \) -print0 | \
  xargs -0 -n1 -P$(sysctl -n hw.ncpu) strip -x 2>/dev/null || true

if command -v upx &> /dev/null; then
    find dist/pyprophet/_internal -type f \( -name '*.dylib' -o -name '*.so' \) -print0 | \
      xargs -0 -n1 -P$(sysctl -n hw.ncpu) upx --best --lzma 2>/dev/null || true
fi

# Remove collected dev packages if any slipped through
echo "Removing any dev/doc packages..."
rm -rf dist/pyprophet/_internal/{sphinx,babel,alabaster,docutils,mypy,pytest,black,ruff,tomli} 2>/dev/null || true

echo "============================================"
echo "Build complete! Executable at: dist/pyprophet/pyprophet"
du -sh dist/pyprophet/_internal
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

# Generate SHA256 checksum
if command -v shasum &> /dev/null; then
    shasum -a 256 "${ARCHIVE_NAME}" > "${ARCHIVE_NAME}.sha256"
    echo "Checksum: ${ARCHIVE_NAME}.sha256"
    cat "${ARCHIVE_NAME}.sha256"
fi

echo ""
echo "To test locally:"
echo "  cd dist/pyprophet && ./pyprophet --help"