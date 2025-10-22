#!/usr/bin/env bash
# Build script for PyProphet Linux executable using PyInstaller

set -euo pipefail

PYTHON=${PYTHON:-python3}

echo "============================================"
echo "Building PyProphet for Linux"
echo "============================================"

# Install system tools for stripping and compression
echo "Installing build tools (strip, upx)..."
sudo apt-get update -qq
sudo apt-get install -y binutils upx

# Install/upgrade build dependencies
$PYTHON -m pip install --upgrade pip setuptools wheel cython numpy pyinstaller

# Install ONLY runtime dependencies (no dev/docs/testing extras)
echo "Installing runtime dependencies only..."
$PYTHON -m pip install -e . --no-deps

# Parse and install runtime dependencies from pyproject.toml
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
rm -rf build dist pyprophet.spec
mkdir -p build-dist

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
$PYTHON -m PyInstaller \
  --clean \
  --noconfirm \
  --onefile \
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

# Post-build: aggressive strip + UPX on all .so files
echo "Post-processing: stripping and compressing native libraries..."
find dist/pyprophet/_internal -type f \( -name '*.so*' -o -name '*.so.*' \) -print0 | \
  xargs -0 -n1 -P$(nproc) strip --strip-unneeded 2>/dev/null || true

find dist/pyprophet/_internal -type f \( -name '*.so*' -o -name '*.so.*' \) -print0 | \
  xargs -0 -n1 -P$(nproc) upx --best --lzma 2>/dev/null || true

# Remove collected dev packages if any slipped through
echo "Removing any dev/doc packages..."
rm -rf dist/pyprophet/_internal/{sphinx,babel,alabaster,docutils,mypy,pytest,black,ruff,tomli} 2>/dev/null || true

# Copy to build-dist for local use
cp -r dist/pyprophet build-dist/

echo "============================================"
echo "Build complete! Executable at: build-dist/pyprophet/pyprophet"
du -sh build-dist/pyprophet/_internal
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

# Generate SHA256 checksum
if command -v sha256sum &> /dev/null; then
    sha256sum "${ARCHIVE_NAME}" > "${ARCHIVE_NAME}.sha256"
    echo "Checksum: ${ARCHIVE_NAME}.sha256"
    cat "${ARCHIVE_NAME}.sha256"
fi

echo ""
echo "To test locally:"
echo "  cd build-dist/pyprophet && ./pyprophet --help"