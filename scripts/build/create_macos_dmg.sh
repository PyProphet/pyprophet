#!/bin/bash
# filepath: scripts/build/create_macos_dmg.sh
# Create macOS DMG installer for PyProphet

set -e

echo "============================================"
echo "Creating macOS DMG Installer"
echo "============================================"

# Get version (prefer GITHUB_REF_NAME, then pyproject.toml, then git tag, then default)
if [ -n "${GITHUB_REF_NAME:-}" ]; then
    VERSION="${GITHUB_REF_NAME#v}"
else
    VERSION=$(
        python3 - <<'PY'
import sys, subprocess
try:
    import tomllib
    with open("pyproject.toml","rb") as f:
        cfg = tomllib.load(f)
        v = cfg.get("project",{}).get("version")
        if v:
            print(v); sys.exit(0)
except Exception:
    pass
try:
    tag = subprocess.check_output(["git","describe","--tags","--abbrev=0"], stderr=subprocess.DEVNULL).decode().strip()
    print(tag.lstrip("v")); sys.exit(0)
except Exception:
    print("3.0.4")
PY
    )
fi

# sanitize VERSION for filenames
VERSION_SAFE="${VERSION//\//-}"
VERSION_SAFE="${VERSION_SAFE// /-}"
VERSION_SAFE="$(echo "$VERSION_SAFE" | tr -cd 'A-Za-z0-9._-')"

ARCH=$(uname -m)
echo "Version: ${VERSION} (safe: ${VERSION_SAFE})"
echo "Architecture: ${ARCH}"

# Verify single-file executable exists
if [ ! -f "dist/pyprophet" ]; then
    echo "ERROR: dist/pyprophet not found. Build the executable first."
    exit 1
fi

echo "Found single-file executable: dist/pyprophet"
ls -lh dist/pyprophet

# Create a simple directory structure for DMG
echo "Preparing DMG contents..."
DMG_TEMP="dmg-temp"
rm -rf "${DMG_TEMP}"
mkdir -p "${DMG_TEMP}/PyProphet"

# Copy the single-file executable
echo "Copying PyProphet executable..."
cp dist/pyprophet "${DMG_TEMP}/PyProphet/pyprophet"
chmod +x "${DMG_TEMP}/PyProphet/pyprophet"

# Create a wrapper script for convenience
cat > "${DMG_TEMP}/PyProphet/pyprophet.sh" << 'EOF'
#!/bin/bash
# PyProphet launcher script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
exec "${DIR}/pyprophet" "$@"
EOF
chmod +x "${DMG_TEMP}/PyProphet/pyprophet.sh"

# Create a README for installation instructions
cat > "${DMG_TEMP}/README.txt" << EOF
PyProphet ${VERSION} for macOS (${ARCH})
==========================================

Installation:
1. Copy the PyProphet folder to your Applications folder or any location
2. Add to your PATH by adding this line to ~/.zshrc or ~/.bash_profile:
   export PATH="/Applications/PyProphet:\$PATH"
3. Restart your terminal or run: source ~/.zshrc

Usage:
- Run from Applications: /Applications/PyProphet/pyprophet --help
- Or if added to PATH: pyprophet --help

Note: On first run, macOS may show a security warning.
Right-click the pyprophet file and select "Open" to allow it.

For more information:
https://github.com/pyprophet/pyprophet
EOF

# Create symbolic link to Applications for easy drag-and-drop install
ln -s /Applications "${DMG_TEMP}/Applications"

# Copy documentation
if [ -f "README.md" ]; then
    cp README.md "${DMG_TEMP}/PyProphet/README.md"
fi
if [ -f "LICENSE" ]; then
    cp LICENSE "${DMG_TEMP}/PyProphet/LICENSE"
fi

# Create DMG
DMG_NAME="pyprophet-${VERSION_SAFE}-macos-${ARCH}.dmg"
echo "Creating DMG: ${DMG_NAME}"

# Remove any existing DMG
rm -f "${DMG_NAME}"

# Create DMG with proper settings
hdiutil create -volname "PyProphet ${VERSION}" \
  -srcfolder "${DMG_TEMP}" \
  -ov -format UDZO \
  -imagekey zlib-level=9 \
  "${DMG_NAME}"

# Generate checksum
shasum -a 256 "${DMG_NAME}" > "${DMG_NAME}.sha256"

echo "============================================"
echo "DMG created successfully!"
echo "File: ${DMG_NAME}"
echo "============================================"

ls -lh "${DMG_NAME}" "${DMG_NAME}.sha256"

# Clean up
rm -rf "${DMG_TEMP}"

echo ""
echo "To test the DMG:"
echo "1. Mount: hdiutil attach ${DMG_NAME}"
echo "2. Run: /Volumes/PyProphet*/PyProphet/pyprophet --help"
echo "3. Unmount: hdiutil detach /Volumes/PyProphet*"