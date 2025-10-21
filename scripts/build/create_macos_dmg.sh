#!/bin/bash
# filepath: scripts/build/create_macos_dmg.sh
# Create macOS DMG installer for PyProphet

set -e

echo "============================================"
echo "Creating macOS DMG Installer"
echo "============================================"

# Get version and architecture
if [ -n "${GITHUB_REF_NAME:-}" ]; then
    VERSION="${GITHUB_REF_NAME#v}"
else
    VERSION=$(git describe --tags --abbrev=0 2>/dev/null | sed 's/^v//' || echo "3.0.4")
fi

ARCH=$(uname -m)
echo "Version: ${VERSION}"
echo "Architecture: ${ARCH}"

# Verify dist directory exists
if [ ! -f "dist/pyprophet/pyprophet" ]; then
    echo "ERROR: dist/pyprophet/pyprophet not found. Build the executable first."
    exit 1
fi

# Create app bundle structure
echo "Creating app bundle..."
APP_NAME="PyProphet.app"
rm -rf "${APP_NAME}"
mkdir -p "${APP_NAME}/Contents/MacOS"
mkdir -p "${APP_NAME}/Contents/Resources"

# Copy the entire pyprophet directory
echo "Copying executable from dist/pyprophet..."
cp -r dist/pyprophet/* "${APP_NAME}/Contents/MacOS/"

# Ensure main executable is present and executable
if [ ! -f "${APP_NAME}/Contents/MacOS/pyprophet" ]; then
    echo "ERROR: pyprophet executable not found in dist/pyprophet/"
    exit 1
fi

chmod +x "${APP_NAME}/Contents/MacOS/pyprophet"

# Create launcher script
cat > "${APP_NAME}/Contents/MacOS/pyprophet-launcher" << 'EOF'
#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
exec "${DIR}/pyprophet" "$@"
EOF
chmod +x "${APP_NAME}/Contents/MacOS/pyprophet-launcher"

# Create Info.plist
cat > "${APP_NAME}/Contents/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>pyprophet-launcher</string>
    <key>CFBundleIdentifier</key>
    <string>org.pyprophet.pyprophet</string>
    <key>CFBundleName</key>
    <string>PyProphet</string>
    <key>CFBundleDisplayName</key>
    <string>PyProphet</string>
    <key>CFBundleVersion</key>
    <string>${VERSION}</string>
    <key>CFBundleShortVersionString</key>
    <string>${VERSION}</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>LSMinimumSystemVersion</key>
    <string>11.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
EOF

# Create DMG
DMG_NAME="pyprophet-${VERSION}-macos-${ARCH}.dmg"
echo "Creating DMG: ${DMG_NAME}"

# Create temporary directory for DMG contents
DMG_TEMP="dmg-temp"
rm -rf "${DMG_TEMP}"
mkdir -p "${DMG_TEMP}"

# Copy app to temp directory
cp -r "${APP_NAME}" "${DMG_TEMP}/"

# Create symbolic link to Applications
ln -s /Applications "${DMG_TEMP}/Applications"

# Copy README if exists
if [ -f "README.md" ]; then
    cp README.md "${DMG_TEMP}/README.txt"
fi

# Create DMG
hdiutil create -volname "PyProphet ${VERSION}" \
  -srcfolder "${DMG_TEMP}" \
  -ov -format UDZO \
  "${DMG_NAME}"

# Generate checksum
shasum -a 256 "${DMG_NAME}" > "${DMG_NAME}.sha256"

echo "============================================"
echo "DMG created successfully!"
echo "File: ${DMG_NAME}"
echo "============================================"

ls -lh "${DMG_NAME}" "${DMG_NAME}.sha256"

# Clean up
rm -rf "${APP_NAME}" "${DMG_TEMP}"

echo ""
echo "To test the DMG:"
echo "1. Mount: hdiutil attach ${DMG_NAME}"
echo "2. Copy app to /Applications"
echo "3. Run: /Applications/PyProphet.app/Contents/MacOS/pyprophet --help"