#!/usr/bin/env bash
# filepath: scripts/build/create_linux_packages.sh
# Create DEB and RPM packages for PyProphet

set -euo pipefail

echo "============================================"
echo "Creating Linux Packages (DEB/RPM)"
echo "============================================"

# Get version from git tag or default
# DEB packages require versions to start with a digit
if [ -n "${GITHUB_REF_NAME:-}" ]; then
    # If it's a tag (starts with 'v'), strip the 'v'
    if [[ "${GITHUB_REF_NAME}" =~ ^v[0-9] ]]; then
        VERSION="${GITHUB_REF_NAME#v}"
    else
        # For branches or non-version tags, use a default version
        VERSION="0.0.0+${GITHUB_REF_NAME//\//.}"
    fi
else
    # Try to get the latest tag
    VERSION=$(git describe --tags --abbrev=0 2>/dev/null | sed 's/^v//' || echo "")
    
    # If no tags exist, use development version with commit hash
    if [ -z "$VERSION" ] || [[ ! "$VERSION" =~ ^[0-9] ]]; then
        COMMIT_HASH=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
        VERSION="0.0.0+dev.${COMMIT_HASH}"
    fi
fi

echo "Package version: ${VERSION}"

# Validate version starts with a digit
if [[ ! "$VERSION" =~ ^[0-9] ]]; then
    echo "ERROR: Version '$VERSION' does not start with a digit (required for DEB packages)"
    echo "Using fallback version: 0.0.0+local"
    VERSION="0.0.0+local"
fi

# Install fpm if not available
if ! command -v fpm &> /dev/null; then
    echo "Installing fpm..."
    sudo apt-get update
    sudo apt-get install -y ruby ruby-dev rubygems build-essential
    sudo gem install --no-document fpm
fi

# Verify single-file executable exists
if [ ! -f "dist/pyprophet" ]; then
    echo "ERROR: dist/pyprophet executable not found. Build the executable first."
    exit 1
fi

echo "Found single-file executable: dist/pyprophet"
ls -lh dist/pyprophet

# Clean any previous package directory
rm -rf package

# Create package structure
echo "Creating package structure..."
mkdir -p package/usr/local/bin
mkdir -p package/usr/share/doc/pyprophet

# Copy single-file executable
echo "Copying pyprophet executable..."
cp dist/pyprophet package/usr/local/bin/pyprophet
chmod +x package/usr/local/bin/pyprophet

# Copy documentation
if [ -f "README.md" ]; then
    cp README.md package/usr/share/doc/pyprophet/
fi
if [ -f "LICENSE" ]; then
    cp LICENSE package/usr/share/doc/pyprophet/copyright
fi

# Create a changelog
cat > package/usr/share/doc/pyprophet/changelog << EOF
pyprophet (${VERSION}) stable; urgency=medium

  * Release ${VERSION}
  * See https://github.com/pyprophet/pyprophet/releases

 -- The PyProphet Developers <rocksportrocker@gmail.com>  $(date -R)
EOF
gzip -9 package/usr/share/doc/pyprophet/changelog

# List what will be packaged
echo "Package contents:"
find package -type f

# Count total files
TOTAL_FILES=$(find package -type f 2>/dev/null | wc -l)
echo "Total files to package: ${TOTAL_FILES}"

# Determine system dependencies
# Single-file PyInstaller executable is mostly self-contained
DEPS="libc6 (>= 2.34)"

# Create DEB package
echo "Creating DEB package..."
fpm -s dir -t deb \
  -n pyprophet \
  -v "${VERSION}" \
  --description "PyProphet: Semi-supervised learning and scoring of OpenSWATH results
 PyProphet is a Python package for semi-supervised learning and scoring
 of OpenSWATH results. It provides tools for peptide-level, protein-level,
 and IPF (Integrated Peptidoform) scoring." \
  --url "https://github.com/pyprophet/pyprophet" \
  --maintainer "The PyProphet Developers <rocksportrocker@gmail.com>" \
  --license "BSD-3-Clause" \
  --category science \
  --architecture amd64 \
  --depends "${DEPS}" \
  --deb-priority optional \
  --deb-compression xz \
  -C package \
  .

DEB_FILE=$(ls pyprophet_*.deb)

# Create RPM package
echo "Creating RPM package..."
fpm -s dir -t rpm \
  -n pyprophet \
  -v "${VERSION}" \
  --description "PyProphet: Semi-supervised learning and scoring of OpenSWATH results" \
  --url "https://github.com/pyprophet/pyprophet" \
  --maintainer "The PyProphet Developers <rocksportrocker@gmail.com>" \
  --license "BSD-3-Clause" \
  --category science \
  --architecture x86_64 \
  --rpm-compression xz \
  --depends "glibc >= 2.34" \
  -C package \
  .

RPM_FILE=$(ls pyprophet-*.rpm)

# Verify DEB package
echo "Verifying DEB package..."
echo "Package info:"
dpkg --info "${DEB_FILE}"
echo ""
echo "Package contents:"
dpkg --contents "${DEB_FILE}"

# Generate checksums
echo "Generating checksums..."
sha256sum "${DEB_FILE}" > "${DEB_FILE}.sha256"
sha256sum "${RPM_FILE}" > "${RPM_FILE}.sha256"

echo "============================================"
echo "Packages created successfully!"
echo "DEB: ${DEB_FILE}"
echo "RPM: ${RPM_FILE}"
echo "Version: ${VERSION}"
echo "============================================"

ls -lh *.deb *.rpm *.sha256

# Clean up
rm -rf package

echo ""
echo "To test the DEB package:"
echo "  sudo dpkg -i ${DEB_FILE}"
echo "  pyprophet --help"
echo ""
echo "To remove:"
echo "  sudo dpkg -r pyprophet"