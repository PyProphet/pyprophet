#!/usr/bin/env bash
# filepath: scripts/build/create_linux_packages.sh
# Create DEB and RPM packages for PyProphet

set -euo pipefail

echo "============================================"
echo "Creating Linux Packages (DEB/RPM)"
echo "============================================"

# Get version from git tag or default
if [ -n "${GITHUB_REF_NAME:-}" ]; then
    VERSION="${GITHUB_REF_NAME#v}"
else
    VERSION=$(git describe --tags --abbrev=0 2>/dev/null | sed 's/^v//' || echo "3.0.4")
fi

echo "Package version: ${VERSION}"

# Install fpm if not available
if ! command -v fpm &> /dev/null; then
    echo "Installing fpm..."
    sudo apt-get update
    sudo apt-get install -y ruby ruby-dev rubygems build-essential
    sudo gem install --no-document fpm
fi

# Verify dist directory exists
if [ ! -d "dist/pyprophet" ]; then
    echo "ERROR: dist/pyprophet directory not found. Build the executable first."
    exit 1
fi

# Create package structure
echo "Creating package structure..."
mkdir -p package/usr/local/bin
mkdir -p package/usr/share/pyprophet
mkdir -p package/usr/share/doc/pyprophet

# Copy built executable and libraries
echo "Copying executable from dist/pyprophet..."
cp -r dist/pyprophet/* package/usr/share/pyprophet/

# Ensure the main executable is present
if [ ! -f "package/usr/share/pyprophet/pyprophet" ]; then
    echo "ERROR: pyprophet executable not found in dist/pyprophet/"
    exit 1
fi

# Make executable
chmod +x package/usr/share/pyprophet/pyprophet

# Create wrapper script in /usr/local/bin
cat > package/usr/local/bin/pyprophet << 'EOF'
#!/bin/bash
exec /usr/share/pyprophet/pyprophet "$@"
EOF
chmod +x package/usr/local/bin/pyprophet

# Copy documentation
if [ -f "README.md" ]; then
    cp README.md package/usr/share/doc/pyprophet/
fi
if [ -f "LICENSE" ]; then
    cp LICENSE package/usr/share/doc/pyprophet/
fi

# Create DEB package
echo "Creating DEB package..."
fpm -s dir -t deb \
  -n pyprophet \
  -v "${VERSION}" \
  --description "PyProphet: Semi-supervised learning and scoring of OpenSWATH results" \
  --url "https://github.com/pyprophet/pyprophet" \
  --maintainer "The PyProphet Developers <rocksportrocker@gmail.com>" \
  --license "BSD-3-Clause" \
  --category science \
  --architecture amd64 \
  -C package \
  usr/local/bin/pyprophet \
  usr/share/pyprophet \
  usr/share/doc/pyprophet

DEB_FILE="pyprophet_${VERSION}_amd64.deb"

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
  -C package \
  usr/local/bin/pyprophet \
  usr/share/pyprophet \
  usr/share/doc/pyprophet

RPM_FILE="pyprophet-${VERSION}-1.x86_64.rpm"

# Generate checksums
echo "Generating checksums..."
sha256sum "${DEB_FILE}" > "${DEB_FILE}.sha256"
sha256sum "${RPM_FILE}" > "${RPM_FILE}.sha256"

echo "============================================"
echo "Packages created successfully!"
echo "DEB: ${DEB_FILE}"
echo "RPM: ${RPM_FILE}"
echo "============================================"

ls -lh *.deb *.rpm *.sha256

# Clean up
rm -rf package