# Release Process

## Prerequisites
- Ensure all tests pass
- Update CHANGELOG.md
- Update version in pyproject.toml

## Creating a Release

1. **Tag the release**
   ```bash
   git tag -a v3.0.3 -m "Release v3.0.3"
   git push origin v3.0.3
   ```

2. **GitHub Actions will automatically:**
   - Build executables for Linux, Windows, macOS (Intel + ARM)
   - Create a GitHub release
   - Upload all build artifacts

3. **Manual verification (after workflow completes):**
   - Check the GitHub Release page
   - Download and test each platform's executable
   - Verify release notes are accurate

## Building locally

### Linux
```bash
bash scripts/build/build_linux.sh
```

### Windows
```bat
scripts\build\build_windows.bat
```

### macOS
```bash
bash scripts/build/build_macos.sh
```

## Distribution
- Executables are available as GitHub Release assets
- PyPI package: `pip install pyprophet`

## Troubleshooting
- If builds fail, check GitHub Actions logs
- Test locally on the target platform first
- Ensure all dependencies are in pyproject.toml
