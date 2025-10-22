@echo off
REM Build script for PyProphet Windows executable using PyInstaller

echo ============================================
echo Building PyProphet for Windows
echo ============================================

REM Install UPX for compression
echo Installing UPX...
choco install upx -y >nul 2>&1 || echo UPX install skipped (may already be installed)

REM Clean previous builds
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist pyprophet.spec del /q pyprophet.spec

REM Install/upgrade build dependencies
python -m pip install --upgrade pip setuptools wheel cython numpy pyinstaller

REM Install ONLY runtime dependencies (no dev/docs/testing extras)
echo Installing runtime dependencies only...
python -m pip install -e . --no-deps

REM Parse and install runtime dependencies from pyproject.toml
python -c "import tomllib, subprocess, sys; config = tomllib.load(open('pyproject.toml', 'rb')); deps = config['project']['dependencies']; [subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', dep.strip()]) for dep in deps if dep.strip() and not dep.strip().startswith('#')]"

REM Build Cython extensions in-place
python setup.py build_ext --inplace

REM Create PyInstaller spec with dev package excludes
echo Creating PyInstaller spec...
(
echo # -*- mode: python ; coding: utf-8 -*-
echo block_cipher = None
echo.
echo a = Analysis^(
echo     ['packaging/pyinstaller/run_pyprophet.py'],
echo     pathex=[],
echo     binaries=[],
echo     datas=[],
echo     hiddenimports=['pyprophet', 'pyprophet.main'],
echo     hookspath=['packaging/pyinstaller/hooks'],
echo     hooksconfig={},
echo     runtime_hooks=[],
echo     excludes=[
echo         # Exclude dev/docs/testing packages
echo         'sphinx', 'sphinx_rtd_theme', 'pydata_sphinx_theme', 'sphinx_copybutton',
echo         'sphinx.ext', 'alabaster', 'babel', 'docutils',
echo         'mypy', 'pytest', 'pytest-regtest', 'pytest-xdist',
echo         'black', 'ruff',
echo         'tomli',  # build-time only
echo     ],
echo     win_no_prefer_redirects=False,
echo     win_private_assemblies=False,
echo     cipher=block_cipher,
echo     noarchive=False,
echo ^)
echo.
echo # Remove pyopenms data files ^(not needed at runtime^)
echo a.datas = [
echo     d for d in a.datas
echo     if not any^(x in d[0] for x in ['pyopenms/share/OpenMS/CHEMISTRY/', 'pyopenms/share/OpenMS/CV/']^)
echo ]
echo.
echo pyz = PYZ^(a.pure, a.zipped_data, cipher=block_cipher^)
echo.
echo exe = EXE^(
echo     pyz,
echo     a.scripts,
echo     [],
echo     exclude_binaries=True,
echo     name='pyprophet',
echo     debug=False,
echo     bootloader_ignore_signals=False,
echo     strip=False,
echo     upx=True,    # compress exe with UPX
echo     console=True,
echo     disable_windowed_traceback=False,
echo     argv_emulation=False,
echo     target_arch=None,
echo     codesign_identity=None,
echo     entitlements_file=None,
echo ^)
echo.
echo coll = COLLECT^(
echo     exe,
echo     a.binaries,
echo     a.zipfiles,
echo     a.datas,
echo     strip=False,
echo     upx=True,    # compress all binaries with UPX
echo     upx_exclude=[],
echo     name='pyprophet',
echo ^)
) > pyprophet.spec

REM Run PyInstaller with the spec
echo Running PyInstaller...
python -m PyInstaller ^
    --clean ^
    --noconfirm ^
    --onefile ^
    --log-level INFO ^
    --collect-submodules pyprophet ^
    --collect-all numpy ^
    --collect-all pandas ^
    --collect-all scipy ^
    --collect-all sklearn ^
    --collect-all pyopenms ^
    --copy-metadata duckdb ^
    --copy-metadata duckdb-extensions ^
    --copy-metadata duckdb-extension-sqlite-scanner ^
    --copy-metadata pyopenms ^
    pyprophet.spec

if errorlevel 1 (
    echo Build failed!
    exit /b 1
)

REM Post-build: remove dev packages if any slipped through
echo Removing any dev/doc packages...
for %%d in (sphinx babel alabaster docutils mypy pytest black ruff tomli) do (
    if exist dist\pyprophet\_internal\%%d rmdir /s /q dist\pyprophet\_internal\%%d 2>nul
)

echo ============================================
echo Build complete! Executable at: dist\pyprophet\pyprophet.exe
dir dist\pyprophet\_internal | find "bytes"
echo ============================================

REM Create ZIP archive
echo Creating ZIP archive...
powershell -Command "Compress-Archive -Path dist\pyprophet -DestinationPath pyprophet-windows-x86_64.zip -Force"

if errorlevel 1 (
    echo Archive creation failed!
    exit /b 1
)

echo Archive created: pyprophet-windows-x86_64.zip

REM Generate SHA256 checksum
powershell -Command "(Get-FileHash pyprophet-windows-x86_64.zip -Algorithm SHA256).Hash | Out-File -FilePath pyprophet-windows-x86_64.zip.sha256 -Encoding ASCII"
echo Checksum created: pyprophet-windows-x86_64.zip.sha256

echo.
echo To test locally:
echo   dist\pyprophet\pyprophet.exe --help