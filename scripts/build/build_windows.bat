@echo off
REM Build script for PyProphet Windows executable using PyInstaller

echo ============================================
echo Building PyProphet for Windows
echo ============================================

REM Clean previous builds
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

REM Install/upgrade build dependencies
python -m pip install --upgrade pip setuptools wheel cython numpy pyinstaller

REM Install the package in editable mode to ensure it's importable
python -m pip install -e .

REM Build Cython extensions in-place
python setup.py build_ext --inplace

REM Run PyInstaller with metadata collection
python -m PyInstaller ^
    --name pyprophet ^
    --onedir ^
    --console ^
    --clean ^
    --noconfirm ^
    --log-level INFO ^
    --additional-hooks-dir=packaging/pyinstaller/hooks ^
    --hidden-import=pyprophet ^
    --hidden-import=pyprophet.main ^
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
    packaging/pyinstaller/run_pyprophet.py

if errorlevel 1 (
    echo Build failed!
    exit /b 1
)

echo ============================================
echo Build complete! Executable at: dist\pyprophet\pyprophet.exe
echo ============================================

REM Create ZIP archive (more native for Windows)
powershell -Command "Compress-Archive -Path dist\pyprophet -DestinationPath pyprophet-windows-x86_64.zip -Force"

if errorlevel 1 (
    echo Archive creation failed!
    exit /b 1
)

echo Archive created: pyprophet-windows-x86_64.zip

REM Optional: Generate SHA256 checksum
powershell -Command "(Get-FileHash pyprophet-windows-x86_64.zip -Algorithm SHA256).Hash | Out-File -FilePath pyprophet-windows-x86_64.zip.sha256 -Encoding ASCII"
echo Checksum created: pyprophet-windows-x86_64.zip.sha256