# Create Windows installer using Inno Setup

$ErrorActionPreference = "Stop"

Write-Host "============================================"
Write-Host "Creating Windows Installer"
Write-Host "============================================"

# Get version from environment or git tag
if ($env:GITHUB_REF_NAME) {
    $VERSION = $env:GITHUB_REF_NAME -replace '^v', ''
} else {
    $VERSION = (git describe --tags --abbrev=0 2>$null) -replace '^v', ''
    if (-not $VERSION) { $VERSION = "3.0.4" }
}

Write-Host "Installer version: $VERSION"

# Verify single-file executable exists
if (-not (Test-Path "dist\pyprophet.exe")) {
    Write-Error "ERROR: dist\pyprophet.exe not found. Build the executable first."
    exit 1
}

Write-Host "Found single-file executable: dist\pyprophet.exe"
Get-Item "dist\pyprophet.exe" | Format-Table Name, Length

# Install Inno Setup if not available
if (-not (Test-Path "C:\Program Files (x86)\Inno Setup 6\ISCC.exe")) {
    Write-Host "Installing Inno Setup..."
    choco install innosetup -y
}

# Create Inno Setup script
$issContent = @"
; PyProphet Inno Setup Script
[Setup]
AppName=PyProphet
AppVersion=$VERSION
AppPublisher=The PyProphet Developers
AppPublisherURL=https://github.com/pyprophet/pyprophet
AppSupportURL=https://github.com/pyprophet/pyprophet/issues
AppUpdatesURL=https://github.com/pyprophet/pyprophet/releases
DefaultDirName={autopf}\PyProphet
DefaultGroupName=PyProphet
AllowNoIcons=yes
LicenseFile=LICENSE
OutputDir=.
OutputBaseFilename=pyprophet-setup-x86_64
Compression=lzma
SolidCompression=yes
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
WizardStyle=modern
UninstallDisplayIcon={app}\pyprophet.exe

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "addtopath"; Description: "Add to PATH environment variable"; GroupDescription: "System integration:"; Flags: unchecked

[Files]
Source: "dist\pyprophet.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "README.md"; DestDir: "{app}"; Flags: ignoreversion; Check: FileExists('README.md')
Source: "LICENSE"; DestDir: "{app}"; Flags: ignoreversion; Check: FileExists('LICENSE')

[Icons]
Name: "{group}\PyProphet"; Filename: "{app}\pyprophet.exe"
Name: "{group}\{cm:UninstallProgram,PyProphet}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\PyProphet"; Filename: "{app}\pyprophet.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\pyprophet.exe"; Parameters: "--version"; Description: "Verify installation"; Flags: postinstall skipifsilent nowait

[Registry]
Root: HKLM; Subkey: "SYSTEM\CurrentControlSet\Control\Session Manager\Environment"; ValueType: expandsz; ValueName: "Path"; ValueData: "{olddata};{app}"; Tasks: addtopath; Check: NeedsAddPath('{app}')

[Code]
function FileExists(FileName: string): Boolean;
begin
  Result := FileOrDirExists(FileName);
end;

function NeedsAddPath(Param: string): boolean;
var
  OrigPath: string;
begin
  if not RegQueryStringValue(HKEY_LOCAL_MACHINE,
    'SYSTEM\CurrentControlSet\Control\Session Manager\Environment',
    'Path', OrigPath)
  then begin
    Result := True;
    exit;
  end;
  Result := Pos(';' + Param + ';', ';' + OrigPath + ';') = 0;
end;
"@

# Write Inno Setup script
$issContent | Out-File -FilePath "pyprophet.iss" -Encoding UTF8

Write-Host "Building installer..."
& "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" "pyprophet.iss"

if ($LASTEXITCODE -ne 0) {
    Write-Error "Installer build failed!"
    exit 1
}

# Generate checksum
$installerFile = "pyprophet-setup-x86_64.exe"
if (Test-Path $installerFile) {
    $hash = (Get-FileHash $installerFile -Algorithm SHA256).Hash
    "$hash  $installerFile" | Out-File -FilePath "$installerFile.sha256" -Encoding ASCII
    
    Write-Host "============================================"
    Write-Host "Installer created successfully!"
    Write-Host "File: $installerFile"
    Write-Host "============================================"
    
    Get-ChildItem $installerFile, "$installerFile.sha256" | Format-Table Name, Length
} else {
    Write-Error "Installer file not found: $installerFile"
    exit 1
}

# Clean up
Remove-Item "pyprophet.iss" -ErrorAction SilentlyContinue