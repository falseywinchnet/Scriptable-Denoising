param(
    [Parameter(Mandatory=$true)][string]$SdrSharpDirectory,
    [Parameter(Mandatory=$true)][string]$PythonDirectory
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
Set-Location $repo
& "$PythonDirectory/python.exe" -m pip install -r requirements.txt
if ($LASTEXITCODE) { throw 'Python dependencies failed' }
if (Test-Path plugin/SDRSharp.Cleanup.csproj) {
    & "$PythonDirectory/python.exe" tools/extract_sdrsharp.py "$SdrSharpDirectory/SDRSharp.exe" build/sdrsharp-sdk
    if ($LASTEXITCODE) { throw 'SDR# reference extraction failed' }
}
cmake -S . -B build/windows-msvc -A x64 "-DPython3_ROOT_DIR=$PythonDirectory"
if ($LASTEXITCODE) { throw 'CMake configure failed' }
cmake --build build/windows-msvc --config Release --target Cleanup CleanupNative cleanupctl
if ($LASTEXITCODE) { throw 'Native build failed' }
if (Test-Path plugin/SDRSharp.Cleanup.csproj) {
    dotnet build plugin/SDRSharp.Cleanup.csproj -c Release
    if ($LASTEXITCODE) { throw 'Managed build failed' }
} elseif (!(Test-Path plugin/precompiled/windows-x64/SDRSharp.Cleanup.dll)) {
    throw 'Missing precompiled SDR# wrapper'
}
cmake -S viewer -B build/viewer-msvc -A x64
if ($LASTEXITCODE) { throw 'Viewer configure failed' }
cmake --build build/viewer-msvc --config Release --target CleanupViewer
if ($LASTEXITCODE) { throw 'Viewer build failed' }
Copy-Item build/viewer-msvc/VIEWER-LICENSES.txt build/viewer-msvc/Release/VIEWER-LICENSES.txt
& "$PythonDirectory/python.exe" tools/stage.py --sdrsharp $SdrSharpDirectory --python-runtime $PythonDirectory --native-build build/windows-msvc/Release --viewer-build build/viewer-msvc/Release
if ($LASTEXITCODE) { throw 'Staging failed' }
