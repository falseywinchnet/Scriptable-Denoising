#!/bin/sh
# Existing MinGW toolchain on the Mac; the Mini has no Windows cross compiler.
set -eu
: "${CLEANUP_WINDOWS_PYTHON:?Set to the Windows Python 3.12 directory containing include/ and libs/}"
: "${CLEANUP_SDRSHARP:?Set to the directory containing SDRSharp.exe}"
if [ -f plugin/SDRSharp.Cleanup.csproj ]; then
  : "${CLEANUP_DOTNET:?Set to the dotnet SDK executable}"
  python3 tools/extract_sdrsharp.py "$CLEANUP_SDRSHARP/SDRSharp.exe" build/sdrsharp-sdk
else
  test -f plugin/precompiled/windows-x64/SDRSharp.Cleanup.dll
fi
cmake -S . -B build/windows -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=cmake/mingw.cmake \
  -DCLEANUP_GUIDE_THREADS=4 \
  -DCLEANUP_DYNAMIC_GUIDE=ON -DCLEANUP_PARALLEL_OBSERVATIONS=ON \
  "-DPython3_INCLUDE_DIR=$CLEANUP_WINDOWS_PYTHON/include" \
  "-DPython3_LIBRARY=$CLEANUP_WINDOWS_PYTHON/libs/python312.lib"
cmake --build build/windows --target Cleanup CleanupNative cleanupctl -j4
compiler=$(sed -n 's|^CMAKE_CXX_COMPILER:[^=]*=||p' build/windows/CMakeCache.txt)
pthread=$($compiler -print-file-name=libwinpthread-1.dll)
if [ ! -f "$pthread" ]; then
  pthread="$(dirname "$compiler")/../x86_64-w64-mingw32/bin/libwinpthread-1.dll"
fi
cp "$pthread" build/windows/
if [ -f plugin/SDRSharp.Cleanup.csproj ]; then
  "$CLEANUP_DOTNET" build plugin/SDRSharp.Cleanup.csproj -c Release
fi

sh tools/build-viewer-cross.sh
