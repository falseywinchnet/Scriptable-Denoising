#!/bin/sh
# Local cross compiler; no Windows cross toolchain is installed on the Mini.
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cmake -S "$root/viewer" -B "$root/build/viewer-windows" -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE="$root/cmake/mingw.cmake"
cmake --build "$root/build/viewer-windows" -j4
