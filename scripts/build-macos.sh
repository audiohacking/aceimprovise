#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/build"
mkdir -p "$BUILD"
cd "$BUILD"
cmake "$ROOT" -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)"
echo "Built: $BUILD/ace-improvise-server"
