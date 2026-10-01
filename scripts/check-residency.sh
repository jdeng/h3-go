#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="$ROOT/deps/stable-diffusion.cpp"
mkdir -p "$ROOT/build/checks"
g++ -std=c++17 -O2 -I"$SOURCE/src" -I"$SOURCE/ggml/include" \
    "$ROOT/scripts/check-residency.cpp" -L"$ROOT/dist/lib" -lstable-diffusion -lggml-base \
    -Wl,-rpath,"$ROOT/dist/lib" -o "$ROOT/build/checks/check-residency"
"$ROOT/build/checks/check-residency"
