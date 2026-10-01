#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="$ROOT/deps/stable-diffusion.cpp"
BASE=$(awk '$1 == "stable-diffusion.cpp" {print $4}' "$ROOT/dependencies.lock")
mkdir -p "$ROOT/build/checks"
INCLUDES=(-DGGML_MAX_NAME=160 -I"$SOURCE/src" -I"$SOURCE/include" -I"$SOURCE/ggml/include" -I"$SOURCE/ggml/src" -I"$SOURCE/thirdparty")
git -C "$SOURCE" show "$BASE:src/runtime/tiling.cpp" |
    g++ -std=c++17 -O3 "${INCLUDES[@]}" -Dprocess_tiles_2d=reference_process_tiles_2d \
        -x c++ -c - -o "$ROOT/build/checks/vae-reference-tiling.o"
g++ -std=c++17 -O3 "${INCLUDES[@]}" "$ROOT/scripts/check-vae.cpp" \
    "$ROOT/build/checks/vae-reference-tiling.o" -L"$ROOT/dist/lib" \
    -lstable-diffusion -lggml -lggml-base -Wl,-rpath,"$ROOT/dist/lib" -o "$ROOT/build/checks/check-vae"
"$ROOT/build/checks/check-vae"
