#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/build/checks"
g++ -std=c++17 -O2 -I"$ROOT/deps/stable-diffusion.cpp/ggml/include" \
    "$ROOT/scripts/check-attention.cpp" -L"$ROOT/dist/lib" -lggml-cuda -lggml -lggml-base \
    -Wl,-rpath,"$ROOT/dist/lib" -o "$ROOT/build/checks/check-attention"
"$ROOT/build/checks/check-attention" "$@"
