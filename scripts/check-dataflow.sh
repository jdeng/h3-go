#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
LIBRARY="${H3_CHECK_LIB:-$ROOT/dist/lib}"
mkdir -p "$ROOT/build/checks"
g++ -std=c++17 -O2 -I"$ROOT/deps/stable-diffusion.cpp/ggml/include" \
    "$ROOT/scripts/check-dataflow.cpp" -L"$LIBRARY" -lggml-cuda -lggml -lggml-base \
    -Wl,-rpath,"$LIBRARY" -o "$ROOT/build/checks/check-dataflow"
"$ROOT/build/checks/check-dataflow"
