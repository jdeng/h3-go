#!/usr/bin/env bash
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CGO_ENABLED=1
export CGO_CFLAGS="\"-I$ROOT/deps/stable-diffusion.cpp/include\""
export CGO_LDFLAGS="\"-L$ROOT/dist/lib\" -lstable-diffusion -lstdc++ -lm -Wl,-rpath,\$ORIGIN/../lib"
export LD_LIBRARY_PATH="$ROOT/dist/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
