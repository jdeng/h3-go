#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/outputs"
cp "$ROOT/review.html" "$ROOT/outputs/index.html"
exec python3 -m http.server 8090 --bind 0.0.0.0 --directory "$ROOT/outputs"
