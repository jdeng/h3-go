#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
: "${H3_PASSWORD:?Set H3_PASSWORD before starting the server}"
exec "$ROOT/scripts/run.sh" --serve "${H3_LISTEN:-0.0.0.0:8092}" \
    --output-dir "${H3_OUTPUT_DIR:-$ROOT/web-outputs}" "$@"
