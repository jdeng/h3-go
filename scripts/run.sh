#!/usr/bin/env bash
set -euo pipefail
umask 077
ulimit -c 0
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec "$ROOT/dist/bin/h3-go" --models "${H3_MODELS:-$ROOT/../ComfyUI/models}" "$@"
