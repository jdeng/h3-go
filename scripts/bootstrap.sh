#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
while IFS=$'\t' read -r name path url revision; do
    if [[ -e "$path/.git" && "$(git -C "$path" rev-parse HEAD)" != "$revision" ]]; then
        echo "$name is not at pinned revision $revision; refusing to change your checkout." >&2
        exit 1
    fi
done < dependencies.lock
git submodule sync -- deps/stable-diffusion.cpp deps/cutlass
git submodule update --init -- deps/stable-diffusion.cpp deps/cutlass
git -C deps/stable-diffusion.cpp submodule sync -- ggml
git -C deps/stable-diffusion.cpp submodule update --init -- ggml
while IFS=$'\t' read -r name path url revision; do
    [[ "$(git -C "$path" rev-parse HEAD)" == "$revision" ]] || {
        echo "Unexpected $name revision; see dependencies.lock." >&2
        exit 1
    }
done < dependencies.lock
"$ROOT/scripts/apply-patches.sh"
