#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
sha256sum --check --quiet patches/SHA256SUMS
apply_series() {
    local source="$1" series="$2" patch
    if (cd "$source" && sha256sum --check --status "$series/FILES.sha256"); then
        echo "$(basename "$series"): patches already applied and verified"
        return
    fi
    if [[ -n "$(git -C "$source" status --porcelain --untracked-files=normal --ignore-submodules=all)" ]]; then
        echo "$source contains changes that do not match the patch manifest; refusing to overwrite." >&2
        exit 1
    fi
    for patch in "$series"/*.patch; do
        git -C "$source" apply --check "$patch"
        git -C "$source" apply "$patch"
    done
    (cd "$source" && sha256sum --check --quiet "$series/FILES.sha256")
}
apply_series "$ROOT/deps/stable-diffusion.cpp" "$ROOT/patches/sd.cpp"
apply_series "$ROOT/deps/stable-diffusion.cpp/ggml" "$ROOT/patches/ggml"
