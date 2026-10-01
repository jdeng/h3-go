#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CUDA="${CUDA_HOME:-/usr/local/cuda-13.4}"
[[ -x "$CUDA/bin/nvcc" ]] || { echo "CUDA toolkit not found: $CUDA" >&2; exit 1; }
"$ROOT/scripts/bootstrap.sh"
mkdir -p "$ROOT/build" "$ROOT/dist/bin" "$ROOT/dist/lib"
cmake -S "$ROOT/deps/stable-diffusion.cpp" -B "$ROOT/build/native" \
    -DCMAKE_BUILD_TYPE=Release -DSD_CUDA=ON \
    -DSD_BUILD_SHARED_LIBS=ON -DSD_BUILD_SHARED_GGML_LIB=ON \
    -DSD_BUILD_EXAMPLES=OFF -DSD_WEBP=OFF -DSD_WEBM=OFF \
    -DGGML_BUILD_TESTS=OFF -DGGML_CUDA_NCCL=OFF -DGGML_CUDA_GRAPHS=OFF \
    -DH3_SERVICE_ROOT="$ROOT" -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON \
    -DCMAKE_CUDA_COMPILER="$CUDA/bin/nvcc" -DCUDAToolkit_ROOT="$CUDA" \
    -DCMAKE_CUDA_ARCHITECTURES=120 2>&1 | tee "$ROOT/build/configure.log"
cmake --build "$ROOT/build/native" --target stable-diffusion --parallel "${JOBS:-4}" \
    2>&1 | tee "$ROOT/build/native-build.log"
STAGING="$(mktemp -d "$ROOT/dist/.native-install.XXXXXX")"
trap 'rm -rf -- "$STAGING"' EXIT
cp -a "$ROOT/build/native/bin/"lib*.so* "$STAGING/"
for library in "$STAGING/"*; do
    mv -Tf -- "$library" "$ROOT/dist/lib/${library##*/}"
done
rmdir "$STAGING"
trap - EXIT
SOURCE="$ROOT/deps/stable-diffusion.cpp"
g++ -std=c++17 -O2 -pthread \
    -I"$SOURCE/include" -I"$SOURCE/examples" -I"$SOURCE/src" -I"$SOURCE/thirdparty" \
    "$SOURCE/examples/common/common.cpp" "$SOURCE/examples/common/log.cpp" \
    "$SOURCE/examples/common/media_io.cpp" "$SOURCE/examples/cli/image_metadata.cpp" \
    "$SOURCE/examples/cli/main.cpp" -L"$ROOT/dist/lib" -lstable-diffusion \
    -Wl,-rpath,'$ORIGIN/../lib' -o "$ROOT/dist/bin/sd-cli"
mkdir -p "$ROOT/dist/share/licenses/fused_int8"
cp "$ROOT"/third_party/fused_int8/{LICENSE,NOTICE,UPSTREAM.md} "$ROOT/dist/share/licenses/fused_int8/"
mkdir -p "$ROOT/dist/share/licenses/flash_attn"
cp "$ROOT"/third_party/flash_attn/{LICENSE,AUTHORS,UPSTREAM.md} "$ROOT/dist/share/licenses/flash_attn/"
cp "$SOURCE/LICENSE" "$ROOT/dist/share/licenses/stable-diffusion.cpp.LICENSE"
cp "$SOURCE/ggml/LICENSE" "$ROOT/dist/share/licenses/ggml.LICENSE"
cp "$ROOT/deps/cutlass/LICENSE.txt" "$ROOT/dist/share/licenses/cutlass.LICENSE"
source "$ROOT/scripts/go-env.sh"
cd "$ROOT"
go build -mod=readonly -trimpath -buildvcs=false -o dist/bin/h3-go .
go vet ./...
"$ROOT/dist/bin/h3-go" --help
