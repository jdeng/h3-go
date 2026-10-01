# h3-go

An independent MiniMax H3 CLI and password-protected web service:
Go → cgo → stable-diffusion.cpp → GGML/CUDA. Supports text-to-video and
first/last-frame image-to-video, including audio.
It does not invoke ComfyUI, Python, PyTorch, or another Go inference service.
FFmpeg encodes the returned RGB frames and audio into an MP4.

This checkout has its own submodules, patches, build products, and outputs.
Only the existing model files are shared. It does not change `qwen-image-go`.
The embedded English UI follows `qwen-image-go`'s simple upload/paste/generate
workflow. It has a single-job HTTP API, not a persistent production job queue.

## Web UI

```bash
export H3_PASSWORD='choose-a-password'
./scripts/serve.sh
```

Open **http://localhost:8092** (or `http://<host>:8092` remotely). Username defaults
to `admin`; set `H3_USERNAME` to override it. The server binds `0.0.0.0:8092` by
default; `H3_LISTEN` changes the address. It does not stop Qwen on 8091 or the
existing static review server on 8090. Run the build below first.

Features: prompt, independently removable upload/paste/drop first and last frames,
aspect ratio/custom dimensions, duration, seed, Turbo/base mode, steps, CFG, flow
shift, LoRA strength, optional audio, cancellation, MP4 playback/download and
deletion. Frame/dimension rounding is shown before generation. Defaults remain
8-step Turbo, CFG 1 and shift 12. Only PNG/JPEG keyframes are accepted.

Generated web videos go into **`web-outputs/`**, not the public review server's
`outputs/` tree. `H3_OUTPUT_DIR` overrides this; keep it outside any unauthenticated
static root. Prompts/uploads are not retained or logged; output videos are saved
without prompt metadata. Passwords are required, never hardcoded in source.
Basic auth over HTTP is suitable only for a trusted LAN; use TLS on untrusted
networks. See [model capabilities, parameters and API](docs/model-and-ui.md).

The local running instance is a transient user service named `h3-go.service`;
`systemctl --user restart h3-go` / `stop h3-go` manage it. This does not install a
boot-time unit. Its ignored, mode-0600 `.env` supplies credentials; the script
itself reads environment variables, not `.env` automatically.

## Requirements and build

Validated on Ubuntu, RTX 5090 (SM120, 32 GB), 64 GB RAM, CUDA Toolkit 13.4,
GCC 13.3, and Go 1.25.4. The default is automatic VRAM management (`--max-vram 0`):
use live free device memory, reclaim idle workspaces, and evict reloadable weights
as necessary, retaining the backend's 512-MiB allocation safety margin.
An explicit positive `--max-vram` still limits managed weights and runner buffers;
it is not a strict total-process allocation cap. Other processes' allocations
remain unavailable, and concurrent allocation spikes can still cause an OOM.
Keep adequate free system RAM and VRAM. Larger/longer clips cost more memory.

Required tools: Git, CMake, a C/C++ compiler, Go 1.25+, CUDA's `nvcc`, FFmpeg,
and `ffprobe`. For example, Ubuntu's non-CUDA prerequisites can be installed with
`sudo apt-get install build-essential cmake git ffmpeg`.
Install Go and CUDA separately if absent. Nothing here changes a CUDA 12.8
installation, the driver, system CUDA symlinks, or an existing Python environment.

```bash
cd h3-go
CUDA_HOME=/usr/local/cuda-13.4 JOBS=4 ./scripts/build.sh
```

`build.sh` initializes the pinned submodules, applies and verifies the patches,
builds the native shared libraries and upstream CLI, then builds/vets the Go CLI.
It targets SM120 and does not rebuild PyTorch. The first CUDA build takes time.
Re-running it is incremental. `dist/bin/h3-go` finds its own `dist/lib` via RPATH;
deploy the entire `dist` tree, not just the executable.
Library installation uses staged files and atomic renames rather than overwriting
memory-mapped shared objects. For a complete version upgrade, finish active jobs,
stop the HTTP service, build, and start it again so all components use one version.

Pins are recorded in `dependencies.lock` and submodule gitlinks:

| Dependency | Revision |
| --- | --- |
| stable-diffusion.cpp | `3f8527a46c54ecf4cb4ed6003da8e8982283c73c` |
| Nested GGML | `89c4413f5da6fb20cc796f16033d37f129be81fd` |
| CUTLASS | `d4b4b494c3c51bf6507e7ab09fbafd1e9fa94f39` |

These dependencies are submodules, not vendored source. Two Apache-2.0 INT8
kernel source files are copied under `third_party/fused_int8`, with provenance,
license, notices, and the local workspace adaptation retained. No comfy-kitchen
checkout or Python package is needed. See `patches/README.md` for PR provenance.

Eleven BSD-3-Clause FlashAttention forward headers are copied under
`third_party/flash_attn` from release v2.8.3, commit
`060c9188beec3a8b62b33a3bfa6d5d2d44975fab`, with license, authors, and adaptations
documented there. Only the dense forward CUDA specialization is compiled; it
reuses the existing CUTLASS submodule and has no ATen/PyTorch dependency.

`bootstrap.sh` rejects unexpected revisions or edited patched files rather than
resetting them. Patched submodules intentionally appear dirty. Update the patches
and checksum manifests when deliberately changing managed native code.
See [code review](docs/code-review.md) for the service cleanup, remaining risks,
and possible reductions to the inherited patch set.

## Models

Set `H3_MODELS` to a directory containing:

```text
diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors
text_encoders/qwen3vl_32b_minimax_h3-Q4_K_M.gguf
vae/minimax_h3_video_vae_fp16.safetensors
vae/minimax_h3_audio_vae_fp32.safetensors
loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors
```

`run.sh` defaults to the existing sibling `../ComfyUI/models` directory, only as
file storage; ComfyUI need not be running. The bare binary defaults to `models`
relative to the current working directory, or accepts `--models /path/to/models`.
No weights are downloaded or converted by the build.

The tested GGUF encoder is from `leejet/MiniMax-H3-GGUF`, revision
`345fc13a6c4e722a0cbe6bcb0f583e204645e1ac`. Its SHA-256 is
`11e6efe70a57ce7f4838c47bdbd1a1c4b8ce10e2b7747f1b065990b70f4b05fc`.
The existing NVFP4/AWQ encoder is not used or deleted. Q4_K_M and NVFP4 are
different quantizations, not interchangeable containers; this backend does not
claim to run the NVFP4/AWQ file.

## Generate

```bash
./scripts/run.sh \
  --prompt '视频描述：晨光中的红色茶壶，蒸汽缓缓升起，镜头向右平移。声音描述：轻微沸水声，没有人声。' \
  --width 640 --height 384 --frames 56 --steps 8 \
  --output outputs/teapot.mp4

./scripts/run.sh \
  --input-image reference.png \
  --prompt '视频描述：保持参考图的主体，镜头缓慢向右移动。声音描述：安静的室内环境。' \
  --duration 10 --width 640 --height 384 \
  --output outputs/reference-video.mp4
```

Defaults: Turbo LoRA strength 1, CFG 1, 8 steps, Res Multistep + Simple,
flow shift 12, CPU RNG, seed 42, 24 fps, Flash Attention, and automatic VRAM.
CLI input images are PNG/JPEG, at most 64 megapixels each. `--input-image` and
`--end-image` select first/last keyframes independently. Multi-reference images,
reference video and reference audio are not supported by this FL2VA service.
`--turbo=false` disables the adapter and defaults to 20 steps unless `--steps` is
explicit; `--audio=false` skips audio decoding. `--cfg`, `--flow-shift`,
`--lora-strength`, and `--negative-prompt` expose the corresponding UI controls.
Width/height must be multiples of 32. Native frame counts round up to `17k+5`:
10 seconds requests 240 frames and produces 243 (10.125 seconds). The CLI accepts
up to 15 requested seconds/362 aligned frames; trim externally for exact durations.
A 10-second request is validated below; 15-second and higher-resolution clips
have not been benchmarked in this optimization pass.

The CLI writes an MP4 and prints timing JSON on stdout; native diagnostics go to
stderr. `--report build/timing.json` optionally saves timing data without a prompt.
There is no prompt/history database, result cache, or cross-request model cache.
Each CLI invocation or HTTP request frees its native context. Temporary audio/video files are created
beside the destination and removed on normal completion/error; SIGKILL or a crash
can leave a `.h3-encode-*` directory. Completed output replaces the requested path
atomically. Benchmark logs deliberately include the public synthetic test prompt.

## Implemented optimizations

1. **Request-local text-refiner cache.** Compute H3's condition projection and token
   refiner once, then reuse the result in later sampling steps. This does not cache
   the 32B text encoder across requests. Shape/content changes invalidate the cache;
   runner shutdown frees it. Alternating CFG conditions recompute rather than reuse
   the wrong tensor. The tested CFG=1 run records seven cache hits over eight steps.
2. **H3 INT8 fusion eligibility.** Enable the existing CUTLASS INT8 GEMM +
   scale/bias epilogue for H3's QKV, FFN-up, attention-output, and FFN-down matrix
   sizes on SM120 (128–65535 rows). Accumulation/output precision is unchanged;
   unsupported shapes or rejected kernel configurations retain the cuBLAS path.
3. **Less redundant VAE tiling.** Use 256×256 tiles with 25% target overlap instead
   of the previous CLI's 50%. This test uses six rather than eight spatial tiles
   per temporal chunk. Overlap changes decoding slightly; it is not bit-identical.
4. **Actual H3 graph fusion.** Fuse RMSNorm, learned channel scale, non-interleaved
   partial RoPE, and attention-layout output in one CUDA kernel. It reads strided
   QKV views directly, preserves the unrotated tail, and retains the original graph
   on unsupported backends/layouts. This is distinct from the inherited Qwen matcher.
5. **Remove redundant graph work.** Use the existing fused SwiGLU operator, zero-copy
   QKV/modulation slices, and omit identity-only runtime LoRA scale operations.
6. **Pinned prefetch staging.** Reuse small pinned-host buffers for pageable weight
   uploads, keeping them alive until the transfer event completes. The pool is
   capped at 1 GiB (512 MiB per buffer); this benchmark allocates 320 + 384 MiB.
   Oversized transfers, unavailable buffers, or unsupported backends use the old
   path. This copies weights for transfer; it does not cache prompts or results.
7. **Automatic budget and cyclic weight reuse.** H3 uses live free VRAM by default.
   During segmented H3 sampling, evict weights whose next use is farthest away in
   the repeated block sequence, rather than always evicting the highest-numbered
   blocks. Shared parameters use their nearest upcoming use; active/prefetched
   weights remain protected. Other runners keep their existing eviction policy.
   `params_backend=cpu` remains the reloadable backing store, not CPU inference.
   Startup `auto_fit` placement is distinct from this runtime residency manager.
8. **Block dataflow fusion.** Fuse RMSNorm, learned channel scale, and per-modality
   AdaLN shift/scale into one CUDA kernel, and fuse gated residual updates across
   modality spans. This avoids materializing the intermediate FP32 normalization,
   multiply/add, and concatenation tensors. Reductions and residual math remain
   FP32; the original graph remains available on unsupported layouts/backends.
9. **Direct half-precision attention storage.** Produce scaled K and V directly
   in the FP16 layout already consumed by Flash Attention, combining K's
   RMSNorm/RoPE/scale/cast and V's layout/scale/cast. Q and reduction math remain
   FP32. This removes redundant FP32 temporaries, not an additional model
   quantization or an unvalidated conversion of all activations to half precision.
10. **FFN adapter/activation fusion.** Where the H3 FFN projection has a compatible
    runtime adapter, combine the projection/adapter rescaling, addition, and
    SwiGLU into one CUDA kernel. Keep the existing matmuls and FP32 arithmetic,
    but do not write/read their full-width rescaled and combined intermediates.
    SwiGLU uses a separate contiguous output: an attempted in-place strided FFN
    variant changed the full model's output despite passing isolated kernel
    checks, and was rejected. The allocator can reuse dead activation/update
    buffers for modulation and residual fusion, but never their shared weights.
11. **Overlapped H3 block execution.** Prepare the next block's GPU/pinned buffers
    before submission, then launch the current block asynchronously before filling
    and uploading those pinned buffers. Transfer events protect staging-buffer
    lifetime. Recycle at most two expired graph-cut tensors, queue cut copies on
    the compute stream, and skip empty request-cache synchronization. Keep the
    necessary block-completion boundary before releasing weights or reusing the
    workspace, and complete the final graph before reading results on the CPU.
    This path is automatic for segmented single-CUDA H3 without debug callbacks;
    other runners and scheduler/CPU paths retain synchronous execution.
12. **Standalone FlashAttention forward.** For SM120, contiguous, unmasked,
    noncausal, equal-head self-attention with head dimension 64 or 128 and at least
    1024 tokens, use the upstream FlashAttention-2 CUDA forward implementation.
    It computes 128-query/64-key tiles with four warps and FP32 accumulation.
    GGML supplies the stream, graph-planned scratch buffers, FP32-to-FP16 Q conversion and
    FP16-to-FP32 output conversion. Existing FP16 K/V storage is reused directly.
    Masked, strided, short, grouped-query and other unsupported cases keep the
    original GGML attention. No user-facing switch or Python dependency is added.
    Different reduction order and half-precision output rounding mean this path
    is numerically validated, not bit-identical to the original GGML kernel.
    Scratch storage is included in GGML's allocation-size hook so the automatic
    VRAM planner sees it rather than discovering a large hidden allocation later.
13. **Batched H3 video decoder.** Decode up to four equal-length temporal chunks
    together on CUDA with the default 256-pixel tiles, preserving independent batch
    axes and the original temporal overlap. Flatten batch/token rows for decoder
    GEMMs, fuse SwiGLU and scaled residuals, and use fused unaffine RMS/partial-RoPE
    with direct FP16 K/V and head-64 FlashAttention. Keep VAE weights resident for
    this decode request, releasing them on completion or failure. CPU and larger
    custom tiles use one chunk per batch; unsupported fused ops retain fallbacks.
14. **Contiguous VAE assembly.** Traverse spatial tiles plane-by-plane rather than
    jumping across entire frame/channel planes for each pixel. Compute blending
    weights once per axis, remove per-pixel vector allocations from H3 temporal
    blending, and write chunks into one preallocated output instead of repeatedly
    concatenating the growing full video. Spatial/circular and temporal blending
    retain the previous arithmetic and pass exact CPU comparisons.
15. **BF16 sampling activation storage.** On H3 CUDA, fused RMSNorm/AdaLN and
    adapter/SwiGLU write BF16 directly. ConvRot consumes those values without
    expanding the entire activation back to FP32, and BF16 LoRA matmuls reuse them.
    Reductions, nonlinear math, GEMM outputs and the residual stream remain FP32.
    This halves these particular activation buffers, not the whole model's VRAM.
    Unsupported layouts/backends keep the existing unfused path.
16. **Native BF16 LoRA operands.** The old FFN `GGML_PREC_F32` flag caused CUDA to
    promote BF16 adapter weights and inputs to FP32, not merely accumulate in FP32.
    H3's BF16 runtime adapter now uses BF16 operands with FP32 accumulation/output.
    Remove the redundant `1/128` input and `128` output rescaling for this path
    and for H3 ConvRot INT8 projections. Non-BF16 adapters retain their original
    precision/scaling policy. There is no offline weight conversion or LoRA merge.
17. **Fewer ConvRot barriers.** Advance each butterfly stage across all independent
    groups before synchronizing, instead of synchronizing each group wave at every
    stage. The direct BF16 and existing FP32 paths preserve the rotation arithmetic
    and agree exactly with a CPU reference. Oversized rows retain a pooled FP32
    conversion and the original split-kernel fallback.

Turbo LoRA still executes as runtime adapter matmuls; there is no merged/requantized
INT8 LoRA, fully BF16 sampling stream, activation/quantization fusion, GPU-only VAE tile
assembly, or native NVFP4 text encoder.
CUDA graphs remain disabled as in the comparison baseline.

## Measured results

### Latest sampling dataflow round

The rebuilt public Go binary takes **52.03 s including MP4**, with **38.32 s
sampling**, for the same 640×384, 243-frame, 8-step/CFG-1/seed-42 request. An
immediately repeated old-library baseline takes **55.59 s / 41.85 s**: **6.4%
less total time and 8.4% less sampling time**. The development candidate measured
51.94 s / 38.18 s. The earlier VAE-round baseline was 55.95 s / 42.18 s.

| Stage | Repeated baseline | Installed dataflow build | Earlier ComfyUI |
| --- | ---: | ---: | ---: |
| End-to-end | 55.59 s | **52.03 s** | 42.82 s |
| Sampling | 41.85 s | **38.32 s** | 30.05 s |
| Video VAE | 8.78 s | 8.81 s | 4.96 s |

This is a partial sampling-dataflow improvement, not parity with ComfyUI. Total
time remains approximately **21.5% higher**, with an 8.27 s sampling-stage gap.
Runtime LoRA GEMMs, FP32 projection outputs/residuals, weight streaming, and the
separate activation-to-INT8 pass remain. The deployed text encoder quantizations
also differ (Q4_K_M versus NVFP4/AWQ). These are individual runs, not statistical
confidence bounds or guarantees for other sizes.

At 56 frames the activation-storage change reduces the reported peak segmented
sampling workspace from **1462.48 to 1232.16 MiB**; this is not total VRAM.
Isolated BF16-operand LoRA-up matmuls take 0.379/1.296 ms at 4350/16800 tokens,
versus 0.660/2.299 ms with promoted FP32 operands. Those microbenchmark gains
must not be interpreted as the full request's speedup.

All eight native check scripts, clean source patch replays, public build and
Go vet pass. The installed binary also passes first-frame I2V and a 5-frame
cropped-tail test. The 10-second result has 243 frames, 24 fps and stereo 32-kHz
AAC, and fully decodes. BF16 rounding intentionally changes the sampling
trajectory: the development result's old/new video SSIM is **0.806329**, not
pixel equivalence. Synthetic review frames are coherent; general quality has
not been exhaustively evaluated.

This round changes nine native files (**139 added / 61 removed lines**) and adds
a dedicated dataflow numerical check. Reports are under
`build/bench/dataflow-{before,final}-10s.{json,log}` and
`build/checks/dataflow-final-*.log`. No other service was stopped or modified.

Earlier optimization rounds and their individual measurements are preserved in
[optimization history](docs/optimization-history.md). These are historical snapshots,
not cumulative speedup claims or results for the current build.

### Validation and reproduction

```bash
./scripts/check-int8.sh
./scripts/check-rope.sh
./scripts/check-block.sh
./scripts/check-residency.sh
./scripts/check-pipeline.sh
./scripts/check-attention.sh
./scripts/check-vae.sh
./scripts/check-dataflow.sh
./scripts/test.sh
./scripts/benchmark.sh
./scripts/check-video.sh outputs/baseline.mp4 outputs/h3-optimized.mp4
source ./scripts/go-env.sh
go vet ./...
node --test scripts/ui.test.cjs
```

- All four H3 matrix shapes pass sampled CPU-reference comparisons at 127, 129,
  and 4149 rows, with and without bias: 12 cases, zero measured sample error.
- An Nsight Systems trace confirms eight CUTLASS fused launches and four fallback
  cuBLAS/dequant launches in that check. This verifies execution, not just eligibility.
- The H3 RMS/RoPE kernel passes full-output CPU-reference checks with strided QKV,
  two batches, 2/96/128 rotated channels, unrotated tails, and a single-token case.
  Maximum measured absolute error is `4.77e-7`.
- Decoder checks additionally cover unaffine 64-channel RMS, interleaved QKV,
  48/64 rotary channels, and four batches. Head-64 FlashAttention is checked at
  129 and 1797 tokens, including a 32-head/four-batch VAE tile configuration.
- The host VAE check compares spatial tiles with the original pinned implementation,
  including circular axes, odd sizes, zero overlap, oversized tiles and batches.
  Temporal blending is checked against the original coordinate-indexing formula.
  All comparisons require exact element equality, not a relaxed image metric.
- The block checks compare fused modulation, residual, and FFN math against their
  unfused CUDA graphs, including modality boundaries, repeated modulation rows,
  batches, partial channel tiles, and allocator aliasing. K/V and Flash Attention
  checks also compare the direct-FP16 path with the old scale/cast/layout chain.
- The dataflow check requires exact BF16 activation equality with FP32 fused
  output followed by a BF16 cast. Direct BF16 quantization must match an explicit
  FP32 expansion. ConvRot outputs/scales also match an independent CPU reference
  for 5376/14336/32768 channels, 64/256 groups, zero and high-magnitude rows, row
  padding, and the large-row fallback. BF16-operand LoRA-up GEMMs at 4350/16800
  tokens are sampled against double-precision CPU dot products. These operator
  checks do not establish whole-video equivalence after BF16 activation rounding.
- A CPU-only residency-policy check covers cyclic ordering, wraparound, shared
  parameters, protected current weights, and the unchanged non-H3 policy.
- CUDA cache checks cover request-local stream ordering, two-buffer reuse across
  consecutive cuts, cache clearing, and the non-reusing fallback.
- Attention checks cover 129/1024/4350/16800 tokens, 2/56 heads and one/two
  batches, including partial query/key tiles. Outputs must be finite and selected
  full rows must agree with a double-precision CPU softmax reference to `5e-6`
  absolute error, with H3's `1/128` K/V scaling. Optional old-library dumps check
  every output element (`2e-5` maximum absolute error, `0.006` relative RMS error),
  and ensure the CPU-reference error does not exceed the old kernel's by more
  than 25% plus `2e-7`. The old GGML kernel uses FP16 value accumulators; the
  FlashAttention path uses FP32 accumulation and is more accurate in these checks.
  These are numerical tolerances, not an exact-frame guarantee. For a matched
  library comparison, run `check-attention.sh build/checks/attention-before` with
  the old `LD_LIBRARY_PATH`, then `check-attention.sh - build/checks/attention-before`
  with the new one. Timings include conversion kernels and CUDA synchronization.
- The eight-step diagnostic benchmark records seven text-refiner cache reuses.
- Both patch series replay cleanly against pinned source and match final hashes;
  repeated bootstrap succeeds without changing anything.
- FFprobe validates both Go MP4s. First/middle/last synthetic frames show coherent
  teapot/steam/camera motion without obvious corruption. Before/after video SSIM
  is 0.98909; this is a regression sanity check, not a general quality evaluation.
- Pinned staging gives identical decoded video pixels to the fusion-only Go run
  (SSIM 1.0), confirming no observed transfer corruption in this sample.

Local evidence is under ignored `build/bench` and `build/checks`; output videos are
under ignored `outputs` (benchmarks) and `web-outputs` (HTTP). `scripts/test.sh`
runs Go race tests/vet for request validation, keyframe decoding, authentication,
CSRF protection, single-job exclusion, upload limits, cancellation propagation,
error privacy, byte ranges, deletion and output confinement. Local real-HTTP smoke
tests cover T2V, first+last-frame generation without audio, and last-only Standard
mode with CFG above 1. Browser checks cover pasting/removal, parameters, playback,
mobile layout and cancel. These are integration checks, not model quality guarantees.
The optional Node.js test uses only built-in modules, mocks image decode completion,
and reproduces overlapping uploads/removal, combined limits, and late deletion
responses without running inference. Node.js is not a build or runtime dependency.
The full default-size HTTP request took 52.36 s and its decoded video frames and
timestamps exactly match the optimized CLI result. A concurrent incremental build
with atomic library replacement completed without restarting the running server.

`benchmark.sh` uses the native CLI for comparison and debug
logging to show cache reuse. It does not automatically stop other GPU services.
Use `H3_MAX_VRAM=16 H3_BENCH_NAME=limited ./scripts/benchmark.sh` to compare an
explicit budget without changing defaults or overwriting another benchmark.
The optional video check requires your own saved baseline from the same prompt,
seed, request path, and generation settings. It fails on any decoded-frame or
timing difference, so do not use a different encoder/backend as that reference.

## Review videos

```bash
./scripts/review.sh
```

This optional Python static server binds `0.0.0.0:8090` and serves **only `outputs`**,
not model files, source, or benchmark logs. It has no authentication; use it only
on a trusted network. Stop any existing listener on that port first. The included
review page links the synthetic validation clips; the optional ComfyUI comparison
clip comes from the earlier local benchmark and is not produced by `benchmark.sh`.
