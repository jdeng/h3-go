# H3 optimization history

These are earlier benchmark snapshots, newest first. See the [README](../README.md)
for the current configuration, latest measurements, and validation commands.
Paths below are relative to the repository root; local evidence is not committed.

### Previous VAE round

The same 640×384, 243-frame (10.125-second), 8-step/CFG-1/seed-42 request now takes
**55.95 seconds including MP4**, down from **66.53 seconds**: 10.58 seconds saved,
or **15.9% less end-to-end time**. Video VAE drops from **19.48 to 8.91 seconds**
(54.3% less time, 2.19× decoder throughput). Sampling is unchanged: 42.19 → 42.18 s.

| Stage | Before this round | After this round | Previously measured ComfyUI |
| --- | ---: | ---: | ---: |
| End-to-end | 66.53 s | **55.95 s** | 42.82 s |
| Sampling | 42.19 s | 42.18 s | 30.05 s |
| Video VAE | 19.48 s | **8.91 s** | 4.96 s |
| Audio VAE | 2.05 s | 2.04 s | 0.22 s |

The first batching/fusion/residency pass reached 62.62 s total / 15.95 s video VAE.
Fixing host assembly and redundant copies then reached the final result. This
was not just a GPU-kernel problem: the old temporal blend allocated coordinate
vectors for every pixel access, and the spatial tiler used cache-unfriendly strides.
The host-assembly pass preserves every decoded video pixel and timestamp of the
first pass. Against the pre-VAE baseline, SSIM is 0.991834; batch GEMMs and attention
rounding change pixels, so the complete optimization is not bit-identical.

**Correction to the earlier diagnosis:** H3's video decoder is a 36-layer ViT,
preceded by a single 1×1×1 post-quant convolution. The CNN/GroupNorm/SiLU/padding
path discussed previously is mainly the encoder, not this decoding bottleneck.

Compared with the earlier ComfyUI run, native is still **13.13 s / 30.7% slower**.
Sampling is now the largest remaining stage gap (12.13 s); video VAE adds 3.95 s
and audio VAE 1.82 s. Stage gaps do not add to the total because loading, text
encoding and MP4 handling differ. This uses the two deployed encoder quantizations
(native Q4_K_M versus ComfyUI NVFP4/AWQ), not identical numerical pipelines.

Next substantial targets are half-precision activation/dataflow in sampling,
fixed-LoRA merge/requantization or cached patched weights, and half-precision/GPU
assembly in the VAE. Those are unmeasured opportunities, not promised speedups.
More small fusions or changing Go to another host language will not by themselves
explain away the remaining sampling gap. CUDA graphs remain a secondary option
until capture lifetimes are compatible with segmented weight streaming.

Native patches for this round touch seven files: **233 added / 117 removed lines**,
plus a new 107-line host VAE check and extensions to the existing numerical checks.
Reports: `build/bench/vae-final-10s.{json,log}`, `build/checks/vae-final-*.log`.
The public build and both clean patch replays pass. All seven check scripts pass;
the installed Go binary also passes 56-frame image-to-video and 5-frame cropped-tail
smoke tests. The 10-second MP4 has 243 frames and stereo 32-kHz AAC and decodes fully.

### Previous attention round

2026-10-01, same RTX 5090, 640×384, 8 steps, CFG 1, seed 42, automatic VRAM,
synthetic teapot prompt and shared model files. Other GPU services stayed running
and idle. The old shared libraries were preserved and used for matched references.

| Measurement | Previous GGML attention | Standalone FlashAttention |
| --- | ---: | ---: |
| Isolated attention, 4350 tokens × 56 heads × 128 channels | 5.943 ms | **2.943 ms** |
| Isolated attention, 16800 tokens × 56 heads × 128 channels | 86.963 ms | **39.788 ms** |
| Go 10-second request, including MP4 | 91.29 s | **67.88 s** |
| Sampling in that 10-second request | 65.68 s | **42.05 s** |
| Video VAE decode in that request | 20.74 s | 20.96 s |
| Short native CLI process, excluding MP4 | 17.19 s | 16.98 s |
| Sampling in that short native run | 10.25 s | 9.77 s |

The long request improves by **25.6% overall**, saving **23.41 seconds**, and
**36.0% in sampling**. Both long outputs contain 243 frames / 10.125 seconds,
24-fps H.264 and stereo 32-kHz AAC, and decode completely without errors or OOM.
The short 56-frame request shows **little overall improvement**; the kernel's
roughly 2× speedup is not a 2× application speedup. Video VAE, runtime LoRA,
weight transfers and other FP32 passes are unchanged. That optimization round
did not include a 10-second ComfyUI comparison; a fresh comparison follows below.

Numerical checks pass against a double-precision CPU reference. At 4350/16800
tokens, selected-row maximum absolute errors fall from `2.74e-6`/`1.87e-6` to
`1.41e-7`/`8.74e-8`, respectively, for H3-scaled K/V. New FP32 accumulation and
different rounding intentionally change the old FP16-accumulator results:
full-tensor relative RMS differences are 0.245%/0.477%. **Videos are not
pixel-identical** to the old backend (short-clip SSIM 0.822). Synthetic review
frames are coherent, but this is not an exhaustive image-quality evaluation.

The public build script, clean patch replay, attention CPU-reference checks,
RMS/RoPE, modulation/FFN, INT8, cache stream ordering and residency checks pass.
Reports are under `build/bench/attention-*` and `build/checks/attention-*`.
The rebuilt public Go binary also passes a 56-frame image-to-video smoke test
in 19.07 seconds including MP4; that is not a matched old/new timing comparison.
The new scratch space is graph-planned, and the CUDA shared library links no
PyTorch/ATen libraries. Simple GGML tile-size tuning was tried first and discarded;
the shipped implementation uses the standalone forward kernel instead.

### Pre-VAE 10-second ComfyUI comparison

A fresh sequential comparison on 2026-10-01 uses the public native binary and
the existing ComfyUI API: 640×384, 243 frames / 10.125 seconds, 24 fps, 8 steps,
CFG 1, seed 42, the same Chinese synthetic prompt and Turbo LoRA strength 1.
ComfyUI model/node caches were freed first; its report has no cached nodes.
Other services remained running and idle. ComfyUI was unloaded again before
running the native binary. Both totals include producing an MP4.

| Stage | h3-go | ComfyUI |
| --- | ---: | ---: |
| End-to-end | **66.53 s** | **42.82 s** |
| Sampling | 42.19 s | 30.05 s |
| Video VAE decode | 19.48 s | 4.96 s |
| Audio VAE decode | 2.05 s | 0.22 s |

Native takes **23.71 seconds longer**, or **55.4% more time** (1.55×). The largest
individual stage gap is video VAE: **14.52 seconds**, about 3.93× the ComfyUI
decode time. Sampling adds 12.14 seconds. These stage differences do not sum to
the total gap because text conditioning, loading and MP4 encoding also differ.

At this point the native ViT VAE decoder still used FP32 activation storage,
individually processed tiles, repeated weight eviction between temporal chunks,
and expensive CPU stitching. ComfyUI used half-precision decoder activations and
small tile batches. The latest VAE round above addresses several of these issues.
Native sampling still evaluates runtime LoRA projections,
keeps many FP32 intermediates, and has weight streaming/block-completion boundaries.
ComfyUI applies LoRA through its weight-patching path and uses BF16 DiT activations.
The remaining sampling gap has not been apportioned among those individual causes.

This compares the two deployed stacks, not identical numerical execution:
native uses the Q4_K_M text encoder, ComfyUI uses NVFP4/AWQ, and the latter is a
long-lived process with different kernel caches and encoding behavior. Do not
attribute the entire gap to GGML, Go or the text encoder. These reports are the
preserved baseline from before the VAE optimization, not the latest native result.

Raw reports: `build/bench/current-comfy-10s.json`,
`build/bench/current-native-10s.json`, and `build/bench/current-native-10s.log`.
The exact API workflow is `build/bench/current-comfy-workflow.json`.

### Previous execution-pipeline round

2026-10-01, RTX 5090, 640×384, 56 frames, 8 steps, CFG 1, seed 42, automatic
VRAM and the same synthetic prompt. Existing services remained running. These
are warmed-filesystem before/after runs, not a comparison against a cold start.

| Measurement | Before | After |
| --- | ---: | ---: |
| Native CLI process, excluding MP4 encoding | 20.11 s | **17.21 s** |
| Native sampling | 13.17 s | **10.30 s** |
| Direct Go T2V, including MP4 encoding | 19.93 s | **17.09 s** |
| Direct Go I2V, including MP4 encoding | 21.82 s | **19.03 s** |

The native repeat pair reduces total time by **14.4%** and sampling by **21.8%**.
An earlier pair measured 21.35 → 17.14 s; the repeated pair above is the more
conservative result. Both native comparisons and the matched Go T2V/I2V runs
pass exact decoded-frame and timing comparisons. Go references were regenerated
with the preserved old library and identical arguments, rather than assuming
older test clips used identical prompt whitespace/settings.

The first asynchronous prototype failed output regression: the old cache/readback
helpers used the per-thread default CUDA stream, not the in-flight compute stream.
Cache copies now use the compute stream; view fallback reads and final CPU readback
wait correctly. Rejected outputs are excluded from public review. Focused checks
exercise cache stream ordering, bounded two-buffer reuse, and unchanged synchronous
fallback behavior. Patch `0008-h3-overlapped-block-pipeline.patch` replays cleanly
with the preceding series and verified source hashes.

This fixes a useful part of the pipeline, **not parity with ComfyUI**. A required
completion boundary still exists between blocks, streamed weight buffers still
allocate, and attention/BF16/runtime-LoRA/VAE differences remain. No new GPU trace
was collected for this round: the measured improvement is end-to-end wall time,
not a claim that all previously observed stalls disappeared. Logs and reports are
under `build/bench/pipeline-*`; focused checks are in `build/checks/pipeline-*`.

The rebuilt public Go binary also completes a **10-second request in 89.90 s**
without OOM, producing 243 frames / 10.125 seconds at 24 fps with stereo 32-kHz
AAC audio. This is a long-video smoke test, not a matched old/new timing or
pixel-equivalence comparison. See `pipeline-go-10s.json` and its ffprobe report.

### Previous memory/dataflow round

2026-10-01, same RTX 5090, synthetic prompt, 640×384, 56 frames, 8 steps,
CFG 1 and seed 42. Other GPU applications remained resident (~13 GiB total);
the idle Qwen service was paused only during tests and restored afterward.

| Native CLI, warmed filesystem, excluding MP4 encoding | Previous code, 16 GiB | Previous code, automatic | New code, automatic |
| --- | ---: | ---: | ---: |
| Whole process | 22.55 s | 22.18 s | 20.09 s |
| Sampling | 15.36 s | 15.10 s | 13.14 s |
| Video VAE | 4.18 s | 4.06 s | 3.84 s |
| Peak H3 graph workspace | 1253.70 MiB | 1253.70 MiB | 1484.92 MiB |

The default-to-default change reduces observed total time by **10.9%** and
sampling by **14.5%** in this small test. Removing the budget alone saves only
0.37 seconds here. With both versions on automatic VRAM, the code changes reduce
total time by 9.4%; keeping the new code at 16 GiB gives 20.23 seconds total and
13.33 seconds sampling. These are single-run observations, not a speed guarantee.
The VAE implementation did not change in this round; its small timing variation
must not be attributed to a new VAE optimization.

Matched automatic-budget Nsight traces (`round3-before` → `memory-safe`) show:

- CUDA kernel instances: **102,051 → 84,045** (17.6% fewer).
- Summed GPU kernel time: **11.19 → 9.55 s**; this is not additive wall time.
- Whole-request host-to-device traffic: **108.73 → 105.72 GB** (decimal units,
  only 2.8% less). The new residency order helps, but does not eliminate uploads.
- Actual fused launches: 800 RMSNorm/modulation, 800 gated residual, 400 direct
  FP16 K, 400 direct FP16 V, and 402 FFN adapter/SwiGLU kernels.

Fusion reduces intermediate read/write traffic, **not every peak allocation**.
The separate FFN output raises peak workspace by about 231 MiB; it is retained
because it is correct and faster in the measured workload. The attempted smaller
in-place FFN output was rejected by full-video regression testing. All retained
native automatic/16-GiB/profile runs produce exactly the baseline's decoded video
frames and timing, checked with `check-video.sh`, not only a visual inspection.

Final direct Go integration runs take **19.82 s T2V** and **21.83 s I2V** including
MP4 encoding, with the same small output dimensions/frame count. They use the C
API directly and are not interchangeable with native CLI timings. The earlier
ComfyUI API measurement remains 12.49 seconds, with different encoder precision,
process caches, and VRAM policy: the native backend is still roughly 1.6× slower.
Evidence is in `build/bench/h3-memory-optimized.log`, `memory-safe16.log`,
`memory-t2v.json`, `memory-i2v.json`, and `build/checks/memory-safe*`.

A **10-second T2V request** at 640×384/8 steps completes in **95.84 seconds**
including MP4 encoding, producing 243 frames / **10.125 seconds** at 24 fps with
stereo 32-kHz AAC audio. Automatic VRAM completes without OOM while the other GPU
applications remain resident. The integration monitor's minimum observed free
VRAM is 702 MiB across the short/long runs; this is not a concurrency guarantee.
First/middle/last frames of the I2V and long T2V clips were inspected for obvious
corruption. `build/bench/memory-10s.json` and `memory-10s-probe.json` record the run.

### Pre-pipeline GPU timeline comparison

A separate headless ComfyUI diagnostic on 2026-10-01 profiles the same synthetic
workflow without restarting or changing any service. It saves no new video. Its
sampling node takes 6.95 s and video VAE 1.42 s under Nsight; initialization and
profiling overhead make these different from the earlier 6.26 s / 1.57 s API
measurements. The native trace is the retained `memory-safe` run. These are not
simultaneous runs or a promise of recoverable speedup.

Comparing a representative steady step with 50 main transformer blocks:

| Timeline component | Native | ComfyUI |
| --- | ---: | ---: |
| Step window | 1,366 ms | 482 ms |
| Union of GPU kernel intervals | 929 ms | 471 ms |
| Copies/memsets with no overlapping kernel | 169 ms | 5 ms |
| No traced work from this process on GPU | 268 ms | 5 ms |
| Attention kernels, summed | 301 ms | 135 ms |
| Main INT8 GEMMs, summed | 217 ms | 233 ms |
| Host-to-device bytes | 7.26 GB | 4.64 GB |
| Device-to-device bytes | 4.86 GB | 0.13 GB |
| `cudaStreamSynchronize` calls | 461 | 7 |

The first three time components partition the step window; the attention/GEMM
rows are subsets, not additional time. The four measured steady windows agree
closely. Native boundaries use successive first-block RMS kernels; ComfyUI uses
successive model-forward submissions, including the following GPU completion
wait. Each selected window contains 50 main attention calls. Native has 4,350
tokens versus ComfyUI's 4,349, so this is a near-matched execution comparison,
not a bitwise-equivalent model test. ComfyUI activations are BF16; native's main
residual/FFN path is still FP32.

The largest actionable issue is **execution scheduling, not INT8 GEMM speed**.
Exposed copies plus idle gaps account for about 427 ms of the 885 ms steady-step
gap. Native stages pinned buffers synchronously before submitting the current
block, allocates/copies graph-cut outputs, and synchronizes at cache capture and
segment boundaries. Merely changing the VRAM limit or eviction order does not
make this an overlapped pipeline. Synchronization API duration includes actual
GPU work and must not be counted as wholly wasted time.

Attention accounts for another roughly 165 ms difference per step. Both kernels
use tensor cores, but the current ggml attention is slower than PyTorch Flash
Attention for this shape. The remaining GPU-time gap includes full FP32 memory
passes, scaling/layout operations and runtime LoRA projections. Native's scale
kernels alone take about 123 ms/step. ComfyUI merges/requantizes LoRA weights on
loading/faulting rather than executing two low-rank projections for every linear
forward. This does not mean merging is free or that every weight remains resident.

The ComfyUI "model compiler" log refers here to allocation management: the H3
path does **not** enable its per-block CUDA-graph replay, and this trace contains
no CUDA graph launches. CUDA graphs alone are therefore not the explanation.
Priority is reusable block buffers and event-based asynchronous staging, then
the attention kernel and end-to-end BF16/activation-quantization dataflow. VAE
remains a separate gap, including grouped tiles versus native serial tile work;
its individual causes have not yet been quantified by ablation.

Local evidence: `build/checks/comfy-diagnostic.{json,nsys-rep,sqlite}`,
`memory-safe.sqlite`, and `steady-step-comparison.json`. The local diagnostic
helpers `profile-comfy.py` and `compare-traces.py` are in the same ignored
directory; the former uses the existing synthetic workflow and ComfyUI install,
not a new runtime dependency of h3-go. No inference code changed in this audit.

### Earlier optimization rounds

2026-10-01, RTX 5090, synthetic red-teapot video, 640×384, 56 frames (2.333 seconds),
8 steps, CFG 1, seed 42, Turbo LoRA 1, identical native prompt and sampling settings.
Another GPU application remained resident (~12.2 GiB), and idle ComfyUI used ~0.8 GiB.
The idle Qwen service was temporarily stopped during generation tests and restored.
Do not benchmark concurrently with active generation in other applications.

| Warmed-filesystem native comparison | Before | Final |
| --- | ---: | ---: |
| Text conditioning | 1.61 s | 1.42 s |
| Sampling, including first-step preparation | 19.74 s | 15.27 s |
| Audio VAE | 0.61 s | 0.60 s |
| Video VAE | 4.61 s | 3.99 s |
| Whole native CLI process, excluding MP4 encoding | 27.60 s | 22.33 s |

The final measured reduction is **19.1% overall and 22.6% in sampling**. Steady
sampling improves from approximately 2.17 to 1.64 seconds per step. These are local
single-run warmed-filesystem observations, not a statistical performance guarantee.
The baseline and first-round pair ran adjacent to each other; the final run came
after rebuilding and warming the new implementation, with the same GPU budget and
background GPU applications. No model/prompt-result cache is used.

The first three changes alone were insufficient: 27.60 → 26.38 seconds (4.4%).
Profiling that version revealed substantial layout-copy/pointwise work and roughly
115 GB of host-to-device transfers over a complete request. The second round's
graph changes reached 23.52 seconds, and pinned prefetch staging reached 22.33.
The 115-GB trace describes the first-round version, not a measurement of the final
transfer volume; GPU operation totals can overlap and should not be summed as wall time.

Earlier less-warmed runs were 41.97 s before and 40.37 s after only the first round.
Do not attribute the entire 42→22-second difference to code: disk/page-cache warming
matters greatly. Disabling mmap was slower in a trial. Read-only whole-file host
registration was rejected by this driver, so that experiment was removed rather
than shipping a non-working optimization or locking tens of GiB of RAM.

The earlier ComfyUI/NVFP4 run took 12.49 seconds via its API (sampling 6.26 s,
video VAE 1.57 s). Its graph/model cache was cleared, but its process/kernel caches
and automatic VRAM policy differ from this native 16-GiB-budget process. This is
not a controlled encoder-only comparison. Native remains substantially slower;
runtime LoRA, segmented weight movement, FP32 intermediates, and unbatched VAE
work are still potential bottlenecks, not individually measured speedup claims.

Separate final direct Go integration tests passed: T2V 22.60 s and I2V 25.21 s
including MP4 encoding, both 640×384/56 frames with stereo 32-kHz AAC audio. They
are not the native-CLI performance comparison; the I2V prompt/input differ and Go
uses the C API directly rather than the upstream CLI's request preparation.
Same-seed outputs across different request paths/backends are not asserted to be
pixel-identical. The earlier first-round Go tests were T2V 30.46 s and I2V 29.51 s.
