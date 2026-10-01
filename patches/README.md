# Patch provenance and order

Base revisions are pinned in `dependencies.lock` and Git submodule entries.
These are adapted local patches, not verbatim PR merges or claims of upstream acceptance.
CPU/CUDA code is included; the original proposals' other backend changes are not included.

| Target | Patch | Source / purpose |
| --- | --- | --- |
| sd.cpp | `0001-pr2010-w4a8.patch` | [PR 2010](https://github.com/leejet/stable-diffusion.cpp/pull/2010), head `631907c311053846797d0dae7289734d437fb00f`; adapted W4 metadata/loading, logical dimensions, shared activation quantization and local compatibility fixes |
| sd.cpp | `0002-loader-wakeup.patch` | Local condition-variable completion notification instead of worker polling |
| sd.cpp | `0003-swiglu.patch` | Local contiguous F32 gate/up SwiGLU fusion, original fallback retained |
| sd.cpp | `0004-h3-text-cache.patch` | Request-local H3 condition projection/token-refiner output cache, invalidated by input shape/content and runner lifetime |
| sd.cpp | `0005-h3-graph-fusions.patch` | H3 RMSNorm/partial-RoPE integration with fallback, zero-copy QKV/modulation views, existing SwiGLU operator, removal of identity LoRA scaling |
| sd.cpp | `0006-pinned-prefetch.patch` | Reusable pinned-host staging for asynchronous weight prefetch, at most 512 MiB per buffer and 1 GiB pooled, with event-protected lifetimes and fallback |
| sd.cpp | `0007-h3-memory-dataflow.patch` | H3-scoped cyclic next-use weight eviction, fused per-modality RMSNorm/AdaLN and gated residual graphs, direct FP16 K/V attention storage, with original graph fallback |
| sd.cpp | `0008-h3-overlapped-block-pipeline.patch` | Asynchronous H3 block submission, deferred pinned weight uploads, compute-stream cut copies and bounded graph-cut buffer reuse |
| sd.cpp | `0009-h3-vae-decode.patch` | Bounded temporal chunk batching, flattened decoder GEMMs, SwiGLU/residual/RMS-RoPE integration, decode-local weight retention, contiguous CPU tile assembly and allocation-free overlap blending |
| sd.cpp | `0010-h3-sampling-dataflow.patch` | H3 CUDA BF16 normalization/FFN activation storage, direct INT8 quantization input, BF16 runtime-LoRA operands with FP32 accumulation, and removal of redundant INT8 scaling |
| GGML | `0001-pr10-w4a8.patch` | [PR 10](https://github.com/leejet/ggml/pull/10), head `025d7a5009ca59000652257cd3f992abdb3474e9`; adapted W4 CPU/CUDA operators and ConvRot fixes |
| GGML | `0002-cuda-fusions.patch` | Local RMSNorm + RoPE matcher/kernel and CUTLASS INT8 GEMM + scale/bias epilogue integration |
| GGML | `0003-h3-int8.patch` | H3-specific INT8 matrix eligibility on SM120 and independent H3 build-root configuration |
| GGML | `0004-h3-rms-rope.patch` | CUDA F32 RMSNorm + learned scale + non-interleaved partial RoPE + attention-layout kernel, with explicit CPU rejection and model-side unfused fallback |
| GGML | `0005-h3-block-dataflow.patch` | CUDA span-aware RMSNorm/modulation and gated residual operators with safe dead-buffer reuse, direct FP16 K/V storage, and FP32 rescale/LoRA-add/SwiGLU fusion with separate contiguous output |
| GGML | `0006-h3-flash-attention.patch` | Standalone FlashAttention-2 dense forward specialization for SM120/head-dimension-128, layout/type guards, existing GGML fallback and planner-visible scratch storage |
| GGML | `0007-h3-vae-attention.patch` | Extend the existing RMS/RoPE operator to unaffine 64-channel VAE heads and the standalone FlashAttention adapter to head dimension 64; preserve the original fallbacks |
| GGML | `0008-h3-bf16-dataflow.patch` | BF16-output RMS/AdaLN and SwiGLU, direct BF16 ConvRot row quantization, and fewer independent-group butterfly barriers; retain FP32 reductions, residuals, and existing operator signatures |

Apply in numbered order within each target, sd.cpp then GGML. `scripts/bootstrap.sh`
does this automatically and verifies both patch checksums and resulting source hashes.
Already patched trees are accepted only when all managed source hashes match.
Unexpected edits or revisions are never reset or overwritten; resolve them manually.
Patched submodules intentionally show dirty worktrees. The root stores patches, not fork commits.

The INT8 source is copied, not linked to a kitchen checkout; provenance and license
are in `third_party/fused_int8/`. Eligibility checks keep the tested SM120/shapes
on the fused path and leave other shapes on the existing implementation.

FlashAttention's BSD-3-Clause forward headers and retained notices are under
`third_party/flash_attn/`. The local adaptation removes the unused PyTorch RNG
dependency and statically forbids dropout; the forward arithmetic is unchanged.
The GGML adapter launches the specialization directly, without ATen/libtorch.

The earlier patches preserve the previously validated W4/Qwen integration as a base;
they do not make the H3 CLI depend on a Qwen service or another local repository.
The existing Qwen RoPE matcher does not cover H3's partial RoPE; the additional
H3 kernel handles that distinct layout. INT8 LoRA merge/requantization is not included.
Sampling dataflow changes round selected activations to BF16 and are not
pixel-identical to the preceding FP32 path. Matrix outputs and residual streams
remain FP32; this is not a complete BF16 conversion or activation/quantization fusion.
VAE overlap is configured to 0.25 by the C bridge and benchmark script, not by
changing global upstream defaults for other model families.
