# Vendored INT8 CUDA kernels

Source: https://github.com/Comfy-Org/comfy-kitchen/tree/178050feb376efb17f5080aecc4161bee83fd390

Only these source files are copied from `comfy_kitchen/backends/cuda/ops/`:

- `cutlass_gemm_int8.cu`: unchanged upstream INT8 GEMM/dequant CUDA implementation.
- `cutlass_gemm_common.cuh`: upstream helpers, with the local external-workspace hook described below.

The upstream Apache-2.0 `LICENSE`, `NOTICE`, and source notices are retained here.
This directory is compiled directly as a CUDA object library. No comfy-kitchen checkout,
Python package, PyTorch, extension loader, or other kitchen source is needed.
CUTLASS is a separate pinned submodule at `deps/cutlass`.

The retained upstream NOTICE describes the full upstream project. Its references
to Python packages and other kernels do not add those dependencies to this two-file adaptation.

## Local modification

`cutlass_gemm_common.cuh` adds `COMFY_CUTLASS_EXTERNAL_WORKSPACE` and
`comfy_cutlass_external_workspace(size_t, cudaStream_t)`. When compiled with this
definition, temporary storage comes from GGML's stream-local memory pool instead
of the original persistent allocation cache. The adapter is in the GGML fusion patch.
The kernel arithmetic is unchanged. These are internal build definitions, not runtime switches.
