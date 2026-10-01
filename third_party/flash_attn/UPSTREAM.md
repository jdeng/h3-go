# Standalone FlashAttention forward headers

Source: https://github.com/Dao-AILab/flash-attention/tree/060c9188beec3a8b62b33a3bfa6d5d2d44975fab
(release v2.8.3).

This directory contains only the eleven headers needed by the CUDA forward
kernel, copied from `csrc/flash_attn/src`, plus upstream `LICENSE` and `AUTHORS`.
The BSD-3-Clause license and source notices are retained. CUTLASS/CuTe comes from
the existing pinned `deps/cutlass` submodule; it is not copied here.

Local changes remove the ATen generator include and Philox state from `flash.h`.
In `flash_fwd_kernel.h`, dropout is rejected at compile time, and unused dropout
state is initialized to zero instead of calling PyTorch's RNG unpacker. There
are no changes to attention arithmetic. Training/backward/dropout are not exposed.

The GGML patches supply a narrow dense, noncausal adapter for head dimensions 64 and 128.
The adapter uses GGML's CUDA stream, conversion helpers and graph-planned scratch
space, and compiles just these two forward specializations. No PyTorch, Python, ATen, libtorch,
FlashAttention Python package, or upstream launch/extension code is required.
