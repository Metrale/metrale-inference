// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-14: The two activations of the Qwen3.8-Flash-Next low-rank hyper-connection mixer,
// shared by its GB10 kernels (qwen3.8-flash-next/nvfp4/hyper_connection.cu) and the Hopper
// decode GEMVs (hopper/common/hc_lowrank_gemv_hopper.cu), so both round the same formula.
//
// Owner: gb10 kernels.
// Invariants: every caller applies these to FP32 values that it has already rounded as the
// reference does (BF16 GEMM outputs, scaled by 1 / hc where the formula says so).

#pragma once

__device__ __forceinline__ float qhc_silu(float v) {
    return v / (1.0f + __expf(-v));
}

__device__ __forceinline__ float qhc_sigmoid(float v) {
    return 1.0f / (1.0f + __expf(-v));
}
