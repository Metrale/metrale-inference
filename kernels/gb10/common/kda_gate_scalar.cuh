// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-10: The KDA gate and beta element functions, shared by their kernels (kda_gate.cu,
// kda_layer_ops.cu) and the fused epilogues of kda_gates_w8a8.cu, so every caller computes the
// same expression.
//
// Owner: gb10 kernels.
// Invariants:
// - kda_gate_scalar: lower_bound * sigmoid(decay * (g_raw + dt_bias)) with expf.
// - kda_beta_scalar: sigmoid(x) with __expf, as kda_sigmoid_bf16_f32 has always computed it.

#pragma once
#include <math.h>

__device__ __forceinline__ float kda_gate_scalar(float g_raw, float dt_bias,
                                                 float decay, float lower_bound) {
    return lower_bound * (1.0f / (1.0f + expf(-(decay * (g_raw + dt_bias)))));
}

__device__ __forceinline__ float kda_beta_scalar(float x) {
    return 1.0f / (1.0f + __expf(-x));
}
