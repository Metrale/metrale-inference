// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-14: GLM-5.3's clamped SwiGLU of one element, in FP32, shared by the element-wise
// entries glm5next_swiglu_clamp[_f32out] (glm5next_ffn.cu) and the fused routed gate/up epilogue
// (w4a4_grouped_mx.cu, w4a4_grouped_gu_*), so both compute the same expression on the same
// inputs: silu(min(gate, limit)) * clamp(up, -limit, limit).
//
// Owner: gb10 kernels.
// Invariants: every includer builds with the same nvcc flags (--fmad=false; the common tree's
// KERNEL.toml), so the one expression rounds the same way everywhere.
#pragma once

__device__ __forceinline__ float glm5next_swiglu_clamp_f32(float g, float u, const float limit) {
    g = fminf(g, limit);
    u = fminf(fmaxf(u, -limit), limit);
    float s = g / (1.0f + expf(-g));
    return s * u;
}
