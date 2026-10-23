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

#include <cuda_bf16.h>

__device__ __forceinline__ float glm5next_swiglu_clamp_f32(float g, float u, const float limit) {
    g = fminf(g, limit);
    u = fminf(fmaxf(u, -limit), limit);
    float s = g / (1.0f + expf(-g));
    return s * u;
}

// 2026-10-23: Eight BF16 SwiGLU outputs from eight BF16 gate and up values (one 16-byte word of
// each): out[i] = bf16(glm5next_swiglu_clamp_f32(gate[i], up[i], limit)), rounded to nearest
// even as glm5next_swiglu_clamp's __float2bfloat16, so the word is that kernel's output bytes.
__device__ __forceinline__ uint4 glm5next_swiglu_clamp_bf16x8(uint4 g, uint4 u, const float limit) {
    const unsigned int gw[4] = {g.x, g.y, g.z, g.w};
    const unsigned int uw[4] = {u.x, u.y, u.z, u.w};
    unsigned int ow[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const float2 gf = __bfloat1622float2(*(const __nv_bfloat162*)&gw[i]);
        const float2 uf = __bfloat1622float2(*(const __nv_bfloat162*)&uw[i]);
        const __nv_bfloat162 o = __halves2bfloat162(
            __float2bfloat16(glm5next_swiglu_clamp_f32(gf.x, uf.x, limit)),
            __float2bfloat16(glm5next_swiglu_clamp_f32(gf.y, uf.y, limit)));
        ow[i] = *(const unsigned int*)&o;
    }
    return make_uint4(ow[0], ow[1], ow[2], ow[3]);
}
