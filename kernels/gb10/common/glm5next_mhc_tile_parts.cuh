// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-15: The per-token device parts of glm5next_mhc_tile.cu (hc_mult 4, 256-thread groups),
// moved here unchanged so glm5next_mhc_post_tile.cu runs the same trees and pre weights: one
// token's 256-wide RMS and mix trees on its group's named barrier, and its pre weights.
//
// Owner: gb10 kernels (family glm_mhc).
// Invariants: glm5next_mhc_tile.cu's. Every sum and every expression is
// glm5next_hc_pre_norm_tok4's (glm5next_mhc_tok.cu), in its order.

#pragma once
#include "glm5next_mhc_fused_parts.cuh"

#define GLM_HCT_HC 4
#define GLM_HCT_MIX ((2 + GLM_HCT_HC) * GLM_HCT_HC)

// 2026-10-15: Named barrier over the 256 threads of group G (barrier 2 + G); the tails of the
// group's warps 1..7 use barrier 4 + G (glm_hcf_collapse_norm_at, glm_hcf_norm_at).
template <int G>
__device__ __forceinline__ void glm_hct_group_bar() {
    asm volatile("bar.sync %0, %1;" ::"n"(2 + G), "n"(GLM_HCF_BLOCK));
}

// 2026-10-15: One token's trees on group G: thread t's chains `ss` and `acc` go through tok4's
// 256-wide tree in `red` ([MIX + 1][256]), then threads 0..MIX-1 scale the mix sums by the RMS
// and keep them in `s_mix`, and in `mix` row `row` when `store`.
template <int G>
__device__ __forceinline__ void glm_hct_tree(
    const float ss, const float (&acc)[GLM_HCT_MIX], float (*red)[GLM_HCF_BLOCK], float* s_mix,
    float* __restrict__ mix, const unsigned int row, const bool store, const unsigned int t,
    const unsigned int hc_dim, const float norm_eps) {
    constexpr unsigned int MIX = GLM_HCT_MIX;
    red[0][t] = ss;
#pragma unroll
    for (int r = 0; r < (int)MIX; ++r) red[1 + r][t] = acc[r];
    glm_hct_group_bar<G>();
    for (unsigned int s = GLM_HCF_BLOCK / 2; s > 0; s >>= 1) {
        if (t < s) {
#pragma unroll
            for (int r = 0; r <= (int)MIX; ++r) red[r][t] += red[r][t + s];
        }
        glm_hct_group_bar<G>();
    }
    if (t < MIX) {
        const float rsq = rsqrtf(red[0][0] / (float)hc_dim + norm_eps);
        const float m = red[1 + t][0] * rsq;
        s_mix[t] = m;
        if (store) mix[(size_t)row * MIX + t] = m;
    }
    glm_hct_group_bar<G>();
}

// 2026-10-15: One token's pre weights from its scaled mixes, on threads 0..GLM_HCT_HC-1 of the
// group; the caller's group barrier publishes them.
__device__ __forceinline__ void glm_hct_pre(const float* s_mix, float* s_pre,
                                            const unsigned int t,
                                            const float* __restrict__ hc_scale,
                                            const float* __restrict__ hc_base,
                                            const float hc_eps) {
    if (t < GLM_HCT_HC) {
        float pr = s_mix[t] * hc_scale[0] + hc_base[t];
        s_pre[t] = 1.f / (1.f + expf(-pr)) + hc_eps;
    }
}
