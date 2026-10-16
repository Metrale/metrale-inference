// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-15: The cp.async ring and the per-group finish of glm5next_mhc_tile.cu (hc_mult 4,
// four tokens per 256-thread group), moved here unchanged so glm5next_mhc_post_tile.cu's ring
// entry points run tile4's ring, trees and tails on the streams they have just written.
//
// Owner: gb10 kernels (family glm_mhc).
// Invariants: glm5next_mhc_tile.cu's. The layout, the ring stage and the finish are tile4's.

#pragma once
#include "e4m3_mma_pipe.cuh"
#include "glm5next_mhc_fused_parts.cuh"
#include "glm5next_mhc_tile_parts.cuh"

#ifndef GLM_HCT_STAGES
#define GLM_HCT_STAGES 4
#endif

#define GLM_HCT_TOK 4

// 2026-10-15: Shared memory, in bytes from the dynamic base: the ring (stage s: hc_fn columns
// [MIX][256] BF16, then the block's stream columns [4 NG][256] FP32), which the trees then reuse
// as red[NG][MIX + 1][256] and the tails as y[NG][H]; then mix[4 NG][MIX], pre[NG][HC] and
// ws[NG][33] (FP32).
template <int NG>
struct GlmHctLayout {
    static constexpr unsigned int W_BYTES = GLM_HCT_MIX * GLM_HCF_BLOCK * 2;
    static constexpr unsigned int V_BYTES = GLM_HCT_TOK * NG * GLM_HCF_BLOCK * 4;
    static constexpr unsigned int STAGE = W_BYTES + V_BYTES;
    static constexpr unsigned int RING = GLM_HCT_STAGES * STAGE;
    static constexpr unsigned int RED = NG * (GLM_HCT_MIX + 1) * GLM_HCF_BLOCK * 4;
    static constexpr unsigned int Y = NG * GLM_HCF_MAX_H * 2;
    static constexpr unsigned int A = RING > RED ? RING : RED;
    static constexpr unsigned int REUSE = A > Y ? A : Y;
    static constexpr unsigned int MIX_OFF = REUSE;
    static constexpr unsigned int PRE_OFF = MIX_OFF + GLM_HCT_TOK * NG * GLM_HCT_MIX * 4;
    static constexpr unsigned int WS_OFF = PRE_OFF + NG * GLM_HCT_HC * 4;
    static constexpr unsigned int BYTES = WS_OFF + NG * 33 * 4;
};

// 2026-10-15: One ring stage: the hc_fn columns base .. base + 255 of every mix row and the same
// stream columns of the block's tokens; a token at or past `ntb` gets zeros (no read).
template <int NG>
__device__ __forceinline__ void glm_hct_issue(unsigned char* stage, const float* xb,
                                              const float* streams,
                                              const __nv_bfloat16* hc_fn, unsigned int hc_dim,
                                              unsigned int base, unsigned int ntb) {
    constexpr unsigned int NT = NG * GLM_HCF_BLOCK;
    constexpr unsigned int W_PIECES = GLM_HCT_MIX * GLM_HCF_BLOCK / 8;
    constexpr unsigned int V_PIECES = GLM_HCT_TOK * NG * GLM_HCF_BLOCK / 4;
    const unsigned int sw = e4m3g::smem_addr(stage);
    const unsigned int sv = sw + GlmHctLayout<NG>::W_BYTES;
    for (unsigned int i = threadIdx.x; i < W_PIECES; i += NT) {
        const unsigned int r = i / (GLM_HCF_BLOCK / 8), c = (i % (GLM_HCF_BLOCK / 8)) * 8;
        e4m3g::cp16(sw + (r * GLM_HCF_BLOCK + c) * 2, hc_fn + (size_t)r * hc_dim + base + c, true);
    }
    for (unsigned int i = threadIdx.x; i < V_PIECES; i += NT) {
        const unsigned int q = i / (GLM_HCF_BLOCK / 4), c = (i % (GLM_HCF_BLOCK / 4)) * 4;
        const bool ok = q < ntb;
        e4m3g::cp16(sv + (q * GLM_HCF_BLOCK + c) * 4,
                    ok ? (const void*)(xb + (size_t)q * hc_dim + base + c) : (const void*)streams,
                    ok);
    }
    e4m3g::cp_commit();
}

// 2026-10-15: Group G's four tokens: the trees, then the tails. `acc` and `ss` are thread t's
// chains (tok4's names); red, mix, pre, ws and y are the group's slices.
template <int NG, int G>
__device__ __forceinline__ void glm_hct_group_finish(
    float (&ss)[GLM_HCT_TOK], float (&acc)[GLM_HCT_TOK][GLM_HCT_MIX], unsigned char* smem,
    unsigned int t, unsigned int tg0, unsigned int ntg, unsigned int hc_dim,
    const float* xg, GLM_HCF_ARGS) {
    using L = GlmHctLayout<NG>;
    constexpr unsigned int MIX = GLM_HCT_MIX;
    float (*red)[GLM_HCF_BLOCK] = reinterpret_cast<float (*)[GLM_HCF_BLOCK]>(smem) + G * (MIX + 1);
    float (*s_mix)[MIX] =
        reinterpret_cast<float (*)[MIX]>(smem + L::MIX_OFF) + G * GLM_HCT_TOK;
    float* s_pre = reinterpret_cast<float*>(smem + L::PRE_OFF) + G * GLM_HCT_HC;
    float* s_ws = reinterpret_cast<float*>(smem + L::WS_OFF) + G * 33;
    __nv_bfloat16* s_y = reinterpret_cast<__nv_bfloat16*>(smem) + G * GLM_HCF_MAX_H;
    const unsigned int H = hidden_size;

    // 2026-10-15: tok4's trees, token by token, on the group's barrier.
#pragma unroll
    for (int q = 0; q < GLM_HCT_TOK; ++q)
        glm_hct_tree<G>(ss[q], acc[q], red, s_mix[q], mix, tg0 + q, (unsigned int)q < ntg, t,
                        hc_dim, norm_eps);

    // 2026-10-15: tok4's tails. y (the group's slice of the reused ring) overlaps `red`, whose
    // last read precedes the barrier above.
    for (unsigned int q = 0; q < ntg; ++q) {
        const unsigned int tk = tg0 + q;
        glm_hct_pre(s_mix[q], s_pre, t, hc_scale, hc_base, hc_eps);
        glm_hct_group_bar<G>();
        if (t < 32) {
            glm_hcf_sinkhorn<GLM_HCT_HC>(s_mix[q], hc_scale, hc_base, post_out, comb_out, tk,
                                         sinkhorn_iters, hc_eps);
        } else {
            glm_hcf_collapse_norm_at<GLM_HCT_HC, false, 4 + G>(
                t - 32u, xg + (size_t)q * hc_dim, s_pre, norm_w, y_out + (size_t)tk * H,
                normed_out + (size_t)tk * H, s_y, s_ws, H, rms_eps);
        }
        glm_hct_group_bar<G>();
    }
}

#define GLM_HCT_PASS                                                                          \
    streams, hc_fn, hc_scale, hc_base, mix, done, norm_w, y_out, post_out, comb_out,          \
        normed_out, hidden_size, hc_mult, sinkhorn_iters, norm_eps, hc_eps, rms_eps
