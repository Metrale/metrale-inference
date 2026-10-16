// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-15: The shape and the argument list shared by the fused mHC post+pre kernels
// (glm5next_mhc_post_tile.cu, glm5next_mhc_post_ring.cu; host side in model-arch
// glm5next_mhc_post_tile.rs, which passes the arguments in this order).
//
// Owner: gb10 kernels (family glm_mhc).
// Invariants: hidden GLM_HCPT_H only; at most GLM_HCPT_MAX_RANKS ranks summed in registers.

#pragma once

#define GLM_HCPT_H 4096
#define GLM_HCPT_COLS (GLM_HCPT_H / GLM_HCF_BLOCK)
#define GLM_HCPT_MAX_RANKS 4

#define GLM_HCPT_ARGS                                                                         \
    __nv_bfloat16 *__restrict__ block_out, const __nv_bfloat16 *__restrict__ peers,           \
        long long slot_elems, int n_ranks, int my_rank, const float *residual,                \
        const float *post, const float *comb, float *streams_out, int write_x,                \
        const __nv_bfloat16 *__restrict__ hc_fn, const float *__restrict__ hc_scale,          \
        const float *__restrict__ hc_base, float *__restrict__ mix,                           \
        const __nv_bfloat16 *__restrict__ norm_w, __nv_bfloat16 *__restrict__ y_out,          \
        float *post_out, float *comb_out, __nv_bfloat16 *__restrict__ normed_out,             \
        unsigned int hidden_size, unsigned int sinkhorn_iters, float norm_eps, float hc_eps,  \
        float rms_eps, unsigned int num_tokens

#define GLM_HCPT_PASS                                                                         \
    block_out, peers, slot_elems, n_ranks, my_rank, residual, post, comb, streams_out,        \
        write_x, hc_fn, hc_scale, hc_base, mix, norm_w, y_out, post_out, comb_out,            \
        normed_out, hidden_size, sinkhorn_iters, norm_eps, hc_eps, rms_eps, num_tokens
