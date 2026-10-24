// SPDX-License-Identifier: MIT OR Apache-2.0
// metrale: instantiates glm_mhc (glm5next_mhc_c1_body.cuh)

// 2026-10-24: glm5next_hc_post_pre_norm_c1 (R3): glm5next_hc_post_rank_sum (the one-shot
// all-reduce's sum and the mHC post-mix of the attention site) and the FFN site's decode-width
// pre-norm glm5next_hc_pre_norm_c1 in one launch, at 1..2 tokens: the c1 body over
// glmc1::PostRankSum's stream values instead of the stored row.
//
// Owner: gb10 kernels.
// Invariants:
// - Every output byte equals glm5next_hc_post_rank_sum then glm5next_hc_pre_norm_c1 on the same
//   inputs: `residual` (the highway, updated in place) and `block_out` get what the post-sum
//   leaves, y/post/comb/normed what the pre-norm writes (glm5next_hc_post_pre_c1_gate, gpu
//   examples, checks every byte). `post`/`comb` may alias `post_out`/`comb_out`.
// - hc_mult == 4, H == 4096, n_ranks <= 8; grid (T, 24, 1), block (GLM_HCF_BLOCK, 1, 1).

#include "glm5next_mhc_c1_body.cuh"

extern "C" __global__ void __launch_bounds__(GLM_HCF_BLOCK)
glm5next_hc_post_pre_norm_c1(GLM_HCF_ARGS, __nv_bfloat16* block_out,
                             const __nv_bfloat16* peers, long long slot_elems, int n_ranks,
                             int my_rank, const float* post, const float* comb, float* residual) {
    const glmc1::PostRankSum src{block_out, peers, slot_elems, n_ranks, my_rank, residual, post, comb, residual};
    glmc1::glm_hcc1_body(streams, hc_fn, hc_scale, hc_base, mix, done, norm_w, y_out, post_out,
                         comb_out, normed_out, hidden_size, hc_mult, sinkhorn_iters, norm_eps,
                         hc_eps, rms_eps, src);
}
