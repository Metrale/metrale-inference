// SPDX-License-Identifier: MIT OR Apache-2.0
// metrale: instantiates glm_mhc (glm5next_mhc_c1_body.cuh)

// 2026-10-21: glm5next_hc_pre_norm for the decode widths (fewer than 64 tokens) at GLM-5.3's shape
// (hc_mult 4, H 4096). 2026-10-24: The body moved to glm5next_mhc_c1_body.cuh (its notes), this
// entry instantiates it over the stored streams row (glmc1::LoadStreams).
//
// Owner: gb10 kernels.
// Invariants:
// - Every output is byte-identical to glm5next_hc_pre_norm (glm5next_mhc_fused.cu), which is
//   byte-identical to the unfused hc_mix_bf16 -> hc_finish -> rms_norm_vanilla
//   (--fmad=false on every side). glm5next_hc_pre_c1_gate (gpu examples) checks y, post, comb and
//   the normed row byte for byte.
// - hc_mult == 4 and H == 4096, else the kernel returns without writing (the host never launches
//   it there). Grid (T, 24, 1), block (GLM_HCF_BLOCK, 1, 1); `done` as glm5next_hc_pre_norm's.

#include "glm5next_mhc_c1_body.cuh"

extern "C" __global__ void __launch_bounds__(GLM_HCF_BLOCK)
glm5next_hc_pre_norm_c1(GLM_HCF_ARGS) {
    glmc1::glm_hcc1_body(streams, hc_fn, hc_scale, hc_base, mix, done, norm_w, y_out, post_out,
                         comb_out, normed_out, hidden_size, hc_mult, sinkhorn_iters, norm_eps,
                         hc_eps, rms_eps, glmc1::LoadStreams{});
}
