// SPDX-License-Identifier: MIT OR Apache-2.0
// metrale: instantiates causal_conv1d_l2norm (causal_conv1d_l2norm_body.cuh)

// 2026-10-23: causal_conv1d_update_l2norm_tokens_snap: a K-row verify's conv, its conv
// intermediates and the live window advance in one launch, in place of
// causal_conv1d_window_advance_to per intermediate, causal_conv1d_update_l2norm_tokens and
// causal_conv1d_window_advance (METRALE_GLM_KDA_CONV_ONE, crates/model-arch/src/glm5next_kda/
// seq_tokens.rs). Grid (ceil(dim / 256)), block 256.
//
// Owner: gb10 kernels.
// Invariants: every byte written equals the three kernels' (the template's comment,
// causal_conv1d_update_l2norm_tokens_snap_body); bias may be null.

#include "causal_conv1d_l2norm_body.cuh"
#include "pdl.cuh"

extern "C" __global__ void causal_conv1d_update_l2norm_tokens_snap(
    float* __restrict__ conv_state,
    const __nv_bfloat16* __restrict__ new_input,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ bias,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ snap0,
    float* __restrict__ snap1,
    float* __restrict__ snap2,
    unsigned int nsnap,
    unsigned int dim,
    unsigned int d_conv,
    unsigned int qk_channels,
    unsigned int head_dim,
    float l2_eps,
    unsigned int T
) {
    // 2026-10-23: PDL (pdl.cuh): every input is another kernel's output.
    pdl_enter();
    causal_conv1d_update_l2norm_tokens_snap_body(conv_state, new_input, weight, bias, output, snap0,
                                                 snap1, snap2, nsnap, dim, d_conv, qk_channels,
                                                 head_dim, l2_eps, T);
}
