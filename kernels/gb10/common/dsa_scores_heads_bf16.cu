// SPDX-License-Identifier: MIT OR Apache-2.0
// metrale: instantiates dsa_indexer (dsa_scores_heads.cuh)

// 2026-10-23: The persisted pool keys stored as BF16 (`METRALE_GLM_DSA_POOL_KEYS_BF16`): the
// store and the two lane-per-head scores walks over them, so the captured decode / verify
// selection reads half the key bytes.
//
// Owner: gb10 kernels.
// Invariants:
// - dsa_pool_keys_store_rows_bf16 writes __float2bfloat16_rn of the key dsa_pool_keys_store_rows
//   writes, in the first half of the same slot.
// - dsa_index_scores_heads_paged_bf16 / dsa_index_scores_heads_window_bf16 write the bytes the
//   FP32 entries write for the keys widened from BF16 (dsa_sh_key4), with their arguments and
//   launch (gate examples/glm5next_dsa_ceiling_scores_gate.rs).

#include "dsa_scores_heads.cuh"

extern "C" __global__ void dsa_pool_keys_store_rows_bf16(
    const int* __restrict__ pos,
    int pos0,
    const __nv_bfloat16* __restrict__ k_pool,
    const __nv_bfloat16* __restrict__ gate_pool,
    float* __restrict__ pk,
    const float* __restrict__ ape,
    unsigned int D,
    unsigned int KP,
    const int* __restrict__ bt,
    unsigned int bt_stride,
    unsigned int bs,
    unsigned int blk_elems,
    unsigned int pk_blk
) {
    dsa_pool_keys_store_body<true>(pos, pos0, k_pool, gate_pool, pk, ape, D, KP, bt, bt_stride,
                                   bs, blk_elems, pk_blk);
}

#define DSA_SH_BF16_PARAMS                                                                    \
    const float* __restrict__ q, const float* __restrict__ pk,                                \
        const float* __restrict__ weights, const int* __restrict__ bt,                        \
        const int* __restrict__ q_pos, float* __restrict__ out,                               \
        unsigned char* __restrict__ valid_cand, unsigned int Q, unsigned int P, unsigned int H, \
        unsigned int D, unsigned int KP, unsigned int S, unsigned int bs, unsigned int pk_blk, \
        float scale, const int* __restrict__ geom
#define DSA_SH_BF16_ARGS \
    q, pk, weights, bt, q_pos, out, valid_cand, Q, P, H, D, KP, S, bs, pk_blk, scale, geom

extern "C" __global__ void __launch_bounds__(256, 1)
    dsa_index_scores_heads_paged_bf16(DSA_SH_BF16_PARAMS) {
    dsa_sh_paged_walk<true>(DSA_SH_BF16_ARGS);
}

extern "C" __global__ void __launch_bounds__(256, 1)
    dsa_index_scores_heads_window_bf16(DSA_SH_BF16_PARAMS) {
    dsa_sh_window_walk<true>(DSA_SH_BF16_ARGS);
}
