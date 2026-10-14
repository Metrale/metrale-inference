// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-14: The softmax tail of the one-warp-per-row top-k (`moe_topk_softmax_rows_warp` in
// moe_topk.cu), shared with the Hopper fused router (hopper/common/moe_route_fused_hopper.cu) so
// both write a row's weights from one copy of the arithmetic.
//
// Owner: gb10 kernels.
// Invariants:
// - Lane l holds the logits of experts l + 32 j (j < TOPK_WARP_SLOTS) in v[j], with every picked
//   expert's slot already set to -1e30; top_vals / top_idxs hold the `rounds` picks in pick order
//   (written before a __syncwarp when lanes other than 0 wrote them) and global_max is the first
//   pick's value.
// - The sum keeps the 256-thread block body's tree (moe_topk.cu `topk_softmax_lowidx_block`): lane
//   l plays thread 32 w + l of every body warp w (its slots are w and w + 8), the shuffle-down
//   steps run in the body's order, and lane 0 adds the 8 warp sums and then the picks' terms, in
//   order, as thread 0 does. Lane 0 alone writes the outputs.

#pragma once

#define TOPK_WARP_MAX_EXPERTS 512
#define TOPK_WARP_SLOTS (TOPK_WARP_MAX_EXPERTS / 32)
#define TOPK_WARP_BODY_WARPS 8

__device__ __forceinline__ void topk_warp_softmax_tail(
    const float (&v)[TOPK_WARP_SLOTS],
    unsigned int lane,
    unsigned int actual_n,
    unsigned int rounds,
    float global_max,
    float* top_vals,
    const unsigned int* top_idxs,
    unsigned int normalize,
    unsigned int* idx_out,
    float* w_out
) {
    float warp_sum[TOPK_WARP_BODY_WARPS];
    #pragma unroll
    for (unsigned int w = 0; w < TOPK_WARP_BODY_WARPS; w++) {
        float local_sum = 0.0f;
        #pragma unroll
        for (unsigned int j = w; j < TOPK_WARP_SLOTS; j += TOPK_WARP_BODY_WARPS) {
            if (lane + 32 * j < actual_n) local_sum += __expf(v[j] - global_max);
        }
        warp_sum[w] = local_sum;
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        #pragma unroll
        for (unsigned int w = 0; w < TOPK_WARP_BODY_WARPS; w++) {
            warp_sum[w] += __shfl_down_sync(0xFFFFFFFF, warp_sum[w], offset);
        }
    }

    if (lane == 0) {
        float exp_sum = 0.0f;
        #pragma unroll
        for (unsigned int w = 0; w < TOPK_WARP_BODY_WARPS; w++) {
            exp_sum += warp_sum[w];
        }
        for (unsigned int t = 0; t < rounds; t++) {
            exp_sum += __expf(top_vals[t] - global_max);
        }
        for (unsigned int t = 0; t < rounds; t++) {
            idx_out[t] = top_idxs[t];
            float softmax_weight = __expf(top_vals[t] - global_max) / exp_sum;
            if (normalize) {
                top_vals[t] = softmax_weight;
            } else {
                w_out[t] = softmax_weight;
            }
        }
        if (normalize) {
            float topk_sum = 0.0f;
            for (unsigned int t = 0; t < rounds; t++) {
                topk_sum += top_vals[t];
            }
            for (unsigned int t = 0; t < rounds; t++) {
                w_out[t] = top_vals[t] / topk_sum;
            }
        }
    }
}
