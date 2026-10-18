// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-09: causal_conv1d_update_l2norm's body, moved unchanged from causal_conv1d.cu (which
// includes it where it stood) so kda_conv_tokens.cu runs the same code.
//
// Owner: gb10 kernels.
// Invariants: see causal_conv1d.cu's causal_conv1d_update_l2norm comment.

#pragma once

#include <cuda_bf16.h>

// 2026-10-09: The body of causal_conv1d_update_l2norm, for row `b` of the input and output and
// window `b_state` of `conv_state`; the entries below pick them.
__device__ __forceinline__ void causal_conv1d_update_l2norm_body(
    float* __restrict__ conv_state,
    const __nv_bfloat16* __restrict__ new_input,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ bias,
    __nv_bfloat16* __restrict__ output,
    unsigned int b_state,
    unsigned int b,
    bool row_valid,
    unsigned int dim,
    unsigned int d_conv,
    unsigned int qk_channels,
    unsigned int head_dim,
    float l2_eps,
    // 2026-10-09: When non-null, this thread's d_conv-float window to shift and convolve in
    // place of its window in conv_state (causal_conv1d_update_l2norm_tokens passes a private
    // copy); the entries that update conv_state pass nullptr.
    float* __restrict__ win
) {
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int tid = threadIdx.x;


    const unsigned int block_start = blockIdx.x * blockDim.x;
    const bool block_needs_l2 = (block_start < qk_channels);

    const bool valid = (ch < dim && row_valid);
    float silu = 0.0f;


    if (valid) {
        float* state = win != nullptr ? win : conv_state + (b_state * dim + ch) * d_conv;

        for (unsigned int i = 0; i < d_conv - 1; i++)
            state[i] = state[i + 1];
        state[d_conv - 1] = (float)new_input[b * dim + ch];

        const __nv_bfloat16* w = weight + ch * d_conv;
        float acc = (bias != nullptr) ? bias[ch] : 0.0f;
        for (unsigned int k = 0; k < d_conv; k++)
            acc += state[k] * (float)w[k];

        float sigmoid_acc = 1.0f / (1.0f + __expf(-acc));
        silu = acc * sigmoid_acc;
    }




    if (block_needs_l2) {
        float sq = valid ? (silu * silu) : 0.0f;


        const unsigned int warp_id = tid / 32;
        const unsigned int lane = tid % 32;
        for (int offset = 16; offset >= 1; offset >>= 1)
            sq += __shfl_down_sync(0xFFFFFFFF, sq, offset);


        __shared__ float warp_sums[8];
        if (lane == 0) warp_sums[warp_id] = sq;
        __syncthreads();



        const unsigned int head_in_block = tid / head_dim;
        const unsigned int base_warp = head_in_block * (head_dim / 32);


        if (tid == 0 || tid == head_dim) {
            float total = warp_sums[base_warp] + warp_sums[base_warp + 1]
                        + warp_sums[base_warp + 2] + warp_sums[base_warp + 3];
            warp_sums[base_warp] = rsqrtf(total + l2_eps);
        }
        __syncthreads();


        if (valid) {
            silu *= warp_sums[base_warp];
        }
    }

    if (valid) {
        output[b * dim + ch] = __float2bfloat16(silu);
    }
}

