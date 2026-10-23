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


// 2026-10-23: T consecutive tokens of ONE sequence (a K-row verify) through the conv, SiLU and
// L2 with every conv window step in the same launch: grid (ceil(dim / 256)), block 256, each
// thread carrying its channel's window in registers through the T tokens. Token t runs
// causal_conv1d_update_l2norm_body on the window after tokens 0..t-1 (the window
// causal_conv1d_update_l2norm_tokens builds for block (x, t): the same floats), so its output
// is that kernel's. After token t < nsnap (at most three) the window is stored to snap[t]
// (causal_conv1d_window_advance_to at T = t + 1, from the live window: the same floats), and
// after the last token to conv_state (causal_conv1d_window_advance at T). Each thread reads its
// window whole before writing it; no other thread touches that channel. d_conv <= 4.
__device__ __forceinline__ void causal_conv1d_update_l2norm_tokens_snap_body(
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
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    const bool live = ch < dim;
    float win[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (live) {
        for (unsigned int s = 0; s < d_conv && s < 4u; ++s) win[s] = conv_state[(size_t)ch * d_conv + s];
    }
    float* const snaps[3] = {snap0, snap1, snap2};
    for (unsigned int t = 0; t < T; ++t) {
        // 2026-10-23: The body reuses its shared warp sums: every thread has read token t - 1's
        // before token t writes them (block-uniform: no thread returns early).
        if (t > 0) __syncthreads();
        causal_conv1d_update_l2norm_body(nullptr, new_input, weight, bias, output, 0u, t, true, dim,
                                         d_conv, qk_channels, head_dim, l2_eps, win);
        if (live && t < nsnap && t < 3u) {
            for (unsigned int s = 0; s < d_conv && s < 4u; ++s)
                snaps[t][(size_t)ch * d_conv + s] = win[s];
        }
    }
    if (live) {
        for (unsigned int s = 0; s < d_conv && s < 4u; ++s) conv_state[(size_t)ch * d_conv + s] = win[s];
    }
}
