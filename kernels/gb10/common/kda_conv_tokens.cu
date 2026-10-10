// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-09: The KDA conv over T consecutive tokens of ONE sequence (a prefill sub-chunk) in
// one launch, and the conv window advance past them: causal_conv1d_update_l2norm's body
// (causal_conv1d_l2norm_body.cuh) per token, on the window the per-token path holds then.
//
// Owner: gb10 kernels.
// Invariants:
// - Token t's output equals causal_conv1d_update_l2norm's for that token after tokens 0..t-1;
//   after causal_conv1d_window_advance the window equals the per-token path's after T tokens.

#include "causal_conv1d_l2norm_body.cuh"

// 2026-10-09: causal_conv1d_update_l2norm over T consecutive tokens of ONE sequence in one
// launch: grid (ceil(dim / 256), T), block 256; block (x, t) is the block (x, 0) the per-token
// path launches for token t. Token t's window before its shift is [x_{t-K} .. x_{t-1}] (K =
// d_conv, x_j the input row j as float, x_j for j < 0 the conv_state slot j + K), the window the
// per-token path's conv_state holds when token t arrives; each thread builds it in registers and
// the body shifts and convolves it there, so token t's output is the per-token path's. conv_state
// is only read: causal_conv1d_window_advance, launched after this, moves it past the T tokens.
// d_conv must be at most 4 (Glm5NextKdaConfig::validate).
extern "C" __global__ void causal_conv1d_update_l2norm_tokens(
    const float* __restrict__ conv_state,
    const __nv_bfloat16* __restrict__ new_input,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ bias,
    __nv_bfloat16* __restrict__ output,
    unsigned int dim,
    unsigned int d_conv,
    unsigned int qk_channels,
    unsigned int head_dim,
    float l2_eps
) {
    const unsigned int t = blockIdx.y;
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    float win[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (ch < dim) {
        for (unsigned int s = 0; s < d_conv && s < 4u; ++s) {
            const int j = (int)t - (int)d_conv + (int)s;
            win[s] = j >= 0 ? (float)new_input[(size_t)j * dim + ch]
                            : conv_state[(size_t)ch * d_conv + (unsigned int)(j + (int)d_conv)];
        }
    }
    causal_conv1d_update_l2norm_body(nullptr, new_input, weight, bias, output, 0u, t, true, dim,
                                     d_conv, qk_channels, head_dim, l2_eps, win);
}

// 2026-10-09: The conv_state the per-token path leaves after T tokens of new_input ([T, dim]):
// slot s holds x_{T-K+s} (from conv_state slot T + s when T - K + s < 0). Grid (ceil(dim / 256)),
// block 256; each thread moves its own channel, reading it whole before writing.
extern "C" __global__ void causal_conv1d_window_advance(
    float* __restrict__ conv_state,
    const __nv_bfloat16* __restrict__ new_input,
    unsigned int dim,
    unsigned int d_conv,
    unsigned int T
) {
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= dim) return;
    float* st = conv_state + (size_t)ch * d_conv;
    float nw[4];
    for (unsigned int s = 0; s < d_conv && s < 4u; ++s) {
        const int j = (int)T - (int)d_conv + (int)s;
        nw[s] = j >= 0 ? (float)new_input[(size_t)j * dim + ch] : st[j + (int)d_conv];
    }
    for (unsigned int s = 0; s < d_conv && s < 4u; ++s) st[s] = nw[s];
}

