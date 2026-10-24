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
#include "pdl.cuh"

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
    // 2026-10-21: PDL (pdl.cuh): every input is another kernel's output.
    pdl_enter();
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
    // 2026-10-21: PDL (pdl.cuh): every input is another kernel's output.
    pdl_enter();
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

// 2026-10-21: A copy of `src` into `dst` then causal_conv1d_window_advance on `dst`, in one
// launch: slot s of dst gets x_{T-K+s}, or src slot T + s when T - K + s < 0. With d_conv <= 4
// (Glm5NextKdaConfig::validate) every slot of dst is written, so dst's bytes equal the copy then
// the advance; src is only read. Same grid and block as causal_conv1d_window_advance.
extern "C" __global__ void causal_conv1d_window_advance_to(
    const float* __restrict__ src,
    float* __restrict__ dst,
    const __nv_bfloat16* __restrict__ new_input,
    unsigned int dim,
    unsigned int d_conv,
    unsigned int T
) {
    // 2026-10-21: PDL (pdl.cuh): every input is another kernel's output.
    pdl_enter();
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= dim) return;
    const float* st = src + (size_t)ch * d_conv;
    float* out = dst + (size_t)ch * d_conv;
    for (unsigned int s = 0; s < d_conv && s < 4u; ++s) {
        const int j = (int)T - (int)d_conv + (int)s;
        out[s] = j >= 0 ? (float)new_input[(size_t)j * dim + ch] : st[j + (int)d_conv];
    }
}


// 2026-10-13: Tokens per pass of causal_conv1d_update_l2norm_tokens_parts: each thread issues
// this many input loads before it convolves them, and one pair of barriers serves them all.
#define KDA_CONV_PARTS_PASS 8

// 2026-10-13: kda_pack_qkv_bf16 and causal_conv1d_update_l2norm_tokens in one launch, a run of
// tokens per block. Inputs are the three [T, part] projections (dim = 3 * part); channel ch reads
// part ch / part at column ch % part, the element kda_pack_qkv_bf16 copies to packed[t, ch], and
// writes that copy to `packed` as the pack does (the window advance and the verify record read
// it). Grid (ceil(dim / 256), G), block 256: block (x, y) runs tokens [y * span, (y + 1) * span),
// span = ceil(T / G), so G may come from the SM count; each token's bytes do not depend on G.
// Byte-identical to the pack then causal_conv1d_update_l2norm_tokens, token by token:
// - a thread's window holds the floats the per-token block builds (x_j = the BF16 input widened,
//   x_j for j < 0 the conv_state slot j + K), shifted and appended in the body's order;
// - acc, SiLU and the L2 are the body's expressions in its order (bias, then k = 0 .. K-1;
//   __expf; the shuffle-down warp sums; the four warp sums from base_warp, summed and rsqrtf'd by
//   threads 0 and head_dim, read back by every thread of the head);
// - the store is the body's __float2bfloat16 of the same float.
// Only the issue order changes: KDA_CONV_PARTS_PASS loads per thread in flight, each input read
// once (the per-token grid read each one d_conv + 1 times), two barriers per pass instead of per
// token. conv_state is only read; causal_conv1d_window_advance runs after this, on `packed`.
template <int K>
__device__ __forceinline__ void kda_conv_parts_body(
    const float* __restrict__ conv_state,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ bias,
    __nv_bfloat16* __restrict__ packed,
    __nv_bfloat16* __restrict__ output,
    unsigned int dim,
    unsigned int part,
    unsigned int T,
    unsigned int qk_channels,
    unsigned int head_dim,
    float l2_eps,
    float (*warp_sums)[KDA_CONV_PARTS_PASS][8]
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int ch = blockIdx.x * blockDim.x + tid;
    const bool ch_valid = ch < dim;
    const bool block_needs_l2 = blockIdx.x * blockDim.x < qk_channels;
    const unsigned int span = (T + gridDim.y - 1) / gridDim.y;
    const unsigned int t_begin = blockIdx.y * span;
    const unsigned int t_end = min(T, t_begin + span);
    if (t_begin >= t_end) return;

    const unsigned int pi = ch_valid ? ch / part : 0u;
    const unsigned int off = ch_valid ? ch - pi * part : 0u;
    const __nv_bfloat16* src = pi == 0u ? q : (pi == 1u ? k : v);

    float win[K];
    float w[K];
    float b0 = 0.0f;
    if (ch_valid) {
#pragma unroll
        for (int s = 0; s < K; ++s) {
            const int j = (int)t_begin - K + s;
            win[s] = j >= 0 ? (float)src[(size_t)j * part + off]
                            : conv_state[(size_t)ch * K + (unsigned int)(j + K)];
            w[s] = (float)weight[ch * K + s];
        }
        b0 = (bias != nullptr) ? bias[ch] : 0.0f;
    }

    const unsigned int warp_id = tid / 32;
    const unsigned int lane = tid % 32;
    const unsigned int head_in_block = tid / head_dim;
    const unsigned int base_warp = head_in_block * (head_dim / 32);
    unsigned int buf = 0;
    for (unsigned int t0 = t_begin; t0 < t_end; t0 += KDA_CONV_PARTS_PASS, buf ^= 1u) {
        __nv_bfloat16 xin[KDA_CONV_PARTS_PASS];
#pragma unroll
        for (int p = 0; p < KDA_CONV_PARTS_PASS; ++p) {
            const unsigned int t = t0 + p;
            xin[p] = (ch_valid && t < t_end) ? src[(size_t)t * part + off] : __ushort_as_bfloat16(0);
        }
        float silu[KDA_CONV_PARTS_PASS];
#pragma unroll
        for (int p = 0; p < KDA_CONV_PARTS_PASS; ++p) {
            const unsigned int t = t0 + p;
            const bool valid = ch_valid && t < t_end;
            silu[p] = 0.0f;
            if (valid) {
                packed[(size_t)t * dim + ch] = xin[p];
#pragma unroll
                for (int i = 0; i < K - 1; ++i) win[i] = win[i + 1];
                win[K - 1] = (float)xin[p];
                float acc = b0;
#pragma unroll
                for (int i = 0; i < K; ++i) acc += win[i] * w[i];
                float sigmoid_acc = 1.0f / (1.0f + __expf(-acc));
                silu[p] = acc * sigmoid_acc;
            }
        }
        if (block_needs_l2) {
#pragma unroll
            for (int p = 0; p < KDA_CONV_PARTS_PASS; ++p) {
                float sq = (ch_valid && t0 + p < t_end) ? (silu[p] * silu[p]) : 0.0f;
                for (int offset = 16; offset >= 1; offset >>= 1)
                    sq += __shfl_down_sync(0xFFFFFFFF, sq, offset);
                if (lane == 0) warp_sums[buf][p][warp_id] = sq;
            }
            __syncthreads();
            if (tid < KDA_CONV_PARTS_PASS * 2) {
                const unsigned int p = tid / 2;
                const unsigned int bw = (tid % 2) * (head_dim / 32);
                if (bw + 3 < 8) {
                    float* ws = warp_sums[buf][p];
                    float total = ws[bw] + ws[bw + 1] + ws[bw + 2] + ws[bw + 3];
                    ws[bw] = rsqrtf(total + l2_eps);
                }
            }
            __syncthreads();
#pragma unroll
            for (int p = 0; p < KDA_CONV_PARTS_PASS; ++p)
                if (ch_valid && t0 + p < t_end) silu[p] *= warp_sums[buf][p][base_warp];
        }
#pragma unroll
        for (int p = 0; p < KDA_CONV_PARTS_PASS; ++p) {
            const unsigned int t = t0 + p;
            if (ch_valid && t < t_end) output[(size_t)t * dim + ch] = __float2bfloat16(silu[p]);
        }
    }
}

extern "C" __global__ void __launch_bounds__(256) causal_conv1d_update_l2norm_tokens_parts(
    const float* __restrict__ conv_state,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ bias,
    __nv_bfloat16* __restrict__ packed,
    __nv_bfloat16* __restrict__ output,
    unsigned int dim,
    unsigned int part,
    unsigned int T,
    unsigned int d_conv,
    unsigned int qk_channels,
    unsigned int head_dim,
    float l2_eps
) {
    __shared__ float warp_sums[2][KDA_CONV_PARTS_PASS][8];
#define KDA_CONV_PARTS_PASS_ARGS                                                              \
    conv_state, q, k, v, weight, bias, packed, output, dim, part, T, qk_channels, head_dim,   \
        l2_eps, warp_sums
    switch (d_conv) {
    case 1: kda_conv_parts_body<1>(KDA_CONV_PARTS_PASS_ARGS); break;
    case 2: kda_conv_parts_body<2>(KDA_CONV_PARTS_PASS_ARGS); break;
    case 3: kda_conv_parts_body<3>(KDA_CONV_PARTS_PASS_ARGS); break;
    case 4: kda_conv_parts_body<4>(KDA_CONV_PARTS_PASS_ARGS); break;
    default: break;
    }
#undef KDA_CONV_PARTS_PASS_ARGS
}

// 2026-10-14: causal_conv1d_window_advance reading the conv inputs where they lie instead of the
// packed rows: channel ch is element ch % part of projection ch / part (q, k, v), rows
// in_stride apart (the three parts: in_stride = part; the packed rows: q = the row start,
// k = q + part, v = q + 2 part, in_stride = dim). Byte-identical to the packed advance; the
// chunked prefill whose conv runs inside kda_chunk64_prepare_conv advances the window with it
// after the last slab (the prepares read the window before it moves).
extern "C" __global__ void causal_conv1d_window_advance_parts(
    float* __restrict__ conv_state,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    unsigned int dim,
    unsigned int part,
    unsigned int in_stride,
    unsigned int d_conv,
    unsigned int T
) {
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= dim) return;
    const unsigned int pi = ch / part, off = ch - pi * part;
    const __nv_bfloat16* src = (pi == 0u ? q : (pi == 1u ? k : v)) + off;
    float* st = conv_state + (size_t)ch * d_conv;
    float nw[4];
    for (unsigned int s = 0; s < d_conv && s < 4u; ++s) {
        const int j = (int)T - (int)d_conv + (int)s;
        nw[s] = j >= 0 ? (float)src[(size_t)j * in_stride] : st[j + (int)d_conv];
    }
    for (unsigned int s = 0; s < d_conv && s < 4u; ++s) st[s] = nw[s];
}
