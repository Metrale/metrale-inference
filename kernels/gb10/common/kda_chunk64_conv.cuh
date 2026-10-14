// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-14: The KDA depthwise conv, SiLU and q/k L2 norm of one (chunk, head) tile, run by
// kda_chunk64_prepare_conv (kda_chunk64_prepare.cu) ahead of its factors, so the conv output of
// a prefill sub-chunk never makes a DRAM round trip: the block reads its 64 rows (and the
// d_conv - 1 rows before them) of the q/k/v projections, writes the conv'd rows to a slab-local
// stage the rest of the prepare reads back from L2, and writes k straight into its shared K tile.
//
// Owner: gb10 kernels.
// Invariants:
// - Byte-identical to causal_conv1d_update_l2norm_tokens_parts (kda_conv_tokens.cu) for every
//   row t < T of head h, given the same conv_state, weight, bias and inputs:
//   - the window holds the floats that kernel builds (x_j the BF16 input widened; j < 0 reads
//     conv_state slot j + K), shifted and appended in its order;
//   - acc, SiLU and the L2 are its expressions in its order: bias, then products k = 0 .. K-1
//     (the common module builds with --fmad=false, so neither side contracts); __expf; a
//     shuffle-down tree over the 32 channels of each warp; the four warp sums of the head added
//     left to right and rsqrtf'd; one multiply; __float2bfloat16.
//   - That kernel's 256-channel blocks hold two whole heads (q|k channel count a multiple of
//     256), so its warps are this tile's warps channel for channel: q threads 0..127 and k
//     threads 128..255, lane = channel % 32. The launcher refuses other shapes.
// - v takes no L2 (v channels start at the q|k channel count, a 256 multiple, as there).
// - Rows t >= T write nothing to the stage and zero to the K tile (the zero-filled copy the
//   plain prepare takes there).
// - No input row t >= T is read (a v half whose rows all lie at or past T is skipped whole).

#pragma once

#include <cuda_bf16.h>

#include "kda_chunk64.cuh"

// 2026-10-14: Rows per conv pass: each thread holds this many inputs in flight and loads the next
// pass's before reducing this one; one pair of barriers serves the pass.
#define KDA64_CONV_PASS 8

template <int K>
__device__ __forceinline__ void k64_conv_window(
    float (&win)[K],
    float (&w)[K],
    float& b0,
    const __nv_bfloat16* __restrict__ src,
    unsigned int in_stride,
    const float* __restrict__ conv_state,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ bias,
    unsigned int ch,
    int t_first
) {
#pragma unroll
    for (int s = 0; s < K; ++s) {
        const int j = t_first - K + s;
        win[s] = j >= 0 ? (float)src[(size_t)j * in_stride]
                        : conv_state[(size_t)ch * K + (unsigned int)(j + K)];
        w[s] = (float)weight[ch * K + s];
    }
    b0 = (bias != nullptr) ? bias[ch] : 0.0f;
}

// 2026-10-14: causal_conv1d_update_l2norm_tokens_parts's per-token step: shift x in, convolve,
// SiLU.
template <int K>
__device__ __forceinline__ float k64_conv_step(
    float (&win)[K],
    const float (&w)[K],
    float b0,
    __nv_bfloat16 x
) {
#pragma unroll
    for (int i = 0; i < K - 1; ++i) win[i] = win[i + 1];
    win[K - 1] = (float)x;
    float acc = b0;
#pragma unroll
    for (int i = 0; i < K; ++i) acc += win[i] * w[i];
    float sigmoid_acc = 1.0f / (1.0f + __expf(-acc));
    return acc * sigmoid_acc;
}

// 2026-10-14: The tile (rows t0 .. t0 + 63 of head h). x{q,k,v} point at head h's first channel
// of each projection (rows in_stride apart); stage_{q,k,v} at head h's first channel of the
// block's stage rows (qkv_stride apart); ch_{q,k,v} are the head's first conv channels.
template <int K>
__device__ __forceinline__ void k64_conv_tile(
    const __nv_bfloat16* __restrict__ xq,
    const __nv_bfloat16* __restrict__ xk,
    const __nv_bfloat16* __restrict__ xv,
    unsigned int in_stride,
    const float* __restrict__ conv_state,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ bias,
    unsigned int ch_q,
    unsigned int ch_k,
    unsigned int ch_v,
    float l2_eps,
    __nv_bfloat16* stage_q,
    __nv_bfloat16* stage_k,
    __nv_bfloat16* stage_v,
    unsigned int qkv_stride,
    __nv_bfloat16* sK,
    unsigned int t0,
    unsigned int T
) {
    constexpr int C = KDA64_C, D = KDA64_D, P = KDA64_CONV_PASS;
    __shared__ float ws[2][P][KDA64_PREP_THREADS / 32];
    const unsigned int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
    const unsigned int d = tid & (unsigned int)(D - 1), pi = tid / (unsigned int)D;

    // 2026-10-14: q (pi 0) and k (pi 1), every row, the L2 per row and head.
    {
        const __nv_bfloat16* src = (pi == 0u ? xq : xk) + d;
        __nv_bfloat16* dst = (pi == 0u ? stage_q : stage_k) + d;
        const unsigned int ch = (pi == 0u ? ch_q : ch_k) + d;
        const unsigned int base_warp = pi * (unsigned int)(D / 32);
        float win[K], w[K], b0;
        k64_conv_window<K>(win, w, b0, src, in_stride, conv_state, weight, bias, ch, (int)t0);
        __nv_bfloat16 xa[P], xb[P];
#pragma unroll
        for (int p = 0; p < P; ++p) {
            const unsigned int t = t0 + (unsigned int)p;
            xa[p] = t < T ? src[(size_t)t * in_stride] : __ushort_as_bfloat16(0);
        }
#pragma unroll
        for (int p0 = 0; p0 < C; p0 += P) {
            const int buf = (p0 / P) & 1;
            if (p0 + P < C) {
#pragma unroll
                for (int p = 0; p < P; ++p) {
                    const unsigned int t = t0 + (unsigned int)(p0 + P + p);
                    xb[p] = t < T ? src[(size_t)t * in_stride] : __ushort_as_bfloat16(0);
                }
            }
            float silu[P];
#pragma unroll
            for (int p = 0; p < P; ++p) {
                const bool valid = t0 + (unsigned int)(p0 + p) < T;
                silu[p] = valid ? k64_conv_step<K>(win, w, b0, xa[p]) : 0.0f;
                float sq = valid ? (silu[p] * silu[p]) : 0.0f;
                for (int offset = 16; offset >= 1; offset >>= 1)
                    sq += __shfl_down_sync(0xFFFFFFFF, sq, offset);
                if (lane == 0) ws[buf][p][warp] = sq;
            }
            __syncthreads();
            if (tid < (unsigned int)(P * 2)) {
                const unsigned int p = tid / 2;
                const unsigned int bw = (tid % 2) * (unsigned int)(D / 32);
                float* s = ws[buf][p];
                float total = s[bw] + s[bw + 1] + s[bw + 2] + s[bw + 3];
                s[bw] = rsqrtf(total + l2_eps);
            }
            __syncthreads();
#pragma unroll
            for (int p = 0; p < P; ++p) {
                const unsigned int i = (unsigned int)(p0 + p);
                const bool valid = t0 + i < T;
                if (valid) silu[p] *= ws[buf][p][base_warp];
                const __nv_bfloat16 o = __float2bfloat16(silu[p]);
                if (valid) dst[(size_t)i * qkv_stride] = o;
                if (pi == 1u) sK[i * D + d] = valid ? o : __ushort_as_bfloat16(0);
            }
#pragma unroll
            for (int p = 0; p < P; ++p) xa[p] = xb[p];
        }
    }

    // 2026-10-14: v, no L2: threads 0..127 rows 0..31, threads 128..255 rows 32..63.
    // 2026-10-14: A half with no row below T is skipped: its window (rows t0 + r0 - K ..) would
    // read past the inputs' last row, and it writes nothing. No barrier follows in this pass.
    const unsigned int r0 = pi * (unsigned int)(C / 2);
    if (t0 + r0 < T) {
        const __nv_bfloat16* src = xv + d;
        float win[K], w[K], b0;
        k64_conv_window<K>(win, w, b0, src, in_stride, conv_state, weight, bias, ch_v + d,
                           (int)(t0 + r0));
#pragma unroll
        for (int p0 = 0; p0 < C / 2; p0 += P) {
            __nv_bfloat16 x[P];
#pragma unroll
            for (int p = 0; p < P; ++p) {
                const unsigned int t = t0 + r0 + (unsigned int)(p0 + p);
                x[p] = t < T ? src[(size_t)t * in_stride] : __ushort_as_bfloat16(0);
            }
#pragma unroll
            for (int p = 0; p < P; ++p) {
                const unsigned int i = r0 + (unsigned int)(p0 + p);
                if (t0 + i < T) {
                    const float silu = k64_conv_step<K>(win, w, b0, x[p]);
                    stage_v[(size_t)i * qkv_stride + d] = __float2bfloat16(silu);
                }
            }
        }
    }
}
