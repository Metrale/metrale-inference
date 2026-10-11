// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-28: W8A8 decode projections for 1..128 rows per launch on the WxAy skinny engine (wxay_engine.cuh): E4M3 weights times
// E4M3 activations, FP32 accumulation on `mma.sync.m16n8k32.e4m3`, BF16 output. Two policies:
//   rowscale:  C[m, n] = bf16( S[m, n] * (w_scale[n] * a_scale[m]) ),     S = sum_k A[m, k] * W[n, k]
//   blk128:    C[m, n] = bf16( sum_c S_c[m, n] * (w_scale[n / 128, c] * a_scale[m, c]) ), S_c over k in [128c, 128c + 128)
// A [M, lda] E4M3 (the first K columns), a_scale [M] (rowscale) or [M, K / 128] (blk128) FP32; C [M, ldc] BF16 (the
// first N columns are written).
//
// The weight is up to three row segments stacked into one output (attention Q|K|V, GDN QKV|Z, FFN gate|up), so a
// checkpoint's separate tensors are read in place with no concatenated copy: output rows [0, n1) come from (W0, S0),
// [n1, n2) from (W1, S1), [n2, N) from (W2, S2), rows and scales indexed from each segment's own start. One segment
// passes n1 = n2 = N.
//
// Weights sit on the MMA's m16 side, tokens on n8 (MB tiles). The K permutation is shared by both operands: thread t
// owns bytes [16t, 16t + 16) and [64 + 16t, 64 + 16t + 16) of every 128-byte chunk, so each load is 64 contiguous
// bytes per row and no shuffles are needed (the scales sit outside the k sum, or fold per chunk).
//
// Owner: gb10 kernels.
// Invariants:
// - K % 128 == 0, lda % 16 == 0, A and every W 16-byte aligned; 1 <= M <= 8 * MB (MB 16: 128 rows). n1 and n2 are multiples of 16
//   (rowscale) or 128 (blk128) with 0 < n1 <= n2 <= N. Only C[m, n] with m < M, n < N is written.
// - Grid (ceil(N / 16), 1, 1), block (256, 1, 1) (the engine's geometry).

#include "w8a8_policy.cuh"

#define W8A8_TPL(NAME, BLK, LAZY, MB, KU)                                                                          \
    extern "C" __global__ __launch_bounds__(256) void NAME(                                                        \
        const unsigned char* __restrict__ Aq, const float* __restrict__ As, const unsigned char* __restrict__ W0, \
        const float* __restrict__ S0, const unsigned char* __restrict__ W1, const float* __restrict__ S1,         \
        const unsigned char* __restrict__ W2, const float* __restrict__ S2, __nv_bfloat16* __restrict__ C,        \
        unsigned M, unsigned N, unsigned K, unsigned lda, unsigned ldc, unsigned n1, unsigned n2) {               \
        const w8a8::Args a = {Aq, As, {W0, W1, W2}, {S0, S1, S2}, n1, n2, lda, ldc};                              \
        wxay::mma_gemv<w8a8::W8A8<BLK, LAZY>, MB, KU>(a, C, M, N, K);                                              \
    }

// 2026-09-28: (MB, KU) per token-tile width. KU does not change the arithmetic (a KU2 twin of the MB 1 entry was
// measured bit-identical, and no faster at any decode shape on dgx1, so it is not built).
W8A8_TPL(w8a8_gemv_rowscale_mb1_ku8, false, false, 1, 8)
W8A8_TPL(w8a8_gemv_rowscale_mb2, false, false, 2, 4)
W8A8_TPL(w8a8_gemv_rowscale_mb4, false, true, 4, 4)
W8A8_TPL(w8a8_gemv_rowscale_mb8, false, true, 8, 2)
W8A8_TPL(w8a8_gemv_rowscale_mb16, false, true, 16, 2)
W8A8_TPL(w8a8_gemv_blk128_mb1_ku8, true, false, 1, 8)
W8A8_TPL(w8a8_gemv_blk128_mb2, true, false, 2, 4)
W8A8_TPL(w8a8_gemv_blk128_mb4, true, true, 4, 4)
W8A8_TPL(w8a8_gemv_blk128_mb8, true, true, 8, 2)
W8A8_TPL(w8a8_gemv_blk128_mb16, true, true, 16, 1)
