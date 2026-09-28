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

#include "wxay_engine.cuh"

namespace w8a8 {

using wxay::Lane;
using u8 = unsigned char;

struct Args {
    const u8* Aq;
    const float* As;
    const u8* W[3];
    const float* S[3];
    unsigned n1, n2, lda, ldc;
};

// 2026-09-28: The segment holding output row n (a tile never straddles two): its index and first output row.
__device__ __forceinline__ unsigned seg_of(const Args& a, unsigned n, unsigned& base) {
    const unsigned s = n >= a.n2 ? 2u : (n >= a.n1 ? 1u : 0u);
    base = s == 2u ? a.n2 : (s == 1u ? a.n1 : 0u);
    return s;
}

__device__ __forceinline__ uint4 ld_nc(const u8* p) { return __ldg(reinterpret_cast<const uint4*>(p)); }

template <bool BLK, bool LAZY>
struct W8A8 {
    using Args = w8a8::Args;
    static constexpr unsigned ROWS = 16;
    static constexpr bool LAZY_A = LAZY;
    struct Tile {
        const u8 *w0, *w1;
        const float* ws;
        bool l0, l1;
    };
    struct Tok {
        const u8* aq;
        const float *s0, *s1;
        bool live, l0, l1;
    };
    struct WReg {
        uint4 l0, l1, h0, h1;
        float ws;
    };
    struct AReg {
        uint4 b0, b1;
        float s0, s1;
    };
    using WFrag = WReg;

    __device__ static Tile tile(const Args& a, unsigned n0, const Lane& L, unsigned N, unsigned K) {
        unsigned base;
        const unsigned s = seg_of(a, n0, base);
        const unsigned lr = n0 - base;
        Tile t;
        t.l0 = n0 + L.g < N;
        t.l1 = n0 + L.g + 8u < N;
        t.w0 = a.W[s] + (unsigned long long)(lr + L.g) * K + L.t * 16u;
        t.w1 = a.W[s] + (unsigned long long)(lr + L.g + 8u) * K + L.t * 16u;
        t.ws = a.S[s] + (unsigned long long)(lr >> 7) * (K >> 7);
        return t;
    }
    __device__ static Tok tok(const Args& a, unsigned j, const Lane& L, unsigned M, unsigned K) {
        const unsigned tk = j * 8u + L.g, c0 = j * 8u + L.t * 2u;
        Tok t;
        t.live = tk < M;
        t.aq = a.Aq + (unsigned long long)tk * a.lda + L.t * 16u;
        t.l0 = c0 < M;
        t.l1 = c0 + 1u < M;
        t.s0 = a.As + (unsigned long long)c0 * (K >> 7);
        t.s1 = t.s0 + (K >> 7);
        return t;
    }
    __device__ static WReg load_w(const Tile& t, unsigned c, bool live) {
        const uint4 z = make_uint4(0u, 0u, 0u, 0u);
        const bool a = live && t.l0, b = live && t.l1;
        WReg w;
        w.l0 = a ? __ldcs(reinterpret_cast<const uint4*>(t.w0 + c * 128u)) : z;
        w.l1 = a ? __ldcs(reinterpret_cast<const uint4*>(t.w0 + c * 128u + 64u)) : z;
        w.h0 = b ? __ldcs(reinterpret_cast<const uint4*>(t.w1 + c * 128u)) : z;
        w.h1 = b ? __ldcs(reinterpret_cast<const uint4*>(t.w1 + c * 128u + 64u)) : z;
        w.ws = (BLK && live) ? __ldg(t.ws + c) : 0.0f;
        return w;
    }
    __device__ static AReg load_a(const Tok& t, unsigned c, bool live) {
        const uint4 z = make_uint4(0u, 0u, 0u, 0u);
        AReg b;
        b.b0 = (live && t.live) ? ld_nc(t.aq + c * 128u) : z;
        b.b1 = (live && t.live) ? ld_nc(t.aq + c * 128u + 64u) : z;
        b.s0 = (BLK && live && t.l0) ? __ldg(t.s0 + c) : 0.0f;
        b.s1 = (BLK && live && t.l1) ? __ldg(t.s1 + c) : 0.0f;
        return b;
    }
    __device__ static WFrag prep_w(const WReg& w, const Lane&) { return w; }
    __device__ static void mma_frag(float (&acc)[4], const WFrag& w, const AReg& b, const Lane&) {
        if constexpr (BLK) {
            float s[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            chunk(s, w, b);
            const float f0 = w.ws * b.s0, f1 = w.ws * b.s1;
            acc[0] = __fmaf_rn(s[0], f0, acc[0]);
            acc[1] = __fmaf_rn(s[1], f1, acc[1]);
            acc[2] = __fmaf_rn(s[2], f0, acc[2]);
            acc[3] = __fmaf_rn(s[3], f1, acc[3]);
        } else {
            chunk(acc, w, b);
        }
    }
    __device__ static void mma_chunk(float (&acc)[4], const WReg& w, const AReg& b, const Lane& L) {
        mma_frag(acc, w, b, L);
    }
    // 2026-09-28: The four k32 MMAs of one chunk. l0/l1: row g, bytes [16t, 16t + 16) and [64 + 16t, ..); h0/h1 the
    // same for row g + 8; b0/b1 the token's matching bytes.
    __device__ static void chunk(float (&d)[4], const WReg& w, const AReg& b) {
        wxay::mma_e4m3_16832(d, w.l0.x, w.h0.x, w.l0.y, w.h0.y, b.b0.x, b.b0.y);
        wxay::mma_e4m3_16832(d, w.l0.z, w.h0.z, w.l0.w, w.h0.w, b.b0.z, b.b0.w);
        wxay::mma_e4m3_16832(d, w.l1.x, w.h1.x, w.l1.y, w.h1.y, b.b1.x, b.b1.y);
        wxay::mma_e4m3_16832(d, w.l1.z, w.h1.z, w.l1.w, w.h1.w, b.b1.z, b.b1.w);
    }
    __device__ static void store(const Args& a, __nv_bfloat16* C, const float (&r)[4], unsigned j, unsigned n0,
                                 const Lane& L, unsigned M, unsigned N) {
        unsigned base;
        const unsigned s = seg_of(a, n0, base);
#pragma unroll
        for (int c = 0; c < 4; c++) {
            const unsigned n = n0 + L.g + (c < 2 ? 0u : 8u);
            const unsigned tk = j * 8u + L.t * 2u + (unsigned)(c & 1);
            if (n >= N || tk >= M) continue;
            const float v = BLK ? r[c] : r[c] * (__ldg(a.S[s] + (n - base)) * __ldg(a.As + tk));
            C[(unsigned long long)tk * a.ldc + n] = __float2bfloat16_rn(v);
        }
    }
};

}

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
