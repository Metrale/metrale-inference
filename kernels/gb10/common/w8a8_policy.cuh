// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-10: The W8A8 rowscale and blk128 policies of the WxAy skinny engine (wxay_engine.cuh), moved here unchanged
// from w8a8_gemv.cu so a fused entry (kda_gates_w8a8.cu) runs the same loads, MMAs and scale fold. The formats and the
// operand layout are described at the top of w8a8_gemv.cu.
//
// Owner: gb10 kernels.
// Invariants: as w8a8_gemv.cu's.

#pragma once
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
        return tile_in(a.W[s], a.S[s], n0, n0 - base, L, N, K);
    }
    // 2026-10-21: The tile at output row n0, row lr of the segment (W, S) holding it.
    __device__ static Tile tile_in(const u8* W, const float* S, unsigned n0, unsigned lr, const Lane& L, unsigned N,
                                   unsigned K) {
        Tile t;
        t.l0 = n0 + L.g < N;
        t.l1 = n0 + L.g + 8u < N;
        t.w0 = W + (unsigned long long)(lr + L.g) * K + L.t * 16u;
        t.w1 = W + (unsigned long long)(lr + L.g + 8u) * K + L.t * 16u;
        t.ws = S + (unsigned long long)(lr >> 7) * (K >> 7);
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
        store_in(a.S[s], a.As, base, C, a.ldc, 0u, r, j, n0, L, M, N);
    }
    // 2026-10-21: Store token tile j of the tile at output row n0 of the segment starting at output row `base`
    // (scales S), output row n to column n - col0 of C (row pitch ldc).
    __device__ static void store_in(const float* S, const float* As, unsigned base, __nv_bfloat16* C, unsigned ldc,
                                    unsigned col0, const float (&r)[4], unsigned j, unsigned n0, const Lane& L,
                                    unsigned M, unsigned N) {
#pragma unroll
        for (int c = 0; c < 4; c++) {
            const unsigned n = n0 + L.g + (c < 2 ? 0u : 8u);
            const unsigned tk = j * 8u + L.t * 2u + (unsigned)(c & 1);
            if (n >= N || tk >= M) continue;
            const float v = BLK ? r[c] : r[c] * (__ldg(S + (n - base)) * __ldg(As + tk));
            C[(unsigned long long)tk * ldc + (n - col0)] = __float2bfloat16_rn(v);
        }
    }
};

// 2026-10-21: Up to four per-row-scaled weights of one K over one quantized activation, each into its own output
// (`Split4`): output rows [nb[s-1], nb[s]) are segment s (nb[-1] = 0, nb[3] = N), read from (W[s], S[s]) and
// written to column n - nb[s-1] of Cs[s] (row pitch ldcs[s]). Every tile is W8A8<false, false>'s tile on the same
// operands, so each output holds the bytes the segment's own launch writes; only the destination differs.
struct SplitArgs : Args {
    const u8* W4[4];
    const float* S4[4];
    __nv_bfloat16* Cs[4];
    unsigned ldcs[4];
    unsigned nb[3];
};

struct Split4 : W8A8<false, false> {
    using Args = SplitArgs;
    using Base = W8A8<false, false>;
    __device__ static unsigned seg4(const SplitArgs& a, unsigned n, unsigned& base) {
        const unsigned s = n >= a.nb[2] ? 3u : (n >= a.nb[1] ? 2u : (n >= a.nb[0] ? 1u : 0u));
        base = s == 0u ? 0u : a.nb[s - 1u];
        return s;
    }
    __device__ static Tile tile(const SplitArgs& a, unsigned n0, const Lane& L, unsigned N, unsigned K) {
        unsigned base;
        const unsigned s = seg4(a, n0, base);
        return Base::tile_in(a.W4[s], a.S4[s], n0, n0 - base, L, N, K);
    }
    __device__ static void store(const SplitArgs& a, __nv_bfloat16*, const float (&r)[4], unsigned j, unsigned n0,
                                 const Lane& L, unsigned M, unsigned N) {
        unsigned base;
        const unsigned s = seg4(a, n0, base);
        Base::store_in(a.S4[s], a.As, base, a.Cs[s], a.ldcs[s], base, r, j, n0, L, M, N);
    }
};

}
