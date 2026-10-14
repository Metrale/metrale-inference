// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-14: The W8A8 rowscale projection at prefill row counts: a tensor-core GEMM whose every output value is
// bit-identical to w8a8_gemv.cu's rowscale entries (the WxAy engine, wxay_engine.cuh) for the same operands:
//   C[m, n] = bf16( S[m, n] * (w_scale[n] * a_scale[m]) ),   S = ((P_0 + P_1) + P_2) + ... + P_7
//   P_w = the m16n8k32 E4M3 MMA chain, F32 accumulation from +0.0, over the 128-byte K chunks c = w, w + 8, w + 16, ...
//         in ascending c, each chunk the four MMAs of w8a8::W8A8<false, *>::chunk (the engine's K permutation).
// The engine computes P_w in warp w and sums the eight partials in warp order; here one warp computes all eight
// classes for its own sub-tile, chunk by chunk in class order (every chunk of class 0, then class 1, ...), with an
// inner accumulator per class folded into the outer one as outer = P_0, then outer = outer + P_w. A class with no
// chunk (K < 1024) folds +0.0, as the engine's idle warps contribute. Operand roles are the engine's: weight rows on
// the MMA's m16 side, tokens on n8, so each MMA sees the registers the engine's MMA sees for that (row, token).
//
// So a token's output does not depend on M, the tile, the launch or which kernel (GEMV or GEMM) served it. The weight
// is up to three row segments stacked into one output, as w8a8_gemv.cu's (n1, n2 boundaries, per-row resolution).
//
// Pipeline: STAGES-deep cp.async ring of whole 128-byte K chunks of the BN weight rows and BM token rows, staged in
// class order. A row's 16-byte unit u sits at unit u ^ ((row & 1) << 2), so the LDS.128 of rows g and g + 1 (one
// quarter-warp) land in disjoint bank halves.
//
// Owner: gb10 kernels.
// Invariants:
// - K % 128 == 0, lda % 16 == 0, Aq and every W 16-byte aligned; n1 and n2 multiples of 16 with 0 < n1 <= n2 <= N.
// - Grid (ceil(N / BN), ceil(M / BM), 1) (tile counts; `raster` assigns the tiles), block (256, 1, 1), dynamic shared
//   memory STAGES * (BN + BM) * 128 bytes.
// - Only C[m, n] with m < M, n < N is written.

#include "e4m3_mma_pipe.cuh"
#include "w8a8_policy.cuh"

namespace w8a8g {

constexpr unsigned CHUNK = 128;
constexpr unsigned UNITS = CHUNK / 16;
constexpr unsigned CLASSES = wxay::WARPS;

__device__ __forceinline__ unsigned unit_off(unsigned row, unsigned u) {
    return row * CHUNK + ((u ^ ((row & 1u) << 2)) << 4);
}

__device__ __forceinline__ uint4 lds128(const unsigned char* base, unsigned off) {
    return *reinterpret_cast<const uint4*>(base + off);
}

// 2026-10-14: Position s of the class-order walk over num_c = 8q + r chunks: classes 0..r-1 hold q + 1 chunks, the
// rest q. Returns the chunk index and sets `last` when it is its class's final chunk.
__device__ __forceinline__ unsigned class_chunk(unsigned s, unsigned q, unsigned r, bool& last) {
    const unsigned head = r * (q + 1u);
    unsigned w, i, cnt;
    if (s < head) {
        w = s / (q + 1u);
        i = s - w * (q + 1u);
        cnt = q + 1u;
    } else {
        const unsigned s2 = s - head;
        w = r + s2 / q;
        i = s2 - (w - r) * q;
        cnt = q;
    }
    last = i + 1u == cnt;
    return w + CLASSES * i;
}

// 2026-10-14: The tile a CTA computes, from its launch order (x fastest): GROUP token tiles share each weight tile in
// turn, so a weight tile is read from DRAM once per GROUP token tiles while those CTAs run together, and the GROUP
// token tiles stay in L2 across the sweep over N. With x over weight tiles directly, a weight larger than L2 was
// streamed again for every token tile (8448 x 4096 at M = 8192: 21.6 ms vs 11.9 ms for three 2816-row weights).
// Changes which CTA computes a tile, never what it computes.
constexpr unsigned GROUP = 8;

__device__ __forceinline__ void raster(unsigned& nt, unsigned& mt) {
    const unsigned num_n = gridDim.x, num_m = gridDim.y;
    const unsigned lin = blockIdx.y * num_n + blockIdx.x, per_group = GROUP * num_n;
    const unsigned first = lin / per_group * GROUP, size = min(num_m - first, GROUP), at = lin % per_group;
    mt = first + at % size;
    nt = at / size;
}

template <int BN, int BM, int WARPS_N, int WARPS_M, int STAGES>
__device__ __forceinline__ void gemm_rowscale(const w8a8::Args& a, __nv_bfloat16* __restrict__ C, unsigned M,
                                              unsigned N, unsigned K) {
    constexpr int THREADS = 256;
    static_assert(WARPS_N * WARPS_M == 8, "8 warps");
    constexpr int WN = BN / WARPS_N, WM = BM / WARPS_M;
    constexpr int MI = WN / 16, NJ = WM / 8;
    static_assert(MI >= 1 && NJ >= 1 && WN % 16 == 0 && WM % 8 == 0, "warp tile");
    constexpr int W_UNITS = BN * UNITS / THREADS, A_UNITS = BM * UNITS / THREADS;
    static_assert(W_UNITS >= 1 && A_UNITS >= 1 && BN * UNITS % THREADS == 0 && BM * UNITS % THREADS == 0, "staging");
    constexpr unsigned STAGE_BYTES = (BN + BM) * CHUNK;

    extern __shared__ __align__(128) unsigned char smem[];
    const wxay::Lane L = wxay::lane_ids();
    const unsigned tid = threadIdx.x;
    unsigned nt, mt;
    raster(nt, mt);
    const unsigned n0 = nt * BN, m0 = mt * BM;
    const unsigned num_c = K / CHUNK, q = num_c / CLASSES, r = num_c % CLASSES;

    // 2026-10-14: Each thread stages the same rows and unit at every step: row tid / 8 + 32 i, unit tid % 8.
    const unsigned su = tid % UNITS;
    const unsigned char* wsrc[W_UNITS];
    bool wlive[W_UNITS];
    unsigned wdst[W_UNITS];
#pragma unroll
    for (int i = 0; i < W_UNITS; i++) {
        const unsigned row = tid / UNITS + (unsigned)i * (THREADS / UNITS), n = n0 + row;
        unsigned base;
        const unsigned s = w8a8::seg_of(a, n, base);
        wlive[i] = n < N;
        wsrc[i] = (wlive[i] ? a.W[s] + (unsigned long long)(n - base) * K : a.W[0]) + su * 16u;
        wdst[i] = unit_off(row, su);
    }
    const unsigned char* asrc[A_UNITS];
    bool alive[A_UNITS];
    unsigned adst[A_UNITS];
#pragma unroll
    for (int i = 0; i < A_UNITS; i++) {
        const unsigned row = tid / UNITS + (unsigned)i * (THREADS / UNITS), tk = m0 + row;
        alive[i] = tk < M;
        asrc[i] = a.Aq + (unsigned long long)(alive[i] ? tk : 0u) * a.lda + su * 16u;
        adst[i] = BN * CHUNK + unit_off(row, su);
    }
    const unsigned sbase = e4m3g::smem_addr(smem);
    auto load = [&](unsigned s, unsigned stage) {
        bool last;
        const unsigned c = class_chunk(s, q, r, last);
        const unsigned st = sbase + stage * STAGE_BYTES;
#pragma unroll
        for (int i = 0; i < W_UNITS; i++) e4m3g::cp16(st + wdst[i], wsrc[i] + c * CHUNK, wlive[i]);
#pragma unroll
        for (int i = 0; i < A_UNITS; i++) e4m3g::cp16(st + adst[i], asrc[i] + c * CHUNK, alive[i]);
    };

    const unsigned wn0 = (L.warp % WARPS_N) * WN, wm0 = (L.warp / WARPS_N) * WM;
    float inner[MI][NJ][4], outer[MI][NJ][4];
#pragma unroll
    for (int mi = 0; mi < MI; mi++)
#pragma unroll
        for (int nj = 0; nj < NJ; nj++)
#pragma unroll
            for (int e = 0; e < 4; e++) inner[mi][nj][e] = 0.0f, outer[mi][nj][e] = 0.0f;

#pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
        if ((unsigned)s < num_c) load((unsigned)s, (unsigned)s);
        e4m3g::cp_commit();
    }
    bool first = true;
    for (unsigned s = 0; s < num_c; s++) {
        e4m3g::cp_wait<STAGES - 2>();
        __syncthreads();  // 2026-10-14: stage s visible; the stage read at s - 1 is free for the next load
        {
            const unsigned nxt = s + STAGES - 1;
            if (nxt < num_c) load(nxt, nxt % STAGES);
            e4m3g::cp_commit();
        }
        const unsigned char* sw = smem + (s % STAGES) * STAGE_BYTES;
        const unsigned char* sa = sw + BN * CHUNK;
        using P = w8a8::W8A8<false, false>;
        P::WReg w[MI];
#pragma unroll
        for (int mi = 0; mi < MI; mi++) {
            const unsigned r0 = wn0 + mi * 16 + L.g, r1 = r0 + 8u;
            w[mi].l0 = lds128(sw, unit_off(r0, L.t));
            w[mi].l1 = lds128(sw, unit_off(r0, 4u + L.t));
            w[mi].h0 = lds128(sw, unit_off(r1, L.t));
            w[mi].h1 = lds128(sw, unit_off(r1, 4u + L.t));
            w[mi].ws = 0.0f;
        }
#pragma unroll
        for (int nj = 0; nj < NJ; nj++) {
            const unsigned row = wm0 + nj * 8 + L.g;
            P::AReg b;
            b.b0 = lds128(sa, unit_off(row, L.t));
            b.b1 = lds128(sa, unit_off(row, 4u + L.t));
            b.s0 = 0.0f;
            b.s1 = 0.0f;
#pragma unroll
            for (int mi = 0; mi < MI; mi++) P::chunk(inner[mi][nj], w[mi], b);
        }
        bool last;
        class_chunk(s, q, r, last);
        if (last) {
#pragma unroll
            for (int mi = 0; mi < MI; mi++)
#pragma unroll
                for (int nj = 0; nj < NJ; nj++)
#pragma unroll
                    for (int e = 0; e < 4; e++) {
                        outer[mi][nj][e] = first ? inner[mi][nj][e] : outer[mi][nj][e] + inner[mi][nj][e];
                        inner[mi][nj][e] = 0.0f;
                    }
            first = false;
        }
    }
    e4m3g::cp_wait<0>();
    // 2026-10-14: The classes with no chunk (num_c < 8) add the engine's idle-warp +0.0 partials.
    for (unsigned w = num_c; w < CLASSES; w++)
#pragma unroll
        for (int mi = 0; mi < MI; mi++)
#pragma unroll
            for (int nj = 0; nj < NJ; nj++)
#pragma unroll
                for (int e = 0; e < 4; e++) outer[mi][nj][e] = outer[mi][nj][e] + inner[mi][nj][e];

#pragma unroll
    for (int mi = 0; mi < MI; mi++)
#pragma unroll
        for (int e = 0; e < 4; e++) {
            const unsigned n = n0 + wn0 + mi * 16 + L.g + (e < 2 ? 0u : 8u);
            if (n >= N) continue;
            unsigned base;
            const unsigned sg = w8a8::seg_of(a, n, base);
            const float ws = __ldg(a.S[sg] + (n - base));
#pragma unroll
            for (int nj = 0; nj < NJ; nj++) {
                const unsigned tk = m0 + wm0 + nj * 8 + L.t * 2u + (unsigned)(e & 1);
                if (tk >= M) continue;
                const float v = outer[mi][nj][e] * (ws * __ldg(a.As + tk));
                C[(unsigned long long)tk * a.ldc + n] = __float2bfloat16_rn(v);
            }
        }
}

}

#define W8A8G_TPL(NAME, BN, BM, WN, WM, STAGES, MINB)                                                              \
    extern "C" __global__ __launch_bounds__(256, MINB) void NAME(                                                  \
        const unsigned char* __restrict__ Aq, const float* __restrict__ As, const unsigned char* __restrict__ W0, \
        const float* __restrict__ S0, const unsigned char* __restrict__ W1, const float* __restrict__ S1,         \
        const unsigned char* __restrict__ W2, const float* __restrict__ S2, __nv_bfloat16* __restrict__ C,        \
        unsigned M, unsigned N, unsigned K, unsigned lda, unsigned ldc, unsigned n1, unsigned n2) {               \
        const w8a8::Args a = {Aq, As, {W0, W1, W2}, {S0, S1, S2}, n1, n2, lda, ldc};                              \
        w8a8g::gemm_rowscale<BN, BM, WN, WM, STAGES>(a, C, M, N, K);                                               \
    }

// 2026-10-14: Tile candidates (weight rows x tokens, warps over weights x tokens, stages). Same arithmetic in each.
W8A8G_TPL(w8a8_gemm_rowscale_128x64_s3, 128, 64, 4, 2, 3, 1)
W8A8G_TPL(w8a8_gemm_rowscale_128x64_s2, 128, 64, 4, 2, 2, 2)
W8A8G_TPL(w8a8_gemm_rowscale_64x128_s3, 64, 128, 2, 4, 3, 1)
W8A8G_TPL(w8a8_gemm_rowscale_128x128_s2, 128, 128, 4, 2, 2, 1)
