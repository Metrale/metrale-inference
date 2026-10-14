// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-14: The grouped W4A4 expert GEMM's body (w4g_body) and its shared-memory layout, moved
// here unchanged from w4a4_grouped_mx.cu (whose header comment documents the entries, the
// operands and why the bits match the slot GEMV) so that file stays under the size limit, plus
// the fused gate/up epilogue (GU = true).
//
// Owner: gb10 kernels.
// Invariants: as w4a4_grouped_mx.cu. Include inside a METRALE_NO_WARP_BLOCKSCALE_MMA guard.
#pragma once

#include "w4a4_mx_core.cuh"
#include "glm5next_swiglu.cuh"

#define W4G_THREADS 256
#define W4G_NT 128
#define W4G_MAX_CHUNKS 64

__device__ __forceinline__ unsigned int w4g_smem(const void* p) {
    return (unsigned int)__cvta_generic_to_shared(p);
}
// 2026-10-10: cp.async with zero fill: `bytes` = 0 writes zeros and reads nothing.
__device__ __forceinline__ void w4g_cp16(void* dst, const void* src, bool pred) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(w4g_smem(dst)), "l"(src),
                 "r"(pred ? 16 : 0));
}
__device__ __forceinline__ void w4g_cp8(void* dst, const void* src, bool pred) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" ::"r"(w4g_smem(dst)), "l"(src),
                 "r"(pred ? 8 : 0));
}
__device__ __forceinline__ void w4g_cp4(void* dst, const void* src, bool pred) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(w4g_smem(dst)), "l"(src),
                 "r"(pred ? 4 : 0));
}
__device__ __forceinline__ void w4g_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void w4g_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// 2026-10-10: One pipeline stage: a k128 chunk of NT weight rows and MT activation rows.
template <int MT, int NT>
struct W4gStage {
    unsigned char w[NT][64];
    unsigned char ws[NT][8];
    unsigned char a[MT][64];
    unsigned char as[MT][8];
};

template <int MT, int NT, int S>
struct W4gSmem {
    union {
        W4gStage<MT, NT> st[S];
        __nv_bfloat16 out[MT][NT + 8];
    } u;
    int arow[MT];
    float ag[MT];
    unsigned char order[W4G_MAX_CHUNKS];
};

// 2026-10-14: The fused gate/up epilogue's extra operands (GU = true; unused otherwise): the up
// projection's pointer tables, the down input's quantization (codes, scales, globals) at row
// width Kp under the down projection's static global scale gs, and the SwiGLU limit.
struct W4gGu {
    const unsigned long long* up_packed;
    const unsigned long long* up_scales;
    const float* up_scale2;
    unsigned char* Dq;
    unsigned char* Ds;
    float* Dg;
    unsigned int Kp;
    float gs;
    float limit;
};

// 2026-10-14: Shared-memory weight row r of a GU tile holds projection (r >> 4) & 1 (0 gate,
// 1 up) at output column w4g_gu_col(n0, r): the rows alternate 16 gate and 16 up rows of the same
// 16 columns, so column n sits at row n % 16 of its 16-row MMA fragment exactly as in the
// one-projection tiles (n0 is a multiple of 16 there too), and its gate and up values meet in the
// output tile at rows 32 * (n - n0) / 16 + n % 16 and that + 16.
__device__ __forceinline__ unsigned int w4g_gu_col(unsigned int n0, unsigned int r) {
    return n0 + (r >> 5) * 16u + (r & 15u);
}

// 2026-10-14: The fused gate/up epilogue: `out` holds every live slot's BF16 gate and up values
// (the bytes the one-projection launches store to their C); each thread takes a (slot, 16-column
// group), applies the clamped SwiGLU and its BF16 rounding (glm5next_swiglu_clamp's expression)
// and quantizes the group as w4a4_quant_rows_static[_k64] does, into the slot's own row of
// (Dq, Ds) at width Kp. The n0 == 0 CTA writes each slot's global, and the CTA holding column
// N - 1 the zero groups N / 16 .. Kp / 16 (PAD128). TH threads.
template <int MT, int NT, int TH = W4G_THREADS>
__device__ __forceinline__ void w4g_gu_epilogue(
    const __nv_bfloat16 (&out)[MT][NT + 8], const W4gGu& gu, const int m_start, const int m_valid,
    const unsigned int n0, const unsigned int N)
{
    constexpr unsigned int GPR = NT / 32u;  // 16-column groups per slot row
    const float inv_gs = 1.0f / gu.gs;
    for (unsigned int i = threadIdx.x; i < (unsigned int)MT * GPR; i += TH) {
        const unsigned int slot = i / GPR, gl = i % GPR;
        const unsigned int col = n0 + gl * 16u;
        if ((int)slot >= m_valid || col >= N) continue;
        unsigned int w[8];
#pragma unroll
        for (unsigned int h = 0; h < 8u; ++h) {
            unsigned int pair = 0u;
#pragma unroll
            for (unsigned int e = 0; e < 2u; ++e) {
                const unsigned int k = gl * 32u + 2u * h + e;
                const float a = glm5next_swiglu_clamp_f32(
                    __bfloat162float(out[slot][k]), __bfloat162float(out[slot][k + 16u]), gu.limit);
                pair |= (unsigned int)__bfloat16_as_ushort(__float2bfloat16(a)) << (16u * e);
            }
            w[h] = pair;
        }
        w4a4_quant_group(make_uint4(w[0], w[1], w[2], w[3]), make_uint4(w[4], w[5], w[6], w[7]),
                         gu.Dq, gu.Ds, (unsigned int)(m_start + (int)slot), gu.Kp, col >> 4, gu.gs,
                         inv_gs);
    }
    // 2026-10-14: The first column tile writes the rows' global scale (keyed on n0, not the grid
    // coordinate, so any tile raster keeps exactly one writer).
    if (n0 == 0u) {
        for (unsigned int i = threadIdx.x; i < (unsigned int)MT; i += TH)
            if ((int)i < m_valid) gu.Dg[m_start + (int)i] = gu.gs;
    }
    if (n0 + NT / 2u >= N) {
        const unsigned int g0 = N >> 4, np = (gu.Kp >> 4) - g0;
        for (unsigned int i = threadIdx.x; i < (unsigned int)MT * np; i += TH) {
            const unsigned int slot = i / np;
            if ((int)slot < m_valid)
                w4a4_quant_pad_group(gu.Dq, gu.Ds, (unsigned int)(m_start + (int)slot), gu.Kp,
                                     g0 + i % np);
        }
    }
}

// 2026-10-14: TH threads (TH / 32 warps over the MT x NT tile, 32 slots x NT / WN rows each); the
// warp tiling never changes an output's K chain, so every TH gives the same bytes.
template <int MT, int NT, int S, bool KH, bool GU = false, int TH = W4G_THREADS>
__device__ __forceinline__ void w4g_body(
    const unsigned char* __restrict__ Aq, const unsigned char* __restrict__ As,
    const float* __restrict__ Ag, const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs, const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C, const int* __restrict__ expert_offsets,
    const int* __restrict__ tile_off, const int* __restrict__ sorted_token_ids,
    unsigned int num_experts, unsigned int N, unsigned int K, const W4gGu gu = W4gGu{},
    const unsigned int raster = 0u)
{
    constexpr int WM = MT / 32;          // warps along the slots
    constexpr int WN = TH / 32 / WM;     // warps along the weight rows
    constexpr int FN = NT / WN / 16;     // 16-row weight fragments per warp
    constexpr int FM = 4;                // 8-slot fragments per warp (32 slots)
    static_assert(FN >= 1 && NT % (WN * 16) == 0, "NT must give each warp whole fragments");
    static_assert(TH % 32 == 0 && (TH / 32) % WM == 0, "TH must give each 32-slot band whole warps");
    extern __shared__ __align__(16) unsigned char w4g_dyn[];
    W4gSmem<MT, NT, S>& sm = *reinterpret_cast<W4gSmem<MT, NT, S>*>(w4g_dyn);

    // 2026-10-14: raster = 0: (column tile, tile) = (blockIdx.x, blockIdx.y). raster = G > 0: the
    // linear block index walks G tiles fastest, then the column tiles, then the next G tiles, so
    // the G tiles' weight tiles stay in L2 across the column sweep. Only which CTA computes which
    // (tile, column tile) changes; every output's K chain is the same, so the bytes are the same.
    int tile = (int)blockIdx.y;
    unsigned int cx = blockIdx.x;
    if (raster != 0u) {
        const unsigned int nx = gridDim.x, band = raster * nx;
        const unsigned int lin = blockIdx.x + nx * blockIdx.y, g = lin / band, r = lin % band;
        const unsigned int gt = min(raster, gridDim.y - g * raster);
        tile = (int)(g * raster + r % gt);
        cx = r / gt;
    }
    // 2026-10-10: The tile's expert: the last e with tile_off[e] <= tile (block-uniform).
    if (tile >= tile_off[num_experts]) return;
    int lo = 0, hi = (int)num_experts - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (tile_off[mid] <= tile) lo = mid; else hi = mid - 1;
    }
    const unsigned int e = (unsigned int)lo;
    const unsigned long long bq_ptr = packed_ptrs[e];
    if (bq_ptr == 0ull) return;
    const unsigned char* Bq = (const unsigned char*)bq_ptr;
    const unsigned char* Bs = (const unsigned char*)scale_ptrs[e];
    const float scale2 = scale2_vals[e];
    // 2026-10-14: GU: Bq / Bs / scale2 are the gate's; the up projection's beside them.
    const unsigned char* Uq = nullptr;
    const unsigned char* Us = nullptr;
    float uscale2 = 0.0f;
    if constexpr (GU) {
        const unsigned long long uq_ptr = gu.up_packed[e];
        if (uq_ptr == 0ull) return;
        Uq = (const unsigned char*)uq_ptr;
        Us = (const unsigned char*)gu.up_scales[e];
        uscale2 = gu.up_scale2[e];
    }
    const int m_start = expert_offsets[e] + (tile - tile_off[e]) * MT;
    const int m_valid = min(MT, expert_offsets[e + 1] - m_start);
    // 2026-10-14: GU: the tile's NT weight rows are NT / 2 output columns of gate and of up.
    const unsigned int n0 = cx * (GU ? NT / 2 : NT);

    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid >> 5, lane = tid & 31u;
    const unsigned int g = lane >> 2, t = lane & 3u;
    const bool odd = (t & 1u) != 0u;
    const unsigned int wn = warp % WN, wm = warp / WN;

    const unsigned int half_K = K >> 1;
    const unsigned int groups = K >> 4;
    const unsigned int num_c = KH ? (K + 127u) >> 7 : K >> 7;
    const unsigned int a_half_K = KH ? num_c << 6 : half_K;
    const unsigned int a_groups = KH ? num_c << 3 : groups;
    const unsigned int tail_c = (KH && (K & 127u) != 0u) ? num_c - 1u : 0xFFFFFFFFu;

    for (unsigned int i = tid; i < (unsigned int)MT; i += TH) {
        const int r = m_start + (int)i;
        const bool live = (int)i < m_valid;
        const int ar = live ? (sorted_token_ids != nullptr ? sorted_token_ids[r] : r) : 0;
        sm.arow[i] = ar;
        sm.ag[i] = live ? Ag[ar] : 0.0f;
    }
    // 2026-10-10: Chunk visiting order: class w = chunks w, w + 8, .. for w = 0..7.
    if (tid == 0) {
        unsigned int p = 0;
        for (unsigned int w = 0; w < 8u; ++w)
            for (unsigned int c = w; c < num_c; c += 8u) sm.order[p++] = (unsigned char)c;
    }
    __syncthreads();

    auto load = [&](unsigned int s, unsigned int c) {
        W4gStage<MT, NT>& st = sm.u.st[s];
        const bool tail = c == tail_c;
#pragma unroll
        for (unsigned int i = tid; i < NT * 4u; i += TH) {
            const unsigned int r = i >> 2, part = i & 3u;
            const unsigned int n = GU ? w4g_gu_col(n0, r) : n0 + r;
            const unsigned char* bq = (GU && ((r >> 4) & 1u)) ? Uq : Bq;
            const bool ok = n < N && !(tail && part >= 2u);
            const unsigned char* src =
                ok ? bq + (unsigned long long)n * half_K + c * 64u + part * 16u : bq;
            w4g_cp16(&st.w[r][part * 16u], src, ok);
        }
        for (unsigned int i = tid; i < NT * 2u; i += TH) {
            const unsigned int r = i >> 1, h = i & 1u;
            const unsigned int n = GU ? w4g_gu_col(n0, r) : n0 + r;
            const unsigned char* bs = (GU && ((r >> 4) & 1u)) ? Us : Bs;
            const bool ok = n < N && !(tail && h == 1u);
            const unsigned char* src =
                ok ? bs + (unsigned long long)n * groups + c * 8u + h * 4u : bs;
            w4g_cp4(&st.ws[r][h * 4u], src, ok);
        }
#pragma unroll
        for (unsigned int i = tid; i < (unsigned int)MT * 4u; i += TH) {
            const unsigned int r = i >> 2, part = i & 3u;
            const bool ok = (int)r < m_valid;
            const unsigned char* src =
                ok ? Aq + (unsigned long long)sm.arow[r] * a_half_K + c * 64u + part * 16u : Aq;
            w4g_cp16(&st.a[r][part * 16u], src, ok);
        }
        for (unsigned int i = tid; i < (unsigned int)MT; i += TH) {
            const bool ok = (int)i < m_valid;
            const unsigned char* src =
                ok ? As + (unsigned long long)sm.arow[i] * a_groups + c * 8u : As;
            w4g_cp8(&st.as[i][0], src, ok);
        }
    };

    float acc[FN][FM][4];
    float tot[FN][FM][4];
#pragma unroll
    for (int f = 0; f < FN; ++f)
#pragma unroll
        for (int j = 0; j < FM; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[f][j][q] = tot[f][j][q] = 0.0f;

#pragma unroll
    for (unsigned int s = 0; s < S - 1; ++s) {
        if (s < num_c) load(s, sm.order[s]);
        w4g_commit();
    }

    unsigned int cls = 0;
    for (unsigned int p = 0; p < num_c; ++p) {
        w4g_wait<S - 2>();
        __syncthreads();
        {
            const unsigned int nxt = p + S - 1;
            if (nxt < num_c) load(nxt % S, sm.order[nxt]);
            w4g_commit();
        }
        const unsigned int c = sm.order[p];
        // 2026-10-10: Entering a new class folds the finished one, and every empty class
        // between them, into the total (the GEMV's `v += s_w`, `v = s_0` for class 0).
        while (cls < (c & 7u)) {
#pragma unroll
            for (int f = 0; f < FN; ++f)
#pragma unroll
                for (int j = 0; j < FM; ++j)
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        tot[f][j][q] = cls == 0 ? acc[f][j][q] : tot[f][j][q] + acc[f][j][q];
                        acc[f][j][q] = 0.0f;
                    }
            ++cls;
        }
        const W4gStage<MT, NT>& st = sm.u.st[p % S];
        uint32_t bx[FM][4];
        uint2 sb[FM];
#pragma unroll
        for (int j = 0; j < FM; ++j) {
            const unsigned int r = wm * 32u + (unsigned int)j * 8u + g;
            const uint4 b = *(const uint4*)&st.a[r][t * 16u];
            bx[j][0] = b.x; bx[j][1] = b.y; bx[j][2] = b.z; bx[j][3] = b.w;
            sb[j] = *(const uint2*)&st.as[r][0];
        }
#pragma unroll
        for (int f = 0; f < FN; ++f) {
            const unsigned int rb = wn * (NT / WN) + (unsigned int)f * 16u;
            const uint4 wl = *(const uint4*)&st.w[rb + g][t * 16u];
            const uint4 wh = *(const uint4*)&st.w[rb + g + 8u][t * 16u];
            const uint2 sw = *(const uint2*)&st.ws[rb + g + (odd ? 8u : 0u)][0];
            // 2026-10-10: The fragments of w4a4_gemv_mx_tok_impl, verbatim.
            const uint32_t x0l = __shfl_xor_sync(0xFFFFFFFFu, odd ? wl.x : wl.y, 1);
            const uint32_t x0h = __shfl_xor_sync(0xFFFFFFFFu, odd ? wh.x : wh.y, 1);
            const uint32_t x1l = __shfl_xor_sync(0xFFFFFFFFu, odd ? wl.z : wl.w, 1);
            const uint32_t x1h = __shfl_xor_sync(0xFFFFFFFFu, odd ? wh.z : wh.w, 1);
            const uint32_t a00 = odd ? x0l : wl.x, a02 = odd ? wl.y : x0l;
            const uint32_t a01 = odd ? x0h : wh.x, a03 = odd ? wh.y : x0h;
            const uint32_t a10 = odd ? x1l : wl.z, a12 = odd ? wl.w : x1l;
            const uint32_t a11 = odd ? x1h : wh.z, a13 = odd ? wh.w : x1h;
            const uint32_t s0 = __byte_perm(sw.x, sw.y, 0x6240);
            const uint32_t s1 = __byte_perm(sw.x, sw.y, 0x7351);
#pragma unroll
            for (int j = 0; j < FM; ++j) {
                w4a4_mma(acc[f][j], a00, a01, a02, a03, bx[j][0], bx[j][1], s0, sb[j].x);
                w4a4_mma(acc[f][j], a10, a11, a12, a13, bx[j][2], bx[j][3], s1, sb[j].y);
            }
        }
    }
    // 2026-10-10: Fold the last non-empty class and the empty ones after it (num_c < 8).
    while (cls < 8u) {
#pragma unroll
        for (int f = 0; f < FN; ++f)
#pragma unroll
            for (int j = 0; j < FM; ++j)
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    tot[f][j][q] = cls == 0 ? acc[f][j][q] : tot[f][j][q] + acc[f][j][q];
                    acc[f][j][q] = 0.0f;
                }
        ++cls;
    }
    w4g_wait<0>();
    __syncthreads();

    // 2026-10-10: D (16 x 8): c0, c1 = weight row g, slots 2t, 2t + 1; c2, c3 = row g + 8. The
    // tile is staged in shared memory, then stored as 16-byte row pieces.
#pragma unroll
    for (int f = 0; f < FN; ++f) {
        const unsigned int rb = wn * (NT / WN) + (unsigned int)f * 16u;
#pragma unroll
        for (int j = 0; j < FM; ++j) {
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const unsigned int col = rb + g + (q < 2 ? 0u : 8u);
                const unsigned int slot = wm * 32u + (unsigned int)j * 8u + t * 2u + (q & 1);
                const float s2 = (GU && ((col >> 4) & 1u)) ? uscale2 : scale2;
                sm.u.out[slot][col] = __float2bfloat16_rn(tot[f][j][q] * (sm.ag[slot] * s2));
            }
        }
    }
    __syncthreads();
    if constexpr (GU) {
        w4g_gu_epilogue<MT, NT, TH>(sm.u.out, gu, m_start, m_valid, n0, N);
    } else {
        for (unsigned int i = tid; i < (unsigned int)MT * (NT / 8u); i += TH) {
            const unsigned int slot = i / (NT / 8u), piece = i % (NT / 8u);
            const unsigned int n = n0 + piece * 8u;
            if ((int)slot < m_valid && n < N) {
                *(uint4*)(C + (unsigned long long)(m_start + (int)slot) * N + n) =
                    *(const uint4*)&sm.u.out[slot][piece * 8u];
            }
        }
    }
}
