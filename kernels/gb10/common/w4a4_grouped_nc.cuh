// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-15: The grouped W4A4 expert GEMM with `nc` column tiles per CTA (w4n_body), for the
// one-projection entries (the down GEMM, a separate gate or up). w4g_body (w4a4_grouped_core.cuh)
// gives each CTA one (slot tile, column tile); at a 32k prefill the down GEMM runs 41k CTAs of six
// k128 chunks each, so every CTA's prologue (the tile's expert, its pointers, the slot rows and
// globals), its pipeline fill and its staged epilogue are paid per six chunks. Here a CTA walks
// column tiles k0 .. k0 + nc - 1 of one slot tile as ONE continuous cp.async stream (chunk p of
// column tile k is stream position k * num_c + p), so the prologue and the fill are paid once per
// nc tiles, and each finished tile is stored from registers while the next tile's chunks load.
//
// Why the bytes equal w4g_body's. Every output keeps its K chain: per column tile the chunks are
// visited in the same class order (sm.order), the MMA fragments are built by the same shared-memory
// reads, shuffles and byte permutes from the same bytes, the class sums fold into the total by the
// same expression (`tot = acc` for class 0, `tot += acc` after, empty classes adding +0.0), and the
// output is the same `bf16_rn(tot * (Ag * scale2))`. Folding class 0 assigns `tot`, so the next
// column tile needs no reset; the final fold zeroes `acc`. The expert lookup counts the tile
// offsets at or below the tile (tile_off is nondecreasing with tile_off[0] = 0), which is the last
// e with tile_off[e] <= tile, the binary search's answer. The store moves the same BF16 values: a
// shuffle with lane ^ 4 (weight row g ^ 1) pairs two adjacent output columns of one slot row.
//
// Owner: gb10 kernels.
// Invariants: as w4a4_grouped_mx.cu, except the grid: (ceil(ceil(N / NT) / nc), ceil(slots / MT)
// + E, 1), nc >= 1; `raster` walks that grid as w4g_body walks its own. N % 8 == 0. Include inside
// a METRALE_NO_WARP_BLOCKSCALE_MMA guard, after w4a4_grouped_core.cuh.
#pragma once

#include "w4a4_grouped_core.cuh"

// 2026-10-15: The tile's expert, the last e with tile_off[e] <= tile, as the number of experts
// whose first tile is at or below `tile`, minus one: one block-wide count per TH experts (the first
// two from values loaded before the caller's early exit; `v0`, `v1` = tile_off[tid], [tid + TH]).
template <int TH>
__device__ __forceinline__ unsigned int w4n_expert(const int* __restrict__ tile_off,
                                                   const unsigned int num_experts, const int tile,
                                                   const int v0, const int v1)
{
    const unsigned int tid = threadIdx.x;
    int x = __syncthreads_count(tid < num_experts && v0 <= tile);
    x += __syncthreads_count(tid + TH < num_experts && v1 <= tile);
    for (unsigned int b = 2u * TH; b < num_experts; b += TH) {
        const unsigned int i = b + tid;
        x += __syncthreads_count(i < num_experts && tile_off[i] <= tile);
    }
    return (unsigned int)(x - 1);
}

template <int MT, int NT, int S, bool KH, int TH = W4G_THREADS>
__device__ __forceinline__ void w4n_body(
    const unsigned char* __restrict__ Aq, const unsigned char* __restrict__ As,
    const float* __restrict__ Ag, const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs, const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C, const int* __restrict__ expert_offsets,
    const int* __restrict__ tile_off, const int* __restrict__ sorted_token_ids,
    unsigned int num_experts, unsigned int N, unsigned int K, const unsigned int raster,
    const unsigned int nc)
{
    constexpr int WM = MT / 32;
    constexpr int WN = TH / 32 / WM;
    constexpr int FN = NT / WN / 16;
    constexpr int FM = 4;
    static_assert(FN >= 1 && NT % (WN * 16) == 0, "NT must give each warp whole fragments");
    static_assert(TH % 32 == 0 && (TH / 32) % WM == 0, "TH must give each 32-slot band whole warps");
    extern __shared__ __align__(16) unsigned char w4g_dyn[];
    W4gSmem<MT, NT, S>& sm = *reinterpret_cast<W4gSmem<MT, NT, S>*>(w4g_dyn);

    int tile = (int)blockIdx.y;
    unsigned int cx = blockIdx.x;
    if (raster != 0u) {
        const unsigned int nx = gridDim.x, band = raster * nx;
        const unsigned int lin = blockIdx.x + nx * blockIdx.y, g = lin / band, r = lin % band;
        const unsigned int gt = min(raster, gridDim.y - g * raster);
        tile = (int)(g * raster + r % gt);
        cx = r / gt;
    }
    const unsigned int tid = threadIdx.x;
    const int total = tile_off[num_experts];
    const int v0 = tid < num_experts ? tile_off[tid] : 0;
    const int v1 = tid + TH < num_experts ? tile_off[tid + TH] : 0;
    const unsigned int nx = (N + NT - 1u) / NT, k0 = cx * nc;
    if (tile >= total || k0 >= nx) return;
    const unsigned int e = w4n_expert<TH>(tile_off, num_experts, tile, v0, v1);
    const unsigned long long bq_ptr = packed_ptrs[e];
    if (bq_ptr == 0ull) return;
    const unsigned char* Bq = (const unsigned char*)bq_ptr;
    const unsigned char* Bs = (const unsigned char*)scale_ptrs[e];
    const float scale2 = scale2_vals[e];
    const int m_start = expert_offsets[e] + (tile - tile_off[e]) * MT;
    const int m_valid = min(MT, expert_offsets[e + 1] - m_start);
    const unsigned int nk = min(nc, nx - k0);

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
    if (tid == 0) {
        unsigned int p = 0;
        for (unsigned int w = 0; w < 8u; ++w)
            for (unsigned int c = w; c < num_c; c += 8u) sm.order[p++] = (unsigned char)c;
    }
    __syncthreads();

    // 2026-10-15: The load stream's position: column tile lk (of nk), chunk position lp (of num_c).
    unsigned int lk = 0, lp = 0;
    auto load = [&](unsigned int s) {
        W4gStage<MT, NT>& st = sm.u.st[s];
        const unsigned int c = sm.order[lp];
        const unsigned int n0 = (k0 + lk) * NT;
        const bool tail = c == tail_c;
#pragma unroll
        for (unsigned int i = tid; i < NT * 4u; i += TH) {
            const unsigned int r = i >> 2, part = i & 3u;
            const unsigned int n = n0 + r;
            const bool ok = n < N && !(tail && part >= 2u);
            const unsigned char* src =
                ok ? Bq + (unsigned long long)n * half_K + c * 64u + part * 16u : Bq;
            w4g_cp16(&st.w[r][part * 16u], src, ok);
        }
        for (unsigned int i = tid; i < NT * 2u; i += TH) {
            const unsigned int r = i >> 1, h = i & 1u;
            const unsigned int n = n0 + r;
            const bool ok = n < N && !(tail && h == 1u);
            const unsigned char* src = ok ? Bs + (unsigned long long)n * groups + c * 8u + h * 4u : Bs;
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
            const unsigned char* src = ok ? As + (unsigned long long)sm.arow[i] * a_groups + c * 8u : As;
            w4g_cp8(&st.as[i][0], src, ok);
        }
        if (++lp == num_c) { lp = 0; ++lk; }
    };

    float acc[FN][FM][4];
    float tot[FN][FM][4];
#pragma unroll
    for (int f = 0; f < FN; ++f)
#pragma unroll
        for (int j = 0; j < FM; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[f][j][q] = tot[f][j][q] = 0.0f;

    const unsigned int nq = nk * num_c;
#pragma unroll
    for (unsigned int s = 0; s < S - 1; ++s) {
        if (s < nq) load(s);
        w4g_commit();
    }

    unsigned int cls = 0, pk = 0, pp = 0;
    for (unsigned int q = 0; q < nq; ++q) {
        w4g_wait<S - 2>();
        __syncthreads();
        {
            const unsigned int nxt = q + S - 1;
            if (nxt < nq) load(nxt % S);
            w4g_commit();
        }
        const unsigned int c = sm.order[pp];
        while (cls < (c & 7u)) {
#pragma unroll
            for (int f = 0; f < FN; ++f)
#pragma unroll
                for (int j = 0; j < FM; ++j)
#pragma unroll
                    for (int qq = 0; qq < 4; ++qq) {
                        tot[f][j][qq] = cls == 0 ? acc[f][j][qq] : tot[f][j][qq] + acc[f][j][qq];
                        acc[f][j][qq] = 0.0f;
                    }
            ++cls;
        }
        const W4gStage<MT, NT>& st = sm.u.st[q % S];
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
        if (++pp < num_c) continue;
        // 2026-10-15: Column tile pk is complete: fold the last class and the empty ones after it,
        // then store it. D (16 x 8): c0, c1 = weight row g, slots 2t, 2t + 1; c2, c3 = row g + 8.
        while (cls < 8u) {
#pragma unroll
            for (int f = 0; f < FN; ++f)
#pragma unroll
                for (int j = 0; j < FM; ++j)
#pragma unroll
                    for (int qq = 0; qq < 4; ++qq) {
                        tot[f][j][qq] = cls == 0 ? acc[f][j][qq] : tot[f][j][qq] + acc[f][j][qq];
                        acc[f][j][qq] = 0.0f;
                    }
            ++cls;
        }
        const unsigned int n0 = (k0 + pk) * NT;
        const bool ev = (g & 1u) == 0u;
#pragma unroll
        for (int f = 0; f < FN; ++f) {
            const unsigned int rb = wn * (NT / WN) + (unsigned int)f * 16u;
#pragma unroll
            for (int j = 0; j < FM; ++j) {
                const unsigned int s0 = wm * 32u + (unsigned int)j * 8u + t * 2u;
                const float g0 = sm.ag[s0] * scale2, g1 = sm.ag[s0 + 1u] * scale2;
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const unsigned int b0 = __bfloat16_as_ushort(__float2bfloat16_rn(tot[f][j][2 * h] * g0));
                    const unsigned int b1 =
                        __bfloat16_as_ushort(__float2bfloat16_rn(tot[f][j][2 * h + 1] * g1));
                    const unsigned int rv = __shfl_xor_sync(0xFFFFFFFFu, ev ? b1 : b0, 4);
                    const unsigned int slot = s0 + (ev ? 0u : 1u);
                    const unsigned int col = n0 + rb + (g & ~1u) + (h != 0 ? 8u : 0u);
                    if ((int)slot < m_valid && col < N)
                        *(unsigned int*)(C + (unsigned long long)(m_start + (int)slot) * N + col) =
                            ev ? (b0 | (rv << 16)) : (rv | (b1 << 16));
                }
            }
        }
        cls = 0;
        pp = 0;
        ++pk;
    }
    w4g_wait<0>();
}
