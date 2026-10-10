// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-10: Grouped W4A4 expert GEMM for wide routed-expert launches (prefill): every routed
// slot of every expert in one launch, on the w4a4_mx core's FP4 block-scale MMA
// (w4a4_mx_core.cuh, no new MMA code), with each output BIT-IDENTICAL to the slot GEMV
// w4a4_gemv_mx8_moe_slots (w4a4_gemv_mx_moe.cu) on the same slot.
//
// - w4a4_grouped_tiles: one block; tile_off[e] = sum over e' < e of ceil(M_e' / MT), the first
//   tile of expert e, from the expert-sorted row offsets (moe_sort_by_expert); tile_off[E] is the
//   tile count. The GEMM's grid.y is an upper bound, ceil(slots / MT) + E; tiles past the count
//   exit. No host read, so the launch can be captured.
// - w4a4_grouped_mx_m{32,64}[_k64]: C rows [expert_offsets[e], expert_offsets[e + 1]) are expert
//   e's sorted slots; C row r reads activation row sorted_token_ids[r] when that pointer is
//   non-null (gate/up: the token's quantized row), else row r (down: the slot's own row). Weights
//   from the global-id pointer tables; a null packed pointer writes nothing (another EP rank's
//   expert). blockIdx.x is a 128-column weight tile, blockIdx.y a linear (expert, MT-slot) tile.
//
// Why the bits match the slot GEMV. The slot GEMV runs w4a4_gemv_mx_tok_impl<1, 4>: warp w
// (0..7) accumulates the k128 chunks c = w, w + 8, w + 16, .. in increasing order, two MMAs per
// chunk (K halves, fragments built by the same shuffles and byte permutes), then the 8 warp sums
// are added as v = s0; v += s1; .. v += s7, and the output is bf16_rn(v * (Ag * scale2)). Here
// each thread visits the chunks in that class order (class w = chunks c = w mod 8, ascending),
// keeps the class sum in `acc` and the running total in `tot` (tot = class 0; tot += class w for
// w = 1..7, empty classes adding +0.0 as the GEMV's zero warp sums do), loads each operand into
// the registers the GEMV loads it into (staged through shared memory, same bytes), and finishes
// with the same expression. An MMA output column (a token) depends only on that column's
// operands, which the union GEMV's byte gate already relies on (model-arch example
// glm5next_moe_wide_bench). The model-arch example glm5next_moe_prefill_w4a4_bench checks every
// output byte against the slot GEMV.
//
// Owner: gb10 kernels.
// Invariants:
// - Launch: tiles grid (1, 1, 1), block 256. GEMM grid (ceil(N / NT), ceil(slots / MT) + E, 1),
//   block 256, dynamic shared memory of sizeof(W4gSmem<MT, NT, S>) (host:
//   `w4a4_grouped::smem_bytes`); MT must match the tile pass's `mt`. Entries: m32 and m64 with
//   NT 128, m128 with NT 64; S = 3 pipeline stages (m128_s6: 6, one CTA per SM).
// - K % 128 == 0 (_k64: K % 64 == 0, activations at Kp = round_up(K, 128) as
//   w4a4_quant_rows_static_k64 writes them); K <= 8192 (the class-order table). N % 8 == 0.
// - Activations as w4a4_quant_rows_static writes them (fragment order, row stride Kp / 2, Kp / 16
//   scales, one F32 global per row).
#ifndef METRALE_NO_WARP_BLOCKSCALE_MMA

#include "w4a4_mx_core.cuh"

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

extern "C" __global__ void __launch_bounds__(W4G_THREADS) w4a4_grouped_tiles(
    const int* __restrict__ expert_offsets, int* __restrict__ tile_off, unsigned int num_experts,
    unsigned int mt)
{
    if (threadIdx.x != 0) return;
    int acc = 0;
    for (unsigned int e = 0; e < num_experts; ++e) {
        tile_off[e] = acc;
        const int m = expert_offsets[e + 1] - expert_offsets[e];
        acc += m > 0 ? (m + (int)mt - 1) / (int)mt : 0;
    }
    tile_off[num_experts] = acc;
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

template <int MT, int NT, int S, bool KH>
__device__ __forceinline__ void w4g_body(
    const unsigned char* __restrict__ Aq, const unsigned char* __restrict__ As,
    const float* __restrict__ Ag, const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs, const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C, const int* __restrict__ expert_offsets,
    const int* __restrict__ tile_off, const int* __restrict__ sorted_token_ids,
    unsigned int num_experts, unsigned int N, unsigned int K)
{
    constexpr int WM = MT / 32;          // warps along the slots
    constexpr int WN = 8 / WM;           // warps along the weight rows
    constexpr int FN = NT / WN / 16;     // 16-row weight fragments per warp
    constexpr int FM = 4;                // 8-slot fragments per warp (32 slots)
    static_assert(FN >= 1 && NT % (WN * 16) == 0, "NT must give each warp whole fragments");
    extern __shared__ __align__(16) unsigned char w4g_dyn[];
    W4gSmem<MT, NT, S>& sm = *reinterpret_cast<W4gSmem<MT, NT, S>*>(w4g_dyn);

    // 2026-10-10: The tile's expert: the last e with tile_off[e] <= tile (block-uniform).
    const int tile = (int)blockIdx.y;
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
    const int m_start = expert_offsets[e] + (tile - tile_off[e]) * MT;
    const int m_valid = min(MT, expert_offsets[e + 1] - m_start);
    const unsigned int n0 = blockIdx.x * NT;

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

    for (unsigned int i = tid; i < (unsigned int)MT; i += W4G_THREADS) {
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
        for (unsigned int i = tid; i < NT * 4u; i += W4G_THREADS) {
            const unsigned int r = i >> 2, part = i & 3u;
            const unsigned int n = n0 + r;
            const bool ok = n < N && !(tail && part >= 2u);
            const unsigned char* src =
                ok ? Bq + (unsigned long long)n * half_K + c * 64u + part * 16u : Bq;
            w4g_cp16(&st.w[r][part * 16u], src, ok);
        }
        for (unsigned int i = tid; i < NT * 2u; i += W4G_THREADS) {
            const unsigned int r = i >> 1, h = i & 1u;
            const unsigned int n = n0 + r;
            const bool ok = n < N && !(tail && h == 1u);
            const unsigned char* src =
                ok ? Bs + (unsigned long long)n * groups + c * 8u + h * 4u : Bs;
            w4g_cp4(&st.ws[r][h * 4u], src, ok);
        }
#pragma unroll
        for (unsigned int i = tid; i < (unsigned int)MT * 4u; i += W4G_THREADS) {
            const unsigned int r = i >> 2, part = i & 3u;
            const bool ok = (int)r < m_valid;
            const unsigned char* src =
                ok ? Aq + (unsigned long long)sm.arow[r] * a_half_K + c * 64u + part * 16u : Aq;
            w4g_cp16(&st.a[r][part * 16u], src, ok);
        }
        for (unsigned int i = tid; i < (unsigned int)MT; i += W4G_THREADS) {
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
                sm.u.out[slot][col] =
                    __float2bfloat16_rn(tot[f][j][q] * (sm.ag[slot] * scale2));
            }
        }
    }
    __syncthreads();
    for (unsigned int i = tid; i < (unsigned int)MT * (NT / 8u); i += W4G_THREADS) {
        const unsigned int slot = i / (NT / 8u), piece = i % (NT / 8u);
        const unsigned int n = n0 + piece * 8u;
        if ((int)slot < m_valid && n < N) {
            *(uint4*)(C + (unsigned long long)(m_start + (int)slot) * N + n) =
                *(const uint4*)&sm.u.out[slot][piece * 8u];
        }
    }
}

#define W4G_ARGS                                                                               \
    const unsigned char* __restrict__ Aq, const unsigned char* __restrict__ As,               \
        const float* __restrict__ Ag, const unsigned long long* __restrict__ packed_ptrs,     \
        const unsigned long long* __restrict__ scale_ptrs,                                    \
        const float* __restrict__ scale2_vals, __nv_bfloat16* __restrict__ C,                 \
        const int* __restrict__ expert_offsets, const int* __restrict__ tile_off,             \
        const int* __restrict__ sorted_token_ids, unsigned int num_experts, unsigned int N,   \
        unsigned int K
#define W4G_PASS                                                                               \
    Aq, As, Ag, packed_ptrs, scale_ptrs, scale2_vals, C, expert_offsets, tile_off,             \
        sorted_token_ids, num_experts, N, K

extern "C" __global__ void __launch_bounds__(W4G_THREADS, 2) w4a4_grouped_mx_m64(W4G_ARGS) {
    w4g_body<64, W4G_NT, 3, false>(W4G_PASS);
}
extern "C" __global__ void __launch_bounds__(W4G_THREADS, 2) w4a4_grouped_mx_m64_k64(W4G_ARGS) {
    w4g_body<64, W4G_NT, 3, true>(W4G_PASS);
}
extern "C" __global__ void __launch_bounds__(W4G_THREADS, 2) w4a4_grouped_mx_m32(W4G_ARGS) {
    w4g_body<32, W4G_NT, 3, false>(W4G_PASS);
}
extern "C" __global__ void __launch_bounds__(W4G_THREADS, 2) w4a4_grouped_mx_m32_k64(W4G_ARGS) {
    w4g_body<32, W4G_NT, 3, true>(W4G_PASS);
}
// 2026-10-10: 128 slots by 64 weight rows: an expert's weights are read once for up to 128 of
// its slots (a 2048-row prefill sub-chunk averages 57 per expert at top 8 of 288).
extern "C" __global__ void __launch_bounds__(W4G_THREADS, 2) w4a4_grouped_mx_m128(W4G_ARGS) {
    w4g_body<128, 64, 3, false>(W4G_PASS);
}
extern "C" __global__ void __launch_bounds__(W4G_THREADS) w4a4_grouped_mx_m128_s6(W4G_ARGS) {
    w4g_body<128, 64, 6, false>(W4G_PASS);
}
extern "C" __global__ void __launch_bounds__(W4G_THREADS, 2) w4a4_grouped_mx_m128_k64(W4G_ARGS) {
    w4g_body<128, 64, 3, true>(W4G_PASS);
}

#endif // METRALE_NO_WARP_BLOCKSCALE_MMA
