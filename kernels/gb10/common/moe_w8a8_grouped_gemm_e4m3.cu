// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-27: `moe_w8a8_grouped_gemm_e4m3_*`: the routed-expert grouped GEMM of `moe_w8a8_grouped_gemm_pm4`
// (same inputs, same work-list, same output) on the native FP8 tensor-core MMA
// `mma.sync.m16n8k32.row.col.f32.e4m3.e4m3.f32` instead of software E4M3 -> BF16 decode plus BF16 MMAs:
//   C[m, n] = bf16( sum_c ( sum_(k in c) A[token(m), k] * B[n, k] ) * (a_scale[token(m), c / 2] * b_scale[n / 128, c / 2]) )
// with c over 64-value chunks of K, as in the PM4 kernel. Every FP8 x FP8 product is exact in F32 on both paths, and
// each MMA sums the same 16 products in the same K order as PM4's BF16 m16n8k16 MMAs (see the split below), so the
// output equals PM4's bit for bit on every input tested. It is deterministic: a row's result depends only on its own
// inputs (fixed tile and fold order, no atomics, no split-K).
//
// Work-list: item w is (expert_id, mt << 6 | nt) from `moe_build_tile_worklist` with m_tile = BM and
// n_tiles = N / BN, written earlier on the same stream. Expert e owns rows expert_offsets[e] .. expert_offsets[e+1];
// token(m) is sorted_token_ids[m], or m when that pointer is NULL. An expert with a NULL weight pointer is skipped.
//
// Pipeline: STAGES-deep cp.async ring of BK = 64-byte K slices of A (gathered by token id) and B, with the per-row
// a_scale of the slice's 128-K block; XOR-swizzled 64-byte smem rows read with ldmatrix. Rows past the expert are
// zero-filled (and their a_scale is 0), and m16 sub-tiles that lie wholly past the expert skip their MMAs.
//
// Requirements (checked by the caller): K % 128 == 0, N % BN == 0, BN divides 128 (one b_scale row per item).
// Dynamic shared memory: STAGES * (BM + BN) * 64 + STAGES * BM * 4 + BM * 4 bytes.
//
// Owner: gb10 kernels.
// Invariants: none beyond the types.

#include <cuda_bf16.h>

namespace e4m3g {

constexpr int BK = 64;
constexpr int SCALE_BLOCK = 128;

__device__ __forceinline__ unsigned smem_addr(const void* p) {
    return (unsigned)__cvta_generic_to_shared(p);
}

// 2026-09-27: 16-byte async copy; src_bytes 0 zero-fills the destination without reading `src`.
__device__ __forceinline__ void cp16(unsigned dst, const void* src, bool pred) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(pred ? 16 : 0));
}
__device__ __forceinline__ void cp4(unsigned dst, const void* src, bool pred) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(dst), "l"(src), "r"(pred ? 4 : 0));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// 2026-09-27: Byte offset of 16-byte chunk `ch` (0..3) of row `row` in a [rows][64] tile. The XOR spreads the 8 rows
// one ldmatrix phase reads over all 8 16-byte bank groups.
__device__ __forceinline__ unsigned swz(unsigned row, unsigned ch) {
    return row * BK + ((ch ^ ((row >> 1) & 3u)) << 4);
}

__device__ __forceinline__ void ldsm_x4(unsigned addr, unsigned& d0, unsigned& d1, unsigned& d2, unsigned& d3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
                 : "r"(addr));
}

__device__ __forceinline__ void mma_e4m3(float* acc, const unsigned* a, unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
        : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1),
          "f"(acc[0]), "f"(acc[1]), "f"(acc[2]), "f"(acc[3]));
}

template <int BM, int BN, int WARPS_M, int WARPS_N, int STAGES>
__device__ __forceinline__ void grouped_body(
    const unsigned char* __restrict__ A_fp8,
    const float* __restrict__ a_scale,
    const unsigned long long* __restrict__ B_weight_ptrs,
    const unsigned long long* __restrict__ B_scale_ptrs,
    __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    unsigned int N,
    unsigned int K,
    const unsigned int* __restrict__ worklist,
    const int* __restrict__ total_tiles
) {
    constexpr int THREADS = WARPS_M * WARPS_N * 32;
    constexpr int WM = BM / WARPS_M;
    constexpr int WN = BN / WARPS_N;
    constexpr int MI = WM / 16;
    constexpr int NI = WN / 8;
    static_assert(MI >= 1 && NI >= 2 && NI % 2 == 0, "warp tile");
    static_assert(128 % BN == 0, "BN divides 128");

    extern __shared__ __align__(128) unsigned char smem[];
    unsigned char* sA = smem;
    unsigned char* sB = sA + STAGES * BM * BK;
    float* sAS = (float*)(sB + STAGES * BN * BK);
    int* sTok = (int*)(sAS + STAGES * BM);

    const unsigned tid = threadIdx.x;
    const unsigned warp = tid >> 5;
    const unsigned lane = tid & 31u;
    const unsigned gid = lane >> 2;
    const unsigned tig = lane & 3u;
    const unsigned wm0 = (warp / WARPS_N) * WM;
    const unsigned wn0 = (warp % WARPS_N) * WN;
    const unsigned lmat = lane >> 3;
    const unsigned lrow = lane & 7u;

    const unsigned k_blocks = K / SCALE_BLOCK;
    const unsigned n_steps = K / BK;
    const int total = *total_tiles;

    for (int wid = blockIdx.x; wid < total; wid += (int)gridDim.x) {
        __syncthreads();  // 2026-09-27: the previous item is done with shared memory

        const unsigned expert_id = worklist[wid * 2 + 0];
        const unsigned packed = worklist[wid * 2 + 1];
        const unsigned cta_m = (packed >> 6) * BM;
        const unsigned cta_n = (packed & 0x3Fu) * BN;
        const int m_start = expert_offsets[expert_id];
        const int M_expert = expert_offsets[expert_id + 1] - m_start;
        const unsigned char* B_exp = (const unsigned char*)B_weight_ptrs[expert_id];
        const float* S_exp = (const float*)B_scale_ptrs[expert_id];
        if (B_exp == 0) continue;

        for (unsigned i = tid; i < (unsigned)BM; i += THREADS) {
            const unsigned m = cta_m + i;
            int t = -1;
            if (m < (unsigned)M_expert) {
                const int s = m_start + (int)m;
                t = sorted_token_ids ? sorted_token_ids[s] : s;
            }
            sTok[i] = t;
        }
        __syncthreads();

        const unsigned n_block = cta_n / SCALE_BLOCK;
        const unsigned char* B_tile = B_exp + (unsigned long long)cta_n * K;

        auto load_stage = [&](unsigned step, unsigned stage) {
            const unsigned k0 = step * BK;
            unsigned char* a_dst = sA + stage * BM * BK;
            unsigned char* b_dst = sB + stage * BN * BK;
            #pragma unroll
            for (unsigned c = tid; c < (unsigned)BM * 4; c += THREADS) {
                const unsigned row = c >> 2, ch = c & 3u;
                const int t = sTok[row];
                const unsigned char* src = A_fp8 + (unsigned long long)(t < 0 ? 0 : t) * K + k0 + ch * 16;
                cp16(smem_addr(a_dst + swz(row, ch)), src, t >= 0);
            }
            #pragma unroll
            for (unsigned c = tid; c < (unsigned)BN * 4; c += THREADS) {
                const unsigned row = c >> 2, ch = c & 3u;
                cp16(smem_addr(b_dst + swz(row, ch)), B_tile + (unsigned long long)row * K + k0 + ch * 16, true);
            }
            const unsigned kb = k0 / SCALE_BLOCK;
            for (unsigned i = tid; i < (unsigned)BM; i += THREADS) {
                const int t = sTok[i];
                cp4(smem_addr(sAS + stage * BM + i), a_scale + (unsigned long long)(t < 0 ? 0 : t) * k_blocks + kb, t >= 0);
            }
        };

        // 2026-09-27: The m16 sub-tiles of this warp that hold at least one of the expert's rows.
        const int rows_left = M_expert - (int)(cta_m + wm0);
        const int mi_valid = rows_left <= 0 ? 0 : min(MI, (rows_left + 15) / 16);

        float inner[MI][NI][4];
        float outer[MI][NI][4];
        #pragma unroll
        for (int mi = 0; mi < MI; mi++)
            #pragma unroll
            for (int ni = 0; ni < NI; ni++)
                #pragma unroll
                for (int r = 0; r < 4; r++) { inner[mi][ni][r] = 0.0f; outer[mi][ni][r] = 0.0f; }

        #pragma unroll
        for (int s = 0; s < STAGES - 1; s++) {
            if ((unsigned)s < n_steps) load_stage(s, s);
            cp_commit();
        }

        for (unsigned step = 0; step < n_steps; step++) {
            cp_wait<STAGES - 2>();
            __syncthreads();  // 2026-09-27: stage `step` visible; stage `step - 1` free for the next load
            {
                const unsigned nxt = step + STAGES - 1;
                if (nxt < n_steps) load_stage(nxt, nxt % STAGES);
                cp_commit();
            }
            const unsigned stage = step % STAGES;
            const unsigned a_base = smem_addr(sA + stage * BM * BK);
            const unsigned b_base = smem_addr(sB + stage * BN * BK);

            #pragma unroll
            for (int ks = 0; ks < 2; ks++) {
                unsigned bf[NI][2];
                #pragma unroll
                for (int j = 0; j < NI / 2; j++) {
                    const unsigned nrow = wn0 + j * 16 + lrow + ((lmat >> 1) << 3);
                    const unsigned ch = 2 * ks + (lmat & 1u);
                    ldsm_x4(b_base + swz(nrow, ch), bf[2 * j][0], bf[2 * j][1], bf[2 * j + 1][0], bf[2 * j + 1][1]);
                }
                #pragma unroll
                for (int mi = 0; mi < MI; mi++) {
                    if (mi < mi_valid) {
                        unsigned af[4];
                        const unsigned arow = wm0 + mi * 16 + lrow + ((lmat & 1u) << 3);
                        const unsigned ch = 2 * ks + (lmat >> 1);
                        ldsm_x4(a_base + swz(arow, ch), af[0], af[1], af[2], af[3]);
                        // 2026-09-27: Each k32 MMA is issued as two with one K half zeroed, so every MMA sums the
                        // same 16 products as PM4's m16n8k16 BF16 MMA, in the same K order. One k32 MMA sums 32
                        // products per step and rounds differently on wide dynamic ranges (measured: 4 of 4.5M
                        // outputs on normal data, 1396 of 18M over every E4M3 code); the split form matched
                        // PM4 bit for bit on every input tested, for 2x the MMAs (still under the MMA peak).
                        const unsigned alo[4] = {af[0], af[1], 0u, 0u}, ahi[4] = {0u, 0u, af[2], af[3]};
                        #pragma unroll
                        for (int ni = 0; ni < NI; ni++) {
                            mma_e4m3(inner[mi][ni], alo, bf[ni][0], 0u);
                            mma_e4m3(inner[mi][ni], ahi, 0u, bf[ni][1]);
                        }
                    }
                }
            }

            // 2026-09-27: Fold the 64-K chunk: outer += inner * (a_scale * b_scale), per row, then reset inner.
            const float bs = S_exp[n_block * k_blocks + (step * BK) / SCALE_BLOCK];
            const float* as = sAS + stage * BM;
            #pragma unroll
            for (int mi = 0; mi < MI; mi++) {
                const float s0 = as[wm0 + mi * 16 + gid] * bs;
                const float s1 = as[wm0 + mi * 16 + gid + 8] * bs;
                #pragma unroll
                for (int ni = 0; ni < NI; ni++) {
                    outer[mi][ni][0] += inner[mi][ni][0] * s0;
                    outer[mi][ni][1] += inner[mi][ni][1] * s0;
                    outer[mi][ni][2] += inner[mi][ni][2] * s1;
                    outer[mi][ni][3] += inner[mi][ni][3] * s1;
                    inner[mi][ni][0] = 0.0f; inner[mi][ni][1] = 0.0f;
                    inner[mi][ni][2] = 0.0f; inner[mi][ni][3] = 0.0f;
                }
            }
        }
        cp_wait<0>();

        #pragma unroll
        for (int mi = 0; mi < MI; mi++) {
            const unsigned r0 = cta_m + wm0 + mi * 16 + gid;
            const unsigned r1 = r0 + 8;
            #pragma unroll
            for (int ni = 0; ni < NI; ni++) {
                const unsigned col = cta_n + wn0 + ni * 8 + tig * 2;
                if (r0 < (unsigned)M_expert) {
                    __nv_bfloat162 v = __floats2bfloat162_rn(outer[mi][ni][0], outer[mi][ni][1]);
                    *(__nv_bfloat162*)&C[(unsigned long long)(m_start + r0) * N + col] = v;
                }
                if (r1 < (unsigned)M_expert) {
                    __nv_bfloat162 v = __floats2bfloat162_rn(outer[mi][ni][2], outer[mi][ni][3]);
                    *(__nv_bfloat162*)&C[(unsigned long long)(m_start + r1) * N + col] = v;
                }
            }
        }
    }
}

}  // namespace e4m3g

// 2026-09-27: The entry points' shared parameter list and forwarding list.
#define E4M3G_PARAMS                                                                                   \
    const unsigned char* __restrict__ A_fp8,              /* [total_tokens, K] FP8 E4M3 */             \
    const float* __restrict__ a_scale,                    /* [total_tokens, K / 128] F32 */            \
    const unsigned long long* __restrict__ B_weight_ptrs, /* [num_experts] -> [N, K] FP8 */            \
    const unsigned long long* __restrict__ B_scale_ptrs,  /* [num_experts] -> [N / 128, K / 128] F32 */\
    __nv_bfloat16* __restrict__ C,                        /* [total_expanded, N] BF16 */               \
    const int* __restrict__ expert_offsets,               /* [num_experts + 1] */                      \
    const int* __restrict__ sorted_token_ids,             /* [total_expanded] or NULL */               \
    unsigned int N, unsigned int K,                                                                    \
    const unsigned int* __restrict__ worklist,            /* [*total_tiles * 2] */                     \
    const int* __restrict__ total_tiles
#define E4M3G_ARGS A_fp8, a_scale, B_weight_ptrs, B_scale_ptrs, C, expert_offsets, sorted_token_ids, N, K, worklist, total_tiles

// 2026-09-27: Gate/up shape (N = inter, K = hidden): 64 x 128 items, 8 warps as 2 x 4 (warp tile 32 x 32), 3
// stages, 37.8 KiB of dynamic shared memory, 2 CTAs per SM. Work-list m_tile 64, n_tiles N / 128.
extern "C" __global__ void __launch_bounds__(256, 2) moe_w8a8_grouped_gemm_e4m3_gu(E4M3G_PARAMS) {
    e4m3g::grouped_body<64, 128, 2, 4, 3>(E4M3G_ARGS);
}

// 2026-09-27: Down shape (N = hidden, K = inter): 128 x 64 items, 8 warps as 4 x 2 (warp tile 32 x 32), 3 stages,
// 38.8 KiB, 2 CTAs per SM. Work-list m_tile 128, n_tiles N / 64.
extern "C" __global__ void __launch_bounds__(256, 2) moe_w8a8_grouped_gemm_e4m3_dn(E4M3G_PARAMS) {
    e4m3g::grouped_body<128, 64, 4, 2, 3>(E4M3G_ARGS);
}
