// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-09: The NVFP4 W4A16 row tiles of w4a16_tc_rows.cu (tc_rows.cuh, point Nvfp4G16) with two
// more parameters: up to three weight segments that share the input (one launch for a block's
// q/k/v, its gate/up or its f_a/g_a/b projections), and a K-split S of 1, 2 or 4 whose S
// blocks form one thread-block cluster and merge their FP32 partial sums through distributed
// shared memory.
//   C_i[r, n] = s2_i * sum_k A[r, k] * BF16(E2M1(W_i[n, k]) * E4M3(scale_i[n, k / 16])),
//   r < M, n < N_i, i < 3
//
// Why: at 2..16 decode rows the 64-column row tile puts at most one 4-warp block on an SM, and a
// narrow projection leaves most SMs idle and runs K serially: in the GLM-5.3 C16 profile a
// 768 x 4096 projection (12 blocks) took 21 us (~85 GB/s) and a 128 x 4096 one (2 blocks) 16 us.
// K-splitting multiplies the blocks per column tile and shortens each block's K; one launch per
// group of projections that share the input removes the other launches.
//
// Owner: gb10 kernels.
// Invariants:
// - Each segment as in w4a16_tc_rows.cu, with C_i contiguous (pitch N_i); N_i = 0 leaves segment
//   i out. The host guarantees 1 <= M <= 8 * NT, K a positive multiple of 256 (TR_SPLIT_K) with
//   S <= K / 256, lda a multiple of 8 and >= K, and N_0 > 0.
// - Grid (sum_i ceil(N_i / TR_COLS), S, 1), block TR_THREADS: block (x, s) computes split s of
//   column tile x; the S blocks of a tile are one cluster (__cluster_dims__(1, S, 1)).
// - S = 1: every output byte equals w4a16_tc_rows.cu's for the same segment (the same
//   tr_block_split<..., 1, TrWhole> code). S > 1: split s sums K's 256-wide units
//   [s * u / S, (s + 1) * u / S), u = K / 256, in w4a16_tc_rows.cu's order, and the cluster's
//   rank 0 adds the other splits' sums in rank order, ((p_0 + p_1) + p_2) + ..., then scales
//   once by s2: other bits than S = 1, which depend on K and S alone. For each S, a row's bits
//   do not depend on M, on the other rows, or on the entry (16, 32 or 64 rows).

#include <cooperative_groups.h>

#include "tc_rows.cuh"

// 2026-10-09: tr_block_split's merge for a K-split over the S blocks of one cluster: ranks
// 1..S-1 publish their partial sums in their own shared memory, rank 0 adds them in rank order
// and stores; the second barrier keeps the publishing blocks resident until rank 0 has read them.
template <int S>
struct TrClusterMerge {
    template <int NT>
    static __device__ __forceinline__ bool merge(float (&acc)[NT][4], unsigned int nt_live, unsigned char* scratch) {
        cooperative_groups::cluster_group cl = cooperative_groups::this_cluster();
        const unsigned int rank = cl.block_rank();
        float* part = (float*)scratch;
        if (rank != 0) {
            #pragma unroll
            for (int n = 0; n < NT; n++) {
                if (n >= (int)nt_live) break;
                #pragma unroll
                for (int e = 0; e < 4; e++) part[(n * 4 + e) * TR_THREADS + threadIdx.x] = acc[n][e];
            }
        }
        cl.sync();
        if (rank == 0) {
            #pragma unroll
            for (int r = 1; r < S; r++) {
                const float* peer = cl.map_shared_rank(part, r);
                #pragma unroll
                for (int n = 0; n < NT; n++) {
                    if (n >= (int)nt_live) break;
                    #pragma unroll
                    for (int e = 0; e < 4; e++) acc[n][e] += peer[(n * 4 + e) * TR_THREADS + threadIdx.x];
                }
            }
        }
        cl.sync();
        return rank == 0;
    }
};

template <int S>
struct TrSegMerge {
    using T = TrClusterMerge<S>;
};
template <>
struct TrSegMerge<1> {
    using T = TrWhole;
};

// 2026-10-09: Block (blockIdx.x, blockIdx.y): the segment and column tile blockIdx.x names, split
// blockIdx.y of S.
template <int NT, int G, int S>
__device__ __forceinline__ void w4a16_seg_block(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* p0, const unsigned char* sc0, float s20, __nv_bfloat16* C0, unsigned int N0,
    const unsigned char* p1, const unsigned char* sc1, float s21, __nv_bfloat16* C1, unsigned int N1,
    const unsigned char* p2, const unsigned char* sc2, float s22, __nv_bfloat16* C2, unsigned int N2,
    unsigned int M, unsigned int K, unsigned int lda
) {
    const unsigned int t0 = (N0 + TR_COLS - 1) / TR_COLS, t01 = t0 + (N1 + TR_COLS - 1) / TR_COLS;
    const unsigned int b = blockIdx.x;
    const unsigned int seg = b >= t01 ? 2u : (b >= t0 ? 1u : 0u);
    const unsigned char* packed = seg == 2u ? p2 : (seg == 1u ? p1 : p0);
    const unsigned char* scale = seg == 2u ? sc2 : (seg == 1u ? sc1 : sc0);
    const float s2 = seg == 2u ? s22 : (seg == 1u ? s21 : s20);
    __nv_bfloat16* C = seg == 2u ? C2 : (seg == 1u ? C1 : C0);
    const unsigned int N = seg == 2u ? N2 : (seg == 1u ? N1 : N0);
    const unsigned int tile = b - (seg == 2u ? t01 : (seg == 1u ? t0 : 0u));
    tr_block_split<Nvfp4G16, NT, G, true, S, typename TrSegMerge<S>::T>(
        A, {packed, scale, s2}, C, M, N, K, lda, N, tile, blockIdx.y);
}

// 2026-10-09: Entry w4a16_tc_rows_seg_<R>_k<S>: the row tiles of w4a16_tc_rows_<R> (NT, G) with
// a K-split of S.
#define W4A16_SEG_ENTRY(R, NT, G, S, CLUSTER)                                                              \
    extern "C" __global__ void CLUSTER __launch_bounds__(TR_THREADS) w4a16_tc_rows_seg_##R##_k##S(        \
        const __nv_bfloat16* __restrict__ A,                                                               \
        const unsigned char* p0, const unsigned char* sc0, float s20, __nv_bfloat16* C0, unsigned int N0, \
        const unsigned char* p1, const unsigned char* sc1, float s21, __nv_bfloat16* C1, unsigned int N1, \
        const unsigned char* p2, const unsigned char* sc2, float s22, __nv_bfloat16* C2, unsigned int N2, \
        unsigned int M, unsigned int K, unsigned int lda                                                   \
    ) {                                                                                                    \
        w4a16_seg_block<NT, G, S>(A, p0, sc0, s20, C0, N0, p1, sc1, s21, C1, N1, p2, sc2, s22, C2, N2,    \
                                  M, K, lda);                                                              \
    }

#define W4A16_SEG_SPLITS(R, NT, G)                                  \
    W4A16_SEG_ENTRY(R, NT, G, 1, )                                  \
    W4A16_SEG_ENTRY(R, NT, G, 2, __cluster_dims__(1, 2, 1))         \
    W4A16_SEG_ENTRY(R, NT, G, 4, __cluster_dims__(1, 4, 1))

// 2026-10-09: 1..=16, 1..=32 and 1..=64 rows, as w4a16_tc_rows_16 / _32 / _64.
W4A16_SEG_SPLITS(16, 2, 2)
W4A16_SEG_SPLITS(32, 4, 1)
W4A16_SEG_SPLITS(64, 8, 1)
