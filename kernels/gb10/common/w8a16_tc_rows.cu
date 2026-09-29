// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-28: Block-scaled W8A16 projection for 1..=64 decode rows on mma.sync m16n8k16 BF16
// tiles with the rows as the MMA's N columns:
//   C[r, n] = sum_k A[r, k] * E4M3(B[n, k]) * block_scale[n / 128, k / 128],  r < M, n < N
//
// Why: the tile kernels this replaces under the canonical row tiers (w8a16_gemm_pipelined_m32 /
// _m64) decode every weight byte through a shared-memory table into a BF16 shared tile and
// read it back per MMA; on GB10 that draws 50-75 W at 1..32 rows. Here a weight byte goes
// straight from a 16-byte global load to a BF16 fragment register (a byte permute and two
// logic ops) and is applied to every row tile; the activations are staged once per block in
// shared memory. Standalone at Qwen3.6-35B-A3B shapes (dgx2/dgx3): 30-50% fewer GPU-rail
// joules per launch; the same or less time from 16 rows up; at 1..8 rows up to 5% more time
// on the widest projection (12288 x 2048) and less on the others.
//
// Owner: gb10 kernels.
// Invariants:
// - A [M, lda] BF16 (the first K of each row read), B [N, K] E4M3 bytes, block_scale
//   [N / 128, K / 128] FP32, C [M, ldc] BF16 (the first N of each row written). The host
//   guarantees 1 <= M <= 8 * NT, N a positive multiple of TR_COLS and of 128, K a positive
//   multiple of 128, lda a multiple of 8 (16-byte rows) and lda >= K, ldc >= N.
// - A weight byte b becomes the BF16 whose bits are sign(b) | (b & 0x7F) << 4, which equals
//   E4M3(b) * 2^-120 exactly; activations are staged times 2^60 (exact in BF16) and each
//   128-K block's partial sum is scaled by block_scale * 2^60, so products and partial sums
//   stay FP32 normals (moe_fp8_grouped_tc.cu uses the same scheme).
// - Inside a 64-wide K chunk, lane t = lane & 3 holds K = 16t .. 16t + 15 of its weight rows
//   and activation rows; MMA j (0..3) takes K = 16t + 4j + {0,1} as fragment slots 2t, 2t+1
//   and K = 16t + 4j + {2,3} as 2t+8, 2t+9. A row's sum order is fixed by K alone and its
//   column of the MMA reads only its own activations, so a row's output bits do not depend
//   on M or on the other rows (the three entry points agree row for row).
// - Grid (N / TR_COLS, 1, 1), block TR_THREADS, static shared memory only.

#include <cuda_bf16.h>

#define TR_WARPS 4
#define TR_THREADS (TR_WARPS * 32)
#define TR_COLS (TR_WARPS * 16)

__device__ __forceinline__ unsigned int tr_e4m3_pair_bf16(unsigned int w, unsigned int sel) {
    const unsigned int x = __byte_perm(w, 0u, sel);
    return (x & 0x80008000u) | ((x >> 4) & 0x07F007F0u);
}

__device__ __forceinline__ void tr_mma_bf16(float* c, const unsigned int* a, unsigned int b0, unsigned int b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// 2026-09-28: One block: TR_COLS output columns, one m-tile (16 columns) per warp, NT row tiles
// of 8. The weights stream in load groups of G 64-K chunks (a weight row in 64 * G-byte runs),
// one group ahead; each group's activations are loaded one group ahead into registers and
// stored (times 2^60) into the other shared buffer after the current group's MMAs. G changes
// only how loads are batched, never the arithmetic, so the entry points agree bit for bit.
template <int NT, int G>
__device__ __forceinline__ void tr_block(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ B,
    const float* __restrict__ S, __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K, unsigned int lda, unsigned int ldc
) {
    constexpr int GK = 64 * G;
    // 2026-09-28: Shared row pitch in bytes: 16 bytes of padding keep the eight rows of one
    // 16-byte fragment load on distinct bank groups.
    constexpr int RS = GK * 2 + 16;
    constexpr int ROWS = 8 * NT;
    constexpr int U4 = ROWS * GK * 2 / 16;
    constexpr int PER = (U4 + TR_THREADS - 1) / TR_THREADS;
    __shared__ __align__(16) unsigned char xs[2][ROWS * RS];
    const unsigned int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const unsigned int f0 = blockIdx.x * TR_COLS + warp * 16;
    const unsigned int kblocks = K / 128, ngroups = K / GK;
    const unsigned int nt_live = (M + 7) / 8;
    const float two60 = 1152921504606846976.0f;
    const __nv_bfloat162 two60x2 = __floats2bfloat162_rn(two60, two60);

    uint4 xr[PER];
    auto group_load = [&](unsigned int gi) {
        #pragma unroll
        for (int i = 0; i < PER; i++) {
            const unsigned int u = threadIdx.x + i * TR_THREADS;
            const unsigned int row = u / (GK / 8), col = u % (GK / 8);
            xr[i] = (u < U4 && row < M)
                ? *(const uint4*)(A + (unsigned long long)row * lda + gi * GK + col * 8)
                : make_uint4(0u, 0u, 0u, 0u);
        }
    };
    auto group_store = [&](unsigned int buf) {
        #pragma unroll
        for (int i = 0; i < PER; i++) {
            const unsigned int u = threadIdx.x + i * TR_THREADS;
            if (u >= U4) break;
            const unsigned int row = u / (GK / 8), col = u % (GK / 8);
            unsigned int* w = (unsigned int*)&xr[i];
            #pragma unroll
            for (int q = 0; q < 4; q++) {
                __nv_bfloat162 v = *(__nv_bfloat162*)&w[q];
                v = __hmul2(v, two60x2);
                w[q] = *(unsigned int*)&v;
            }
            *(uint4*)(&xs[buf][row * RS + col * 16]) = xr[i];
        }
    };

    const unsigned char* w_lo = B + (unsigned long long)(f0 + g) * K + t * 16;
    const unsigned char* w_hi = B + (unsigned long long)(f0 + g + 8) * K + t * 16;
    const float* srow = S + (f0 / 128) * kblocks;
    float acc[NT][4], tmp[NT][4];
    #pragma unroll
    for (int n = 0; n < NT; n++)
        #pragma unroll
        for (int e = 0; e < 4; e++) acc[n][e] = tmp[n][e] = 0.f;

    group_load(0);
    group_store(0);
    uint4 wn[G][2];
    #pragma unroll
    for (int c = 0; c < G; c++) { wn[c][0] = *(const uint4*)(w_lo + c * 64); wn[c][1] = *(const uint4*)(w_hi + c * 64); }
    __syncthreads();
    for (unsigned int gi = 0; gi < ngroups; gi++) {
        const unsigned int buf = gi & 1;
        uint4 w[G][2];
        #pragma unroll
        for (int c = 0; c < G; c++) { w[c][0] = wn[c][0]; w[c][1] = wn[c][1]; }
        if (gi + 1 < ngroups) {
            #pragma unroll
            for (int c = 0; c < G; c++) {
                wn[c][0] = *(const uint4*)(w_lo + ((gi + 1) * G + c) * 64);
                wn[c][1] = *(const uint4*)(w_hi + ((gi + 1) * G + c) * 64);
            }
            group_load(gi + 1);
        }
        #pragma unroll
        for (int c = 0; c < G; c++) {
            const unsigned int chunk = gi * G + c;
            unsigned int a[4][4];
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                const uint4& lo = w[c][0];
                const uint4& hi = w[c][1];
                const unsigned int wg = (j == 0) ? lo.x : (j == 1) ? lo.y : (j == 2) ? lo.z : lo.w;
                const unsigned int wh = (j == 0) ? hi.x : (j == 1) ? hi.y : (j == 2) ? hi.z : hi.w;
                a[j][0] = tr_e4m3_pair_bf16(wg, 0x1404u);
                a[j][1] = tr_e4m3_pair_bf16(wh, 0x1404u);
                a[j][2] = tr_e4m3_pair_bf16(wg, 0x3424u);
                a[j][3] = tr_e4m3_pair_bf16(wh, 0x3424u);
            }
            #pragma unroll
            for (int n = 0; n < NT; n++) {
                if (n >= (int)nt_live) break;
                const uint4* xp = (const uint4*)(&xs[buf][(n * 8 + g) * RS + c * 128 + t * 32]);
                const uint4 xa = xp[0], xb = xp[1];
                const unsigned int xw[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
                #pragma unroll
                for (int j = 0; j < 4; j++) tr_mma_bf16(tmp[n], a[j], xw[2 * j], xw[2 * j + 1]);
            }
            // 2026-09-28: Chunks 2kb and 2kb + 1 make 128-K block kb: scale it once.
            if (chunk & 1) {
                const float s = srow[chunk >> 1] * two60;
                #pragma unroll
                for (int n = 0; n < NT; n++) {
                    if (n >= (int)nt_live) break;
                    #pragma unroll
                    for (int e = 0; e < 4; e++) { acc[n][e] += tmp[n][e] * s; tmp[n][e] = 0.f; }
                }
            }
        }
        if (gi + 1 < ngroups) group_store(buf ^ 1);
        __syncthreads();
    }
    // 2026-09-28: acc[n][e]: column f0 + g (+ 8 for e >= 2) of row 8n + 2t + (e & 1).
    #pragma unroll
    for (int n = 0; n < NT; n++) {
        if (n >= (int)nt_live) break;
        #pragma unroll
        for (int e = 0; e < 4; e++) {
            const unsigned int r = n * 8 + 2 * t + (e & 1);
            if (r < M) C[(unsigned long long)r * ldc + f0 + g + ((e >> 1) ? 8 : 0)] = __float2bfloat16(acc[n][e]);
        }
    }
}

// 2026-09-28: 1..=16 rows, 256-byte weight runs.
extern "C" __global__ void __launch_bounds__(TR_THREADS) w8a16_tc_rows_16(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ B,
    const float* __restrict__ block_scale, __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K, unsigned int lda, unsigned int ldc
) {
    tr_block<2, 4>(A, B, block_scale, C, M, N, K, lda, ldc);
}

// 2026-09-28: 1..=32 rows (from 17 rows the 256-byte runs of `_16` cost more than they save).
extern "C" __global__ void __launch_bounds__(TR_THREADS) w8a16_tc_rows_32(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ B,
    const float* __restrict__ block_scale, __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K, unsigned int lda, unsigned int ldc
) {
    tr_block<4, 2>(A, B, block_scale, C, M, N, K, lda, ldc);
}

// 2026-09-28: 33..=64 rows (any 1..=64; a row's bits equal the other entry points').
extern "C" __global__ void __launch_bounds__(TR_THREADS) w8a16_tc_rows_64(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ B,
    const float* __restrict__ block_scale, __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K, unsigned int lda, unsigned int ldc
) {
    tr_block<8, 2>(A, B, block_scale, C, M, N, K, lda, ldc);
}
