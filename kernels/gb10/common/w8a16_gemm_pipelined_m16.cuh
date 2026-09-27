// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-27: Skinny-M body of w8a16_gemm_pipelined_m32, for 1..=16 rows. Included by
// w8a16_gemm_pipelined_m32.cu, whose entry point runs it when M <= PM16_MAX_M and
// K <= PM16_MAX_K.
//
// Owner: gb10 kernels.
// Invariants:
// - Every output gets the canonical tile's arithmetic: the m16n8k16 BF16 sub-MMAs over the
//   16-wide K windows of each 128-K step in ascending order into a zeroed FP32 inner sum,
//   then outer = outer + inner * block_scale for the steps in ascending order from 0, then
//   one BF16 rounding. A step's product inner * block_scale depends on that step alone, so
//   the products are computed in parallel and only the additions run in step order. The
//   A fragment rows past M are zero, as the 32-, 64- and 128-row tiles zero-fill theirs, and
//   an MMA row never reads another row, so the bits equal those tiles' at every M
//   (native_fp8_gdn_proj_m32_microtest compares them byte for byte at M = 1..64).
// - Block PM16_WARPS warps, grid (ceil(N / 8), 1, 1): block b owns columns 8b .. 8b + 7.
//   Warp w loads weight row 8b + w whole into shared memory (LDG.128 into registers, then
//   STS; cp.async streamed about 20% slower on GB10), then after one block barrier computes
//   the steps w * spw .. (w + 1) * spw - 1, spw = ceil(K / 128 / PM16_WARPS): each step's
//   sub-MMAs, with its A fragments read straight from global memory (rows at or past M are
//   zero registers) and its weight bytes converted in registers (pm16_e4m3x2_bf16x2), and
//   the step's scaled product kept in registers. After a second barrier the products replace
//   the weight rows, and after a third warp 0 adds them in step order. Weight rows are
//   K + 16 bytes apart (K a multiple of 128, so 4 words mod 32), so the 8 rows of a fragment
//   read land on distinct banks.
// - The caller's shared buffer is 16-byte aligned and PM16_SMEM_BYTES(K) long (dynamic
//   shared memory sized by the launcher).

#ifndef METRALE_W8A16_GEMM_PIPELINED_M16_CUH
#define METRALE_W8A16_GEMM_PIPELINED_M16_CUH

#define PM16_MAX_M 16
// 2026-09-27: The largest K whose slots fit 48 KiB of dynamic shared memory.
#define PM16_MAX_K 5120
#define PM16_WARPS 8
#define PM16_K_STEP 128
// 2026-09-27: Steps per warp at PM16_MAX_K.
#define PM16_MAX_SPW ((PM16_MAX_K / PM16_K_STEP + PM16_WARPS - 1) / PM16_WARPS)
// 2026-09-27: Dynamic shared memory for K: the block's 8 weight rows, K + 16 bytes apart; the
// ops launcher's w8a16_m16_smem_bytes must agree.
#define PM16_SMEM_BYTES(K) (8 * ((K) + 16))

// 2026-09-27: The BF16 pair of the E4M3 bytes in the low half of pair (low byte first),
// bit for bit the value the other tiles stage (E4M3_LUT to FP32, then BF16). A code c
// becomes the FP32 with bits (c & 0x80) << 24 | (c & 0x7F) << 20, c's value times 2^-120
// for every non-NaN code (subnormals included; these kernels do not flush denormals), and
// times 2^120 it is c's value exactly, whose top 16 bits are its BF16. The NaN codes are
// cleared to signed zero first, as the table maps them.
__device__ __forceinline__ unsigned int pm16_e4m3x2_bf16x2(unsigned int pair) {
    const unsigned int nan = ((pair & 0x7F7Fu) + 0x0101u) & 0x8080u;
    pair &= ~(nan - (nan >> 7));
    const float f0 = __uint_as_float(((pair & 0x80u) << 24) | ((pair & 0x7Fu) << 20)) * 0x1p120f;
    const float f1 = __uint_as_float(((pair & 0x8000u) << 16) | ((pair & 0x7F00u) << 12)) * 0x1p120f;
    return __byte_perm(__float_as_uint(f0), __float_as_uint(f1), 0x7632);
}

__device__ __forceinline__ void pm16_skinny(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B,
    const float* __restrict__ block_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K, unsigned int lda, unsigned int ldc,
    unsigned char* smem
) {
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;
    const unsigned int n0 = blockIdx.x * 8;
    const unsigned int n_steps = K / PM16_K_STEP;
    const float* scale_row = block_scale + (unsigned long long)(n0 / PM16_K_STEP) * n_steps;

    // 2026-09-27: Warp w copies weight row n0 + w whole, 512 contiguous bytes per warp load.
    const unsigned int row_stride = K + 16;
    {
        unsigned char* dst_row = smem + warp_id * row_stride;
        if (n0 + warp_id < N) {
            const unsigned char* src_row = B + (unsigned long long)(n0 + warp_id) * K;
            uint4 v[PM16_MAX_K / 512];
            #pragma unroll
            for (unsigned int i = 0; i < PM16_MAX_K / 512; i++) {
                const unsigned int c = (lane_id + 32 * i) * 16;
                if (c < K) v[i] = __ldg((const uint4*)(src_row + c));
            }
            #pragma unroll
            for (unsigned int i = 0; i < PM16_MAX_K / 512; i++) {
                const unsigned int c = (lane_id + 32 * i) * 16;
                if (c < K) *(uint4*)(dst_row + c) = v[i];
            }
        } else {
            for (unsigned int c = lane_id * 16; c < K; c += 32 * 16) *(uint4*)(dst_row + c) = make_uint4(0u, 0u, 0u, 0u);
        }
    }

    // 2026-09-27: This lane's fragment bytes of a 16-byte window: 2t, 2t + 1 of the low
    // eight (b0) and of the high eight (b1).
    const unsigned int pair_sel = (2 * tid) | ((2 * tid + 1) << 4);
    const bool lo_live = group_id < M;
    const bool hi_live = group_id + 8 < M;
    const __nv_bfloat16* a_lo = A + (unsigned long long)group_id * lda;
    const __nv_bfloat16* a_hi = A + (unsigned long long)(group_id + 8) * lda;
    const unsigned int spw = (n_steps + PM16_WARPS - 1) / PM16_WARPS;
    const unsigned int first = warp_id * spw;
    float prod[PM16_MAX_SPW][4];

    __syncthreads();
    const unsigned char* b_row = smem + group_id * row_stride;
    #pragma unroll
    for (unsigned int j = 0; j < PM16_MAX_SPW; j++) {
        const unsigned int step = first + j;
        if (j >= spw || step >= n_steps) break;
        const unsigned int k_base = step * PM16_K_STEP;
        const float scale = scale_row[step];
        float inner[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        #pragma unroll
        for (int s = 0; s < PM16_K_STEP / 16; s++) {
            const unsigned int c0 = s * 16 + tid * 2;
            const unsigned int c1 = c0 + 8;
            const uint4 w = *(const uint4*)(b_row + k_base + s * 16);
            const unsigned int b0 = pm16_e4m3x2_bf16x2(__byte_perm(w.x, w.y, pair_sel));
            const unsigned int b1 = pm16_e4m3x2_bf16x2(__byte_perm(w.z, w.w, pair_sel));
            const unsigned int a0 = lo_live ? __ldg((const unsigned int*)(a_lo + k_base + c0)) : 0u;
            const unsigned int a2 = lo_live ? __ldg((const unsigned int*)(a_lo + k_base + c1)) : 0u;
            const unsigned int a1 = hi_live ? __ldg((const unsigned int*)(a_hi + k_base + c0)) : 0u;
            const unsigned int a3 = hi_live ? __ldg((const unsigned int*)(a_hi + k_base + c1)) : 0u;
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0, %1, %2, %3}, "
                "{%4, %5, %6, %7}, "
                "{%8, %9}, "
                "{%10, %11, %12, %13};"
                : "=f"(inner[0]), "=f"(inner[1]), "=f"(inner[2]), "=f"(inner[3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                  "r"(b0), "r"(b1),
                  "f"(inner[0]), "f"(inner[1]), "f"(inner[2]), "f"(inner[3])
            );
        }
        #pragma unroll
        for (int i = 0; i < 4; i++) prod[j][i] = inner[i] * scale;
    }
    // 2026-09-27: Every warp is done with the weight rows; the products take their place,
    // step-major, 512 bytes per step (lane l's four at byte 16l).
    __syncthreads();
    #pragma unroll
    for (unsigned int j = 0; j < PM16_MAX_SPW; j++) {
        const unsigned int step = first + j;
        if (j >= spw || step >= n_steps) break;
        *(float4*)(smem + step * 512 + lane_id * 16) = make_float4(prod[j][0], prod[j][1], prod[j][2], prod[j][3]);
    }
    __syncthreads();
    if (warp_id != 0) return;

    float outer[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (unsigned int step = 0; step < n_steps; step++) {
        const float4 p = *(const float4*)(smem + step * 512 + lane_id * 16);
        outer[0] += p.x;
        outer[1] += p.y;
        outer[2] += p.z;
        outer[3] += p.w;
    }

    const unsigned int col0 = n0 + tid * 2;
    const unsigned int col1 = col0 + 1;
    const unsigned int row0 = group_id;
    const unsigned int row1 = row0 + 8;
    if (row0 < M && col0 < N) C[(unsigned long long)row0 * ldc + col0] = __float2bfloat16(outer[0]);
    if (row0 < M && col1 < N) C[(unsigned long long)row0 * ldc + col1] = __float2bfloat16(outer[1]);
    if (row1 < M && col0 < N) C[(unsigned long long)row1 * ldc + col0] = __float2bfloat16(outer[2]);
    if (row1 < M && col1 < N) C[(unsigned long long)row1 * ldc + col1] = __float2bfloat16(outer[3]);
}

#endif
