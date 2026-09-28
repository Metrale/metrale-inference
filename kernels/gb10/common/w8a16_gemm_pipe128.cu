// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-28: `w8a16_gemm_pipe128`: a bit-identical twin of `w8a16_gemm_pipelined` (w8a16_gemm_pipelined.cu) on
// 128 x 128 tiles:
//   C[m, n] = sum_k A[m, k] * E4M3(B[n, k]) * block_scale[n / 128, k / 128]
// Per output element the arithmetic is the original's: E4M3 bytes decoded through the same E4M3_LUT (exact in BF16),
// BF16 m16n8k16 MMAs over K in ascending 16-wide steps into an inner FP32 accumulator, folded as
// outer += inner * block_scale at every 128-K boundary, one BF16 rounding at the end. Only the organization differs:
// 8 warps as 2 x 4 (warp tile 64 x 32) instead of 8 x 1 on a 128 x 32 tile, so each A tile is read from L2 once per
// 128 output columns instead of once per 32; ldmatrix fragments from XOR-swizzled shared memory; a 3-stage cp.async
// ring for A and the raw weight bytes; grouped rasterization (8 M tiles per N column).
//
// Owner: gb10 kernels.
// Invariants:
// - A [M, K] BF16, B [N, K] E4M3, C [M, N] BF16, block_scale [N / 128, K / 128] F32; K % 128 == 0 and N % 128 == 0
//   (the caller falls back to w8a16_gemm_pipelined otherwise).
// - Grid (N / 128, ceil(M / 128), 1), block 256, dynamic shared memory 3 x (8 KiB + 4 KiB) + 2 x 8 KiB + 1 KiB =
//   54,272 bytes.

#include <cuda_bf16.h>
#include "e4m3_lut.cuh"

namespace w8p {

constexpr int BM = 128, BN = 128, BK = 32, STAGES = 3, THREADS = 256;
constexpr int WARPS_N = 4, WM = 64, WN = 32, MI = WM / 16, NI = WN / 8;
constexpr int ROW_A = BK * 2;   // 2026-09-28: 64-byte A rows (BF16)
constexpr int ROW_BR = BK;      // 2026-09-28: 32-byte raw weight rows (E4M3)
constexpr int ROW_B = BK * 2;   // 2026-09-28: 64-byte decoded weight rows (BF16)
constexpr int A_BYTES = BM * ROW_A, BR_BYTES = BN * ROW_BR, B_BYTES = BN * ROW_B;

__device__ __forceinline__ unsigned smem_addr(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp16(unsigned dst, const void* src, bool pred) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(pred ? 16 : 0));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// 2026-09-28: Byte offset of 16-byte chunk `ch` (0..3) of row `row` in a tile of 64-byte rows; the XOR spreads each
// ldmatrix phase's 8 rows over the 8 bank groups.
__device__ __forceinline__ unsigned swz(unsigned row, unsigned ch) {
    return row * 64u + ((ch ^ ((row >> 1) & 3u)) << 4);
}
__device__ __forceinline__ void ldsm_x4(unsigned a, unsigned& d0, unsigned& d1, unsigned& d2, unsigned& d3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3) : "r"(a));
}
__device__ __forceinline__ void mma_bf16(float* c, const unsigned* a, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                 : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1),
                   "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

}  // namespace w8p

extern "C" __global__ void __launch_bounds__(256, 1) w8a16_gemm_pipe128(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B,
    const float* __restrict__ block_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    using namespace w8p;
    extern __shared__ __align__(128) unsigned char smem[];
    unsigned char* sA = smem;                         // 2026-09-28: STAGES x A_BYTES
    unsigned char* sBr = sA + STAGES * A_BYTES;       // 2026-09-28: STAGES x BR_BYTES
    unsigned char* sB = sBr + STAGES * BR_BYTES;      // 2026-09-28: two decoded steps, 2 x B_BYTES
    float* lut = (float*)(sB + 2 * B_BYTES);          // 2026-09-28: 256 floats
    lut[threadIdx.x] = E4M3_LUT[threadIdx.x];

    // 2026-09-28: Grouped rasterization; it only changes which CTA computes which tile.
    constexpr unsigned GROUP_M = 8;
    const unsigned num_n = gridDim.x, num_m = gridDim.y;
    const unsigned pid = blockIdx.y * num_n + blockIdx.x;
    const unsigned first_m = (pid / (GROUP_M * num_n)) * GROUP_M;
    const unsigned group_rows = min(num_m - first_m, GROUP_M);
    const unsigned cta_m = (first_m + pid % group_rows) * BM;
    const unsigned cta_n = ((pid % (GROUP_M * num_n)) / group_rows) * BN;

    const unsigned tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
    const unsigned gid = lane >> 2, tig = lane & 3u, lmat = lane >> 3, lrow = lane & 7u;
    const unsigned wm0 = (warp / WARPS_N) * WM, wn0 = (warp % WARPS_N) * WN;
    const unsigned k_blocks = K / 128, n_block = cta_n / 128, n_steps = K / BK;
    const int rows_left = (int)M - (int)(cta_m + wm0);
    const int mi_valid = rows_left <= 0 ? 0 : min(MI, (rows_left + 15) / 16);

    auto load_stage = [&](unsigned step, unsigned stage) {
        const unsigned k0 = step * BK;
        #pragma unroll
        for (unsigned c = tid; c < (unsigned)BM * 4; c += THREADS) {
            const unsigned row = c >> 2, ch = c & 3u, gr = cta_m + row;
            const bool ok = gr < M;
            cp16(smem_addr(sA + stage * A_BYTES + swz(row, ch)),
                 A + (unsigned long long)(ok ? gr : 0) * K + k0 + ch * 8, ok);
        }
        {
            const unsigned row = tid >> 1, half = tid & 1u;
            cp16(smem_addr(sBr + stage * BR_BYTES + row * ROW_BR + half * 16),
                 B + (unsigned long long)(cta_n + row) * K + k0 + half * 16, true);
        }
    };

    float inner[MI][NI][4], outer[MI][NI][4];
    #pragma unroll
    for (int mi = 0; mi < MI; mi++)
        #pragma unroll
        for (int ni = 0; ni < NI; ni++)
            #pragma unroll
            for (int r = 0; r < 4; r++) { inner[mi][ni][r] = 0.0f; outer[mi][ni][r] = 0.0f; }

    // 2026-09-28: Thread t copies and decodes the same 16 weight bytes (row t / 2, half t % 2), so it may decode a
    // stage once its own cp.async group landed; the one barrier per step publishes the decoded slice and the A
    // tile for the next step's MMAs while this step's MMAs of other warps overlap the decode.
    auto decode = [&](unsigned stage, unsigned slot) {
        const unsigned row = tid >> 1, half = tid & 1u;
        const uint4 raw = *(const uint4*)(sBr + stage * BR_BYTES + row * ROW_BR + half * 16);
        const unsigned char* b = (const unsigned char*)&raw;
        #pragma unroll
        for (int c = 0; c < 2; c++) {
            uint4 out;
            unsigned* w = (unsigned*)&out;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const unsigned lo = __bfloat16_as_ushort(__float2bfloat16(lut[b[c * 8 + 2 * i]]));
                const unsigned hi = __bfloat16_as_ushort(__float2bfloat16(lut[b[c * 8 + 2 * i + 1]]));
                w[i] = lo | (hi << 16);
            }
            *(uint4*)(sB + slot * B_BYTES + swz(row, half * 2 + c)) = out;
        }
    };

    #pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
        if ((unsigned)s < n_steps) load_stage(s, s);
        cp_commit();
    }
    cp_wait<STAGES - 2>();
    __syncthreads();  // 2026-09-28: the LUT is written
    decode(0, 0);
    __syncthreads();
    for (unsigned step = 0; step < n_steps; step++) {
        {
            const unsigned nxt = step + STAGES - 1;
            if (nxt < n_steps) load_stage(nxt, nxt % STAGES);
            cp_commit();
        }
        const unsigned stage = step % STAGES;
        const unsigned a_base = smem_addr(sA + stage * A_BYTES), b_base = smem_addr(sB + (step & 1u) * B_BYTES);
        #pragma unroll
        for (int ks = 0; ks < 2; ks++) {
            unsigned bf[NI][2];
            #pragma unroll
            for (int j = 0; j < NI / 2; j++) {
                const unsigned nrow = wn0 + j * 16 + lrow + ((lmat >> 1) << 3);
                ldsm_x4(b_base + swz(nrow, 2 * ks + (lmat & 1u)), bf[2 * j][0], bf[2 * j][1], bf[2 * j + 1][0],
                        bf[2 * j + 1][1]);
            }
            #pragma unroll
            for (int mi = 0; mi < MI; mi++) {
                if (mi < mi_valid) {
                    unsigned af[4];
                    ldsm_x4(a_base + swz(wm0 + mi * 16 + (lane & 15u), 2 * ks + (lane >> 4)), af[0], af[1], af[2],
                            af[3]);
                    #pragma unroll
                    for (int ni = 0; ni < NI; ni++) mma_bf16(inner[mi][ni], af, bf[ni][0], bf[ni][1]);
                }
            }
        }
        if ((step + 1) % (128 / BK) == 0) {
            const float scale = block_scale[n_block * k_blocks + (step * BK) / 128];
            #pragma unroll
            for (int mi = 0; mi < MI; mi++)
                #pragma unroll
                for (int ni = 0; ni < NI; ni++)
                    #pragma unroll
                    for (int r = 0; r < 4; r++) {
                        outer[mi][ni][r] += inner[mi][ni][r] * scale;
                        inner[mi][ni][r] = 0.0f;
                    }
        }
        if (step + 1 < n_steps) {
            cp_wait<STAGES - 2>();
            decode((step + 1) % STAGES, (step + 1) & 1u);
        }
        __syncthreads();  // 2026-09-28: slice step + 1 decoded and its A tile landed; stage `step` is free
    }
    cp_wait<0>();

    #pragma unroll
    for (int mi = 0; mi < MI; mi++) {
        const unsigned r0 = cta_m + wm0 + mi * 16 + gid, r1 = r0 + 8;
        #pragma unroll
        for (int ni = 0; ni < NI; ni++) {
            const unsigned col = cta_n + wn0 + ni * 8 + tig * 2;
            if (r0 < M) {
                const unsigned lo = __bfloat16_as_ushort(__float2bfloat16(outer[mi][ni][0]));
                const unsigned hi = __bfloat16_as_ushort(__float2bfloat16(outer[mi][ni][1]));
                *(unsigned*)&C[(unsigned long long)r0 * N + col] = lo | (hi << 16);
            }
            if (r1 < M) {
                const unsigned lo = __bfloat16_as_ushort(__float2bfloat16(outer[mi][ni][2]));
                const unsigned hi = __bfloat16_as_ushort(__float2bfloat16(outer[mi][ni][3]));
                *(unsigned*)&C[(unsigned long long)r1 * N + col] = lo | (hi << 16);
            }
        }
    }
}
