// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: BF16-weight small-M GEMV on tensor cores: C[t, n] = sum_k A[t, k] * W[n, k].
//
// Owner: gb10 kernels.
// Invariants:
// - Arguments match dense_gemv_bf16_batchm: A is [M, K] contiguous, W is [N, K] row-major,
//   row t of C starts at C + t * out_stride.
// - Requires K % 64 == 0; the host route() declines any other K. Any N: a weight row past N
//   loads zeros and is never stored. A token row past M feeds zeros and is never stored.
// - Launch: grid (ceil(N / 16), 1, 1), block (DTC_WARPS * 32 = 256, 1, 1); NT = 1 in every
//   entry, so a CTA owns one 16-row weight tile.
// - The DTC_WARPS warps split K (interleaved 64-k blocks) and reduce through shared memory
//   in a fixed warp order with no atomics, so the output is deterministic, and a token's
//   partials never mix with another token's.
// - The result is not bit-identical to the CUDA-core GEMV: accumulation happens inside
//   the MMA.
//
// Caller: the MTP drafter (mtp_head), through ops/dense_gemv_tc.rs, for M = 2..=32.
//
// 2026-10-17: The kernel body, its operand placement and its K permutation live in
// dense_gemv_bf16_tc_body.cuh, shared with the FP32-output entries (dense_gemm_tc_f32out.cu).

#include "dense_gemv_bf16_tc_body.cuh"

#define DTC_ENTRY(NAME, NB, NT, KU)                                                     \
    extern "C" __global__ __launch_bounds__(DTC_WARPS * 32) void NAME(                   \
        const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ W,         \
        __nv_bfloat16* __restrict__ C, unsigned int M, unsigned int N, unsigned int K,    \
        unsigned int out_stride) {                                                        \
        dense_gemv_bf16_tc_impl<NB, NT, KU>(A, W, C, M, N, K, out_stride);               \
    }

// 2026-09-25: Entries per token-tile count NB (M <= 8 / 16 / 32); each covers every M up to
// its cap. Geometry measured 2026-09-23 on GB10, one 27B draft position (849 MB of cold
// BF16 weights): weight bytes in flight per thread moved the time, not occupancy. KU=8
// (512 B/thread, 1 CTA/SM) beat KU=4 (2 CTAs/SM) by 3.5-4.4%; KU=2 at 4 CTAs/SM and KU=3/4
// at 3 CTAs/SM were slower than KU=4; NT=2 was slower than KU=8. Against
// dense_gemv_bf16_batchm at M=2/4/8: 3554/3583/3583 us vs 3487/3564/3631 us, and
// 97/117/134 mJ vs 149/168/264 mJ per position. tc16 at KU=8: 3602 us (KU=4: 3737).
// tc32 at KU=4: 3803 us (KU=2: 3885; KU=8 does not fit).




DTC_ENTRY(dense_gemv_bf16_tc8, 1, 1, 8)
DTC_ENTRY(dense_gemv_bf16_tc16, 2, 1, 8)
DTC_ENTRY(dense_gemv_bf16_tc32, 4, 1, 4)
