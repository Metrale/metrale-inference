// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-21: dense_gemv_bf16_batchm (dense_gemv_bf16_batchm.cu) for 2..FEW_MAX_M rows with the
// latency taken out. The runtime-M body walks K in 64-vector slabs: each slab stages every A row
// in shared memory, waits at a barrier, loads one B vector per thread, accumulates (acc[] in local
// memory, as M is not a compile-time count) and waits at a second barrier. At GLM-5.3's decode
// widths (two verify rows) a call is a chain of K / 512 such rounds, each a DRAM trip for B; on
// GB10 the router (288 x 4096) took 16.6 us a call against a 10 us DRAM floor.
// Here, per thread:
//   - every B vector of up to FEW_PF slabs is loaded before any is used (one DRAM trip, not one
//     per slab);
//   - the A vectors are read from global memory (L1/L2: the four output groups of a block and
//     every block read the same rows), so no slab barrier;
//   - the row count is a template parameter, so acc[] is in registers.
//
// Owner: gb10 kernels.
// Invariants:
// - Each row's result is bit-identical to dense_gemv_bf16_batchm's row (and so dense_gemv_bf16's):
//   the thread's kv sequence (lane, lane + 64, ...), the lo-then-hi adds per row, the scalar tail
//   over K % 8, the warp shuffle and the two-warp sum are the same operations in the same order;
//   the common build passes --fmad=false. Only where the operands are read from changes.
//   glm5next_batchm_few_gate (gpu examples) checks it byte for byte.
// - A is [M, K] contiguous, B is [N, K] row-major, row t of C starts at C + t * out_stride. K % 8 == 0
//   (16-byte rows, as dense_gemv_bf16_batchm).
// - Launch: grid (ceil(N / 4), 1, 1), block (256, 1, 1) (the _pair entry: the sum of both). M outside 2..FEW_MAX_M writes nothing
//   (the host sends only those).

#include "dense_gemv_bf16_batchm_few.cuh"


extern "C" __global__ void __launch_bounds__(bmfew::BLOCK) dense_gemv_bf16_batchm_few(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int out_stride
) {
    bmfew::few_dispatch<bmfew::N_PER_BLOCK, __nv_bfloat16>(A, B, C, M, N, K, out_stride, blockIdx.x);
}

extern "C" __global__ void __launch_bounds__(bmfew::BLOCK) dense_gemv_bf16_batchm_few_fp32out(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    float* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int out_stride
) {
    bmfew::few_dispatch<bmfew::N_PER_BLOCK, float>(A, B, C, M, N, K, out_stride, blockIdx.x);
}

// 2026-10-21: Two FP32-out projections of M rows in one launch (default-on
// METRALE_GLM_DSA_SELECTOR_PAIR: a DSA verify's selector query wq_b over q_resid and head weights
// weights_proj over the layer input). Blocks [0, ceil(N0 / 4)) are dense_gemv_bf16_batchm_few_fp32out
// on (A0, B0, C0, N0, K0, out_stride0) at block blockIdx.x, the rest the same on the second operand
// set at block blockIdx.x - ceil(N0 / 4); each output has that entry's bytes. Grid
// (ceil(N0 / 4) + ceil(N1 / 4), 1, 1).
extern "C" __global__ void __launch_bounds__(bmfew::BLOCK) dense_gemv_bf16_batchm_few_fp32out_pair(
    const __nv_bfloat16* __restrict__ A0,
    const __nv_bfloat16* __restrict__ B0,
    float* __restrict__ C0,
    unsigned int N0,
    unsigned int K0,
    unsigned int out_stride0,
    const __nv_bfloat16* __restrict__ A1,
    const __nv_bfloat16* __restrict__ B1,
    float* __restrict__ C1,
    unsigned int N1,
    unsigned int K1,
    unsigned int out_stride1,
    unsigned int M
) {
    const unsigned int g0 = (N0 + bmfew::N_PER_BLOCK - 1) / bmfew::N_PER_BLOCK;
    if (blockIdx.x < g0) {
        bmfew::few_dispatch<bmfew::N_PER_BLOCK, float>(A0, B0, C0, M, N0, K0, out_stride0, blockIdx.x);
    } else {
        bmfew::few_dispatch<bmfew::N_PER_BLOCK, float>(A1, B1, C1, M, N1, K1, out_stride1, blockIdx.x - g0);
    }
}
