// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-24: Mean pooling for embedding models: the FP32 mean over each packed sequence's
// BF16 token rows. The rows of all sequences sit back to back in `hidden` ([total, dim]); the
// sequence `s` owns rows [cu_seqlens[s], cu_seqlens[s + 1]). Each thread owns one column of
// one sequence and sums its rows in order (row 0 first) in FP32, then divides by the row
// count once, so the result does not depend on the launch shape.
//
// Owner: gb10 kernels.
// Invariants:
// - Grid (num_seqs, ceil(dim / blockDim.x), 1); any block size.
// - cu_seqlens is non-decreasing with cu_seqlens[0] = 0; an empty sequence writes zeros.

#include <cuda_bf16.h>

extern "C" __global__ void mean_pool_masked_f32(
    const __nv_bfloat16* __restrict__ hidden,
    const unsigned int* __restrict__ cu_seqlens,
    float* __restrict__ out,
    unsigned int dim
) {
    const unsigned int s = blockIdx.x;
    const unsigned int col = blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= dim) return;
    const unsigned int lo = cu_seqlens[s];
    const unsigned int hi = cu_seqlens[s + 1];
    float acc = 0.0f;
    for (unsigned int r = lo; r < hi; ++r) {
        acc += __bfloat162float(hidden[(unsigned long long)r * dim + col]);
    }
    const unsigned int n = hi - lo;
    out[(unsigned long long)s * dim + col] = n > 0 ? acc / (float)n : 0.0f;
}
