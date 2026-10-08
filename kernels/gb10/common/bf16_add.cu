// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: `bf16_add_inplace`: dst[i] += src[i] for every i < n, one thread per element.
// Owner: gb10 kernels. Callers: `NcclBackend::all_reduce_2rank` and the GLM-5-Next MTP layer's residual add.
// Invariants: none beyond the types.
#include <cuda_bf16.h>

extern "C" __global__ void bf16_add_inplace(
    __nv_bfloat16* __restrict__ dst,
    const __nv_bfloat16* __restrict__ src,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = __hadd(dst[i], src[i]);
    }
}

// 2026-10-08: `bf16_add_rank_sum`: the reduce step of the one-shot all-reduce at world >= 3.
// `dst` holds this rank's partial; `peers` holds the other ranks' partials, one slot of
// `slot_elems` elements per peer, peer r in slot (r < my_rank ? r : r - 1). Every rank sums the
// world's partials in rank order 0..n_ranks-1 in FP32 and rounds once, so all ranks write the same
// bytes. Callers: `NcclBackend::all_reduce_oneshot`.
extern "C" __global__ void bf16_add_rank_sum(
    __nv_bfloat16* __restrict__ dst,
    const __nv_bfloat16* __restrict__ peers,
    long long slot_elems,
    int n_ranks,
    int my_rank,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    float acc = 0.0f;
    for (int r = 0; r < n_ranks; ++r) {
        __nv_bfloat16 v = (r == my_rank) ? dst[i] : peers[(long long)(r < my_rank ? r : r - 1) * slot_elems + i];
        acc += __bfloat162float(v);
    }
    dst[i] = __float2bfloat16_rn(acc);
}
