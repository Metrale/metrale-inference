// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-10: l2_warm_ldcg: read `n16` 16-byte words at `p` through L2 (ld.global.cg) and drop
// them, so a later kernel reading the same bytes hits L2. GB10 (24 MB L2) streams 8 MB from
// DRAM in ~22 us and from L2 in ~7 us; the prefetch.global.L2 hint alone barely moved that.
//
// Owner: gb10 kernels.
// Invariants:
// - Reads only; the one store is behind a condition no input can meet in practice and goes to
//   `sink`, a scratch word no caller reads, so the compiler keeps the loads.
// - `p` is 16-byte aligned (an allocation base or a 16-byte multiple into one) and the
//   `n16 * 16` bytes from it lie inside one allocation.
// Callers: model-arch glm5next_layer::l2_warm (opt-in METRALE_GLM_AR_L2_PREFETCH), on a side
// stream forked at an all-reduce.

#include <cstdint>

extern "C" __global__ void l2_warm_ldcg(
    const uint4* __restrict__ p,
    unsigned long long n16,
    unsigned int* __restrict__ sink
) {
    unsigned int acc = 0;
    const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < n16;
         i += stride) {
        const uint4 v = __ldcg(p + i);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x5a17c0deu) {
        *sink = acc;
    }
}
