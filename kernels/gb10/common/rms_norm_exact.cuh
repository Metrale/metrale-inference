// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-28: The row RMSNorm steps of rms_norm.cu, for fused kernels that must reproduce its
// bytes: BF16 pair unpack/pack, the xor-shuffle warp sum, and the block sum of the per-thread
// partial sums of squares.
//
// Owner: gb10 kernels.
// Invariants:
// - Each helper is the code rms_norm.cu inlines into rms_norm and rms_norm_residual, operation
//   for operation: pairs are widened exactly, packed with __float2bfloat16 (round to nearest
//   even), warp sums use the offsets 16, 8, 4, 2, 1, and the block sum has warp 0 add the
//   first ceil(blockDim.x / 32) warp partials and zero for the rest. Compiled under the same
//   target flags (--fmad=false in every gb10 KERNEL.toml), the fused kernels give the same
//   bits; model-arch examples residual_add_rms_norm_exact_microtest and
//   rms_norm_act_quant_microtest check it byte for byte.
// - rmsx_block_sum needs a __shared__ float[32] and a block of at most 1024 threads, and ends
//   with a __syncthreads so every thread reads the same total.

#pragma once

#include <cuda_bf16.h>

__device__ __forceinline__ void rmsx_unpack(unsigned int packed, float& v0, float& v1) {
    v0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed & 0xFFFF)));
    v1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed >> 16)));
}

__device__ __forceinline__ unsigned int rmsx_pack(float v0, float v1) {
    unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v0));
    unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v1));
    return lo | (hi << 16);
}

__device__ __forceinline__ float rmsx_warp_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// 2026-09-28: The block total of `sum_sq`, reduced as rms_norm reduces it.
__device__ __forceinline__ float rmsx_block_sum(float sum_sq, float* warp_sums) {
    sum_sq = rmsx_warp_sum(sum_sq);
    unsigned int warp_id = threadIdx.x / 32;
    unsigned int lane_id = threadIdx.x % 32;
    if (lane_id == 0) {
        warp_sums[warp_id] = sum_sq;
    }
    __syncthreads();
    if (warp_id == 0) {
        float val = (lane_id < (blockDim.x + 31) / 32) ? warp_sums[lane_id] : 0.0f;
        val = rmsx_warp_sum(val);
        if (lane_id == 0) {
            warp_sums[0] = val;
        }
    }
    __syncthreads();
    return warp_sums[0];
}
