// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-07: Projection bias before the final BF16 store. Reuses the FP32
// output of dense_gemv_bf16_fp32out, avoiding a second matrix-vector kernel.
// Owner: gb10 kernels. Named LKB residual: projection_bias_before_cast.
// accum [elements] FP32, bias [cols] BF16, output [elements] BF16.
// Host validates nonzero geometry, alignment, address overflow and non-aliasing.
// Launch ceil(elements/256) blocks of 256. cols broadcasts across token rows.
// Target linkage and end-to-end projection parity are pending; the standalone
// GPU epilogue test passes. No native model support or performance claimed.

#include <cuda_bf16.h>

extern "C" __global__ void projection_bias_bf16(
    const float* __restrict__ accum,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ output,
    unsigned int elements,
    unsigned int cols
) {
    const unsigned long long i =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= elements) return;
    const float sum = accum[i] + __bfloat162float(bias[i % cols]);
    output[i] = __float2bfloat16_rn(sum);
}
