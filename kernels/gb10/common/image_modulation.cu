// SPDX-License-Identifier: MIT OR Apache-2.0
// 2026-10-07: Image-transformer residual family point, not a registered model.
// Qwen Image 2.1 rounds each eager BF16 elementwise result separately. Do not
// contract the residual product/add or scale add/multiply into a single cast.
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>

extern "C" __global__ void image_modulation_scale_bf16(
    const __nv_bfloat16* normalized, const __nv_bfloat16* params,
    const uint32_t* selected_rows, __nv_bfloat16* output,
    uint32_t rows, uint32_t width, uint32_t param_stride, uint32_t offset) {
    const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= uint64_t(rows) * width) return;
    const uint32_t row = i / width, col = i % width;
    const float scale = __bfloat162float(params[uint64_t(selected_rows[row]) * param_stride + offset + col]);
    const float factor = __bfloat162float(__float2bfloat16_rn(1.0f + scale));
    output[i] = __float2bfloat16_rn(__bfloat162float(normalized[i]) * factor);
}

extern "C" __global__ void image_modulation_residual_bf16(
    const __nv_bfloat16* hidden, const __nv_bfloat16* branch,
    const __nv_bfloat16* params, const uint32_t* selected_rows,
    __nv_bfloat16* output, uint32_t rows, uint32_t width,
    uint32_t param_stride, uint32_t offset) {
    const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= uint64_t(rows) * width) return;
    const uint32_t row = i / width, col = i % width;
    const float gate = __bfloat162float(params[uint64_t(selected_rows[row]) * param_stride + offset + col]);
    const float activated = __bfloat162float(__float2bfloat16_rn(tanhf(gate)));
    const float product = __bfloat162float(__float2bfloat16_rn(activated * __bfloat162float(branch[i])));
    output[i] = __float2bfloat16_rn(__bfloat162float(hidden[i]) + product);
}
