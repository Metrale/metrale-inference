// SPDX-License-Identifier: MIT OR Apache-2.0
// 2026-10-07: Correctness residual: row-major packed GPT-OSS expert GEMV.
// Layout [N,K/32,16] E2M1 bytes and [N,K/32] raw exponent bytes.
// Low nibble is the even column. Decode follows pinned Transformers v4.55.0
// convert_moe_packed_tensors: BF16(ldexp(E2M1, scale_byte-127)). In particular
// byte 0 is NOT zero; byte 255 is not silently zeroed and can overflow. This
// is the pinned unpack policy, not a claim that reserved E8M0 codes are valid.
// FP32 multiply/accumulate; a warp reduction; one BF16 output BEFORE bias.
// Bias, activation and routing reduction are separate operators. No model
// registration or performance claim. Four warps/block, one output row/warp.
#include <cuda_bf16.h>
#include <math.h>

__device__ __forceinline__ float gpt_oss_unpack(unsigned char code, unsigned char scale) {
    const float magnitude[8] = {0.0f,0.5f,1.0f,1.5f,2.0f,3.0f,4.0f,6.0f};
    float value = magnitude[code & 7];
    if (code & 8) value = -value;
    return __bfloat162float(__float2bfloat16_rn(ldexpf(value, int(scale) - 127)));
}

__device__ __forceinline__ void gpt_oss_mxfp4_rows(
    const unsigned char* __restrict__ blocks,
    const unsigned char* __restrict__ scales,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    unsigned int rows, unsigned int cols) {
    const unsigned row = blockIdx.x * 4 + threadIdx.x / 32;
    const unsigned lane = threadIdx.x & 31;
    if (row >= rows) return;
    float accum = 0.0f;
    for (unsigned col = lane; col < cols; col += 32) {
        const unsigned char byte = blocks[size_t(row) * (cols / 2) + col / 2];
        const unsigned char code = (col & 1) ? byte >> 4 : byte & 15;
        const unsigned char scale = scales[size_t(row) * (cols / 32) + col / 32];
        accum = fmaf(gpt_oss_unpack(code, scale), __bfloat162float(input[col]), accum);
    }
    for (unsigned shift = 16; shift; shift >>= 1)
        accum += __shfl_down_sync(0xffffffffU, accum, shift);
    if (lane == 0) output[row] = __float2bfloat16_rn(accum);
}

// 2026-10-07: Legacy launch retains its original row, lane, and FMA ordering.
extern "C" __global__ void gpt_oss_mxfp4_gemv_bf16(
    const unsigned char* blocks, const unsigned char* scales,
    const __nv_bfloat16* input, __nv_bfloat16* output,
    unsigned rows, unsigned cols) {
    gpt_oss_mxfp4_rows(blocks, scales, input, output, rows, cols);
}

// 2026-10-07: Grid Y is the selected slot, not expert ID. Raw contiguous checkpoint
// expert strides are derived from validated rows/cols; input_stride is zero for
// a shared normalized token or cols for per-slot activations. Host ID validation
// remains mandatory; these guards additionally prevent bad IDs reading weights.
extern "C" __global__ void gpt_oss_mxfp4_selected_bf16(
    const unsigned char* blocks, const unsigned char* scales,
    const __nv_bfloat16* input, const unsigned* ids,
    __nv_bfloat16* output, unsigned rows, unsigned cols, unsigned input_stride) {
    const unsigned slot = blockIdx.y;
    if (slot >= 4) return;
    const unsigned row = blockIdx.x * 4 + threadIdx.x / 32;
    bool valid = true;
    for (unsigned a = 0; a < 4; ++a) {
        valid &= ids[a] < 32;
        for (unsigned b = 0; b < a; ++b) valid &= ids[a] != ids[b];
    }
    if (!valid) {
        if (row < rows && !(threadIdx.x & 31))
            output[size_t(slot) * rows + row] = __float2bfloat16_rn(nanf(""));
        return;
    }
    const size_t expert = ids[slot];
    gpt_oss_mxfp4_rows(blocks + expert * rows * (cols / 2),
        scales + expert * rows * (cols / 32), input + size_t(slot) * input_stride,
        output + size_t(slot) * rows, rows, cols);
}
