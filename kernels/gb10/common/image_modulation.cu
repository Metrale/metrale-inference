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

// 2026-10-07: Separate BF16 weight multiplication after a normalized row has
// already rounded to BF16. This preserves Diffusers RMSNorm's storage boundary.
extern "C" __global__ void image_head_weight_bf16(
    const __nv_bfloat16* normalized, const __nv_bfloat16* weight,
    __nv_bfloat16* output, uint32_t rows, uint32_t width) {
    const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= uint64_t(rows) * width) return;
    output[i] = __float2bfloat16_rn(__bfloat162float(normalized[i]) * __bfloat162float(weight[i % width]));
}

// 2026-10-07: Adjacent real/imaginary pairs, shared FP32 cis[sequence,64,2].
// Axes partition the 64 pairs as 8 frame / 28 height / 28 width.
// Pinned Torch CUDA contracts opposite products for real/imaginary components.
// Identical-operand controls distinguish this ordering; native CPU cis generation
// still differs, so the overall rotary gate remains diagnostic.
extern "C" __global__ void image_rope_complex_bf16(
    const __nv_bfloat16* input, const float* cis, __nv_bfloat16* output,
    uint32_t samples, uint32_t sequence, uint32_t heads) {
    const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= uint64_t(samples) * sequence * heads * 64) return;
    const uint64_t token = (i / (heads * 64)) % sequence;
    const uint64_t pair = i % 64;
    const float c = cis[(token * 64 + pair) * 2], s = cis[(token * 64 + pair) * 2 + 1];
    const float x = __bfloat162float(input[2*i]), y = __bfloat162float(input[2*i+1]);
    output[2*i] = __float2bfloat16_rn(__fmaf_rn(x,c,-__fmul_rn(y,s)));
    output[2*i+1] = __float2bfloat16_rn(__fmaf_rn(y,c,__fmul_rn(x,s)));
}

// 2026-10-07: Eager BF16 SiLU storage boundary before the separate up multiply.
// Existing fused MoE SiLU kernels keep that intermediate in FP32; do not alias
// this entry to them or change their rounding. In-place output is supported.
// 2026-10-07: Null up explicitly selects unary SiLU for time conditioning.
extern "C" __global__ void image_silu_staged_mul_bf16(
    const __nv_bfloat16* gate, const __nv_bfloat16* up,
    __nv_bfloat16* output, uint32_t elements) {
    const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= elements) return;
    const float g = __bfloat162float(gate[i]);
    const float activated = __bfloat162float(__float2bfloat16_rn(g / (1.0f + expf(-g))));
    output[i] = __float2bfloat16_rn(up ? activated * __bfloat162float(up[i]) : activated);
}

// 2026-10-07: Pinned 256-channel time embedding, cosine half before sine half.
// Input timestep has already rounded to BF16; freqs are explicit FP32[128].
extern "C" __global__ void image_timestep_bf16(
    const __nv_bfloat16* timestep, const float* freqs, __nv_bfloat16* output,
    uint32_t rows) {
    const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= uint64_t(rows) * 256) return;
    const uint32_t channel = i % 256;
    const float scaled = __fmul_rn(__bfloat162float(timestep[i / 256]), 1000.0f);
    const float angle = __fmul_rn(scaled, freqs[channel % 128]);
    output[i] = __float2bfloat16_rn(channel < 128 ? cosf(angle) : sinf(angle));
}

// 2026-10-07: Pinned Qwen3-VL text RoPE uses split halves and BF16 products
// before the BF16 sum. Coefficients are BF16 [tokens,64], already generated.
// This additive entry leaves the existing FP32-coefficient RoPE kernels intact.
extern "C" __global__ void image_text_rope_bf16(
    __nv_bfloat16* values, const __nv_bfloat16* cosines,
    const __nv_bfloat16* sines, uint32_t tokens, uint32_t heads) {
    const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= uint64_t(tokens) * heads * 64) return;
    const uint32_t pair = i % 64;
    const uint64_t row = i / 64;
    const uint64_t coefficient = (row / heads) * 64 + pair;
    const float x = __bfloat162float(values[row * 128 + pair]);
    const float y = __bfloat162float(values[row * 128 + pair + 64]);
    const float c = __bfloat162float(cosines[coefficient]);
    const float s = __bfloat162float(sines[coefficient]);
    const float xc = __bfloat162float(__float2bfloat16_rn(x*c));
    const float ys = __bfloat162float(__float2bfloat16_rn(-y*s));
    const float yc = __bfloat162float(__float2bfloat16_rn(y*c));
    const float xs = __bfloat162float(__float2bfloat16_rn(x*s));
    values[row*128+pair] = __float2bfloat16_rn(xc+ys);
    values[row*128+pair+64] = __float2bfloat16_rn(yc+xs);
}
