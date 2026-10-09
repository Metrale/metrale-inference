// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: NVFP4 -> BF16 dequantization on the device:
// out[n, k] = E2M1(nibble) * E4M3(scale[n, k / 16]) * combined_global, rounded to BF16.
//
// Owner: gb10 kernels.
// Invariants:
// - packed is [N, K/2] bytes (column k in the low nibble when k is even, the high nibble
//   when odd), scales is [N, K/16] FP8 E4M3, out is [N, K] row-major.
// - combined_global is always a multiplier: the host (weight_map/fp8_lut.rs) passes
//   1 / weight_global_scale for compressed-tensors checkpoints and weight_scale_2
//   for the others.
// - Launch: grid (N, 1, 1), block (256, 1, 1); one block per row, threads stride over K.






#include <cuda_bf16.h>

#define DQ_GROUP_SIZE 16

// 2026-09-25: E2M1 nibble -> float. Bits: [sign(1)][exp(2)][mantissa(1)].

__device__ __constant__ float DQ_E2M1_LUT[16] = {
    0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

// 2026-09-25: FP8 E4M3 byte -> float, the same values as metrale_core's fp8_e4m3_to_f32:
// bias 7, exp == 0 is subnormal (man * 2^-9), and the NaN code (exp 15, man 7) reads as 0.

__device__ __forceinline__ float dq_fp8_e4m3_decode(unsigned char b) {
    unsigned int sign = (b >> 7) & 1u;
    unsigned int exp = (b >> 3) & 0xFu;
    unsigned int man = b & 0x7u;
    float val;
    if (exp == 0u) {
        val = (float)man * 0.001953125f;
    } else if (exp == 15u && man == 7u) {
        val = 0.0f;
    } else {
        val = (1.0f + (float)man * 0.125f) * exp2f((float)((int)exp - 7));
    }
    return sign ? -val : val;
}


extern "C" __global__ void dequant_nvfp4_to_bf16(
    const unsigned char* __restrict__ packed,
    const unsigned char* __restrict__ scales,
    __nv_bfloat16* __restrict__ out,
    float combined_global,
    unsigned int N,
    unsigned int K
) {
    unsigned int row = blockIdx.x;
    if (row >= N) return;

    const unsigned char* row_packed = packed + (unsigned long long)row * (K / 2);
    const unsigned char* row_scale = scales + (unsigned long long)row * (K / DQ_GROUP_SIZE);
    __nv_bfloat16* row_out = out + (unsigned long long)row * K;

    for (unsigned int col = threadIdx.x; col < K; col += blockDim.x) {
        unsigned int g = col / DQ_GROUP_SIZE;
        float s = dq_fp8_e4m3_decode(row_scale[g]) * combined_global;
        unsigned char byte = row_packed[col >> 1];

        unsigned int nib = (col & 1u) ? ((byte >> 4) & 0xFu) : (byte & 0xFu);
        row_out[col] = __float2bfloat16(DQ_E2M1_LUT[nib] * s);
    }
}

// 2026-10-09: The per-group form for a transient BF16 copy of a weight that a vendor BF16 GEMM
// then reads (metrale-model-layers layers/dense_ffn_lt.rs): out[n, k] = E2M1(nibble) *
// E4M3(scale[n, k / 16]), with NO global scale (the GEMM applies it as alpha). A 2-bit E2M1
// significand times a 4-bit E4M3 significand fits BF16's 8 bits, and every product of these
// formats is a BF16 normal or zero, so each element is exact.
//
// One thread per 16-element group: 8 packed bytes, one scale byte, two 16-byte stores. The row-
// major [N, K/2] and [N, K/16] layouts make group g's bytes packed[8g .. 8g + 8) and its scale
// scales[g]. Launch: grid (min(ceil(groups / 256), 65535 * ...), 1, 1) by the caller, block 256;
// the loop strides by the grid.
extern "C" __global__ void __launch_bounds__(256) dequant_nvfp4_to_bf16_g16(
    const unsigned char* __restrict__ packed,
    const unsigned char* __restrict__ scales,
    __nv_bfloat16* __restrict__ out,
    unsigned long long groups
) {
    for (unsigned long long g = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; g < groups;
         g += (unsigned long long)gridDim.x * blockDim.x) {
        const uint2 q = ((const uint2*)packed)[g];
        const float s = dq_fp8_e4m3_decode(scales[g]);
        __align__(16) __nv_bfloat16 v[16];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const unsigned int byte = ((i < 4 ? q.x : q.y) >> ((i & 3) * 8)) & 0xFFu;
            v[2 * i] = __float2bfloat16_rn(DQ_E2M1_LUT[byte & 0xFu] * s);
            v[2 * i + 1] = __float2bfloat16_rn(DQ_E2M1_LUT[byte >> 4] * s);
        }
        uint4* o = (uint4*)(out + g * 16ull);
        o[0] = ((const uint4*)v)[0];
        o[1] = ((const uint4*)v)[1];
    }
}
