// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

// 2026-10-19: The device helpers attn_v41.cu and attn_v41_kv.cu share: bf16 rounding, the block
// reductions, and the reference quantisers' rounding and element codes (deepseek_v41_ref attn.rs and
// compress.rs; mat.rs holds the host decoders the codes are checked against).
//
// Owner: gb10 kernels (deepseek-v4-flash).
// Invariants: included by attn_v41.cu and attn_v41_kv.cu only; every function here is the one
// definition its name has in those two modules.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <math_constants.h>

#define AV_BLOCK 256

__device__ __forceinline__ float av_bf16r(float x) {
    return __bfloat162float(__float2bfloat16(x));
}

__device__ __forceinline__ float av_block_sum(float v, float* red) {
    const unsigned int tid = threadIdx.x;
    red[tid] = v;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    const float r = red[0];
    __syncthreads();
    return r;
}

__device__ __forceinline__ float av_block_max(float v, float* red) {
    const unsigned int tid = threadIdx.x;
    red[tid] = v;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
        __syncthreads();
    }
    const float r = red[0];
    __syncthreads();
    return r;
}

// 2026-09-25: `2 ** ceil(log2(v))` for a positive normal v, via the IEEE fields (`pow2_ceil` in deepseek_v41_ref/attn.rs).
__device__ __forceinline__ float av_pow2_ceil(float v) {
    const unsigned int bits = __float_as_uint(v);
    const int e = (int)((bits >> 23) & 0xFF) - 127;
    const unsigned int man = bits & 0x7FFFFF;
    const int n = e + (man != 0 ? 1 : 0);
    return __uint_as_float((unsigned int)(n + 127) << 23);
}

// 2026-09-25: e4m3 round trip, RNE, saturating at +-448. act_quant clamps to +-448 first, as the reference does.
__device__ __forceinline__ float av_e4m3_round_trip(float x) {
    const __nv_fp8_storage_t q = __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
    return __half2float(__nv_cvt_fp8_to_halfraw(q, __NV_E4M3));
}

// 2026-10-19: The e4m3 byte of x (RNE, saturating); exact for a value already on the e4m3 grid.
__device__ __forceinline__ unsigned int av_e4m3_code(float x) {
    return (unsigned int)__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
}

// 2026-10-19: The value of an e4m3 byte (`mat::e4m3`).
__device__ __forceinline__ float av_e4m3_value(unsigned int code) {
    return __half2float(__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)code, __NV_E4M3));
}

// 2026-10-19: The e8m0 byte of a power-of-two scale: its biased exponent. The quantisers' scales are
// normal (their amax floors keep them at or above 2^-126), so the byte is never 0 or 255.
__device__ __forceinline__ unsigned int av_e8m0_code(float pow2) {
    return (__float_as_uint(pow2) >> 23) & 0xFF;
}

// 2026-10-19: The value of an e8m0 byte, `2^(b - 127)`; byte 0 is 2^-127 (`mat::e8m0`).
__device__ __forceinline__ float av_e8m0_value(unsigned int b) {
    return b != 0 ? __uint_as_float(b << 23) : __uint_as_float(0x00400000u);
}

// 2026-10-19: The e2m1 code of y: the magnitude rounded to the grid {0, .5, 1, 1.5, 2, 3, 4, 6}, ties to the
// even code, in bits 0..2 and the sign of y (signbit, so -0 keeps it) in bit 3 (`_to_e2m1_rne` in
// deepseek_v41_ref/ds41_ref_shims.py).
__device__ __forceinline__ unsigned int av_e2m1_code(float y) {
    const float grid[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float a = fabsf(y);
    int lo = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        if (grid[i] <= a) lo = i;
    }
    const int hi = lo + 1 < 7 ? lo + 1 : 7;
    const float dlo = a - grid[lo];
    const float dhi = grid[hi] - a;
    const bool pick_hi = dhi < dlo || (dhi == dlo && (hi % 2) == 0 && (lo % 2) == 1);
    return (unsigned int)(pick_hi ? hi : lo) | (signbit(y) ? 8u : 0u);
}

// 2026-10-19: The value of the low nibble of `code` (`mat::e2m1`).
__device__ __forceinline__ float av_e2m1_value(unsigned int code) {
    const float grid[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float v = grid[code & 7];
    return (code & 8) ? -v : v;
}

// 2026-09-25: y rounded to the e2m1 grid, keeping its sign: the value of its code.
__device__ __forceinline__ float av_e2m1_rne(float y) {
    return av_e2m1_value(av_e2m1_code(y));
}

// 2026-10-19: The scale of an fp4_act_quant block of amax `amax`: mode 0 = e8m0 (pow2 ceil of amax/6, amax
// >= 6*2^-126), mode 1 = e4m3 (amax/6 through e4m3, amax >= 6*2^-9).
__device__ __forceinline__ float av_fp4_scale(float amax, unsigned int mode) {
    if (mode == 0) return av_pow2_ceil(fmaxf(amax, 6.0f * 1.1754943508222875e-38f) / 6.0f);
    return av_e4m3_round_trip(fmaxf(amax, 6.0f * 0.001953125f) / 6.0f);
}

// 2026-10-19: The scale of an act_quant block of amax `amax`: pow2 ceil of amax/448, amax >= 1e-4.
__device__ __forceinline__ float av_fp8_scale(float amax) {
    return av_pow2_ceil(fmaxf(amax, 1e-4f) / 448.0f);
}
