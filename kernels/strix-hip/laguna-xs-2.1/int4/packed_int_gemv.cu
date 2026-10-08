// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-07: Packed INT4 / INT8 weight-only GEMVs for gfx1151 (wave32): a dense W4A16 /
// W8A16 decode GEMV and a grouped-expert GEMV over a per-expert pointer table. Weights
// are dequantized in registers; products and sums are fp32.
//
// Owner: strix-hip kernels (laguna-xs-2.1/int4).
// Invariants:
// - Layout and lane arithmetic are packed_int_dequant.cuh's. K % 128 == 0, N >= 1.
// - Launch (both families): block (256, 1, 1) = 8 waves of 32; wave w of block b computes
//   output column n = b * 8 + w of one output row; grid (ceil(N / 8), rows, 1), where rows
//   is M (dense) or S = tokens * top_k slots (grouped). Lane l reads packed words l, l + 32,
//   ...; the 32 lane partials reduce by xor-shuffle tree 16, 8, 4, 2, 1 and lane 0 stores
//   BF16 (round to nearest even).
// - Dense: y[m, n] = sum_k x[m, k] * W[n, k]; x [M, K], y [M, N] BF16 row-major;
//   W words [N, K * BITS / 32], scales BF16 [N, K / 128].
// - Grouped: slot s (0 <= s < S) uses expert e = expert_ids[s] and activation row
//   s / x_row_div (x_row_div = top_k when every slot of a token reads the token's hidden
//   state, gate/up; 1 when each slot has its own row, down). It writes y[s, n]. Expert e's
//   words and scales are at w_ptrs[e] and scale_ptrs[e] in the dense layout. A slot whose
//   expert id is outside [0, num_experts) or whose pointers are null writes 0: padding
//   slots never read memory.
// - Entry points: packed_int4_gemv_g128, packed_int8_gemv_g128,
//   moe_packed_int4_gemv_ptrtable_g128, moe_packed_int8_gemv_ptrtable_g128 (module
//   packed_int_gemv). No Rust dispatch launches them yet; the lookup names are pinned by
//   crates/kernels/tests/strix_hip_laguna_int4.rs.
//
// TODO(gfx1151): hipcc compile, then launch against the CPU reference
// (metrale_model_layers::quant_format::packed_int) on the device; nothing here has run on
// a GPU. Optimization (several columns per wave, 128-bit word loads, LDS-staged
// activations, V_DOT4_I32_IU8) comes after that parity gate.

#include "packed_int_dequant.cuh"

#define PI_THREADS 256
#define PI_WAVE 32
#define PI_COLS_PER_BLOCK (PI_THREADS / PI_WAVE)

// 2026-10-07: Sum over the 32 lanes of a wave, xor-shuffle tree 16, 8, 4, 2, 1. The HIP
// source mirror widens the mask literal to 64 bits (crates/kernels/build_hip.rs).
__device__ __forceinline__ float pi_wave_sum(float v) {
    v += __shfl_xor_sync(0xffffffff, v, 16);
    v += __shfl_xor_sync(0xffffffff, v, 8);
    v += __shfl_xor_sync(0xffffffff, v, 4);
    v += __shfl_xor_sync(0xffffffff, v, 2);
    v += __shfl_xor_sync(0xffffffff, v, 1);
    return v;
}

// 2026-10-07: One output value: row `words`/`scales` of the weight against activation row x.
template <int BITS>
__device__ __forceinline__ void pi_column(const unsigned int* __restrict__ words,
                                          const unsigned short* __restrict__ scales,
                                          const unsigned short* __restrict__ x,
                                          unsigned short* __restrict__ out, int n, int k) {
    const int lane = threadIdx.x % PI_WAVE;
    const unsigned int* row_words = words + (size_t)n * (k / (32 / BITS));
    const unsigned short* row_scales = scales + (size_t)n * (k / PI_GROUP);
    float partial = pi_row_lane_partial<BITS>(row_words, row_scales, x, k, lane, PI_WAVE);
    float sum = pi_wave_sum(partial);
    if (lane == 0) {
        *out = pi_f32_to_bf16_rn(sum);
    }
}

template <int BITS>
__device__ __forceinline__ void pi_dense(const unsigned short* __restrict__ x,
                                         const unsigned int* __restrict__ words,
                                         const unsigned short* __restrict__ scales,
                                         unsigned short* __restrict__ y, int n_total, int k) {
    const int n = blockIdx.x * PI_COLS_PER_BLOCK + threadIdx.x / PI_WAVE;
    const int m = blockIdx.y;
    if (n >= n_total) {
        return;
    }
    pi_column<BITS>(words, scales, x + (size_t)m * k, y + (size_t)m * n_total + n, n, k);
}

template <int BITS>
__device__ __forceinline__ void pi_grouped(const unsigned short* __restrict__ x,
                                           const unsigned long long* __restrict__ w_ptrs,
                                           const unsigned long long* __restrict__ scale_ptrs,
                                           const int* __restrict__ expert_ids,
                                           unsigned short* __restrict__ y, int num_experts,
                                           int x_row_div, int n_total, int k) {
    const int n = blockIdx.x * PI_COLS_PER_BLOCK + threadIdx.x / PI_WAVE;
    const int s = blockIdx.y;
    if (n >= n_total) {
        return;
    }
    unsigned short* out = y + (size_t)s * n_total + n;
    const int e = expert_ids[s];
    const unsigned int* words = 0;
    const unsigned short* scales = 0;
    if (e >= 0 && e < num_experts) {
        words = (const unsigned int*)w_ptrs[e];
        scales = (const unsigned short*)scale_ptrs[e];
    }
    if (words == 0 || scales == 0) {
        if (threadIdx.x % PI_WAVE == 0) {
            *out = 0;
        }
        return;
    }
    pi_column<BITS>(words, scales, x + (size_t)(s / x_row_div) * k, out, n, k);
}

extern "C" __global__ void __launch_bounds__(PI_THREADS) packed_int4_gemv_g128(
    const unsigned short* __restrict__ x, const unsigned int* __restrict__ words,
    const unsigned short* __restrict__ scales, unsigned short* __restrict__ y, int n, int k) {
    pi_dense<4>(x, words, scales, y, n, k);
}

extern "C" __global__ void __launch_bounds__(PI_THREADS) packed_int8_gemv_g128(
    const unsigned short* __restrict__ x, const unsigned int* __restrict__ words,
    const unsigned short* __restrict__ scales, unsigned short* __restrict__ y, int n, int k) {
    pi_dense<8>(x, words, scales, y, n, k);
}

extern "C" __global__ void __launch_bounds__(PI_THREADS) moe_packed_int4_gemv_ptrtable_g128(
    const unsigned short* __restrict__ x, const unsigned long long* __restrict__ w_ptrs,
    const unsigned long long* __restrict__ scale_ptrs, const int* __restrict__ expert_ids,
    unsigned short* __restrict__ y, int num_experts, int x_row_div, int n, int k) {
    pi_grouped<4>(x, w_ptrs, scale_ptrs, expert_ids, y, num_experts, x_row_div, n, k);
}

extern "C" __global__ void __launch_bounds__(PI_THREADS) moe_packed_int8_gemv_ptrtable_g128(
    const unsigned short* __restrict__ x, const unsigned long long* __restrict__ w_ptrs,
    const unsigned long long* __restrict__ scale_ptrs, const int* __restrict__ expert_ids,
    unsigned short* __restrict__ y, int num_experts, int x_row_div, int n, int k) {
    pi_grouped<8>(x, w_ptrs, scale_ptrs, expert_ids, y, num_experts, x_row_div, n, k);
}
