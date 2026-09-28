// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-28: rms_norm (rms_norm.cu) with an activation-quantizer epilogue, in one launch with
// the bytes of the two-kernel chain:
//   rms_norm_quant_fp8_row   = rms_norm, then quant_rowwise_fp8 (quant_rowwise_fp8.cu):
//                              per-token E4M3, one FP32 scale per row (W8A8 per-token).
//   rms_norm_quant_fp8_g128  = rms_norm, then per_token_group_quant_fp8
//                              (per_token_group_quant_fp8.cu): E4M3 with one FP32 scale per
//                              128-wide K group, [M, K / 128] row-major.
//   rms_norm_quant_nvfp4     = rms_norm, then w4a4_quant_rows (w4a4_gemv_mx.cu): E2M1 values
//                              and E4M3 group-16 scales in the W4A4 GEMV's fragment order, and
//                              an FP32 per-row global scale.
// The normalized row is rounded to BF16 exactly as rms_norm stores it and kept in shared
// memory, so the quantizer reads the values the unfused chain reads from global memory.
//
// Numerics each epilogue reproduces (from the quantizer it replaces):
// - FP8: s = max(amax / 448, 1e-12), correctly rounded FP32 division; q = E4M3(clamp(v / s,
//   +-448)) with __nv_cvt_float_to_fp8(.., __NV_SATFINITE, __NV_E4M3). This is also the
//   convention w8a8_act_quant.cu (branch perf/w8a8-decode) documents for W8A8 decode.
// - NVFP4: gs = amax > 0 ? amax / (6 * 448) : 1; per 16 values s8 = e4m3(gm * (1 / 6) * (1 /
//   gs)), s = float(s8); E2M1 round-to-nearest-even of v * (s > 0 ? 1 / (s * gs) : 0).
// - Every amax is a max of fabsf over BF16-widened values: exact and independent of the order
//   the threads combine it in, so the epilogue need not copy the quantizer's thread layout.
//
// Owner: gb10 kernels.
// Invariants:
// - One block per row with blockDim.x = min(K, 1024), the launch of ops::rms_norm, so the sum
//   of squares is rms_norm's (rms_norm_exact.cuh). Dynamic shared memory K * 2 bytes.
// - K % 16 == 0 (NVFP4), K % 128 == 0 (g128), K % 2 == 0 (row); X, Q and the NVFP4 outputs
//   are 16-byte aligned.
// - Bit identity with each chain is checked by the model-arch example
//   rms_norm_act_quant_microtest.
// - CUDA only: SCALE and HIP builds encode E4M3 in software (per_token_group_quant_fp8.cu), so
//   this module has no entry points there.

#include <cuda_fp8.h>
#include <stdint.h>

#include "rms_norm_exact.cuh"

#if !defined(__SCALE__) && !defined(__HIP_PLATFORM_AMD__)

#define RMSQ_E4M3_MAX 448.0f

// 2026-09-28: rms_norm's two passes over row blockIdx.x, the second storing the BF16 pairs to
// `y` (shared memory) instead of global memory.
__device__ __forceinline__ void rmsq_norm_row(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    unsigned int* y,
    unsigned int hidden_size,
    float eps
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int half_size = hidden_size / 2;
    const unsigned int* x32 = (const unsigned int*)(input + blockIdx.x * hidden_size);
    float sum_sq = 0.0f;
    for (unsigned int i = tid; i < half_size; i += blockDim.x) {
        float v0, v1;
        rmsx_unpack(x32[i], v0, v1);
        sum_sq += v0 * v0 + v1 * v1;
    }
    __shared__ float warp_sums[32];
    float total = rmsx_block_sum(sum_sq, warp_sums);
    float rms = rsqrtf(total / (float)hidden_size + eps);
    const unsigned int* w32 = (const unsigned int*)weight;
    for (unsigned int i = tid; i < half_size; i += blockDim.x) {
        float xv0, xv1, wv0, wv1;
        rmsx_unpack(x32[i], xv0, xv1);
        rmsx_unpack(w32[i], wv0, wv1);
        y[i] = rmsx_pack(xv0 * rms * (1.0f + wv0), xv1 * rms * (1.0f + wv1));
    }
    __syncthreads();
}

// 2026-09-28: Block-wide max of non-negative `m`: exact whatever the combining order.
__device__ __forceinline__ float rmsq_block_max(float m, float* smem) {
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, o));
    if ((threadIdx.x & 31u) == 0u) smem[threadIdx.x >> 5] = m;
    __syncthreads();
    float r = 0.0f;
    for (unsigned int w = 0; w < (blockDim.x + 31u) / 32u; w++) r = fmaxf(r, smem[w]);
    return r;
}

__device__ __forceinline__ float rmsq_abs_max_pair(unsigned int p) {
    float a, b;
    rmsx_unpack(p, a, b);
    return fmaxf(fabsf(a), fabsf(b));
}

__device__ __forceinline__ unsigned char rmsq_e4m3(float v, float s) {
    float q = v / s;
    q = fmaxf(fminf(q, RMSQ_E4M3_MAX), -RMSQ_E4M3_MAX);
    return (unsigned char)__nv_cvt_float_to_fp8(q, __NV_SATFINITE, __NV_E4M3);
}

extern "C" __global__ void __launch_bounds__(1024) rms_norm_quant_fp8_row(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    unsigned char* __restrict__ q,
    float* __restrict__ scale,
    unsigned int hidden_size,
    float eps
) {
    extern __shared__ unsigned int y[];
    rmsq_norm_row(input, weight, y, hidden_size, eps);
    const unsigned int half_size = hidden_size / 2;
    float m = 0.0f;
    for (unsigned int i = threadIdx.x; i < half_size; i += blockDim.x) {
        m = fmaxf(m, rmsq_abs_max_pair(y[i]));
    }
    __shared__ float wm[32];
    float gmax = rmsq_block_max(m, wm);
    float s = gmax / RMSQ_E4M3_MAX;
    if (s < 1e-12f) s = 1e-12f;
    if (threadIdx.x == 0) scale[blockIdx.x] = s;
    unsigned char* qr = q + (unsigned long long)blockIdx.x * hidden_size;
    for (unsigned int i = threadIdx.x; i < half_size; i += blockDim.x) {
        float a, b;
        rmsx_unpack(y[i], a, b);
        qr[2 * i] = rmsq_e4m3(a, s);
        qr[2 * i + 1] = rmsq_e4m3(b, s);
    }
}

// 2026-09-28: Each 128-wide group's 64 pairs are scanned by one thread, so no group spans two
// threads' partial maxima.
extern "C" __global__ void __launch_bounds__(1024) rms_norm_quant_fp8_g128(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    unsigned char* __restrict__ q,
    float* __restrict__ a_scale,
    unsigned int hidden_size,
    float eps
) {
    extern __shared__ unsigned int y[];
    rmsq_norm_row(input, weight, y, hidden_size, eps);
    const unsigned int groups = hidden_size / 128;
    unsigned char* qr = q + (unsigned long long)blockIdx.x * hidden_size;
    for (unsigned int g = threadIdx.x; g < groups; g += blockDim.x) {
        float m = 0.0f;
        for (unsigned int j = 0; j < 64; j++) m = fmaxf(m, rmsq_abs_max_pair(y[g * 64 + j]));
        float s = m / RMSQ_E4M3_MAX;
        if (s < 1e-12f) s = 1e-12f;
        a_scale[(unsigned long long)blockIdx.x * groups + g] = s;
        for (unsigned int j = 0; j < 64; j++) {
            float a, b;
            rmsx_unpack(y[g * 64 + j], a, b);
            qr[g * 128 + 2 * j] = rmsq_e4m3(a, s);
            qr[g * 128 + 2 * j + 1] = rmsq_e4m3(b, s);
        }
    }
}

// 2026-09-28: E2M1 code of x, rounded to nearest with ties to even and saturated at 6: the
// thresholds of w4a4_e2m1_rne (w4a4_gemv_mx.cu).
__device__ __forceinline__ unsigned int rmsq_e2m1_rne(float x) {
    const float a = fabsf(x);
    unsigned int c;
    if (a <= 0.25f) c = 0u;
    else if (a < 0.75f) c = 1u;
    else if (a <= 1.25f) c = 2u;
    else if (a < 1.75f) c = 3u;
    else if (a <= 2.5f) c = 4u;
    else if (a < 3.5f) c = 5u;
    else if (a <= 5.0f) c = 6u;
    else c = 7u;
    return (x < 0.0f && c != 0u) ? (c | 8u) : c;
}

// 2026-09-28: Outputs as w4a4_quant_rows writes them: Aq [M, K / 2], As [M, K / 16] in the
// fragment order of the W4A4 GEMV, Ag [M].
extern "C" __global__ void __launch_bounds__(1024) rms_norm_quant_nvfp4(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    unsigned char* __restrict__ Aq,
    unsigned char* __restrict__ As,
    float* __restrict__ Ag,
    unsigned int hidden_size,
    float eps
) {
    extern __shared__ unsigned int y[];
    rmsq_norm_row(input, weight, y, hidden_size, eps);
    const unsigned int row = blockIdx.x;
    const unsigned int K = hidden_size;
    float m = 0.0f;
    for (unsigned int i = threadIdx.x; i < K / 2; i += blockDim.x) {
        m = fmaxf(m, rmsq_abs_max_pair(y[i]));
    }
    __shared__ float wm[32];
    const float amax = rmsq_block_max(m, wm);
    const float gs = amax > 0.0f ? amax / (6.0f * 448.0f) : 1.0f;
    if (threadIdx.x == 0) Ag[row] = gs;
    const float inv_gs = 1.0f / gs;
    for (unsigned int grp = threadIdx.x; grp < (K >> 4); grp += blockDim.x) {
        float f[16];
        float gm = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            rmsx_unpack(y[grp * 8u + i], f[2 * i], f[2 * i + 1]);
            gm = fmaxf(gm, fmaxf(fabsf(f[2 * i]), fabsf(f[2 * i + 1])));
        }
        const __nv_fp8_e4m3 s8(gm * (1.0f / 6.0f) * inv_gs);
        const float s = (float)s8;
        const unsigned int chunk = grp >> 3, q = grp & 7u;
        const unsigned int spos = ((q & 1u) << 2) | ((q >> 2) & 1u) | (q & 2u);
        As[(unsigned long long)row * (K >> 4) + chunk * 8u + spos] = *(const unsigned char*)&s8;
        const float inv = s > 0.0f ? 1.0f / (s * gs) : 0.0f;
        unsigned int p[2] = {0u, 0u};
        #pragma unroll
        for (int i = 0; i < 16; i++) p[i >> 3] |= rmsq_e2m1_rne(f[i] * inv) << ((i & 7) * 4);
        const unsigned int pr = (q & 3u) == 1u ? 2u : (q & 3u) == 2u ? 1u : (q & 3u);
        unsigned char* dst = Aq + (unsigned long long)row * (K >> 1) + chunk * 64u;
        #pragma unroll
        for (unsigned int h = 0; h < 2u; h++) {
            const unsigned int slot = (2u * (q >> 2) + h) * 4u + pr;
            *(uint32_t*)(dst + slot * 4u) = p[h];
        }
    }
}

#endif
