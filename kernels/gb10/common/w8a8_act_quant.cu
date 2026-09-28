// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-28: Dynamic E4M3 activation quantizers for the W8A8 decode projections (w8a8_gemv.cu).
//   w8a8_act_quant_row:  s[m]    = max(max_k |X[m, k]| / 448, 1e-12);  Q[m, k] = E4M3(clamp(X[m, k] / s[m]))
//   w8a8_act_quant_g128: s[m, u] = the same over k in [128u, 128u + 128), row-major [M, K / 128]
//   w8a8_act_quant_silu_{row,g128}: the same applied to X[m, k] = bf16(silu(G[m, k]) * U[m, k]), with silu and the
//     rounding of moe_silu_mul (moe_silu_mul.cu), so the fused kernel writes the bytes and scales of moe_silu_mul
//     followed by the plain quantizer.
// X [M, ldx] BF16, Q [M, ldq] E4M3; the first K columns are read and written.
//
// Exact numerics (a fused producer, e.g. an RmsNorm -> ActQuant fusion feeding w8a8_gemv, must reproduce these bytes):
// - Input: each BF16 value widened exactly to FP32. The SiLU variants first form v = bf16_rn(g * sig * u) with
//   sig = 1.0f / (1.0f + __expf(-g)), products left to right (g * sig) * u, built with --fmad=false (no FMA
//   contraction); the quantizer then reads v as above.
// - amax = max over the group of fabsf(v) with fmaxf, in FP32 (exact; independent of reduction order). fmaxf drops
//   NaN, so a NaN input does not set amax.
// - s = fmaxf(amax / 448.0f, 1e-12f): IEEE correctly rounded FP32 division (nvcc's default -prec-div=true), then
//   the floor. An all-zero group gets s = 1e-12 and encodes zeros.
// - q = E4M3(fmaxf(fminf(v / s, 448.0f), -448.0f)): correctly rounded FP32 division, clamp, then
//   __nv_cvt_float2_to_fp8x2(.., __NV_SATFINITE, __NV_E4M3): round to nearest, ties to even, saturating. -0.0
//   encodes as 0x80. A NaN input clamps to +448 (fminf/fmaxf return the non-NaN operand); with an Inf in the group
//   s = Inf, so the Inf element gives Inf / Inf = NaN -> +448 and every finite element encodes 0.
// - Scale storage: s as FP32, one per row ([M]) or per (row, 128-wide group) ([M, K / 128] row-major).
// This is the convention of quant_rowwise_fp8.cu and per_token_group_quant_fp8.cu (same formula and encoder).
//
// Owner: gb10 kernels.
// Invariants:
// - One CTA per row m (grid (M, 1, 1)), block (QT, 1, 1) with QT = 256. K % 8 == 0 (row), K % 128 == 0 (g128);
//   every leading dimension % 8 == 0, X/G/U and Q 16-byte aligned.
// - A row's bytes and scale depend on that row alone.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

namespace w8a8q {

constexpr int QT = 256;
constexpr float E4M3_MAX = 448.0f;
constexpr int REGS = 9;  // uint4 (8 values) per thread held in registers: K <= QT * 8 * REGS = 18432 in one read.

__device__ __forceinline__ uint32_t enc2(float a, float b) {
    a = fmaxf(fminf(a, E4M3_MAX), -E4M3_MAX);
    b = fmaxf(fminf(b, E4M3_MAX), -E4M3_MAX);
    return (uint32_t)__nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE, __NV_E4M3);
}

__device__ __forceinline__ void unpack8(const uint4& v, float (&f)[8]) {
    const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        f[2 * i] = __uint_as_float(w[i] << 16);
        f[2 * i + 1] = __uint_as_float(w[i] & 0xFFFF0000u);
    }
}

__device__ __forceinline__ float amax8(const uint4& v) {
    float f[8];
    unpack8(v, f);
    float m = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) m = fmaxf(m, fabsf(f[i]));
    return m;
}

__device__ __forceinline__ uint2 quant8(const uint4& v, float s) {
    float f[8];
    unpack8(v, f);
    uint2 o;
    o.x = enc2(f[0] / s, f[1] / s) | (enc2(f[2] / s, f[3] / s) << 16);
    o.y = enc2(f[4] / s, f[5] / s) | (enc2(f[6] / s, f[7] / s) << 16);
    return o;
}

__device__ __forceinline__ float block_max(float v, float* smem) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = v;
    __syncthreads();
    float m = 0.f;
#pragma unroll
    for (int i = 0; i < QT / 32; ++i) m = fmaxf(m, smem[i]);
    return m;
}

// 2026-09-28: The BF16 input of one row: a plain row, or bf16(silu(g) * u) of a gate row and an up row.
struct Plain {
    const uint4* x;
    __device__ uint4 operator()(uint32_t i) const { return x[i]; }
};
struct SiluMul {
    const uint4 *g, *u;
    __device__ uint4 operator()(uint32_t i) const {
        const uint4 gv = g[i], uv = u[i];
        float gf[8], uf[8];
        unpack8(gv, gf);
        unpack8(uv, uf);
        uint32_t o[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float a = gf[2 * j] * (1.0f / (1.0f + __expf(-gf[2 * j]))) * uf[2 * j];
            const float b = gf[2 * j + 1] * (1.0f / (1.0f + __expf(-gf[2 * j + 1]))) * uf[2 * j + 1];
            const __nv_bfloat162 h = __halves2bfloat162(__float2bfloat16(a), __float2bfloat16(b));
            o[j] = *reinterpret_cast<const uint32_t*>(&h);
        }
        return make_uint4(o[0], o[1], o[2], o[3]);
    }
};

template <class In>
__device__ __forceinline__ void quant_row(const In& in, uint2* q, float* scale, uint32_t K) {
    __shared__ float smem[QT / 32];
    const uint32_t v8 = K / 8;
    uint4 held[REGS];
    float m = 0.f;
#pragma unroll
    for (int i = 0; i < REGS; ++i) {
        const uint32_t idx = threadIdx.x + i * QT;
        held[i] = idx < v8 ? in(idx) : make_uint4(0, 0, 0, 0);
        m = fmaxf(m, amax8(held[i]));
    }
    for (uint32_t idx = threadIdx.x + REGS * QT; idx < v8; idx += QT) m = fmaxf(m, amax8(in(idx)));
    const float s = fmaxf(block_max(m, smem) / E4M3_MAX, 1e-12f);
    if (threadIdx.x == 0) scale[blockIdx.x] = s;
#pragma unroll
    for (int i = 0; i < REGS; ++i) {
        const uint32_t idx = threadIdx.x + i * QT;
        if (idx < v8) q[idx] = quant8(held[i], s);
    }
    for (uint32_t idx = threadIdx.x + REGS * QT; idx < v8; idx += QT) q[idx] = quant8(in(idx), s);
}

// 2026-09-28: 16 threads per 128-wide group; the group max is a 16-lane xor-shuffle reduction. The loop bound is
// uniform per warp (v8 % 16 == 0, so a 16-lane group is all in or all out), which keeps every lane in the shuffles.
template <class In>
__device__ __forceinline__ void quant_g128(const In& in, uint2* q, float* scale, uint32_t K) {
    const uint32_t v8 = K / 8, kb = K / 128;
    for (uint32_t idx = threadIdx.x; idx - (threadIdx.x & 31) < v8; idx += QT) {
        const bool ok = idx < v8;
        const uint4 v = ok ? in(idx) : make_uint4(0, 0, 0, 0);
        float m = amax8(v);
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        const float s = fmaxf(m / E4M3_MAX, 1e-12f);
        if (ok && (idx & 15) == 0) scale[(size_t)blockIdx.x * kb + idx / 16] = s;
        if (ok) q[idx] = quant8(v, s);
    }
}

}  // namespace w8a8q

#define W8A8Q_PLAIN(NAME, BODY)                                                                                   \
    extern "C" __global__ void __launch_bounds__(w8a8q::QT)                                                       \
        NAME(const __nv_bfloat16* __restrict__ X, unsigned char* __restrict__ Q, float* __restrict__ scale,       \
             uint32_t K, uint32_t ldx, uint32_t ldq) {                                                            \
        const w8a8q::Plain in{reinterpret_cast<const uint4*>(X + (size_t)blockIdx.x * ldx)};                      \
        w8a8q::BODY(in, reinterpret_cast<uint2*>(Q + (size_t)blockIdx.x * ldq), scale, K);                        \
    }
#define W8A8Q_SILU(NAME, BODY)                                                                                    \
    extern "C" __global__ void __launch_bounds__(w8a8q::QT)                                                       \
        NAME(const __nv_bfloat16* __restrict__ G, const __nv_bfloat16* __restrict__ U, unsigned char* __restrict__ Q, \
             float* __restrict__ scale, uint32_t K, uint32_t ldg, uint32_t ldu, uint32_t ldq) {                  \
        const w8a8q::SiluMul in{reinterpret_cast<const uint4*>(G + (size_t)blockIdx.x * ldg),                     \
                                reinterpret_cast<const uint4*>(U + (size_t)blockIdx.x * ldu)};                    \
        w8a8q::BODY(in, reinterpret_cast<uint2*>(Q + (size_t)blockIdx.x * ldq), scale, K);                        \
    }

W8A8Q_PLAIN(w8a8_act_quant_row, quant_row)
W8A8Q_PLAIN(w8a8_act_quant_g128, quant_g128)
W8A8Q_SILU(w8a8_act_quant_silu_row, quant_row)
W8A8Q_SILU(w8a8_act_quant_silu_g128, quant_g128)
