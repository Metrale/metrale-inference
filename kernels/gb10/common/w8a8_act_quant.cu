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

#include "w8a8_quant.cuh"
#include "pdl.cuh"

#define W8A8Q_PLAIN(NAME, BODY)                                                                                   \
    extern "C" __global__ void __launch_bounds__(w8a8q::QT)                                                       \
        NAME(const __nv_bfloat16* __restrict__ X, unsigned char* __restrict__ Q, float* __restrict__ scale,       \
             uint32_t K, uint32_t ldx, uint32_t ldq) {                                                            \
        pdl_enter(); /* 2026-10-21: PDL (pdl.cuh): X is another kernel's output. */                              \
        const w8a8q::Plain in{reinterpret_cast<const uint4*>(X + (size_t)blockIdx.x * ldx)};                      \
        w8a8q::BODY(in, reinterpret_cast<uint2*>(Q + (size_t)blockIdx.x * ldq), scale, K);                        \
    }
#define W8A8Q_SILU(NAME, BODY)                                                                                    \
    extern "C" __global__ void __launch_bounds__(w8a8q::QT)                                                       \
        NAME(const __nv_bfloat16* __restrict__ G, const __nv_bfloat16* __restrict__ U, unsigned char* __restrict__ Q, \
             float* __restrict__ scale, uint32_t K, uint32_t ldg, uint32_t ldu, uint32_t ldq) {                  \
        pdl_enter();                                                                                              \
        const w8a8q::SiluMul in{reinterpret_cast<const uint4*>(G + (size_t)blockIdx.x * ldg),                     \
                                reinterpret_cast<const uint4*>(U + (size_t)blockIdx.x * ldu)};                    \
        w8a8q::BODY(in, reinterpret_cast<uint2*>(Q + (size_t)blockIdx.x * ldq), scale, K);                        \
    }

W8A8Q_PLAIN(w8a8_act_quant_row, quant_row)
W8A8Q_PLAIN(w8a8_act_quant_g128, quant_g128)
W8A8Q_SILU(w8a8_act_quant_silu_row, quant_row)
W8A8Q_SILU(w8a8_act_quant_silu_g128, quant_g128)
