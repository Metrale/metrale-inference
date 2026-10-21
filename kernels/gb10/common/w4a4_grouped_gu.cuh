// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-21: The grouped W4A4 GEMM's fused gate/up epilogue (W4gGu, w4g_gu_col,
// w4g_gu_epilogue), moved unchanged from w4a4_grouped_core.cuh so that file stays under the size
// limit with the compact-scale staging (w4a4_grouped_cs.cuh).
//
// Owner: gb10 kernels.
// Invariants: as w4a4_grouped_mx.cu. Included by w4a4_grouped_core.cuh only (after W4G_THREADS).
#pragma once

#include "w4a4_mx_core.cuh"
#include "glm5next_swiglu.cuh"

// 2026-10-14: The fused gate/up epilogue's extra operands (GU = true; unused otherwise): the up
// projection's pointer tables, the down input's quantization (codes, scales, globals) at row
// width Kp under the down projection's static global scale gs, and the SwiGLU limit.
struct W4gGu {
    const unsigned long long* up_packed;
    const unsigned long long* up_scales;
    const float* up_scale2;
    unsigned char* Dq;
    unsigned char* Ds;
    float* Dg;
    unsigned int Kp;
    float gs;
    float limit;
    // 2026-10-21: The `_cs` entries' up base-code table (w4g_body CS); null otherwise.
    const unsigned int* up_cs = nullptr;
};

// 2026-10-14: Shared-memory weight row r of a GU tile holds projection (r >> 4) & 1 (0 gate,
// 1 up) at output column w4g_gu_col(n0, r): the rows alternate 16 gate and 16 up rows of the same
// 16 columns, so column n sits at row n % 16 of its 16-row MMA fragment exactly as in the
// one-projection tiles (n0 is a multiple of 16 there too), and its gate and up values meet in the
// output tile at rows 32 * (n - n0) / 16 + n % 16 and that + 16.
__device__ __forceinline__ unsigned int w4g_gu_col(unsigned int n0, unsigned int r) {
    return n0 + (r >> 5) * 16u + (r & 15u);
}

// 2026-10-14: The fused gate/up epilogue: `out` holds every live slot's BF16 gate and up values
// (the bytes the one-projection launches store to their C); each thread takes a (slot, 16-column
// group), applies the clamped SwiGLU and its BF16 rounding (glm5next_swiglu_clamp's expression)
// and quantizes the group as w4a4_quant_rows_static[_k64] does, into the slot's own row of
// (Dq, Ds) at width Kp. The n0 == 0 CTA writes each slot's global, and the CTA holding column
// N - 1 the zero groups N / 16 .. Kp / 16 (PAD128). TH threads.
template <int MT, int NT, int TH = W4G_THREADS>
__device__ __forceinline__ void w4g_gu_epilogue(
    const __nv_bfloat16 (&out)[MT][NT + 8], const W4gGu& gu, const int m_start, const int m_valid,
    const unsigned int n0, const unsigned int N)
{
    constexpr unsigned int GPR = NT / 32u;  // 16-column groups per slot row
    const float inv_gs = 1.0f / gu.gs;
    for (unsigned int i = threadIdx.x; i < (unsigned int)MT * GPR; i += TH) {
        const unsigned int slot = i / GPR, gl = i % GPR;
        const unsigned int col = n0 + gl * 16u;
        if ((int)slot >= m_valid || col >= N) continue;
        unsigned int w[8];
#pragma unroll
        for (unsigned int h = 0; h < 8u; ++h) {
            unsigned int pair = 0u;
#pragma unroll
            for (unsigned int e = 0; e < 2u; ++e) {
                const unsigned int k = gl * 32u + 2u * h + e;
                const float a = glm5next_swiglu_clamp_f32(
                    __bfloat162float(out[slot][k]), __bfloat162float(out[slot][k + 16u]), gu.limit);
                pair |= (unsigned int)__bfloat16_as_ushort(__float2bfloat16(a)) << (16u * e);
            }
            w[h] = pair;
        }
        w4a4_quant_group(make_uint4(w[0], w[1], w[2], w[3]), make_uint4(w[4], w[5], w[6], w[7]),
                         gu.Dq, gu.Ds, (unsigned int)(m_start + (int)slot), gu.Kp, col >> 4, gu.gs,
                         inv_gs);
    }
    // 2026-10-14: The first column tile writes the rows' global scale (keyed on n0, not the grid
    // coordinate, so any tile raster keeps exactly one writer).
    if (n0 == 0u) {
        for (unsigned int i = threadIdx.x; i < (unsigned int)MT; i += TH)
            if ((int)i < m_valid) gu.Dg[m_start + (int)i] = gu.gs;
    }
    if (n0 + NT / 2u >= N) {
        const unsigned int g0 = N >> 4, np = (gu.Kp >> 4) - g0;
        for (unsigned int i = threadIdx.x; i < (unsigned int)MT * np; i += TH) {
            const unsigned int slot = i / np;
            if ((int)slot < m_valid)
                w4a4_quant_pad_group(gu.Dq, gu.Ds, (unsigned int)(m_start + (int)slot), gu.Kp,
                                     g0 + i % np);
        }
    }
}
