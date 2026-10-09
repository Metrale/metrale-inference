// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-08: W4A4 for a checkpoint that declares STATIC NVFP4 activation scales (ModelOpt
// `input_scale`), and its routed-expert form. Two entries over the w4a4_mx core
// (w4a4_mx_core.cuh), no new MMA code:
//
// - w4a4_quant_rows_static: w4a4_quant_rows with the row's global scale fixed to the caller's
//   `gs` (the projection's input_scale) instead of amax(row) / (6 * 448). The w4a4_gemv_mx*
//   entries then compute C = scale2 * gs * sum, i.e. weight_scale_2 * input_scale, the
//   checkpoint's GEMM alpha.
// - w4a4_gemv_mx8_moe_slots: the one-tile mx8 GEMV with M = 1, once per routed slot. blockIdx.y
//   is the slot s; its expert is ids[s], whose packed weights, block scales and weight_scale_2
//   come from the global-id pointer tables (null packed pointer: another EP rank's expert). The
//   activation row is s / act_div (act_div = top_k: every slot of a token reads the token's
//   quantized row, the gate/up input; act_div = 1: each slot reads its own row, the down input).
//   C row s, N wide. A slot whose expert is negative, out of range or remote writes nothing; the
//   caller zeroes C first where that matters.
//
// Owner: gb10 kernels.
// Invariants:
// - A slot's output does not depend on the other slots, on the slot count or on the launch: each
//   block runs w4a4_gemv_mx_impl<1, 4> (the mx8 entry's body) on one row.
// - Launch: quant grid (rows, 1, 1), block 256; slots grid (ceil(N / 16), slots, 1), block 256.
//   K % 128 == 0 (whole k128 chunks; a tail is not read), K <= 32768. Activations as
//   w4a4_quant_rows writes them (fragment order), row stride K / 2 bytes, K / 16 scales.
#ifndef METRALE_NO_WARP_BLOCKSCALE_MMA

#include "w4a4_mx_core.cuh"

extern "C" __global__ __launch_bounds__(256) void w4a4_quant_rows_static(
    const __nv_bfloat16* __restrict__ A, unsigned char* __restrict__ Aq,
    unsigned char* __restrict__ As, float* __restrict__ Ag, unsigned int K, float gs)
{
    w4a4_quant_rows_impl<true>(A, Aq, As, Ag, K, gs);
}

extern "C" __global__ __launch_bounds__(W4A4_WARPS * 32) void w4a4_gemv_mx8_moe_slots(
    const unsigned char* __restrict__ Aq, const unsigned char* __restrict__ As,
    const float* __restrict__ Ag, const int* __restrict__ ids,
    const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs, const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C, unsigned int N, unsigned int K, unsigned int act_div,
    unsigned int num_experts)
{
    const unsigned int s = blockIdx.y;
    const int id = ids[s];
    // 2026-10-08: Block-uniform exits (one slot per blockIdx.y), before the impl's barrier.
    if (id < 0 || (unsigned int)id >= num_experts) return;
    const unsigned long long bq = packed_ptrs[id];
    if (bq == 0ull) return;
    const unsigned int row = s / act_div;
    w4a4_gemv_mx_impl<1, 4>(
        Aq + (unsigned long long)row * (K >> 1), As + (unsigned long long)row * (K >> 4), Ag + row,
        (const unsigned char*)bq, (const unsigned char*)scale_ptrs[id], scale2_vals[id],
        C + (unsigned long long)s * N, 1u, N, K);
}

#endif
