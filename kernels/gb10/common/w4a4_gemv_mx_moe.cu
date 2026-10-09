// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-08: W4A4 for a checkpoint that declares STATIC NVFP4 activation scales (ModelOpt
// `input_scale`), and its routed-expert form. Entries over the w4a4_mx core
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
// - w4a4_gemv_mx{8,16}_moe_union (2026-10-09): the same per (row, slot), with each expert of the
//   rows' union swept once for all the rows that chose it (below).
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

// 2026-10-09: Token rows of one routed expert from shared-memory tables (union entry u).
struct W4a4RowsTable {
    unsigned int m;
    const unsigned int* ta;
    const unsigned int* tc;
    __device__ __forceinline__ unsigned int a(unsigned int tok) const { return ta[tok]; }
    __device__ __forceinline__ unsigned int c(unsigned int tok) const { return tc[tok]; }
};

// 2026-10-09: The union form of w4a4_gemv_mx8_moe_slots: blockIdx.y is union entry u of
// glm5next_moe_row_union (u_eid[u] the expert, u_slot[u * rows + r] the slot row r gave it or
// -1), and the block sweeps the expert's weight tile ONCE for every (row, slot) that chose it,
// as the MMA's token columns (MB * 8 of them: rows <= 8 * MB). Token (r, s) reads activation row
// r (act_div = top_k, the gate/up input) or r * top_k + s (act_div = 1, the down input) and
// writes output row r * top_k + s. Each token's output is bit-identical to the slot kernel's:
// the MMA's token columns do not mix and the K order is the same.
template <int MB>
__device__ __forceinline__ void w4a4_moe_union(
    const unsigned char* __restrict__ Aq, const unsigned char* __restrict__ As,
    const float* __restrict__ Ag, const int* __restrict__ u_eid, const int* __restrict__ u_slot,
    const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs, const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C, unsigned int N, unsigned int K, unsigned int rows,
    unsigned int top_k, unsigned int act_div, unsigned int num_experts)
{
    const unsigned int u = blockIdx.y;
    const int id = u_eid[u];
    // 2026-10-09: Block-uniform exits before any barrier.
    if (id < 0 || (unsigned int)id >= num_experts || rows > 8u * MB) return;
    const unsigned long long bq = packed_ptrs[id];
    if (bq == 0ull) return;
    __shared__ unsigned int s_a[8 * MB], s_c[8 * MB], s_m;
    if (threadIdx.x == 0) {
        unsigned int m = 0;
        for (unsigned int r = 0; r < rows; r++) {
            const int sl = u_slot[u * rows + r];
            if (sl < 0) continue;
            const unsigned int row_slot = r * top_k + (unsigned int)sl;
            s_a[m] = act_div == 1u ? row_slot : r;
            s_c[m] = row_slot;
            m++;
        }
        for (unsigned int j = m; j < 8u * MB; j++) { s_a[j] = 0u; s_c[j] = 0u; }
        s_m = m;
    }
    __syncthreads();
    if (s_m == 0u) return;
    w4a4_gemv_mx_tok_impl<MB, 4>(Aq, As, Ag, (const unsigned char*)bq,
                                 (const unsigned char*)scale_ptrs[id], scale2_vals[id], C,
                                 W4a4RowsTable{s_m, s_a, s_c}, N, K);
}

#define W4A4_UNION_ENTRY(NAME, MB)                                                         \
    extern "C" __global__ __launch_bounds__(W4A4_WARPS * 32) void NAME(                     \
        const unsigned char* __restrict__ Aq, const unsigned char* __restrict__ As,        \
        const float* __restrict__ Ag, const int* __restrict__ u_eid,                       \
        const int* __restrict__ u_slot, const unsigned long long* __restrict__ packed_ptrs, \
        const unsigned long long* __restrict__ scale_ptrs,                                 \
        const float* __restrict__ scale2_vals, __nv_bfloat16* __restrict__ C,              \
        unsigned int N, unsigned int K, unsigned int rows, unsigned int top_k,             \
        unsigned int act_div, unsigned int num_experts) {                                  \
        w4a4_moe_union<MB>(Aq, As, Ag, u_eid, u_slot, packed_ptrs, scale_ptrs, scale2_vals, \
                           C, N, K, rows, top_k, act_div, num_experts);                    \
    }

// 2026-10-09: Up to 8 and up to 16 rows (one or two 8-token MMA tiles).
W4A4_UNION_ENTRY(w4a4_gemv_mx8_moe_union, 1)
W4A4_UNION_ENTRY(w4a4_gemv_mx16_moe_union, 2)

#endif
