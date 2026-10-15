// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-10: Grouped W4A4 expert GEMM for wide routed-expert launches (prefill): every routed
// slot of every expert in one launch, on the w4a4_mx core's FP4 block-scale MMA
// (w4a4_mx_core.cuh, no new MMA code), with each output BIT-IDENTICAL to the slot GEMV
// w4a4_gemv_mx8_moe_slots (w4a4_gemv_mx_moe.cu) on the same slot.
//
// - w4a4_grouped_tiles: one block; tile_off[e] = sum over e' < e of ceil(M_e' / MT), the first
//   tile of expert e, from the expert-sorted row offsets (moe_sort_by_expert); tile_off[E] is the
//   tile count. The GEMM's grid.y is an upper bound, ceil(slots / MT) + E; tiles past the count
//   exit. No host read, so the launch can be captured.
// - w4a4_grouped_one_group (2026-10-13): the offsets and tiles of one group of m rows, m a launch
//   argument (the dense MLP's one-group launch).
// - w4a4_grouped_mx_m{32,64}[_k64]: C rows [expert_offsets[e], expert_offsets[e + 1]) are expert
//   e's sorted slots; C row r reads activation row sorted_token_ids[r] when that pointer is
//   non-null (gate/up: the token's quantized row), else row r (down: the slot's own row). Weights
//   from the global-id pointer tables; a null packed pointer writes nothing (another EP rank's
//   expert). blockIdx.x is a 128-column weight tile, blockIdx.y a linear (expert, MT-slot) tile.
// - w4a4_grouped_gu_m{32,64,128} (2026-10-14): gate, up, the clamped SwiGLU and the down input's
//   static NVFP4 quantization in one launch (the fused epilogue, w4a4_grouped_core.cuh); below.
// The GEMM body and its shared-memory layout live in w4a4_grouped_core.cuh (2026-10-14).
//
// Why the bits match the slot GEMV. The slot GEMV runs w4a4_gemv_mx_tok_impl<1, 4>: warp w
// (0..7) accumulates the k128 chunks c = w, w + 8, w + 16, .. in increasing order, two MMAs per
// chunk (K halves, fragments built by the same shuffles and byte permutes), then the 8 warp sums
// are added as v = s0; v += s1; .. v += s7, and the output is bf16_rn(v * (Ag * scale2)). Here
// each thread visits the chunks in that class order (class w = chunks c = w mod 8, ascending),
// keeps the class sum in `acc` and the running total in `tot` (tot = class 0; tot += class w for
// w = 1..7, empty classes adding +0.0 as the GEMV's zero warp sums do), loads each operand into
// the registers the GEMV loads it into (staged through shared memory, same bytes), and finishes
// with the same expression. An MMA output column (a token) depends only on that column's
// operands, which the union GEMV's byte gate already relies on (model-arch example
// glm5next_moe_wide_bench). The model-arch example glm5next_moe_prefill_w4a4_bench checks every
// output byte against the slot GEMV.
//
// Owner: gb10 kernels.
// Invariants:
// - Launch: tiles grid (1, 1, 1), block 256. GEMM grid (ceil(N / NT), ceil(slots / MT) + E, 1),
//   block 256, dynamic shared memory of sizeof(W4gSmem<MT, NT, S>) (host:
//   `w4a4_grouped::smem_bytes`); MT must match the tile pass's `mt`. Entries: m32 and m64 with
//   NT 128, m128 with NT 64; S = 3 pipeline stages (m128_s6: 6, one CTA per SM).
// - K % 128 == 0 (_k64: K % 64 == 0, activations at Kp = round_up(K, 128) as
//   w4a4_quant_rows_static_k64 writes them); K <= 8192 (the class-order table). N % 8 == 0.
// - Activations as w4a4_quant_rows_static writes them (fragment order, row stride Kp / 2, Kp / 16
//   scales, one F32 global per row).
#ifndef METRALE_NO_WARP_BLOCKSCALE_MMA

#include "w4a4_grouped_core.cuh"
#include "w4a4_grouped_nc.cuh"

extern "C" __global__ void __launch_bounds__(W4G_THREADS) w4a4_grouped_tiles(
    const int* __restrict__ expert_offsets, int* __restrict__ tile_off, unsigned int num_experts,
    unsigned int mt)
{
    if (threadIdx.x != 0) return;
    int acc = 0;
    for (unsigned int e = 0; e < num_experts; ++e) {
        tile_off[e] = acc;
        const int m = expert_offsets[e + 1] - expert_offsets[e];
        acc += m > 0 ? (m + (int)mt - 1) / (int)mt : 0;
    }
    tile_off[num_experts] = acc;
}

// 2026-10-13: The offsets and tile pass of ONE group of m rows (a dense projection,
// w4a4_dense_grouped.rs): expert_offsets = [0, m], tile_off = [0, ceil(m / mt)], exactly what
// w4a4_grouped_tiles writes from those offsets. m arrives by value, so no host buffer is read
// and a captured launch replays the m it was captured with. Launch: grid (1, 1, 1), block 32.
extern "C" __global__ void __launch_bounds__(W4G_THREADS) w4a4_grouped_one_group(
    int* __restrict__ expert_offsets, int* __restrict__ tile_off, unsigned int m, unsigned int mt)
{
    if (threadIdx.x != 0) return;
    expert_offsets[0] = 0;
    expert_offsets[1] = (int)m;
    tile_off[0] = 0;
    tile_off[1] = (int)((m + mt - 1u) / mt);
}

#define W4G_ARGS                                                                               \
    const unsigned char* __restrict__ Aq, const unsigned char* __restrict__ As,               \
        const float* __restrict__ Ag, const unsigned long long* __restrict__ packed_ptrs,     \
        const unsigned long long* __restrict__ scale_ptrs,                                    \
        const float* __restrict__ scale2_vals, __nv_bfloat16* __restrict__ C,                 \
        const int* __restrict__ expert_offsets, const int* __restrict__ tile_off,             \
        const int* __restrict__ sorted_token_ids, unsigned int num_experts, unsigned int N,   \
        unsigned int K
#define W4G_PASS                                                                               \
    Aq, As, Ag, packed_ptrs, scale_ptrs, scale2_vals, C, expert_offsets, tile_off,             \
        sorted_token_ids, num_experts, N, K

extern "C" __global__ void __launch_bounds__(W4G_THREADS, 2) w4a4_grouped_mx_m64(W4G_ARGS) {
    w4g_body<64, W4G_NT, 3, false>(W4G_PASS);
}
extern "C" __global__ void __launch_bounds__(W4G_THREADS, 2) w4a4_grouped_mx_m64_k64(W4G_ARGS) {
    w4g_body<64, W4G_NT, 3, true>(W4G_PASS);
}
extern "C" __global__ void __launch_bounds__(W4G_THREADS, 2) w4a4_grouped_mx_m32(W4G_ARGS) {
    w4g_body<32, W4G_NT, 3, false>(W4G_PASS);
}
extern "C" __global__ void __launch_bounds__(W4G_THREADS, 2) w4a4_grouped_mx_m32_k64(W4G_ARGS) {
    w4g_body<32, W4G_NT, 3, true>(W4G_PASS);
}
// 2026-10-10: 128 slots by 64 weight rows: an expert's weights are read once for up to 128 of
// its slots (a 2048-row prefill sub-chunk averages 57 per expert at top 8 of 288).
extern "C" __global__ void __launch_bounds__(W4G_THREADS, 2) w4a4_grouped_mx_m128(W4G_ARGS) {
    w4g_body<128, 64, 3, false>(W4G_PASS);
}
extern "C" __global__ void __launch_bounds__(W4G_THREADS) w4a4_grouped_mx_m128_s6(W4G_ARGS) {
    w4g_body<128, 64, 6, false>(W4G_PASS);
}
extern "C" __global__ void __launch_bounds__(W4G_THREADS, 2) w4a4_grouped_mx_m128_k64(W4G_ARGS) {
    w4g_body<128, 64, 3, true>(W4G_PASS);
}
// 2026-10-14: The m128 entries with a raster group (w4g_body `raster`; 0 is the plain entry's
// order): same grid, same bytes, a different CTA-to-tile walk (lever METRALE_GLM_MOE_W4A4_RASTER).
extern "C" __global__ void __launch_bounds__(W4G_THREADS, 2) w4a4_grouped_mx_m128_r(
    W4G_ARGS, unsigned int raster) {
    w4g_body<128, 64, 3, false>(W4G_PASS, W4gGu{}, raster);
}
extern "C" __global__ void __launch_bounds__(W4G_THREADS, 2) w4a4_grouped_mx_m128_k64_r(
    W4G_ARGS, unsigned int raster) {
    w4g_body<128, 64, 3, true>(W4G_PASS, W4gGu{}, raster);
}
// 2026-10-15: The m128 one-projection entries with `nc` column tiles per CTA (w4n_body,
// w4a4_grouped_nc.cuh): grid (ceil(ceil(N / 64) / nc), ceil(slots / 128) + E, 1), same bytes as the
// entries above at every nc and raster (lever METRALE_GLM_MOE_W4A4_NC).
extern "C" __global__ void __launch_bounds__(W4G_THREADS, 2) w4a4_grouped_mx_m128_n(
    W4G_ARGS, unsigned int raster, unsigned int nc) {
    w4n_body<128, 64, 3, false>(W4G_PASS, raster, nc);
}
extern "C" __global__ void __launch_bounds__(W4G_THREADS, 2) w4a4_grouped_mx_m128_k64_n(
    W4G_ARGS, unsigned int raster, unsigned int nc) {
    w4n_body<128, 64, 3, true>(W4G_PASS, raster, nc);
}

// 2026-10-14: Gate, up, the clamped SwiGLU and the down input's static NVFP4 quantization in one
// launch (w4g_body GU): each CTA's NT weight rows are NT / 2 columns of gate and of up
// (w4g_gu_col), so the BF16 gate and up tiles and the SwiGLU product never reach global memory.
// Every byte of (Dq, Ds, Dg) on a live slot row of a local expert equals what the separate
// launches write there: gate and up through the one-projection entry of the same MT,
// glm5next_swiglu_clamp, then w4a4_quant_rows_static (Kp = N) or w4a4_quant_rows_static_k64
// (Kp = round_up(N, 128)) under `gs` (model-arch example glm5next_moe_epi_bench checks it).
// Grid (ceil(N / (NT / 2)), ceil(slots / MT) + E, 1); N % 16 == 0; Kp = N or round_up(N, 128).
#define W4G_GU_ARGS                                                                            \
    const unsigned char* __restrict__ Aq, const unsigned char* __restrict__ As,               \
        const float* __restrict__ Ag, const unsigned long long* __restrict__ gate_packed,     \
        const unsigned long long* __restrict__ gate_scales,                                   \
        const float* __restrict__ gate_scale2, const unsigned long long* __restrict__ up_packed, \
        const unsigned long long* __restrict__ up_scales, const float* __restrict__ up_scale2, \
        unsigned char* __restrict__ Dq, unsigned char* __restrict__ Ds, float* __restrict__ Dg, \
        const int* __restrict__ expert_offsets, const int* __restrict__ tile_off,             \
        const int* __restrict__ sorted_token_ids, unsigned int num_experts, unsigned int N,   \
        unsigned int K, unsigned int Kp, float gs, float limit
#define W4G_GU_PASS                                                                            \
    Aq, As, Ag, gate_packed, gate_scales, gate_scale2, nullptr, expert_offsets, tile_off,      \
        sorted_token_ids, num_experts, N, K,                                                   \
        W4gGu{up_packed, up_scales, up_scale2, Dq, Ds, Dg, Kp, gs, limit}

extern "C" __global__ void __launch_bounds__(W4G_THREADS, 2) w4a4_grouped_gu_m32(W4G_GU_ARGS) {
    w4g_body<32, W4G_NT, 3, false, true>(W4G_GU_PASS);
}
extern "C" __global__ void __launch_bounds__(W4G_THREADS, 2) w4a4_grouped_gu_m64(W4G_GU_ARGS) {
    w4g_body<64, W4G_NT, 3, false, true>(W4G_GU_PASS);
}
extern "C" __global__ void __launch_bounds__(W4G_THREADS, 2) w4a4_grouped_gu_m128(W4G_GU_ARGS) {
    w4g_body<128, 64, 3, false, true>(W4G_GU_PASS);
}
extern "C" __global__ void __launch_bounds__(W4G_THREADS, 2) w4a4_grouped_gu_m128_r(
    W4G_GU_ARGS, unsigned int raster) {
    w4g_body<128, 64, 3, false, true>(W4G_GU_PASS, raster);
}

#endif // METRALE_NO_WARP_BLOCKSCALE_MMA
