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
//   rows' union swept once for all the rows that chose it (below). Since 2026-10-09 the forward
//   runs its persistent form w4a4_gemv_mx{8,16}_moe_union_sweep (gate and up in one launch); the
//   grid form stays as the reference the model-arch example glm5next_moe_wide_bench compares to.
// - w4a4_gemv_mx8_moe_slots_sweep (2026-10-09): the sweep at one row over the row's own slots
//   (no union build), which the forward runs at one row; the slot GEMV stays as its reference
//   (model-arch example glm5next_moe_narrow_bench).
// - 2026-10-10: `_k64` twins of the quantizer, the slot GEMV and the sweeps, for a routed down
//   projection whose K (the expert width) is a multiple of 64 but not of 128: an expert sliced
//   over TP ranks in 64-column units (glm5next_mlp::expert_tp). The quantizer writes the rows at
//   the padded width (w4a4_quant_rows_impl<true, true>), the GEMVs read the weights at their
//   natural K and the activations at the padded one (w4a4_gemv_mx_tok_impl KH); each output is
//   bit-identical to the plain entry at the padded K over zero-padded weights. 2026-10-10: the
//   one-row own-slots sweep has its twin too (w4a4_gemv_mx8_moe_slots_sweep_k64).
//
// Owner: gb10 kernels.
// Invariants:
// - A slot's output does not depend on the other slots, on the slot count or on the launch: each
//   block runs w4a4_gemv_mx_impl<1, 4> (the mx8 entry's body) on one row.
// - Launch: quant grid (rows, 1, 1), block 256; slots grid (ceil(N / 16), slots, 1), block 256.
//   K % 128 == 0 (whole k128 chunks; a tail is not read), K <= 32768. Activations as
//   w4a4_quant_rows writes them (fragment order), row stride K / 2 bytes, K / 16 scales.
//   2026-10-10: The `_k64` twins: K % 64 == 0, activation rows at round_up(K, 128).
// - 2026-10-21: PDL: the static quantizers and the sweeps call pdl_enter() before any global access.
//   2026-10-21: The `_sl` gate/up union sweeps (serial launch) hold no pdl_wait(), so the host launches
//   them stream-ordered (gpu-runtime ptx_pdl.rs): their persistent grid starts on an idle device
//   instead of beside its predecessor's CTAs, and they call pdl_trigger() after their last item.
//   Same bytes as the plain entries; measured on GB10 (pdl_chain_bench) -4.9 us per MoE layer.
#ifndef METRALE_NO_WARP_BLOCKSCALE_MMA

#include "w4a4_mx_core.cuh"
#include "pdl.cuh"

extern "C" __global__ __launch_bounds__(256) void w4a4_quant_rows_static(
    const __nv_bfloat16* __restrict__ A, unsigned char* __restrict__ Aq,
    unsigned char* __restrict__ As, float* __restrict__ Ag, unsigned int K, float gs)
{
    pdl_enter();
    w4a4_quant_rows_impl<true>(A, Aq, As, Ag, K, gs);
}

extern "C" __global__ __launch_bounds__(256) void w4a4_quant_rows_static_k64(
    const __nv_bfloat16* __restrict__ A, unsigned char* __restrict__ Aq,
    unsigned char* __restrict__ As, float* __restrict__ Ag, unsigned int K, float gs)
{
    pdl_enter();
    w4a4_quant_rows_impl<true, true>(A, Aq, As, Ag, K, gs);
}

// 2026-10-10: The slot GEMV body; KH: the `_k64` twin (activation rows at round_up(K, 128)).
// 2026-10-21: CS: `cs` is the per-expert compact-scale base table (w4a4_mx_core.cuh CS).
template <bool KH, bool CS = false>
__device__ __forceinline__ void w4a4_moe_slots(
    const unsigned char* __restrict__ Aq, const unsigned char* __restrict__ As,
    const float* __restrict__ Ag, const int* __restrict__ ids,
    const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs, const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C, unsigned int N, unsigned int K, unsigned int act_div,
    unsigned int num_experts, const unsigned int* __restrict__ cs = nullptr)
{
    const unsigned int s = blockIdx.y;
    const int id = ids[s];
    // 2026-10-08: Block-uniform exits (one slot per blockIdx.y), before the impl's barrier.
    if (id < 0 || (unsigned int)id >= num_experts) return;
    const unsigned long long bq = packed_ptrs[id];
    if (bq == 0ull) return;
    const unsigned int row = s / act_div;
    const unsigned int ka = KH ? (K + 127u) & ~127u : K;
    if (CS && cs[id] != 0xFFFFFFFFu) {
        w4a4_gemv_mx_tok_impl<1, 4, W4a4RowsIdentity, KH, true>(
            Aq + (unsigned long long)row * (ka >> 1), As + (unsigned long long)row * (ka >> 4),
            Ag + row, (const unsigned char*)bq, (const unsigned char*)scale_ptrs[id],
            scale2_vals[id], C + (unsigned long long)s * N, W4a4RowsIdentity{1u}, N, K, blockIdx.x,
            cs[id]);
        return;
    }
    w4a4_gemv_mx_tok_impl<1, 4, W4a4RowsIdentity, KH>(
        Aq + (unsigned long long)row * (ka >> 1), As + (unsigned long long)row * (ka >> 4),
        Ag + row, (const unsigned char*)bq, (const unsigned char*)scale_ptrs[id],
        scale2_vals[id], C + (unsigned long long)s * N, W4a4RowsIdentity{1u}, N, K, blockIdx.x);
}

#define W4A4_SLOTS_ENTRY(NAME, KH)                                                         \
    extern "C" __global__ __launch_bounds__(W4A4_WARPS * 32) void NAME(                     \
        const unsigned char* __restrict__ Aq, const unsigned char* __restrict__ As,        \
        const float* __restrict__ Ag, const int* __restrict__ ids,                         \
        const unsigned long long* __restrict__ packed_ptrs,                                \
        const unsigned long long* __restrict__ scale_ptrs,                                 \
        const float* __restrict__ scale2_vals, __nv_bfloat16* __restrict__ C,              \
        unsigned int N, unsigned int K, unsigned int act_div, unsigned int num_experts) {  \
        w4a4_moe_slots<KH>(Aq, As, Ag, ids, packed_ptrs, scale_ptrs, scale2_vals, C, N, K,  \
                           act_div, num_experts);                                          \
    }

W4A4_SLOTS_ENTRY(w4a4_gemv_mx8_moe_slots, false)
W4A4_SLOTS_ENTRY(w4a4_gemv_mx8_moe_slots_k64, true)

// 2026-10-21: The `_cs` twins (METRALE_GLM_MOE_SCALE_DICT): the per-expert base table as one more
// argument; every output is the plain entry's bytes.
#define W4A4_SLOTS_CS_ENTRY(NAME, KH)                                                      \
    extern "C" __global__ __launch_bounds__(W4A4_WARPS * 32) void NAME(                     \
        const unsigned char* __restrict__ Aq, const unsigned char* __restrict__ As,        \
        const float* __restrict__ Ag, const int* __restrict__ ids,                         \
        const unsigned long long* __restrict__ packed_ptrs,                                \
        const unsigned long long* __restrict__ scale_ptrs,                                 \
        const float* __restrict__ scale2_vals, __nv_bfloat16* __restrict__ C,              \
        unsigned int N, unsigned int K, unsigned int act_div, unsigned int num_experts,    \
        const unsigned int* __restrict__ cs) {                                             \
        w4a4_moe_slots<KH, true>(Aq, As, Ag, ids, packed_ptrs, scale_ptrs, scale2_vals, C, \
                                 N, K, act_div, num_experts, cs);                          \
    }
W4A4_SLOTS_CS_ENTRY(w4a4_gemv_mx8_moe_slots_cs, false)
W4A4_SLOTS_CS_ENTRY(w4a4_gemv_mx8_moe_slots_k64_cs, true)

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
                                 W4a4RowsTable{s_m, s_a, s_c}, N, K, blockIdx.x);
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

// 2026-10-09: The persistent form of w4a4_moe_union, for up to two projections that read the
// same activations (gate and up). The grid is fixed (a few CTAs per SM, any count works), so a
// CUDA graph replays it whatever the routing. Each CTA lists the union entries whose expert this
// rank holds (u_eid in range, a non-null pointer in either projection's table) in union order, then
// takes a contiguous share of the work items (projection p, live entry e, weight tile x), x
// fastest, and runs each as w4a4_moe_union's block with blockIdx.x = x. The grid form spends a
// CTA on every (tile, entry) pair, ~3 in 4 of them empty at 16 rows on one of three ranks, and
// one launch per projection.
// Every output element is computed by the same w4a4_gemv_mx_tok_impl call, so each token's
// output is bit-identical to the grid form's (and so to the slot kernel's).
// Launch: grid (CTAs, 1, 1), block 256; rows * top_k <= 256, rows <= 8 * MB and
// num_experts <= 65536, else no CTA writes. A projection whose table holds a null pointer for a live entry's expert skips it, as
// the grid form's CTA would. nproj 1 reads only the first table.
// 2026-10-09: One CTA per SM (161 registers at mx16; two per SM caps it at 128 and spills) and
// the L2 prefetch one item ahead. Measured at 16 rows against 2 per SM and 0 or 2 items ahead:
// the model-arch example glm5next_moe_wide_bench. The host's grid is
// glm5next_mlp::W4A4_SWEEP_CTAS_PER_SM per SM.
#define W4A4_SWEEP_MIN_CTAS_PER_SM 1

#define W4A4_SWEEP_PF 1

// 2026-10-09: L2 prefetch of weight tile x (rows 16x .. 16x + 15) of one expert: its packed rows
// and its scale rows are two contiguous ranges. Skipped for a null or unaligned base; the size
// is rounded down to the 16 bytes the bulk prefetch moves in.
// 2026-10-21: CS (compact scales, w4a4_mx_core.cuh): the tile's lo-plane words and hi-plane
// bytes, two more contiguous ranges, in place of its scale rows.
__device__ __forceinline__ void w4a4_prefetch_range(const unsigned char* p, unsigned int n)
{
    n &= ~15u;
    if (p != nullptr && ((unsigned long long)p & 15ull) == 0ull && n != 0u)
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(p), "r"(n) : "memory");
}

template <bool CS = false>
__device__ __forceinline__ void w4a4_sweep_prefetch(const unsigned char* bq,
                                                    const unsigned char* bs, unsigned int x,
                                                    unsigned int N, unsigned int K)
{
    const unsigned int r0 = x * 16u, nr = min(16u, N - r0);
    if constexpr (CS) {
        const unsigned int nc = (K + 127u) >> 7;
        if (bq != nullptr) w4a4_prefetch_range(bq + (unsigned long long)r0 * (K >> 1), nr * (K >> 1));
        if (bs == nullptr) return;
        w4a4_prefetch_range(bs + (unsigned long long)r0 * nc * 4u, nr * nc * 4u);
        w4a4_prefetch_range(bs + (unsigned long long)N * nc * 4u + (unsigned long long)r0 * nc, nr * nc);
    } else {
        const unsigned char* q = bq + (unsigned long long)r0 * (K >> 1);
        const unsigned char* sc = bs + (unsigned long long)r0 * (K >> 4);
        const unsigned int qn = (nr * (K >> 1)) & ~15u, sn = (nr * (K >> 4)) & ~15u;
        if (bq != nullptr && ((unsigned long long)q & 15ull) == 0ull && qn != 0u)
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(q), "r"(qn) : "memory");
        if (bs != nullptr && ((unsigned long long)sc & 15ull) == 0ull && sn != 0u)
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(sc), "r"(sn) : "memory");
    }
}

// 2026-10-09: OWN: one row (rows == 1) whose entries are its own slots: u_eid is the router's ids
// row and entry u is slot u (u_slot is not read). Every slot of a local expert runs, a repeated id
// included, as in the slot GEMV; the union tables would keep one slot of a repeated id.
// 2026-10-10: OWN also writes +0.0 over every output row of a slot it computes nothing for, so
// C needs no zeroing first (N % 8 == 0: 16-byte rows). Every live entry's table pointers and
// weight_scale_2 are read once at setup into shared memory, so no item waits on a global table
// load before its first weight byte (measured on the tp shapes: -4 to -7% at 2 to 16 rows, the
// same at one row; glm5next_moe_narrow_bench GLM_BENCH_TP=1).
// 2026-10-21: SL: launched stream-ordered (no wait), the successor triggered after the last item;
// else pdl_enter() before any global access.
// 2026-10-21: CS: each projection also has a per-expert table cs0 / cs1: the expert's scale base
// code in every byte (its scales compact, w4a4_mx_core.cuh), or 0xFFFFFFFF (plain scales: a
// tensor whose codes span more than 32). Each item runs the impl of its expert's format.
template <int MB, int PF, bool KH, bool OWN, bool SL, bool CS = false>
__device__ __forceinline__ void w4a4_moe_union_sweep(
    const unsigned char* __restrict__ Aq, const unsigned char* __restrict__ As,
    const float* __restrict__ Ag, const int* __restrict__ u_eid, const int* __restrict__ u_slot,
    const unsigned long long* __restrict__ packed0, const unsigned long long* __restrict__ scale0,
    const float* __restrict__ s2_0, __nv_bfloat16* __restrict__ C0,
    const unsigned long long* __restrict__ packed1, const unsigned long long* __restrict__ scale1,
    const float* __restrict__ s2_1, __nv_bfloat16* __restrict__ C1, unsigned int nproj,
    unsigned int N, unsigned int K, unsigned int rows, unsigned int top_k, unsigned int act_div,
    unsigned int num_experts, const unsigned int* __restrict__ cs0 = nullptr,
    const unsigned int* __restrict__ cs1 = nullptr)
{
    const unsigned int T = rows * top_k;
    // 2026-10-09: Block-uniform refusals before any barrier.
    // s_live packs (entry << 16) | expert, so an expert id must fit 16 bits.
    if (T > W4A4_WARPS * 32u || rows > 8u * MB || nproj == 0u || nproj > 2u ||
        num_experts > 0x10000u || (OWN && (rows != 1u || (N & 7u) != 0u)))
        return;
    // 2026-10-21: PDL (pdl.cuh): every load below follows the routing (u_eid, the predecessor's
    // output), so the whole body waits.
    if constexpr (!SL) pdl_enter();
    __shared__ int s_live[W4A4_WARPS * 32];
    // 2026-10-10: Per live entry e and projection p: packed and scale bases, weight_scale_2.
    __shared__ unsigned long long s_q[2][W4A4_WARPS * 32], s_s[2][W4A4_WARPS * 32];
    __shared__ float s_f[2][W4A4_WARPS * 32];
    __shared__ unsigned int s_b[CS ? 2 : 1][CS ? W4A4_WARPS * 32 : 1];
    __shared__ unsigned int s_wcnt[W4A4_WARPS], s_dead[W4A4_WARPS];
    __shared__ int s_sl[8 * MB];
    __shared__ unsigned int s_a[8 * MB], s_c[8 * MB], s_m;
    const unsigned int tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;

    int id = -1;
    unsigned long long q0 = 0ull, q1 = 0ull;
    if (tid < T) {
        id = u_eid[tid];
        if (id < 0 || (unsigned int)id >= num_experts) {
            id = -1;
        } else {
            q0 = packed0[id];
            q1 = nproj == 2u ? packed1[id] : 0ull;
            if (q0 == 0ull && q1 == 0ull) id = -1;
        }
    }
    const unsigned int bal = __ballot_sync(0xFFFFFFFFu, id >= 0);
    // 2026-10-10: OWN: the slots this rank writes no product for (expert invalid or remote).
    const unsigned int dead = __ballot_sync(0xFFFFFFFFu, tid < T && id < 0);
    if (lane == 0u) {
        s_wcnt[warp] = (unsigned int)__popc(bal);
        s_dead[warp] = dead;
    }
    __syncthreads();
    unsigned int before = 0u, count = 0u;
    #pragma unroll
    for (unsigned int w = 0; w < W4A4_WARPS; w++) {
        before += w < warp ? s_wcnt[w] : 0u;
        count += s_wcnt[w];
    }
    // 2026-10-09: s_live[e] = (union entry << 16) | expert of the e-th live entry.
    if (id >= 0) {
        const unsigned int e = before + (unsigned int)__popc(bal & ((1u << lane) - 1u));
        s_live[e] = (int)((tid << 16) | (unsigned int)id);
        s_q[0][e] = q0;
        s_s[0][e] = scale0[id];
        s_f[0][e] = s2_0[id];
        if constexpr (CS) s_b[0][e] = cs0[id];
        if (nproj == 2u) {
            s_q[1][e] = q1;
            s_s[1][e] = scale1[id];
            s_f[1][e] = s2_1[id];
            if constexpr (CS) s_b[1][e] = cs1[id];
        }
    }
    __syncthreads();

    // 2026-10-10: OWN: every output row of a dead slot is zeroed here, in each projection (the
    // slot GEMV writes nothing there, and a zeroed C reads +0.0, which these stores write). The
    // CTAs share the rows' 16-byte words.
    if constexpr (OWN) {
        const unsigned int words = N >> 3, cta_threads = gridDim.x * blockDim.x;
        for (unsigned int dw = 0; dw < W4A4_WARPS; dw++) {
            for (unsigned int bits = s_dead[dw]; bits != 0u; bits &= bits - 1u) {
                const unsigned int slot = dw * 32u + (unsigned int)__ffs(bits) - 1u;
                for (unsigned int p = 0; p < nproj; p++) {
                    uint4* row = (uint4*)((p == 0u ? C0 : C1) + (unsigned long long)slot * N);
                    for (unsigned int i = blockIdx.x * blockDim.x + tid; i < words; i += cta_threads)
                        row[i] = make_uint4(0u, 0u, 0u, 0u);
                }
            }
        }
    }

    // 2026-10-09: Work item w = (p * count + e) * tiles + x; this CTA takes [w0, w1).
    const unsigned int tiles = (N + 15u) >> 4;
    const unsigned int work = nproj * count * tiles;
    const unsigned int w0 = (unsigned int)((unsigned long long)work * blockIdx.x / gridDim.x);
    const unsigned int w1 = (unsigned int)((unsigned long long)work * (blockIdx.x + 1u) / gridDim.x);
    unsigned int cur = 0xFFFFFFFFu;
    for (unsigned int w = w0; w < w1; w++) {
        const unsigned int x = w % tiles, pe = w / tiles;
        const unsigned int p = pe / count, e = pe % count;
        const unsigned int u = (unsigned int)s_live[e] >> 16;
        // 2026-10-09: While this item computes, pull the weight tile of the item PF
        // ahead into L2 (a hint: it changes no value).
        if (PF != 0 && tid == 0u && w + PF < w1) {
            const unsigned int wn = w + PF, pn = wn / tiles;
            const unsigned int qn = pn / count, en = pn % count;
            if (CS && s_b[CS ? qn : 0][CS ? en : 0] != 0xFFFFFFFFu)
                w4a4_sweep_prefetch<true>((const unsigned char*)s_q[qn][en],
                                          (const unsigned char*)s_s[qn][en], wn % tiles, N, K);
            else
                w4a4_sweep_prefetch((const unsigned char*)s_q[qn][en],
                                    (const unsigned char*)s_s[qn][en], wn % tiles, N, K);
        }
        // 2026-10-09: The previous item's reduction and token tables are done with.
        __syncthreads();
        if (u != cur) {
            if (tid < rows) s_sl[tid] = OWN ? (int)u : u_slot[u * rows + tid];
            __syncthreads();
            if (tid == 0u) {
                unsigned int m = 0;
                for (unsigned int r = 0; r < rows; r++) {
                    const int sl = s_sl[r];
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
            cur = u;
        }
        const unsigned long long bq = s_q[p][e];
        if (s_m == 0u || bq == 0ull) continue;
        const unsigned long long bs = s_s[p][e];
        const float s2 = s_f[p][e];
        if (CS && s_b[CS ? p : 0][CS ? e : 0] != 0xFFFFFFFFu) {
            w4a4_gemv_mx_tok_impl<MB, 4, W4a4RowsTable, KH, true>(
                Aq, As, Ag, (const unsigned char*)bq, (const unsigned char*)bs, s2, p == 0u ? C0 : C1,
                W4a4RowsTable{s_m, s_a, s_c}, N, K, x, s_b[CS ? p : 0][CS ? e : 0]);
            continue;
        }
        w4a4_gemv_mx_tok_impl<MB, 4, W4a4RowsTable, KH>(
            Aq, As, Ag, (const unsigned char*)bq, (const unsigned char*)bs, s2, p == 0u ? C0 : C1,
            W4a4RowsTable{s_m, s_a, s_c}, N, K, x);
    }
    if constexpr (SL) pdl_trigger();
}

#define W4A4_SWEEP_ENTRY(NAME, MB, PF, MINB, KH, OWN, SL)                                                 \
    extern "C" __global__ __launch_bounds__(W4A4_WARPS * 32, MINB) void NAME( \
        const unsigned char* __restrict__ Aq, const unsigned char* __restrict__ As,             \
        const float* __restrict__ Ag, const int* __restrict__ u_eid,                            \
        const int* __restrict__ u_slot, const unsigned long long* __restrict__ packed0,         \
        const unsigned long long* __restrict__ scale0, const float* __restrict__ s2_0,          \
        __nv_bfloat16* __restrict__ C0, const unsigned long long* __restrict__ packed1,         \
        const unsigned long long* __restrict__ scale1, const float* __restrict__ s2_1,          \
        __nv_bfloat16* __restrict__ C1, unsigned int nproj, unsigned int N, unsigned int K,     \
        unsigned int rows, unsigned int top_k, unsigned int act_div, unsigned int num_experts) { \
        w4a4_moe_union_sweep<MB, PF, KH, OWN, SL>(Aq, As, Ag, u_eid, u_slot, packed0, scale0, s2_0, C0,     \
                                 packed1,                                                        \
                                 scale1, s2_1, C1, nproj, N, K, rows, top_k, act_div,            \
                                 num_experts);                                                   \
    }

W4A4_SWEEP_ENTRY(w4a4_gemv_mx8_moe_union_sweep, 1, W4A4_SWEEP_PF, W4A4_SWEEP_MIN_CTAS_PER_SM, false, false, false)
W4A4_SWEEP_ENTRY(w4a4_gemv_mx16_moe_union_sweep, 2, W4A4_SWEEP_PF, W4A4_SWEEP_MIN_CTAS_PER_SM, false, false, false)
// 2026-10-09: One row over its own slots (OWN above): the slot GEMV's outputs on the persistent
// grid, gate and up in one launch, with no union build. u_eid is the ids row; rows must be 1.
W4A4_SWEEP_ENTRY(w4a4_gemv_mx8_moe_slots_sweep, 1, W4A4_SWEEP_PF, W4A4_SWEEP_MIN_CTAS_PER_SM, false, true, false)
// 2026-10-10: The `_k64` twins (the routed down projection of a 64-unit expert slice).
W4A4_SWEEP_ENTRY(w4a4_gemv_mx8_moe_union_sweep_k64, 1, W4A4_SWEEP_PF, W4A4_SWEEP_MIN_CTAS_PER_SM, true, false, false)
W4A4_SWEEP_ENTRY(w4a4_gemv_mx16_moe_union_sweep_k64, 2, W4A4_SWEEP_PF, W4A4_SWEEP_MIN_CTAS_PER_SM, true, false, false)
W4A4_SWEEP_ENTRY(w4a4_gemv_mx8_moe_slots_sweep_k64, 1, W4A4_SWEEP_PF, W4A4_SWEEP_MIN_CTAS_PER_SM, true, true, false)
// 2026-10-21: The gate/up sweeps launched stream-ordered (METRALE_GLM_MOE_SWEEP_SERIAL).
W4A4_SWEEP_ENTRY(w4a4_gemv_mx8_moe_union_sweep_sl, 1, W4A4_SWEEP_PF, W4A4_SWEEP_MIN_CTAS_PER_SM, false, false, true)
W4A4_SWEEP_ENTRY(w4a4_gemv_mx16_moe_union_sweep_sl, 2, W4A4_SWEEP_PF, W4A4_SWEEP_MIN_CTAS_PER_SM, false, false, true)

// 2026-10-21: The `_cs` twins (METRALE_GLM_MOE_SCALE_DICT): the same sweeps over experts whose
// block scales may be compact (w4a4_mx_core.cuh CS), with the per-expert base tables cs0 / cs1 as
// two more arguments. Every expert's outputs are the plain twin's bytes.
#define W4A4_SWEEP_CS_ENTRY(NAME, MB, KH, OWN, SL)                                                       \
    extern "C" __global__ __launch_bounds__(W4A4_WARPS * 32, W4A4_SWEEP_MIN_CTAS_PER_SM) void NAME(      \
        const unsigned char* __restrict__ Aq, const unsigned char* __restrict__ As,                     \
        const float* __restrict__ Ag, const int* __restrict__ u_eid,                                    \
        const int* __restrict__ u_slot, const unsigned long long* __restrict__ packed0,                 \
        const unsigned long long* __restrict__ scale0, const float* __restrict__ s2_0,                  \
        __nv_bfloat16* __restrict__ C0, const unsigned long long* __restrict__ packed1,                 \
        const unsigned long long* __restrict__ scale1, const float* __restrict__ s2_1,                  \
        __nv_bfloat16* __restrict__ C1, unsigned int nproj, unsigned int N, unsigned int K,             \
        unsigned int rows, unsigned int top_k, unsigned int act_div, unsigned int num_experts,          \
        const unsigned int* __restrict__ cs0, const unsigned int* __restrict__ cs1) {                   \
        w4a4_moe_union_sweep<MB, W4A4_SWEEP_PF, KH, OWN, SL, true>(                                     \
            Aq, As, Ag, u_eid, u_slot, packed0, scale0, s2_0, C0, packed1, scale1, s2_1, C1, nproj, N,  \
            K, rows, top_k, act_div, num_experts, cs0, cs1);                                            \
    }

W4A4_SWEEP_CS_ENTRY(w4a4_gemv_mx8_moe_union_sweep_cs, 1, false, false, false)
W4A4_SWEEP_CS_ENTRY(w4a4_gemv_mx16_moe_union_sweep_cs, 2, false, false, false)
W4A4_SWEEP_CS_ENTRY(w4a4_gemv_mx8_moe_slots_sweep_cs, 1, false, true, false)
W4A4_SWEEP_CS_ENTRY(w4a4_gemv_mx8_moe_union_sweep_k64_cs, 1, true, false, false)
W4A4_SWEEP_CS_ENTRY(w4a4_gemv_mx16_moe_union_sweep_k64_cs, 2, true, false, false)
W4A4_SWEEP_CS_ENTRY(w4a4_gemv_mx8_moe_slots_sweep_k64_cs, 1, true, true, false)
W4A4_SWEEP_CS_ENTRY(w4a4_gemv_mx8_moe_union_sweep_sl_cs, 1, false, false, true)
W4A4_SWEEP_CS_ENTRY(w4a4_gemv_mx16_moe_union_sweep_sl_cs, 2, false, false, true)

#endif
