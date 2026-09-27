// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-27: moe_sort_by_expert and the active-expert list in one launch, for the grouped
// FP8 MoE decode (moe_shared_expert_fused_fp8_grouped.cu). It replaces the single-thread
// moe_fp8_grouped_compact that followed the sort.
//
// Owner: gb10 kernels.
// Invariants:
// - Outputs are moe_sort_by_expert's (moe_permute.cu) and the active-expert list:
//   expert_offsets[0..=num_experts] the exclusive prefix of the
//   per-expert slot counts; sorted_token_ids / sorted_expert_ids / token_to_perm the
//   scatter, where the order of an expert's slots among themselves follows shared-memory
//   atomics, as in moe_sort_by_expert (the grouped kernels' sums do not depend on it);
//   active_experts[0..count] the experts with at least one slot in ascending id order and
//   active_count[0] = count.
// - Launch: grid 1, block PMS_BLOCK; num_experts <= PMS_BLOCK * PMS_PER_THREAD. The two
//   prefix sums (slots and active experts) run across the block (four experts per thread,
//   warp shuffles, one shared array of warp totals) instead of on one thread.

#define PMS_BLOCK 256
#define PMS_PER_THREAD 4
#define PMS_MAX_EXPERTS (PMS_BLOCK * PMS_PER_THREAD)

extern "C" __global__ void __launch_bounds__(PMS_BLOCK) moe_fp8_grouped_sort(
    const unsigned int* __restrict__ topk_ids,
    int* __restrict__ sorted_token_ids,
    int* __restrict__ sorted_expert_ids,
    int* __restrict__ expert_offsets,
    int* __restrict__ token_to_perm,
    int* __restrict__ active_experts,
    int* __restrict__ active_count,
    unsigned int total_expanded,
    unsigned int num_experts,
    unsigned int topk
) {
    __shared__ unsigned int counts[PMS_MAX_EXPERTS];
    __shared__ unsigned int offsets[PMS_MAX_EXPERTS];
    __shared__ unsigned int warp_rows[PMS_BLOCK / 32];
    __shared__ unsigned int warp_live[PMS_BLOCK / 32];
    const unsigned int lane = threadIdx.x % 32;
    const unsigned int warp = threadIdx.x / 32;

    for (unsigned int i = threadIdx.x; i < num_experts; i += PMS_BLOCK) counts[i] = 0;
    __syncthreads();
    for (unsigned int i = threadIdx.x; i < total_expanded; i += PMS_BLOCK)
        atomicAdd(&counts[topk_ids[i]], 1u);
    __syncthreads();

    // 2026-09-27: This thread's experts e0 .. e0 + 3: slot counts and how many are active.
    const unsigned int e0 = threadIdx.x * PMS_PER_THREAD;
    unsigned int c[PMS_PER_THREAD];
    unsigned int rows = 0, live = 0;
    #pragma unroll
    for (int j = 0; j < PMS_PER_THREAD; j++) {
        c[j] = (e0 + j < num_experts) ? counts[e0 + j] : 0u;
        rows += c[j];
        live += (c[j] != 0u);
    }
    unsigned int rows_incl = rows, live_incl = live;
    #pragma unroll
    for (unsigned int o = 1; o < 32; o <<= 1) {
        const unsigned int r = __shfl_up_sync(0xFFFFFFFFu, rows_incl, o);
        const unsigned int l = __shfl_up_sync(0xFFFFFFFFu, live_incl, o);
        if (lane >= o) {
            rows_incl += r;
            live_incl += l;
        }
    }
    if (lane == 31) {
        warp_rows[warp] = rows_incl;
        warp_live[warp] = live_incl;
    }
    __syncthreads();
    unsigned int rows_base = 0, live_base = 0;
    for (unsigned int w = 0; w < warp; w++) {
        rows_base += warp_rows[w];
        live_base += warp_live[w];
    }
    unsigned int r = rows_base + rows_incl - rows;
    unsigned int a = live_base + live_incl - live;
    #pragma unroll
    for (int j = 0; j < PMS_PER_THREAD; j++) {
        const unsigned int e = e0 + j;
        if (e < num_experts) {
            offsets[e] = r;
            expert_offsets[e] = (int)r;
            if (c[j] != 0u) active_experts[a++] = (int)e;
            r += c[j];
            counts[e] = 0;
        }
    }
    if (threadIdx.x == PMS_BLOCK - 1) {
        expert_offsets[num_experts] = (int)r;
        active_count[0] = (int)a;
    }
    __syncthreads();

    for (unsigned int i = threadIdx.x; i < total_expanded; i += PMS_BLOCK) {
        const unsigned int expert_id = topk_ids[i];
        const unsigned int pos = offsets[expert_id] + atomicAdd(&counts[expert_id], 1u);
        sorted_token_ids[pos] = (int)(i / topk);
        sorted_expert_ids[pos] = (int)expert_id;
        token_to_perm[i] = (int)pos;
    }
}
