// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-27: moe_sort_by_expert and the active-expert list in one launch, for the grouped
// FP8 MoE decode (moe_shared_expert_fused_fp8_grouped.cu). It replaces the single-thread
// moe_fp8_grouped_compact that followed the sort.
// 2026-10-14: The body is pms_sort_block (moe_fp8_grouped_sort.cuh), which the Hopper fused
// router also runs.
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

#include "moe_fp8_grouped_sort.cuh"

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
    pms_sort_block(topk_ids, sorted_token_ids, sorted_expert_ids, expert_offsets, token_to_perm, active_experts,
                   active_count, total_expanded, num_experts, topk);
}
