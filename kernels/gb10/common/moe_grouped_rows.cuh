// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-27: The block-row mapping of the grouped MoE decode kernels: which expert and which
// sorted positions one block row serves.
//
// Owner: gb10 kernels.
// Invariants:
// - Rows are in sorted order (moe_sort_by_expert): expert e owns positions
//   [expert_offsets[e], expert_offsets[e + 1]).
// - blockIdx.y < S = ceil(num_tokens / rows) is the shared expert, tokens [y * rows,
//   (y + 1) * rows); blockIdx.y = S + i is active_experts[i] (moe_fp8_grouped_compact), and a
//   block with i at or past active_count[0] serves nothing. The grid does not depend on the
//   routing. moe_shared_expert_fused_fp8_grouped.cu defines the same mapping inline.

#pragma once

// 2026-09-27: The rows of this block; false when it serves none.
__device__ __forceinline__ bool moe_grouped_block_rows(
    const int* __restrict__ expert_offsets, const int* __restrict__ active_experts,
    const int* __restrict__ active_count, unsigned int num_tokens, unsigned int rows,
    bool* is_shared, unsigned int* expert, unsigned int* begin, unsigned int* end
) {
    const unsigned int y = blockIdx.y;
    const unsigned int shared_slots = (num_tokens + rows - 1) / rows;
    *is_shared = (y < shared_slots);
    if (*is_shared) {
        *expert = 0; *begin = y * rows; *end = min(num_tokens, (y + 1) * rows);
    } else {
        const unsigned int i = y - shared_slots;
        if ((int)i >= active_count[0]) return false;
        *expert = (unsigned int)active_experts[i];
        *begin = (unsigned int)expert_offsets[*expert];
        *end = (unsigned int)expert_offsets[*expert + 1];
    }
    return *begin < *end;
}

// 2026-09-27: Input row of sorted position pos: pos itself for the shared expert, else
// sorted_token_ids[pos].
__device__ __forceinline__ unsigned int moe_grouped_a_row(
    const int* __restrict__ sorted_token_ids, bool is_shared, unsigned int pos
) {
    return is_shared ? pos : (unsigned int)sorted_token_ids[pos];
}
