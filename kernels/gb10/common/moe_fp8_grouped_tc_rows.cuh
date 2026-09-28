// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-28: The block-row contract shared by the tensor-core grouped FP8 MoE decode kernels
// (moe_fp8_grouped_tc.cu, moe_fp8_grouped_tc_w8a8.cu): rows run TC_ROWS at a time; block row
// y < ceil(num_tokens / TC_ROWS) is the shared expert's slot y, block row S + i is active
// expert i (moe_fp8_grouped_sort's list), and blocks past active_count[0] return.
//
// Owner: gb10 kernels.
// Invariants: TC_ROWS must equal FP8_GROUPED_TC_ROWS_PER_PASS in fp8_moe_grouped.rs.

#pragma once

#define TC_ROWS 8

// 2026-09-28: The row range of this block row: the shared expert's slot or active expert i.
__device__ __forceinline__ bool tc_block_rows(
    const int* __restrict__ expert_offsets, const int* __restrict__ active_experts,
    const int* __restrict__ active_count, unsigned int num_tokens,
    bool* is_shared, unsigned int* expert, unsigned int* begin, unsigned int* end
) {
    const unsigned int y = blockIdx.y;
    const unsigned int shared_slots = (num_tokens + TC_ROWS - 1) / TC_ROWS;
    *is_shared = (y < shared_slots);
    if (*is_shared) {
        *expert = 0; *begin = y * TC_ROWS; *end = min(num_tokens, (y + 1) * TC_ROWS);
    } else {
        const unsigned int i = y - shared_slots;
        if ((int)i >= active_count[0]) return false;
        *expert = (unsigned int)active_experts[i];
        *begin = (unsigned int)expert_offsets[*expert];
        *end = (unsigned int)expert_offsets[*expert + 1];
    }
    return *begin < *end;
}
