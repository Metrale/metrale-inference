// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-15: The selected-pools payload of the sequence-parallel DSA selection
// (`METRALE_GLM_DSA_SP_SELECT_SEL`). A split rank folds its own rows' candidacy into their
// selected pools (dsa_sel_fold), the ranks all-gather the folded `[Q, select_k]` rows, and every
// rank expands all rows from them (dsa_expand_selection_folded), which needs no candidacy row:
// the expansion of a folded row is dsa_expand_selection's row, byte for byte.
//
// Owner: gb10 kernels.
// Invariants:
// - dsa_sel_fold leaves entry j of row r as p = selected[r][j] when q_mask[r] != 0,
//   j < row_select_k(r) = min((q_pos[r] + 1) / KP, select_k), p >= 0 and valid_cand[r][p] != 0,
//   and DSA_INVALID otherwise: the test dsa_expand_selection_body applies to the entries it
//   reads, on the same operands, so every entry it would take is kept and every other is -1.
// - dsa_expand_selection_folded runs dsa_expand_selection_body<false, true>: the body of
//   dsa_expand_selection with the candidacy test reduced to p >= 0. Over a folded row it takes
//   the same pools and writes the same token ids and tail as dsa_expand_selection over the
//   unfolded row and its candidacy row.

#include "dsa_indexer_body.cuh"

// 2026-10-15: Fold row blockIdx.x's candidacy into its selected pools, in place. Launch: grid
// (Q, 1, 1), block ROW_BLOCK. `selected`, `q_pos` and `q_mask` start at the pass's first row,
// `valid_cand` at its candidacy row 0 (P bytes per row).
extern "C" __global__ void dsa_sel_fold(
    int* __restrict__ selected,
    const unsigned char* __restrict__ valid_cand,
    const int* __restrict__ q_pos,
    const unsigned char* __restrict__ q_mask,
    unsigned int P,
    unsigned int KP,
    unsigned int select_k
) {
    const unsigned int r = blockIdx.x;
    int* row = selected + (size_t)r * select_k;
    const bool live = q_mask[r] != 0;
    const unsigned int row_pools = (unsigned int)(q_pos[r] + 1) / KP;
    const unsigned int row_select_k = (row_pools < select_k) ? row_pools : select_k;
    for (unsigned int j = threadIdx.x; j < select_k; j += blockDim.x) {
        const int p = row[j];
        const bool ok = live && j < row_select_k && p >= 0
                        && valid_cand[(size_t)r * P + p] != 0;
        row[j] = ok ? p : DSA_INVALID;
    }
}

// 2026-10-15: dsa_expand_selection over folded selected pools: the same arguments without the
// candidacy row. Launch: as dsa_expand_selection (grid (Q, 1, 1), block ROW_BLOCK).
extern "C" __global__ void dsa_expand_selection_folded(
    const int* __restrict__ selected,
    const int* __restrict__ pool_indices,
    const unsigned char* __restrict__ valid_keys,
    const int* __restrict__ q_pos,
    const unsigned char* __restrict__ q_mask,
    int* __restrict__ out,
    unsigned int Q,
    unsigned int P,
    unsigned int KP,
    unsigned int S,
    unsigned int select_k,
    unsigned int width,
    int first_key,
    int always_tail
) {
    dsa_expand_selection_body<false, true>(selected, pool_indices, nullptr, valid_keys, q_pos,
                                           q_mask, out, P, KP, S, select_k, width, first_key,
                                           always_tail, nullptr, blockIdx.x);
}
