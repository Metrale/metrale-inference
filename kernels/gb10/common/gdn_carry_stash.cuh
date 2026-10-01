// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-01: The carried-state GDN verify's stash layout, shared by every kernel that reads or
// writes it: gated_delta_rule_carry.cu (the WY verify, the fold and the conv twins) and the
// exact per-token chain twins (gdn_exact_carry.cu in the model directories).
//
// Owner: gb10 kernels.
// Invariants:
// - Stash per (layer, slot), in floats: vn[CAP][num_v_heads][v_dim] | g[CAP][num_v_heads] |
//   sk[CAP][num_v_heads][k_dim], with k_dim == v_dim == 128 (checked on the host);
//   ops::gdn_carry_seq_floats is the host side of this layout.
// - CARRY_CAP is ops::GDN_CARRY_CAP.

#ifndef GDN_CARRY_STASH_CUH
#define GDN_CARRY_STASH_CUH

#define CARRY_KD 128u
#define CARRY_VD 128u
// 2026-09-26: Rows the stash holds per slot; ops::GDN_CARRY_CAP on the host.
#define CARRY_CAP 8u

#define CARRY_VN(S, T, VH) ((S) + ((T) * num_v_heads + (VH)) * CARRY_VD)
#define CARRY_G(S, T, VH)  ((S) + CARRY_CAP * num_v_heads * CARRY_VD + (T) * num_v_heads + (VH))
#define CARRY_SK(S, T, VH) ((S) + CARRY_CAP * num_v_heads * CARRY_VD + CARRY_CAP * num_v_heads \
                            + ((T) * num_v_heads + (VH)) * CARRY_KD)

// 2026-09-26: The pending rows t < np applied to one element of row j of a column, from the
// stashed key rows `pk` and this thread's gates `pgr` and vn values `pvr` (registers: the
// loop is unrolled over CARRY_CAP and guarded, so the arrays index statically).
__device__ __forceinline__ float carry_fold(
    float x, unsigned int np, unsigned int j,
    const float (*pk)[CARRY_KD], const float* pgr, const float* pvr
) {
    #pragma unroll
    for (unsigned int t = 0; t < CARRY_CAP; ++t) if (t < np) x = pgr[t] * x + pk[t][j] * pvr[t];
    return x;
}

// 2026-10-01: gdn_carry_flush's body: apply a slot's pending rows to H and write it back, with the
// fold above (the update `g * h + k * vn` of decodes without a state clamp). Grid (num_v_heads,
// batch, layers), block 128; see gdn_carry_flush for the layout.
__device__ __forceinline__ void carry_flush_body(float* const* __restrict__ h_table,
    unsigned long long table_layer_entries,
    const float* __restrict__ carry_base,
    unsigned long long carry_layer_floats,
    const unsigned int* __restrict__ slot_tab,
    const unsigned int* __restrict__ pend,
    unsigned int pend_layer_entries,
    unsigned int seq_floats,
    unsigned int batch_size,
    unsigned int num_v_heads) {
    const unsigned int vh = blockIdx.x;
    const unsigned int b = blockIdx.y;
    const unsigned int l = blockIdx.z;
    if (vh >= num_v_heads || b >= batch_size) return;
    const unsigned int slot = slot_tab[b];
    const unsigned int np = pend[(unsigned long long)l * pend_layer_entries + slot];
    if (np == 0) return;
    const unsigned int tid = threadIdx.x;
    float* H = h_table[l * table_layer_entries + b] + (unsigned long long)vh * CARRY_KD * CARRY_VD;
    const float* S = carry_base + l * carry_layer_floats + (unsigned long long)slot * seq_floats;
    __shared__ float pk[CARRY_CAP][CARRY_KD], pg[CARRY_CAP];
    float pvr[CARRY_CAP], pgr[CARRY_CAP];
    #pragma unroll
    for (unsigned int t = 0; t < CARRY_CAP; ++t) {
        if (t < np) { pk[t][tid] = CARRY_SK(S, t, vh)[tid]; pvr[t] = CARRY_VN(S, t, vh)[tid]; }
        else pvr[t] = 0.0f;
    }
    if (tid < np) pg[tid] = *CARRY_G(S, tid, vh);
    __syncthreads();
    #pragma unroll
    for (unsigned int t = 0; t < CARRY_CAP; ++t) pgr[t] = t < np ? pg[t] : 0.0f;
    #pragma unroll 4
    for (unsigned int j = 0; j < CARRY_KD; ++j)
        H[j * CARRY_VD + tid] = carry_fold(H[j * CARRY_VD + tid], np, j, pk, pgr, pvr);
}

#endif
