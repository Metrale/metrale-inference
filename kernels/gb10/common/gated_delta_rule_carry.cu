// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-26: Carried-state GDN verify for K = 2, 3, 4 rows per sequence
// (gdn_carry_wy2/3/4) and its standalone fold (gdn_carry_flush), with the same treatment of
// the conv window (gdn_carry_conv, gdn_carry_conv_flush, at the end of this file).
//
// Owner: gb10 kernels.
// Invariants:
// - The state in memory lags the sequence by `pend[slot]` accepted rows (at most CARRY_CAP),
//   held in the slot's stash. A verify kernel applies them to its copy of H with the parent
//   kernels' update expression in row order, so that copy is what the parent kernels would
//   have committed. It writes that H back and restarts the stash on every verify (the
//   eager form) or only when this verify's K rows might not fit behind the pending ones,
//   pend + K > CARRY_CAP (the lazy form).
// - It then computes `output` with gated_delta_rule_wy{2,3,4}'s expressions and
//   accumulation order, writes no other h-state, and stashes its K rows behind the pending
//   ones (at row 0 after a write-back): the vn vectors, the clamped gates and the key
//   rows, as the floats it used. The key rows are stored once per v-head, so each block
//   reads and writes only its own part of the stash.
// - gdn_carry_flush applies `pend` rows the same way and writes H. Neither kernel clears
//   `pend`; the host owns it.
// gdn_carry_microtest (model-arch examples) checks outputs and states over several verify
// rounds bitwise against the parent kernels.
//
// State traffic per verify: one read of H, plus one write (eager form) or one write every
// few verifies, when the stash fills (lazy form). The parent kernels read H once (resident) or twice and write K blobs, and
// a partial accept costs one more read and write for the restore copy.
//
// Stash per (layer, slot), in floats: vn[CAP][num_v_heads][v_dim] | g[CAP][num_v_heads] |
// sk[CAP][num_v_heads][k_dim]. k_dim == v_dim == 128 is checked on the host. Grid
// (num_v_heads, batch), block 128. The h-state is a per-sequence pointer table (slab 0 of
// the verify WY tables); slot_tab maps a batch position to its carry slot, and each
// position's engaged word (engaged_flag[b]) is set to 1, or to 2 when the kernel wrote the
// state back; the host reads it to learn how many rows stay pending.
//
// Two forms: gdn_carry_wy{K} writes the state back on every verify (pass 2 then re-reads
// it, which stays in L2 at moderate width) and folds at most four pending rows, and
// gdn_carry_wy{K}_lazy writes it back only when the stash is full and keeps half of the
// column in registers across the passes, which pays off from 16 sequences
// (ops::GDN_CARRY_LAZY_MIN_SEQS). Both give the same bits.

#include <cuda_bf16.h>
#include "gdn_reduce.cuh"
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

// 2026-09-26: The same fold for the eager form, at most four rows from registers (`pv0`..
// `pv3`), predicated so that the unrolled pass keeps its loads independent. The host never
// hands an eager launch a slot with more than four pending rows (model-engine
// gdn_carry.rs folds such a slot first).
__device__ __forceinline__ float carry_fold4(
    float x, unsigned int np, unsigned int j, const float (*pk)[CARRY_KD], const float* pg,
    float pv0, float pv1, float pv2, float pv3
) {
    if (np > 0) x = pg[0] * x + pk[0][j] * pv0;
    if (np > 1) x = pg[1] * x + pk[1][j] * pv1;
    if (np > 2) x = pg[2] * x + pk[2][j] * pv2;
    if (np > 3) x = pg[3] * x + pk[3][j] * pv3;
    return x;
}

template <int K, bool LAZY>
__device__ __forceinline__ void gdn_carry_verify(
    float* const* __restrict__ h_table,
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ carry_base,
    const unsigned int* __restrict__ slot_tab,
    const unsigned int* __restrict__ pend,
    unsigned int seq_floats,
    unsigned int batch_size,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int qk_stride,
    unsigned int v_stride,
    unsigned int gb_stride,
    unsigned int k_dim,
    unsigned int* __restrict__ engaged_flag
) {
    const unsigned int vh = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (vh >= num_v_heads || b >= batch_size) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int hr = num_v_heads / num_k_heads;
    const unsigned int kh = vh / hr;
    const unsigned int hv = CARRY_KD * CARRY_VD;
    float* H = h_table[b] + (unsigned long long)vh * hv;
    const unsigned int slot = slot_tab[b];
    float* S = carry_base + (unsigned long long)slot * seq_floats;
    const unsigned int np = pend[slot];
    // 2026-09-26: Write back now if this verify's rows might not fit behind the pending ones;
    // this verify's rows then go to row 0 of the stash, else behind the pending rows.
    const bool wb = !LAZY || np + K > CARRY_CAP;
    const unsigned int base = wb ? 0u : np;
    if (threadIdx.x == 0 && vh == 0) engaged_flag[b] = wb ? 2u : 1u;

    __shared__ float sk[K][CARRY_KD], sq[K][CARRY_KD];
    __shared__ float pk[CARRY_CAP][CARRY_KD], pg[CARRY_CAP];
    __shared__ float smem_warp[4];
    __shared__ float kd[4][4];

    #pragma unroll
    for (int t = 0; t < K; ++t) {
        const unsigned int row = b * K + t;
        sk[t][tid] = (float)key[row * qk_stride + kh * CARRY_KD + tid];
        sq[t][tid] = (float)query[row * qk_stride + kh * CARRY_KD + tid];
    }
    // 2026-09-26: The pending rows are read before the barrier; this block may overwrite the
    // same stash rows further down.
    // The lazy form keeps this thread's vn values and the gates in registers, since it folds
    // the second half of the column again in pass 2; the eager form keeps four vn values in
    // registers and reads the gates from shared memory.
    float pvr[LAZY ? CARRY_CAP : 1], pgr[LAZY ? CARRY_CAP : 1];
    if (LAZY) {
        #pragma unroll
        for (unsigned int t = 0; t < CARRY_CAP; ++t) {
            if (t < np) { pk[t][tid] = CARRY_SK(S, t, vh)[tid]; pvr[t] = CARRY_VN(S, t, vh)[tid]; }
            else pvr[t] = 0.0f;
        }
    } else {
        for (unsigned int t = 0; t < np; ++t) pk[t][tid] = CARRY_SK(S, t, vh)[tid];
    }
    const float pv0 = !LAZY && np > 0 ? CARRY_VN(S, 0, vh)[tid] : 0.0f;
    const float pv1 = !LAZY && np > 1 ? CARRY_VN(S, 1, vh)[tid] : 0.0f;
    const float pv2 = !LAZY && np > 2 ? CARRY_VN(S, 2, vh)[tid] : 0.0f;
    const float pv3 = !LAZY && np > 3 ? CARRY_VN(S, 3, vh)[tid] : 0.0f;
    if (tid < np) pg[tid] = *CARRY_G(S, tid, vh);
    __syncthreads();
    if (LAZY) {
        #pragma unroll
        for (unsigned int t = 0; t < CARRY_CAP; ++t) pgr[t] = t < np ? pg[t] : 0.0f;
    }

    // 2026-09-26: kd[a][c] = k_a . k_c for c < a, in the parents' order (kd10, kd20, kd21,
    // kd30, kd31, kd32), each a separate block reduction.
    #pragma unroll
    for (int a = 1; a < K; ++a) {
        #pragma unroll
        for (int c = 0; c < a; ++c) {
            float p = sk[a][tid] * sk[c][tid];
            float r = metrale_block_reduce_sum(p, smem_warp, tid);
            if (tid == 0) kd[a][c] = r;
            __syncthreads();
        }
    }

    // 2026-09-26: Pass 1 over the column: H with the pending rows applied, written back
    // under `wb`, and summed into hk_t = H . k_t in groups of four j as the parents sum
    // them. The first half of the column stays in registers for pass 2.
    float* __restrict__ Hc = H + tid;
    float H_lo[LAZY ? CARRY_KD / 2 : 1];
    float hk[K];
    #pragma unroll
    for (int t = 0; t < K; ++t) hk[t] = 0.0f;
    #pragma unroll
    for (unsigned int j = 0; j < CARRY_KD; j += 4) {
        float h[4];
        #pragma unroll
        for (unsigned int e = 0; e < 4; ++e) {
            const float x = LAZY
                ? carry_fold(Hc[(j + e) * CARRY_VD], np, j + e, pk, pgr, pvr)
                : carry_fold4(Hc[(j + e) * CARRY_VD], np, j + e, pk, pg, pv0, pv1, pv2, pv3);
            if (wb && np > 0) Hc[(j + e) * CARRY_VD] = x;
            h[e] = x;
            if (LAZY && j < CARRY_KD / 2) H_lo[j + e] = x;
        }
        #pragma unroll
        for (int t = 0; t < K; ++t)
            hk[t] += h[0] * sk[t][j] + h[1] * sk[t][j + 1] + h[2] * sk[t][j + 2] + h[3] * sk[t][j + 3];
    }

    // 2026-09-26: The parents' vn chain (gated_delta_rule_wy4 spells out all four), from
    // the gate clamped as they clamp it.
    float vi[K], g[K], bt[K];
    #pragma unroll
    for (int t = 0; t < K; ++t) {
        const unsigned int row = b * K + t;
        vi[t] = (float)value[row * v_stride + vh * CARRY_VD + tid];
        g[t] = fminf(fmaxf(gate[row * gb_stride + vh], 1e-6f), 1.0f - 1e-6f);
        bt[t] = beta[row * gb_stride + vh];
    }
    float vn[K];
    vn[0] = (vi[0] - g[0] * hk[0]) * bt[0];
    if (K > 1) {
        const float hk1c = g[0] * hk[1] + kd[1][0] * vn[0];
        vn[1] = (vi[1] - g[1] * hk1c) * bt[1];
    }
    if (K > 2) {
        const float hk2c = g[0] * g[1] * hk[2] + g[1] * kd[2][0] * vn[0] + kd[2][1] * vn[1];
        vn[2] = (vi[2] - g[2] * hk2c) * bt[2];
    }
    if (K > 3) {
        const float hk3c = g[0] * g[1] * g[2] * hk[3] + g[1] * g[2] * kd[3][0] * vn[0]
                         + g[2] * kd[3][1] * vn[1] + kd[3][2] * vn[2];
        vn[3] = (vi[3] - g[3] * hk3c) * bt[3];
    }

    // 2026-09-26: Pass 2: the column again, the first half from registers and the second
    // from memory (mostly from L2; pending rows are applied again unless pass 1 wrote them
    // back). Row t updates it in registers, then qd_t += h_t . q_t, per group of four j in
    // the parents' order. Nothing is written back.
    float qd[K];
    #pragma unroll
    for (int t = 0; t < K; ++t) qd[t] = 0.0f;
    const unsigned int refold = wb ? 0u : np;
    #pragma unroll
    for (unsigned int j = 0; j < CARRY_KD; j += 4) {
        float h0, h1, h2, h3;
        if (LAZY && j < CARRY_KD / 2) {
            h0 = H_lo[j]; h1 = H_lo[j + 1]; h2 = H_lo[j + 2]; h3 = H_lo[j + 3];
        } else {
            h0 = Hc[j * CARRY_VD]; h1 = Hc[(j + 1) * CARRY_VD];
            h2 = Hc[(j + 2) * CARRY_VD]; h3 = Hc[(j + 3) * CARRY_VD];
            if (LAZY) {
                h0 = carry_fold(h0, refold, j, pk, pgr, pvr);
                h1 = carry_fold(h1, refold, j + 1, pk, pgr, pvr);
                h2 = carry_fold(h2, refold, j + 2, pk, pgr, pvr);
                h3 = carry_fold(h3, refold, j + 3, pk, pgr, pvr);
            }
        }
        #pragma unroll
        for (int t = 0; t < K; ++t) {
            h0 = g[t] * h0 + sk[t][j] * vn[t];
            h1 = g[t] * h1 + sk[t][j + 1] * vn[t];
            h2 = g[t] * h2 + sk[t][j + 2] * vn[t];
            h3 = g[t] * h3 + sk[t][j + 3] * vn[t];
            qd[t] += h0 * sq[t][j] + h1 * sq[t][j + 1] + h2 * sq[t][j + 2] + h3 * sq[t][j + 3];
        }
    }
    // 2026-09-26: From the runtime k_dim, as the parents compute it: rsqrtf of the
    // literal 128 is folded at compile time and can differ from the hardware result.
    const float s = rsqrtf((float)k_dim);
    #pragma unroll
    for (int t = 0; t < K; ++t)
        output[((b * K + t) * num_v_heads + vh) * CARRY_VD + tid] = __float2bfloat16(qd[t] * s);

    #pragma unroll
    for (int t = 0; t < K; ++t) {
        CARRY_VN(S, base + t, vh)[tid] = vn[t];
        CARRY_SK(S, base + t, vh)[tid] = sk[t][tid];
    }
    if (tid < K) *CARRY_G(S, base + tid, vh) = g[tid];
}

#define CARRY_ENTRY(NAME, K, LAZY) \
extern "C" __global__ void __launch_bounds__(128) NAME( \
    float* const* __restrict__ h_table, const __nv_bfloat16* __restrict__ query, \
    const __nv_bfloat16* __restrict__ key, const __nv_bfloat16* __restrict__ value, \
    const float* __restrict__ gate, const float* __restrict__ beta, \
    __nv_bfloat16* __restrict__ output, float* __restrict__ carry_base, \
    const unsigned int* __restrict__ slot_tab, const unsigned int* __restrict__ pend, \
    unsigned int seq_floats, unsigned int batch_size, unsigned int num_k_heads, \
    unsigned int num_v_heads, unsigned int qk_stride, unsigned int v_stride, \
    unsigned int gb_stride, unsigned int k_dim, unsigned int* __restrict__ engaged_flag) { \
    gdn_carry_verify<K, LAZY>(h_table, query, key, value, gate, beta, output, carry_base, slot_tab, \
        pend, seq_floats, batch_size, num_k_heads, num_v_heads, qk_stride, v_stride, \
        gb_stride, k_dim, engaged_flag); \
}
CARRY_ENTRY(gdn_carry_wy2, 2, false)
CARRY_ENTRY(gdn_carry_wy3, 3, false)
CARRY_ENTRY(gdn_carry_wy4, 4, false)
CARRY_ENTRY(gdn_carry_wy2_lazy, 2, true)
CARRY_ENTRY(gdn_carry_wy3_lazy, 3, true)
CARRY_ENTRY(gdn_carry_wy4_lazy, 4, true)

// 2026-09-26: Apply pending rows without a verify and write H. Grid (num_v_heads, batch,
// layers): layer l reads its h pointers at h_table + l * table_layer_entries, its stash at
// carry_base + l * carry_layer_floats and its counts at pend + l * pend_layer_entries. A
// count of 0 leaves H alone.
extern "C" __global__ void gdn_carry_flush(
    float* const* __restrict__ h_table,
    unsigned long long table_layer_entries,
    const float* __restrict__ carry_base,
    unsigned long long carry_layer_floats,
    const unsigned int* __restrict__ slot_tab,
    const unsigned int* __restrict__ pend,
    unsigned int pend_layer_entries,
    unsigned int seq_floats,
    unsigned int batch_size,
    unsigned int num_v_heads
) {
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

// 2026-09-26: Carried-state twin of gdn_verify_fused_conv_kn_batched (gdn_verify_fused_conv_kn.cu).
// Thread ch owns channel ch of sequence blockIdx.y. It shifts the `pend[slot]` pending input
// rows into its window, which is then what the parent kernels would have committed, and
// writes the window back under the rule of the verify kernel it runs with (always, or with
// `lazy` when pend + num_tokens > CARRY_CAP).
// Positions 0..num_tokens-1 then run the parent's arithmetic in the parent's order; no
// snapshot and no final window is written, and the position inputs are stashed (as the BF16
// the window converts) behind the pending rows, or from row 0 after a write-back, at
// conv_stash + slot * stash_seq_elems, row t at t * dim.
// Grid (ceil(dim / 256), batch), block 256; the parent's contract on d_conv, head_dim and
// qk_channels applies.
extern "C" __global__ void gdn_carry_conv(
    float* __restrict__ conv_state,
    const __nv_bfloat16* __restrict__ new_input,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    __nv_bfloat16* __restrict__ conv_stash,
    const unsigned int* __restrict__ slot_tab,
    const unsigned int* __restrict__ pend,
    unsigned int stash_seq_elems,
    unsigned int num_tokens,
    unsigned int dim,
    unsigned int d_conv,
    unsigned int qk_channels,
    unsigned int head_dim,
    unsigned int input_stride,
    unsigned int output_stride,
    float l2_eps,
    unsigned int conv_state_seq_stride,
    unsigned int input_seq_stride,
    unsigned int output_seq_stride,
    unsigned int lazy
) {
    const unsigned int seq = blockIdx.y;
    conv_state += (size_t) seq * conv_state_seq_stride;
    new_input  += (size_t) seq * input_seq_stride;
    output     += (size_t) seq * output_seq_stride;
    const unsigned int slot = slot_tab[seq];
    __nv_bfloat16* stash = conv_stash + (size_t) slot * stash_seq_elems;
    const unsigned int np = pend[slot];
    const bool wb = !lazy || np + num_tokens > CARRY_CAP;
    const unsigned int base = wb ? 0u : np;

    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int block_start = blockIdx.x * blockDim.x;
    const bool block_needs_l2 = (block_start < qk_channels);
    const bool valid = (ch < dim);

    float win[8];
    if (valid) {
        const float* state = conv_state + ch * d_conv;
        for (unsigned int i = 0; i < d_conv; i++) win[i] = state[i];
        for (unsigned int t = 0; t < np; t++) {
            for (unsigned int i = 0; i < d_conv - 1; i++) win[i] = win[i + 1];
            win[d_conv - 1] = (float)stash[t * dim + ch];
        }
        if (wb && np > 0) {
            float* out_state = conv_state + ch * d_conv;
            for (unsigned int i = 0; i < d_conv; i++) out_state[i] = win[i];
        }
    }

    const __nv_bfloat16* w = valid ? (weight + ch * d_conv) : nullptr;
    float wcoef[8];
    if (valid) {
        for (unsigned int k = 0; k < d_conv; k++) wcoef[k] = (float)w[k];
    }

    __shared__ float warp_sums[8];

    for (unsigned int t = 0; t < num_tokens; t++) {
        float silu = 0.0f;
        if (valid) {
            const __nv_bfloat16 x = new_input[t * input_stride + ch];
            for (unsigned int i = 0; i < d_conv - 1; i++) win[i] = win[i + 1];
            win[d_conv - 1] = (float)x;
            stash[(base + t) * dim + ch] = x;

            float acc = 0.0f;
            for (unsigned int k = 0; k < d_conv; k++) acc += win[k] * wcoef[k];
            float sigmoid_acc = 1.0f / (1.0f + __expf(-acc));
            silu = acc * sigmoid_acc;
        }

        if (block_needs_l2) {
            float sq = valid ? (silu * silu) : 0.0f;
            const unsigned int warp_id = tid / 32;
            const unsigned int lane = tid % 32;
            for (int offset = 16; offset >= 1; offset >>= 1)
                sq += __shfl_down_sync(0xFFFFFFFF, sq, offset);
            if (lane == 0) warp_sums[warp_id] = sq;
            __syncthreads();
            const unsigned int head_in_block = tid / head_dim;
            const unsigned int base_warp = head_in_block * (head_dim / 32);
            if (tid == 0 || tid == head_dim) {
                float total = warp_sums[base_warp] + warp_sums[base_warp + 1]
                            + warp_sums[base_warp + 2] + warp_sums[base_warp + 3];
                warp_sums[base_warp] = rsqrtf(total + l2_eps);
            }
            __syncthreads();
            if (valid) silu *= warp_sums[base_warp];
            // 2026-09-26: Keeps the next position's lane-0 write to warp_sums behind this read.
            __syncthreads();
        }

        if (valid) output[t * output_stride + ch] = __float2bfloat16(silu);
    }
}

// 2026-09-26: Shift pending input rows into the conv windows without a verify. Grid
// (ceil(dim / 256), batch, layers), block 256: layer l reads its conv-state pointers at
// state_table + l * table_layer_entries, its stash at conv_stash + l * stash_layer_elems
// and its counts at pend + l * pend_layer_entries. A count of 0 leaves the window alone.
extern "C" __global__ void gdn_carry_conv_flush(
    float* const* __restrict__ state_table,
    unsigned long long table_layer_entries,
    const __nv_bfloat16* __restrict__ conv_stash,
    unsigned long long stash_layer_elems,
    const unsigned int* __restrict__ slot_tab,
    const unsigned int* __restrict__ pend,
    unsigned int pend_layer_entries,
    unsigned int stash_seq_elems,
    unsigned int batch_size,
    unsigned int dim,
    unsigned int d_conv
) {
    const unsigned int b = blockIdx.y;
    const unsigned int l = blockIdx.z;
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size || ch >= dim) return;
    const unsigned int slot = slot_tab[b];
    const unsigned int np = pend[(unsigned long long)l * pend_layer_entries + slot];
    if (np == 0) return;
    float* state = state_table[l * table_layer_entries + b] + ch * d_conv;
    const __nv_bfloat16* stash =
        conv_stash + l * stash_layer_elems + (unsigned long long)slot * stash_seq_elems;
    float win[8];
    for (unsigned int i = 0; i < d_conv; i++) win[i] = state[i];
    for (unsigned int t = 0; t < np; t++) {
        for (unsigned int i = 0; i < d_conv - 1; i++) win[i] = win[i + 1];
        win[d_conv - 1] = (float)stash[t * dim + ch];
    }
    for (unsigned int i = 0; i < d_conv; i++) state[i] = win[i];
}
