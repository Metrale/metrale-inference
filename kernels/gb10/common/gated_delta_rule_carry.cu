// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-26: Carried-state GDN verify for K = 2, 3, 4 rows per sequence
// (gdn_carry_wy2/3/4) and its standalone fold (gdn_carry_flush).
//
// Owner: gb10 kernels.
// Invariants:
// - A verify kernel first applies the `pend[slot]` accepted rows of the previous verify
//   from the carry stash to H, with the parent kernels' update expression in row order,
//   and writes H once when there were any. H then holds what the parent kernel of that
//   verify left in Hi(pend-1), or in its final H when every row was accepted.
// - It then computes `output` with gated_delta_rule_wy{2,3,4}'s expressions and
//   accumulation order from that H, writes no h-state, and stashes what a later fold
//   needs: the K vn vectors, the K clamped gates and the K key rows, as the floats it
//   used. The key rows are stored once per v-head, so each block reads and writes only
//   its own part of the stash.
// - gdn_carry_flush applies `pend` rows the same way, reading and writing H once.
//   Neither kernel clears `pend`; the host owns it.
// gdn_carry_microtest (model-arch examples) checks output and every accepted count
// bitwise against the parent kernels.
//
// State traffic per verify: one read of H, plus one write when rows were pending; pass 2
// reads the column again, mostly from L2. The parent kernels read H once (resident) or
// twice and write K blobs, and a partial accept costs one more read and write for the
// restore copy.
//
// Stash per (layer, slot), in floats: vn[4][num_v_heads][v_dim] | g[4][num_v_heads] |
// sk[4][num_v_heads][k_dim]. k_dim == v_dim == 128 is checked on the host; the kernels
// index with the compile-time 128 and use the k_dim argument only for the output scale. Grid
// (num_v_heads, batch), block 128. The h-state is a per-sequence pointer table (slab 0 of
// the verify WY tables); slot_tab maps a batch position to its carry slot, and each
// position's engaged word (engaged_flag[b]) is set to 1.

#include <cuda_bf16.h>
#include "gdn_reduce.cuh"
#define CARRY_KD 128u
#define CARRY_VD 128u

#define CARRY_VN(S, T, VH) ((S) + ((T) * num_v_heads + (VH)) * CARRY_VD)
#define CARRY_G(S, T, VH)  ((S) + 4 * num_v_heads * CARRY_VD + (T) * num_v_heads + (VH))
#define CARRY_SK(S, T, VH) ((S) + 4 * num_v_heads * CARRY_VD + 4 * num_v_heads + ((T) * num_v_heads + (VH)) * CARRY_KD)

template <int K>
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
    if (threadIdx.x == 0 && vh == 0) engaged_flag[b] = 1u;
    const unsigned int tid = threadIdx.x;
    const unsigned int hr = num_v_heads / num_k_heads;
    const unsigned int kh = vh / hr;
    const unsigned int hv = CARRY_KD * CARRY_VD;
    float* H = h_table[b] + (unsigned long long)vh * hv;
    const unsigned int slot = slot_tab[b];
    float* S = carry_base + (unsigned long long)slot * seq_floats;
    const unsigned int np = pend[slot];

    __shared__ float sk[K][CARRY_KD], sq[K][CARRY_KD];
    __shared__ float pk[4][CARRY_KD], pg[4];
    __shared__ float smem_warp[4];
    __shared__ float kd[4][4];

    #pragma unroll
    for (int t = 0; t < K; ++t) {
        const unsigned int row = b * K + t;
        sk[t][tid] = (float)key[row * qk_stride + kh * CARRY_KD + tid];
        sq[t][tid] = (float)query[row * qk_stride + kh * CARRY_KD + tid];
    }
    // 2026-09-26: The pending rows are read before the barrier; this block overwrites the
    // same stash rows further down.
    for (unsigned int t = 0; t < np; ++t) pk[t][tid] = CARRY_SK(S, t, vh)[tid];
    if (tid < np) pg[tid] = *CARRY_G(S, tid, vh);
    const float pv0 = np > 0 ? CARRY_VN(S, 0, vh)[tid] : 0.0f;
    const float pv1 = np > 1 ? CARRY_VN(S, 1, vh)[tid] : 0.0f;
    const float pv2 = np > 2 ? CARRY_VN(S, 2, vh)[tid] : 0.0f;
    const float pv3 = np > 3 ? CARRY_VN(S, 3, vh)[tid] : 0.0f;
    __syncthreads();

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

    // 2026-09-26: Pass 1 over the column: H with the pending rows applied (g_t * h +
    // sk_t[j] * vn_t for t < np in row order, the parents' update expression), written back
    // when there were any, and summed into hk_t = H . k_t in groups of four j as the
    // parents sum them.
    float* __restrict__ Hc = H + tid;
    float hk[K];
    #pragma unroll
    for (int t = 0; t < K; ++t) hk[t] = 0.0f;
    #pragma unroll
    for (unsigned int j = 0; j < CARRY_KD; j += 4) {
        float h[4];
        #pragma unroll
        for (unsigned int e = 0; e < 4; ++e) {
            float x = Hc[(j + e) * CARRY_VD];
            if (np > 0) x = pg[0] * x + pk[0][j + e] * pv0;
            if (np > 1) x = pg[1] * x + pk[1][j + e] * pv1;
            if (np > 2) x = pg[2] * x + pk[2][j + e] * pv2;
            if (np > 3) x = pg[3] * x + pk[3][j + e] * pv3;
            if (np > 0) Hc[(j + e) * CARRY_VD] = x;
            h[e] = x;
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

    // 2026-09-26: Pass 2 reads the column again (this thread wrote it in pass 1 when rows
    // were pending; at the verify widths it is mostly still in L2): row t updates it in
    // registers, then qd_t += h_t . q_t, per group of
    // four j in the parents' order. Nothing is written back.
    float qd[K];
    #pragma unroll
    for (int t = 0; t < K; ++t) qd[t] = 0.0f;
    #pragma unroll
    for (unsigned int j = 0; j < CARRY_KD; j += 4) {
        float h0 = Hc[j * CARRY_VD], h1 = Hc[(j + 1) * CARRY_VD];
        float h2 = Hc[(j + 2) * CARRY_VD], h3 = Hc[(j + 3) * CARRY_VD];
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
        CARRY_VN(S, t, vh)[tid] = vn[t];
        CARRY_SK(S, t, vh)[tid] = sk[t][tid];
    }
    if (tid < K) *CARRY_G(S, tid, vh) = g[tid];
}

#define CARRY_ENTRY(K) \
extern "C" __global__ void __launch_bounds__(128) gdn_carry_wy##K( \
    float* const* __restrict__ h_table, const __nv_bfloat16* __restrict__ query, \
    const __nv_bfloat16* __restrict__ key, const __nv_bfloat16* __restrict__ value, \
    const float* __restrict__ gate, const float* __restrict__ beta, \
    __nv_bfloat16* __restrict__ output, float* __restrict__ carry_base, \
    const unsigned int* __restrict__ slot_tab, const unsigned int* __restrict__ pend, \
    unsigned int seq_floats, unsigned int batch_size, unsigned int num_k_heads, \
    unsigned int num_v_heads, unsigned int qk_stride, unsigned int v_stride, \
    unsigned int gb_stride, unsigned int k_dim, unsigned int* __restrict__ engaged_flag) { \
    gdn_carry_verify<K>(h_table, query, key, value, gate, beta, output, carry_base, slot_tab, \
        pend, seq_floats, batch_size, num_k_heads, num_v_heads, qk_stride, v_stride, \
        gb_stride, k_dim, engaged_flag); \
}
CARRY_ENTRY(2)
CARRY_ENTRY(3)
CARRY_ENTRY(4)

// 2026-09-26: Apply pending rows without a verify. Grid (num_v_heads, batch, layers):
// layer l reads its h pointers at h_table + l * table_layer_entries, its stash at
// carry_base + l * carry_layer_floats and its counts at pend + l * pend_layer_entries.
// A count of 0 leaves H alone.
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
    __shared__ float pk[4][CARRY_KD], pg[4];
    for (unsigned int t = 0; t < np; ++t) pk[t][tid] = CARRY_SK(S, t, vh)[tid];
    if (tid < np) pg[tid] = *CARRY_G(S, tid, vh);
    __syncthreads();
    float fvn[4];
    for (unsigned int t = 0; t < np; ++t) fvn[t] = CARRY_VN(S, t, vh)[tid];
    #pragma unroll 4
    for (unsigned int j = 0; j < CARRY_KD; ++j) {
        float h = H[j * CARRY_VD + tid];
        for (unsigned int t = 0; t < np; ++t) h = pg[t] * h + pk[t][j] * fvn[t];
        H[j * CARRY_VD + tid] = h;
    }
}
