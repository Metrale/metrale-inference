// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: GLM-5.3-Flash DSA (DeepSeek Sparse Attention) kernels: the kpool indexer and
// the NoPE MLA oracle.
//
// Owner: gb10 kernels.
// Invariants:
// - Pipeline order: dsa_kpool_compress (pool keys, token ids, validity) -> dsa_index_scores
//   (per-(query, pool) score and candidacy) -> dsa_topk_pools (deterministic top-k) ->
//   dsa_expand_selection (raw token ids, tail appended, -1 padded). dsa_topk_to_mask and
//   dsa_mla_masked_attn are the oracle's dense path; dsa_compact_pools gathers kept pools.
// - -1 (DSA_INVALID) marks an invalid index. dsa_expand_selection writes -1 over its whole
//   output row and synchronises before it stores any real index, so every slot of the row
//   is written on every path, including a masked query.
// - dsa_mla_masked_attn has no rope section: q and k share one head dim qd, and `scale` is
//   an argument, never derived here.

#include "dsa_indexer_body.cuh"

__device__ __forceinline__ float dsa_block_sum(float v, float* smem, unsigned tid, unsigned nthreads) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    if ((tid & 31u) == 0u) smem[tid >> 5] = v;
    __syncthreads();
    if (tid < 32u) {
        float x = (tid < ((nthreads + 31u) / 32u)) ? smem[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) x += __shfl_down_sync(0xffffffff, x, off);
        if (tid == 0u) smem[0] = x;
    }
    __syncthreads();
    return smem[0];
}

// 2026-09-25: 1. kpool compression. One block per pool; threads stride over channels. The
// softmax runs over the pool-slot axis, independently per channel. Pool p covers tokens
// first_key + p*KP + s, so left padding before first_key is skipped. A pool is valid only
// if every slot is in range and valid, so a trailing partial pool is never valid. KP must
// be <= 8 (lg[8]); config validation refuses a larger index_kpool (KERNEL_MAX_KPOOL).
// 2026-10-09: The geometry slots, dsa_write_geom's body and the per-channel and per-slot
// halves of the compression are in dsa_indexer_body.cuh, shared with dsa_indexer_rows.cu.

// 2026-09-25: One thread. Fills the geom slots from seq_len[0].
extern "C" __global__ void dsa_write_geom(
    const int* __restrict__ seq_len,
    int* __restrict__ geom,
    unsigned int KP,
    unsigned int topk,
    unsigned int tile
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    dsa_write_geom_body(seq_len, geom, KP, topk, tile);
}

// 2026-09-25: Copies one staged indexer row (k and gate) into the cache at the device-side
// row pos[0] and marks it valid, so a replayed graph writes the live row.

extern "C" __global__ void dsa_indexer_store(
    const __nv_bfloat16* __restrict__ stage_k,
    const __nv_bfloat16* __restrict__ stage_gate,
    const int* __restrict__ pos,
    __nv_bfloat16* __restrict__ k_normed,
    __nv_bfloat16* __restrict__ gate,
    unsigned char* __restrict__ valid,
    unsigned int D,
    const int* __restrict__ bt,
    unsigned int bs,
    unsigned int blk_elems
) {
    dsa_indexer_store_body(stage_k, stage_gate, pos, k_normed, gate, valid, D, bt, bs,
                           blk_elems);
}

extern "C" __global__ void dsa_kpool_compress(
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ gate,
    const unsigned char* __restrict__ valid,
    const float* __restrict__ ape,
    float* __restrict__ pool_keys,
    int* __restrict__ pool_indices,
    unsigned char* __restrict__ pool_valid,
    unsigned int S,
    unsigned int D,
    unsigned int KP,
    int first_key,
    const int* __restrict__ geom,
    const int* __restrict__ bt,
    unsigned int bs,
    unsigned int blk_elems
) {
    const unsigned int p = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    if (geom) {
        S = (unsigned int)geom[DSA_GEOM_S];
        // 2026-09-25: Under a ceiling launch this pool may not be live yet.
        if (p >= (unsigned int)geom[DSA_GEOM_NPOOLS_F]) return;
    }

    // 2026-09-25: Slot bookkeeping is the same for every channel, so thread 0 writes it.
    bool all_valid = true;
    for (unsigned int s = 0; s < KP; ++s) {
        const int idx = dsa_pool_slot(first_key, p, KP, s, S, valid);
        all_valid &= idx != DSA_INVALID;
        if (tid == 0) pool_indices[p * KP + s] = idx;
    }
    if (tid == 0) pool_valid[p] = all_valid ? 1 : 0;

    for (unsigned int d = tid; d < D; d += blockDim.x)
        pool_keys[p * D + d] =
            dsa_pool_key(k, gate, valid, ape, S, D, KP, first_key, bt, bs, blk_elems, p, d);
}

// 2026-09-25: 2. Per-(query, pool) index score, one block per (pool, query). `weights` must
// already carry the index_heads^-0.5 factor; the kernel applies only `scale` (the host
// passes index_head_dim^-0.5). score = sum_h weights[h] * relu(scale * dot(q_h, key)).

extern "C" __global__ void dsa_index_scores(
    const float* __restrict__ q,
    const float* __restrict__ pool_keys,
    const float* __restrict__ weights,
    const int* __restrict__ pool_indices,
    const unsigned char* __restrict__ pool_valid,
    const unsigned char* __restrict__ valid_keys,
    const int* __restrict__ q_pos,
    float* __restrict__ out,
    unsigned char* __restrict__ valid_cand,
    unsigned int Q,
    unsigned int P,
    unsigned int H,
    unsigned int D,
    unsigned int KP,
    unsigned int S,
    float scale,
    const int* __restrict__ geom
) {
    const unsigned int p = blockIdx.x;
    const unsigned int r = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    extern __shared__ float sh[];
    if (geom) {
        S = (unsigned int)geom[DSA_GEOM_S];
        P = (unsigned int)geom[DSA_GEOM_NPOOLS];
        // 2026-09-25: `P` is also the row stride of out / valid_cand, so it may vary with geom
        // only because the device-geometry path is decode-only (Q == 1, every r * P is 0).
        // The launcher refuses a ceiling launch with Q > 1.
        if (p >= P) return;
    }

    // 2026-09-25: A pool is a candidate only when it is complete and its last token, clamped
    // to [0, S-1], is at or before this query's position and valid. Others score -FLT_MAX.
    int end = pool_indices[p * KP + KP - 1];
    int end_c = end < 0 ? 0 : (end >= (int)S ? (int)S - 1 : end);
    bool vis = (end_c <= q_pos[r]) && dsa_row_valid(valid_keys, end_c);
    bool cand = (pool_valid[p] != 0) && vis;
    if (tid == 0) valid_cand[(size_t)r * P + p] = cand ? 1 : 0;
    if (!cand) {
        if (tid == 0) out[(size_t)r * P + p] = -FLT_MAX;
        return;
    }

    const float acc = dsa_score_heads(q + (size_t)r * H * D, pool_keys + (size_t)p * D,
                                      weights + (size_t)r * H, H, D, scale, sh, tid);
    if (tid == 0) out[(size_t)r * P + p] = acc;
}

// 2026-09-25: 3. Deterministic top-k over pools, one block per query. A tiled bitonic
// select: the pool axis is walked in tiles of NP2 and a running best-NP2 list is kept in
// shared memory (two tiles of [f32, i32], 16 * NP2 bytes, whatever the context). The order
// is score descending, then pool index ascending. That comparator is a total order over
// unique indices, so the top-select_k prefix is unique and the tiled result equals a
// whole-axis sort. select_k must be <= NP2; DsaSelectGeometry::plan refuses more. Slots
// never filled (index INT_MAX) are written as -1.
extern "C" __global__ void dsa_topk_pools(
    const float* __restrict__ scores,
    int* __restrict__ selected,
    unsigned int Q,
    unsigned int P,
    unsigned int NP2,
    unsigned int select_k,
    const int* __restrict__ geom
) {
    dsa_topk_pools_body(scores, selected, P, NP2, select_k, geom, blockIdx.x);
}

// 2026-09-25: 4. Expand the selected pools into raw token ids, one block per query, into a
// row of `width` slots (index_topk, plus index_kpool - 1 when always_tail). The row is
// filled with -1 and synchronised before any real index is written, so short rows, invalid
// pools and a missing tail all leave the sentinel.
extern "C" __global__ void dsa_expand_selection(
    const int* __restrict__ selected,
    const int* __restrict__ pool_indices,
    const unsigned char* __restrict__ valid_cand,
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
    int always_tail,
    const int* __restrict__ geom
) {
    dsa_expand_selection_body<false>(selected, pool_indices, valid_cand, valid_keys, q_pos,
                                     q_mask, out, P, KP, S, select_k, width, first_key,
                                     always_tail, geom, blockIdx.x);
}

// 2026-09-25: 5. Index row -> visibility mask (oracle). Duplicates collapse, so a repeated
// token is attended once; out-of-range and -1 entries are dropped.

extern "C" __global__ void dsa_topk_to_mask(
    const int* __restrict__ topk,
    unsigned char* __restrict__ mask,
    unsigned int Q,
    unsigned int width,
    unsigned int S
) {
    const unsigned int r = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    unsigned char* row = mask + (size_t)r * S;
    for (unsigned int i = tid; i < S; i += blockDim.x) row[i] = 0;
    __syncthreads();
    for (unsigned int j = tid; j < width; j += blockDim.x) {
        int i = topk[(size_t)r * width + j];
        if (i >= 0 && i < (int)S) row[i] = 1;
    }
}

// 2026-09-25: 6. NoPE MLA over the selected tokens (oracle), one block per (query, head).
// Scores are parallel over keys and the value sum over dims, with the score row staged in
// shared memory in between, so there is no barrier per key. The score row is S floats of
// dynamic shared memory: at 49,152 B that caps S at 12,288 keys (MASKED_ATTN_MAX_KEYS in
// glm5next_dsa/mod.rs).









extern "C" __global__ void dsa_mla_masked_attn(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const unsigned char* __restrict__ mask,
    float* __restrict__ out,
    unsigned int Q,
    unsigned int S,
    unsigned int H,
    unsigned int qd,
    unsigned int vd,
    float scale,
    // 2026-09-25: Nonzero rounds each scaled score to BF16 before the softmax.



    unsigned int round_scores_bf16
) {
    extern __shared__ float sc[];
    __shared__ float red[32];
    __shared__ float s_m, s_l;
    const unsigned int r = blockIdx.x;
    const unsigned int h = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const __nv_bfloat16* qrow = q + ((size_t)r * H + h) * qd;
    const unsigned char* mrow = mask + (size_t)r * S;


    float local_max = -CUDART_INF_F;
    for (unsigned int t = tid; t < S; t += blockDim.x) {
        if (mrow[t] == 0) { sc[t] = -CUDART_INF_F; continue; }
        float dot = 0.0f;
        const __nv_bfloat16* krow = k + ((size_t)t * H + h) * qd;
        for (unsigned int d = 0; d < qd; ++d)
            dot += __bfloat162float(qrow[d]) * __bfloat162float(krow[d]);
        float sv = dot * scale;
        if (round_scores_bf16) sv = __bfloat162float(__float2bfloat16(sv));
        sc[t] = sv;
        local_max = fmaxf(local_max, sv);
    }
    for (int off = 16; off > 0; off >>= 1)
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, off));
    if ((tid & 31u) == 0u) red[tid >> 5] = local_max;
    __syncthreads();
    if (tid < 32u) {
        float x = (tid < ((blockDim.x + 31u) / 32u)) ? red[tid] : -CUDART_INF_F;
        for (int off = 16; off > 0; off >>= 1) x = fmaxf(x, __shfl_down_sync(0xffffffff, x, off));
        if (tid == 0u) s_m = x;
    }
    __syncthreads();


    const float m = s_m;
    float local_sum = 0.0f;
    for (unsigned int t = tid; t < S; t += blockDim.x) {
        float e = (sc[t] == -CUDART_INF_F) ? 0.0f : __expf(sc[t] - m);
        sc[t] = e;
        local_sum += e;
    }
    local_sum = dsa_block_sum(local_sum, red, tid, blockDim.x);
    if (tid == 0) s_l = local_sum;
    __syncthreads();


    const float inv = (s_l > 0.0f) ? (1.0f / s_l) : 0.0f;
    for (unsigned int d = tid; d < vd; d += blockDim.x) {
        float acc = 0.0f;
        for (unsigned int t = 0; t < S; ++t) {
            float p = sc[t];
            if (p != 0.0f) acc += p * __bfloat162float(v[((size_t)t * H + h) * vd + d]);
        }
        out[((size_t)r * H + h) * vd + d] = acc * inv;
    }
}

// 2026-09-25: 1b. Pool-axis compaction: gathers the pools listed in `keep` (original pool
// ids) into dense arrays; the caller computes `keep`. select_tokens does not launch it:
// over a contiguous cache the kept pools are the prefix 0 .. S / KP.






extern "C" __global__ void dsa_compact_pools(
    const float* __restrict__ keys_in,
    const int* __restrict__ idx_in,
    const unsigned char* __restrict__ valid_in,
    const int* __restrict__ keep,
    float* __restrict__ keys_out,
    int* __restrict__ idx_out,
    unsigned char* __restrict__ valid_out,
    unsigned int P_kept,
    unsigned int D,
    unsigned int KP
) {
    const unsigned int p = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const int src = keep[p];
    for (unsigned int d = tid; d < D; d += blockDim.x)
        keys_out[(size_t)p * D + d] = keys_in[(size_t)src * D + d];
    for (unsigned int s = tid; s < KP; s += blockDim.x)
        idx_out[(size_t)p * KP + s] = idx_in[(size_t)src * KP + s];
    if (tid == 0) valid_out[p] = valid_in[src];
}
