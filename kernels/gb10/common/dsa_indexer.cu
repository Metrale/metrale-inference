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
// - 2026-10-06: dsa_topk_radix_{init,hist,find,gather,sort} (3b, METRALE_GLM_DSA_TOPK_RADIX=1)
//   write the same `selected` bytes as dsa_topk_pools, for pool counts above one top-k tile.







#include <cuda_bf16.h>
#include <math_constants.h>
#include <float.h>
#include <limits.h>

#define DSA_INVALID (-1)

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




// 2026-09-25: Replay-safe geometry. A CUDA graph fixes every scalar argument at capture time,
// but S, the pool counts, the top-k tile and select_k grow with the context. Each selector
// kernel therefore takes `geom`, a 5-int device vector that dsa_write_geom fills once per
// step from seq_len. When non-null it overrides the scalar arguments; when null the scalars
// are used as passed. Pool-indexed blocks past the live pool count return at once, so a
// ceiling launch can fix the grid at the context ceiling. Slots: S (tokens in the cache),
// pools including the trailing partial one, complete pools, select_k, top-k tile width.






#define DSA_GEOM_S        0
#define DSA_GEOM_NPOOLS_F 1
#define DSA_GEOM_NPOOLS   2
#define DSA_GEOM_SELECT_K 3
#define DSA_GEOM_NP2      4

// 2026-09-25: One thread. Fills the geom slots from seq_len[0]. `tile` is the top-k tile width
// (a power of two); np2 = min(max(2, next power of two >= complete pools), tile).
extern "C" __global__ void dsa_write_geom(
    const int* __restrict__ seq_len,
    int* __restrict__ geom,
    unsigned int KP,
    unsigned int topk,
    unsigned int tile
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const int S = seq_len[0];
    const int np = S / (int)KP;


    int np2 = 2;
    while (np2 < np && np2 < (int)tile) np2 <<= 1;
    const int cap = (int)(topk / KP);
    geom[DSA_GEOM_S] = S;
    geom[DSA_GEOM_NPOOLS_F] = (S + (int)KP - 1) / (int)KP;
    geom[DSA_GEOM_NPOOLS] = np;
    geom[DSA_GEOM_SELECT_K] = np < cap ? np : cap;
    geom[DSA_GEOM_NP2] = np2;
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
    unsigned int D
) {
    const size_t base = (size_t)pos[0] * D;
    for (unsigned int d = threadIdx.x; d < D; d += blockDim.x) {
        k_normed[base + d] = stage_k[d];
        gate[base + d] = stage_gate[d];
    }
    if (threadIdx.x == 0) valid[pos[0]] = 1;
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
    const int* __restrict__ geom
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
        long long raw = (long long)first_key + (long long)p * KP + s;
        bool in_range = raw >= 0 && raw < (long long)S;
        bool ok = in_range && valid[in_range ? (unsigned)raw : 0] != 0;
        all_valid &= ok;
        if (tid == 0) pool_indices[p * KP + s] = ok ? (int)raw : DSA_INVALID;
    }
    if (tid == 0) pool_valid[p] = all_valid ? 1 : 0;

    for (unsigned int d = tid; d < D; d += blockDim.x) {
        float mx = -CUDART_INF_F;
        float lg[8];
        for (unsigned int s = 0; s < KP && s < 8; ++s) {
            long long raw = (long long)first_key + (long long)p * KP + s;
            bool in_range = raw >= 0 && raw < (long long)S;
            bool ok = in_range && valid[in_range ? (unsigned)raw : 0] != 0;
            lg[s] = ok ? (__bfloat162float(gate[(size_t)raw * D + d]) + ape[s * D + d])
                       : -CUDART_INF_F;
            mx = fmaxf(mx, lg[s]);
        }
        float sum = 0.0f;
        for (unsigned int s = 0; s < KP && s < 8; ++s) {
            lg[s] = (lg[s] == -CUDART_INF_F) ? 0.0f : __expf(lg[s] - mx);
            sum += lg[s];
        }
        // 2026-09-25: A pool with no valid slot has sum 0, so its weights and key are 0.
        float inv = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
        float acc = 0.0f;
        for (unsigned int s = 0; s < KP && s < 8; ++s) {
            long long raw = (long long)first_key + (long long)p * KP + s;
            bool in_range = raw >= 0 && raw < (long long)S;
            bool ok = in_range && valid[in_range ? (unsigned)raw : 0] != 0;
            if (ok) acc += lg[s] * inv * __bfloat162float(k[(size_t)raw * D + d]);
        }
        pool_keys[p * D + d] = acc;
    }
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
    bool vis = (end_c <= q_pos[r]) && (valid_keys[end_c] != 0);
    bool cand = (pool_valid[p] != 0) && vis;
    if (tid == 0) valid_cand[(size_t)r * P + p] = cand ? 1 : 0;
    if (!cand) {
        if (tid == 0) out[(size_t)r * P + p] = -FLT_MAX;
        return;
    }

    // 2026-09-25: One warp per head: each lane accumulates every 32nd product, a shuffle tree
    // reduces them, and the head's term lands in sh[h]. Thread 0 then sums sh[0..H) in head
    // order. sh must hold H floats: the host requests max(SCORES_BLOCK, 4 * H) bytes.

















    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5;
    const unsigned int nwarps = (blockDim.x + 31u) / 32u;
    const float* __restrict__ pk = pool_keys + (size_t)p * D;
    for (unsigned int h = warp; h < H; h += nwarps) {
        const float* __restrict__ qh = q + ((size_t)r * H + h) * D;
        float dot = 0.0f;
        for (unsigned int d = lane; d < D; d += 32u) dot += qh[d] * pk[d];
        for (int off = 16; off > 0; off >>= 1) dot += __shfl_down_sync(0xffffffffu, dot, off);
        if (lane == 0) sh[h] = weights[(size_t)r * H + h] * fmaxf(scale * dot, 0.0f);
    }
    __syncthreads();
    if (tid == 0) {
        float acc = 0.0f;
        for (unsigned int h = 0; h < H; ++h) acc += sh[h];
        out[(size_t)r * P + p] = acc;
    }
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
    if (geom) {
        P = (unsigned int)geom[DSA_GEOM_NPOOLS];
        NP2 = (unsigned int)geom[DSA_GEOM_NP2];
        select_k = (unsigned int)geom[DSA_GEOM_SELECT_K];
    }
    // 2026-09-25: Under a ceiling launch the dynamic shared memory is sized for the ceiling;
    // the walk still runs over this step's NP2 and P.
    extern __shared__ char raw_sh[];
    const unsigned int T = NP2;
    float* sv = (float*)raw_sh;
    int*   si = (int*)(raw_sh + (size_t)(2 * T) * sizeof(float));
    const unsigned int r = blockIdx.x;
    const unsigned int tid = threadIdx.x;

    // 2026-09-25: Running best list, descending; starts as padding that sorts last on both keys.
    for (unsigned int i = tid; i < T; i += blockDim.x) {
        sv[i] = -FLT_MAX;
        si[i] = INT_MAX;
    }
    __syncthreads();

#define DSA_TOPK_GT(a, b) ((sv[(a)] > sv[(b)]) || (sv[(a)] == sv[(b)] && si[(a)] < si[(b)]))
#define DSA_TOPK_SWAP(a, b)                                                                  \
    do {                                                                                     \
        float tv_ = sv[(a)]; sv[(a)] = sv[(b)]; sv[(b)] = tv_;                               \
        int   ti_ = si[(a)]; si[(a)] = si[(b)]; si[(b)] = ti_;                               \
    } while (0)

    for (unsigned int base = 0; base < P; base += T) {
        // 2026-09-25: Candidate tile into [T, 2T); a short tile pads with the same sentinel.
        for (unsigned int i = tid; i < T; i += blockDim.x) {
            unsigned int idx = base + i;
            sv[T + i] = (idx < P) ? scores[(size_t)r * P + idx] : -FLT_MAX;
            si[T + i] = (idx < P) ? (int)idx : INT_MAX;
        }
        __syncthreads();

        // 2026-09-25: Bitonic sort of the candidate tile, descending.
        for (unsigned int k = 2; k <= T; k <<= 1) {
            for (unsigned int j = k >> 1; j > 0; j >>= 1) {
                for (unsigned int i = tid; i < T; i += blockDim.x) {
                    unsigned int l = i ^ j;
                    if (l > i) {
                        bool gt = DSA_TOPK_GT(T + i, T + l);
                        bool want_desc = ((i & k) == 0);
                        if (want_desc != gt) DSA_TOPK_SWAP(T + i, T + l);
                    }
                }
                __syncthreads();
            }
        }

        // 2026-09-25: Half-cleaner across the two descending runs: pairing best[i] with
        // cand[T-1-i] leaves the top T of both in [0, T), bitonic but not yet sorted.
        for (unsigned int i = tid; i < T; i += blockDim.x) {
            unsigned int a = i, b = T + (T - 1 - i);
            if (!DSA_TOPK_GT(a, b)) DSA_TOPK_SWAP(a, b);
        }
        __syncthreads();

        // 2026-09-25: Bitonic merge restores descending order over [0, T).
        for (unsigned int j = T >> 1; j > 0; j >>= 1) {
            for (unsigned int i = tid; i < T; i += blockDim.x) {
                unsigned int l = i ^ j;
                if (l > i && !DSA_TOPK_GT(i, l)) DSA_TOPK_SWAP(i, l);
            }
            __syncthreads();
        }
    }

#undef DSA_TOPK_GT
#undef DSA_TOPK_SWAP

    for (unsigned int i = tid; i < select_k; i += blockDim.x)
        selected[(size_t)r * select_k + i] = (si[i] == INT_MAX) ? DSA_INVALID : si[i];
}

// 2026-10-06: 3b. Exact radix top-k over pools (METRALE_GLM_DSA_TOPK_RADIX=1, opt-in): the
// same output as dsa_topk_pools, byte for byte, without its per-tile bitonic sorts. The host
// (crates/model-arch/src/glm5next_dsa/select/radix.rs) takes it only when the pool count it
// dispatches on (n_pools, or the ceiling pool count of a graph launch) exceeds topk_tile();
// below that dsa_topk_pools sorts one tile and is already cheap.
//
// Order (the contract of dsa_topk_pools): score descending, then pool index ascending, with
// float compare semantics: -0.0 ties +0.0, and -FLT_MAX (a non-candidate) is an ordinary
// value that ties with itself. dsa_radix_key maps a score to a uint32 that orders exactly so:
// -0.0 is canonicalised to +0.0 first, then a non-negative float gets its sign bit set and a
// negative one is bit-inverted. NaN is out of scope (scores are finite sums of relu terms).
//
// Sequence per call, all on one stream (9 launches; every block of a launch takes the same
// early exit when select_k is 0, so no barrier is skipped by part of a block):
//   init        (grid Q)     zero the two global histograms and the state words of each row.
//   hist pass 0 (grid C x Q) histogram of key bits 31..20 (4096 bins) over the row's chunk,
//                            warp-aggregated into shared memory, then atomicAdd into hist0.
//   find pass 0 (grid Q)     scan the bins from the top until the count reaches select_k:
//                            that bin is the threshold's top 12 bits; `need` becomes the count
//                            still wanted inside it.
//   hist/find pass 1         bits 19..8 (4096 bins, hist1) over keys matching the prefix.
//   hist/find pass 2         bits 7..0 (256 bins) over keys matching the 24-bit prefix; each
//                            chunk stores its own 256-bin histogram (no atomics), so find can
//                            read the per-chunk count of the threshold key tau. After it:
//                            tau, m = ties at tau to take (>= 1), n_gt = select_k - m =
//                            count(key > tau), and tie_base[c] = ties at tau in chunks < c.
//   gather      (grid C x Q) keys > tau go to candidate slots [0, n_gt) through an atomic
//                            counter (any order); a key == tau at global tie rank t (index
//                            order: tie_base[c] + rank within the chunk from a block scan) goes
//                            to slot n_gt + t when t < m. So exactly select_k candidates, and
//                            threshold ties are the m lowest indices, deterministically, with no
//                            tie buffer however many -FLT_MAX ties there are.
//   sort        (grid Q)     bitonic sort of the select_k [key, index] pairs, key descending
//                            then index ascending (a total order over unique indices), and the
//                            indices written to selected[r * select_k + i].
// The candidate set equals the first select_k pools of the total order, and the sort orders it
// by that order, so `selected` equals dsa_topk_pools'. The row stride of `scores` is P and that
// of `selected` is select_k, both from `geom` when it is non-null, as in dsa_topk_pools.
//
// Ceiling launch: P and select_k come from geom; the grids are fixed (C chunks per row) and the
// chunk width ceil(P / C) follows the live P, so a chunk past the live pools is empty (it still
// stores its zero pass-2 histogram and gathers nothing). init re-zeroes the global scratch in
// every call, inside the captured sequence.
//
// Work buffer per row (u32 words, row stride DSA_RADIX_CAND + 2 * kcap):
//   [0, 4096) hist0, [4096, 8192) hist1; pass 2's per-chunk histograms [c * 256, c * 256 + 256)
//   reuse [0, C * 256) (C <= 32, so at most both) once find pass 1 has read hist1.
//   [8192, 8208) state, [8208, 8240) tie_base per chunk, then kcap candidate keys and kcap
//   candidate indices.

#define DSA_RADIX_THREADS 256u
#define DSA_RADIX_MAX_CHUNKS 32u
#define DSA_RADIX_BINS 4096u
#define DSA_RADIX_BINS_LAST 256u
#define DSA_RADIX_STATE 8192u
#define DSA_RADIX_STATE_WORDS 16u
#define DSA_RADIX_TIE 8208u
#define DSA_RADIX_CAND 8240u
#define DSA_RADIX_SORT_MAX 2048u

// 2026-10-06: State word slots (offsets from DSA_RADIX_STATE).
#define DSA_RS_PREFIX 0u
#define DSA_RS_NEED   1u
#define DSA_RS_GT_CNT 2u
#define DSA_RS_NGT    3u

// 2026-10-06: Monotone key: a > b as floats (with -0.0 == +0.0) iff key(a) > key(b).
__device__ __forceinline__ unsigned int dsa_radix_key(float v) {
    unsigned int u = __float_as_uint(v);
    if (u == 0x80000000u) u = 0u;
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// 2026-10-06: Whether `key` matches the threshold prefix fixed by the passes before `pass`.
__device__ __forceinline__ bool dsa_radix_match(unsigned int key, unsigned int prefix,
                                                unsigned int pass) {
    if (pass == 0u) return true;
    if (pass == 1u) return (key >> 20) == (prefix >> 20);
    return (key >> 8) == (prefix >> 8);
}

// 2026-10-06: The bin of `key` in `pass`: bits 31..20, 19..8, then 7..0.
__device__ __forceinline__ unsigned int dsa_radix_digit(unsigned int key, unsigned int pass) {
    if (pass == 0u) return key >> 20;
    if (pass == 1u) return (key >> 8) & 0xFFFu;
    return key & 0xFFu;
}

// 2026-10-06: Exclusive block scan of v over threadIdx.x (blockDim.x a multiple of 32, at most
// 1024); `*total` gets the block sum. Every thread of the block must call it (it has barriers);
// `ws` is a 32-word shared scratch, free again when it returns.
__device__ __forceinline__ unsigned int dsa_radix_excl_scan(unsigned int v, unsigned int* ws,
                                                            unsigned int* total) {
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int nwarps = blockDim.x >> 5;
    unsigned int x = v;
    for (unsigned int off = 1; off < 32u; off <<= 1) {
        const unsigned int y = __shfl_up_sync(0xffffffffu, x, off);
        if (lane >= off) x += y;
    }
    if (lane == 31u) ws[warp] = x;
    __syncthreads();
    if (warp == 0u) {
        unsigned int s = (lane < nwarps) ? ws[lane] : 0u;
        for (unsigned int off = 1; off < 32u; off <<= 1) {
            const unsigned int y = __shfl_up_sync(0xffffffffu, s, off);
            if (lane >= off) s += y;
        }
        ws[lane] = s;
    }
    __syncthreads();
    const unsigned int base = (warp == 0u) ? 0u : ws[warp - 1u];
    *total = ws[nwarps - 1u];
    __syncthreads();
    return base + x - v;
}

// 2026-10-06: One row's work buffer.
__device__ __forceinline__ unsigned int* dsa_radix_row(unsigned int* work, unsigned int r,
                                                       unsigned int kcap) {
    return work + (size_t)r * (DSA_RADIX_CAND + 2u * kcap);
}

// 2026-10-06: Zero hist0, hist1 and the state words of row blockIdx.x.
extern "C" __global__ void __launch_bounds__(DSA_RADIX_THREADS) dsa_topk_radix_init(
    unsigned int* __restrict__ work,
    unsigned int select_k,
    unsigned int kcap,
    const int* __restrict__ geom
) {
    if (geom) select_k = (unsigned int)geom[DSA_GEOM_SELECT_K];
    if (select_k == 0u) return;
    // 2026-10-06: The host sizes kcap >= every select_k it plans; a larger one would overrun
    // the candidate buffer and the sort's shared arrays.
    if (select_k > kcap || kcap > DSA_RADIX_SORT_MAX) __trap();
    unsigned int* w = dsa_radix_row(work, blockIdx.x, kcap);
    for (unsigned int i = threadIdx.x; i < DSA_RADIX_STATE + DSA_RADIX_STATE_WORDS; i += blockDim.x)
        w[i] = 0u;
}

// 2026-10-06: Histogram pass `pass` over chunk blockIdx.x of row blockIdx.y.
extern "C" __global__ void __launch_bounds__(DSA_RADIX_THREADS) dsa_topk_radix_hist(
    const float* __restrict__ scores,
    unsigned int* __restrict__ work,
    unsigned int P,
    unsigned int select_k,
    unsigned int kcap,
    unsigned int pass,
    const int* __restrict__ geom
) {
    if (geom) {
        P = (unsigned int)geom[DSA_GEOM_NPOOLS];
        select_k = (unsigned int)geom[DSA_GEOM_SELECT_K];
    }
    if (select_k == 0u) return;
    if (gridDim.x > DSA_RADIX_MAX_CHUNKS) __trap();
    __shared__ unsigned int sh[DSA_RADIX_BINS];
    const unsigned int tid = threadIdx.x;
    const unsigned int c = blockIdx.x;
    const unsigned int r = blockIdx.y;
    unsigned int* w = dsa_radix_row(work, r, kcap);
    const unsigned int nbins = (pass == 2u) ? DSA_RADIX_BINS_LAST : DSA_RADIX_BINS;
    for (unsigned int b = tid; b < nbins; b += blockDim.x) sh[b] = 0u;
    __syncthreads();

    const unsigned int prefix = (pass == 0u) ? 0u : w[DSA_RADIX_STATE + DSA_RS_PREFIX];
    const unsigned int chunk = (P + gridDim.x - 1u) / gridDim.x;
    const unsigned int lo = min(c * chunk, P);
    const unsigned int hi = min(lo + chunk, P);
    const float* row = scores + (size_t)r * P;
    const unsigned int lane = tid & 31u;
    // 2026-10-06: The trip count depends on lo, hi and blockDim only, so every lane of every
    // warp reaches the ballot; lanes past `hi` vote false.
    for (unsigned int base = lo; base < hi; base += blockDim.x) {
        const unsigned int i = base + tid;
        bool live = i < hi;
        unsigned int digit = 0u;
        if (live) {
            const unsigned int key = dsa_radix_key(row[i]);
            live = dsa_radix_match(key, prefix, pass);
            digit = dsa_radix_digit(key, pass);
        }
        const unsigned int active = __ballot_sync(0xffffffffu, live);
        if (live) {
            // 2026-10-06: One shared atomic per distinct bin in the warp (thousands of
            // -FLT_MAX ties fall in one bin).
            const unsigned int peers = __match_any_sync(active, digit);
            if (lane == (unsigned int)(__ffs(peers) - 1))
                atomicAdd(&sh[digit], (unsigned int)__popc(peers));
        }
    }
    __syncthreads();

    if (pass < 2u) {
        unsigned int* g = w + pass * DSA_RADIX_BINS;
        for (unsigned int b = tid; b < DSA_RADIX_BINS; b += blockDim.x)
            if (sh[b] != 0u) atomicAdd(&g[b], sh[b]);
    } else {
        // 2026-10-06: Every chunk, empty or not, stores all its bins.
        unsigned int* g = w + c * DSA_RADIX_BINS_LAST;
        for (unsigned int b = tid; b < DSA_RADIX_BINS_LAST; b += blockDim.x) g[b] = sh[b];
    }
}

// 2026-10-06: Count of bin `b` in `pass`: the global histogram, or (pass 2) the per-chunk
// histograms summed over the `chunks` chunks.
__device__ __forceinline__ unsigned int dsa_radix_bin_count(const unsigned int* w,
                                                            unsigned int pass, unsigned int b,
                                                            unsigned int chunks) {
    if (pass < 2u) return w[pass * DSA_RADIX_BINS + b];
    unsigned int s = 0u;
    for (unsigned int c = 0; c < chunks; ++c) s += w[c * DSA_RADIX_BINS_LAST + b];
    return s;
}

// 2026-10-06: Find pass `pass` for row blockIdx.x: the bin where the count from the top reaches
// `need`. Thread t owns bins [nbins - (t + 1) * per, nbins - t * per), walked from the top, so
// the exclusive scan over t is the count above its bins. Exactly one thread has
// excl < need <= excl + local (need >= 1 and the bins hold at least `need` keys); it extends the
// prefix and, after pass 2, writes n_gt and the per-chunk tie bases. `chunks` is the grid x of
// the hist launches.
extern "C" __global__ void __launch_bounds__(DSA_RADIX_THREADS) dsa_topk_radix_find(
    unsigned int* __restrict__ work,
    unsigned int select_k,
    unsigned int kcap,
    unsigned int pass,
    unsigned int chunks,
    const int* __restrict__ geom
) {
    if (geom) select_k = (unsigned int)geom[DSA_GEOM_SELECT_K];
    if (select_k == 0u) return;
    // 2026-10-06: The bin split assumes DSA_RADIX_THREADS threads (16 bins, then 1, each).
    if (blockDim.x != DSA_RADIX_THREADS || chunks > DSA_RADIX_MAX_CHUNKS) __trap();
    __shared__ unsigned int ws[32];
    const unsigned int tid = threadIdx.x;
    unsigned int* w = dsa_radix_row(work, blockIdx.x, kcap);
    const unsigned int nbins = (pass == 2u) ? DSA_RADIX_BINS_LAST : DSA_RADIX_BINS;
    const unsigned int per = nbins / blockDim.x;
    const unsigned int need = (pass == 0u) ? select_k : w[DSA_RADIX_STATE + DSA_RS_NEED];
    const unsigned int prefix = (pass == 0u) ? 0u : w[DSA_RADIX_STATE + DSA_RS_PREFIX];
    const unsigned int top = nbins - 1u - tid * per;
    unsigned int local = 0u;
    for (unsigned int j = 0; j < per; ++j) local += dsa_radix_bin_count(w, pass, top - j, chunks);
    unsigned int total;
    const unsigned int excl = dsa_radix_excl_scan(local, ws, &total);
    // 2026-10-06: The scan's barriers order every read of the state above before the write
    // below.
    if (excl < need && need <= excl + local) {
        unsigned int cum = excl;
        unsigned int b = top;
        for (unsigned int j = 0; j < per; ++j) {
            b = top - j;
            const unsigned int h = dsa_radix_bin_count(w, pass, b, chunks);
            if (cum + h >= need) break;
            cum += h;
        }
        const unsigned int shift = (pass == 0u) ? 20u : ((pass == 1u) ? 8u : 0u);
        const unsigned int left = need - cum;
        w[DSA_RADIX_STATE + DSA_RS_PREFIX] = prefix | (b << shift);
        w[DSA_RADIX_STATE + DSA_RS_NEED] = left;
        if (pass == 2u) {
            w[DSA_RADIX_STATE + DSA_RS_NGT] = select_k - left;
            unsigned int run = 0u;
            for (unsigned int c = 0; c < chunks; ++c) {
                w[DSA_RADIX_TIE + c] = run;
                run += w[c * DSA_RADIX_BINS_LAST + b];
            }
        }
    }
}

// 2026-10-06: Gather the select_k candidates of chunk blockIdx.x of row blockIdx.y (see 3b).
extern "C" __global__ void __launch_bounds__(DSA_RADIX_THREADS) dsa_topk_radix_gather(
    const float* __restrict__ scores,
    unsigned int* __restrict__ work,
    unsigned int P,
    unsigned int select_k,
    unsigned int kcap,
    const int* __restrict__ geom
) {
    if (geom) {
        P = (unsigned int)geom[DSA_GEOM_NPOOLS];
        select_k = (unsigned int)geom[DSA_GEOM_SELECT_K];
    }
    if (select_k == 0u) return;
    __shared__ unsigned int ws[32];
    const unsigned int tid = threadIdx.x;
    const unsigned int c = blockIdx.x;
    const unsigned int r = blockIdx.y;
    unsigned int* w = dsa_radix_row(work, r, kcap);
    const unsigned int tau = w[DSA_RADIX_STATE + DSA_RS_PREFIX];
    const unsigned int m = w[DSA_RADIX_STATE + DSA_RS_NEED];
    const unsigned int ngt = w[DSA_RADIX_STATE + DSA_RS_NGT];
    unsigned int* cand_key = w + DSA_RADIX_CAND;
    unsigned int* cand_idx = w + DSA_RADIX_CAND + kcap;

    const unsigned int chunk = (P + gridDim.x - 1u) / gridDim.x;
    const unsigned int lo = min(c * chunk, P);
    const unsigned int hi = min(lo + chunk, P);
    const float* row = scores + (size_t)r * P;
    // 2026-10-06: Ties at tau in chunks before this one; the loop adds this chunk's in index
    // order.
    unsigned int run = w[DSA_RADIX_TIE + c];
    for (unsigned int base = lo; base < hi; base += blockDim.x) {
        const unsigned int i = base + tid;
        const bool live = i < hi;
        const unsigned int key = live ? dsa_radix_key(row[i]) : 0u;
        if (live && key > tau) {
            const unsigned int slot = atomicAdd(&w[DSA_RADIX_STATE + DSA_RS_GT_CNT], 1u);
            if (slot < ngt) {
                cand_key[slot] = key;
                cand_idx[slot] = i;
            }
        }
        const unsigned int tie = (live && key == tau) ? 1u : 0u;
        unsigned int total;
        const unsigned int rank = run + dsa_radix_excl_scan(tie, ws, &total);
        if (tie != 0u && rank < m) {
            cand_key[ngt + rank] = key;
            cand_idx[ngt + rank] = i;
        }
        run += total;
    }
}

// 2026-10-06: Sort row blockIdx.x's select_k candidates (key descending, then index ascending)
// and write their pool indices.
extern "C" __global__ void __launch_bounds__(DSA_RADIX_THREADS) dsa_topk_radix_sort(
    const unsigned int* __restrict__ work,
    int* __restrict__ selected,
    unsigned int select_k,
    unsigned int kcap,
    const int* __restrict__ geom
) {
    if (geom) select_k = (unsigned int)geom[DSA_GEOM_SELECT_K];
    if (select_k == 0u) return;
    __shared__ unsigned int sk[DSA_RADIX_SORT_MAX];
    __shared__ int si[DSA_RADIX_SORT_MAX];
    const unsigned int tid = threadIdx.x;
    const unsigned int r = blockIdx.x;
    const unsigned int* w = work + (size_t)r * (DSA_RADIX_CAND + 2u * kcap);
    const unsigned int* cand_key = w + DSA_RADIX_CAND;
    const unsigned int* cand_idx = w + DSA_RADIX_CAND + kcap;
    unsigned int n2 = 2u;
    while (n2 < select_k) n2 <<= 1;
    // 2026-10-06: Padding sorts last on both keys (key 0 is below every finite score's key).
    for (unsigned int i = tid; i < n2; i += blockDim.x) {
        sk[i] = (i < select_k) ? cand_key[i] : 0u;
        si[i] = (i < select_k) ? (int)cand_idx[i] : INT_MAX;
    }
    __syncthreads();

#define DSA_RADIX_GT(a, b) ((sk[(a)] > sk[(b)]) || (sk[(a)] == sk[(b)] && si[(a)] < si[(b)]))
    for (unsigned int k = 2; k <= n2; k <<= 1) {
        for (unsigned int j = k >> 1; j > 0; j >>= 1) {
            for (unsigned int i = tid; i < n2; i += blockDim.x) {
                const unsigned int l = i ^ j;
                if (l > i) {
                    const bool gt = DSA_RADIX_GT(i, l);
                    const bool want_desc = ((i & k) == 0);
                    if (want_desc != gt) {
                        const unsigned int tk = sk[i]; sk[i] = sk[l]; sk[l] = tk;
                        const int ti = si[i]; si[i] = si[l]; si[l] = ti;
                    }
                }
            }
            __syncthreads();
        }
    }
#undef DSA_RADIX_GT

    for (unsigned int i = tid; i < select_k; i += blockDim.x)
        selected[(size_t)r * select_k + i] = si[i];
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
    const unsigned int r = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    if (geom) {
        S = (unsigned int)geom[DSA_GEOM_S];
        P = (unsigned int)geom[DSA_GEOM_NPOOLS];
        select_k = (unsigned int)geom[DSA_GEOM_SELECT_K];
    }
    int* row = out + (size_t)r * width;

    for (unsigned int i = tid; i < width; i += blockDim.x) row[i] = DSA_INVALID;
    __syncthreads();
    if (q_mask[r] == 0) return;   // 2026-09-25: a masked query selects nothing; the row stays all -1

    // 2026-09-25: Per-row select_k. The scalar select_k is the pass's, planned from the pass's
    // cache length, and a multi-row pass plans it from the group's final length. Clamping it
    // to this row's own pool count (q_pos[r] + 1) / KP gives the row the select_k, and so the
    // tail base below, that a single-row pass at that row's length plans. `selected` keeps
    // the pass stride; only the count is per row. The tail slot matters because
    // glm5next_dsa_mla_decode_fp8 splits the row into NUM_WARPS = 8 slices merged across
    // warps. Measured 2026-09-06 on that kernel without the clamp: 14 of 18 configurations
    // where a tail token crossed a slice boundary differed, by up to 2 BF16 ulp.












    const unsigned int row_pools = (unsigned int)(q_pos[r] + 1) / KP;
    const unsigned int row_select_k = (row_pools < select_k) ? row_pools : select_k;

    for (unsigned int j = tid; j < row_select_k; j += blockDim.x) {
        int p = selected[(size_t)r * select_k + j];
        bool ok = (p >= 0) && (valid_cand[(size_t)r * P + p] != 0);
        for (unsigned int s = 0; s < KP; ++s) {
            unsigned int w = j * KP + s;
            if (w < width) row[w] = ok ? pool_indices[(size_t)p * KP + s] : DSA_INVALID;
        }
    }

    if (always_tail) {
        // 2026-09-25: The in-progress pool as raw indices. vis_count, the number of valid keys
        // at or before q_pos[r], is an integer count split across all threads, so it is exact.
        // It is not taken as q_pos[r] + 1 - first_key, because valid_keys may mark keys at or
        // below q_pos[r] invalid. always_tail is kernel-uniform and r is blockIdx.x, so every
        // thread reaches the barrier below; the q_mask return above is whole-block as well.













        __shared__ int vis_warp[32];
        const int qp = q_pos[r];
        int local = 0;
        for (unsigned int t = tid; t < S; t += blockDim.x)
            if ((int)t <= qp && valid_keys[t] != 0) ++local;
        for (int off = 16; off > 0; off >>= 1)
            local += __shfl_down_sync(0xffffffffu, local, off);
        const unsigned int lane = tid & 31u;
        const unsigned int warp = tid >> 5;
        if (lane == 0) vis_warp[warp] = local;
        __syncthreads();
        if (tid != 0) return;
        const unsigned int nwarps = (blockDim.x + 31u) / 32u;
        int vis_count = 0;
        for (unsigned int w = 0; w < nwarps; ++w) vis_count += vis_warp[w];

        int tail_count = vis_count % (int)KP;
        int tail_start = first_key + vis_count - tail_count;
        unsigned int base = row_select_k * KP;
        for (unsigned int t = 0; t + 1 < KP; ++t) {
            long long idx = (long long)tail_start + t;
            bool ok = ((int)t < tail_count) && idx >= 0 && idx < (long long)S
                      && ((int)idx <= q_pos[r]) && valid_keys[(unsigned)idx] != 0;
            if (base + t < width) row[base + t] = ok ? (int)idx : DSA_INVALID;
        }
    }
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
