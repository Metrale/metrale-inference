// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

// 2026-09-25: DFlash2 drafter kernels: grouped dynamic two-tap conv, per-row top-16, and
// the candidate-selector chain walk. Arithmetic in f32; activations stored as BF16.
//
// Owner: gb10 kernels.
// Invariants:
// - Rows are packed sequence-major in bands (gamma rows for the conv, `rows` rows for the
//   walk), and neither kernel reads a row of another band.


#include <cuda_bf16.h>

// 2026-09-25: Two-tap grouped dynamic causal convolution, one application (prepare or
// finish). For row t, channel c, group g = c / group_size:
//   coef_k(t, c) = base[k*h + c] + dyn[t*dyn_stride + app_off + k*groups + g]
//   out[t][c]    = coef_0 * x[t][c] + coef_1 * x[t-1][c]
// The second tap is dropped on the first row of each band of gamma rows (gamma == 0: only
// row 0), so no row convolves with another sequence's row. dyn rows hold
// [2 applications, kernel_size, groups]; app_off selects the application (0 = prepare,
// kernel_size * groups = finish). kernel_size is 2 (the host checks it). out must not
// alias x: block t reads row t - 1 of x. Launch: grid (rows, 1, 1), block (256, 1, 1).







extern "C" __global__ void dflash2_conv2(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ dyn,
    const __nv_bfloat16* __restrict__ base,
    __nv_bfloat16* __restrict__ out,
    unsigned int h,
    unsigned int group_size,
    unsigned int dyn_stride,
    unsigned int app_off,
    unsigned int gamma
) {
    const unsigned int t = blockIdx.x;
    const unsigned int groups = h / group_size;
    const __nv_bfloat16* xr = x + (size_t)t * h;





    const bool first_in_band = (gamma == 0u) ? (t == 0u) : (t % gamma == 0u);
    const __nv_bfloat16* xp = first_in_band ? nullptr : x + (size_t)(t - 1) * h;
    const __nv_bfloat16* dr = dyn + (size_t)t * dyn_stride + app_off;
    __nv_bfloat16* orow = out + (size_t)t * h;

    for (unsigned int c = threadIdx.x; c < h; c += blockDim.x) {
        const unsigned int g = c / group_size;
        const float c0 = __bfloat162float(base[c]) + __bfloat162float(dr[g]);
        float acc = c0 * __bfloat162float(xr[c]);
        if (xp != nullptr) {
            const float c1 =
                __bfloat162float(base[h + c]) + __bfloat162float(dr[groups + g]);
            acc += c1 * __bfloat162float(xp[c]);
        }
        orow[c] = __float2bfloat16(acc);
    }
}

// 2026-09-25: Per-row top-16 over BF16 logits, one block per row, 16 sequential
// block-argmax passes, so top_vals / top_idx come out in descending order. Destructive:
// each selected entry of `logits` is overwritten with -1e30. The tree reduction assumes
// blockDim.x is a power of two <= 1024 (the host launches 1024).

extern "C" __global__ void dflash2_topk16(
    __nv_bfloat16* __restrict__ logits,
    float* __restrict__ top_vals,
    unsigned int* __restrict__ top_idx,
    unsigned int n
) {
    __shared__ float s_val[1024];
    __shared__ unsigned int s_idx[1024];

    const unsigned int row = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int stride = blockDim.x;
    __nv_bfloat16* lrow = logits + (size_t)row * n;

    for (unsigned int k = 0; k < 16; ++k) {
        float local_max = -1e30f;
        unsigned int local_idx = 0;
        for (unsigned int i = tid; i < n; i += stride) {
            const float v = __bfloat162float(lrow[i]);
            if (v > local_max) {
                local_max = v;
                local_idx = i;
            }
        }
        s_val[tid] = local_max;
        s_idx[tid] = local_idx;
        __syncthreads();
        for (unsigned int off = stride >> 1; off > 0; off >>= 1) {
            if (tid < off && s_val[tid + off] > s_val[tid]) {
                s_val[tid] = s_val[tid + off];
                s_idx[tid] = s_idx[tid + off];
            }
            __syncthreads();
        }
        if (tid == 0) {
            top_vals[(size_t)row * 16 + k] = s_val[0];
            top_idx[(size_t)row * 16 + k] = s_idx[0];
            lrow[s_idx[0]] = __float2bfloat16(-1e30f);
        }
        __syncthreads();
    }
}

// 2026-10-10: dflash2_topk16 in two stages over many blocks, with the same result and the
// same writes to `logits`. dflash2_topk16 runs 16 block-argmax passes on one 1024-thread
// block per row, ~440 us for 8 rows of a 154880 vocabulary on GB10 (8 SMs busy). Its passes
// select, in order, the 16 greatest eligible entries (value > -1e30f; NaN never qualifies)
// under one total order: the value descending, then, among equal values, the entry its
// reduction keeps. Thread tid scans i = tid, tid + 1024, ... keeping the first of equal
// maxima (lowest i / 1024), and the tree (off = 512 .. 1, replace only on a strictly greater
// value) keeps the lower position at every level, which prefers the smaller bit-reversed
// 10-bit tid. topk16_tie below is that order's key (smaller first). The selected entries are
// overwritten with bf16(-1e30), which reads back below -1e30f, so a pass never re-selects
// one. When fewer than 16 entries qualify, a pass yields (-1e30f, 0) and writes index 0.
//
// Stage 1, dflash2_topk16_part: grid (rows, parts), block TOPK16_PART_THREADS; block (r, p)
// takes the 16 best of its TOPK16_PART_SPAN entries of row r under that order into
// cand_vals / cand_idx [rows, parts, 16] (an unfilled slot is (-inf, 0xFFFFFFFF)). Stage 2,
// dflash2_topk16_merge: grid (rows), block TOPK16_PART_THREADS; the 16 best of a row's
// parts * 16 candidates, written as dflash2_topk16 writes them. parts * 16 must be at most
// TOPK16_PART_THREADS * TOPK16_PER_THREAD (the host checks it).

#define TOPK16_PART_THREADS 256u
#define TOPK16_PER_THREAD 8u
#define TOPK16_PART_SPAN (TOPK16_PART_THREADS * TOPK16_PER_THREAD)
#define TOPK16_NEG_INF __int_as_float(0xff800000)

// 2026-10-10: dflash2_topk16's tie key of entry i (1024-thread block): the bit-reversed
// lane, then the stride index. Smaller wins among equal values.
__device__ __forceinline__ unsigned int topk16_tie(unsigned int i) {
    return ((__brev(i & 1023u) >> 22) << 22) | (i >> 10);
}

// 2026-10-10: (va, ta) ranks before (vb, tb).
__device__ __forceinline__ bool topk16_before(float va, unsigned int ta, float vb, unsigned int tb) {
    return va > vb || (va == vb && ta < tb);
}

// 2026-10-10: The block's best (value, entry) of every thread's TOPK16_PER_THREAD slots, 16
// times; each pass's winner is removed from its owner's slots. Ineligible slots hold
// (-inf, tie 0xFFFFFFFF, entry 0xFFFFFFFF), which every eligible value beats; a removed winner
// becomes one. out_v / out_i get the 16 winners in order, (-inf, 0xFFFFFFFF) where none was
// left.
__device__ __forceinline__ void topk16_block_select(
    float (&v)[TOPK16_PER_THREAD],
    unsigned int (&t)[TOPK16_PER_THREAD],
    unsigned int (&ix)[TOPK16_PER_THREAD],
    float* out_v,
    unsigned int* out_i
) {
    __shared__ float s_v[TOPK16_PART_THREADS / 32u];
    __shared__ unsigned int s_t[TOPK16_PART_THREADS / 32u];
    __shared__ unsigned int s_i[TOPK16_PART_THREADS / 32u];
    __shared__ unsigned int s_win;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;
    for (unsigned int k = 0; k < 16u; ++k) {
        float bv = v[0];
        unsigned int bt = t[0], bi = ix[0];
        #pragma unroll
        for (unsigned int j = 1; j < TOPK16_PER_THREAD; ++j) {
            if (topk16_before(v[j], t[j], bv, bt)) { bv = v[j]; bt = t[j]; bi = ix[j]; }
        }
        #pragma unroll
        for (unsigned int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_down_sync(0xffffffffu, bv, off);
            const unsigned int ot = __shfl_down_sync(0xffffffffu, bt, off);
            const unsigned int oi = __shfl_down_sync(0xffffffffu, bi, off);
            if (topk16_before(ov, ot, bv, bt)) { bv = ov; bt = ot; bi = oi; }
        }
        if (lane == 0u) { s_v[warp] = bv; s_t[warp] = bt; s_i[warp] = bi; }
        __syncthreads();
        if (threadIdx.x == 0u) {
            float wv = s_v[0];
            unsigned int wt = s_t[0], wi = s_i[0];
            for (unsigned int w = 1; w < TOPK16_PART_THREADS / 32u; ++w) {
                if (topk16_before(s_v[w], s_t[w], wv, wt)) { wv = s_v[w]; wt = s_t[w]; wi = s_i[w]; }
            }
            out_v[k] = wv;
            out_i[k] = wi;
            s_win = wt;
        }
        __syncthreads();
        // 2026-10-10: Remove the winner (ties are unique per entry); an ineligible winner
        // leaves every slot ineligible already.
        const unsigned int win = s_win;
        #pragma unroll
        for (unsigned int j = 0; j < TOPK16_PER_THREAD; ++j) {
            if (t[j] == win && win != 0xFFFFFFFFu) {
                v[j] = TOPK16_NEG_INF;
                t[j] = 0xFFFFFFFFu;
                ix[j] = 0xFFFFFFFFu;
            }
        }
        __syncthreads();
    }
}

extern "C" __global__ void __launch_bounds__(TOPK16_PART_THREADS) dflash2_topk16_part(
    const __nv_bfloat16* __restrict__ logits,
    float* __restrict__ cand_vals,
    unsigned int* __restrict__ cand_idx,
    unsigned int n
) {
    const unsigned int row = blockIdx.x;
    const unsigned int part = blockIdx.y;
    const unsigned int parts = gridDim.y;
    const __nv_bfloat16* lrow = logits + (size_t)row * n;
    const unsigned int lo = part * TOPK16_PART_SPAN;
    float v[TOPK16_PER_THREAD];
    unsigned int t[TOPK16_PER_THREAD], ix[TOPK16_PER_THREAD];
    #pragma unroll
    for (unsigned int j = 0; j < TOPK16_PER_THREAD; ++j) {
        const unsigned int i = lo + j * TOPK16_PART_THREADS + threadIdx.x;
        const float x = i < n ? __bfloat162float(lrow[i]) : TOPK16_NEG_INF;
        const bool ok = i < n && x > -1e30f;
        v[j] = ok ? x : TOPK16_NEG_INF;
        t[j] = ok ? topk16_tie(i) : 0xFFFFFFFFu;
        ix[j] = ok ? i : 0xFFFFFFFFu;
    }
    __shared__ float o_v[16];
    __shared__ unsigned int o_i[16];
    topk16_block_select(v, t, ix, o_v, o_i);
    if (threadIdx.x < 16u) {
        const size_t o = ((size_t)row * parts + part) * 16u + threadIdx.x;
        cand_vals[o] = o_v[threadIdx.x];
        cand_idx[o] = o_i[threadIdx.x];
    }
}

extern "C" __global__ void __launch_bounds__(TOPK16_PART_THREADS) dflash2_topk16_merge(
    __nv_bfloat16* __restrict__ logits,
    const float* __restrict__ cand_vals,
    const unsigned int* __restrict__ cand_idx,
    float* __restrict__ top_vals,
    unsigned int* __restrict__ top_idx,
    unsigned int n,
    unsigned int parts
) {
    const unsigned int row = blockIdx.x;
    const unsigned int m = parts * 16u;
    const float* cv = cand_vals + (size_t)row * m;
    const unsigned int* ci = cand_idx + (size_t)row * m;
    float v[TOPK16_PER_THREAD];
    unsigned int t[TOPK16_PER_THREAD], ix[TOPK16_PER_THREAD];
    #pragma unroll
    for (unsigned int j = 0; j < TOPK16_PER_THREAD; ++j) {
        const unsigned int c = j * TOPK16_PART_THREADS + threadIdx.x;
        const unsigned int i = c < m ? ci[c] : 0xFFFFFFFFu;
        const bool ok = i != 0xFFFFFFFFu;
        v[j] = ok ? cv[c] : TOPK16_NEG_INF;
        t[j] = ok ? topk16_tie(i) : 0xFFFFFFFFu;
        ix[j] = i;
    }
    __shared__ float o_v[16];
    __shared__ unsigned int o_i[16];
    topk16_block_select(v, t, ix, o_v, o_i);
    if (threadIdx.x < 16u) {
        const bool ok = o_i[threadIdx.x] != 0xFFFFFFFFu;
        const unsigned int i = ok ? o_i[threadIdx.x] : 0u;
        top_vals[(size_t)row * 16u + threadIdx.x] = ok ? o_v[threadIdx.x] : -1e30f;
        top_idx[(size_t)row * 16u + threadIdx.x] = i;
        logits[(size_t)row * n + i] = __float2bfloat16(-1e30f);
    }
}

// 2026-09-25: Candidate-selector chain walk, one block per sequence band:
//   S_t(a, b) = U_t(b) + sum_r pred[a][r] * hproj[t][r] * succ[b][r]
// U_t is top_vals and the candidates b are top_idx. Block i walks rows first_row..rows-1
// of band i (rows = rows per band). The predecessor a of the first walked row is
// tokens[i * rows]; each chosen candidate is written to tokens[row] and becomes the next
// row's predecessor. Launch: block (512, 1, 1), 16 warps, warp k scores candidate k.
// rank must be <= 256 (s_gate); the host checks only rank > 0.







extern "C" __global__ void dflash2_selector_walk(
    const float* __restrict__ top_vals,
    const unsigned int* __restrict__ top_idx,
    const __nv_bfloat16* __restrict__ hproj,
    const __nv_bfloat16* __restrict__ pred,
    const __nv_bfloat16* __restrict__ succ,
    unsigned int* __restrict__ tokens,
    unsigned int rows,
    unsigned int rank,
    unsigned int first_row
) {



    const size_t base_row = (size_t)blockIdx.x * rows;
    __shared__ float s_gate[256];
    __shared__ float s_score[16];
    __shared__ unsigned int s_prev;

    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & 31;

    if (tid == 0) {
        s_prev = tokens[base_row];
    }
    __syncthreads();

    for (unsigned int t = first_row; t < rows; ++t) {
        const size_t row = base_row + t;

        const __nv_bfloat16* pr = pred + (size_t)s_prev * rank;
        const __nv_bfloat16* hr = hproj + row * rank;
        for (unsigned int r = tid; r < rank; r += blockDim.x) {
            s_gate[r] = __bfloat162float(pr[r]) * __bfloat162float(hr[r]);
        }
        __syncthreads();


        if (warp < 16) {
            const unsigned int cand = top_idx[row * 16 + warp];
            const __nv_bfloat16* sr = succ + (size_t)cand * rank;
            float acc = 0.0f;
            for (unsigned int r = lane; r < rank; r += 32) {
                acc += s_gate[r] * __bfloat162float(sr[r]);
            }
            #pragma unroll
            for (unsigned int off = 16; off > 0; off >>= 1) {
                acc += __shfl_down_sync(0xffffffff, acc, off);
            }
            if (lane == 0) {
                s_score[warp] = top_vals[row * 16 + warp] + acc;
            }
        }
        __syncthreads();

        if (tid == 0) {
            float best = s_score[0];
            unsigned int best_k = 0;
            #pragma unroll
            for (unsigned int k = 1; k < 16; ++k) {
                if (s_score[k] > best) {
                    best = s_score[k];
                    best_k = k;
                }
            }
            const unsigned int chosen = top_idx[row * 16 + best_k];
            tokens[row] = chosen;
            s_prev = chosen;
        }
        __syncthreads();
    }
}
