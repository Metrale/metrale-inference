// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-24: The body of glm5next_hc_pre_norm_c1 as a template over where the token's stream
// values come from (`Src`), moved verbatim from glm5next_mhc_c1.cu. Two sources:
//   - glmc1::LoadStreams: the streams row as stored (glm5next_hc_pre_norm_c1, glm5next_mhc_c1.cu);
//   - glmc1::PostRankSum: the streams glm5next_hc_post_rank_sum would store (the one-shot
//     all-reduce's rank sum, then the mHC post-mix), computed in registers from this rank's
//     partial, the peers' slots and the old highway; the last block of the token stores them and
//     the summed row (glm5next_hc_post_pre_norm_c1, glm5next_mhc_post_pre_c1.cu, R3).
// The original kernel's notes follow.
//
// 2026-10-21: glm5next_hc_pre_norm for the decode widths (fewer than 64 tokens) at GLM-5.3's shape
// (hc_mult 4, H 4096: hc_mult * H == 64 * GLM_HCF_BLOCK), with the latency taken out of the last
// block's tail. Same grid (T, 24) and the same mix chains as glm5next_hc_pre_norm_pf64 (every
// thread's 64 stream and 64 hc_fn loads in flight at once, accumulated in the loop's order).
// What changes, measured on GB10 at T = 2 (90 distinct sites in a graph): the old tail took
// 3.4 us after a 6.3 us mix, the Sinkhorn on warp 0 (about 3 us) beside the collapse + norm on
// warps 1..7 (about 2.8 us, five dependent L2 round trips for norm_w among them).
//   - The 256-wide reduction trees run on warp 0 (three in-lane levels, then shfl_down): the
//     block tree's pairs, so the same sums, with one barrier instead of eight.
//   - Everything the tail reads from global memory except the other blocks' mixes (norm_w, the
//     scales and biases) is loaded before the mix, so no tail step waits on it.
//   - The collapse y[d] = sum_i pre[i] x[i][d] reads the stream values the threads already hold
//     (thread tid holds x[i][tid + 256 m] for every i and m < 16; warp 0 parks its share in shared
//     memory during the mix) instead of a second pass over the 64 KB of streams through L2.
//   - Warp 0 starts the Sinkhorn as soon as the mixes are in shared memory; warps 1..7 form pre
//     (the four sigmoids, the same expression), the collapse and the RMS norm meanwhile.
// The Sinkhorn (glm_hcf_sinkhorn_at), the RMS (glm_hcf_rms_at) and the normalize
// (glm_hcf_norm_quad) are glm5next_mhc_fused_parts.cuh's.
//
// Owner: gb10 kernels.
// Invariants:
// - Every output is byte-identical to glm5next_hc_pre_norm (glm5next_mhc_fused.cu), which is
//   byte-identical to the unfused hc_mix_bf16 -> hc_finish -> rms_norm_vanilla
//   (--fmad=false on every side). glm5next_hc_pre_c1_gate (gpu examples) checks y, post, comb and
//   the normed row byte for byte.
// - hc_mult == 4 and H == 4096, else the kernel returns without writing (the host never launches
//   it there). Grid (T, 24, 1), block (GLM_HCF_BLOCK, 1, 1); `done` as glm5next_hc_pre_norm's.
//
// 2026-10-24 (template) Invariants:
// - With LoadStreams every byte is the pre-template kernel's (same expressions, same order).
// - With PostRankSum each xv value is glm5next_hc_post_rank_sum's output element (FP32 rank sum
//   from 0.0f in rank order, one __float2bfloat16_rn, then post[j] * x + comb[i * hc + j] * res[i]
//   for i ascending; --fmad=false), so every pre-norm output byte equals the two launches'. The
//   highway is updated in place: only the token's last block stores the new streams and the
//   summed row, after every block of the token counted itself in `done` (all their loads of the
//   old row are complete), so no block reads a value another block of the launch wrote.

#pragma once

#include "glm5next_mhc_fused_parts.cuh"
#include "pdl.cuh"

namespace glmc1 {

constexpr unsigned int HC = 4;
constexpr unsigned int MIX = (2 + HC) * HC;
constexpr unsigned int PF = 64;
constexpr unsigned int H = PF * GLM_HCF_BLOCK / HC;
constexpr unsigned int COLS = PF / HC;
constexpr unsigned int NQ = H / 4u;
constexpr unsigned int TAIL_PASSES = (NQ + GLM_HCF_TAIL_THREADS - 1) / GLM_HCF_TAIL_THREADS;

// 2026-10-21: The block tree `for s = 128..1: if (tid < s) red[tid] += red[tid + s]` on one warp:
// lane l holds red[l + 32 q], q < 8; levels 128, 64 and 32 pair q with q + 4, q + 2, q + 1 inside
// the lane, levels 16..1 pair lane l with lane l + s (shfl_down). Lane 0 ends with red[0].
__device__ __forceinline__ float warp_tree(const float* red, unsigned int lane) {
    float a[8];
#pragma unroll
    for (int q = 0; q < 8; ++q) a[q] = red[lane + 32u * q];
#pragma unroll
    for (int q = 0; q < 4; ++q) a[q] += a[q + 4];
#pragma unroll
    for (int q = 0; q < 2; ++q) a[q] += a[q + 2];
    a[0] += a[1];
#pragma unroll
    for (int s = 16; s > 0; s >>= 1) a[0] += __shfl_down_sync(0xFFFFFFFFu, a[0], s);
    return a[0];
}

struct Smem {
    float red_ss[GLM_HCF_BLOCK];
    float red[GLM_HCF_BLOCK];
    float mix[MIX];
    float ws[33];
    bool last;
    __nv_bfloat16 y[H];
    // 2026-10-21: Warp 0's stream values, [j][lane] (j < PF), for the tail's collapse.
    float x0[PF][32];
};

// 2026-10-21: Column d of y (glm_hcf_collapse_norm_at's sum, 0 + pre[0] x[0][d] + .. in i order),
// stored to the output row and to shared memory.
__device__ __forceinline__ void collapse_col(const float (&pre)[HC], const float (&xc)[HC], unsigned int d,
                                             __nv_bfloat16* __restrict__ yo, __nv_bfloat16* s_y) {
    float a = 0.f;
#pragma unroll
    for (int i = 0; i < (int)HC; ++i) a += pre[i] * xc[i];
    const __nv_bfloat16 yb = __float2bfloat16(a);
    yo[d] = yb;
    s_y[d] = yb;
}


// 2026-10-24: Src for the stored streams row.
struct LoadStreams {
    __device__ __forceinline__ void load(float (&xv)[PF], const float* x, unsigned int, unsigned int tid) const {
#pragma unroll
        for (int j = 0; j < (int)PF; ++j) xv[j] = x[tid + (unsigned int)j * GLM_HCF_BLOCK];
    }
    __device__ __forceinline__ void store_last(const float (&)[PF], unsigned int, unsigned int) const {}
};

// 2026-10-24: The most ranks PostRankSum sums (glm5next_hc_post_sum.cu's GLM_HCPS_MAX_RANKS).
constexpr int MAX_RANKS = 8;

// 2026-10-24: Src for the streams glm5next_hc_post_rank_sum would write: `block_out` holds this
// rank's partial (row t at t * H), `peers` the others' (peer r in slot r < my_rank ? r : r - 1,
// `slot_elems` apart), `residual` the old highway (also the output: `out` may equal it), `post`
// and `comb` the site's post-mix weights.
struct PostRankSum {
    __nv_bfloat16* block_out;
    const __nv_bfloat16* peers;
    long long slot_elems;
    int n_ranks;
    int my_rank;
    const float* residual;
    const float* post;
    const float* comb;
    float* out;

    // 2026-10-24: bf16_add_rank_sum's sum of column d of row t.
    __device__ __forceinline__ __nv_bfloat16 sum_at(unsigned int t, unsigned int d) const {
        const long long e = (long long)t * H + d;
        float acc = 0.0f;
#pragma unroll
        for (int r = 0; r < MAX_RANKS; ++r) {
            if (r < n_ranks) {
                const __nv_bfloat16 v = (r == my_rank) ? block_out[e]
                    : peers[(long long)(r < my_rank ? r : r - 1) * slot_elems + e];
                acc += __bfloat162float(v);
            }
        }
        return __float2bfloat16_rn(acc);
    }

    __device__ __forceinline__ void load(float (&xv)[PF], const float*, unsigned int t, unsigned int tid) const {
        float p[HC], c[HC * HC];
#pragma unroll
        for (int i = 0; i < (int)HC; ++i) p[i] = post[(size_t)t * HC + i];
#pragma unroll
        for (int i = 0; i < (int)(HC * HC); ++i) c[i] = comb[(size_t)t * HC * HC + i];
        const float* res = residual + (size_t)t * HC * H;
#pragma unroll
        for (int m = 0; m < (int)COLS; ++m) {
            const unsigned int d = tid + (unsigned int)m * GLM_HCF_BLOCK;
            const float xd = (float)sum_at(t, d);
            float rv[HC];
#pragma unroll
            for (int i = 0; i < (int)HC; ++i) rv[i] = res[(size_t)i * H + d];
            // glm5next_hc_post's arithmetic for column d: stream j is xv[j * COLS + m].
#pragma unroll
            for (int j = 0; j < (int)HC; ++j) {
                float a = p[j] * xd;
#pragma unroll
                for (int i = 0; i < (int)HC; ++i) a += c[i * HC + j] * rv[i];
                xv[j * COLS + m] = a;
            }
        }
    }

    // 2026-10-24: The token's last block: the new streams over the old ones and the summed row
    // over this rank's partial (what glm5next_hc_post_rank_sum leaves in both).
    __device__ __forceinline__ void store_last(const float (&xv)[PF], unsigned int t, unsigned int tid) const {
        float* o = out + (size_t)t * HC * H;
#pragma unroll
        for (int j = 0; j < (int)PF; ++j) o[tid + (unsigned int)j * GLM_HCF_BLOCK] = xv[j];
        __nv_bfloat16 xs[COLS];
#pragma unroll
        for (int m = 0; m < (int)COLS; ++m) xs[m] = sum_at(t, tid + (unsigned int)m * GLM_HCF_BLOCK);
        // Each thread reads and writes only its own columns of the row.
#pragma unroll
        for (int m = 0; m < (int)COLS; ++m) block_out[(size_t)t * H + tid + (unsigned int)m * GLM_HCF_BLOCK] = xs[m];
    }
};

// 2026-10-24: The kernel body; `src` fills each thread's 64 stream values after pdl_enter() and
// runs its last-block store before the tail.
template <class Src>
__device__ __forceinline__ void glm_hcc1_body(GLM_HCF_ARGS, const Src& src) {
    __shared__ Smem sm;
    if (hc_mult != HC || hidden_size != H) return;
    const unsigned int t = blockIdx.x;
    const unsigned int m0 = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int u = tid - 32u;
    const unsigned int hc_dim = HC * H;
    const float* x = streams + (size_t)t * hc_dim;
    const __nv_bfloat16* __restrict__ wrow = hc_fn + (size_t)m0 * hc_dim;

    // 2026-10-21: PDL (pdl.cuh): the hc_fn row is loaded before the wait; the streams (the
    // predecessor's output) and every store after it. The tail's weights stay after the streams'
    // loads, as before: hoisting them too held them live through the mix (126 -> 158 registers,
    // +0.6 us per launch without PDL).
    float xv[PF];
    __nv_bfloat16 wv[PF];
#pragma unroll
    for (int j = 0; j < (int)PF; ++j) wv[j] = wrow[tid + (unsigned int)j * GLM_HCF_BLOCK];
    pdl_enter();
    src.load(xv, x, t, tid);

    // 2026-10-21: The tail's operands that do not depend on the mixes. Warp 0: its lane's post and
    // comb scale and bias (glm_hcf_sinkhorn's loads); warps 1..7: the pre scale and biases and
    // the norm weights of the quads they will write.
    const unsigned int l16 = lane % (HC * HC);
    float sc = 0.f, base_c = 0.f, base_p = 0.f, sc_p = 0.f;
    float base_pre[HC];
    uint2 nw[TAIL_PASSES];
    if (tid < 32) {
        sc = hc_scale[2];
        base_c = hc_base[2 * HC + l16];
        if (lane < HC) {
            sc_p = hc_scale[1];
            base_p = hc_base[HC + lane];
        }
    } else {
        sc = hc_scale[0];
#pragma unroll
        for (int i = 0; i < (int)HC; ++i) base_pre[i] = hc_base[i];
        const uint2* w4 = reinterpret_cast<const uint2*>(norm_w);
#pragma unroll
        for (int n = 0; n < (int)TAIL_PASSES; ++n) {
            const unsigned int q = u + (unsigned int)n * GLM_HCF_TAIL_THREADS;
            nw[n] = q < NQ ? w4[q] : make_uint2(0u, 0u);
        }
    }

    float ss = 0.f, acc = 0.f;
#pragma unroll
    for (int j = 0; j < (int)PF; ++j) {
        const float v = xv[j];
        ss += v * v;
        acc += __bfloat162float(wv[j]) * v;
    }
    sm.red_ss[tid] = ss;
    sm.red[tid] = acc;
    if (tid < 32) {
#pragma unroll
        for (int j = 0; j < (int)PF; ++j) sm.x0[j][lane] = xv[j];
    }
    __syncthreads();
    if (tid < 32) {
        const float tss = warp_tree(sm.red_ss, lane);
        const float tacc = warp_tree(sm.red, lane);
        if (lane == 0) {
            const float rsq = rsqrtf(tss / (float)hc_dim + norm_eps);
            mix[(size_t)t * MIX + m0] = tacc * rsq;
            __threadfence();
            const unsigned int prev = atomicAdd(&done[t], 1u);
            sm.last = (prev == gridDim.y - 1);
        }
    }
    __syncthreads();
    if (!sm.last) return;

    // 2026-10-21: Last block of token t: every other block's mix is visible after the fence. The
    // counter goes back to zero for the next launch.
    __threadfence();
    if (tid == 0) done[t] = 0;
    src.store_last(xv, t, tid);
    if (tid < MIX) sm.mix[tid] = __ldcg(&mix[(size_t)t * MIX + tid]);
    __syncthreads();

    if (tid < 32) {
        float po = 0.f;
        if (lane < HC) po = sm.mix[HC + lane] * sc_p + base_p;
        const float v = sm.mix[2 * HC + l16] * sc + base_c;
        glm_hcf_sinkhorn_at<HC>(po, v, post_out, comb_out, t, sinkhorn_iters, hc_eps);
        return;
    }
    float pre[HC];
#pragma unroll
    for (int i = 0; i < (int)HC; ++i) {
        float pr = sm.mix[i] * sc + base_pre[i];
        pre[i] = 1.f / (1.f + expf(-pr)) + hc_eps;
    }
    __nv_bfloat16* yo = y_out + (size_t)t * H;
#pragma unroll
    for (int m = 0; m < (int)COLS; ++m) {
        const float xc[HC] = {xv[m], xv[m + COLS], xv[m + 2 * COLS], xv[m + 3 * COLS]};
        collapse_col(pre, xc, tid + (unsigned int)m * GLM_HCF_BLOCK, yo, sm.y);
    }
    // 2026-10-21: Warp 0's columns d = l + 256 m (l < 32, m < COLS) from its parked values.
    for (unsigned int k = u; k < 32u * COLS; k += GLM_HCF_TAIL_THREADS) {
        const unsigned int l = k & 31u, m = k >> 5;
        const float xc[HC] = {sm.x0[m][l], sm.x0[m + COLS][l], sm.x0[m + 2 * COLS][l], sm.x0[m + 3 * COLS][l]};
        collapse_col(pre, xc, l + m * GLM_HCF_BLOCK, yo, sm.y);
    }
    glm_hcf_tail_bar_at<1>();
    const float rms = glm_hcf_rms_at<1>(u, sm.y, sm.ws, H, rms_eps);
    uint2* out = reinterpret_cast<uint2*>(normed_out + (size_t)t * H);
#pragma unroll
    for (int n = 0; n < (int)TAIL_PASSES; ++n) {
        const unsigned int q = u + (unsigned int)n * GLM_HCF_TAIL_THREADS;
        if (q < NQ) out[q] = glm_hcf_norm_quad(sm.y, q, nw[n], rms);
    }
}

}  // namespace glmc1
