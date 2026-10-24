// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-24: The body of glm5next_router_topk_rank (moved from glm5next_ffn.cu, which still
// launches it over a whole block) as a template over the thread group that runs one row (`G`:
// tid(), size(), sync()) and the logit load (`ld`), so a fused kernel can run several rows' top-k
// in one block (glm5next_router_gemv_topk, R2). Same expressions in the same order; the group
// size only strides the experts, so every group size gives the same ids and weights.
// The original notes:
//
// 2026-10-10: glm5next_router_topk_rank: glm5next_router_topk's ids and weights, byte for byte,
// without its top_k rounds of block-wide argmax (at one row on GB10 that kernel is ~15 us of
// barriers). The order is the same total order: by choice descending, an exact tie to the lower
// id, and an expert takes part exactly when the old kernel's seed (-1e30, id INT_MAX) loses to
// it, that is when its choice is >= -1e30 (a NaN or a lower choice is never selected). The k-th
// slot of the old kernel is the expert with exactly k participating experts ahead of it, so:
//   1. the scores are computed once per expert, with glm5next_router_topk's expressions;
//   2. each run of 32 consecutive experts ranks its members within the run (warp shuffles) and
//      keeps those with fewer than top_k ahead: every expert of the global top_k is kept,
//      because the experts ahead of it within its run are also ahead of it globally;
//   3. each kept expert counts the kept experts ahead of it; for an expert of the global top_k
//      every expert ahead of it is kept, so that count is its global rank and its slot, and an
//      expert outside the global top_k has all top_k winners ahead of it among the kept;
//   4. thread 0 forms the weights from the slots in slot order, as glm5next_router_topk does.
// Slots no expert fills keep id -1 and weight 0. Block: a multiple of 32 threads; the experts
// are strided over the block (expert e = i * blockDim.x + tid). The host launches
// 32 * ceil(num_experts / 32) threads (at most 1024) and keeps 1 <= top_k <= 16 and
// num_experts <= ROUTER_RANK_MAX_EXPERTS. n_group != 1 returns without writing anything.
//
// Owner: gb10 kernels (GLM-5.3).
// Invariants: G's size is a multiple of 32 and its threads are whole warps starting at a warp
// boundary (lanes are hardware lanes); MAX_EXPERTS bounds num_experts (above it nothing is
// written).
#pragma once

#include <cuda_bf16.h>

#define ROUTER_RANK_NEG_INF __int_as_float(0xff800000)
#define ROUTER_RANK_NO_EXPERT 2147483647

namespace glm_router_rank {

// 2026-10-24: One row's scratch: top_k (<= 16) kept slots per run of 32 experts.
template <unsigned int MAX_EXPERTS>
struct Smem {
    float2 kept_ce[(MAX_EXPERTS / 32) * 16];
    float kept_s[(MAX_EXPERTS / 32) * 16];
    int sel_id[16];
    float sel_w[16];
};

// 2026-10-24: The whole block runs one row (the original kernel).
struct BlockGroup {
    __device__ __forceinline__ unsigned int tid() const { return threadIdx.x; }
    __device__ __forceinline__ unsigned int size() const { return blockDim.x; }
    __device__ __forceinline__ void sync() const { __syncthreads(); }
};

// 2026-10-24: Threads [base, base + n) of the block run one row, synchronised on named barrier id
// (1..15; 0 is __syncthreads's).
struct BarGroup {
    unsigned int base, n, id;
    __device__ __forceinline__ unsigned int tid() const { return threadIdx.x - base; }
    __device__ __forceinline__ unsigned int size() const { return n; }
    __device__ __forceinline__ void sync() const { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory"); }
};

// 2026-10-24: Logit loads: as stored (another kernel wrote them), or through L2 only (this grid
// wrote them and every writer fenced).
struct LdPlain {
    __device__ __forceinline__ float operator()(const float* p) const { return *p; }
};
struct LdCg {
    __device__ __forceinline__ float operator()(const float* p) const { return __ldcg(p); }
};

template <unsigned int MAX_EXPERTS, class G, class Ld>
__device__ __forceinline__ void topk_rank_body(
    const G& g,
    const Ld& ld,
    Smem<MAX_EXPERTS>& sm,
    const float* __restrict__ row,
    const float* __restrict__ bias,
    int* __restrict__ ids,
    float* __restrict__ wts,
    const unsigned int num_experts,
    const unsigned int top_k,
    const unsigned int n_group,
    const float routed_scale,
    const unsigned int renormalize,
    const unsigned int bf16_ladder
) {
    const unsigned int tid = g.tid();
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5;
    if (n_group != 1 || num_experts > MAX_EXPERTS) return;
    float2* kept_ce = sm.kept_ce;
    float* kept_s = sm.kept_s;
    int* sel_id = sm.sel_id;
    float* sel_w = sm.sel_w;

    const unsigned int runs = (num_experts + 31u) / 32u;
    const unsigned int kept = (runs * top_k + 7u) & ~7u;
    if (tid < top_k) {
        sel_id[tid] = -1;
        sel_w[tid] = 0.0f;
    }
    for (unsigned int i = runs * top_k + tid; i < kept; i += g.size())
        kept_ce[i] = make_float2(ROUTER_RANK_NEG_INF, __int_as_float(ROUTER_RANK_NO_EXPERT));
    for (unsigned int base = 0; base < num_experts; base += g.size()) {
        const unsigned int e = base + tid;
        const unsigned int run = (base >> 5) + warp;
        float s = 0.0f;
        float c = ROUTER_RANK_NEG_INF;
        bool part = false;
        if (e < num_experts) {
            s = 1.0f / (1.0f + expf(-ld(row + e)));
            if (bf16_ladder) s = (float)__float2bfloat16(s);
            float ch = s + bias[e];
            if (bf16_ladder) ch = (float)__float2bfloat16(ch);
            part = ch >= -1e30f;
            if (part) c = ch;
        }
        unsigned int ahead = 0;
        #pragma unroll
        for (unsigned int j = 0; j < 32u; ++j) {
            const float cj = __shfl_sync(0xffffffffu, c, j);
            ahead += (unsigned int)(cj > c) | ((unsigned int)(cj == c) & (unsigned int)(j < lane));
        }
        if (run < runs && lane < top_k)
            kept_ce[run * top_k + lane] =
                make_float2(ROUTER_RANK_NEG_INF, __int_as_float(ROUTER_RANK_NO_EXPERT));
        __syncwarp();
        if (part && ahead < top_k) {
            kept_ce[run * top_k + ahead] = make_float2(c, __int_as_float((int)e));
            kept_s[run * top_k + ahead] = s;
        }
    }
    g.sync();

    for (unsigned int i = tid; i < kept; i += g.size()) {
        const float2 mine = kept_ce[i];
        const int ei = __float_as_int(mine.y);
        if (ei == ROUTER_RANK_NO_EXPERT) continue;
        const float ci = mine.x;
        unsigned int ahead = 0;
        for (unsigned int j0 = 0; j0 < kept; j0 += 8u) {
            #pragma unroll
            for (unsigned int u = 0; u < 8u; ++u) {
                const float2 o = kept_ce[j0 + u];
                ahead += (unsigned int)(o.x > ci) |
                         ((unsigned int)(o.x == ci) & (unsigned int)(__float_as_int(o.y) < ei));
            }
        }
        if (ahead < top_k) {
            sel_id[ahead] = ei;
            sel_w[ahead] = kept_s[i];
        }
    }
    g.sync();

    // 2026-10-10: Slot k's thread forms its weight: the slot sum in slot order (every slot
    // thread adds the same values in the same order), then glm5next_router_topk's steps.
    if (tid >= top_k) return;
    float sum = 0.0f;
    for (unsigned int k = 0; k < top_k; ++k) {
        sum += sel_w[k];
        if (bf16_ladder) sum = (float)__float2bfloat16(sum);
    }
    float w = sel_w[tid];
    if (renormalize) {
        w = w / (sum + 1e-20f);
        if (bf16_ladder) w = (float)__float2bfloat16(w);
    }
    w *= routed_scale;
    if (bf16_ladder) w = (float)__float2bfloat16(w);
    ids[tid] = sel_id[tid];
    wts[tid] = w;
}

}  // namespace glm_router_rank
