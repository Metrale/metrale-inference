// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-14: Decode-width GEMVs of the Qwen3.8-Flash-Next low-rank hyper-connection collapse.
// At small T, `hc_pre_gemm` (crates/model-layers/src/layers/ops/hyper_connection_lowrank_gemm.rs)
// runs these two after `hc_pre_stage_bf16`, in place of its cuBLAS projections (five launches at
// T = 1: split-K down + reduce, up, dot + reduce for the injection) and of `hc_silu_scale` and
// `hc_pre_mix`:
//
//   hc_lowrank_down_inject  low = bf16(silu(bf16(normed x down_w^T) / hc))       [T, rank] BF16
//                           inj = 2 * sigmoid(bf16(normed x inject_w^T) / hc)     [T, hc]   FP32
//   hc_lowrank_up_mix       y[d] = bf16(mean_s sigmoid(bf16(low x up_w^T)[s*H+d]) * normed[s*H+d])
//
// The epilogues are `hc_silu_scale`'s and `hc_pre_mix`'s formulas (qhc_act.cuh), applied to the
// projections rounded to BF16, as the GEMM path rounds its GEMM outputs; the mean adds the streams
// in ascending order from zero, as `hc_pre_mix` does. Only the projections' summation order differs
// from the cuBLAS path.
//
// Summation: each thread sums its K slice (8-element vectors at stride HLG_THREADS or 32) in
// ascending order in FP32; a fixed butterfly reduce-scatter combines a warp's 32 lanes, and (down
// only) the block's warps are added in ascending order. The tree depends on K and the block shape
// alone, never on T or on a token's place in the batch: a token's outputs are the same bits at
// every T and on every run.
//
// Requirements (the launcher admits only these; the kernels trap on the rest): blockDim.x ==
// HLG_THREADS; K = hc * H and rank multiples of 8; rank <= HLG_MAX_RANK; hc <= HLG_STREAMS; every
// pointer 16-byte aligned.
//
// Owner: hopper kernels.
// Invariants: each token's outputs are independent of T (`ops/hyper_connection_gemv_tests.rs`).

#include <cuda_bf16.h>

#include "qhc_act.cuh"  // 2026-10-14: qhc_silu, qhc_sigmoid

#define HLG_THREADS 256
#define HLG_WARPS (HLG_THREADS / 32)
// 2026-10-14: Tokens per block (grid y counts token groups). HLG_ROWS * HLG_DOWN_TOKENS and
// HLG_STREAMS * HLG_TOKENS are 32: one value per lane after the reduce-scatter. The down GEMV takes
// one weight row per block (rank + hc = 324 blocks at T <= 32, not 81 at four rows: 81 left 51 SMs
// idle and ran at ~1 TB/s); the reduce-scatter adds every index by the same tree, so a row's bits do
// not depend on how many rows or tokens share its block.
#define HLG_TOKENS 8
#define HLG_ROWS 1
#define HLG_DOWN_TOKENS 32
#define HLG_STREAMS 4
// 2026-10-14: K-loop iterations a lane of the up GEMV runs at HLG_MAX_RANK (rank / 8 / 32).
#define HLG_UP_ITERS (HLG_MAX_RANK / 8 / 32)
#define HLG_MAX_RANK 512

// 2026-10-14: acc += dot(w, x) over one 8-element BF16 vector, in ascending element order.
__device__ __forceinline__ void hlg_dot8(float& acc, const uint4& w, const uint4& x) {
    const __nv_bfloat162* wp = reinterpret_cast<const __nv_bfloat162*>(&w);
    const __nv_bfloat162* xp = reinterpret_cast<const __nv_bfloat162*>(&x);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 wf = __bfloat1622float2(wp[i]);
        const float2 xf = __bfloat1622float2(xp[i]);
        acc += wf.x * xf.x;
        acc += wf.y * xf.y;
    }
}

// 2026-10-14: One butterfly step over the first 2 * HALF values: a lane keeps the half its lane
// bit HALF selects and adds its partner's copy of that half into v[0, HALF). The selects are bit
// masks, not `?:`, which the compiler turns into a dynamically indexed load (v to local memory).
template <int HALF>
__device__ __forceinline__ void hlg_butterfly_step(float (&v)[32], unsigned int lane) {
    const unsigned int up = 0u - ((lane / (unsigned int)HALF) & 1u);
    #pragma unroll
    for (int j = 0; j < HALF; ++j) {
        const unsigned int a = __float_as_uint(v[j]);
        const unsigned int b = __float_as_uint(v[j + HALF]);
        const float send = __uint_as_float((a & up) | (b & ~up));
        const float keep = __uint_as_float((b & up) | (a & ~up));
        v[j] = keep + __shfl_xor_sync(0xFFFFFFFFu, send, HALF);
    }
}

// 2026-10-14: Butterfly reduce-scatter: returns, on lane L, the sum over the warp's lanes of v[L].
// After the step of bit HALF a lane's live values are those whose index bit HALF equals its lane
// bit, so after five steps lane L holds index L. v is clobbered.
__device__ __forceinline__ float hlg_reduce_scatter32(float (&v)[32]) {
    const unsigned int lane = threadIdx.x & 31u;
    hlg_butterfly_step<16>(v, lane);
    hlg_butterfly_step<8>(v, lane);
    hlg_butterfly_step<4>(v, lane);
    hlg_butterfly_step<2>(v, lane);
    hlg_butterfly_step<1>(v, lane);
    return v[0];
}

// 2026-10-14: Grid [ceil((rank + hc_inj) / HLG_ROWS), ceil(T / HLG_DOWN_TOKENS)], block HLG_THREADS,
// where hc_inj = hc with an injection weight and 0 without. Block x owns weight rows
// [x * HLG_ROWS, +HLG_ROWS) of [down_w ; inject_w].
extern "C" __global__ void __launch_bounds__(HLG_THREADS) hc_lowrank_down_inject(
    const __nv_bfloat16* __restrict__ normed,    // 2026-10-14: [T, K]
    const __nv_bfloat16* __restrict__ down_w,    // 2026-10-14: [rank, K]
    const __nv_bfloat16* __restrict__ inject_w,  // 2026-10-14: [hc, K] or null
    __nv_bfloat16* __restrict__ low,             // 2026-10-14: [T, rank]
    float* __restrict__ inj_out,                 // 2026-10-14: [T, hc]; unused without inject_w
    const unsigned int T,
    const unsigned int K,
    const unsigned int rank,
    const unsigned int hc,
    const float inv_hc
) {
    if (blockDim.x != HLG_THREADS || (K & 7u) != 0u) __trap();
    const unsigned int rows = rank + (inject_w != nullptr ? hc : 0u);
    const unsigned int r0 = blockIdx.x * HLG_ROWS;
    const unsigned int t0 = blockIdx.y * HLG_DOWN_TOKENS;
    if (r0 >= rows || t0 >= T) return;
    const unsigned int nt = min((unsigned int)HLG_DOWN_TOKENS, T - t0);
    const unsigned int k8n = K / 8u;

    const uint4* wrow[HLG_ROWS];
    #pragma unroll
    for (int r = 0; r < HLG_ROWS; ++r) {
        const unsigned int row = r0 + r;
        wrow[r] = row < rank ? reinterpret_cast<const uint4*>(down_w + (size_t)row * K)
                : row < rows ? reinterpret_cast<const uint4*>(inject_w + (size_t)(row - rank) * K)
                : nullptr;
    }
    const uint4* xrow = reinterpret_cast<const uint4*>(normed + (size_t)t0 * K);

    float acc[HLG_ROWS * HLG_DOWN_TOKENS];
    #pragma unroll
    for (int i = 0; i < HLG_ROWS * HLG_DOWN_TOKENS; ++i) acc[i] = 0.0f;

    // 2026-10-14: Unrolled so a thread's weight loads issue together (the loop is load-latency
    // bound at decode); each accumulator still adds its K slice in ascending order.
    const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
    #pragma unroll 4
    for (unsigned int k = threadIdx.x; k < k8n; k += HLG_THREADS) {
        uint4 w[HLG_ROWS];
        #pragma unroll
        for (int r = 0; r < HLG_ROWS; ++r) w[r] = wrow[r] != nullptr ? __ldg(wrow[r] + k) : zero;
        #pragma unroll
        for (int t = 0; t < HLG_DOWN_TOKENS; ++t) {
            if ((unsigned int)t < nt) {
                const uint4 x = __ldg(xrow + (size_t)t * k8n + k);
                #pragma unroll
                for (int r = 0; r < HLG_ROWS; ++r) hlg_dot8(acc[r * HLG_DOWN_TOKENS + t], w[r], x);
            }
        }
    }

    __shared__ float red[HLG_WARPS][32];
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;
    red[warp][lane] = hlg_reduce_scatter32(acc);
    __syncthreads();
    if (threadIdx.x >= 32u) return;

    float sum = 0.0f;
    #pragma unroll
    for (int w2 = 0; w2 < HLG_WARPS; ++w2) sum += red[w2][lane];
    const unsigned int row = r0 + lane / HLG_DOWN_TOKENS;
    const unsigned int t = lane % HLG_DOWN_TOKENS;
    if (row >= rows || t >= nt) return;
    const float p = __bfloat162float(__float2bfloat16(sum)) * inv_hc;
    if (row < rank) {
        low[(size_t)(t0 + t) * rank + row] = __float2bfloat16(qhc_silu(p));
    } else {
        inj_out[(size_t)(t0 + t) * hc + (row - rank)] = 2.0f * qhc_sigmoid(p);
    }
}

// 2026-10-14: Grid [ceil(H / HLG_WARPS), ceil(T / HLG_TOKENS)], block HLG_THREADS. Warp w of
// block x owns hidden index d = x * HLG_WARPS + w: up_w rows s * H + d for every stream s.
extern "C" __global__ void __launch_bounds__(HLG_THREADS) hc_lowrank_up_mix(
    const __nv_bfloat16* __restrict__ low,     // 2026-10-14: [T, rank]
    const __nv_bfloat16* __restrict__ up_w,    // 2026-10-14: [hc * H, rank]
    const __nv_bfloat16* __restrict__ normed,  // 2026-10-14: [T, hc * H]
    __nv_bfloat16* __restrict__ y_out,         // 2026-10-14: [T, H]
    const unsigned int T,
    const unsigned int H,
    const unsigned int hc,
    const unsigned int rank,
    const float inv_hc
) {
    if (blockDim.x != HLG_THREADS || (rank & 7u) != 0u || rank > HLG_MAX_RANK || hc > HLG_STREAMS) {
        __trap();
    }
    __shared__ uint4 lows[HLG_TOKENS * (HLG_MAX_RANK / 8)];
    const unsigned int t0 = blockIdx.y * HLG_TOKENS;
    if (t0 >= T) return;
    const unsigned int nt = min((unsigned int)HLG_TOKENS, T - t0);
    const unsigned int r8n = rank / 8u;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int d = blockIdx.x * HLG_WARPS + (threadIdx.x >> 5);
    const bool live = d < H;
    const unsigned int s = lane / HLG_TOKENS;
    const unsigned int t = lane % HLG_TOKENS;

    // 2026-10-14: The weights and the lane's `normed` element do not depend on `low`: their loads
    // issue before `low` is staged and the barrier, so the three latencies overlap.
    const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
    uint4 w[HLG_UP_ITERS][HLG_STREAMS];
    #pragma unroll
    for (int i = 0; i < HLG_UP_ITERS; ++i) {
        const unsigned int k = lane + 32u * (unsigned int)i;
        #pragma unroll
        for (int s2 = 0; s2 < HLG_STREAMS; ++s2) {
            w[i][s2] = live && k < r8n && (unsigned int)s2 < hc
                ? __ldg(reinterpret_cast<const uint4*>(up_w + ((size_t)s2 * H + d) * rank) + k)
                : zero;
        }
    }
    const size_t hc_dim = (size_t)hc * H;
    const bool term_live = live && s < hc && t < nt;
    const float nv =
        term_live ? __bfloat162float(normed[(t0 + t) * hc_dim + (size_t)s * H + d]) : 0.0f;

    const uint4* lrow = reinterpret_cast<const uint4*>(low + (size_t)t0 * rank);
    for (unsigned int i = threadIdx.x; i < nt * r8n; i += HLG_THREADS) {
        lows[(i / r8n) * (HLG_MAX_RANK / 8) + i % r8n] = __ldg(lrow + i);
    }
    __syncthreads();
    if (!live) return;

    float acc[HLG_STREAMS * HLG_TOKENS];
    #pragma unroll
    for (int i = 0; i < HLG_STREAMS * HLG_TOKENS; ++i) acc[i] = 0.0f;

    #pragma unroll
    for (int i = 0; i < HLG_UP_ITERS; ++i) {
        const unsigned int k = lane + 32u * (unsigned int)i;
        if (k >= r8n) break;
        #pragma unroll
        for (int t2 = 0; t2 < HLG_TOKENS; ++t2) {
            if ((unsigned int)t2 < nt) {
                const uint4 x = lows[t2 * (HLG_MAX_RANK / 8) + k];
                #pragma unroll
                for (int s2 = 0; s2 < HLG_STREAMS; ++s2) {
                    hlg_dot8(acc[s2 * HLG_TOKENS + t2], w[i][s2], x);
                }
            }
        }
    }

    // 2026-10-14: Lane s * HLG_TOKENS + t holds stream s, token t; lanes t < HLG_TOKENS gather the
    // streams in ascending order.
    const float part = hlg_reduce_scatter32(acc);
    float term = 0.0f;
    if (term_live) {
        const float u = __bfloat162float(__float2bfloat16(part));
        term = qhc_sigmoid(u) * nv;
    }
    float mixed = 0.0f;
    #pragma unroll
    for (int s2 = 0; s2 < HLG_STREAMS; ++s2) {
        const float v = __shfl_sync(0xFFFFFFFFu, term, s2 * HLG_TOKENS + (int)t);
        if ((unsigned int)s2 < hc) mixed += v;
    }
    if (lane < nt) y_out[(size_t)(t0 + lane) * H + d] = __float2bfloat16(mixed * inv_hc);
}
