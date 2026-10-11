// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-09: GLM-5.3 mHC pre-mix in one launch: glm5next_hc_mix_bf16 (or its 8-row variant),
// glm5next_hc_finish and the rms_norm_vanilla that follows it, fused with a last-block-done
// tail.
//
// Owner: gb10 kernels.
// Invariants:
// - Every output is byte-identical to the unfused sequence glm5next_hc_mix_bf16 (or
//   glm5next_hc_mix_bf16_rows) -> glm5next_hc_finish -> rms_norm_vanilla launched with
//   blockDim min(H, 1024): each value is computed with the same operations in the same order
//   (the build passes --fmad=false, so no product is contracted into an FMA on either side).
//   glm5next_hc_fused_gate (model-arch examples) checks y, post, comb and the normed row
//   byte for byte against those kernels.
// - blockDim == GLM_HCF_BLOCK (256); hc_mult is 1..=GLM_HCF_MAX_MULT (4); H is a multiple of
//   4, at most GLM_HCF_MAX_H, and min(H, 1024) is a multiple of 32. glm_hc_pre_norm (model-arch
//   glm5next_mhc_fused.rs) refuses any other shape.
// - `done` holds one u32 per token, zero before the launch; the last block of each token puts
//   it back to zero, so the next launch finds it zeroed.
//
// Mix phase: block (t, y) computes ROWS rows y*ROWS .. of token t's mixes. The RMS partials
// and the dot partials are taken in one pass over the streams; each partial is the same
// thread-strided sum in the same k order as the unfused kernel, and each 256-wide tree runs
// in the same order, so the RMS and every mix keep their bits.
//
// Tail: the block that finds itself last for its token (an atomic on done[t], after a fence
// on the mix stores) reads the mixes back. Warp 0 runs post and the Sinkhorn on 16 lanes, one
// comb element per lane in a register; row and column sums gather the operands by shuffle and
// add them in the unfused order. Warps 1..7 collapse y = sum_i pre[i] * x[i] and run the RMS
// norm of y, emulating rms_norm_vanilla's min(H, 1024) threads, their warp butterflies and its
// final warp butterfly over the warp sums.

#include <cuda_bf16.h>

#define GLM_HCF_BLOCK 256
#define GLM_HCF_MAX_MULT 4
#define GLM_HCF_MAX_MIX 24
#define GLM_HCF_MAX_H 8192
#define GLM_HCF_TAIL_THREADS (GLM_HCF_BLOCK - 32)

// 2026-10-09: rms_norm_vanilla's warp_reduce_sum: an xor butterfly, offsets 16 down to 1.
__device__ __forceinline__ float glm_hcf_warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
    return v;
}

// 2026-10-09: Named barrier over warps 1..7, the tail's y + norm threads; warp 0 runs the
// Sinkhorn meanwhile and never reaches it.
__device__ __forceinline__ void glm_hcf_tail_bar() {
    asm volatile("bar.sync 1, %0;" ::"n"(GLM_HCF_TAIL_THREADS));
}

// 2026-10-09: post and the Sinkhorn of glm5next_hc_finish block 0, on warp 0. Lane l holds comb
// element (i, j) = (l / HC, l % HC); lanes at or past HC * HC hold a copy of lane l % (HC * HC)
// so every shuffle runs on the full warp. Each sum starts where the unfused one does and adds
// the same operands in the same order; every division stays a division.
template <int HC>
__device__ __forceinline__ void glm_hcf_sinkhorn(
    const float* s_mix,
    const float* __restrict__ hc_scale,
    const float* __restrict__ hc_base,
    float* __restrict__ post_out,
    float* __restrict__ comb_out,
    unsigned int t,
    unsigned int sinkhorn_iters,
    float hc_eps
) {
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int l = lane % (HC * HC);
    const unsigned int i = l / HC;
    const unsigned int j = l % HC;

    if (lane < HC) {
        float po = s_mix[HC + lane] * hc_scale[1] + hc_base[HC + lane];
        post_out[(size_t)t * HC + lane] = 2.f * (1.f / (1.f + expf(-po)));
    }

    float v = s_mix[2 * HC + l] * hc_scale[2] + hc_base[2 * HC + l];

    float mx = -1e30f;
#pragma unroll
    for (int jj = 0; jj < HC; ++jj) mx = fmaxf(mx, __shfl_sync(0xFFFFFFFFu, v, i * HC + jj));
    const float e = expf(v - mx);
    float sum = 0.f;
#pragma unroll
    for (int jj = 0; jj < HC; ++jj) sum += __shfl_sync(0xFFFFFFFFu, e, i * HC + jj);
    v = e / sum + hc_eps;

    float c = hc_eps;
#pragma unroll
    for (int ii = 0; ii < HC; ++ii) c += __shfl_sync(0xFFFFFFFFu, v, ii * HC + j);
    v /= c;

    for (unsigned int it = 0; it + 1 < sinkhorn_iters; ++it) {
        float r = hc_eps;
#pragma unroll
        for (int jj = 0; jj < HC; ++jj) r += __shfl_sync(0xFFFFFFFFu, v, i * HC + jj);
        v /= r;
        float cc = hc_eps;
#pragma unroll
        for (int ii = 0; ii < HC; ++ii) cc += __shfl_sync(0xFFFFFFFFu, v, ii * HC + j);
        v /= cc;
    }
    if (lane < HC * HC) comb_out[(size_t)t * HC * HC + lane] = v;
}

// 2026-10-10: CG reads the streams through L2 (__ldcg), for a caller whose other blocks wrote
// them in this launch (glm5next_hc_post_pre_norm); the values read are the same.
// 2026-10-09: y and its RMS norm on warps 1..7 (u = threadIdx.x - 32). y[d] is
// glm5next_hc_finish's collapse; the norm is rms_norm_vanilla over the BF16 y with
// NT = min(H, 1024) virtual threads: virtual thread v sums pairs v, v + NT, .. of y as
// `s += a * a + b * b`, each virtual warp of 32 runs the same butterfly on one real warp, and
// the warp sums go through the same final butterfly, zeros past the last warp included.
template <int HC, bool CG = false>
__device__ __forceinline__ void glm_hcf_collapse_norm(
    const float* __restrict__ x,
    const float* s_pre,
    const __nv_bfloat16* __restrict__ norm_w,
    __nv_bfloat16* __restrict__ y_out,
    __nv_bfloat16* __restrict__ normed_out,
    __nv_bfloat16* s_y,
    float* s_ws,
    unsigned int H,
    float rms_eps
) {
    const unsigned int u = threadIdx.x - 32u;
    const unsigned int rw = u >> 5;
    const unsigned int lane = threadIdx.x & 31u;

    float pre[HC];
#pragma unroll
    for (int i = 0; i < HC; ++i) pre[i] = s_pre[i];
    // 2026-10-09: Four columns per thread and pass through 16-byte loads: one pass per
    // GLM_HCF_TAIL_THREADS quads keeps the single tail block from waiting on one L2 round trip
    // per column (measured 3.3 us scalar, 1.2 us here at H 4096). Each column's sum is still
    // acc = 0 + pre[0] x[0] + .. + pre[HC-1] x[HC-1].
    const unsigned int nq = H / 4u;
    const float4* x4 = reinterpret_cast<const float4*>(x);
    for (unsigned int q = u; q < nq; q += GLM_HCF_TAIL_THREADS) {
        float4 v[HC];
#pragma unroll
        for (int i = 0; i < HC; ++i) v[i] = CG ? __ldcg(&x4[i * nq + q]) : x4[i * nq + q];
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll
        for (int i = 0; i < HC; ++i) {
            a0 += pre[i] * v[i].x;
            a1 += pre[i] * v[i].y;
            a2 += pre[i] * v[i].z;
            a3 += pre[i] * v[i].w;
        }
        const __nv_bfloat162 lo = __halves2bfloat162(__float2bfloat16(a0), __float2bfloat16(a1));
        const __nv_bfloat162 hi = __halves2bfloat162(__float2bfloat16(a2), __float2bfloat16(a3));
        reinterpret_cast<__nv_bfloat162*>(y_out)[2 * q] = lo;
        reinterpret_cast<__nv_bfloat162*>(y_out)[2 * q + 1] = hi;
        reinterpret_cast<__nv_bfloat162*>(s_y)[2 * q] = lo;
        reinterpret_cast<__nv_bfloat162*>(s_y)[2 * q + 1] = hi;
    }
    glm_hcf_tail_bar();

    const unsigned int nt = H < 1024u ? H : 1024u;
    const unsigned int nwv = nt / 32u;
    const unsigned int half = H / 2u;
    for (unsigned int vw = rw; vw < nwv; vw += GLM_HCF_TAIL_THREADS / 32) {
        float s = 0.f;
        for (unsigned int p = vw * 32u + lane; p < half; p += nt) {
            const float a = __bfloat162float(s_y[2 * p]);
            const float b = __bfloat162float(s_y[2 * p + 1]);
            s += a * a + b * b;
        }
        s = glm_hcf_warp_sum(s);
        if (lane == 0) s_ws[vw] = s;
    }
    glm_hcf_tail_bar();
    if (rw == 0) {
        float v = (lane < nwv) ? s_ws[lane] : 0.0f;
        v = glm_hcf_warp_sum(v);
        if (lane == 0) s_ws[32] = v;
    }
    glm_hcf_tail_bar();
    const float rms = rsqrtf(s_ws[32] / (float)H + rms_eps);

    // 2026-10-09: Four columns per thread and pass, the weights through one 8-byte load; each
    // output is rms_norm_vanilla's x * rms * w.
    const uint2* w4 = reinterpret_cast<const uint2*>(norm_w);
    for (unsigned int q = u; q < nq; q += GLM_HCF_TAIL_THREADS) {
        const uint2 wq = w4[q];
        const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(&wq);
        unsigned short o[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float xv = __bfloat162float(s_y[4 * q + e]);
            const float wv = __bfloat162float(wb[e]);
            o[e] = __bfloat16_as_ushort(__float2bfloat16(xv * rms * wv));
        }
        uint2 pk;
        pk.x = (unsigned int)o[0] | ((unsigned int)o[1] << 16);
        pk.y = (unsigned int)o[2] | ((unsigned int)o[3] << 16);
        reinterpret_cast<uint2*>(normed_out)[q] = pk;
    }
}

// 2026-10-09: The block's shared memory, declared once in each entry point: a __shared__
// declared inside the per-HC instantiations would be reserved once per instantiation.
template <int ROWS>
struct GlmHcfSmem {
    float red_ss[GLM_HCF_BLOCK];
    float red[ROWS][GLM_HCF_BLOCK];
    float mix[GLM_HCF_MAX_MIX];
    float pre[GLM_HCF_MAX_MULT];
    float ws[33];
    __nv_bfloat16 y[GLM_HCF_MAX_H];
    bool last;
};

// 2026-10-09: The whole kernel for ROWS mix rows per block (1: glm5next_hc_mix_bf16's grid,
// 8: glm5next_hc_mix_bf16_rows's) and hc_mult HC.
template <int ROWS, int HC>
__device__ __forceinline__ void glm_hcf_body(
    const float* __restrict__ streams,
    const __nv_bfloat16* __restrict__ hc_fn,
    const float* __restrict__ hc_scale,
    const float* __restrict__ hc_base,
    float* __restrict__ mix,
    unsigned int* __restrict__ done,
    const __nv_bfloat16* __restrict__ norm_w,
    __nv_bfloat16* __restrict__ y_out,
    float* __restrict__ post_out,
    float* __restrict__ comb_out,
    __nv_bfloat16* __restrict__ normed_out,
    unsigned int H,
    unsigned int sinkhorn_iters,
    float norm_eps,
    float hc_eps,
    float rms_eps,
    GlmHcfSmem<ROWS>& sm
) {
    constexpr unsigned int MIX = (2 + HC) * HC;
    const unsigned int t = blockIdx.x;
    const unsigned int m0 = blockIdx.y * ROWS;
    const unsigned int tid = threadIdx.x;
    const unsigned int hc_dim = HC * H;
    const unsigned int rows = min((unsigned int)ROWS, MIX - m0);
    const float* x = streams + (size_t)t * hc_dim;

    float* red_ss = sm.red_ss;
    float (*red)[GLM_HCF_BLOCK] = sm.red;
    float* s_mix = sm.mix;
    float* s_pre = sm.pre;

    float ss = 0.f;
    float acc[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) acc[r] = 0.f;
    // 2026-10-09: Eight iterations in flight per thread: at one token the 24 blocks are bound by
    // load latency (whole kernel at C1: 13.4 us at unroll 4, 11.1 us at 8, 16 no better).
#pragma unroll 8
    for (unsigned int k = tid; k < hc_dim; k += GLM_HCF_BLOCK) {
        const float v = x[k];
        ss += v * v;
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            if ((unsigned int)r < rows) {
                acc[r] += __bfloat162float(hc_fn[(size_t)(m0 + r) * hc_dim + k]) * v;
            }
        }
    }
    red_ss[tid] = ss;
#pragma unroll
    for (int r = 0; r < ROWS; ++r) red[r][tid] = acc[r];
    __syncthreads();
    for (unsigned int s = GLM_HCF_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            red_ss[tid] += red_ss[tid + s];
#pragma unroll
            for (int r = 0; r < ROWS; ++r) red[r][tid] += red[r][tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        const float rsq = rsqrtf(red_ss[0] / (float)hc_dim + norm_eps);
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            if ((unsigned int)r < rows) mix[(size_t)t * MIX + m0 + r] = red[r][0] * rsq;
        }
        __threadfence();
        const unsigned int prev = atomicAdd(&done[t], 1u);
        sm.last = (prev == gridDim.y - 1);
    }
    __syncthreads();
    if (!sm.last) return;

    // 2026-10-09: Last block of token t: every other block's mixes are visible after the
    // fence. The counter goes back to zero for the next launch.
    __threadfence();
    if (tid == 0) done[t] = 0;
    if (tid < MIX) s_mix[tid] = __ldcg(&mix[(size_t)t * MIX + tid]);
    __syncthreads();
    if (tid < HC) {
        float pr = s_mix[tid] * hc_scale[0] + hc_base[tid];
        s_pre[tid] = 1.f / (1.f + expf(-pr)) + hc_eps;
    }
    __syncthreads();

    if (tid < 32) {
        glm_hcf_sinkhorn<HC>(s_mix, hc_scale, hc_base, post_out, comb_out, t, sinkhorn_iters,
                             hc_eps);
    } else {
        glm_hcf_collapse_norm<HC>(x, s_pre, norm_w, y_out + (size_t)t * H,
                                  normed_out + (size_t)t * H, sm.y, sm.ws, H, rms_eps);
    }
}

#define GLM_HCF_ARGS                                                                          \
    const float* __restrict__ streams, const __nv_bfloat16* __restrict__ hc_fn,               \
        const float* __restrict__ hc_scale, const float* __restrict__ hc_base,                \
        float* __restrict__ mix, unsigned int* __restrict__ done,                             \
        const __nv_bfloat16* __restrict__ norm_w, __nv_bfloat16* __restrict__ y_out,          \
        float* __restrict__ post_out, float* __restrict__ comb_out,                           \
        __nv_bfloat16* __restrict__ normed_out, const unsigned int hidden_size,               \
        const unsigned int hc_mult, const unsigned int sinkhorn_iters, const float norm_eps,  \
        const float hc_eps, const float rms_eps

#define GLM_HCF_DISPATCH(ROWS)                                                                \
    switch (hc_mult) {                                                                        \
    case 1: glm_hcf_body<ROWS, 1>(GLM_HCF_PASS); break;                                       \
    case 2: glm_hcf_body<ROWS, 2>(GLM_HCF_PASS); break;                                       \
    case 3: glm_hcf_body<ROWS, 3>(GLM_HCF_PASS); break;                                       \
    case 4: glm_hcf_body<ROWS, 4>(GLM_HCF_PASS); break;                                       \
    default: break;                                                                           \
    }

#define GLM_HCF_PASS                                                                          \
    streams, hc_fn, hc_scale, hc_base, mix, done, norm_w, y_out, post_out, comb_out,          \
        normed_out, hidden_size, sinkhorn_iters, norm_eps, hc_eps, rms_eps, sm

// 2026-10-09: One block per (token, mix row): grid (T, (2 + hc) * hc, 1).
extern "C" __global__ void __launch_bounds__(GLM_HCF_BLOCK)
glm5next_hc_pre_norm(GLM_HCF_ARGS) {
    __shared__ GlmHcfSmem<1> sm;
    GLM_HCF_DISPATCH(1)
}

// 2026-10-09: One block per (token, 8 mix rows): grid (T, ceil((2 + hc) * hc / 8), 1).
extern "C" __global__ void __launch_bounds__(GLM_HCF_BLOCK)
glm5next_hc_pre_norm_rows(GLM_HCF_ARGS) {
    __shared__ GlmHcfSmem<8> sm;
    GLM_HCF_DISPATCH(8)
}


// 2026-10-10: glm5next_hc_post_pre_norm: one token's mHC post-mix of the previous site (with the
// one-shot all-reduce's sum in front when `n_ranks` > 0) and the next site's glm5next_hc_pre_norm,
// in one launch. Unfused, the decode runs bf16_add_rank_sum (or nothing, when the sum is already
// in `own`), glm5next_hc_post (streams updated in place) and glm5next_hc_pre_norm.
//
// Each block (row m) computes the post-mixed streams its 256 threads sum, in registers: thread
// tid owns k = tid + 256 jj, i.e. column d = tid + 256 c of every stream, and for each such d
// forms x[d] with bf16_add_rank_sum's arithmetic (FP32 sum in rank order from 0.0f, one BF16
// rounding; or `own` as it is) and the hc new values with glm5next_hc_post's (post[j] * x then
// + comb[i][j] * residual[i][d], i ascending). It then runs glm5next_hc_pre_norm's mix on them in
// its order (k ascending: stream-major, column-minor) and its trees, tail and norm, the tail
// reading the new streams back through L2. Block row 0 also stores the new streams to
// `streams_out`, which must not alias `residual` (every block reads `residual` while row 0 writes),
// before its arrival, so the tail sees them. `post`/`comb` may be the buffers the tail writes:
// every block reads them before it arrives. The summed x is not stored back to `own`.
//
// Invariants: one token, grid (1, mix_hc, 1), block GLM_HCF_BLOCK; H a multiple of 256 and at
// most GLM_HCF_POST_FOLD_MAX_H (otherwise nothing is written); hc_mult <= 4; `peers` holds peer r
// in slot (r < my_rank ? r : r - 1) of `slot_elems` elements; `done` as for glm5next_hc_pre_norm.
#define GLM_HCF_POST_FOLD_MAX_H 4096

template <int HC>
__device__ __forceinline__ void glm_hcf_post_pre_body(
    const __nv_bfloat16* __restrict__ own,
    const __nv_bfloat16* __restrict__ peers,
    long long slot_elems,
    int n_ranks,
    int my_rank,
    const float* __restrict__ residual,
    const float* __restrict__ p_post,
    const float* __restrict__ p_comb,
    float* __restrict__ streams_out,
    const __nv_bfloat16* __restrict__ hc_fn,
    const float* __restrict__ hc_scale,
    const float* __restrict__ hc_base,
    float* __restrict__ mix,
    unsigned int* __restrict__ done,
    const __nv_bfloat16* __restrict__ norm_w,
    __nv_bfloat16* __restrict__ y_out,
    float* __restrict__ post_out,
    float* __restrict__ comb_out,
    __nv_bfloat16* __restrict__ normed_out,
    unsigned int H,
    unsigned int sinkhorn_iters,
    float norm_eps,
    float hc_eps,
    float rms_eps,
    GlmHcfSmem<1>& sm
) {
    constexpr unsigned int MIX = (2 + HC) * HC;
    constexpr unsigned int MAXC = GLM_HCF_POST_FOLD_MAX_H / GLM_HCF_BLOCK;
    const unsigned int m = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int hc_dim = HC * H;
    const unsigned int cols = H / GLM_HCF_BLOCK;
    __shared__ float s_p[GLM_HCF_MAX_MULT];
    __shared__ float s_c[GLM_HCF_MAX_MULT * GLM_HCF_MAX_MULT];
    if (tid < HC) s_p[tid] = p_post[tid];
    if (tid < HC * HC) s_c[tid] = p_comb[tid];
    __syncthreads();

    float nv[HC][MAXC];
#pragma unroll
    for (unsigned int c = 0; c < MAXC; ++c) {
        if (c >= cols) break;
        const unsigned int d = tid + GLM_HCF_BLOCK * c;
        float xd;
        if (n_ranks > 0) {
            float acc = 0.0f;
            for (int r = 0; r < n_ranks; ++r) {
                const __nv_bfloat16 v =
                    (r == my_rank) ? own[d]
                                   : peers[(long long)(r < my_rank ? r : r - 1) * slot_elems + d];
                acc += __bfloat162float(v);
            }
            xd = (float)__float2bfloat16_rn(acc);
        } else {
            xd = (float)own[d];
        }
        float rv[HC];
#pragma unroll
        for (int i = 0; i < HC; ++i) rv[i] = residual[i * H + d];
#pragma unroll
        for (int j = 0; j < HC; ++j) {
            float a = s_p[j] * xd;
#pragma unroll
            for (int i = 0; i < HC; ++i) a += s_c[i * HC + j] * rv[i];
            nv[j][c] = a;
            if (m == 0) streams_out[j * H + d] = a;
        }
    }

    float ss = 0.f;
    float acc = 0.f;
    const __nv_bfloat16* fn = hc_fn + (size_t)m * hc_dim;
#pragma unroll
    for (int j = 0; j < HC; ++j) {
#pragma unroll
        for (unsigned int c = 0; c < MAXC; ++c) {
            if (c >= cols) break;
            const float v = nv[j][c];
            ss += v * v;
            acc += __bfloat162float(fn[j * H + tid + GLM_HCF_BLOCK * c]) * v;
        }
    }
    sm.red_ss[tid] = ss;
    sm.red[0][tid] = acc;
    __syncthreads();
    for (unsigned int s = GLM_HCF_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sm.red_ss[tid] += sm.red_ss[tid + s];
            sm.red[0][tid] += sm.red[0][tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        const float rsq = rsqrtf(sm.red_ss[0] / (float)hc_dim + norm_eps);
        mix[m] = sm.red[0][0] * rsq;
        __threadfence();
        const unsigned int prev = atomicAdd(&done[0], 1u);
        sm.last = (prev == gridDim.y - 1);
    }
    __syncthreads();
    if (!sm.last) return;

    __threadfence();
    if (tid == 0) done[0] = 0;
    if (tid < MIX) sm.mix[tid] = __ldcg(&mix[tid]);
    __syncthreads();
    if (tid < HC) {
        float pr = sm.mix[tid] * hc_scale[0] + hc_base[tid];
        sm.pre[tid] = 1.f / (1.f + expf(-pr)) + hc_eps;
    }
    __syncthreads();
    if (tid < 32) {
        glm_hcf_sinkhorn<HC>(sm.mix, hc_scale, hc_base, post_out, comb_out, 0, sinkhorn_iters,
                             hc_eps);
    } else {
        glm_hcf_collapse_norm<HC, true>(streams_out, sm.pre, norm_w, y_out, normed_out, sm.y,
                                        sm.ws, H, rms_eps);
    }
}

extern "C" __global__ void __launch_bounds__(GLM_HCF_BLOCK) glm5next_hc_post_pre_norm(
    const __nv_bfloat16* __restrict__ own, const __nv_bfloat16* __restrict__ peers,
    long long slot_elems, int n_ranks, int my_rank, const float* __restrict__ residual,
    const float* __restrict__ p_post, const float* __restrict__ p_comb,
    float* __restrict__ streams_out, const __nv_bfloat16* __restrict__ hc_fn,
    const float* __restrict__ hc_scale, const float* __restrict__ hc_base,
    float* __restrict__ mix, unsigned int* __restrict__ done,
    const __nv_bfloat16* __restrict__ norm_w, __nv_bfloat16* __restrict__ y_out,
    float* __restrict__ post_out, float* __restrict__ comb_out,
    __nv_bfloat16* __restrict__ normed_out, const unsigned int hidden_size,
    const unsigned int hc_mult, const unsigned int sinkhorn_iters, const float norm_eps,
    const float hc_eps, const float rms_eps) {
    __shared__ GlmHcfSmem<1> sm;
    if (hidden_size % GLM_HCF_BLOCK != 0 || hidden_size > GLM_HCF_POST_FOLD_MAX_H) return;
#define GLM_HCF_PP_PASS                                                                       \
    own, peers, slot_elems, n_ranks, my_rank, residual, p_post, p_comb, streams_out, hc_fn,   \
        hc_scale, hc_base, mix, done, norm_w, y_out, post_out, comb_out, normed_out,          \
        hidden_size, sinkhorn_iters, norm_eps, hc_eps, rms_eps, sm
    switch (hc_mult) {
    case 1: glm_hcf_post_pre_body<1>(GLM_HCF_PP_PASS); break;
    case 2: glm_hcf_post_pre_body<2>(GLM_HCF_PP_PASS); break;
    case 3: glm_hcf_post_pre_body<3>(GLM_HCF_PP_PASS); break;
    case 4: glm_hcf_post_pre_body<4>(GLM_HCF_PP_PASS); break;
    default: break;
    }
#undef GLM_HCF_PP_PASS
}
