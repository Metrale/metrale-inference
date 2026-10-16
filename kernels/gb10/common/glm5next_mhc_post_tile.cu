// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-15: GLM-5.3 mHC post-mix + next pre-norm in one launch with the streams kept in
// registers (METRALE_GLM_MHC_POST_PRE_TILE, host side in model-arch glm5next_mhc_post_tile.rs):
// NG groups of 256 threads per block, one token per group, hidden 4096 and hc_mult 4 only.
//
// Owner: gb10 kernels (family glm_mhc).
// Invariants:
// - Every output is byte-identical to glm5next_hc_post_rank_sum_vec8 (glm5next_hc_post_sum.cu)
//   followed by glm5next_hc_pre_norm_tile4 (glm5next_mhc_tile.cu) on the same inputs. Thread t of
//   a group owns columns d = cc * 256 + t (cc = 0..15) of its token. The post half is vec8's per
//   element: the FP32 rank sum in rank order from 0.0f, one __float2bfloat16_rn, then
//   s_p[j] * xd + s_c[i * 4 + j] * rv[i] for i = 0..3 (--fmad=false, KERNEL.toml); the four
//   stream values of each column stay in registers. Column d of stream i is tile4's chain step
//   k = i * 4096 + d = (i * 16 + cc) * 256 + t, so thread t runs tile4's thread-t chains in tile4's
//   order (ss += v * v, then acc[r] += w * v) from its registers, with only hc_fn through the
//   cp.async ring; then tile4's tree (glm_hct_tree), pre (glm_hct_pre), Sinkhorn (warp 0) and
//   collapse (sum of pre[i] * x[i] from 0, i ascending, rounded once) + RMS norm (warps 1..7,
//   glm_hcf_norm_at). examples/glm5next_hc_post_pre_tile_gate.rs checks every output byte.
// - Aliasing (the SP prefill call): `streams_out` may be `residual` and `post_out`/`comb_out`
//   may be `post`/`comb`: a stream element is read and then written by one thread, and a
//   token's post/comb rows are read into shared memory before its group writes them.
// - write_x != 0 stores the summed row to block_out as vec8 does; 0 leaves this rank's partial.
// - n_ranks 1..GLM_HCPT_MAX_RANKS; hidden_size GLM_HCPT_H (traps otherwise); hc_fn 16-byte
//   aligned (cp.async; every other operand is read per element); blockDim NG * 256; grid
//   (ceil(T / NG), 1, 1); dynamic shared memory GlmHcptLayout<NG>::BYTES
//   (glm5next_mhc_post_tile.rs computes the same). The host refuses every other shape first.

#include "e4m3_mma_pipe.cuh"
#include "glm5next_mhc_fused_parts.cuh"
#include "glm5next_mhc_tile_parts.cuh"
#include "glm5next_mhc_post_args.cuh"

#define GLM_HCPT_STEPS (GLM_HCT_HC * GLM_HCPT_COLS)
#define GLM_HCPT_STAGES 4
#define GLM_HCPT_CB 4

static_assert(GLM_HCPT_COLS % GLM_HCPT_STAGES == 0, "stage index of chain step must be static");
static_assert(GLM_HCPT_COLS % GLM_HCPT_CB == 0, "post chunks must tile the columns");

// 2026-10-15: Shared memory, in bytes from the dynamic base: the hc_fn ring (stage s: columns
// [MIX][256] BF16), which the trees then reuse as red[NG][MIX + 1][256], each group's BF16 y
// inside its own red slice; then mix[NG][MIX], pre[NG][HC], ws[NG][33], p[NG][HC], c[NG][HC*HC].
template <int NG>
struct GlmHcptLayout {
    static constexpr unsigned int STAGE = GLM_HCT_MIX * GLM_HCF_BLOCK * 2;
    static constexpr unsigned int RING = GLM_HCPT_STAGES * STAGE;
    static constexpr unsigned int RED_G = (GLM_HCT_MIX + 1) * GLM_HCF_BLOCK * 4;
    static constexpr unsigned int RED = NG * RED_G;
    static constexpr unsigned int REUSE = RING > RED ? RING : RED;
    static constexpr unsigned int MIX_OFF = REUSE;
    static constexpr unsigned int PRE_OFF = MIX_OFF + NG * GLM_HCT_MIX * 4;
    static constexpr unsigned int WS_OFF = PRE_OFF + NG * GLM_HCT_HC * 4;
    static constexpr unsigned int P_OFF = WS_OFF + NG * 33 * 4;
    static constexpr unsigned int C_OFF = P_OFF + NG * GLM_HCT_HC * 4;
    static constexpr unsigned int BYTES = C_OFF + NG * GLM_HCT_HC * GLM_HCT_HC * 4;
};
static_assert(GlmHcptLayout<1>::RED_G >= GLM_HCPT_H * 2, "y must fit in its group's red slice");

// 2026-10-15: One ring stage: hc_fn columns base .. base + 255 of every mix row.
template <int NG>
__device__ __forceinline__ void glm_hcpt_issue(unsigned char* stage,
                                               const __nv_bfloat16* hc_fn, unsigned int base) {
    constexpr unsigned int NT = NG * GLM_HCF_BLOCK;
    constexpr unsigned int PIECES = GLM_HCT_MIX * GLM_HCF_BLOCK / 8;
    constexpr unsigned int HC_DIM = GLM_HCT_HC * GLM_HCPT_H;
    const unsigned int sw = e4m3g::smem_addr(stage);
    for (unsigned int i = threadIdx.x; i < PIECES; i += NT) {
        const unsigned int r = i / (GLM_HCF_BLOCK / 8), c = (i % (GLM_HCF_BLOCK / 8)) * 8;
        e4m3g::cp16(sw + (r * GLM_HCF_BLOCK + c) * 2, hc_fn + (size_t)r * HC_DIM + base + c, true);
    }
    e4m3g::cp_commit();
}

// 2026-10-15: vec8's post for token tk's columns cc * 256 + t: x[i][cc] gets stream i.
__device__ __forceinline__ void glm_hcpt_post(float (&x)[GLM_HCT_HC][GLM_HCPT_COLS],
                                              const unsigned int tk, const unsigned int t,
                                              const float* s_p, const float* s_c,
                                              GLM_HCPT_ARGS) {
    constexpr unsigned int H = GLM_HCPT_H;
    const float* res = residual + (size_t)tk * GLM_HCT_HC * H;
    float* o = streams_out + (size_t)tk * GLM_HCT_HC * H;
#pragma unroll
    for (int cb = 0; cb < GLM_HCPT_COLS; cb += GLM_HCPT_CB) {
        __nv_bfloat16 v[GLM_HCPT_CB][GLM_HCPT_MAX_RANKS];
        float rv[GLM_HCPT_CB][GLM_HCT_HC];
#pragma unroll
        for (int k = 0; k < GLM_HCPT_CB; ++k) {
            const unsigned int d = (cb + k) * GLM_HCF_BLOCK + t;
            const long long e = (long long)tk * H + d;
#pragma unroll
            for (int r = 0; r < GLM_HCPT_MAX_RANKS; ++r)
                if (r < n_ranks)
                    v[k][r] = (r == my_rank)
                        ? block_out[e]
                        : peers[(long long)(r < my_rank ? r : r - 1) * slot_elems + e];
#pragma unroll
            for (int i = 0; i < GLM_HCT_HC; ++i) rv[k][i] = res[(size_t)i * H + d];
        }
#pragma unroll
        for (int k = 0; k < GLM_HCPT_CB; ++k) {
            const unsigned int d = (cb + k) * GLM_HCF_BLOCK + t;
            float acc = 0.0f;
#pragma unroll
            for (int r = 0; r < GLM_HCPT_MAX_RANKS; ++r)
                if (r < n_ranks) acc += __bfloat162float(v[k][r]);
            const __nv_bfloat16 xb = __float2bfloat16_rn(acc);
            if (write_x) block_out[(size_t)tk * H + d] = xb;
            const float xd = (float)xb;
#pragma unroll
            for (int j = 0; j < GLM_HCT_HC; ++j) {
                float a = s_p[j] * xd;
#pragma unroll
                for (int i = 0; i < GLM_HCT_HC; ++i) a += s_c[i * GLM_HCT_HC + j] * rv[k][i];
                x[j][cb + k] = a;
                o[(size_t)j * H + d] = a;
            }
        }
    }
}

// 2026-10-15: Group G's token after the chains: tile4's tree and pre, then y from the registers
// and tile4's tail (Sinkhorn on warp 0, RMS norm on warps 1..7).
template <int NG, int G>
__device__ __forceinline__ void glm_hcpt_finish(const float ss, const float (&acc)[GLM_HCT_MIX],
                                                const float (&x)[GLM_HCT_HC][GLM_HCPT_COLS],
                                                unsigned char* smem, const unsigned int t,
                                                const unsigned int tk, GLM_HCPT_ARGS) {
    using L = GlmHcptLayout<NG>;
    constexpr unsigned int H = GLM_HCPT_H;
    float (*red)[GLM_HCF_BLOCK] =
        reinterpret_cast<float (*)[GLM_HCF_BLOCK]>(smem + G * L::RED_G);
    __nv_bfloat16* s_y = reinterpret_cast<__nv_bfloat16*>(smem + G * L::RED_G);
    float* s_mix = reinterpret_cast<float*>(smem + L::MIX_OFF) + G * GLM_HCT_MIX;
    float* s_pre = reinterpret_cast<float*>(smem + L::PRE_OFF) + G * GLM_HCT_HC;
    float* s_ws = reinterpret_cast<float*>(smem + L::WS_OFF) + G * 33;

    glm_hct_tree<G>(ss, acc, red, s_mix, mix, tk, true, t, GLM_HCT_HC * H, norm_eps);
    glm_hct_pre(s_mix, s_pre, t, hc_scale, hc_base, hc_eps);
    glm_hct_group_bar<G>();

    // 2026-10-15: y overlaps the group's red slice, whose last read precedes the tree's final
    // barrier.
    float pre[GLM_HCT_HC];
#pragma unroll
    for (int i = 0; i < GLM_HCT_HC; ++i) pre[i] = s_pre[i];
#pragma unroll
    for (int cc = 0; cc < GLM_HCPT_COLS; ++cc) {
        float a = 0.f;
#pragma unroll
        for (int i = 0; i < GLM_HCT_HC; ++i) a += pre[i] * x[i][cc];
        const __nv_bfloat16 yb = __float2bfloat16(a);
        const unsigned int d = cc * GLM_HCF_BLOCK + t;
        y_out[(size_t)tk * H + d] = yb;
        s_y[d] = yb;
    }
    glm_hct_group_bar<G>();
    if (t < 32) {
        glm_hcf_sinkhorn<GLM_HCT_HC>(s_mix, hc_scale, hc_base, post_out, comb_out, tk,
                                     sinkhorn_iters, hc_eps);
    } else {
        glm_hcf_norm_at<4 + G>(t - 32u, norm_w, normed_out + (size_t)tk * H, s_y, s_ws, H,
                               rms_eps);
    }
}

template <int NG>
__device__ __forceinline__ void glm_hcpt_body(GLM_HCPT_ARGS) {
    using L = GlmHcptLayout<NG>;
    constexpr unsigned int MIX = GLM_HCT_MIX;
    constexpr int ST = GLM_HCPT_STAGES;
    constexpr int COLS = GLM_HCPT_COLS;
    extern __shared__ __align__(16) unsigned char glm_hcpt_smem[];
    const unsigned int g = threadIdx.x / GLM_HCF_BLOCK;
    const unsigned int t = threadIdx.x % GLM_HCF_BLOCK;
    // 2026-10-15: A launch outside the layout's shape traps (fail fast); the host refuses first.
    unsigned int dyn_smem;
    asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(dyn_smem));
    if (dyn_smem < L::BYTES || hidden_size != GLM_HCPT_H || n_ranks < 1 ||
        n_ranks > GLM_HCPT_MAX_RANKS)
        __trap();
    const unsigned int tk = blockIdx.x * NG + g;
    const bool valid = tk < num_tokens;
    float* s_p = reinterpret_cast<float*>(glm_hcpt_smem + L::P_OFF) + g * GLM_HCT_HC;
    float* s_c = reinterpret_cast<float*>(glm_hcpt_smem + L::C_OFF) + g * GLM_HCT_HC * GLM_HCT_HC;

#pragma unroll
    for (int s = 0; s < ST - 1; ++s)
        glm_hcpt_issue<NG>(glm_hcpt_smem + s * L::STAGE, hc_fn, s * GLM_HCF_BLOCK);
    if (valid && t < GLM_HCT_HC) s_p[t] = post[(size_t)tk * GLM_HCT_HC + t];
    if (valid && t < GLM_HCT_HC * GLM_HCT_HC)
        s_c[t] = comb[(size_t)tk * GLM_HCT_HC * GLM_HCT_HC + t];
    __syncthreads();

    float x[GLM_HCT_HC][COLS];
    if (valid) {
        glm_hcpt_post(x, tk, t, s_p, s_c, GLM_HCPT_PASS);
    } else {
#pragma unroll
        for (int i = 0; i < GLM_HCT_HC; ++i)
#pragma unroll
            for (int cc = 0; cc < COLS; ++cc) x[i][cc] = 0.f;
    }

    // 2026-10-15: tile4's thread-t chains, step i * COLS + cc reading x[i][cc]; x rotates one
    // stream per pass so the unrolled body reads x[0], and is back in place after HC passes.
    float ss = 0.f;
    float acc[MIX];
#pragma unroll
    for (int r = 0; r < (int)MIX; ++r) acc[r] = 0.f;
#pragma unroll 1
    for (int i = 0; i < GLM_HCT_HC; ++i) {
#pragma unroll
        for (int cc = 0; cc < COLS; ++cc) {
            const unsigned int c = i * COLS + cc;
            e4m3g::cp_wait<ST - 2>();
            __syncthreads();
            // 2026-10-15: Step c + ST - 1 refills the stage step c - 1 used; every thread
            // finished reading it before the barrier above.
            const unsigned int n = c + ST - 1;
            if (n < GLM_HCPT_STEPS)
                glm_hcpt_issue<NG>(glm_hcpt_smem + ((cc + ST - 1) % ST) * L::STAGE, hc_fn,
                                   n * GLM_HCF_BLOCK);
            else
                e4m3g::cp_commit();
            const __nv_bfloat16* sw =
                reinterpret_cast<const __nv_bfloat16*>(glm_hcpt_smem + (cc % ST) * L::STAGE);
            const float v = x[0][cc];
            ss += v * v;
#pragma unroll
            for (int r = 0; r < (int)MIX; ++r) {
                const float w = __bfloat162float(sw[r * GLM_HCF_BLOCK + t]);
                acc[r] += w * v;
            }
        }
#pragma unroll
        for (int cc = 0; cc < COLS; ++cc) {
            const float x0 = x[0][cc];
#pragma unroll
            for (int q = 0; q < GLM_HCT_HC - 1; ++q) x[q][cc] = x[q + 1][cc];
            x[GLM_HCT_HC - 1][cc] = x0;
        }
    }
    e4m3g::cp_wait<0>();
    __syncthreads();
    if (!valid) return;

    if (g == 0) {
        glm_hcpt_finish<NG, 0>(ss, acc, x, glm_hcpt_smem, t, tk, GLM_HCPT_PASS);
    } else if constexpr (NG > 1) {
        glm_hcpt_finish<NG, 1>(ss, acc, x, glm_hcpt_smem, t, tk, GLM_HCPT_PASS);
    }
}

// 2026-10-15: One token per block, two blocks per SM.
extern "C" __global__ void __launch_bounds__(GLM_HCF_BLOCK, 2)
glm5next_hc_post_pre_tile1(GLM_HCPT_ARGS) {
    glm_hcpt_body<1>(GLM_HCPT_PASS);
}

// 2026-10-15: Two tokens per block (one per group) sharing the hc_fn ring, one block per SM.
extern "C" __global__ void __launch_bounds__(2 * GLM_HCF_BLOCK, 1)
glm5next_hc_post_pre_tile2(GLM_HCPT_ARGS) {
    glm_hcpt_body<2>(GLM_HCPT_PASS);
}
