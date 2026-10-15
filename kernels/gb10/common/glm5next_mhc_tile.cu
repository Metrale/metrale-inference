// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-15: GLM-5.3 mHC pre-norm for wide prefill sub-chunks through a cp.async ring
// (METRALE_GLM_MHC_PRE_TILE, host side in model-arch glm5next_mhc_tile.rs): NG groups of 256
// threads per block, four tokens per group, hc_mult 4 only.
//
// Owner: gb10 kernels (family glm_mhc).
// Invariants:
// - Every output is byte-identical to glm5next_hc_pre_norm_tok4 (glm5next_mhc_tok.cu), which is
//   byte-identical to glm5next_hc_pre_norm: thread t (0..255) of group g keeps tok4's thread-t
//   chains for the group's four tokens (k = t, t + 256, .. ascending; ss += v * v, then for each
//   mix row acc += w * v), the RMS and mix sums go through the same 256-wide tree token by token,
//   and each token's tail is tok4's (pre, Sinkhorn on the group's warp 0, collapse + norm on its
//   warps 1..7). Only where the operands wait changes: the hc_fn and stream columns of each
//   256-wide k step come through shared memory, GLM_HCT_STAGES - 1 steps ahead, in 16-byte
//   cp.async copies; a token past the end reads zeros as tok4's does. examples/
//   glm5next_mhc_tile_gate.rs checks every output byte against tok4.
// - blockDim NG * 256; hc_mult 4; H a multiple of 64, at most GLM_HCF_MAX_H; `streams` and
//   `hc_fn` 16-byte aligned; grid (ceil(T / (4 NG)), 1, 1); dynamic shared memory
//   glm_hct_smem_bytes(NG, H) (glm5next_mhc_tile.rs computes the same sum).
// - `done` is not used; `mix` gets tok4's values.

#include "e4m3_mma_pipe.cuh"
#include "glm5next_mhc_fused_parts.cuh"

#ifndef GLM_HCT_STAGES
#define GLM_HCT_STAGES 3
#endif

#define GLM_HCT_HC 4
#define GLM_HCT_MIX ((2 + GLM_HCT_HC) * GLM_HCT_HC)
#define GLM_HCT_TOK 4

// 2026-10-15: Shared memory, in bytes from the dynamic base: the ring (stage s: hc_fn columns
// [MIX][256] BF16, then the block's stream columns [4 NG][256] FP32), which the trees then reuse
// as red[NG][MIX + 1][256] and the tails as y[NG][H]; then mix[4 NG][MIX], pre[NG][HC] and
// ws[NG][33] (FP32).
template <int NG>
struct GlmHctLayout {
    static constexpr unsigned int W_BYTES = GLM_HCT_MIX * GLM_HCF_BLOCK * 2;
    static constexpr unsigned int V_BYTES = GLM_HCT_TOK * NG * GLM_HCF_BLOCK * 4;
    static constexpr unsigned int STAGE = W_BYTES + V_BYTES;
    static constexpr unsigned int RING = GLM_HCT_STAGES * STAGE;
    static constexpr unsigned int RED = NG * (GLM_HCT_MIX + 1) * GLM_HCF_BLOCK * 4;
    static constexpr unsigned int Y = NG * GLM_HCF_MAX_H * 2;
    static constexpr unsigned int A = RING > RED ? RING : RED;
    static constexpr unsigned int REUSE = A > Y ? A : Y;
    static constexpr unsigned int MIX_OFF = REUSE;
    static constexpr unsigned int PRE_OFF = MIX_OFF + GLM_HCT_TOK * NG * GLM_HCT_MIX * 4;
    static constexpr unsigned int WS_OFF = PRE_OFF + NG * GLM_HCT_HC * 4;
    static constexpr unsigned int BYTES = WS_OFF + NG * 33 * 4;
};

// 2026-10-15: Named barrier over the 256 threads of group G (barrier 2 + G); the tails of the
// group's warps 1..7 use barrier 4 + G (glm_hcf_collapse_norm_at).
template <int G>
__device__ __forceinline__ void glm_hct_group_bar() {
    asm volatile("bar.sync %0, %1;" ::"n"(2 + G), "n"(GLM_HCF_BLOCK));
}

// 2026-10-15: One ring stage: the hc_fn columns base .. base + 255 of every mix row and the same
// stream columns of the block's tokens; a token at or past `ntb` gets zeros (no read).
template <int NG>
__device__ __forceinline__ void glm_hct_issue(unsigned char* stage, const float* xb,
                                              const float* streams,
                                              const __nv_bfloat16* hc_fn, unsigned int hc_dim,
                                              unsigned int base, unsigned int ntb) {
    constexpr unsigned int NT = NG * GLM_HCF_BLOCK;
    constexpr unsigned int W_PIECES = GLM_HCT_MIX * GLM_HCF_BLOCK / 8;
    constexpr unsigned int V_PIECES = GLM_HCT_TOK * NG * GLM_HCF_BLOCK / 4;
    const unsigned int sw = e4m3g::smem_addr(stage);
    const unsigned int sv = sw + GlmHctLayout<NG>::W_BYTES;
    for (unsigned int i = threadIdx.x; i < W_PIECES; i += NT) {
        const unsigned int r = i / (GLM_HCF_BLOCK / 8), c = (i % (GLM_HCF_BLOCK / 8)) * 8;
        e4m3g::cp16(sw + (r * GLM_HCF_BLOCK + c) * 2, hc_fn + (size_t)r * hc_dim + base + c, true);
    }
    for (unsigned int i = threadIdx.x; i < V_PIECES; i += NT) {
        const unsigned int q = i / (GLM_HCF_BLOCK / 4), c = (i % (GLM_HCF_BLOCK / 4)) * 4;
        const bool ok = q < ntb;
        e4m3g::cp16(sv + (q * GLM_HCF_BLOCK + c) * 4,
                    ok ? (const void*)(xb + (size_t)q * hc_dim + base + c) : (const void*)streams,
                    ok);
    }
    e4m3g::cp_commit();
}

// 2026-10-15: Group G's four tokens: the trees, then the tails. `acc` and `ss` are thread t's
// chains (tok4's names); red, mix, pre, ws and y are the group's slices.
template <int NG, int G>
__device__ __forceinline__ void glm_hct_group_finish(
    float (&ss)[GLM_HCT_TOK], float (&acc)[GLM_HCT_TOK][GLM_HCT_MIX], unsigned char* smem,
    unsigned int t, unsigned int tg0, unsigned int ntg, unsigned int hc_dim,
    const float* xg, GLM_HCF_ARGS) {
    using L = GlmHctLayout<NG>;
    constexpr unsigned int MIX = GLM_HCT_MIX;
    float (*red)[GLM_HCF_BLOCK] = reinterpret_cast<float (*)[GLM_HCF_BLOCK]>(smem) + G * (MIX + 1);
    float (*s_mix)[MIX] =
        reinterpret_cast<float (*)[MIX]>(smem + L::MIX_OFF) + G * GLM_HCT_TOK;
    float* s_pre = reinterpret_cast<float*>(smem + L::PRE_OFF) + G * GLM_HCT_HC;
    float* s_ws = reinterpret_cast<float*>(smem + L::WS_OFF) + G * 33;
    __nv_bfloat16* s_y = reinterpret_cast<__nv_bfloat16*>(smem) + G * GLM_HCF_MAX_H;
    const unsigned int H = hidden_size;

    // 2026-10-15: tok4's trees, token by token, on the group's barrier.
#pragma unroll
    for (int q = 0; q < GLM_HCT_TOK; ++q) {
        red[0][t] = ss[q];
#pragma unroll
        for (int r = 0; r < (int)MIX; ++r) red[1 + r][t] = acc[q][r];
        glm_hct_group_bar<G>();
        for (unsigned int s = GLM_HCF_BLOCK / 2; s > 0; s >>= 1) {
            if (t < s) {
#pragma unroll
                for (int r = 0; r <= (int)MIX; ++r) red[r][t] += red[r][t + s];
            }
            glm_hct_group_bar<G>();
        }
        if (t < MIX) {
            const float rsq = rsqrtf(red[0][0] / (float)hc_dim + norm_eps);
            const float m = red[1 + t][0] * rsq;
            s_mix[q][t] = m;
            if ((unsigned int)q < ntg) mix[(size_t)(tg0 + q) * MIX + t] = m;
        }
        glm_hct_group_bar<G>();
    }

    // 2026-10-15: tok4's tails. y (the group's slice of the reused ring) overlaps `red`, whose
    // last read precedes the barrier above.
    for (unsigned int q = 0; q < ntg; ++q) {
        const unsigned int tk = tg0 + q;
        if (t < GLM_HCT_HC) {
            float pr = s_mix[q][t] * hc_scale[0] + hc_base[t];
            s_pre[t] = 1.f / (1.f + expf(-pr)) + hc_eps;
        }
        glm_hct_group_bar<G>();
        if (t < 32) {
            glm_hcf_sinkhorn<GLM_HCT_HC>(s_mix[q], hc_scale, hc_base, post_out, comb_out, tk,
                                         sinkhorn_iters, hc_eps);
        } else {
            glm_hcf_collapse_norm_at<GLM_HCT_HC, false, 4 + G>(
                t - 32u, xg + (size_t)q * hc_dim, s_pre, norm_w, y_out + (size_t)tk * H,
                normed_out + (size_t)tk * H, s_y, s_ws, H, rms_eps);
        }
        glm_hct_group_bar<G>();
    }
}

#define GLM_HCT_PASS                                                                          \
    streams, hc_fn, hc_scale, hc_base, mix, done, norm_w, y_out, post_out, comb_out,          \
        normed_out, hidden_size, hc_mult, sinkhorn_iters, norm_eps, hc_eps, rms_eps

template <int NG>
__device__ __forceinline__ void glm_hct_body(GLM_HCF_ARGS, const unsigned int num_tokens) {
    using L = GlmHctLayout<NG>;
    constexpr unsigned int MIX = GLM_HCT_MIX;
    constexpr int ST = GLM_HCT_STAGES;
    extern __shared__ __align__(16) unsigned char glm_hct_smem[];
    const unsigned int g = threadIdx.x / GLM_HCF_BLOCK;
    const unsigned int t = threadIdx.x % GLM_HCF_BLOCK;
    // 2026-10-15: A launch with less dynamic shared memory than the layout traps (fail fast).
    unsigned int dyn_smem;
    asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(dyn_smem));
    if (dyn_smem < L::BYTES) __trap();
    const unsigned int t0 = blockIdx.x * (GLM_HCT_TOK * NG);
    if (t0 >= num_tokens) return;
    const unsigned int ntb = min((unsigned int)(GLM_HCT_TOK * NG), num_tokens - t0);
    const unsigned int hc_dim = GLM_HCT_HC * hidden_size;
    const unsigned int steps = hc_dim / GLM_HCF_BLOCK;
    const float* xb = streams + (size_t)t0 * hc_dim;

    float ss[GLM_HCT_TOK];
    float acc[GLM_HCT_TOK][MIX];
#pragma unroll
    for (int q = 0; q < GLM_HCT_TOK; ++q) {
        ss[q] = 0.f;
#pragma unroll
        for (int r = 0; r < (int)MIX; ++r) acc[q][r] = 0.f;
    }

#pragma unroll
    for (int s = 0; s < ST - 1; ++s) {
        if ((unsigned int)s < steps)
            glm_hct_issue<NG>(glm_hct_smem + s * L::STAGE, xb, streams, hc_fn, hc_dim,
                              s * GLM_HCF_BLOCK, ntb);
        else
            e4m3g::cp_commit();
    }
    for (unsigned int c = 0; c < steps; ++c) {
        e4m3g::cp_wait<ST - 2>();
        __syncthreads();
        // 2026-10-15: Step c + ST - 1 refills the stage step c - 1 used; every thread finished
        // reading it before the barrier above.
        const unsigned int n = c + ST - 1;
        if (n < steps)
            glm_hct_issue<NG>(glm_hct_smem + (n % ST) * L::STAGE, xb, streams, hc_fn, hc_dim,
                              n * GLM_HCF_BLOCK, ntb);
        else
            e4m3g::cp_commit();
        const unsigned char* stage = glm_hct_smem + (c % ST) * L::STAGE;
        const __nv_bfloat16* sw = reinterpret_cast<const __nv_bfloat16*>(stage);
        const float* sv = reinterpret_cast<const float*>(stage + L::W_BYTES) +
                          g * GLM_HCT_TOK * GLM_HCF_BLOCK;
        float v[GLM_HCT_TOK];
#pragma unroll
        for (int q = 0; q < GLM_HCT_TOK; ++q) v[q] = sv[q * GLM_HCF_BLOCK + t];
#pragma unroll
        for (int q = 0; q < GLM_HCT_TOK; ++q) ss[q] += v[q] * v[q];
#pragma unroll
        for (int r = 0; r < (int)MIX; ++r) {
            const float w = __bfloat162float(sw[r * GLM_HCF_BLOCK + t]);
#pragma unroll
            for (int q = 0; q < GLM_HCT_TOK; ++q) acc[q][r] += w * v[q];
        }
    }
    e4m3g::cp_wait<0>();
    __syncthreads();

    const unsigned int tg0 = t0 + g * GLM_HCT_TOK;
    const unsigned int ntg =
        ntb > g * GLM_HCT_TOK ? min((unsigned int)GLM_HCT_TOK, ntb - g * GLM_HCT_TOK) : 0u;
    const float* xg = xb + (size_t)(g * GLM_HCT_TOK) * hc_dim;
    if (g == 0) {
        glm_hct_group_finish<NG, 0>(ss, acc, glm_hct_smem, t, tg0, ntg, hc_dim, xg, GLM_HCT_PASS);
    } else if constexpr (NG > 1) {
        glm_hct_group_finish<NG, 1>(ss, acc, glm_hct_smem, t, tg0, ntg, hc_dim, xg, GLM_HCT_PASS);
    }
}

// 2026-10-15: One group (four tokens, 256 threads, two blocks per SM) and two groups (eight
// tokens, 512 threads: each 16-byte hc_fn copy serves eight tokens).
extern "C" __global__ void __launch_bounds__(GLM_HCF_BLOCK, 2)
glm5next_hc_pre_norm_tile4(GLM_HCF_ARGS, const unsigned int num_tokens) {
    glm_hct_body<1>(GLM_HCT_PASS, num_tokens);
}

extern "C" __global__ void __launch_bounds__(2 * GLM_HCF_BLOCK, 1)
glm5next_hc_pre_norm_tile8(GLM_HCF_ARGS, const unsigned int num_tokens) {
    glm_hct_body<2>(GLM_HCT_PASS, num_tokens);
}
