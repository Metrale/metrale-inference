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

#include "glm5next_mhc_tile_ring.cuh"

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

// 2026-10-15: One group: four tokens, 256 threads, GLM_HCT_STAGES 4 (one block per SM). Two
// groups (eight tokens per block, three stages) measured slower on GB10 and are not built.
extern "C" __global__ void __launch_bounds__(GLM_HCF_BLOCK, 2)
glm5next_hc_pre_norm_tile4(GLM_HCF_ARGS, const unsigned int num_tokens) {
    glm_hct_body<1>(GLM_HCT_PASS, num_tokens);
}
