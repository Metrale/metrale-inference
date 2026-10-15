// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-15: GLM-5.3-Flash unabsorbed-MLA prefill, the per-head halves of `kv_b_proj`
// (module `glm5next_mla_head_bmm`, METRALE_GLM_MLA_UNABSORB). For every head h:
//     C[m, h*N + n] = sum_k A[m, h*K + k] * B[h][n][k]
// `_q` (K 256, N 512): A = q_b_proj's output, B[h][c][r] = kv_b_proj[h*512 + r][c], C = the
// latent query the attend kernel reads. `_o` (K 512, N 256): A = the attend kernel's latent
// output, B[h][r][c] = kv_b_proj[h*512 + 256 + r][c], C = the o_proj input.
// Owner: gb10 kernels (glm-5.3-flash); launched by crates/model-arch/src/glm5next_dsa/unabsorb.
// Invariants:
// - BF16 in, FP32 accumulate on mma.sync m16n8k16, round-to-nearest BF16 out.
// - Every output row reads only its own A row; the k order is fixed (32-wide steps, ascending,
//   no split-K, no atomics), so a row's bytes do not depend on M or on where the row sits.
// - Launch: grid (N/128, ceil(M/128), heads), block 256, static shared memory 48 KiB.
//   A, B and C 16-byte aligned (the launcher refuses otherwise). Rows >= M are zero-filled on
//   load and never stored.

#include <cuda_bf16.h>
#include <stdint.h>

#define HB_TM 128
#define HB_TN 128
#define HB_TK 32
#define HB_STAGES 3
#define HB_THREADS 256
// 2026-10-15: one stage = A tile [128][32] + B tile [128][32] BF16 = 16 KiB.
#define HB_STAGE_BYTES (2 * HB_TM * HB_TK * 2)
// 2026-10-15: epilogue rows padded to 272 B (128 BF16 + 8): conflict-free 4-byte writes.
#define HB_C_STRIDE 272

// 2026-10-15: byte offset of 16-byte chunk `c` (0..3) of tile row `r` (64 B per row); the
// chunk index is XORed with (r >> 1) & 3 so eight consecutive rows of one logical chunk land in
// eight distinct 16-byte slots of a 128-byte window (ldmatrix conflict-free).
__device__ __forceinline__ uint32_t hbm_off(int r, int c) {
    return (uint32_t)(r * 64 + ((c ^ ((r >> 1) & 3)) << 4));
}

__device__ __forceinline__ void hbm_cp16(uint32_t dst, const void* src, uint32_t bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(bytes));
}

__device__ __forceinline__ void hbm_ldsm4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2,
                                          uint32_t& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

__device__ __forceinline__ void hbm_mma(float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

template <int K, int N>
__device__ __forceinline__ void hbm_body(const __nv_bfloat16* __restrict__ A,
                                         const __nv_bfloat16* __restrict__ B,
                                         __nv_bfloat16* __restrict__ C, uint32_t M,
                                         uint32_t heads) {
    __shared__ __align__(128) unsigned char smem[HB_STAGES * HB_STAGE_BYTES];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp >> 2, wn = warp & 3;  // 2 (M) x 4 (N) warps, warp tile 64 x 32
    const int n0 = blockIdx.x * HB_TN;
    const int m0 = blockIdx.y * HB_TM;
    const int h = blockIdx.z;
    const size_t lda = (size_t)heads * K, ldc = (size_t)heads * N;
    const __nv_bfloat16* Ah = A + (size_t)h * K;
    const __nv_bfloat16* Bh = B + (size_t)h * N * K + (size_t)n0 * K;
    const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(smem);

    // 2026-10-15: stage s, k step kt: 512 A chunks and 512 B chunks, two of each per thread.
    auto load_stage = [&](int s, int kt) {
        const uint32_t sa = sbase + s * HB_STAGE_BYTES, sb = sa + HB_TM * HB_TK * 2;
        const int k0 = kt * HB_TK;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int i = tid + j * HB_THREADS, r = i >> 2, c = i & 3;
            const uint32_t row = (uint32_t)(m0 + r);
            const bool live = row < M;
            const __nv_bfloat16* src = Ah + (size_t)(live ? row : 0u) * lda + k0 + c * 8;
            hbm_cp16(sa + hbm_off(r, c), src, live ? 16u : 0u);
            hbm_cp16(sb + hbm_off(r, c), Bh + (size_t)r * K + k0 + c * 8, 16u);
        }
    };

    float acc[4][4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.f;

    constexpr int KT = K / HB_TK;
    load_stage(0, 0);
    asm volatile("cp.async.commit_group;\n" ::);
    if (KT > 1) load_stage(1, 1);
    asm volatile("cp.async.commit_group;\n" ::);

    for (int kt = 0; kt < KT; ++kt) {
        asm volatile("cp.async.wait_group 1;\n" ::);
        __syncthreads();
        if (kt + 2 < KT) load_stage((kt + 2) % HB_STAGES, kt + 2);
        asm volatile("cp.async.commit_group;\n" ::);
        const uint32_t sa = sbase + (kt % HB_STAGES) * HB_STAGE_BYTES;
        const uint32_t sb = sa + HB_TM * HB_TK * 2;
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            uint32_t af[4][4];
#pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
                const int r = wm * 64 + mi * 16 + (lane & 15);
                hbm_ldsm4(sa + hbm_off(r, ks * 2 + (lane >> 4)), af[mi][0], af[mi][1], af[mi][2],
                          af[mi][3]);
            }
            uint32_t bf[4][2];
#pragma unroll
            for (int nj = 0; nj < 2; ++nj) {
                const int r = wn * 32 + nj * 16 + (lane & 7) + ((lane >> 4) << 3);
                hbm_ldsm4(sb + hbm_off(r, ks * 2 + ((lane >> 3) & 1)), bf[nj * 2][0],
                          bf[nj * 2][1], bf[nj * 2 + 1][0], bf[nj * 2 + 1][1]);
            }
#pragma unroll
            for (int mi = 0; mi < 4; ++mi)
#pragma unroll
                for (int ni = 0; ni < 4; ++ni) hbm_mma(acc[mi][ni], af[mi], bf[ni][0], bf[ni][1]);
        }
    }
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();

    // 2026-10-15: stage C as BF16 in shared memory, then 16-byte row stores.
#pragma unroll
    for (int mi = 0; mi < 4; ++mi)
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
            const int r = wm * 64 + mi * 16 + (lane >> 2);
            const int c = wn * 32 + ni * 8 + (lane & 3) * 2;
            *reinterpret_cast<__nv_bfloat162*>(smem + r * HB_C_STRIDE + c * 2) =
                __floats2bfloat162_rn(acc[mi][ni][0], acc[mi][ni][1]);
            *reinterpret_cast<__nv_bfloat162*>(smem + (r + 8) * HB_C_STRIDE + c * 2) =
                __floats2bfloat162_rn(acc[mi][ni][2], acc[mi][ni][3]);
        }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int i = tid + j * HB_THREADS, r = i >> 4, c = i & 15;
        const uint32_t row = (uint32_t)(m0 + r);
        if (row < M) {
            *reinterpret_cast<uint4*>(C + (size_t)row * ldc + (size_t)h * N + n0 + c * 8) =
                *reinterpret_cast<const uint4*>(smem + r * HB_C_STRIDE + c * 16);
        }
    }
}

extern "C" __global__ void __launch_bounds__(HB_THREADS, 2)
    glm5next_mla_head_bmm_q(const __nv_bfloat16* __restrict__ A,
                            const __nv_bfloat16* __restrict__ B, __nv_bfloat16* __restrict__ C,
                            uint32_t M, uint32_t heads) {
    hbm_body<256, 512>(A, B, C, M, heads);
}

extern "C" __global__ void __launch_bounds__(HB_THREADS, 2)
    glm5next_mla_head_bmm_o(const __nv_bfloat16* __restrict__ A,
                            const __nv_bfloat16* __restrict__ B, __nv_bfloat16* __restrict__ C,
                            uint32_t M, uint32_t heads) {
    hbm_body<512, 256>(A, B, C, M, heads);
}
