// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-13: `moe_w4a16_grouped_gemm_ptrtable_pipe`: the grouped W4A16 MoE GEMM of
// `moe_w4a16_grouped_gemm_ptrtable` (gb10/common/moe_w4a16_grouped_gemm.cu) for Hopper prefill,
// with the same arguments, grid meaning and output bytes, staged through a cp.async pipeline.
//
// - CTA tile 64 rows x 128 columns (published as `<entry>_n_tile` = 128; grid y counts 64-row M
//   tiles, as the base kernel's does), 256 threads = 8 warps as 2 (M) x 4 (N), warp tile 32 x 32.
// - K advances 64 per stage over STAGES cp.async stages: the A rows (gathered through
//   sorted_token_ids, zero-filled past the expert's rows), the packed E2M1 bytes and the E4M3
//   scale bytes. Each stage's weights are dequantised once into a BF16 [128][64] tile.
// - A staged weight is BF16(E2M1 * E4M3 * scale2), the product taken in FP32 left to right and
//   rounded once to BF16, exactly as the base kernel computes it (E2M1 * E4M3 is exact in BF16 and
//   in FP32, so taking it as a BF16 pair first changes no bit).
// - The MMAs are m16n8k16 BF16 into FP32 accumulators starting at zero, in ascending k16 order,
//   and each output is that FP32 sum rounded to BF16: the base kernel's arithmetic, element for
//   element, so the outputs are byte-identical to it.
//
// Requirements (the host admits only these shapes; the kernel traps on the rest): K % 64 == 0,
// N % 128 == 0, each expert's packed pointer 16-byte aligned and scale pointer 4-byte aligned, and
// at least PIPE_SMEM_BYTES of dynamic shared memory (the launcher's
// MOE_W4A16_PIPE_LAUNCH, crates/model-layers/src/layers/ops/moe_grouped_a.rs).
//
// Owner: hopper kernels.
// Invariants: output bytes equal `moe_w4a16_grouped_gemm_ptrtable`'s for every admitted shape.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "tc_weight_formats.cuh"  // 2026-10-13: ntc_e2m1x4_bf16, ntc_e4m3_bf16x2, ntc_hmul2

#define PIPE_MT 64
#define PIPE_NT 128
#define PIPE_KS 64
#define PIPE_STAGES 4
#define PIPE_THREADS 256
#define PIPE_GROUP 16
// 2026-10-13: Shared-memory rows padded by 8 BF16 (16 bytes): a 144-byte row stride puts the eight
// rows of an ldmatrix (and the eight 16-byte stores of a dequant quarter-warp) in distinct banks.
#define PIPE_LD (PIPE_KS + 8)
#define PIPE_A_BYTES (PIPE_MT * PIPE_LD * 2)
#define PIPE_BP_BYTES (PIPE_NT * PIPE_KS / 2)
#define PIPE_BS_BYTES (PIPE_NT * PIPE_KS / PIPE_GROUP)
#define PIPE_STAGE_BYTES (PIPE_A_BYTES + PIPE_BP_BYTES + PIPE_BS_BYTES)
#define PIPE_BD_BYTES (PIPE_NT * PIPE_LD * 2)
#define PIPE_SMEM_BYTES (PIPE_STAGES * PIPE_STAGE_BYTES + PIPE_BD_BYTES)

__device__ __forceinline__ unsigned int pipe_smem_addr(const void* p) {
    return (unsigned int)__cvta_generic_to_shared(p);
}

// 2026-10-13: 16 bytes global -> shared, zero-filled when !valid (no global read then).
__device__ __forceinline__ void pipe_cp16(void* dst, const void* src, bool valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(pipe_smem_addr(dst)),
                 "l"(src), "r"(valid ? 16 : 0));
}

__device__ __forceinline__ void pipe_cp4(void* dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(pipe_smem_addr(dst)),
                 "l"(src));
}

__device__ __forceinline__ void pipe_commit() { asm volatile("cp.async.commit_group;\n" ::); }

template <int N>
__device__ __forceinline__ void pipe_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ void pipe_ldsm_x4(unsigned int* r, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(pipe_smem_addr(p)));
}

__device__ __forceinline__ void pipe_mma(float* c, const unsigned int* a, unsigned int b0,
                                         unsigned int b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// 2026-10-13: One BF16 pair p (each E2M1 * E4M3, exact) times scale2 in FP32, rounded to BF16 as
// the base kernel rounds `lut * fp8 * scale2`.
__device__ __forceinline__ unsigned int pipe_scale_pair(unsigned int p, float s2) {
    const float lo = __uint_as_float(p << 16) * s2;
    const float hi = __uint_as_float(p & 0xFFFF0000u) * s2;
    const __nv_bfloat162 r = __floats2bfloat162_rn(lo, hi);
    return *(const unsigned int*)&r;
}

extern "C" __global__ void __launch_bounds__(PIPE_THREADS)
moe_w4a16_grouped_gemm_ptrtable_pipe(
    const __nv_bfloat16* __restrict__ A,                   // 2026-10-13: [num_tokens, K]
    const unsigned long long* __restrict__ B_packed_ptrs,  // 2026-10-13: [E] -> [N, K/2] E2M1
    const unsigned long long* __restrict__ B_scale_ptrs,   // 2026-10-13: [E] -> [N, K/16] E4M3
    const float* __restrict__ scale2_vals,                 // 2026-10-13: [E]
    __nv_bfloat16* __restrict__ C,                         // 2026-10-13: [total_expanded, N]
    const int* __restrict__ expert_offsets,                // 2026-10-13: [E + 1] prefix sum
    const int* __restrict__ sorted_token_ids,              // 2026-10-13: row of A, or NULL
    unsigned int num_experts, unsigned int N, unsigned int K) {
    extern __shared__ __align__(16) unsigned char pipe_smem[];

    const unsigned int expert_id = blockIdx.z;
    if (expert_id >= num_experts) return;
    const int m_start = expert_offsets[expert_id];
    const int M_expert = expert_offsets[expert_id + 1] - m_start;
    const int cta_m_local = blockIdx.y * PIPE_MT;
    if (M_expert <= 0 || cta_m_local >= M_expert) return;
    const unsigned char* B_expert = (const unsigned char*)B_packed_ptrs[expert_id];
    const unsigned char* S_expert = (const unsigned char*)B_scale_ptrs[expert_id];
    if (B_expert == 0) return;
    {
        unsigned int dyn;
        asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(dyn));
        if (dyn < PIPE_SMEM_BYTES || (K % PIPE_KS) != 0 || (N % PIPE_NT) != 0 ||
            (((unsigned long long)B_expert) & 15ull) != 0 ||
            (((unsigned long long)S_expert) & 3ull) != 0) {
            __trap();
        }
    }
    const float scale2 = scale2_vals[expert_id];
    const unsigned int cta_m = (unsigned int)(m_start + cta_m_local);
    const unsigned int cta_n = blockIdx.x * PIPE_NT;
    const int rows_left = M_expert - cta_m_local;
    const unsigned int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const unsigned int wm = warp & 1, wn = warp >> 1;
    const unsigned int half_K = K / 2, groups_K = K / PIPE_GROUP;

    // 2026-10-13: Per-thread load slots, fixed for the whole K loop.
    const __nv_bfloat16* a_src[2];
    bool a_ok[2];
    unsigned int a_dst[2];
#pragma unroll
    for (int i = 0; i < 2; i++) {
        const unsigned int c = t + i * PIPE_THREADS, row = c >> 3, col = (c & 7) * 8;
        a_ok[i] = (int)row < rows_left;
        const unsigned int a_row =
            a_ok[i] ? (sorted_token_ids ? (unsigned int)sorted_token_ids[cta_m + row] : cta_m + row)
                    : 0u;
        a_src[i] = A + (unsigned long long)a_row * K + col;
        a_dst[i] = (row * PIPE_LD + col) * 2;
    }
    const unsigned int bp_n = t >> 1, bp_c = (t & 1) * 16;
    const unsigned char* bp_src = B_expert + (unsigned long long)(cta_n + bp_n) * half_K + bp_c;
    const unsigned char* bs_src = S_expert + (unsigned long long)(cta_n + (t & 127)) * groups_K;

    auto load_stage = [&](unsigned int kt, unsigned int slot) {
        unsigned char* st = pipe_smem + slot * PIPE_STAGE_BYTES;
        const unsigned int k0 = kt * PIPE_KS;
#pragma unroll
        for (int i = 0; i < 2; i++) pipe_cp16(st + a_dst[i], a_src[i] + k0, a_ok[i]);
        pipe_cp16(st + PIPE_A_BYTES + bp_n * (PIPE_KS / 2) + bp_c, bp_src + k0 / 2, true);
        if (t < PIPE_NT) {
            pipe_cp4(st + PIPE_A_BYTES + PIPE_BP_BYTES + t * (PIPE_KS / PIPE_GROUP),
                     bs_src + k0 / PIPE_GROUP);
        }
    };

    float acc[2][4][4];
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0.0f;

    __nv_bfloat16* Bd = (__nv_bfloat16*)(pipe_smem + PIPE_STAGES * PIPE_STAGE_BYTES);
    const unsigned int num_k = K / PIPE_KS;
    // 2026-10-13: A warp whose 16-row M fragment lies wholly past the expert's rows skips its
    // MMAs: those rows are zero-filled and never stored.
    const bool m_live[2] = {(int)(wm * 32) < rows_left, (int)(wm * 32 + 16) < rows_left};

#pragma unroll
    for (unsigned int s = 0; s < PIPE_STAGES - 1; s++) {
        if (s < num_k) load_stage(s, s);
        pipe_commit();
    }

    for (unsigned int kt = 0; kt < num_k; kt++) {
        pipe_wait<PIPE_STAGES - 2>();
        __syncthreads();
        const unsigned int pf = kt + PIPE_STAGES - 1;
        if (pf < num_k) load_stage(pf, pf % PIPE_STAGES);
        pipe_commit();

        const unsigned char* st = pipe_smem + (kt % PIPE_STAGES) * PIPE_STAGE_BYTES;
        {
            // 2026-10-13: Thread t dequantises row bp_n, K bp_c * 2 .. + 32 of this stage: one
            // 16-byte packed load (32 nibbles, low nibble = even k) and its two scale bytes.
            const uint4 w = *(const uint4*)(st + PIPE_A_BYTES + bp_n * (PIPE_KS / 2) + bp_c);
            const unsigned char* sc =
                st + PIPE_A_BYTES + PIPE_BP_BYTES + bp_n * (PIPE_KS / PIPE_GROUP) + (t & 1) * 2;
            const unsigned int words[4] = {w.x, w.y, w.z, w.w};
            unsigned int out[16];
#pragma unroll
            for (int q = 0; q < 4; q++) {
                const unsigned int s = ntc_e4m3_bf16x2(sc[q >> 1]);
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    unsigned int d[2];
                    ntc_e2m1x4_bf16(words[q] >> (16 * h), d);
                    out[q * 4 + h * 2] = pipe_scale_pair(ntc_hmul2(d[0], s), scale2);
                    out[q * 4 + h * 2 + 1] = pipe_scale_pair(ntc_hmul2(d[1], s), scale2);
                }
            }
            uint4* dst = (uint4*)(Bd + bp_n * PIPE_LD + bp_c * 2);
#pragma unroll
            for (int v = 0; v < 4; v++)
                dst[v] = make_uint4(out[v * 4], out[v * 4 + 1], out[v * 4 + 2], out[v * 4 + 3]);
        }
        __syncthreads();

        const __nv_bfloat16* As = (const __nv_bfloat16*)st;
#pragma unroll
        for (unsigned int kk = 0; kk < PIPE_KS; kk += 16) {
            unsigned int af[2][4], bf[2][4];
#pragma unroll
            for (int i = 0; i < 2; i++)
                pipe_ldsm_x4(af[i], As + (wm * 32 + i * 16 + (lane & 15)) * PIPE_LD + kk +
                                        (lane >> 4) * 8);
#pragma unroll
            for (int p = 0; p < 2; p++) {
                const unsigned int mat = lane >> 3;
                pipe_ldsm_x4(bf[p], Bd + (wn * 32 + p * 16 + (mat >> 1) * 8 + (lane & 7)) * PIPE_LD +
                                        kk + (mat & 1) * 8);
            }
#pragma unroll
            for (int i = 0; i < 2; i++) {
                if (!m_live[i]) continue;
#pragma unroll
                for (int j = 0; j < 4; j++)
                    pipe_mma(acc[i][j], af[i], bf[j >> 1][(j & 1) * 2], bf[j >> 1][(j & 1) * 2 + 1]);
            }
        }
    }
    pipe_wait<0>();

    const unsigned int g = lane >> 2, q = lane & 3;
#pragma unroll
    for (int i = 0; i < 2; i++) {
#pragma unroll
        for (int r = 0; r < 2; r++) {
            const unsigned int lr = wm * 32 + i * 16 + g + r * 8;
            if ((int)lr >= rows_left) continue;
            __nv_bfloat16* crow = C + (unsigned long long)(cta_m + lr) * N + cta_n + wn * 32 + q * 2;
#pragma unroll
            for (int j = 0; j < 4; j++)
                *(__nv_bfloat162*)(crow + j * 8) =
                    __floats2bfloat162_rn(acc[i][j][r * 2], acc[i][j][r * 2 + 1]);
        }
    }
}

// 2026-10-13: Output columns per CTA; the host sizes grid x from it (`GpuBackend::kernel_n_tile`).
extern "C" __device__ unsigned int moe_w4a16_grouped_gemm_ptrtable_pipe_n_tile = PIPE_NT;
