// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: W4A4 small-M GEMV on the FP4 block-scale tensor-core MMA, and the per-row NVFP4
// activation quantiser it reads. All of it sits inside the METRALE_NO_WARP_BLOCKSCALE_MMA guard:
// hopper and b200 compile gb10/common with that define (their HARDWARE.toml), so there the module
// has no entry points, ops::w4a4_proj::prepare records none and projections stay W4A16.
//
// Owner: gb10 kernels.
// Invariants: listed in the first comment inside the guard.
#ifndef METRALE_NO_WARP_BLOCKSCALE_MMA

// 2026-09-25: W4A4 GEMV on the Blackwell block-scale FP4 MMA (`kind::mxf4nvf4`, E2M1 x E2M1,
// UE4M3 scales per 16 k) for 1 to 32 rows under `--w4a4-downcast`, 33 to 64 with
// `--w4a4-downcast-wide` (ops::w4a4_proj). The checkpoint's NVFP4 weight bytes are the MMA
// operand and its E4M3 group scales are the block scales, so the weights are never dequantized.
// Measured 2026-09-23 on GB10: the BF16 tensor-core w4a16_gemv_tc8 drew 46/51/59 W at 1/4/8 live
// rows, and w4a16_gemv_tc16 with 9..16 live rows cost 19% more J/token at C=4.
//
// Shapes: Aq [M, K / 2] E2M1 activations and As [M, K / 16] E4M3 group scales, both in fragment
// order (below); Ag [M] FP32 per-row global scale; Bq [N, K / 2] and Bs [N, K / 16], the
// checkpoint's weight bytes and group scales; C [M, N] BF16.
//   C[m, n] = scale2 * Ag[m] * sum_k (qa * sa)[m, k] * (qw * sw)[n, k]
//
// Invariants:
// - Block 256: the 8 warps take k128 chunks c = warp (mod 8) in increasing order, and the 8
//   partials are summed in warp order, so a result does not depend on scheduling.
// - The GEMVs read K / 128 chunks; values past the last full chunk are not read.
// - An entry with MB 8-token blocks covers M <= 8 * MB. Only C[m, n] with m < M and n < N is
//   written.
// - Grids: one-tile entries (ceil(N / 16), 1, 1); `_ntX` twins (ceil(N / (16 * X)), 1, 1);
//   `_ps` entries (#SMs, 1, 1) with dynamic shared memory and the extra argument sst
//   (w4a4_gemv_mx_ps.cuh). ops::w4a4_proj::mx_plan picks the entry.
//
// Operands: weights sit on the MMA's 16-row side (A, m16) and tokens on the 8-wide side (B, n8).
// m16n8k64 fragments, 8 four-bit elements per 32-bit register, g = lane / 4, t = lane % 4:
// a0 = row g int t, a1 = row g + 8 int t, a2 = row g int t + 4, a3 = row g + 8 int t + 4;
// b0 = column g int t, b1 = column g int t + 4. Scale block i covers ints 2i and 2i + 1.
// Each thread loads 16 bytes (groups 2t, 2t + 1) of a weight row's 64-byte k128 chunk in the
// checkpoint's packing. One MMA takes groups {G0, G4, G2, G6} of the chunk and the other
// {G1, G5, G3, G7}, the same permutation on both operands: w4a4_quant_rows stores the
// activations in that fragment order, so a lane's 16-byte load is its (b0, b1) for both MMAs.
// The scale selectors are {0, 0}, as in mma_block_scaled_fp4
// (kernels/gb10/qwen3.6-27b/nvfp4/q4k_vendor/mma.cuh): a lane supplies A's scales for row
// g + (t & 1) * 8 and B's for column g.
//
// Activations (w4a4_quant_rows, one CTA per row): per-row global scale
// gs = amax(row) / (6 * 448), or 1 for an all-zero row; per-16 E4M3 group scale
// s = e4m3(amax(group) / 6 / gs); E2M1 round-to-nearest-even of x / (s * gs).










#include "w4a4_mx_core.cuh"

// 2026-09-25: Activation-reuse twin of w4a4_gemv_mx_impl: NT 16-row weight tiles per CTA (grid
// ceil(N / (16 * NT))). Each warp keeps its activation fragments in registers and feeds them to
// all NT tiles, so the activations are read from L2 N / (16 * NT) times rather than N / 16; at
// M = 32 that traffic is twice the weight bytes. Warp w accumulates the k128 chunks c = w (mod 8)
// in increasing order and the 8 partials are summed in warp order, as in w4a4_gemv_mx_impl, so
// each twin is bit-identical to the one-tile kernel of its row range (the model-arch example
// w4a4_gemv_nt_oracle compares them). Weight loads are `ld.global.cs` (evict-first): each weight
// byte is read once, and streaming it should not evict the scale sectors that 4 warps share.
//
// The one-tile entries mx8, mx16 and mx32 instantiate w4a4_gemv_mx_impl, not this body with
// NT = 1; METRALE_W4A4_MX_NT=1 routes 1..32 rows to them (w4a4_proj/mx_plan.rs, mx_pick).





template <int MB, int KU, int NT>
__device__ __forceinline__ void w4a4_gemv_mx_nt_impl(
    const unsigned char* __restrict__ Aq,
    const unsigned char* __restrict__ As,
    const float* __restrict__ Ag,
    const unsigned char* __restrict__ Bq,
    const unsigned char* __restrict__ Bs,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K)
{
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int g = lane >> 2;
    const unsigned int t = lane & 3u;
    const bool odd = (t & 1u) != 0u;
    const unsigned int nb = blockIdx.x * 16u * NT;
    const unsigned int half_K = K >> 1;
    const unsigned int groups = K >> 4;
    const unsigned int num_c = K >> 7;

    bool l0[NT], l1[NT], ls[NT];
    const unsigned char* w0[NT];
    const unsigned char* w1[NT];
    const unsigned char* ws[NT];
    #pragma unroll
    for (int q = 0; q < NT; q++) {
        const unsigned int n0 = nb + 16u * (unsigned int)q;
        const unsigned int r0 = n0 + g, r1 = n0 + g + 8u, rs = n0 + g + (odd ? 8u : 0u);
        l0[q] = r0 < N; l1[q] = r1 < N; ls[q] = rs < N;
        w0[q] = Bq + (unsigned long long)r0 * half_K + t * 16u;
        w1[q] = Bq + (unsigned long long)r1 * half_K + t * 16u;
        ws[q] = Bs + (unsigned long long)rs * groups;
    }

    bool tl[MB];
    const unsigned char* aq[MB];
    const unsigned char* as[MB];
    #pragma unroll
    for (int j = 0; j < MB; j++) {
        const unsigned int tok = (unsigned int)j * 8u + g;
        tl[j] = tok < M;
        aq[j] = Aq + (unsigned long long)tok * half_K + t * 16u;
        as[j] = As + (unsigned long long)tok * groups;
    }

    float acc[NT][MB][4];
    #pragma unroll
    for (int q = 0; q < NT; q++) {
        #pragma unroll
        for (int j = 0; j < MB; j++) {
            #pragma unroll
            for (int c = 0; c < 4; c++) acc[q][j][c] = 0.0f;
        }
    }

    for (unsigned int c0 = warp; c0 < num_c; c0 += W4A4_WARPS * KU) {
        uint4 wl[KU][NT], wh[KU][NT], b[KU][MB];
        uint2 sw[KU][NT], sb[KU][MB];
        #pragma unroll
        for (int u = 0; u < KU; u++) {
            const unsigned int c = c0 + (unsigned int)u * W4A4_WARPS;
            const bool live = c < num_c;
            const uint4 z4 = make_uint4(0u, 0u, 0u, 0u);
            const uint2 z2 = make_uint2(0u, 0u);
            #pragma unroll
            for (int q = 0; q < NT; q++) {
                wl[u][q] = (live && l0[q]) ? __ldcs((const uint4*)(w0[q] + c * 64u)) : z4;
                wh[u][q] = (live && l1[q]) ? __ldcs((const uint4*)(w1[q] + c * 64u)) : z4;
                sw[u][q] = (live && ls[q]) ? *(const uint2*)(ws[q] + c * 8u) : z2;
            }
            #pragma unroll
            for (int j = 0; j < MB; j++) {
                b[u][j] = (live && tl[j]) ? *(const uint4*)(aq[j] + c * 64u) : z4;
                sb[u][j] = (live && tl[j]) ? *(const uint2*)(as[j] + c * 8u) : z2;
            }
        }
        #pragma unroll
        for (int u = 0; u < KU; u++) {
            if (c0 + (unsigned int)u * W4A4_WARPS >= num_c) break;
            #pragma unroll
            for (int q = 0; q < NT; q++) {
                // 2026-09-25: Thread t holds groups 2t, 2t + 1 of each row (its ints 0, 1 | 2, 3). MMA0 takes
                // scale blocks {G0, G4, G2, G6} and MMA1 {G1, G5, G3, G7}: each block's two ints are split
                // across the pair (t, t ^ 1), so one shfl_xor(1) per row per MMA completes the fragment.

                const uint4 WL = wl[u][q], WH = wh[u][q];
                const uint32_t x0l = __shfl_xor_sync(0xFFFFFFFFu, odd ? WL.x : WL.y, 1);
                const uint32_t x0h = __shfl_xor_sync(0xFFFFFFFFu, odd ? WH.x : WH.y, 1);
                const uint32_t x1l = __shfl_xor_sync(0xFFFFFFFFu, odd ? WL.z : WL.w, 1);
                const uint32_t x1h = __shfl_xor_sync(0xFFFFFFFFu, odd ? WH.z : WH.w, 1);
                const uint32_t a00 = odd ? x0l : WL.x, a02 = odd ? WL.y : x0l;
                const uint32_t a01 = odd ? x0h : WH.x, a03 = odd ? WH.y : x0h;
                const uint32_t a10 = odd ? x1l : WL.z, a12 = odd ? WL.w : x1l;
                const uint32_t a11 = odd ? x1h : WH.z, a13 = odd ? WH.w : x1h;
                const uint32_t s0 = __byte_perm(sw[u][q].x, sw[u][q].y, 0x6240);
                const uint32_t s1 = __byte_perm(sw[u][q].x, sw[u][q].y, 0x7351);
                #pragma unroll
                for (int j = 0; j < MB; j++) {
                    w4a4_mma(acc[q][j], a00, a01, a02, a03, b[u][j].x, b[u][j].y, s0, sb[u][j].x);
                    w4a4_mma(acc[q][j], a10, a11, a12, a13, b[u][j].z, b[u][j].w, s1, sb[u][j].y);
                }
            }
        }
    }

    __shared__ float red[W4A4_WARPS][MB][4][32];
    #pragma unroll
    for (int q = 0; q < NT; q++) {
        if (q > 0) __syncthreads();
        #pragma unroll
        for (int j = 0; j < MB; j++) {
            #pragma unroll
            for (int c = 0; c < 4; c++) red[warp][j][c][lane] = acc[q][j][c];
        }
        __syncthreads();

        // 2026-09-25: D (16 x 8): c0, c1 = weight row g, tokens 2t, 2t + 1; c2, c3 = row g + 8.
        const unsigned int r0 = nb + 16u * (unsigned int)q + g, r1 = r0 + 8u;
        for (unsigned int j = warp; j < (unsigned int)MB; j += W4A4_WARPS) {
            float r[4];
            #pragma unroll
            for (int c = 0; c < 4; c++) {
                float v = red[0][j][c][lane];
                #pragma unroll
                for (int ww = 1; ww < W4A4_WARPS; ww++) v += red[ww][j][c][lane];
                r[c] = v;
            }
            #pragma unroll
            for (int c = 0; c < 4; c++) {
                const unsigned int n = (c < 2) ? r0 : r1;
                const unsigned int tok = j * 8u + t * 2u + (unsigned int)(c & 1);
                if (n < N && tok < M) {
                    C[(unsigned long long)tok * N + n] = __float2bfloat16_rn(r[c] * (Ag[tok] * scale2));
                }
            }
        }
    }
}

#define W4A4_ENTRY(NAME, MB, KU)                                                          \
    extern "C" __global__ __launch_bounds__(W4A4_WARPS * 32) void NAME(                    \
        const unsigned char* __restrict__ Aq, const unsigned char* __restrict__ As,       \
        const float* __restrict__ Ag, const unsigned char* __restrict__ Bq,               \
        const unsigned char* __restrict__ Bs, const float scale2,                         \
        __nv_bfloat16* __restrict__ C, unsigned int M, unsigned int N, unsigned int K) {  \
        w4a4_gemv_mx_impl<MB, KU>(Aq, As, Ag, Bq, Bs, scale2, C, M, N, K);                \
    }

W4A4_ENTRY(w4a4_gemv_mx8, 1, 4)
W4A4_ENTRY(w4a4_gemv_mx16, 2, 4)
W4A4_ENTRY(w4a4_gemv_mx32, 4, 2)

#define W4A4_NT_ENTRY(NAME, MB, KU, NT)                                                   \
    extern "C" __global__ __launch_bounds__(W4A4_WARPS * 32) void NAME(                    \
        const unsigned char* __restrict__ Aq, const unsigned char* __restrict__ As,       \
        const float* __restrict__ Ag, const unsigned char* __restrict__ Bq,               \
        const unsigned char* __restrict__ Bs, const float scale2,                         \
        __nv_bfloat16* __restrict__ C, unsigned int M, unsigned int N, unsigned int K) {  \
        w4a4_gemv_mx_nt_impl<MB, KU, NT>(Aq, As, Ag, Bq, Bs, scale2, C, M, N, K);         \
    }

// 2026-09-25: Activation-reuse twins (METRALE_W4A4_MX_NT): mx16_nt2 serves 9..16 rows, mx32_nt4 17..32.
W4A4_NT_ENTRY(w4a4_gemv_mx16_nt2, 2, 4, 2)
W4A4_NT_ENTRY(w4a4_gemv_mx32_nt4, 4, 1, 4)
// 2026-09-25: 33..64 rows (`--w4a4-downcast-wide`), MB = 8. Each output row is bit-identical to the
// row mx8, mx16 or mx32 computes for that token (w4a4_gemv_nt_oracle compares with mx32).
W4A4_NT_ENTRY(w4a4_gemv_mx64, 8, 1, 1)
W4A4_NT_ENTRY(w4a4_gemv_mx64_nt2, 8, 1, 2)

#include "w4a4_gemv_mx_ps.cuh"

#define W4A4_PS_ENTRY(NAME, MB, KU, RJ)                                                   \
    __device__ unsigned int NAME##_ctr[2];                                                \
    extern "C" __global__ __launch_bounds__(W4A4_WARPS * 32, 1) void NAME(                 \
        const unsigned char* __restrict__ Aq, const unsigned char* __restrict__ As,       \
        const float* __restrict__ Ag, const unsigned char* __restrict__ Bq,               \
        const unsigned char* __restrict__ Bs, const float scale2,                         \
        __nv_bfloat16* __restrict__ C, unsigned int M, unsigned int N, unsigned int K,    \
        unsigned int sst) {                                                               \
        w4a4_gemv_mx_ps_impl<MB, KU, RJ>(Aq, As, Ag, Bq, Bs, scale2, C, M, N, K, sst,     \
                                         NAME##_ctr);                                     \
    }

// 2026-09-25: Persistent activation-staged entries (w4a4_gemv_mx_ps.cuh): mx16_ps serves 9..16
// rows and mx32_ps 17..32, each bit-identical to mx16 / mx32.
W4A4_PS_ENTRY(w4a4_gemv_mx16_ps, 2, 2, 2)
W4A4_PS_ENTRY(w4a4_gemv_mx32_ps, 4, 2, 2)

// 2026-09-25: One CTA of 256 threads per row; writes the row's Aq and As (fragment order) and Ag
// (w4a4_quant_rows_impl, w4a4_mx_core.cuh), with the per-row dynamic global scale.
extern "C" __global__ __launch_bounds__(256) void w4a4_quant_rows(
    const __nv_bfloat16* __restrict__ A, unsigned char* __restrict__ Aq,
    unsigned char* __restrict__ As, float* __restrict__ Ag, unsigned int K)
{
    w4a4_quant_rows_impl<false>(A, Aq, As, Ag, K, 0.0f);
}

#endif
