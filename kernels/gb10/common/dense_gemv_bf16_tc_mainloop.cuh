// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-17: The dense_gemv_bf16_tc body, one overload per inclusion. dense_gemv_bf16_tc_body.cuh
// includes this file once per output type with DTC_OUT_T defined (__nv_bfloat16, float), so each
// output type is an overload of one template with its own signature rather than a type template
// parameter: the BF16 overload keeps the exact signature, and so the symbol names and PTX, it had
// before the FP32 output existed.
//
// Owner: gb10 kernels.
// Invariants: the invariants of dense_gemv_bf16_tc_body.cuh; include it, not this file.

#ifndef DTC_OUT_T
#error "dense_gemv_bf16_tc_mainloop.cuh is included by dense_gemv_bf16_tc_body.cuh with DTC_OUT_T defined"
#endif

template <int NB, int NT, int KU>
__device__ __forceinline__ void dense_gemv_bf16_tc_impl(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ W,
    DTC_OUT_T* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K, unsigned int out_stride)
{
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int g = lane >> 2;
    const unsigned int t = lane & 3u;
    const unsigned int n0 = blockIdx.x * (16u * NT);
    const unsigned int num_kb = K / DTC_KB;

    // 2026-09-25: Weight rows g and g+8 of each 16-row tile; a row past N loads zeros.
    bool live[NT][2];
    const __nv_bfloat16* wrow[NT][2];
    #pragma unroll
    for (int i = 0; i < NT; i++) {
        #pragma unroll
        for (int h = 0; h < 2; h++) {
            const unsigned int n = n0 + (unsigned int)i * 16u + (unsigned int)h * 8u + g;
            live[i][h] = n < N;
            wrow[i][h] = W + (unsigned long long)(live[i][h] ? n : 0u) * K + t * 8u;
        }
    }
    // 2026-09-25: Token tile b is live iff its first row is < M (warp-uniform). Within a
    // live tile, a token row b*8+g past M feeds zeros.
    bool tile_on[NB];
    bool tok_live[NB];
    const __nv_bfloat16* arow[NB];
    #pragma unroll
    for (int b = 0; b < NB; b++) {
        const unsigned int tok = (unsigned int)b * 8u + g;
        tile_on[b] = (unsigned int)b * 8u < M;
        tok_live[b] = tok < M;
        arow[b] = A + (unsigned long long)(tok_live[b] ? tok : 0u) * K + t * 8u;
    }

    float acc[NT][NB][4];
    #pragma unroll
    for (int i = 0; i < NT; i++) {
        #pragma unroll
        for (int b = 0; b < NB; b++) {
            #pragma unroll
            for (int c = 0; c < 4; c++) acc[i][b][c] = 0.0f;
        }
    }

    // 2026-09-25: KU k-blocks per trip: all their weight loads are issued before any MMA,
    // so each thread has KU * NT * 2 rows x 32 B of weights in flight.
    for (unsigned int kb0 = warp; kb0 < num_kb; kb0 += DTC_WARPS * KU) {
        uint4 w[KU][NT][2][2];
        #pragma unroll
        for (int u = 0; u < KU; u++) {
            const unsigned int kb = kb0 + (unsigned int)u * DTC_WARPS;
            #pragma unroll
            for (int i = 0; i < NT; i++) {
                #pragma unroll
                for (int h = 0; h < 2; h++) {
                    if (live[i][h] && kb < num_kb) {
                        const __nv_bfloat16* p = wrow[i][h] + kb * DTC_KB;
                        w[u][i][h][0] = dtc_ld_stream(p);
                        w[u][i][h][1] = dtc_ld_stream(p + 32);
                    } else {
                        w[u][i][h][0] = make_uint4(0u, 0u, 0u, 0u);
                        w[u][i][h][1] = make_uint4(0u, 0u, 0u, 0u);
                    }
                }
            }
        }
        #pragma unroll
        for (int u = 0; u < KU; u++) {
            const unsigned int kb = kb0 + (unsigned int)u * DTC_WARPS;
            if (kb >= num_kb) break;
            #pragma unroll
            for (int b = 0; b < NB; b++) {
                if (!tile_on[b]) continue;
                uint4 av[2];
                if (tok_live[b]) {
                    const uint4* ap = (const uint4*)(arow[b] + kb * DTC_KB);
                    av[0] = ap[0];
                    av[1] = ap[4];   // 2026-09-25: +32 k, the same split as the weights
                } else {
                    av[0] = make_uint4(0u, 0u, 0u, 0u);
                    av[1] = make_uint4(0u, 0u, 0u, 0u);
                }
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    // 2026-09-25: Words 2j and 2j+1 of this thread's 16 k (j < 2: first piece).
                    const uint32_t b0 = dtc_word(av[j >> 1], (2 * j) & 3);
                    const uint32_t b1 = dtc_word(av[j >> 1], (2 * j + 1) & 3);
                    #pragma unroll
                    for (int i = 0; i < NT; i++) {
                        const uint32_t a0 = dtc_word(w[u][i][0][j >> 1], (2 * j) & 3);
                        const uint32_t a2 = dtc_word(w[u][i][0][j >> 1], (2 * j + 1) & 3);
                        const uint32_t a1 = dtc_word(w[u][i][1][j >> 1], (2 * j) & 3);
                        const uint32_t a3 = dtc_word(w[u][i][1][j >> 1], (2 * j + 1) & 3);
                        dtc_mma(acc[i][b], a0, a1, a2, a3, b0, b1);
                    }
                }
            }
        }
    }

    // 2026-09-25: Fixed-order split-K reduction across the CTA's warps.
    __shared__ float red[DTC_WARPS][NT * NB][4][32];
    #pragma unroll
    for (int i = 0; i < NT; i++) {
        #pragma unroll
        for (int b = 0; b < NB; b++) {
            #pragma unroll
            for (int c = 0; c < 4; c++) red[warp][i * NB + b][c][lane] = acc[i][b][c];
        }
    }
    __syncthreads();

    // 2026-09-25: D fragment: d0, d1 = (weight row g, tokens 2t, 2t+1); d2, d3 = row g+8.
    for (unsigned int f = warp; f < (unsigned int)(NT * NB); f += DTC_WARPS) {
        const unsigned int i = f / NB;
        const unsigned int b = f % NB;
        if (b * 8u >= M) continue;
        float r[4];
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            float v = red[0][f][c][lane];
            #pragma unroll
            for (int ww = 1; ww < DTC_WARPS; ww++) v += red[ww][f][c][lane];
            r[c] = v;
        }
        const unsigned int tok0 = b * 8u + t * 2u;
        const unsigned int n_lo = n0 + i * 16u + g;
        const unsigned int n_hi = n_lo + 8u;
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            const unsigned int tok = tok0 + (unsigned int)(c & 1);
            const unsigned int n = (c < 2) ? n_lo : n_hi;
            if (tok < M && n < N)
                C[(unsigned long long)tok * out_stride + n] = dtc_out<DTC_OUT_T>(r[c]);
        }
    }
}
