#pragma once

// Shared-memory tile x tile products on the tensor cores, one per x-tile layout.

#include "mmq_layout.cuh"

#ifdef BLACKWELL_MMA_AVAILABLE
// NVFP4 x NVFP4 on the Blackwell block-scaled MMA: each m16n8k64 call takes one uint32 of
// four UE4M3 scales per operand.
template <int mmq_x, int mmq_y>
static __device__ __forceinline__ void vec_dot_fp4_fp4_mma(const int * __restrict__ x,
                                                           const int * __restrict__ y,
                                                           float * __restrict__ sum,
                                                           const int k00) {

    typedef tile<16, 8, int>   tile_A;
    typedef tile<8, 8, int>    tile_B;
    typedef tile<16, 8, float> tile_C;

    constexpr int stride        = MMQ_MMA_TILE_X_K_FP4;
    constexpr int granularity   = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp = 2 * granularity;
    constexpr int ntx           = rows_per_warp / tile_C::I;
    constexpr int nfrags        = MMQ_TILE_NE_K / tile_A::J;

    y += (threadIdx.y % ntx) * (tile_C::J * MMQ_TILE_Y_K);

    const int *      x_qs = (const int *) x;
    const uint32_t * x_sc = (const uint32_t *) (x_qs + 2 * MMQ_TILE_NE_K);
    const int *      y_qs = (const int *) y + 4;
    const uint32_t * y_sc = (const uint32_t *) y;

    // 2 threads per quad supply the packed scale register to the block_scale MMA,
    // see https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-block-scaling
    const int tidx_A = threadIdx.x / 4 + (threadIdx.x % 2) * 8;
    const int tidx_B = threadIdx.x / 4;
    const int i0     = (threadIdx.y / ntx) * rows_per_warp;

    tile_A   A[ntx][nfrags];
    uint32_t scaleA[ntx][nfrags];

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int frag = 0; frag < nfrags; ++frag) {
            const int k0 = k00 + frag * tile_A::J;
            load_ldmatrix(A[n][frag], x_qs + (i0 + n * tile_A::I) * stride + k0, stride);
            scaleA[n][frag] = x_sc[(i0 + n * tile_A::I + tidx_A) * stride + k0 / tile_A::J];
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx * tile_C::J) {
        tile_B   B[nfrags];
        uint32_t scaleB[nfrags];

#pragma unroll
        for (int frag = 0; frag < nfrags; ++frag) {
            const int k0 = frag * tile_B::J;
            load_generic(B[frag], y_qs + j0 * MMQ_TILE_Y_K + k0, MMQ_TILE_Y_K);
            scaleB[frag] = y_sc[(j0 + tidx_B) * MMQ_TILE_Y_K + frag];
        }

#pragma unroll
        for (int n = 0; n < ntx; ++n) {
#pragma unroll
            for (int frag = 0; frag < nfrags; ++frag) {
                tile_C C = {};
                mma_block_scaled_fp4(C, A[n][frag], B[frag], scaleA[n][frag], scaleB[frag]);
#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    sum[(j0 / tile_C::J + n) * tile_C::ne + l] += C.x[l];
                }
            }
        }
    }
}
#endif // BLACKWELL_MMA_AVAILABLE

template <int mmq_x, int mmq_y, mmq_q8_1_ds_layout ds_layout>
static __device__ __forceinline__ void vec_dot_q8_0_q8_1_mma(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    typedef tile<16, 8, int> tile_A;
    typedef tile< 8, 8, int> tile_B;
    typedef tile<16, 8, int> tile_C;

    constexpr int granularity = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp = 2 * granularity;
    constexpr int ntx = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + 2*MMQ_TILE_NE_K;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;
    const half2 * y_ds = (const half2 *) y;

    tile_A A[ntx][MMQ_TILE_NE_K/QI8_0];
    float dA[ntx][tile_C::ne/2][MMQ_TILE_NE_K/QI8_0];

    const int i0 = (threadIdx.y/ntx)*rows_per_warp;

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_0) {
            const int k0 = k00 + k01;

            load_ldmatrix(A[n][k01/QI8_0], x_qs + (i0 + n*tile_A::I)*MMQ_MMA_TILE_X_K_Q8_0 + k0, MMQ_MMA_TILE_X_K_Q8_0);
        }

#pragma unroll
        for (int l = 0; l < tile_C::ne/2; ++l) {
            const int i = i0 + n*tile_A::I + tile_C::get_i(2*l);

#pragma unroll
            for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_0) {
                const int k0 = k00 + k01;

                dA[n][l][k01/QI8_0] = x_df[i*MMQ_MMA_TILE_X_K_Q8_0 + k0/QI8_0];
            }
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx*tile_C::J) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_0) {
            tile_B B;
            float dB[tile_C::ne/2];

            load_generic(B, y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K); // faster than load_ldmatrix

#pragma unroll
            for (int l = 0; l < tile_C::ne/2; ++l) {
                const int j = j0 + tile_C::get_j(l);

                if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D4) {
                    dB[l] =             y_df[j*MMQ_TILE_Y_K + k01/QI8_1];
                } else {
                    dB[l] = __low2float(y_ds[j*MMQ_TILE_Y_K + k01/QI8_1]);
                }
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C;
                mma(C, A[n][k01/QI8_0], B);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += C.x[l]*dA[n][l/2][k01/QI8_0]*dB[l%2];
                }
            }
        }
    }
}

template <int mmq_x, int mmq_y>
static __device__ __forceinline__ void vec_dot_q8_1_q8_1_mma(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    typedef tile<16,  8, int> tile_A;
    typedef tile< 8,  8, int> tile_B;
    typedef tile<16,  8, int> tile_C;

    constexpr int granularity = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp = 2 * granularity;
    constexpr int ntx = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + 2*MMQ_TILE_NE_K;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_dm = (const half2 *) y;

    tile_A   A[ntx][MMQ_TILE_NE_K/QI8_1];
    float2 dmA[ntx][tile_C::ne/2][MMQ_TILE_NE_K/QI8_1];

    const int i0 = (threadIdx.y/ntx)*rows_per_warp;

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
            const int k0 = k00 + k01;

            load_ldmatrix(A[n][k01/QI8_1], x_qs + (i0 + n*tile_A::I)*MMQ_MMA_TILE_X_K_Q8_1 + k0, MMQ_MMA_TILE_X_K_Q8_1);
        }

#pragma unroll
        for (int l = 0; l < tile_C::ne/2; ++l) {
            const int i = i0 + n*tile_A::I + tile_C::get_i(2*l);

#pragma unroll
            for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
                const int k0 = k00 + k01;

                dmA[n][l][k01/QI8_1] = __half22float2(x_dm[i*MMQ_MMA_TILE_X_K_Q8_1 + k0/QI8_1]);
            }
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx*tile_C::J) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
            tile_B   B;
            float2 dsB[tile_C::ne/2];

            load_generic(B, y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K); // faster than load_ldmatrix

#pragma unroll
            for (int l = 0; l < tile_C::ne/2; ++l) {
                const int j = j0 + tile_C::get_j(l);

                dsB[l] = __half22float2(y_dm[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C;
                mma(C, A[n][k01/QI8_1], B);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += dmA[n][l/2][k01/QI8_1].x*dsB[l%2].x*C.x[l];
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += dmA[n][l/2][k01/QI8_1].y*dsB[l%2].y;
                }
            }
        }
    }
}

// Used for Q3_K, IQ2_S, and IQ2_XS:
template <int mmq_x, int mmq_y>
static __device__ __forceinline__ void vec_dot_q8_0_16_q8_1_mma(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {

    typedef tile<16, 4, int> tile_A;
    typedef tile<16, 8, int> tile_A_8;
    typedef tile< 8, 4, int> tile_B;
    typedef tile<16, 8, int> tile_C;

    constexpr int granularity = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp = 2 * granularity;
    constexpr int ntx = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + MMQ_TILE_NE_K*2;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

    const int i0 = (threadIdx.y / ntx) * (ntx*tile_A::I);

    tile_A  A[ntx][8];
    float  dA[ntx][tile_C::ne/2][8];

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 8) {
            const int k0 = k00 + k01;

            load_ldmatrix(((tile_A_8 *) A[n])[k01/8], x_qs + (i0 + n*tile_A::I)*MMQ_MMA_TILE_X_K_Q3_K + k0, MMQ_MMA_TILE_X_K_Q3_K);
        }

#pragma unroll
        for (int l = 0; l < tile_C::ne/2; ++l) {
            const int i = i0 + n*tile_C::I + tile_C::get_i(2*l);

#pragma unroll
            for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 4) {
                const int k0 = k00 + k01;

                dA[n][l][k01/4] = x_df[i*MMQ_MMA_TILE_X_K_Q3_K + k0/4];
            }
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx*tile_C::J) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QR3_K*VDR_Q3_K_Q8_1_MMQ) {
            tile_B B[2];
            float dB[tile_C::ne/2];

            // Here load_generic is faster than load_ldmatrix.
            load_generic(B[0], y_qs + j0*MMQ_TILE_Y_K + (k01 + 0),         MMQ_TILE_Y_K);
            load_generic(B[1], y_qs + j0*MMQ_TILE_Y_K + (k01 + tile_B::J), MMQ_TILE_Y_K);

#pragma unroll
            for (int l = 0; l < tile_C::ne/2; ++l) {
                const int j = j0 + tile_C::get_j(l);

                dB[l] = y_df[j*MMQ_TILE_Y_K + k01/QI8_1];
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C[2];
                mma(C[0], A[n][k01/4 + 0], B[0]);
                mma(C[1], A[n][k01/4 + 1], B[1]);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += dB[l%2]*(C[0].x[l]*dA[n][l/2][k01/4 + 0] + C[1].x[l]*dA[n][l/2][k01/4 + 1]);
                }
            }
        }
    }
}

template <int mmq_x, int mmq_y>
static __device__ __forceinline__ void vec_dot_q2_K_q8_1_mma(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {

    typedef tile<16, 4, int> tile_A;
    typedef tile<16, 8, int> tile_A_8;
    typedef tile< 8, 4, int> tile_B;
    typedef tile<16, 8, int> tile_C;

    constexpr int granularity = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp = 2 * granularity;
    constexpr int ntx = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + MMQ_TILE_NE_K*2;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;

    const int i0 = (threadIdx.y / ntx) * (ntx*tile_A::I);

    tile_A  A[ntx][8];
    float  dA[ntx][tile_C::ne/2][8];
    float  mA[ntx][tile_C::ne/2][8];

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
            const int k0 = k00 + k01;

            load_ldmatrix(((tile_A_8 *) A[n])[k01/QI8_1], x_qs + (i0 + n*tile_A::I)*MMQ_MMA_TILE_X_K_Q2_K + k0, MMQ_MMA_TILE_X_K_Q2_K);
        }
    }

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int l = 0; l < tile_C::ne/2; ++l) {
            const int i = i0 + n*tile_C::I + tile_C::get_i(2*l);

#pragma unroll
            for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1/2) {
                const int k0 = k00 + k01;

                const float2 dm = __half22float2(x_dm[i*MMQ_MMA_TILE_X_K_Q2_K + k0/(QI8_1/2)]);

                dA[n][l][k01/(QI8_1/2)] = dm.x;
                mA[n][l][k01/(QI8_1/2)] = dm.y;
            }
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx*tile_C::J) {
        float2 dB[tile_C::ne/2];

#pragma unroll
        for (int l = 0; l < tile_C::ne/2; ++l) {
            const int j = j0 + tile_C::get_j(l);

            dB[l] = __half22float2(y_ds[j*MMQ_TILE_Y_K]);
        }

#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
            tile_B B[2];

            // Here load_generic is faster than load_ldmatrix.
            load_generic(B[0], y_qs + j0*MMQ_TILE_Y_K + (k01 + 0),         MMQ_TILE_Y_K);
            load_generic(B[1], y_qs + j0*MMQ_TILE_Y_K + (k01 + tile_B::J), MMQ_TILE_Y_K);

            tile_C Cm[2];
            if (k01 >= MMQ_TILE_NE_K * 3/4) {
                tile_A A1;
                A1.x[0] = 0x01010101;
                A1.x[1] = 0x01010101;
                mma(Cm[0], A1, B[0]);
                mma(Cm[1], A1, B[1]);
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C Cd[2];

                mma(Cd[0], A[n][k01/4 + 0], B[0]);
                mma(Cd[1], A[n][k01/4 + 1], B[1]);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    float tmp = Cd[0].x[l]*dA[n][l/2][k01/4 + 0] + Cd[1].x[l]*dA[n][l/2][k01/4 + 1];
                    if (k01 >= MMQ_TILE_NE_K * 3/4) {
                        tmp -= Cm[0].x[l]*mA[n][l/2][k01/4 + 0] + Cm[1].x[l]*mA[n][l/2][k01/4 + 1];
                    }
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += tmp*(k01 < MMQ_TILE_NE_K/2 ? dB[l%2].x : dB[l%2].y);
                }
            }
        }

#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K * 3/4; k01 += QI8_1) {
            float2 sB[tile_C::ne/2];

#pragma unroll
            for (int l = 0; l < tile_C::ne/2; ++l) {
                const int j = j0 + tile_C::get_j(l);

                sB[l] = __half22float2(y_ds[j*MMQ_TILE_Y_K + (1 + k01/QI8_1)]);
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] -= mA[n][l/2][k01/4 + 0]*sB[l%2].x;
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] -= mA[n][l/2][k01/4 + 1]*sB[l%2].y;
                }
            }
        }
    }
}
