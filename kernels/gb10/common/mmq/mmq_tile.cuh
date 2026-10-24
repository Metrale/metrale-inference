#pragma once

// One mmq_y x mmq_x output tile: per-format traits, the K loop and the BF16 write-back.

#include "mmq_load_tiles.cuh"
#include "mmq_vec_dot.cuh"

template<mmq_type type, int mmq_x, int mmq_y, bool need_check>
static __device__ __forceinline__ void mmq_write_back_mma(
        const float * __restrict__ sum, const int * __restrict__ ids_dst, __nv_bfloat16 * __restrict__ dst,
        const int stride, const int i_max, const int j_max) {

    constexpr int granularity = mmq_get_granularity_device(mmq_x);
    constexpr int nwarps = mmq_get_nwarps_device();

    typedef tile<16, 8, int> tile_C;
    constexpr int rows_per_warp = 2 * granularity;
    constexpr int ntx = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    const int i0 = (threadIdx.y / ntx) * (ntx*tile_C::I);
    static_assert(nwarps*tile_C::I == mmq_y, "nwarps*tile_C::I != mmq_y");

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx*tile_C::J) {
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int j = j0 + (threadIdx.y % ntx) * tile_C::J + tile_C::get_j(l);

                if (j > j_max) {
                    continue;
                }

                const int i = i0 + n*tile_C::I + tile_C::get_i(l);

                if (need_check && i > i_max) {
                    continue;
                }

                dst[ids_dst[j]*stride + i] = __float2bfloat16(sum[(j0/tile_C::J + n)*tile_C::ne + l]);
            }
        }
    }
}

template <int mmq_x, int mmq_y, bool need_check, mmq_type type>
struct mmq_type_traits;

// PrismML Ternary-Bonsai Q2_0: the 2-bit load_tiles and a DS4 vec_dot (the layout of the
// DS4 activation quantizer). Post-unpack the tile IS q8_0, so the vec_dot is the q8_0 one.
template <int mmq_x, int mmq_y, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, need_check, MMQ_TYPE_Q2_0> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q2_0<mmq_y, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, MMQ_Q8_1_DS_LAYOUT_DS4>;
};

#ifdef BLACKWELL_MMA_AVAILABLE
template <int mmq_x, int mmq_y, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, need_check, MMQ_TYPE_NVFP4> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_nvfp4_nvfp4<mmq_y, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_fp4_fp4_mma<mmq_x, mmq_y>;
};
#endif // BLACKWELL_MMA_AVAILABLE

template <int mmq_x, int mmq_y, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, need_check, MMQ_TYPE_Q2_K> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q2_K<mmq_y, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q2_K_q8_1_mma<mmq_x, mmq_y>;
};

template <int mmq_x, int mmq_y, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, need_check, MMQ_TYPE_Q3_K> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q3_K<mmq_y, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_16_q8_1_mma<mmq_x, mmq_y>;
};

template <int mmq_x, int mmq_y, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, need_check, MMQ_TYPE_Q4_K> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q4_K<mmq_y, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_1_q8_1_mma<mmq_x, mmq_y>;
};

template <mmq_type type, int mmq_x, bool need_check>
static __device__ __forceinline__ void mul_mat_q_process_tile(
        const char * __restrict__ x, const int offset_x, const int * __restrict__ y,
        const int * __restrict__ ids_dst, __nv_bfloat16 * __restrict__ dst,
        const int stride_row_x, const int ncols_y, const int stride_col_dst,
        const int tile_x_max_i, const int tile_y_max_j, const int kb0_start, const int kb0_stop) {

    constexpr int              warp_size  = mmq_physical_warp_size();
    constexpr int              nwarps     = mmq_get_nwarps_device();
    constexpr int              qk         = mmq_quant_traits<type>::qk;
    constexpr int              mmq_y      = get_mmq_y_device();
    constexpr load_tiles_mmq_t load_tiles = mmq_type_traits<mmq_x, mmq_y, need_check, type>::load_tiles;

    extern __shared__ int data_mul_mat_q[];
    int * tile_y = data_mul_mat_q + mmq_x;
    int * tile_x = tile_y + MMQ_PAD(mmq_x*MMQ_TILE_Y_K, nwarps*warp_size);

    constexpr vec_dot_mmq_t    vec_dot    = mmq_type_traits<mmq_x, mmq_y, need_check, type>::vec_dot_mma;

#if defined(BLACKWELL_MMA_AVAILABLE)
    // FP4 tile stores 8 blocks
    constexpr int ne_block = type == MMQ_TYPE_NVFP4 ? QK_K : 4 * QK8_1;
#else
    constexpr int ne_block = 4 * QK8_1;
#endif  // defined(BLACKWELL_MMA_AVAILABLE)

    constexpr int ITER_K          = get_iter_k(type);
    constexpr int blocks_per_iter = ITER_K / qk;

    float sum[mmq_x*mmq_y / (nwarps*warp_size)] = {0.0f};

    constexpr int sz = sizeof(block_q8_1_mmq) / sizeof(int);

    for (int kb0 = kb0_start; kb0 < kb0_stop; kb0 += blocks_per_iter) {
        load_tiles(x, tile_x, offset_x + kb0, tile_x_max_i, stride_row_x);
        {
            const int * by0 = y + ncols_y * (kb0 * qk / ne_block) * sz;
#pragma unroll
            for (int l0 = 0; l0 < mmq_x * MMQ_TILE_Y_K; l0 += nwarps * warp_size) {
                int l = l0 + threadIdx.y*warp_size + threadIdx.x;

                tile_y[l] = by0[l];
            }
        }

        __syncthreads();

        vec_dot(tile_x, tile_y, sum, 0);

        __syncthreads();

        {
            const int * by0 = y + ncols_y * ((kb0 * qk / ne_block) * sz + sz);
#pragma unroll
            for (int l0 = 0; l0 < mmq_x * MMQ_TILE_Y_K; l0 += nwarps * warp_size) {
                int l = l0 + threadIdx.y*warp_size + threadIdx.x;

                tile_y[l] = by0[l];
            }
        }

        __syncthreads();

        vec_dot(tile_x, tile_y, sum, MMQ_TILE_NE_K);

        __syncthreads();
    }

    mmq_write_back_mma<type, mmq_x, mmq_y, need_check>(sum, ids_dst, dst, stride_col_dst, tile_x_max_i, tile_y_max_j);
}
