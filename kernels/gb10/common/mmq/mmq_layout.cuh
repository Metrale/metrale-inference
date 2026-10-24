#pragma once

// MMQ tile geometry: the q8_1 / FP4 activation blocks, the K step, the shared-memory row
// strides of each weight format's x tile, the warp count and the column granularity.

#include "mma_tile.cuh"

using namespace mmq_mma;

#define MMQ_ITER_K             256
#define MMQ_ITER_K_FP4         512

typedef void (*load_tiles_mmq_t)(const char * __restrict__ x, int * x_tile, const int kbx0, const int i_max, const int stride);
typedef void (*vec_dot_mmq_t)(const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00);

enum mmq_q8_1_ds_layout {
    MMQ_Q8_1_DS_LAYOUT_D4,
    MMQ_Q8_1_DS_LAYOUT_DS4,
    MMQ_Q8_1_DS_LAYOUT_D2S6,
};

struct block_q8_1_mmq {
    // The y float data is converted to a data layout that can simply be copied to shared memory as a contiguous block.
    // The y float data is first grouped as blocks of 128 values.
    // These blocks are then treated as individual data values and transposed.
    //
    // To avoid shared memory bank conflicts each block is padded with 16 bytes.
    // This padding is also used to store block scales/partial sums.
    // The scales multiplied with the quantized data are equal to the unquantized values.
    // The partial sums are obtained by summing up a subgroup of the contained values (prior to quantization)
    //     and are only needed for performance reasons.
    //
    // The exact data stored depends on the x data type.
    union {
        float d4[4];    // 1 32 bit scale per 32 values, stored as d0,d1,d2,d3
        half2 ds4[4];   // 1 16 bit scale + 1 16 bit partial sum per 32 values, stored as d0,s0,d1,s1,d2,s2,d3,s3
        half  d2s6[8];  // 1 16 bit scale per 64 values + 1 16 bit partial sum per 16 values for the first 96 values,
                        //     stored as d0,d1,s1,s2,s3,s4,s5
    };
    int8_t qs[4*QK8_1]; // 128 values quantized to 8 bit each
};

// NVFP4 activations (Blackwell): block size 16, each int32 of d4 holds 4 ue4m3 scales.
struct block_fp4_mmq {
    uint32_t d4[4];
    int8_t   qs[4 * 32];  // 256 FP4 values packed as 4-bit pairs (2 per byte)
};

static_assert(sizeof(block_q8_1_mmq) == 4*QK8_1 + 4*sizeof(half2), "Unexpected block_q8_1_mmq size");
static_assert(sizeof(block_q8_1_mmq) == 4*sizeof(block_q8_1),      "Unexpected block_q8_1_mmq size");
static_assert(sizeof(block_fp4_mmq)  == sizeof(block_q8_1_mmq),    "Unexpected block_fp4_mmq size");

static constexpr __device__ int get_iter_k([[maybe_unused]] const mmq_type type) {
#if defined(BLACKWELL_MMA_AVAILABLE)
if (type == MMQ_TYPE_NVFP4) {
    return MMQ_ITER_K_FP4;
}
#endif // defined(BLACKWELL_MMA_AVAILABLE)
    return MMQ_ITER_K;
}

static constexpr __device__ int get_mmq_y_device() {
    return 128;
}

// Decouple shared memory tile sizes from WARP_SIZE to allow for different warp sizes.
// The K dimension of the tiles has either,
// 1*MMQ_TILE_NE_K==32 (always for TILE_Y_K) or 2*MMQ_TILE_NE_K==64 (typically for TILE_X_K),
// 32 bit elements for the quantized data (does not include scales).
// In other words, the size of the quantized data in the K dimension is a multiple of MMQ_TILE_NE_K.
// The final tile size in K direction is padded to avoid shared memory bank conflicts,
// in terms of 32 bit elements that means K % 2 == 1 for dp4a or K % 8 == 4 for mma.
#define MMQ_TILE_NE_K 32

#define MMQ_MMA_TILE_X_K_Q8_0  (2*MMQ_TILE_NE_K + 2*MMQ_TILE_NE_K/QI8_0                   + 4)
#define MMQ_MMA_TILE_X_K_FP4   (2*MMQ_TILE_NE_K + 8                                       + 4) // NVFP4, block-scaled MMA
#define MMQ_MMA_TILE_X_K_Q8_1  (2*MMQ_TILE_NE_K + 2*MMQ_TILE_NE_K/QI8_0                   + 4)
#define MMQ_MMA_TILE_X_K_Q2_K  (2*MMQ_TILE_NE_K + MMQ_TILE_NE_K                           + 4)
#define MMQ_MMA_TILE_X_K_Q3_K  (2*MMQ_TILE_NE_K + MMQ_TILE_NE_K/2                         + 4)

static_assert(MMQ_MMA_TILE_X_K_Q8_0 % 8 == 4, "Wrong padding.");
static_assert(MMQ_MMA_TILE_X_K_Q8_1 % 8 == 4, "Wrong padding.");
static_assert(MMQ_MMA_TILE_X_K_Q2_K % 8 == 4, "Wrong padding.");
static_assert(MMQ_MMA_TILE_X_K_Q3_K % 8 == 4, "Wrong padding.");
static_assert(MMQ_MMA_TILE_X_K_FP4  % 8 == 4, "Wrong padding.");
static_assert(MMQ_MMA_TILE_X_K_FP4 == MMQ_MMA_TILE_X_K_Q8_1, "Wrong tile size for FP4");

// block_q8_1_mmq has (128 8-bit ints == 32 32-bit ints + 4 32-bit scales)
#define MMQ_TILE_Y_K     (MMQ_TILE_NE_K + MMQ_TILE_NE_K / QI8_1)

static constexpr __device__ int mmq_get_granularity_device(const int mmq_x) {
    return mmq_x >= 48 ? 16 : 8;
}

static constexpr __device__ int mmq_get_nwarps_device() {
    return 256/mmq_physical_warp_size();
}
