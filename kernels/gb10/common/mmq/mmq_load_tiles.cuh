#pragma once

// Global -> shared loads of one mmq_y x ITER_K weight tile, one function per weight format.

#include "vecdot_mmvq.cuh"
#include "mmq_layout.cuh"

// PrismML Ternary-Bonsai Q2_0 tile load — clone of load_tiles_q1_0 with a 2-bit
// `(code-1)` unpack. Dequantizes the packed 2-bit codes into int8 in the Q8_0
// shared-mem tile layout (MMQ_MMA_TILE_X_K_Q8_0, x_df holds the fp16 block
// scale d), so the downstream stock `vec_dot_q8_0_q8_1_mma` runs verbatim.
//
// Each thread owns one 32-weight sub-chunk (kqsx) of a 128-weight block. 32
// codes = 64 bits = TWO 32-bit words (Q1_0 read one). Codes are 4-per-byte,
// low-bits-first; code c -> int8 (c-1) in {-1,0,1,2}, packed 4 per int32 into
// the same 8 int32 (=32 int8) per thread that Q1_0 produces. The block scale
// store is byte-identical to Q1_0 (d @front, broadcast to 4 sub-scales).
template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_q2_0(
    const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int nwarps = mmq_get_nwarps_device();
    constexpr int warp_size = mmq_physical_warp_size();

    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + 2*MMQ_TILE_NE_K);

    constexpr int blocks_per_iter = MMQ_ITER_K / QK2_0;
    constexpr int threads_per_row = blocks_per_iter * QI2_0;
    constexpr int nrows = warp_size / threads_per_row;
    constexpr int scale_entries_per_block = QK2_0 / QK8_1;
    constexpr int scale_entries_per_row = blocks_per_iter * scale_entries_per_block;

    const int txi  = threadIdx.x % threads_per_row;
    const int kbx  = txi / QI2_0;
    const int kqsx = txi % QI2_0;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nrows*nwarps) {
        int i = i0 + threadIdx.y*nrows + threadIdx.x/threads_per_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q2_0 * bxi = (const block_q2_0 *) x + kbx0 + i*stride + kbx;
        // 32 weights = 8 code-bytes = two 32-bit words. Byte-assemble (block
        // stride 34B → only 2-byte aligned; a raw uint32 load would fault).
        const int qs_offset = 8*kqsx;
        const int qs0 = bxi->qs[qs_offset + 0] | (bxi->qs[qs_offset + 1] << 8) |
                        (bxi->qs[qs_offset + 2] << 16) | (bxi->qs[qs_offset + 3] << 24);
        const int qs1 = bxi->qs[qs_offset + 4] | (bxi->qs[qs_offset + 5] << 8) |
                        (bxi->qs[qs_offset + 6] << 16) | (bxi->qs[qs_offset + 7] << 24);

        int unpacked_bytes[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int word  = (j < 4) ? qs0 : qs1;   // codes 0..15 / 16..31
            const int shift = (j & 3) * 8;           // 4 codes (2 bits each) per int32
            const int c0 = ((word >> (shift + 0)) & 0x3) - 1;
            const int c1 = ((word >> (shift + 2)) & 0x3) - 1;
            const int c2 = ((word >> (shift + 4)) & 0x3) - 1;
            const int c3 = ((word >> (shift + 6)) & 0x3) - 1;
            unpacked_bytes[j] = (c0 & 0xFF) | ((c1 & 0xFF) << 8) | ((c2 & 0xFF) << 16) | ((c3 & 0xFF) << 24);
        }

        const int dst_offset = kbx*(scale_entries_per_block*QI8_0) + kqsx*QI8_0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + dst_offset + j] = unpacked_bytes[j];
        }
    }

    const int ksx = threadIdx.x % scale_entries_per_row;
    const int scale_block = ksx / scale_entries_per_block;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q2_0 * bxi = (const block_q2_0 *) x + kbx0 + i*stride + scale_block;

        x_df[i*MMQ_MMA_TILE_X_K_Q8_0 + ksx] = bxi->d;
    }
}

#ifdef BLACKWELL_MMA_AVAILABLE
template <int mmq_y, bool need_check>
static __device__ __forceinline__ void load_tiles_nvfp4_nvfp4(const char * __restrict__ x,
                                                            int * __restrict__ x_tile,
                                                            const int kbx0,
                                                            const int i_max,
                                                            const int stride) {
    constexpr int nwarps = mmq_get_nwarps_device();
    constexpr int warp_size = mmq_physical_warp_size();
    constexpr int iter_k = get_iter_k(MMQ_TYPE_NVFP4);
    constexpr int threads_per_row = iter_k / QK_NVFP4; // each thread processes 1 block
    constexpr int rows_per_warp = warp_size / threads_per_row;

    uint32_t * x_u32 = (uint32_t *) x_tile;

    const int txi = threadIdx.x;
    const int kbx = txi % threads_per_row;
    const int row_in_warp = txi / threads_per_row;

    // METRALE (A2): the weight buffer is SoA per row — [qs: bpr*32][d: bpr*4] —
    // not an array of 36-byte block_nvfp4. `stride` still carries blocks-per-row,
    // so the row base is stride*36 bytes and the two regions are derived from it.
    // This makes the 32 qs bytes CONTIGUOUS: two 16-byte loads instead of eight
    // 4-byte loads. Written by metrale_nvfp4_repack; see the note there.
    // ★ `kbx0` is a LINEAR block index that already folds in the CTA's row offset
    // (`offset_x = ... + it*mmq_y*stride_row_x`, line 3640) — it is NOT a pure
    // k-offset. The SoA layout is per-ROW, so recover the two components once per
    // thread. One div+mod, hoisted out of the i-loop.
    const int bpr  = stride;                 // blocks per row
    const int row0 = kbx0 / bpr;             // first row of this tile
    const int kb0  = kbx0 - row0 * bpr;      // k-block offset within the row
    uint32_t * x_u32_scale = x_u32 + 64 + kbx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += rows_per_warp * nwarps) {
        int i = i0 + threadIdx.y * rows_per_warp + row_in_warp;

        if constexpr (need_check) {
            i = min(i, i_max);
        }

        const char * rp = x + (size_t) (row0 + i) * bpr * 36;   // this row's base
        const int kb = kb0 + kbx;                                  // block within row
        const int row_base = i * MMQ_MMA_TILE_X_K_FP4;
        const int q_base = row_base + 8 * kbx;

        // 32 contiguous qs bytes = 2x16B. Aligned because the row base is
        // bpr*36 (16-B aligned for bpr%4==0, i.e. K%256==0 — both live K qualify)
        // and the in-row offset is a multiple of 32.
        const uint4 q0 = *reinterpret_cast<const uint4 *>(rp + (size_t) kb * 32);
        const uint4 q1 = *reinterpret_cast<const uint4 *>(rp + (size_t) kb * 32 + 16);

        x_u32[q_base + 0] = q0.x; x_u32[q_base + 1] = q0.y;
        x_u32[q_base + 2] = q0.z; x_u32[q_base + 3] = q0.w;
        x_u32[q_base + 4] = q1.x; x_u32[q_base + 5] = q1.y;
        x_u32[q_base + 6] = q1.z; x_u32[q_base + 7] = q1.w;

        x_u32_scale[row_base] =
            *reinterpret_cast<const uint32_t *>(rp + (size_t) bpr * 32 + (size_t) kb * 4);
    }
}
#endif // BLACKWELL_MMA_AVAILABLE

template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_q2_K(
    const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int nwarps = mmq_get_nwarps_device();

    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + 2*MMQ_TILE_NE_K);

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR2_K);
    constexpr int nrows = mmq_physical_warp_size() / threads_per_row;
    const int kqsx = threadIdx.x % threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nrows*nwarps) {
        int i = i0 + threadIdx.y*nrows + threadIdx.x/threads_per_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q2_K * bxi = (const block_q2_K *) x + kbx0 + i*stride;

        const int x_ql_0 = get_int_b2(bxi->qs, kqsx);

#pragma unroll
        for (int l = 0; l < QR2_K; ++l) {
            const int k = (kqsx/8)*32 + l*8 + kqsx % 8;

            const int x_qs_k = (x_ql_0 >> (2*l)) & 0x03030303;

            x_qs[i*MMQ_MMA_TILE_X_K_Q2_K + k] = x_qs_k;
        }

        const int sc_m = bxi->scales[kqsx];
        const half2 x_dm_ik = __hmul2(bxi->dm, make_half2(sc_m & 0x0F, sc_m >> 4));

        x_dm[i*MMQ_MMA_TILE_X_K_Q2_K + kqsx] = x_dm_ik;
    }
}

template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_q3_K(
    const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int nwarps = mmq_get_nwarps_device();
    constexpr int warp_size = mmq_physical_warp_size();

    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR3_K);
    constexpr int nrows = warp_size / threads_per_row;
    const int kqsx = threadIdx.x % threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nrows*nwarps) {
        int i = i0 + threadIdx.y*nrows + threadIdx.x/threads_per_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q3_K * bxi = (const block_q3_K *) x + kbx0 + i*stride;

        const int x_ql_0 = get_int_b2(bxi->qs,    kqsx);
        const int x_qh_0 = get_int_b2(bxi->hmask, kqsx % (QI3_K/2)) >> (4 * (kqsx / (QI3_K/2)));

#pragma unroll
        for (int l = 0; l < QR3_K; ++l) {
            const int k = (kqsx/8)*32 + l*8 + kqsx % 8;

            const int x_ql_k =  (x_ql_0 >> (2*l))       & 0x03030303;
            const int x_qh_k = ((x_qh_0 >>    l)  << 2) & 0x04040404;

            const int x_qs_k = __vsubss4(x_ql_k | x_qh_k, 0x04040404);

            x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + k] = x_qs_k;
        }
    }

    constexpr int rows_per_warp = warp_size / 4;
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps*rows_per_warp) {
        int i = i0 + threadIdx.y*rows_per_warp + threadIdx.x/4;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q3_K * bxi = (const block_q3_K *) x + kbx0 + i*stride;

        const int ksc = threadIdx.x % 4;

        const int ksc_low = ksc % (QI3_K/8);
        const int shift_low = 4 * (ksc / (QI3_K/8));
        const int sc_low = (get_int_b2(bxi->scales, ksc_low) >> shift_low) & 0x0F0F0F0F;

        const int ksc_high = QI3_K/8;
        const int shift_high = 2 * ksc;
        const int sc_high = ((get_int_b2(bxi->scales, ksc_high) >> shift_high) << 4) & 0x30303030;

        const int sc = __vsubss4(sc_low | sc_high, 0x20202020);

        const int8_t * sc8 = (const int8_t *) &sc;
        const float d = bxi->d;

#pragma unroll
        for (int l = 0; l < int(sizeof(int)); ++l) {
            x_df[i*MMQ_MMA_TILE_X_K_Q3_K + sizeof(int)*ksc + l] = d*sc8[l];
        }
    }

}

static __device__ __forceinline__ int unpack_scales_q45_K(const int * scales, const int ksc) {
    // scale arrangement after the following two lines:
    //   - ksc == 0: sc0, sc1, sc2, sc3
    //   - ksc == 1: sc4, sc5, sc6, sc7
    //   - ksc == 2:  m0,  m1,  m2,  m3
    //   - ksc == 3:  m4,  m5,  m6,  m7
    return ((scales[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F) | // lower 4 bits
           ((scales[ksc/2]              >> (2 * (ksc % 2)))       & 0x30303030);  // upper 2 bits
}

template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_q4_K(
    const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int nwarps = mmq_get_nwarps_device();
    constexpr int warp_size = mmq_physical_warp_size();

    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + 2*MMQ_TILE_NE_K);

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR4_K);
    constexpr int nrows = warp_size / threads_per_row;
    const int txi = warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_K * bxi = (const block_q4_K *) x + kbx0 + i*stride;
        const int qs0 = get_int_b4(bxi->qs, txi);

        x_qs[i*MMQ_MMA_TILE_X_K_Q8_1 + 16*(txi/8) + txi % 8 + 0] = (qs0 >> 0) & 0x0F0F0F0F;
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_1 + 16*(txi/8) + txi % 8 + 8] = (qs0 >> 4) & 0x0F0F0F0F;
    }

    constexpr int rows_per_warp = warp_size / 2;
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps*rows_per_warp) {
        int i = (i0 + threadIdx.y*rows_per_warp + threadIdx.x/2) % mmq_y;
        {
            if (need_check) {
                i = min(i, i_max);
            }

            const block_q4_K * bxi = (const block_q4_K *) x + kbx0 + i*stride;

            const int * scales = (const int *) bxi->scales;
            const int ksc = threadIdx.x % 2;

            const int sc32 = unpack_scales_q45_K(scales, ksc + 0);
            const int  m32 = unpack_scales_q45_K(scales, ksc + 2);

            const uint8_t * sc8 = (const uint8_t *) &sc32;
            const uint8_t *  m8 = (const uint8_t *)  &m32;

            const half2 dm = bxi->dm * make_half2(1.0f, -1.0f);

    #pragma unroll
            for (int l = 0; l < sizeof(int); ++l) {
                x_dm[i*MMQ_MMA_TILE_X_K_Q8_1 + sizeof(int)*ksc + l] = dm*make_half2(sc8[l], m8[l]);
            }
        }
    }
}
