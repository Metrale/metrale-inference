// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-10: w4a16_gemv_seg_r<R>: the one-row NVFP4 W4A16 GEMV (`w4a16_gemv`, w4a16_gemv.cu)
// over up to six weight segments that share the input row, in one persistent launch.
// Segment i: out_i[n] = A[0, :] . W_i[n, :] for n < n_i. An unused segment has n_i = 0.
//
// Weight layout as in w4a16_gemv.cu: packed [N, K/2] (low nibble first), E4M3 scales
// [N, K/16], one FP32 scale2 per tensor.
//
// Owner: gb10 kernels.
// Invariants:
// - Every output's bytes equal `w4a16_gemv`'s for the same A, weight and scale2. Per output the
//   arithmetic is w4a16_gemv_partial's for orig lanes 0..63 (chunks kk = 2l + c + 128j into
//   acc_c, `part` an fmaf chain over the 16 values in nibble order, acc_c = fmaf(scale, part,
//   acc_c), t_l = acc_0 + acc_1), each half of 32 orig lanes reduced by the same shfl_down tree
//   and the output s_0 + s_1, rounded once to BF16. Only the thread mapping and the loads change:
//   one warp per output tile of R rows, lane l holding orig lanes l and l + 32 (as w4a16_gemv_sw);
//   16-byte weight loads (the two chunks of an orig lane are adjacent); A staged once per block
//   in shared memory, permuted so a warp's A reads are contiguous; a grid-stride loop over the
//   tiles of all segments.
// - Contract (ops::w4a16_gemv_seg_shape_ok): K a positive multiple of 32 (an orig lane's two
//   chunks are both in or both out, rows and scale rows 16- and 2-byte aligned), K <= 24576 (the
//   permuted A fits 48 KB of shared memory), A 16-byte aligned. Dynamic shared memory is
//   ceil(K / 2048) * 4096 bytes.
// - CUDA only (the E4M3 scale goes through __nv_fp8_e4m3, as in w4a16_gemv.cu's CUDA build).

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define SEG1_THREADS 256
#define SEG1_WARPS 8

__device__ __constant__ float SEG1_E2M1_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// 2026-10-10: One chunk's `part` (w4a16_gemv_partial's inner loop): 8 packed bytes in w0, w1
// against the 16 BF16 values in a0 (values 0..7) and a1 (8..15).
__device__ __forceinline__ float seg1_part(unsigned int w0, unsigned int w1, uint4 a0, uint4 a1,
                                           const float* __restrict__ lut) {
    const unsigned int a_raw[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
    const unsigned long long packed8 = (unsigned long long)w0 | ((unsigned long long)w1 << 32);
    float part = 0.0f;
    #pragma unroll
    for (int b = 0; b < 8; b++) {
        const unsigned char byte_val = (unsigned char)(packed8 >> (b * 8));
        const float2 af = __bfloat1622float2(*(const __nv_bfloat162*)&a_raw[b]);
        part = fmaf(af.x, lut[byte_val & 0xF], part);
        part = fmaf(af.y, lut[byte_val >> 4], part);
    }
    return part;
}

// 2026-10-10: Weight loads: read once per launch, so not kept in L1, with a 256-byte L2 fetch
// (GB10 streaming read, 19.5 MB: 222 GB/s with __ldg, 230 GB/s with this hint).
__device__ __forceinline__ uint4 seg1_ld16(const unsigned char* p) {
    uint4 v;
    asm("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

__device__ __forceinline__ unsigned short seg1_ld2(const unsigned char* p) {
    unsigned short v;
    asm("ld.global.nc.L1::no_allocate.L2::256B.u16 %0, [%1];" : "=h"(v) : "l"(p));
    return v;
}

__device__ __forceinline__ float seg1_scale(unsigned char b, float scale2) {
    __nv_fp8_e4m3 fp8;
    *(unsigned char*)&fp8 = b;
    return (float)fp8 * scale2;
}

// 2026-10-10: One segment's weight, read from the kernel parameters.
struct Seg1 {
    const unsigned char* packed;
    const unsigned char* scale;
    float scale2;
    __nv_bfloat16* out;
    unsigned int n;
};

template <int R>
__device__ __forceinline__ void w4a16_gemv_seg_body(const __nv_bfloat16* __restrict__ A,
                                                    const Seg1 (&seg)[6], unsigned int K) {
    extern __shared__ uint4 s_a[];
    __shared__ float s_lut[16];

    const unsigned int K16 = K / 16;
    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / 16;
    const unsigned int J = (K16 + 127) / 128;

    // 2026-10-10: Stage A permuted: uint4 u = 2 kk + h of A (h: values 0..7 or 8..15 of chunk
    // kk = 128 j + 2 lp + c) goes to s_a[256 j + 64 (2 c + h) + lp], so lane lp's reads of one
    // (c, h) across a warp are 32 consecutive uint4.
    if (threadIdx.x < 16) s_lut[threadIdx.x] = SEG1_E2M1_LUT[threadIdx.x];
    const uint4* __restrict__ A4 = (const uint4*)A;
    for (unsigned int u = threadIdx.x; u < K16 * 2; u += SEG1_THREADS) {
        const unsigned int kk = u >> 1, h = u & 1u;
        const unsigned int j = kk >> 7, r = kk & 127u;
        s_a[256u * j + 64u * (2u * (r & 1u) + h) + (r >> 1)] = A4[u];
    }
    __syncthreads();

    unsigned int tiles[6];
    unsigned int total = 0;
    #pragma unroll
    for (int i = 0; i < 6; i++) {
        tiles[i] = (seg[i].n + R - 1) / R;
        total += tiles[i];
    }

    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int nw = gridDim.x * SEG1_WARPS;
    for (unsigned int t = blockIdx.x * SEG1_WARPS + (threadIdx.x >> 5); t < total; t += nw) {
        // 2026-10-10: Segment of tile t (warp-uniform).
        unsigned int local = t;
        int si = 0;
        #pragma unroll
        for (int i = 0; i < 5; i++) {
            if (si == i && local >= tiles[i]) {
                local -= tiles[i];
                si = i + 1;
            }
        }
        const unsigned char* packed = seg[0].packed;
        const unsigned char* scale = seg[0].scale;
        float scale2 = seg[0].scale2;
        __nv_bfloat16* out = seg[0].out;
        unsigned int n_seg = seg[0].n;
        #pragma unroll
        for (int i = 1; i < 6; i++) {
            if (si == i) {
                packed = seg[i].packed;
                scale = seg[i].scale;
                scale2 = seg[i].scale2;
                out = seg[i].out;
                n_seg = seg[i].n;
            }
        }
        const unsigned int n0 = local * R;

        float acc[R][2][2];
        #pragma unroll
        for (int r = 0; r < R; r++) {
            acc[r][0][0] = 0.0f; acc[r][0][1] = 0.0f;
            acc[r][1][0] = 0.0f; acc[r][1][1] = 0.0f;
        }

        for (unsigned int j = 0; j < J; j++) {
            uint4 w[R][2];
            unsigned short sc[R][2];
            #pragma unroll
            for (int h = 0; h < 2; h++) {
                const unsigned int lp = lane + 32u * h;
                const unsigned int kk = 128u * j + 2u * lp;
                const bool on = kk < K16;
                #pragma unroll
                for (int r = 0; r < R; r++) {
                    const unsigned int n = n0 + r;
                    if (on && n < n_seg) {
                        w[r][h] = seg1_ld16(packed + (unsigned long long)n * half_K + kk * 8u);
                        sc[r][h] = seg1_ld2(scale + (unsigned long long)n * num_groups + kk);
                    } else {
                        w[r][h] = make_uint4(0u, 0u, 0u, 0u);
                        sc[r][h] = 0;
                    }
                }
            }
            #pragma unroll
            for (int h = 0; h < 2; h++) {
                const unsigned int lp = lane + 32u * h;
                if (128u * j + 2u * lp >= K16) continue;
                const uint4* sa = s_a + 256u * j + lp;
                const uint4 a00 = sa[0], a01 = sa[64], a10 = sa[128], a11 = sa[192];
                #pragma unroll
                for (int r = 0; r < R; r++) {
                    const float p0 = seg1_part(w[r][h].x, w[r][h].y, a00, a01, s_lut);
                    acc[r][h][0] = fmaf(seg1_scale((unsigned char)(sc[r][h] & 0xFFu), scale2), p0, acc[r][h][0]);
                    const float p1 = seg1_part(w[r][h].z, w[r][h].w, a10, a11, s_lut);
                    acc[r][h][1] = fmaf(seg1_scale((unsigned char)(sc[r][h] >> 8), scale2), p1, acc[r][h][1]);
                }
            }
        }

        #pragma unroll
        for (int r = 0; r < R; r++) {
            float s0 = acc[r][0][0] + acc[r][0][1];
            float s1 = acc[r][1][0] + acc[r][1][1];
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                s0 += __shfl_down_sync(0xFFFFFFFFu, s0, offset);
                s1 += __shfl_down_sync(0xFFFFFFFFu, s1, offset);
            }
            const unsigned int n = n0 + r;
            if (lane == 0 && n < n_seg) out[n] = __float2bfloat16(s0 + s1);
        }
    }
}

#define SEG1_PARAMS(i) \
    const unsigned char* __restrict__ p##i, const unsigned char* __restrict__ s##i, \
    float g##i, __nv_bfloat16* __restrict__ c##i, unsigned int n##i

#define SEG1_ENTRY(R)                                                                         \
extern "C" __global__ void __launch_bounds__(SEG1_THREADS) w4a16_gemv_seg_r##R(               \
    const __nv_bfloat16* __restrict__ A,                                                      \
    SEG1_PARAMS(0), SEG1_PARAMS(1), SEG1_PARAMS(2),                                           \
    SEG1_PARAMS(3), SEG1_PARAMS(4), SEG1_PARAMS(5),                                           \
    unsigned int K) {                                                                         \
    const Seg1 seg[6] = {{p0, s0, g0, c0, n0}, {p1, s1, g1, c1, n1}, {p2, s2, g2, c2, n2},   \
                         {p3, s3, g3, c3, n3}, {p4, s4, g4, c4, n4}, {p5, s5, g5, c5, n5}};  \
    w4a16_gemv_seg_body<R>(A, seg, K);                                                        \
}

SEG1_ENTRY(1)
SEG1_ENTRY(2)
SEG1_ENTRY(4)
