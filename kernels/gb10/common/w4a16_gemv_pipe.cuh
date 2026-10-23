// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-23: The w4a16_gemv_pipe template (moved from w4a16_gemv_pipe.cu, unchanged but for the
// input stage): w4a16_gemv_pipe_body<M, D, In> stages its M input rows through `In`, a functor
// whose `(m, u)` is the u-th 16-byte word (8 BF16 values) of input row m. PipeRowsIn reads them
// from M contiguous rows (w4a16_gemv_pipe.cu's entries); an instantiation may compute them
// instead (glm5next_swiglu_down.cu: the clamped SwiGLU of gate and up), so the stage is the
// only change and every output's arithmetic stays the template's.
//
// 2026-10-21: w4a16_gemv_pipe_m<M>: w4a16_gemv_seg_m<M> (w4a16_gemv_seg.cu, R = 1) with each
// warp's weight loads issued D steps ahead across the rows it owns, the first D before the input
// is staged, on a grid the host may cap to the resident blocks (ops::w4a16_gemv_pipe_policy).
//
// Why (GB10, 2026-10-21, graph replay over a rotated pool, M = 2): w4a16_gemv_seg waits on the
// staging of A before its first weight load, and every block stages its own copy of A. On the
// shared expert (one wave of 192 blocks, or K 768 at one step per row) that prologue is not
// hidden by other blocks: gate|up 18.5 -> 17.1 us, down 10.2 -> 9.9 us, KDA o 31.8 -> 31.1 us
// (vs w4a16_gemv_batch2). Deeper D lost at every shape (D 2/3/4 cost registers, so fewer
// resident blocks: gate|up 18.0 / 20.0 / 21.7 us), so the entries are D = 1. With the staging
// and the arithmetic both removed, the same loads take 16.5 us (gate|up) and 9.5 us (down); a
// linear read of the same bytes 15.8 / 8.4 us: what is left is the load pattern and the
// per-launch fixed cost, not the arithmetic.
//
// Owner: gb10 kernels.
// Invariants:
// - Every output's bytes equal w4a16_gemv_seg_m<M>'s (so w4a16_gemv's for each input row alone):
//   per (row m, output n) the same values in the same order: per orig lane l (0..63) chunks
//   kk = 2l + c + 128j into acc_c (`part` an fmaf chain over the 16 values in nibble order,
//   acc_c = fmaf(scale, part, acc_c), j ascending), t_l = acc_0 + acc_1, each half of 32 orig
//   lanes reduced by the shfl_down tree, output s_0 + s_1 rounded once to BF16. Only the load
//   schedule, the grid and the ownership of rows by warps change.
// - Warp w of nw owns the outputs t = w + i nw (all segments' outputs in order), i ascending,
//   each in J = ceil(K / 2048) steps; step q of the warp is (i, j) = (q / J, q % J).
// - D = 1 and 256 threads at __launch_bounds__(256, 4) (64 registers: four resident blocks per
//   SM, ops::W4A16_GEMV_PIPE_BLOCKS_PER_SM).
// - Contract: as w4a16_gemv_seg (ops::w4a16_gemv_seg_rows_ok): K a positive multiple of 32 up
//   to 24576, A 16-byte aligned, dynamic shared memory M ceil(K / 2048) 4096 + 64 bytes.
// - CUDA only (E4M3 through __nv_fp8_e4m3, as w4a16_gemv_seg.cu).
// - 2026-10-21: PDL: the weight and scale loads of the first D steps are issued before
//   pdl_wait(); A, the only input another kernel writes, is read after it.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include "pdl.cuh"

#define PIPE_THREADS 256
#define PIPE_WARPS 8

__device__ __constant__ float PIPE_E2M1_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// 2026-10-21: The loads of w4a16_gemv_seg.cu (seg1_ld16 / seg1_ld2): no L1 allocation, 256-byte
// L2 fetch.
__device__ __forceinline__ uint4 pipe_ld16(const unsigned char* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

__device__ __forceinline__ unsigned short pipe_ld2(const unsigned char* p) {
    unsigned short v;
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.u16 %0, [%1];" : "=h"(v) : "l"(p));
    return v;
}

__device__ __forceinline__ float pipe_scale(unsigned char b, float scale2) {
    __nv_fp8_e4m3 fp8;
    *(unsigned char*)&fp8 = b;
    return (float)fp8 * scale2;
}

// 2026-10-21: seg1_decode / seg1_part_f of w4a16_gemv_seg.cu: the same values, the same order.
__device__ __forceinline__ void pipe_decode(unsigned int w0, unsigned int w1,
                                            const float* __restrict__ lut, float (&wf)[16]) {
    const unsigned long long packed8 = (unsigned long long)w0 | ((unsigned long long)w1 << 32);
    #pragma unroll
    for (int b = 0; b < 8; b++) {
        const unsigned char byte_val = (unsigned char)(packed8 >> (b * 8));
        wf[2 * b] = lut[byte_val & 0xF];
        wf[2 * b + 1] = lut[byte_val >> 4];
    }
}

__device__ __forceinline__ float pipe_part(const float (&wf)[16], uint4 a0, uint4 a1) {
    const unsigned int a_raw[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
    float part = 0.0f;
    #pragma unroll
    for (int b = 0; b < 8; b++) {
        const float2 af = __bfloat1622float2(*(const __nv_bfloat162*)&a_raw[b]);
        part = fmaf(af.x, wf[2 * b], part);
        part = fmaf(af.y, wf[2 * b + 1], part);
    }
    return part;
}

struct PipeSeg {
    const unsigned char* packed;
    const unsigned char* scale;
    float scale2;
    __nv_bfloat16* out;
    unsigned int n;
};

// 2026-10-21: Output t of the launch (warp-uniform): its segment and its row in it.
__device__ __forceinline__ int pipe_locate(unsigned int t, const PipeSeg (&seg)[6],
                                           unsigned int* local) {
    unsigned int l = t;
    int si = 0;
    #pragma unroll
    for (int i = 0; i < 5; i++) {
        if (si == i && l >= seg[i].n) {
            l -= seg[i].n;
            si = i + 1;
        }
    }
    *local = l;
    return si;
}

// 2026-10-21: One step's codes and scale pairs for this lane's two orig lanes (zero past K).
struct PipeStage {
    uint4 w[2];
    unsigned short sc[2];
};

__device__ __forceinline__ void pipe_load(PipeStage& st, const unsigned char* prow,
                                          const unsigned char* srow, unsigned int j,
                                          unsigned int lane, unsigned int K16) {
    #pragma unroll
    for (int h = 0; h < 2; h++) {
        const unsigned int kk = 128u * j + 2u * (lane + 32u * h);
        if (kk < K16) {
            st.w[h] = pipe_ld16(prow + kk * 8u);
            st.sc[h] = pipe_ld2(srow + kk);
        } else {
            st.w[h] = make_uint4(0u, 0u, 0u, 0u);
            st.sc[h] = 0;
        }
    }
}

// 2026-10-21: The load cursor: the row of output i of this warp, step j of it.
struct PipeCursor {
    unsigned int i, j;
    const unsigned char* prow;
    const unsigned char* srow;
};

__device__ __forceinline__ void pipe_row(PipeCursor& c, const PipeSeg (&seg)[6],
                                         unsigned int warp_g, unsigned int nw,
                                         unsigned int half_K, unsigned int num_groups) {
    unsigned int local;
    const int si = pipe_locate(warp_g + c.i * nw, seg, &local);
    const unsigned char* packed = seg[0].packed;
    const unsigned char* scale = seg[0].scale;
    #pragma unroll
    for (int s = 1; s < 6; s++) {
        if (si == s) {
            packed = seg[s].packed;
            scale = seg[s].scale;
        }
    }
    c.prow = packed + (unsigned long long)local * half_K;
    c.srow = scale + (unsigned long long)local * num_groups;
}

__device__ __forceinline__ void pipe_advance(PipeCursor& c, unsigned int J,
                                             const PipeSeg (&seg)[6], unsigned int warp_g,
                                             unsigned int nw, unsigned int half_K,
                                             unsigned int num_groups) {
    if (++c.j == J) {
        c.j = 0;
        c.i++;
        pipe_row(c, seg, warp_g, nw, half_K, num_groups);
    }
}

// 2026-10-23: The plain input stage: M contiguous BF16 rows of K at A.
struct PipeRowsIn {
    const __nv_bfloat16* __restrict__ A;
    unsigned int K;
    __device__ __forceinline__ uint4 operator()(unsigned int m, unsigned int u) const {
        return ((const uint4*)(A + (unsigned long long)m * K))[u];
    }
};

template <int M, int D, class In>
__device__ __forceinline__ void w4a16_gemv_pipe_body(const In in, const PipeSeg (&seg)[6],
                                                     unsigned int K) {
    extern __shared__ uint4 s_a[];

    const unsigned int K16 = K / 16;
    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / 16;
    const unsigned int J = (K16 + 127) / 128;
    float* s_lut = (float*)(s_a + 256u * J * M);
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int nw = gridDim.x * PIPE_WARPS;
    const unsigned int warp_g = blockIdx.x * PIPE_WARPS + (threadIdx.x >> 5);

    unsigned int total = 0;
    #pragma unroll
    for (int i = 0; i < 6; i++) total += seg[i].n;
    const unsigned int rows = warp_g < total ? (total - warp_g + nw - 1) / nw : 0u;
    const unsigned int steps = rows * J;

    // 2026-10-21: Steps 0..D-1 in flight before the staging (they do not read A).
    PipeStage st[D];
    PipeCursor ld = {0u, 0u, nullptr, nullptr};
    if (steps > 0) pipe_row(ld, seg, warp_g, nw, half_K, num_groups);
    #pragma unroll
    for (int s = 0; s < D; s++) {
        if ((unsigned int)s < steps) {
            pipe_load(st[s], ld.prow, ld.srow, ld.j, lane, K16);
            pipe_advance(ld, J, seg, warp_g, nw, half_K, num_groups);
        }
    }

    // 2026-10-21: PDL (pdl.cuh): the loads above read weights only; the input is the
    // predecessor's output, so it is staged after the wait.
    pdl_enter();

    // 2026-10-21: A staged as w4a16_gemv_seg.cu stages it: uint4 u = 2 kk + h of row m at
    // s_a[256 J m + 256 j + 64 (2 c + h) + lp] for kk = 128 j + 2 lp + c.
    if (threadIdx.x < 16) s_lut[threadIdx.x] = PIPE_E2M1_LUT[threadIdx.x];
    #pragma unroll
    for (int m = 0; m < M; m++) {
        uint4* sm = s_a + 256u * J * m;
        for (unsigned int u = threadIdx.x; u < K16 * 2; u += PIPE_THREADS) {
            const unsigned int kk = u >> 1, h = u & 1u;
            const unsigned int j = kk >> 7, r = kk & 127u;
            sm[256u * j + 64u * (2u * (r & 1u) + h) + (r >> 1)] = in((unsigned int)m, u);
        }
    }
    __syncthreads();

    float acc[M][2][2];
    #pragma unroll
    for (int m = 0; m < M; m++) {
        acc[m][0][0] = 0.0f; acc[m][0][1] = 0.0f;
        acc[m][1][0] = 0.0f; acc[m][1][1] = 0.0f;
    }
    unsigned int ci = 0, cj = 0;
    float scale2 = 0.0f;
    __nv_bfloat16* out = nullptr;
    unsigned int n_seg = 0, n = 0;

    for (unsigned int base = 0; base < steps; base += D) {
        #pragma unroll
        for (int s = 0; s < D; s++) {
            if (base + s >= steps) break;
            if (cj == 0) {
                unsigned int local;
                const int si = pipe_locate(warp_g + ci * nw, seg, &local);
                scale2 = seg[0].scale2;
                out = seg[0].out;
                n_seg = seg[0].n;
                #pragma unroll
                for (int q = 1; q < 6; q++) {
                    if (si == q) {
                        scale2 = seg[q].scale2;
                        out = seg[q].out;
                        n_seg = seg[q].n;
                    }
                }
                n = local;
            }
            #pragma unroll
            for (int h = 0; h < 2; h++) {
                const unsigned int lp = lane + 32u * h;
                if (128u * cj + 2u * lp >= K16) continue;
                #pragma unroll
                for (int c = 0; c < 2; c++) {
                    float wf[16];
                    pipe_decode(c ? st[s].w[h].z : st[s].w[h].x, c ? st[s].w[h].w : st[s].w[h].y,
                                s_lut, wf);
                    const float sc_f = pipe_scale(
                        (unsigned char)(c ? (st[s].sc[h] >> 8) : (st[s].sc[h] & 0xFFu)), scale2);
                    #pragma unroll
                    for (int m = 0; m < M; m++) {
                        const uint4* sa = s_a + 256u * J * m + 256u * cj + 128u * c + lp;
                        const float p = pipe_part(wf, sa[0], sa[64]);
                        acc[m][h][c] = fmaf(sc_f, p, acc[m][h][c]);
                    }
                }
            }
            if (base + s + D < steps) {
                pipe_load(st[s], ld.prow, ld.srow, ld.j, lane, K16);
                pipe_advance(ld, J, seg, warp_g, nw, half_K, num_groups);
            }
            if (++cj == J) {
                #pragma unroll
                for (int m = 0; m < M; m++) {
                    float s0 = acc[m][0][0] + acc[m][0][1];
                    float s1 = acc[m][1][0] + acc[m][1][1];
                    #pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1) {
                        s0 += __shfl_down_sync(0xFFFFFFFFu, s0, offset);
                        s1 += __shfl_down_sync(0xFFFFFFFFu, s1, offset);
                    }
                    if (lane == 0) out[(unsigned long long)m * n_seg + n] = __float2bfloat16(s0 + s1);
                    acc[m][0][0] = 0.0f; acc[m][0][1] = 0.0f;
                    acc[m][1][0] = 0.0f; acc[m][1][1] = 0.0f;
                }
                cj = 0;
                ci++;
            }
        }
    }
}

#define PIPE_PARAMS(i) \
    const unsigned char* __restrict__ p##i, const unsigned char* __restrict__ s##i, \
    float g##i, __nv_bfloat16* __restrict__ c##i, unsigned int n##i

// 2026-10-23: The six segments' parameters of an entry, and the array the body takes.
#define PIPE_SEG_PARAMS \
    PIPE_PARAMS(0), PIPE_PARAMS(1), PIPE_PARAMS(2), PIPE_PARAMS(3), PIPE_PARAMS(4), PIPE_PARAMS(5)

#define PIPE_SEG_ARRAY(seg)                                                                    \
    const PipeSeg seg[6] = {{p0, s0, g0, c0, n0}, {p1, s1, g1, c1, n1}, {p2, s2, g2, c2, n2}, \
                            {p3, s3, g3, c3, n3}, {p4, s4, g4, c4, n4}, {p5, s5, g5, c5, n5}}
