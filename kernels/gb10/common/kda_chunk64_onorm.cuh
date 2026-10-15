// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-15: The sigmoid-gated output RMSNorm of the KDA layer as the epilogue of the 64-token
// chunked scan (kda_chunk64.cu, entry kda_chunk64_scan_onorm, `METRALE_GLM_KDA_SCAN_ONORM`):
// every output row the scan produces is normed, gated and written as BF16 in the scan itself,
// so the FP32 core rows never reach DRAM and kda_o_norm_gated_bf16 (kda_layer_ops.cu) does not
// run.
//
// Byte identity with scan + kda_o_norm_gated_bf16: the norm's block of 128 threads holds one
// element per thread, sums squares warp by warp with shfl_down 16, 8, 4, 2, 1, and adds the four
// warp sums as (w0 + w2) + (w1 + w3). A scan warp of phase C holds 16 rows x 32 consecutive
// columns (one norm warp's 32 columns) in the m16n8 accumulator layout: column
// l = 8 nt + 2 tig + e of its segment sits in lane (g, tig), n-tile nt, element e. So the norm's
// tree over the bits of l, high to low, is: bit 4 (nt with nt + 2) and bit 3 (nt 0 with 1) in the
// lane, bits 2 and 1 (tig) across lanes by xor shuffle, bit 0 (e) in the lane. Float addition is
// commutative, so each pair adds to the same bits in either operand order. The two segments of a
// head's other 64 columns live in the other block of its (1, 2, 1) cluster; each block publishes
// its two segment sums per row in shared memory and reads its peer's over DSMEM. Then
// inv = rsqrtf(sum / 128 + eps) and out = bf16(((x * inv) * w) * s), s = 1 / (1 + __expf(-gate)),
// the norm's expressions in its order; the module is built with the same flags (--fmad=false).
//
// Owner: gb10 kernels.
// Invariants:
// - The two blocks of a head form one thread-block cluster: the hardware schedules both or
//   neither, so the per-chunk cluster barrier never waits on a block that is not resident (a
//   cluster that cannot be placed fails the launch; it never hangs).
// - The partial sums are double-buffered by chunk parity: a block's reads of chunk c's buffer
//   come before its arrival at chunk c + 1's barrier, and chunk c + 2 overwrites that buffer only
//   after that barrier; a final barrier keeps each block resident until its peer's last read.
// - Every thread of both blocks reaches each barrier: rows t >= T are not stored, but their
//   lanes still arrive.

#ifndef KDA_CHUNK64_ONORM_CUH
#define KDA_CHUNK64_ONORM_CUH

#include <cooperative_groups.h>
#include <cuda_bf16.h>

#include "kda_chunk64.cuh"

// 2026-10-15: Floats of the partial-sum buffer: [parity 2][segment 2][row KDA64_C].
#define KDA64_ONORM_PART_FLOATS (2 * 2 * KDA64_C)
#define KDA64_SCAN_ONORM_SMEM (KDA64_SCAN_SMEM + KDA64_ONORM_PART_FLOATS * 4)

// 2026-10-15: The norm's operands: the output gate and the output, BF16 [T, out_stride] at column
// h * D + v; the norm weight, BF16 [D]; the RMS eps.
struct K64Onorm {
    const __nv_bfloat16* __restrict__ gate;
    const __nv_bfloat16* __restrict__ weight;
    __nv_bfloat16* __restrict__ out;
    float eps;
};

__device__ __forceinline__ void k64_cluster_arrive() {
    asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void k64_cluster_wait() {
    asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory");
}

// 2026-10-15: kda_o_norm_gated_bf16's first reduction over one segment (32 columns) of the two
// rows hr = 0, 1 this lane holds: sum[hr] in every lane of the quad.
template <int NTW>
__device__ __forceinline__ void k64_onorm_seg_sums(const float (&o)[NTW][4], float (&sum)[2]) {
    static_assert(NTW == 4, "a segment is four n-tiles of 8 columns");
#pragma unroll
    for (int hr = 0; hr < 2; ++hr) {
        float s8[2];
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            float sq[NTW];
#pragma unroll
            for (int nt = 0; nt < NTW; ++nt)
                sq[nt] = __fmul_rn(o[nt][hr * 2 + e], o[nt][hr * 2 + e]);
            // 2026-10-15: shfl_down 16 (nt with nt + 2), then 8 (nt 0 with 1).
            s8[e] = __fadd_rn(__fadd_rn(sq[0], sq[2]), __fadd_rn(sq[1], sq[3]));
        }
        float s2[2];
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            // 2026-10-15: shfl_down 4 (tig with tig + 2), then 2 (tig 0 with 1).
            const float s4 = __fadd_rn(s8[e], __shfl_xor_sync(0xffffffffu, s8[e], 2));
            s2[e] = __fadd_rn(s4, __shfl_xor_sync(0xffffffffu, s4, 1));
        }
        // 2026-10-15: shfl_down 1 (e 0 with 1).
        sum[hr] = __fadd_rn(s2[0], s2[1]);
    }
}

// 2026-10-15: The epilogue of one chunk: this warp's 16 rows x 32 columns `o` (rows r0 + g + 8 hr,
// columns v0 + cw + 8 nt + 2 tig + e of head h; token t0 + r0 + g + 8 hr), normed against the
// whole 128-column row and stored as BF16. `part` is this block's partial-sum buffer, `wv` the
// norm weight at this lane's columns. Every thread of the block calls it once per chunk.
template <int NTW>
__device__ __forceinline__ void k64_onorm_epilogue(const float (&o)[NTW][4], const K64Onorm& on,
                                                   float* part, const float (&wv)[NTW][2],
                                                   unsigned int parity, unsigned int h,
                                                   unsigned int v0, unsigned int cw,
                                                   unsigned int r0, unsigned int g,
                                                   unsigned int tig, unsigned int t0,
                                                   unsigned int T, unsigned int out_stride) {
    constexpr int C = KDA64_C, D = KDA64_D;
    cooperative_groups::cluster_group cl = cooperative_groups::this_cluster();
    const unsigned int rank = cl.block_rank(), seg = cw >> 5;
    float* mine = part + parity * (2u * C);
    float sum[2];
    k64_onorm_seg_sums(o, sum);
    if (tig == 0u) {
        mine[seg * C + r0 + g] = sum[0];
        mine[seg * C + r0 + g + 8u] = sum[1];
    }
    k64_cluster_arrive();
    // 2026-10-15: The gate's sigmoid does not need the sums: computed while the peer arrives.
    float sg[NTW][4];
#pragma unroll
    for (int hr = 0; hr < 2; ++hr) {
        const unsigned int t = t0 + r0 + g + (unsigned int)hr * 8u;
#pragma unroll
        for (int nt = 0; nt < NTW; ++nt) {
            float2 gv = make_float2(0.0f, 0.0f);
            if (t < T) {
                gv = __bfloat1622float2(*(const __nv_bfloat162*)(on.gate + (size_t)t * out_stride +
                                                                  (size_t)h * D + v0 + cw +
                                                                  (unsigned int)nt * 8u + 2u * tig));
            }
            sg[nt][hr * 2] = 1.0f / (1.0f + __expf(-gv.x));
            sg[nt][hr * 2 + 1] = 1.0f / (1.0f + __expf(-gv.y));
        }
    }
    k64_cluster_wait();
    const float* peer = cl.map_shared_rank(part, rank ^ 1u) + parity * (2u * C);
    const float* lo = rank == 0u ? mine : peer;
    const float* hi = rank == 0u ? peer : mine;
#pragma unroll
    for (int hr = 0; hr < 2; ++hr) {
        const unsigned int row = r0 + g + (unsigned int)hr * 8u, t = t0 + row;
        // 2026-10-15: Warps w0, w1 are block 0's segments, w2, w3 block 1's: (w0 + w2) + (w1 + w3).
        const float tot = __fadd_rn(__fadd_rn(lo[row], hi[row]), __fadd_rn(lo[C + row], hi[C + row]));
        const float inv = rsqrtf(__fadd_rn(tot / (float)D, on.eps));
        if (t < T) {
#pragma unroll
            for (int nt = 0; nt < NTW; ++nt) {
                const float a = __fmul_rn(__fmul_rn(__fmul_rn(o[nt][hr * 2], inv), wv[nt][0]),
                                          sg[nt][hr * 2]);
                const float b = __fmul_rn(__fmul_rn(__fmul_rn(o[nt][hr * 2 + 1], inv), wv[nt][1]),
                                          sg[nt][hr * 2 + 1]);
                __nv_bfloat162 r;
                r.x = __float2bfloat16(a);
                r.y = __float2bfloat16(b);
                *(__nv_bfloat162*)(on.out + (size_t)t * out_stride + (size_t)h * D + v0 + cw +
                                   (unsigned int)nt * 8u + 2u * tig) = r;
            }
        }
    }
}

#endif
