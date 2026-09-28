// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: C[M, N] = A[M, K] * B[N, K]^T with A and B FP8 E4M3 bytes, FP32 accumulation
// (mma m16n8k32 e4m3) and a BF16 C; no scale is applied.
//
// Owner: gb10 kernels.
// Invariants:
// - K is a multiple of 32 (the launcher checks it).
// - Each output accumulates its m16n8k32 MMAs in increasing k from 0.0f, one per 32-wide k
//   slice, then rounds once to BF16. The tile shape, the pipeline and the grid order do not
//   change that sequence, so the output is bit-identical to the 2026-09-25 single-stage kernel.
//
// ops::fp8_gemm_n128 (model-layers gemm_fp8_prefill.rs) launches it when K is a multiple of
// 32 and METRALE_FP8_LDMAB is not 0, after casting A to E4M3 with bf16_to_fp8; B is the
// pre-dequantized E4M3 weight.
//
// 2026-09-28: A 128x256 CTA tile on 8 warps (2 x 4, 64x64 each), K stages of 128 bytes
// double-buffered through cp.async.cg into XOR-swizzled shared memory (98,304 B dynamic),
// ldmatrix.x4 fragments, and a 1-D grid in groups of 8 M tiles so a wave's A and B tiles stay
// in L2. The 128x128 single-stage kernel it replaces ran the dense 27B's GDN qkvz projection
// (8192 x 16384 x 5120) at 27.8 TFLOPS; this runs it at 172 (standalone, dgx2).
//
// It is a module of its own (w4a16_fp8_ldmab), not part of w4a16_gemm.cu, because a model's
// own w4a16_gemm.cu replaces the common one whole (crates/kernels/build.rs,
// shadowed_dropped_pairs), which would drop this kernel for that model.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define LDMAB_BM 128
#define LDMAB_BN 256
#define LDMAB_WM 64
#define LDMAB_WN 64
#define LDMAB_BK 128
#define LDMAB_STAGES 2
#define LDMAB_GROUP_M 8
#define LDMAB_THREADS ((LDMAB_BM / LDMAB_WM) * (LDMAB_BN / LDMAB_WN) * 32)

// 2026-09-28: 16-byte cp.async.cg; `pred` false zero-fills the destination and reads nothing.
__device__ __forceinline__ void ldmab_cp16(unsigned int dst, const void* src, bool pred) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(pred ? 16 : 0));
}

// 2026-09-28: A tile row is LDMAB_BK = 128 bytes, eight 16-byte chunks; chunk c of row r is stored at chunk
// (c ^ (r & 7)), so the eight rows one ldmatrix phase reads at a logical chunk land in eight bank groups.
__device__ __forceinline__ unsigned int ldmab_swz(int row, int chunk) {
    return row * LDMAB_BK + ((chunk ^ (row & 7)) << 4);
}

__device__ __forceinline__ void ldmab_ldsm_x4(unsigned int addr, unsigned int& r0, unsigned int& r1,
                                              unsigned int& r2, unsigned int& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3},[%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

__device__ __forceinline__ void ldmab_mma(float* d, const unsigned int* a, unsigned int b0, unsigned int b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                 "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// 2026-09-28: Grid (ceil(M/128) * ceil(N/256), 1, 1), block 256, dynamic shared memory
// LDMAB_STAGES * (LDMAB_BM + LDMAB_BN) * LDMAB_BK bytes.
extern "C" __global__ void __launch_bounds__(LDMAB_THREADS, 1) fp8_fp8_gemm_ldmab(
    const unsigned char* __restrict__ A,
    const unsigned char* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    constexpr int NWN = LDMAB_BN / LDMAB_WN;
    constexpr int MT = LDMAB_WM / 16, NT = LDMAB_WN / 8;
    extern __shared__ __align__(128) unsigned char ldmab_smem[];
    const unsigned int s_a = (unsigned int)__cvta_generic_to_shared(ldmab_smem);
    const unsigned int s_b = s_a + LDMAB_STAGES * LDMAB_BM * LDMAB_BK;

    // 2026-09-28: Grouped order: consecutive CTAs walk LDMAB_GROUP_M M tiles before the next N tile.
    const unsigned int grid_m = (M + LDMAB_BM - 1) / LDMAB_BM, grid_n = (N + LDMAB_BN - 1) / LDMAB_BN;
    const unsigned int in_group = LDMAB_GROUP_M * grid_n;
    const unsigned int first_m = (blockIdx.x / in_group) * LDMAB_GROUP_M;
    const unsigned int gsz = min(grid_m - first_m, (unsigned int)LDMAB_GROUP_M);
    const unsigned int m0 = (first_m + (blockIdx.x % in_group) % gsz) * LDMAB_BM;
    const unsigned int n0 = ((blockIdx.x % in_group) / gsz) * LDMAB_BN;

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / NWN, wn = warp % NWN;

    float acc[MT][NT][4];
#pragma unroll
    for (int i = 0; i < MT; i++)
#pragma unroll
        for (int j = 0; j < NT; j++) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

    const unsigned int ktiles = (K + LDMAB_BK - 1) / LDMAB_BK;
    auto load = [&](unsigned int kt, int stage) {
        const unsigned int k0 = kt * LDMAB_BK;
#pragma unroll
        for (int c = tid; c < LDMAB_BM * 8; c += LDMAB_THREADS) {
            const int r = c >> 3, ch = c & 7;
            const unsigned int gr = m0 + r, gk = k0 + ch * 16;
            const bool ok = gr < M && gk < K;
            ldmab_cp16(s_a + stage * LDMAB_BM * LDMAB_BK + ldmab_swz(r, ch),
                       ok ? (const void*)(A + (unsigned long long)gr * K + gk) : (const void*)A, ok);
        }
#pragma unroll
        for (int c = tid; c < LDMAB_BN * 8; c += LDMAB_THREADS) {
            const int r = c >> 3, ch = c & 7;
            const unsigned int gn = n0 + r, gk = k0 + ch * 16;
            const bool ok = gn < N && gk < K;
            ldmab_cp16(s_b + stage * LDMAB_BN * LDMAB_BK + ldmab_swz(r, ch),
                       ok ? (const void*)(B + (unsigned long long)gn * K + gk) : (const void*)B, ok);
        }
    };

#pragma unroll
    for (int s = 0; s < LDMAB_STAGES - 1; s++) {
        if ((unsigned int)s < ktiles) load(s, s);
        asm volatile("cp.async.commit_group;\n" ::);
    }
    for (unsigned int kt = 0; kt < ktiles; kt++) {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(LDMAB_STAGES - 2));
        __syncthreads();   // 2026-09-28: stage kt is resident, and every read of the stage refilled below has ended.
        {
            const unsigned int nk = kt + LDMAB_STAGES - 1;
            if (nk < ktiles) load(nk, nk % LDMAB_STAGES);
            asm volatile("cp.async.commit_group;\n" ::);
        }
        const int st = kt % LDMAB_STAGES;
        const unsigned int t_a = s_a + st * LDMAB_BM * LDMAB_BK, t_b = s_b + st * LDMAB_BN * LDMAB_BK;
#pragma unroll
        for (int ks = 0; ks < LDMAB_BK / 32; ks++) {
            if (kt * LDMAB_BK + ks * 32 >= K) break;
            unsigned int a[MT][4], b[NT][2];
#pragma unroll
            for (int mt = 0; mt < MT; mt++) {
                // 2026-09-28: lanes 0-15 rows 0-15 at bytes 0-15 of the k slice, lanes 16-31 at bytes 16-31.
                const int row = wm * LDMAB_WM + mt * 16 + (lane & 15);
                ldmab_ldsm_x4(t_a + ldmab_swz(row, ks * 2 + (lane >> 4)), a[mt][0], a[mt][1], a[mt][2], a[mt][3]);
            }
#pragma unroll
            for (int p = 0; p < NT / 2; p++) {
                // 2026-09-28: lanes 0-7 / 8-15: n tile 2p at k bytes 0-15 / 16-31; lanes 16-31: n tile 2p+1.
                const int row = wn * LDMAB_WN + p * 16 + ((lane >> 4) << 3) + (lane & 7);
                ldmab_ldsm_x4(t_b + ldmab_swz(row, ks * 2 + ((lane >> 3) & 1)),
                              b[2 * p][0], b[2 * p][1], b[2 * p + 1][0], b[2 * p + 1][1]);
            }
#pragma unroll
            for (int mt = 0; mt < MT; mt++)
#pragma unroll
                for (int nt = 0; nt < NT; nt++) ldmab_mma(acc[mt][nt], a[mt], b[nt][0], b[nt][1]);
        }
    }
    asm volatile("cp.async.wait_group 0;\n" ::);

    const int g = lane >> 2, t4 = lane & 3;
#pragma unroll
    for (int mt = 0; mt < MT; mt++) {
#pragma unroll
        for (int nt = 0; nt < NT; nt++) {
            const unsigned int c0 = n0 + wn * LDMAB_WN + nt * 8 + t4 * 2;
            const unsigned int r0 = m0 + wm * LDMAB_WM + mt * 16 + g, r1 = r0 + 8;
            if (c0 + 1 < N && (N & 1u) == 0u) {
                if (r0 < M) *(__nv_bfloat162*)&C[(unsigned long long)r0 * N + c0] = __floats2bfloat162_rn(acc[mt][nt][0], acc[mt][nt][1]);
                if (r1 < M) *(__nv_bfloat162*)&C[(unsigned long long)r1 * N + c0] = __floats2bfloat162_rn(acc[mt][nt][2], acc[mt][nt][3]);
            } else {
                if (r0 < M && c0 < N) C[(unsigned long long)r0 * N + c0] = __float2bfloat16(acc[mt][nt][0]);
                if (r0 < M && c0 + 1 < N) C[(unsigned long long)r0 * N + c0 + 1] = __float2bfloat16(acc[mt][nt][1]);
                if (r1 < M && c0 < N) C[(unsigned long long)r1 * N + c0] = __float2bfloat16(acc[mt][nt][2]);
                if (r1 < M && c0 + 1 < N) C[(unsigned long long)r1 * N + c0 + 1] = __float2bfloat16(acc[mt][nt][3]);
            }
        }
    }
}

// 2026-09-28: fp8_predequant_nvfp4_t: the transposed NVFP4 weight that w4a16_gemm_t_m128 reads (B_packed [K/2][N],
// byte (kp, n) = k 2kp in the low nibble and 2kp+1 in the high one; B_scale [K/16][N] E4M3) to E4M3 B_fp8 [N][K],
// with that kernel's dequant arithmetic: e2m1 value * (float(e4m3 scale) * scale2), rounded by
// cvt.rn.satfinite.e4m3x2.f32. Casting A with bf16_to_fp8 and running fp8_fp8_gemm_ldmab on the result then gives
// w4a16_gemm_t_m128's output bit for bit. One thread per (n, 16-wide k group); consecutive threads take consecutive
// n, so the packed and scale reads coalesce. Grid (ceil(N / 256), K / 16), block 256. K % 16 == 0.
__device__ __constant__ float ldmab_e2m1_lut[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

extern "C" __global__ void fp8_predequant_nvfp4_t(
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    unsigned char* __restrict__ B_fp8,
    unsigned int N, unsigned int K
) {
    const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int q = blockIdx.y;
    if (n >= N) return;
    __nv_fp8_e4m3 f;
    *(unsigned char*)&f = B_scale[(unsigned long long)q * N + n];
    const float sv = (float)f * scale2;
    unsigned int w[4];
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const unsigned char p0 = B_packed[(unsigned long long)(q * 8 + 2 * j) * N + n];
        const unsigned char p1 = B_packed[(unsigned long long)(q * 8 + 2 * j + 1) * N + n];
        unsigned short h0, h1;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(h0)
            : "f"(ldmab_e2m1_lut[p0 >> 4] * sv), "f"(ldmab_e2m1_lut[p0 & 0xF] * sv));
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(h1)
            : "f"(ldmab_e2m1_lut[p1 >> 4] * sv), "f"(ldmab_e2m1_lut[p1 & 0xF] * sv));
        w[j] = (unsigned int)h0 | ((unsigned int)h1 << 16);
    }
    *(uint4*)&B_fp8[(unsigned long long)n * K + q * 16] = make_uint4(w[0], w[1], w[2], w[3]);
}
