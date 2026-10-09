// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Dense BF16 GEMV over M activation rows in one pass over the weight:
// C[t, n] = sum_k A[t, k] * B[n, k] for t in [0, M).
//
// Owner: gb10 kernels.
// Invariants:
// - A is [M, K] contiguous, B is [N, K] row-major, row t of C starts at
//   C + t * out_stride (output elements: BF16, or FP32 for the _fp32out entry).
// - Launch: grid (ceil(N / 4), Y, 1), block (256, 1, 1); 64 threads (2 warps) per output.
//   2026-09-27: Y > 1 splits the rows over block rows of ceil(M / Y) each.
// - Only the first min(ceil(M / Y), MAX_M) rows of a block row are computed; the host
//   wrappers refuse more.
// - Assumes K % 8 == 0: rows of A and B start at byte 2 * row * K, which is 16-byte
//   aligned for the uint4 loads only then. The scalar tail covers K % 8, not that alignment.
// - Each row's result is bit-identical to dense_gemv_bf16 on that row: the same kv order
//   (stride 64), the same lo-then-hi add order, the same warp and cross-warp reduction,
//   and the common build passes --fmad=false. Staging A through shared memory changes
//   where the operands are read from, not their values or order.
//
// The four output groups of a block walk the same kv sequence, so each 64-vector slab of
// every A row is staged once per block in shared memory and read by all four groups.




































#include <cuda_bf16.h>

#define BLOCK_SIZE 256
#define N_PER_BLOCK 4
#define WARP_SIZE 32
#define VEC_SIZE 8
// 2026-09-25: Compile-time cap on batched rows, mirrored by DENSE_GEMV_BATCHM_MAX_M in
// the host wrapper. acc[t] is an independent FP32 chain per row and m enters no row's
// operand order, so every M up to the cap gives each row the same bits.
// Shared memory: As is MAX_M * 64 * 16 B = 16 KB, plus 512 B for the fold.















#define MAX_M 16

// 2026-10-09: The body, parameterized on the output element. `store_out` is the only
// difference between the entry points: BF16 rounds the final FP32 sum once, FP32 stores it.
__device__ __forceinline__ void store_out(__nv_bfloat16* C, unsigned long long i, float r) {
    C[i] = __float2bfloat16(r);
}
__device__ __forceinline__ void store_out(float* C, unsigned long long i, float r) {
    C[i] = r;
}

template <typename OutT>
__device__ __forceinline__ void dense_gemv_bf16_batchm_body(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    OutT* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int out_stride
) {
    // 2026-09-27: With gridDim.y > 1, block row y takes rows [y * R, min((y + 1) * R, M)),
    // R = ceil(M / gridDim.y); each row's arithmetic is the same in every block.
    const unsigned int rows_per_y = (M + gridDim.y - 1) / gridDim.y;
    const unsigned int r0 = blockIdx.y * rows_per_y;
    if (r0 >= M) return;
    A += (unsigned long long)r0 * K;
    C += (unsigned long long)r0 * out_stride;
    M = min(rows_per_y, M - r0);
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    // 2026-09-25: A mask, not a return: every thread of a partial last block has to reach
    // the __syncthreads() calls in the staging loop.
    const bool active = (n < N);

    const unsigned int m = (M > MAX_M) ? MAX_M : M;

    float acc[MAX_M];
    #pragma unroll
    for (int t = 0; t < MAX_M; t++) acc[t] = 0.0f;

    const unsigned int K_VEC = K / VEC_SIZE;
    const uint4* B_vec = (const uint4*)(B + (unsigned long long)(active ? n : 0) * K);

    // 2026-09-25: One 64-vector slab of every A row, shared by the four output groups.

    __shared__ uint4 As[MAX_M][BLOCK_SIZE / N_PER_BLOCK];

    for (unsigned int base = 0; base < K_VEC; base += threads_per_out) {

        for (unsigned int idx = threadIdx.x; idx < m * threads_per_out; idx += BLOCK_SIZE) {
            const unsigned int t = idx / threads_per_out;
            const unsigned int l = idx % threads_per_out;
            const unsigned int kv = base + l;
            if (kv < K_VEC) {
                As[t][l] = ((const uint4*)(A + (unsigned long long)t * K))[kv];
            }
        }
        __syncthreads();

        const unsigned int kv = base + lane;
        if (kv < K_VEC && active) {

            uint4 b_data = B_vec[kv];
            const unsigned int b_raw[4] = {b_data.x, b_data.y, b_data.z, b_data.w};

            float bf[8];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                __nv_bfloat16 b_lo, b_hi;
                *(unsigned short*)&b_lo = (unsigned short)(b_raw[i] & 0xFFFF);
                *(unsigned short*)&b_hi = (unsigned short)(b_raw[i] >> 16);
                bf[2 * i] = __bfloat162float(b_lo);
                bf[2 * i + 1] = __bfloat162float(b_hi);
            }

            for (unsigned int t = 0; t < m; t++) {
                uint4 a_data = As[t][lane];
                const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};
                float a = acc[t];
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    __nv_bfloat16 a_lo, a_hi;
                    *(unsigned short*)&a_lo = (unsigned short)(a_raw[i] & 0xFFFF);
                    *(unsigned short*)&a_hi = (unsigned short)(a_raw[i] >> 16);

                    a += __bfloat162float(a_lo) * bf[2 * i];
                    a += __bfloat162float(a_hi) * bf[2 * i + 1];
                }
                acc[t] = a;
            }
        }
        // 2026-09-25: Before the next slab overwrites what the compute above still reads.
        __syncthreads();
    }

    // 2026-09-25: Scalar tail for the last K % 8 elements.




    if (active) {
        const unsigned int tail_start = K_VEC * VEC_SIZE;
        const __nv_bfloat16* B_row = B + (unsigned long long)n * K;
        for (unsigned int k = tail_start + lane; k < K; k += threads_per_out) {
            const float bfv = __bfloat162float(B_row[k]);
            for (unsigned int t = 0; t < m; t++) {
                acc[t] += __bfloat162float(A[(unsigned long long)t * K + k]) * bfv;
            }
        }
    }

    if (!active) return;

    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;

    for (unsigned int t = 0; t < m; t++) {
        float a = acc[t];
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            a += __shfl_down_sync(0xFFFFFFFF, a, offset);
        }
        acc[t] = a;
    }

    // 2026-09-25: Two warps per output: add the warp partials through shared memory, per row.
    __shared__ float smem[MAX_M][N_PER_BLOCK * 2];

    if (warp_lane == 0) {
        const unsigned int smem_idx = local_out * 2 + (lane / WARP_SIZE);
        for (unsigned int t = 0; t < m; t++) smem[t][smem_idx] = acc[t];
    }
    __syncthreads();

    if (lane == 0) {
        for (unsigned int t = 0; t < m; t++) {
            const float r = smem[t][local_out * 2] + smem[t][local_out * 2 + 1];
            store_out(C, (unsigned long long)t * out_stride + n, r);
        }
    }
}

extern "C" __global__ void dense_gemv_bf16_batchm(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int out_stride
) {
    dense_gemv_bf16_batchm_body<__nv_bfloat16>(A, B, C, M, N, K, out_stride);
}

// 2026-10-09: FP32-output twin, as dense_gemv_bf16_fp32out is dense_gemv_bf16's: each row's
// FP32 sum is the one dense_gemv_bf16_fp32out stores for that row (the same arithmetic as
// above), stored without rounding. The GLM-5.3 DSA indexer's query and head weights use it
// for several rows at once (glm5next_dsa/layer/decode_rows.rs).
extern "C" __global__ void dense_gemv_bf16_batchm_fp32out(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    float* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int out_stride
) {
    dense_gemv_bf16_batchm_body<float>(A, B, C, M, N, K, out_stride);
}

// 2026-10-09: The batched GEMV at a compile-time row count MM. In the runtime-M body above,
// `acc[t]` is indexed by a loop variable the compiler cannot unroll, so the 16 accumulators
// live in local memory and every FMA round-trips through it (measured 2026-10-09 on GLM-5.3
// at M = 16: 127 us a call against 54 us for the M = 1 GEMV). Here every row loop is
// unrolled and the accumulators stay in registers. The arithmetic is the body's, operation for
// operation: the same kv order (stride 64), the same lo-then-hi adds per row, the same scalar
// tail, warp shuffle and cross-warp sum, and one store per row, so each row's bits are the
// runtime-M body's (and dense_gemv_bf16's). Only the GLM-5.3 layers launch it, at
// M = BATCHM_WIDE_MIN..MAX_M (glm5next_layer/wide_gemv.rs); every other M and caller keeps
// the entries above. Launch: grid (ceil(N / 4), 1, 1), block (256, 1, 1).
#define BATCHM_WIDE_MIN 9

template <int MM, typename OutT>
__device__ __forceinline__ void dense_gemv_bf16_batchm_fixed(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    OutT* __restrict__ C,
    unsigned int N,
    unsigned int K,
    unsigned int out_stride,
    uint4 (*As)[BLOCK_SIZE / N_PER_BLOCK],
    float (*smem)[N_PER_BLOCK * 2]
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;
    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    const bool active = (n < N);

    float acc[MM];
    #pragma unroll
    for (int t = 0; t < MM; t++) acc[t] = 0.0f;

    const unsigned int K_VEC = K / VEC_SIZE;
    const uint4* B_vec = (const uint4*)(B + (unsigned long long)(active ? n : 0) * K);

    for (unsigned int base = 0; base < K_VEC; base += threads_per_out) {
        for (unsigned int idx = threadIdx.x; idx < MM * threads_per_out; idx += BLOCK_SIZE) {
            const unsigned int t = idx / threads_per_out;
            const unsigned int l = idx % threads_per_out;
            const unsigned int kv = base + l;
            if (kv < K_VEC) {
                As[t][l] = ((const uint4*)(A + (unsigned long long)t * K))[kv];
            }
        }
        __syncthreads();

        const unsigned int kv = base + lane;
        if (kv < K_VEC && active) {
            uint4 b_data = B_vec[kv];
            const unsigned int b_raw[4] = {b_data.x, b_data.y, b_data.z, b_data.w};
            float bf[8];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                __nv_bfloat16 b_lo, b_hi;
                *(unsigned short*)&b_lo = (unsigned short)(b_raw[i] & 0xFFFF);
                *(unsigned short*)&b_hi = (unsigned short)(b_raw[i] >> 16);
                bf[2 * i] = __bfloat162float(b_lo);
                bf[2 * i + 1] = __bfloat162float(b_hi);
            }
            #pragma unroll
            for (int t = 0; t < MM; t++) {
                uint4 a_data = As[t][lane];
                const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};
                float a = acc[t];
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    __nv_bfloat16 a_lo, a_hi;
                    *(unsigned short*)&a_lo = (unsigned short)(a_raw[i] & 0xFFFF);
                    *(unsigned short*)&a_hi = (unsigned short)(a_raw[i] >> 16);
                    a += __bfloat162float(a_lo) * bf[2 * i];
                    a += __bfloat162float(a_hi) * bf[2 * i + 1];
                }
                acc[t] = a;
            }
        }
        __syncthreads();
    }

    if (active) {
        const unsigned int tail_start = K_VEC * VEC_SIZE;
        const __nv_bfloat16* B_row = B + (unsigned long long)n * K;
        for (unsigned int k = tail_start + lane; k < K; k += threads_per_out) {
            const float bfv = __bfloat162float(B_row[k]);
            #pragma unroll
            for (int t = 0; t < MM; t++) {
                acc[t] += __bfloat162float(A[(unsigned long long)t * K + k]) * bfv;
            }
        }
    }

    if (!active) return;

    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;
    #pragma unroll
    for (int t = 0; t < MM; t++) {
        float a = acc[t];
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            a += __shfl_down_sync(0xFFFFFFFF, a, offset);
        }
        acc[t] = a;
    }
    if (warp_lane == 0) {
        const unsigned int smem_idx = local_out * 2 + (lane / WARP_SIZE);
        #pragma unroll
        for (int t = 0; t < MM; t++) smem[t][smem_idx] = acc[t];
    }
    __syncthreads();
    if (lane == 0) {
        #pragma unroll
        for (int t = 0; t < MM; t++) {
            const float r = smem[t][local_out * 2] + smem[t][local_out * 2 + 1];
            store_out(C, (unsigned long long)t * out_stride + n, r);
        }
    }
}

// 2026-10-09: The shared arrays are declared once here, not in the template, so the eight
// instantiations share one allocation instead of eight.
template <typename OutT>
__device__ __forceinline__ void dense_gemv_bf16_batchm_wide_body(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    OutT* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int out_stride
) {
    __shared__ uint4 As[MAX_M][BLOCK_SIZE / N_PER_BLOCK];
    __shared__ float smem[MAX_M][N_PER_BLOCK * 2];
    switch (M) {
        case 9:  dense_gemv_bf16_batchm_fixed<9,  OutT>(A, B, C, N, K, out_stride, As, smem); break;
        case 10: dense_gemv_bf16_batchm_fixed<10, OutT>(A, B, C, N, K, out_stride, As, smem); break;
        case 11: dense_gemv_bf16_batchm_fixed<11, OutT>(A, B, C, N, K, out_stride, As, smem); break;
        case 12: dense_gemv_bf16_batchm_fixed<12, OutT>(A, B, C, N, K, out_stride, As, smem); break;
        case 13: dense_gemv_bf16_batchm_fixed<13, OutT>(A, B, C, N, K, out_stride, As, smem); break;
        case 14: dense_gemv_bf16_batchm_fixed<14, OutT>(A, B, C, N, K, out_stride, As, smem); break;
        case 15: dense_gemv_bf16_batchm_fixed<15, OutT>(A, B, C, N, K, out_stride, As, smem); break;
        case 16: dense_gemv_bf16_batchm_fixed<16, OutT>(A, B, C, N, K, out_stride, As, smem); break;
        // 2026-10-09: Any other M writes nothing; the launcher sends only 9..=16 here.
        default: break;
    }
}

extern "C" __global__ void dense_gemv_bf16_batchm_wide(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int out_stride
) {
    dense_gemv_bf16_batchm_wide_body<__nv_bfloat16>(A, B, C, M, N, K, out_stride);
}

extern "C" __global__ void dense_gemv_bf16_batchm_wide_fp32out(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    float* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int out_stride
) {
    dense_gemv_bf16_batchm_wide_body<float>(A, B, C, M, N, K, out_stride);
}
