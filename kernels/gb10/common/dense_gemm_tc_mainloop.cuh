// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-17: The dense_gemm_tc mainloop, one overload per inclusion. dense_gemm_tc_body.cuh includes this file once
// per output type with TC_OUT_T defined (__nv_bfloat16, float), so each output type is an overload of one template
// with its own signature rather than a type template parameter: the BF16 overload keeps the exact signature, and so
// the symbol names and PTX, it had before the FP32 output existed.
//
// Owner: gb10 kernels.
// Invariants: the invariants of dense_gemm_tc_body.cuh; include it, not this file.

#ifndef TC_OUT_T
#error "dense_gemm_tc_mainloop.cuh is included by dense_gemm_tc_body.cuh with TC_OUT_T defined"
#endif

// 2026-09-25: Shared body. ACC=false stores C. ACC=true folds C += scale * bf16(A * B^T), which spares the LoRA path an
// [M, N] scratch write and a separate bf16_scaled_add pass over it.
template <bool ACC>
__device__ __forceinline__ void dense_gemm_tc_body(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    TC_OUT_T* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    float scale
) {
    const unsigned int m_block = blockIdx.y * TC_TM;
    const unsigned int n_block = blockIdx.x * TC_TN;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid_in_group = lane_id & 3;



    const unsigned int n_warp_base = warp_id * 16;


    __shared__ __nv_bfloat16 smem_A[TC_TM][TC_TK + TC_PAD];
    __shared__ __nv_bfloat16 smem_B[TC_TK][TC_TN + TC_PAD];


    float acc[2][4];
    #pragma unroll
    for (int t = 0; t < 2; t++) {
        acc[t][0] = 0.0f; acc[t][1] = 0.0f;
        acc[t][2] = 0.0f; acc[t][3] = 0.0f;
    }

    const unsigned short* sA_u16 = (const unsigned short*)smem_A;
    const unsigned short* sB_u16 = (const unsigned short*)smem_B;
    const unsigned int sA_stride = TC_TK + TC_PAD;
    const unsigned int sB_stride = TC_TN + TC_PAD;


    for (unsigned int k_base = 0; k_base < K; k_base += TC_TK) {


        {

            for (unsigned int idx = tid; idx < TC_TM * TC_TK; idx += TC_BLOCK) {
                unsigned int r = idx / TC_TK;
                unsigned int c = idx % TC_TK;
                unsigned int gr = m_block + r;
                unsigned int gc = k_base + c;
                smem_A[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : __float2bfloat16(0.0f);
            }
            // 2026-09-25: The mapping walks k fastest: B is [N, K] row-major, so neighbouring threads read neighbouring
            // elements of one B row and the global loads coalesce.









            for (unsigned int i = tid; i < TC_TK * TC_TN; i += TC_BLOCK) {
                unsigned int bn = i / TC_TK;
                unsigned int bk = i % TC_TK;
                unsigned int gn = n_block + bn;
                unsigned int gk = k_base + bk;
                smem_B[bk][bn] = (gn < N && gk < K) ? B[(unsigned long long)gn * K + gk] : __float2bfloat16(0.0f);
            }
        }
        __syncthreads();





        unsigned int ar0 = group_id;
        unsigned int ar1 = group_id + 8;
        unsigned int ac0 = tid_in_group * 2;
        unsigned int ac1 = tid_in_group * 2 + 8;
        unsigned int a0 = *(const unsigned int*)&sA_u16[ar0 * sA_stride + ac0];
        unsigned int a1 = *(const unsigned int*)&sA_u16[ar1 * sA_stride + ac0];
        unsigned int a2 = *(const unsigned int*)&sA_u16[ar0 * sA_stride + ac1];
        unsigned int a3 = *(const unsigned int*)&sA_u16[ar1 * sA_stride + ac1];


        #pragma unroll
        for (int nt = 0; nt < 2; nt++) {
            unsigned int n_col = n_warp_base + nt * 8 + group_id;
            unsigned int k0 = tid_in_group * 2;
            unsigned int k1 = tid_in_group * 2 + 8;
            unsigned int b0 = ((unsigned int)sB_u16[k0 * sB_stride + n_col] |
                              ((unsigned int)sB_u16[(k0+1) * sB_stride + n_col] << 16));
            unsigned int b1 = ((unsigned int)sB_u16[k1 * sB_stride + n_col] |
                              ((unsigned int)sB_u16[(k1+1) * sB_stride + n_col] << 16));

            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0, %1, %2, %3}, "
                "{%4, %5, %6, %7}, "
                "{%8, %9}, "
                "{%10, %11, %12, %13};"
                : "=f"(acc[nt][0]), "=f"(acc[nt][1]),
                  "=f"(acc[nt][2]), "=f"(acc[nt][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                  "r"(b0), "r"(b1),
                  "f"(acc[nt][0]), "f"(acc[nt][1]),
                  "f"(acc[nt][2]), "f"(acc[nt][3])
            );
        }

        __syncthreads();
    }






    #pragma unroll
    for (int nt = 0; nt < 2; nt++) {
        unsigned int r0 = m_block + group_id;
        unsigned int r1 = m_block + group_id + 8;
        unsigned int c0 = n_block + n_warp_base + nt * 8 + tid_in_group * 2;
        unsigned int c1 = c0 + 1;

        // 2026-09-25: The fold rounds the product to BF16 first and then stores bf16(f32(C) + scale * f32(that)): the
        // arithmetic of dense_gemm_tc followed by bf16_scaled_add (residual_add.cu), which lora_delta.rs runs when this
        // kernel is not loaded. Folding the FP32 accumulator directly would give different results.


        if (r0 < M && c0 < N) store<ACC>(C, r0 * N + c0, acc[nt][0], scale);
        if (r0 < M && c1 < N) store<ACC>(C, r0 * N + c1, acc[nt][1], scale);
        if (r1 < M && c0 < N) store<ACC>(C, r1 * N + c0, acc[nt][2], scale);
        if (r1 < M && c1 < N) store<ACC>(C, r1 * N + c1, acc[nt][3], scale);
    }
}
