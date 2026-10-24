// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-19: The 64 x 64 ptrtable entries of moe_w4a16_grouped_gemm.cu (moe_w4a16_grouped_gemm_ptrtable,
// _e8m0 and _e8m0_f32out), split out of that file unchanged so it stays inside its line cap.
//
// Owner: gb10 kernels (deepseek-v4-flash).
// Invariants: included once, by moe_w4a16_grouped_gemm.cu only, after its tile macros, GROUP_SIZE,
// E2M1_LUT_MOE and mx_block_scale are defined; it declares no macro of its own.

#pragma once

// 2026-10-19: The ptrtable output store: bf16 (rounded) or f32 (the accumulator as is, for a caller that
// reduces partial sums before it rounds).
template<typename OutT> __device__ __forceinline__ OutT moe_out_cast(float v);
template<> __device__ __forceinline__ __nv_bfloat16 moe_out_cast<__nv_bfloat16>(float v) { return __float2bfloat16(v); }
template<> __device__ __forceinline__ float moe_out_cast<float>(float v) { return v; }

// 2026-09-25: moe_w4a16_grouped_gemm_ptrtable(_e8m0): 64 x 64 tiles, 128 threads, BF16 m16n8k16 MMAs on
// weights dequantized to BF16 in shared memory, K steps of 16. Expert e's weight is B_packed [N, K/2] (row n,
// the low nibble holds the even k) and B_scale [N, K / GS]. Grid (ceil(N / 64), max_m_tiles, num_experts)
// (ops::moe_w4a16_grouped_gemm_ptrtable).


template<int GS, bool E8M0, typename OutT>
__device__ __forceinline__ void moe_w4a16_grouped_gemm_ptrtable_impl(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ B_packed_ptrs,
    const unsigned long long* __restrict__ B_scale_ptrs,
    const float* __restrict__ scale2_vals,
    OutT* __restrict__ C,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    unsigned int num_experts,
    unsigned int N,
    unsigned int K
) {
    const unsigned int expert_id = blockIdx.z;
    if (expert_id >= num_experts) return;

    const int m_start = expert_offsets[expert_id];
    const int m_end = expert_offsets[expert_id + 1];
    const int M_expert = m_end - m_start;
    if (M_expert <= 0) return;

    const int cta_m_local = blockIdx.y * M_TILE;
    if (cta_m_local >= M_expert) return;

    const unsigned int cta_m = m_start + cta_m_local;
    const unsigned int cta_n = blockIdx.x * N_TILE_SM;

    const unsigned char* B_expert = (const unsigned char*)B_packed_ptrs[expert_id];
    const unsigned char* S_expert = (const unsigned char*)B_scale_ptrs[expert_id];
    const float scale2 = scale2_vals[expert_id];

    if (B_expert == 0) return;

    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __nv_bfloat16 smem_A[M_TILE][K_STEP + PAD];
    __shared__ __nv_bfloat16 smem_B[K_STEP][N_TILE_SM + PAD];

    float acc[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int a_stride = K_STEP + PAD;
    const unsigned int b_stride = N_TILE_SM + PAD;
    const unsigned int M_eff = (unsigned int)M_expert;
    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GS;

    for (unsigned int k_base = 0; k_base < K; k_base += K_STEP) {
        {
            const unsigned int ept = (M_TILE * K_STEP) / 128;
            #pragma unroll
            for (unsigned int i = 0; i < ept; i++) {
                unsigned int idx = threadIdx.x * ept + i;
                unsigned int row = idx / K_STEP;
                unsigned int col = idx % K_STEP;
                unsigned int gc = k_base + col;
                bool valid = (cta_m_local + row) < M_eff && gc < K;
                if (valid) {
                    unsigned int a_row = sorted_token_ids
                        ? (unsigned int)sorted_token_ids[cta_m + row]
                        : (cta_m + row);
                    smem_A[row][col] = A[a_row * K + gc];
                } else {
                    smem_A[row][col] = __float2bfloat16(0.0f);
                }
            }
        }

        {
            const unsigned int ept = (K_STEP * N_TILE_SM) / 128;
            unsigned int scale_group = k_base / GS;
            #pragma unroll
            for (unsigned int i = 0; i < ept; i++) {
                unsigned int idx = threadIdx.x * ept + i;
                unsigned int k = idx / N_TILE_SM;
                unsigned int n = idx % N_TILE_SM;
                unsigned int gk = k_base + k;
                unsigned int gn = cta_n + n;
                if (gk < K && gn < N) {
                    unsigned int k_pair = gk / 2;
                    unsigned char packed_byte = B_expert[(unsigned long long)gn * half_K + k_pair];
                    unsigned int nibble = (gk & 1) ? (packed_byte >> 4) : (packed_byte & 0xF);
                    unsigned char sb = S_expert[(unsigned long long)gn * num_groups + scale_group];
                    if (E8M0) {
                        // 2026-09-25: MXFP4: 2^(sb - 127), no per-tensor scale.
                        float sc = mx_block_scale<true>(sb, scale2);
                        smem_B[k][n] = __float2bfloat16(E2M1_LUT_MOE[nibble] * sc);
                    } else {
                        // 2026-09-25: NVFP4: (E2M1 x E4M3 scale) x scale2, in that order.
                        __nv_fp8_e4m3 fp8; *(unsigned char*)&fp8 = sb;
                        smem_B[k][n] = __float2bfloat16(E2M1_LUT_MOE[nibble] * (float)fp8 * scale2);
                    }
                } else {
                    smem_B[k][n] = __float2bfloat16(0.0f);
                }
            }
        }
        __syncthreads();

        const unsigned short* sA = (const unsigned short*)smem_A;
        const unsigned short* sB = (const unsigned short*)smem_B;
        unsigned int fr0 = warp_m_offset + group_id;
        unsigned int fr1 = fr0 + 8;
        unsigned int fc0 = tid * 2, fc1 = fc0 + 8;
        unsigned int a0 = ((unsigned int)sA[fr0*a_stride+fc0+1]<<16) | (unsigned int)sA[fr0*a_stride+fc0];
        unsigned int a1 = ((unsigned int)sA[fr1*a_stride+fc0+1]<<16) | (unsigned int)sA[fr1*a_stride+fc0];
        unsigned int a2 = ((unsigned int)sA[fr0*a_stride+fc1+1]<<16) | (unsigned int)sA[fr0*a_stride+fc1];
        unsigned int a3 = ((unsigned int)sA[fr1*a_stride+fc1+1]<<16) | (unsigned int)sA[fr1*a_stride+fc1];

        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            unsigned int nc = nt * 8 + group_id;
            unsigned int k0 = tid * 2, k1 = k0 + 8;
            unsigned int b0 = ((unsigned int)sB[(k0+1)*b_stride+nc]<<16) | (unsigned int)sB[k0*b_stride+nc];
            unsigned int b1 = ((unsigned int)sB[(k1+1)*b_stride+nc]<<16) | (unsigned int)sB[k1*b_stride+nc];
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                :"=f"(acc[nt][0]),"=f"(acc[nt][1]),"=f"(acc[nt][2]),"=f"(acc[nt][3])
                :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
                 "f"(acc[nt][0]),"f"(acc[nt][1]),"f"(acc[nt][2]),"f"(acc[nt][3]));
        }
        __syncthreads();
    }

    #pragma unroll
    for (int nt = 0; nt < 8; nt++) {
        unsigned int c0 = cta_n + nt*8 + tid*2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        bool r0v = (int)(warp_m_offset + group_id + cta_m_local) < M_expert;
        bool r1v = (int)(warp_m_offset + group_id + 8 + cta_m_local) < M_expert;
        if (r0v && c0 < N) C[r0*N+c0] = moe_out_cast<OutT>(acc[nt][0]);
        if (r0v && c1 < N) C[r0*N+c1] = moe_out_cast<OutT>(acc[nt][1]);
        if (r1v && c0 < N) C[r1*N+c0] = moe_out_cast<OutT>(acc[nt][2]);
        if (r1v && c1 < N) C[r1*N+c1] = moe_out_cast<OutT>(acc[nt][3]);
    }
}


extern "C" __global__ void moe_w4a16_grouped_gemm_ptrtable(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ B_packed_ptrs,
    const unsigned long long* __restrict__ B_scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    unsigned int num_experts, unsigned int N, unsigned int K
) {
    moe_w4a16_grouped_gemm_ptrtable_impl<GROUP_SIZE, false, __nv_bfloat16>(
        A, B_packed_ptrs, B_scale_ptrs, scale2_vals, C,
        expert_offsets, sorted_token_ids, num_experts, N, K);
}

// 2026-09-25: MXFP4 weights: an E8M0 scale per 32 values and no per-tensor scale.
extern "C" __global__ void moe_w4a16_grouped_gemm_ptrtable_e8m0(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ B_packed_ptrs,
    const unsigned long long* __restrict__ B_scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    unsigned int num_experts, unsigned int N, unsigned int K
) {
    moe_w4a16_grouped_gemm_ptrtable_impl<32, true, __nv_bfloat16>(
        A, B_packed_ptrs, B_scale_ptrs, scale2_vals, C,
        expert_offsets, sorted_token_ids, num_experts, N, K);
}

// 2026-10-19: The _e8m0 entry with f32 outputs: C is [rows, N] f32, the unrounded accumulators (a tensor-parallel
// rank's partial down-projection, summed across ranks before the one bf16 rounding).
extern "C" __global__ void moe_w4a16_grouped_gemm_ptrtable_e8m0_f32out(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ B_packed_ptrs,
    const unsigned long long* __restrict__ B_scale_ptrs,
    const float* __restrict__ scale2_vals,
    float* __restrict__ C,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    unsigned int num_experts, unsigned int N, unsigned int K
) {
    moe_w4a16_grouped_gemm_ptrtable_impl<32, true, float>(
        A, B_packed_ptrs, B_scale_ptrs, scale2_vals, C,
        expert_offsets, sorted_token_ids, num_experts, N, K);
}
