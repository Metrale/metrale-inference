// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-28: `dense_gemm_ba_gates_prefill_tiled`: a bit-identical twin of `dense_gemm_ba_gates_prefill`
// (ssm_preprocess.cu), the GDN beta / alpha gate projection with its gate epilogue, for BA_TILE tokens per CTA.
//
// Per output (token, n) the arithmetic is the original's: 64 lane partials, lane l summing a * b over the 16-byte
// vectors kv = l, l + 64, ... in ascending order (lo then hi of each 32-bit word, --fmad=false), a shfl_down tree over
// each 32-lane half, the two halves added, then the same sigmoid / softplus-exp epilogue. The original runs one CTA
// per (token, 4 outputs) and so reads the 4 weight rows from L2 once per token; here each thread keeps its weight
// vectors in registers and walks BA_TILE tokens, so they are read once per BA_TILE tokens.
//
// Owner: gb10 kernels.
// Invariants:
// - Same arguments as the original. Grid (ceil(N / 4), ceil(M / BA_TILE), 1), block 256.
// - K % 8 == 0 and K / 8 <= 64 * BA_MAXV (the caller falls back to the original otherwise); A rows and B rows are
//   16-byte aligned (as the original assumes).

#include <cuda_bf16.h>

#define BA_TILE 32
#define BA_MAXV 12

extern "C" __global__ void __launch_bounds__(256) dense_gemm_ba_gates_prefill_tiled(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    float* __restrict__ gate_out,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int K_stride,
    unsigned int gate_stride,
    unsigned int nv,
    unsigned int vheads_per_group
) {
    __shared__ float part[BA_TILE][4][2];
    const unsigned int local_out = threadIdx.x / 64;
    const unsigned int lane = threadIdx.x % 64;
    const unsigned int n = blockIdx.x * 4 + local_out;
    const unsigned int t0 = blockIdx.y * BA_TILE;
    const bool live = n < N;
    const unsigned int K_VEC = K / 8;

    uint4 bv[BA_MAXV];
    unsigned int nvec = 0;
    #pragma unroll
    for (int v = 0; v < BA_MAXV; v++) {
        const unsigned int kv = lane + 64u * v;
        if (live && kv < K_VEC) {
            bv[v] = ((const uint4*)(B + (unsigned long long)n * K))[kv];
            nvec = v + 1;
        } else {
            bv[v] = make_uint4(0, 0, 0, 0);
        }
    }

    for (unsigned int tt = 0; tt < BA_TILE; tt++) {
        const unsigned int token = t0 + tt;
        if (token >= M) break;
        float acc = 0.0f;
        const uint4* A_vec = (const uint4*)(A + (unsigned long long)token * K_stride);
        #pragma unroll
        for (int v = 0; v < BA_MAXV; v++) {
            if ((unsigned int)v < nvec) {
                const uint4 a_data = A_vec[lane + 64u * v];
                const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};
                const unsigned int b_raw[4] = {bv[v].x, bv[v].y, bv[v].z, bv[v].w};
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    __nv_bfloat16 a_lo, a_hi, b_lo, b_hi;
                    *(unsigned short*)&a_lo = (unsigned short)(a_raw[i] & 0xFFFF);
                    *(unsigned short*)&a_hi = (unsigned short)(a_raw[i] >> 16);
                    *(unsigned short*)&b_lo = (unsigned short)(b_raw[i] & 0xFFFF);
                    *(unsigned short*)&b_hi = (unsigned short)(b_raw[i] >> 16);
                    acc += __bfloat162float(a_lo) * __bfloat162float(b_lo);
                    acc += __bfloat162float(a_hi) * __bfloat162float(b_hi);
                }
            }
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
        if ((threadIdx.x % 32) == 0) part[tt][local_out][lane / 32] = acc;
    }
    __syncthreads();

    if (threadIdx.x < BA_TILE * 4) {
        const unsigned int tt = threadIdx.x / 4, o = threadIdx.x % 4;
        const unsigned int token = t0 + tt;
        const unsigned int nn = blockIdx.x * 4 + o;
        if (token < M && nn < N) {
            const float result = part[tt][o][0] + part[tt][o][1];
            const unsigned int group_dim_ba = 2 * vheads_per_group;
            const unsigned int within_group = nn % group_dim_ba;
            const unsigned int group = nn / group_dim_ba;
            float* gate_tok = gate_out + (unsigned long long)token * gate_stride;
            if (within_group < vheads_per_group) {
                const unsigned int vh = group * vheads_per_group + within_group;
                gate_tok[nv + vh] = 1.0f / (1.0f + __expf(-result));
            } else {
                const unsigned int vh = group * vheads_per_group + (within_group - vheads_per_group);
                const float a_log_val = A_log[vh];
                const float dt_b = dt_bias[vh];
                const float A_val = __expf(fminf(a_log_val, 20.0f));
                const float dt = __logf(1.0f + __expf(fminf(result + dt_b, 20.0f)));
                gate_tok[vh] = __expf(-A_val * dt);
            }
        }
    }
}
