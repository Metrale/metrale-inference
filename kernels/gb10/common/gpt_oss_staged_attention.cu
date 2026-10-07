// SPDX-License-Identifier: MIT OR Apache-2.0
// 2026-10-07: Correctness residual: pinned eager staged-BF16 sink attention.
// Same token-major paged layout/argument order as paged_decode_attn_sink.
// Launch one 256-thread block per query head/sequence. Explicit bound: 4096 KV
// tokens; larger nonempty sequences return NaN. Not an optimized policy.
// Finite Q/K/V operand contract: masked nonfinite Q/K/V propagation is not
// qualified by this residual; sink NaN/+inf propagation is explicitly supported.
#include <cuda_bf16.h>
#include <math.h>

__device__ __forceinline__ float staged_bf16(float x) {
    return __bfloat162float(__float2bfloat16_rn(x));
}
extern "C" __global__ void gpt_oss_staged_attention_bf16(
    const __nv_bfloat16* q, const __nv_bfloat16* k,
    const __nv_bfloat16* v, __nv_bfloat16* out,
    const unsigned int* tables, const unsigned int* lengths,
    unsigned int max_blocks, unsigned int q_heads, unsigned int kv_heads,
    unsigned int head_dim, unsigned int block_size, float scale,
    unsigned int q_stride, unsigned int window, const __nv_bfloat16* sinks) {
    const unsigned int h = blockIdx.x, seq = blockIdx.y, lane = threadIdx.x;
    const unsigned int length = lengths[seq];
    if (!length) {
        if (lane < head_dim) out[(seq * q_heads + h) * head_dim + lane] = __float2bfloat16(0);
        return;
    }
    if (length > 4096 || head_dim != 64 || length > max_blocks * block_size) {
        if (lane < head_dim) out[(seq * q_heads + h) * head_dim + lane] = __float2bfloat16(NAN);
        return;
    }
    const unsigned int kvh = h / (q_heads / kv_heads);
    const unsigned int begin = window && length > window ? length - window : 0;
    __shared__ float scores[4097];
    __shared__ float maximum, denominator;
    for (unsigned int t = lane; t < length; t += blockDim.x) {
        if (t < begin) { scores[t] = -INFINITY; continue; }
        const unsigned int physical = tables[seq * max_blocks + t / block_size];
        const unsigned long long base = ((unsigned long long)physical * block_size + t % block_size) * kv_heads * head_dim + kvh * head_dim;
        float dot = 0;
        for (unsigned int d = 0; d < 64; ++d) {
            dot = __fadd_rn(dot, __fmul_rn(__bfloat162float(q[seq * q_stride + h * 64 + d]), __bfloat162float(k[base + d])));
        }
        scores[t] = staged_bf16(staged_bf16(dot) * scale);
    }
    if (lane == 0) scores[length] = __bfloat162float(sinks[h]);
    __syncthreads();
    if (lane == 0) {
        float m = -INFINITY;
        for (unsigned int t = begin; t <= length; ++t) {
            if (isnan(scores[t])) { m = NAN; break; }
            m = fmaxf(m, scores[t]);
        }
        maximum = m;
    }
    __syncthreads();
    // torch subtracts max in BF16 before its FP32 softmax implementation.
    for (unsigned int t = lane; t <= length; t += blockDim.x)
        scores[t] = expf(staged_bf16(scores[t] - maximum));
    __syncthreads();
    if (lane == 0) {
        float sum = 0;
        for (unsigned int t = 0; t <= length; ++t) sum = __fadd_rn(sum, scores[t]);
        denominator = sum;
    }
    __syncthreads();
    for (unsigned int t = lane; t < length; t += blockDim.x)
        scores[t] = staged_bf16(scores[t] / denominator);
    __syncthreads();
    if (lane < 64) {
        float result = 0;
        for (unsigned int t = begin; t < length; ++t) {
            const unsigned int physical = tables[seq * max_blocks + t / block_size];
            const unsigned long long offset = ((unsigned long long)physical * block_size + t % block_size) * kv_heads * 64 + kvh * 64 + lane;
            result = __fadd_rn(result, __fmul_rn(scores[t], __bfloat162float(v[offset])));
        }
        out[(seq * q_heads + h) * 64 + lane] = __float2bfloat16_rn(result);
    }
}
