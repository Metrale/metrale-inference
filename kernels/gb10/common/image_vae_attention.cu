// SPDX-License-Identifier: MIT OR Apache-2.0
// 2026-10-06: Named wide-head residual for original-FP32 single-frame VAE attention.
// Existing NLLB attention requires D==blockDim; D1152 cannot use that path.
// One query per CTA, channel-first QKV [3,1152,pixels], noncausal, no mask.
// FP32 online softmax, separate multiply/add; no fast-math or precision substitution.
#include <math.h>
#include <stddef.h>

extern "C" __global__ void image_vae_attention_f32(
    const float* __restrict__ qkv, float* __restrict__ output, unsigned int pixels) {
    constexpr unsigned D = 1152, T = 128, ITEMS = D / T;
    const unsigned query = blockIdx.x, tid = threadIdx.x;
    if (query >= pixels || blockDim.x != T || blockDim.y != 1 || blockDim.z != 1) return;
    __shared__ float warp_sums[4];
    __shared__ float factors[3];
    float q[ITEMS], accumulated[ITEMS];
    for (unsigned i = 0; i < ITEMS; ++i) {
        q[i] = qkv[size_t(tid + i*T)*pixels + query];
        accumulated[i] = 0.0f;
    }
    float maximum = -INFINITY, denominator = 0.0f;
    const float scale = 1.0f / sqrtf(float(D));
    for (unsigned key = 0; key < pixels; ++key) {
        float dot = 0.0f;
        for (unsigned i = 0; i < ITEMS; ++i)
            dot = __fadd_rn(dot, __fmul_rn(q[i], qkv[size_t(D + tid + i*T)*pixels + key]));
        for (unsigned offset = 16; offset; offset >>= 1)
            dot = __fadd_rn(dot, __shfl_down_sync(0xffffffff, dot, offset));
        if ((tid & 31) == 0) warp_sums[tid >> 5] = dot;
        __syncthreads();
        if (tid == 0) {
            float score = __fmul_rn(__fadd_rn(__fadd_rn(warp_sums[0], warp_sums[1]),
                                          __fadd_rn(warp_sums[2], warp_sums[3])), scale);
            float next_maximum = fmaxf(maximum, score);
            float previous = expf(maximum - next_maximum);
            float current = expf(score - next_maximum);
            denominator = __fadd_rn(__fmul_rn(denominator, previous), current);
            maximum = next_maximum;
            factors[0] = previous; factors[1] = current; factors[2] = denominator;
        }
        __syncthreads();
        for (unsigned i = 0; i < ITEMS; ++i) {
            float v = qkv[size_t(2*D + tid + i*T)*pixels + key];
            accumulated[i] = __fadd_rn(__fmul_rn(accumulated[i], factors[0]), __fmul_rn(v, factors[1]));
        }
        __syncthreads();
    }
    for (unsigned i = 0; i < ITEMS; ++i)
        output[size_t(tid + i*T)*pixels + query] = accumulated[i] / factors[2];
}
