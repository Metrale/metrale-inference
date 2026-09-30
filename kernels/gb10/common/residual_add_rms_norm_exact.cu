// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-28: residual_add_rms_norm_exact: bf16_residual_add (residual_add.cu) followed by
// rms_norm_residual (rms_norm.cu) in one launch, with their bytes. It fuses the layer
// boundary: layer i's FFN residual add and layer i + 1's input norm.
//
//   hidden[k]   = bf16(hidden[k] + src[k])                  (bf16_residual_add)
//   residual[k] = hidden[k]
//   output[k]   = bf16(hidden[k] * rms * (1 + weight[k])),
//   rms         = rsqrt(sum_k hidden[k]^2 / H + eps)        (rms_norm_residual)
//
// Unlike residual_add_rms_norm, which squares the FP32 sums before their BF16 rounding, the sum
// of squares here reads the rounded hidden values, as the unfused pair does.
//
// Owner: gb10 kernels.
// Invariants:
// - One block per row, blockDim.x = min(H, 1024), the launch of ops::rms_norm_residual: thread
//   t handles BF16 pairs t, t + blockDim.x, ... in both passes and thread 0 the odd tail, so
//   every partial sum and the reduction tree (rms_norm_exact.cuh) are rms_norm_residual's.
// - H is even except for the tail element, which thread 0 handles as rms_norm_residual does;
//   rows are 32-bit aligned (H even), as for rms_norm.cu.
// - hidden is read and written in place; src, residual and output must not alias it.
// - Bit identity with the unfused pair is checked by the model-arch example
//   residual_add_rms_norm_exact_microtest.

#include "rms_norm_exact.cuh"

extern "C" __global__ void residual_add_rms_norm_exact(
    __nv_bfloat16* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ src,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    __nv_bfloat16* __restrict__ residual,
    unsigned int hidden_size,
    float eps
) {
    unsigned int token = blockIdx.x;
    unsigned int tid = threadIdx.x;

    __nv_bfloat16* h = hidden + token * hidden_size;
    const __nv_bfloat16* s = src + token * hidden_size;
    __nv_bfloat16* out = output + token * hidden_size;
    __nv_bfloat16* res = residual + token * hidden_size;

    const unsigned int half_size = hidden_size / 2;
    unsigned int* h32 = (unsigned int*)h;
    const unsigned int* s32 = (const unsigned int*)s;

    // 2026-09-28: Pass 1: the residual add, rounded to BF16 and stored, then squared from the
    // rounded values in rms_norm_residual's order.
    float sum_sq = 0.0f;
    for (unsigned int i = tid; i < half_size; i += blockDim.x) {
        float h0, h1, s0, s1;
        rmsx_unpack(h32[i], h0, h1);
        rmsx_unpack(s32[i], s0, s1);
        unsigned int packed = rmsx_pack(h0 + s0, h1 + s1);
        h32[i] = packed;
        float v0, v1;
        rmsx_unpack(packed, v0, v1);
        sum_sq += v0 * v0 + v1 * v1;
    }
    if ((hidden_size & 1) && tid == 0) {
        float r = __bfloat162float(h[hidden_size - 1]);
        float a = __bfloat162float(s[hidden_size - 1]);
        __nv_bfloat16 t = __float2bfloat16(r + a);
        h[hidden_size - 1] = t;
        float val = __bfloat162float(t);
        sum_sq += val * val;
    }

    __shared__ float warp_sums[32];
    float total = rmsx_block_sum(sum_sq, warp_sums);
    float rms = rsqrtf(total / (float)hidden_size + eps);

    // 2026-09-28: Pass 2: each thread reads back the pairs it stored in pass 1.
    const unsigned int* w32 = (const unsigned int*)weight;
    unsigned int* out32 = (unsigned int*)out;
    unsigned int* res32 = (unsigned int*)res;
    for (unsigned int i = tid; i < half_size; i += blockDim.x) {
        unsigned int x_packed = h32[i];
        float xv0, xv1, wv0, wv1;
        rmsx_unpack(x_packed, xv0, xv1);
        rmsx_unpack(w32[i], wv0, wv1);
        out32[i] = rmsx_pack(xv0 * rms * (1.0f + wv0), xv1 * rms * (1.0f + wv1));
        res32[i] = x_packed;
    }
    if ((hidden_size & 1) && tid == 0) {
        float val = __bfloat162float(h[hidden_size - 1]);
        float w = __bfloat162float(weight[hidden_size - 1]);
        out[hidden_size - 1] = __float2bfloat16(val * rms * (1.0f + w));
        res[hidden_size - 1] = h[hidden_size - 1];
    }
}
