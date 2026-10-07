// SPDX-License-Identifier: MIT OR Apache-2.0
// 2026-10-07: GPT-OSS expert residuals following the staged BF16 operations in
// Transformers v4.55.0 GptOssExperts.forward (inference path). No fast math.
// Existing nllb_encoder::nllb_bias_bf16 supplies separate post-bmm BF16 bias.
#include <cuda_bf16.h>
#include <math.h>

__device__ __forceinline__ float gpt_expert_bf16(float value) {
    return __bfloat162float(__float2bfloat16_rn(value));
}

// Interleaved gate/up -> BF16 intermediate activation. Each reference tensor
// operation rounds separately: alpha multiply, sigmoid, gate multiply, up+1,
// final multiply. NaNs propagate through clamp rather than becoming bounds.
extern "C" __global__ void gpt_oss_swiglu_bf16(
    const __nv_bfloat16* __restrict__ gate_up,
    __nv_bfloat16* __restrict__ output, unsigned int count) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float g = __bfloat162float(gate_up[size_t(i) * 2]);
    float u = __bfloat162float(gate_up[size_t(i) * 2 + 1]);
    if (!isnan(g)) g = fminf(g, 7.0f);
    if (!isnan(u)) u = fminf(fmaxf(u, -7.0f), 7.0f);
    float a = gpt_expert_bf16(g * 1.702f);
    float sigmoid = gpt_expert_bf16(1.0f / (1.0f + expf(-a)));
    float glu = gpt_expert_bf16(g * sigmoid);
    float up = gpt_expert_bf16(u + 1.0f);
    output[i] = __float2bfloat16_rn(up * glu);
}

// Selected outputs [4,tokens,hidden]; router scores [tokens,32] BF16 and device
// IDs [tokens,4]. Multiply rounds to BF16 before FP32 reduction. IDs are sorted
// locally so slot order does not change summation order. Unselected experts are
// omitted only under the runtime's finite-expert-output contract; dense NaN*0
// propagation is not modeled by this sparse operator. Invalid IDs yield NaN.
extern "C" __global__ void gpt_oss_expert_reduce_bf16(
    const __nv_bfloat16* __restrict__ selected,
    const __nv_bfloat16* __restrict__ scores,
    const unsigned int* __restrict__ ids,
    __nv_bfloat16* __restrict__ output,
    unsigned int tokens, unsigned int hidden) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t count = size_t(tokens) * hidden;
    if (i >= count) return;
    unsigned token = i / hidden;
    unsigned order[4] = {0,1,2,3};
    for (unsigned a=0;a<4;++a) {
        unsigned id=ids[size_t(token)*4+a];
        if(id>=32) { output[i]=__float2bfloat16_rn(nanf("")); return; }
        for(unsigned b=0;b<a;++b)
            if(id==ids[size_t(token)*4+b]) { output[i]=__float2bfloat16_rn(nanf("")); return; }
    }
    for(unsigned a=1;a<4;++a) {
        unsigned slot=order[a], b=a;
        while(b && ids[size_t(token)*4+order[b-1]]>ids[size_t(token)*4+slot]) { order[b]=order[b-1];--b; }
        order[b]=slot;
    }
    float sum=0.0f;
    for(unsigned a=0;a<4;++a) {
        unsigned slot=order[a], expert=ids[size_t(token)*4+slot];
        float value=__bfloat162float(selected[size_t(slot)*count+i]);
        float weight=__bfloat162float(scores[size_t(token)*32+expert]);
        sum += gpt_expert_bf16(value * weight);
    }
    output[i]=__float2bfloat16_rn(sum);
}
