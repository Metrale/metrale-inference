// SPDX-License-Identifier: MIT OR Apache-2.0
// 2026-10-07: Named residual yarn_half_split_bf16_staged. Existing YaRN kernels
// keep products in FP32; GPT v4.55 materializes BF16 cos/sin and each product.
// Compile with --fmad=false, as common KERNEL.toml requires. HD64 only.
#include <cuda_bf16.h>
#include <math.h>
extern "C" __global__ void gpt_oss_yarn_frequencies(float* out,float base,float factor,float low,float span) {
    unsigned i=threadIdx.x;
    if(i>=32) return;
    float pos_freq=powf(base,float(2*i)/64.0f);
    float extrapolation=1.0f/pos_freq;
    float interpolation=1.0f/(factor*pos_freq);
    float ramp=fminf(fmaxf((float(i)-low)/span,0.0f),1.0f);
    float extra=1.0f-ramp;
    out[i]=interpolation*(1.0f-extra)+extrapolation*extra;
}
// 2026-10-07: Rounding here deliberately prevents algebraic contraction/reassociation.
__device__ __forceinline__ float gpt_rope_round(float value){return __bfloat162float(__float2bfloat16_rn(value));}
extern "C" __global__ void gpt_oss_rope_bf16(
    __nv_bfloat16* q,__nv_bfloat16* k,const unsigned* positions,const float* freq,
    unsigned rows,unsigned q_heads,unsigned kv_heads,float attention_factor
) {
    unsigned long long pair=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
    unsigned heads=q_heads+kv_heads;
    if(pair>=(unsigned long long)rows*heads*32) return;
    unsigned d=pair%32, head=(pair/32)%heads,row=pair/(32*heads);
    bool is_q=head<q_heads;unsigned own_head=is_q?head:head-q_heads;
    __nv_bfloat16* x=(is_q?q:k)+((unsigned long long)row*(is_q?q_heads:kv_heads)+own_head)*64;
    float angle=float(positions[row])*freq[d];
    float c=gpt_rope_round(cosf(angle)*attention_factor),s=gpt_rope_round(sinf(angle)*attention_factor);
    float a=__bfloat162float(x[d]),b=__bfloat162float(x[d+32]);
    x[d]=__float2bfloat16_rn(gpt_rope_round(a*c)-gpt_rope_round(b*s));
    x[d+32]=__float2bfloat16_rn(gpt_rope_round(b*c)+gpt_rope_round(a*s));
}
