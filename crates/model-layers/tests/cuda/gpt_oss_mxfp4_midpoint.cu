// SPDX-License-Identifier: MIT OR Apache-2.0
// 2026-10-07: Isolated diagnostic only; never registered or used by production.
// Preserve incumbent FP32 FMA/warp order. Retry an exact BF16 midpoint in FP64,
// then convert directly from FP64 to BF16 to avoid a second FP32 rounding.
// FP64 is a higher-precision diagnostic, not an exact arbitrary-range dot oracle.
#include <cuda_runtime.h>
#ifdef GPT_MIDPOINT_OVERRIDE
#define gpt_oss_mxfp4_gemv_bf16 gpt_oss_mxfp4_incumbent_unused
#endif
#include "../../../../kernels/gb10/common/gpt_oss_mxfp4_gemv.cu"
#ifdef GPT_MIDPOINT_OVERRIDE
#undef gpt_oss_mxfp4_gemv_bf16
#define GPT_MIDPOINT_ENTRY gpt_oss_mxfp4_gemv_bf16
#else
#define GPT_MIDPOINT_ENTRY gpt_oss_mxfp4_midpoint_bf16
#endif
struct MidpointResult {float total;double precise;bool retry;};
__device__ __forceinline__ MidpointResult midpoint_dot(const unsigned char* blocks,
    const unsigned char* scales,const __nv_bfloat16* input,unsigned row,unsigned cols){
    const unsigned lane=threadIdx.x&31;
    float accum=0.0f;
    for(unsigned col=lane;col<cols;col+=32){
        const unsigned char byte=blocks[size_t(row)*(cols/2)+col/2];
        const unsigned char code=(col&1)?byte>>4:byte&15;
        accum=fmaf(gpt_oss_unpack(code,scales[size_t(row)*(cols/32)+col/32]),
            __bfloat162float(input[col]),accum);
    }
    for(unsigned shift=16;shift;shift>>=1)accum+=__shfl_down_sync(0xffffffffU,accum,shift);
    const float total=__shfl_sync(0xffffffffU,accum,0);
    const bool retry=isfinite(total)&&((__float_as_uint(total)&0xffffU)==0x8000U);
    double precise=double(total);
    if(retry){
        precise=0.0;
        for(unsigned col=lane;col<cols;col+=32){
            const unsigned char byte=blocks[size_t(row)*(cols/2)+col/2];
            const unsigned char code=(col&1)?byte>>4:byte&15;
            precise=fma(double(gpt_oss_unpack(code,scales[size_t(row)*(cols/32)+col/32])),
                double(__bfloat162float(input[col])),precise);
        }
        for(unsigned shift=16;shift;shift>>=1)precise+=__shfl_down_sync(0xffffffffU,precise,shift);
    }
    return {total,precise,retry};
}
// 2026-10-07: Fresh module owns zero-initialized gate/down/other retry counters.
__device__ unsigned long long midpoint_retry_counts[3]={0,0,0};
extern "C" __global__ void gpt_oss_midpoint_get_counts(unsigned long long* output){
    if(threadIdx.x==0&&blockIdx.x==0)for(unsigned i=0;i<3;++i)output[i]=midpoint_retry_counts[i];
}
extern "C" __global__ void GPT_MIDPOINT_ENTRY(const unsigned char* blocks,
    const unsigned char* scales,const __nv_bfloat16* input,__nv_bfloat16* output,
    unsigned rows,unsigned cols){
    const unsigned row=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x&31;
    if(row>=rows)return;
    const auto r=midpoint_dot(blocks,scales,input,row,cols);
    if(lane==0){
        output[row]=r.retry?__double2bfloat16(r.precise):__float2bfloat16_rn(r.total);
        if(r.retry)atomicAdd(midpoint_retry_counts+(rows==5760?0:(rows==2880?1:2)),1ULL);
    }
}
#ifndef GPT_MIDPOINT_OVERRIDE
__global__ void midpoint_rows(const unsigned char* blocks,const unsigned char* scales,
    const __nv_bfloat16* input,__nv_bfloat16* bad,float* accum,double* precise,
    unsigned* flags,unsigned rows,unsigned cols){
    const unsigned row=blockIdx.x*4+threadIdx.x/32;
    if(row>=rows)return;
    const auto r=midpoint_dot(blocks,scales,input,row,cols);
    if(!(threadIdx.x&31)){bad[row]=__float2bfloat16_rn(float(r.precise));accum[row]=r.total;precise[row]=r.precise;flags[row]=r.retry;}
}
extern "C" int run_midpoint(const unsigned char* blocks,const unsigned char* scales,
    const __nv_bfloat16* input,__nv_bfloat16* legacy,__nv_bfloat16* repaired,
    __nv_bfloat16* bad,float* accum,double* precise,unsigned* flags,unsigned rows,unsigned cols){
    if(!rows||!cols||cols%32)return int(cudaErrorInvalidValue);
    gpt_oss_mxfp4_gemv_bf16<<<(rows+3)/4,128>>>(blocks,scales,input,legacy,rows,cols);
    midpoint_rows<<<(rows+3)/4,128>>>(blocks,scales,input,bad,accum,precise,flags,rows,cols);
    GPT_MIDPOINT_ENTRY<<<(rows+3)/4,128>>>(blocks,scales,input,repaired,rows,cols);
    return int(cudaGetLastError());
}
extern "C" int read_midpoint_counts(unsigned long long* output){
    gpt_oss_midpoint_get_counts<<<1,1>>>(output);return int(cudaGetLastError());
}
#endif
