// SPDX-License-Identifier: MIT OR Apache-2.0
// 2026-10-07: Isolated diagnostic only; never registered or used by production.
// Preserve incumbent FP32 FMA/warp order. Retry an exact BF16 midpoint in FP64,
// then convert directly from FP64 to BF16 to avoid a second FP32 rounding.
// FP64 is a higher-precision diagnostic, not an exact arbitrary-range dot oracle.
#include <cuda_runtime.h>
#include "../../../../kernels/gb10/common/gpt_oss_mxfp4_gemv.cu"

__global__ void midpoint_rows(const unsigned char* blocks, const unsigned char* scales,
    const __nv_bfloat16* input, __nv_bfloat16* output, __nv_bfloat16* double_round_bad,
    float* accum_out, double* retry_out, unsigned* retry_flags, unsigned rows, unsigned cols) {
    const unsigned row=blockIdx.x*4+threadIdx.x/32, lane=threadIdx.x&31;
    if(row>=rows)return;
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
    if(lane==0){
        output[row]=retry?__double2bfloat16(precise):__float2bfloat16_rn(total);
        double_round_bad[row]=__float2bfloat16_rn(float(precise));
        accum_out[row]=total;retry_out[row]=precise;retry_flags[row]=retry;
    }
}

extern "C" int run_midpoint(const unsigned char* blocks,const unsigned char* scales,
    const __nv_bfloat16* input,__nv_bfloat16* legacy,__nv_bfloat16* repaired,
    __nv_bfloat16* bad,float* accum,double* precise,unsigned* flags,unsigned rows,unsigned cols){
    if(!rows||!cols||cols%32)return int(cudaErrorInvalidValue);
    gpt_oss_mxfp4_gemv_bf16<<<(rows+3)/4,128>>>(blocks,scales,input,legacy,rows,cols);
    midpoint_rows<<<(rows+3)/4,128>>>(blocks,scales,input,repaired,bad,accum,precise,flags,rows,cols);
    return int(cudaGetLastError());
}
