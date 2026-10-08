// SPDX-License-Identifier: MIT OR Apache-2.0
// 2026-10-07: Serial/token-grid same-operand gate; no throughput qualification.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include "../../../../kernels/gb10/common/gpt_oss_mxfp4_gemv.cu"
#include "../../../../kernels/gb10/common/gpt_oss_expert_ops.cu"
__global__ void token_serial_bias(__nv_bfloat16* out,const __nv_bfloat16* bias,unsigned n) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)out[i]=__float2bfloat16_rn(__bfloat162float(out[i])+__bfloat162float(bias[i]));
}
// 2026-10-07: Mode2 deliberately breaks slot stride; it is a known-bad control.
extern "C" int token_run(uintptr_t bp,uintptr_t sp,uintptr_t xp,uintptr_t ip,
    uintptr_t yp,uintptr_t biasp,unsigned n,unsigned k,unsigned tokens,
    int per_slot,int mode,int add_bias) {
    if(!n||!k||k%32||!tokens||tokens>16)return -1;
    auto b=(const unsigned char*)bp;auto s=(const unsigned char*)sp;
    auto x=(const __nv_bfloat16*)xp;auto ids=(const unsigned*)ip;
    auto y=(__nv_bfloat16*)yp;auto bias=(const __nv_bfloat16*)biasp;
    if(mode) {
        unsigned stride=per_slot?(mode==2?k:tokens*k):0;
        gpt_oss_mxfp4_selected_tokens_bf16<<<dim3((n+3)/4,4,tokens),128>>>(b,s,x,ids,y,n,k,tokens,stride);
        if(add_bias)gpt_oss_selected_bias_tokens_bf16<<<(4*tokens*n+255)/256,256>>>(y,bias,ids,n,tokens);
    }else{
        unsigned selected[64];auto code=cudaMemcpy(selected,ids,tokens*16,cudaMemcpyDeviceToHost);
        if(code!=cudaSuccess)return int(code);
        for(unsigned t=0;t<tokens;++t)for(unsigned slot=0;slot<4;++slot){
            unsigned e=selected[t*4+slot];if(e>=32)return -2;
            auto output=y+(size_t(slot)*tokens+t)*n;
            auto input=x+(per_slot?size_t(slot)*tokens*k:0)+size_t(t)*k;
            gpt_oss_mxfp4_gemv_bf16<<<(n+3)/4,128>>>(b+size_t(e)*n*(k/2),s+size_t(e)*n*(k/32),input,output,n,k);
            if(add_bias)token_serial_bias<<<(n+255)/256,256>>>(output,bias+size_t(e)*n,n);
        }
    }
    auto code=cudaGetLastError();return code==cudaSuccess?int(cudaDeviceSynchronize()):int(code);
}
// 2026-10-07: Exercise the unchanged slot-major weighted reduction as composed.
extern "C" int token_reduce(uintptr_t selected,uintptr_t scores,uintptr_t ids,
    uintptr_t out,unsigned tokens,unsigned n) {
    gpt_oss_expert_reduce_bf16<<<(tokens*n+255)/256,256>>>((const __nv_bfloat16*)selected,
        (const __nv_bfloat16*)scores,(const unsigned*)ids,(__nv_bfloat16*)out,tokens,n);
    auto code=cudaGetLastError();return code==cudaSuccess?int(cudaDeviceSynchronize()):int(code);
}
