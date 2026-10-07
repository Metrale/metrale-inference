// SPDX-License-Identifier: MIT OR Apache-2.0
// 2026-10-07: Frozen BF16 batch and scalar controls; constructed operands only.
#include <cuda_runtime.h>
#define dense_gemv_bf16_batchm dense_gemv_bf16_batchm_frozen
#include "dense_batchm_frozen.cuh"
#undef dense_gemv_bf16_batchm
#include "../../../../kernels/gb10/common/dense_gemv_bf16_batchm.cu"
#include "../../../../kernels/gb10/common/dense_gemv_bf16.cu"
extern "C" int compare_batch(const __nv_bfloat16* a,const __nv_bfloat16* b,
    float* batch,float* scalar,__nv_bfloat16* old_bf16,__nv_bfloat16* new_bf16,
    unsigned m,unsigned n,unsigned k,unsigned stride){
    if(!m||m>128||!n||n%4||!k||k%8||stride<n)return int(cudaErrorInvalidValue);
    dense_gemv_bf16_batchm_fp32out<<<dim3(n/4,(m+15)/16),256>>>(a,b,batch,m,n,k,stride);
    dense_gemv_bf16_batchm_frozen<<<dim3(n/4,(m+15)/16),256>>>(a,b,old_bf16,m,n,k,stride);
    dense_gemv_bf16_batchm<<<dim3(n/4,(m+15)/16),256>>>(a,b,new_bf16,m,n,k,stride);
    for(unsigned t=0;t<m;++t)dense_gemv_bf16_fp32out<<<n/4,256>>>(a+size_t(t)*k,b,scalar+size_t(t)*stride,n,k);
    return int(cudaGetLastError());
}
// 2026-10-07: Existing wider-row split admission remains unchanged.
extern "C" int compare_bf16_split(const __nv_bfloat16* a,const __nv_bfloat16* b,
    __nv_bfloat16* old_bf16,__nv_bfloat16* new_bf16,unsigned m,unsigned n,unsigned k,unsigned stride,unsigned y){
    if(!y||!m||(m+y-1)/y>16||n%4||k%8)return int(cudaErrorInvalidValue);
    dense_gemv_bf16_batchm_frozen<<<dim3(n/4,y),256>>>(a,b,old_bf16,m,n,k,stride);
    dense_gemv_bf16_batchm<<<dim3(n/4,y),256>>>(a,b,new_bf16,m,n,k,stride);
    return int(cudaGetLastError());
}

// 2026-10-07: Known-bad omitted Y split silently truncates after16 rows.
extern "C" int truncated_batch(const __nv_bfloat16* a,const __nv_bfloat16* b,
    float* output,unsigned m,unsigned n,unsigned k,unsigned stride){
    if(m<=16||m>128||!n||n%4||!k||k%8||stride<n)return int(cudaErrorInvalidValue);
    dense_gemv_bf16_batchm_fp32out<<<n/4,256>>>(a,b,output,m,n,k,stride);
    return int(cudaGetLastError());
}
