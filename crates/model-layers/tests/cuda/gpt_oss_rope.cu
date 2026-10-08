// SPDX-License-Identifier: MIT OR Apache-2.0
// 2026-10-07: Standalone bridge exercises production kernels on caller-owned CUDA buffers.
#include "../../../../kernels/gb10/common/gpt_oss_rope.cu"
extern "C" int table(float* out,float base,float factor,float low,float span) {
 gpt_oss_yarn_frequencies<<<1,32>>>(out,base,factor,low,span);return cudaDeviceSynchronize();
}
extern "C" int rotate(void* q,void* k,const unsigned* positions,const float* freq,unsigned rows,unsigned qh,unsigned kh,float scale) {
 gpt_oss_rope_bf16<<<(rows*(qh+kh)*32+255)/256,256>>>((__nv_bfloat16*)q,(__nv_bfloat16*)k,positions,freq,rows,qh,kh,scale);return cudaDeviceSynchronize();
}
