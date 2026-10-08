// SPDX-License-Identifier: MIT OR Apache-2.0
// 2026-10-07: Constructed token-grid/reuse layout gate with independently validated host plan.
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <algorithm>
#include "../../../../kernels/gb10/common/gpt_oss_mxfp4_gemv.cu"
#include "../../../../kernels/gb10/common/gpt_oss_expert_ops.cu"
static bool valid_plan(const std::vector<unsigned>& p,const std::vector<unsigned>& ids,unsigned m){
 bool seen[64]={};
 for(unsigned e=0;e<32;++e){
  unsigned n=p[e*(m+1)];if(n>m)return false;
  for(unsigned j=0;j<n;++j){unsigned at=p[e*(m+1)+1+j];
   if(at>=4*m||seen[at]||ids[at]!=e)return false;seen[at]=true;}
 }
 for(unsigned i=0;i<4*m;++i)if(!seen[i])return false;
 return true;
}
extern "C" int token_run(uintptr_t bp,uintptr_t sp,uintptr_t xp,uintptr_t ip,
 uintptr_t yp,uintptr_t biasp,unsigned n,unsigned k,unsigned m,int per,int mode,int add){
 if(!n||!k||k%32||!m||m>16)return -1;
 auto b=(const unsigned char*)bp;auto s=(const unsigned char*)sp;auto x=(const __nv_bfloat16*)xp;
 auto ids=(const unsigned*)ip;auto y=(__nv_bfloat16*)yp;auto bias=(const __nv_bfloat16*)biasp;
 std::vector<unsigned> host(4*m),plan(32*(m+1));
 auto error=cudaMemcpy(host.data(),ids,host.size()*4,cudaMemcpyDeviceToHost);if(error!=cudaSuccess)return int(error);
 for(unsigned t=0;t<m;++t)for(unsigned slot=0;slot<4;++slot){unsigned e=host[t*4+slot];
  if(e>=32)return -2;for(unsigned j=0;j<slot;++j)if(host[t*4+j]==e)return -2;
  unsigned at=e*(m+1),count=plan[at]++;plan[at+1+count]=t*4+slot;
 }
 if(mode==3)for(unsigned e=0;e<32;++e)std::reverse(plan.begin()+e*(m+1)+1,plan.begin()+e*(m+1)+1+plan[e*(m+1)]);
 if(mode==4)plan[host[0]*(m+1)]=0;
 if(mode==5){unsigned at=host[0]*(m+1);plan[at]=2;if(m>1)plan[at+2]=plan[at+1];}
 if(!valid_plan(plan,host,m))return -3;
 if(!mode){
  gpt_oss_mxfp4_selected_tokens_bf16<<<dim3((n+3)/4,4,m),128>>>(b,s,x,ids,y,n,k,m,per?m*k:0);
 }else{
  unsigned* dp;error=cudaMalloc(&dp,plan.size()*4);if(error!=cudaSuccess)return int(error);
  error=cudaMemcpy(dp,plan.data(),plan.size()*4,cudaMemcpyHostToDevice);if(error!=cudaSuccess){cudaFree(dp);return int(error);}
  gpt_oss_mxfp4_reuse_tokens_bf16<<<dim3((n+3)/4,32,(m+3)/4),128>>>(b,s,x,ids,dp,y,n,k,m,per?(mode==2?k:m*k):0);
  error=cudaGetLastError();if(error==cudaSuccess)error=cudaDeviceSynchronize();cudaFree(dp);if(error!=cudaSuccess)return int(error);
 }
 if(add)gpt_oss_selected_bias_tokens_bf16<<<(4*m*n+255)/256,256>>>(y,bias,ids,n,m);
 error=cudaGetLastError();return error==cudaSuccess?int(cudaDeviceSynchronize()):int(error);
}
extern "C" int token_reduce(uintptr_t selected,uintptr_t scores,uintptr_t ids,uintptr_t out,unsigned m,unsigned n){
 gpt_oss_expert_reduce_bf16<<<(m*n+255)/256,256>>>((const __nv_bfloat16*)selected,(const __nv_bfloat16*)scores,(const unsigned*)ids,(__nv_bfloat16*)out,m,n);
 auto e=cudaGetLastError();return e==cudaSuccess?int(cudaDeviceSynchronize()):int(e);
}
