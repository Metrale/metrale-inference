// SPDX-License-Identifier: MIT OR Apache-2.0
// 2026-10-07: File-backed routing parity against separately generated reference arrays.
#include <cstdio>
#include <vector>
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
namespace old {
#define moe_topk_softmax old_single
#define moe_topk_softmax_rows old_rows
#define moe_topk_softmax_f32 old_f32
#define moe_topk_softmax_batched old_batched
#include "baseline_moe_topk.cu"
#undef moe_topk_softmax
#undef moe_topk_softmax_rows
#undef moe_topk_softmax_f32
#undef moe_topk_softmax_batched
}
#include "../../../../kernels/gb10/common/moe_topk.cu"
#define CUDA(x) do { auto e=(x); if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));return 2;} } while(0)
int main(int argc,char** argv){
 if(argc!=3)return 2;
 FILE* f=fopen(argv[1],"rb"); if(!f)return 2;
 uint32_t rows; if(fread(&rows,4,1,f)!=1||!rows||rows>10000)return 2;
 std::vector<uint16_t> logits(rows*32),scores(rows*32);
 if(fread(logits.data(),2,logits.size(),f)!=logits.size())return 2; fclose(f);
 __nv_bfloat16 *x,*y; unsigned *ids,*oldids,*newids; float *oldweights,*newweights;
 CUDA(cudaMalloc(&x,logits.size()*2));CUDA(cudaMalloc(&y,scores.size()*2));
 CUDA(cudaMalloc(&ids,rows*4*4));CUDA(cudaMalloc(&oldids,rows*4*4));CUDA(cudaMalloc(&newids,rows*4*4));
 CUDA(cudaMalloc(&oldweights,rows*4*4));CUDA(cudaMalloc(&newweights,rows*4*4));
 CUDA(cudaMemcpy(x,logits.data(),logits.size()*2,cudaMemcpyHostToDevice));
 old::old_rows<<<rows,256>>>(x,oldids,oldweights,32,4,1);
 moe_topk_softmax_rows<<<rows,256>>>(x,newids,newweights,32,4,1);
 moe_topk_selected_bf16_rows<<<rows,256>>>(x,ids,y,32,4);
 CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize());
 std::vector<unsigned> selected(rows*4),oi(rows*4),ni(rows*4),ow(rows*4),nw(rows*4);
 CUDA(cudaMemcpy(selected.data(),ids,rows*16,cudaMemcpyDeviceToHost));
 CUDA(cudaMemcpy(scores.data(),y,scores.size()*2,cudaMemcpyDeviceToHost));
 CUDA(cudaMemcpy(oi.data(),oldids,rows*16,cudaMemcpyDeviceToHost));CUDA(cudaMemcpy(ni.data(),newids,rows*16,cudaMemcpyDeviceToHost));
 CUDA(cudaMemcpy(ow.data(),oldweights,rows*16,cudaMemcpyDeviceToHost));CUDA(cudaMemcpy(nw.data(),newweights,rows*16,cudaMemcpyDeviceToHost));
 if(oi!=ni||ow!=nw){fprintf(stderr,"legacy changed\n");return 1;}
 FILE* out=fopen(argv[2],"wb");if(!out)return 2;
 fwrite(&rows,4,1,out);fwrite(selected.data(),4,selected.size(),out);fwrite(scores.data(),2,scores.size(),out);fclose(out);
 printf("legacy bit identity: %u rows; selected results saved\n",rows);
 CUDA(cudaFree(x));CUDA(cudaFree(y));CUDA(cudaFree(ids));CUDA(cudaFree(oldids));CUDA(cudaFree(newids));CUDA(cudaFree(oldweights));CUDA(cudaFree(newweights));
}
