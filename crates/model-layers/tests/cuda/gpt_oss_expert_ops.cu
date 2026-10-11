// SPDX-License-Identifier: MIT OR Apache-2.0
// 2026-10-07: Execute production expert operators on a separately generated Torch
// corpus. Compare with gpt_oss_expert_ops_reference.py; no embedded expected values.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <string>
#include <stdexcept>
#include "../../../../kernels/gb10/common/gpt_oss_expert_ops.cu"
#include "../../../../kernels/gb10/common/nllb_encoder.cu"
#define CUDA(x) do {auto e=(x);if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}while(0)
static std::vector<uint8_t> read(const std::string& name) {
    FILE*f=fopen(name.c_str(),"rb");if(!f)throw std::runtime_error(name);
    fseek(f,0,SEEK_END);long size=ftell(f);rewind(f);if(size<0||size>16000000)throw std::runtime_error("size");
    std::vector<uint8_t> bytes(size);if(fread(bytes.data(),1,size,f)!=size)throw std::runtime_error("read");fclose(f);return bytes;
}
static void* upload(const void* data,size_t bytes) {void*p;CUDA(cudaMalloc(&p,bytes));CUDA(cudaMemcpy(p,data,bytes,cudaMemcpyHostToDevice));return p;}
static void save(void* data,size_t bytes,const std::string& path) {
    CUDA(cudaGetLastError());CUDA(cudaDeviceSynchronize());std::vector<uint8_t> result(bytes);CUDA(cudaMemcpy(result.data(),data,bytes,cudaMemcpyDeviceToHost));
    FILE*f=fopen(path.c_str(),"wb");if(!f||fwrite(result.data(),1,bytes,f)!=bytes)throw std::runtime_error("write");fclose(f);
}
static uint32_t word(const std::vector<uint8_t>& b,size_t offset) {if(offset+4>b.size())throw std::runtime_error("header");uint32_t n;memcpy(&n,b.data()+offset,4);return n;}
int main(int argc,char**argv) {try {
    if(argc!=2)return 2;std::string root=argv[1];
    {
        auto b=read(root+"/bias-input.bin");unsigned rows=word(b,0),cols=word(b,4);size_t count=size_t(rows)*cols;
        if(!rows||!cols||count>1000000||b.size()!=8+count*2+cols*2)throw std::runtime_error("bias shape");
        auto*x=(__nv_bfloat16*)upload(b.data()+8,count*2);auto*bias=(__nv_bfloat16*)upload(b.data()+8+count*2,cols*2);
        nllb_bias_bf16<<<(count+255)/256,256>>>(x,bias,rows,cols);save(x,count*2,root+"/bias-observed.bin");CUDA(cudaFree(x));CUDA(cudaFree(bias));
    }
    {
        auto b=read(root+"/activation-input.bin");unsigned count=word(b,0);
        if(!count||count>1000000||b.size()!=4+size_t(count)*4)throw std::runtime_error("activation shape");
        auto*x=(__nv_bfloat16*)upload(b.data()+4,size_t(count)*4);__nv_bfloat16*y;CUDA(cudaMalloc(&y,size_t(count)*2));
        gpt_oss_swiglu_bf16<<<(count+255)/256,256>>>(x,y,count);save(y,size_t(count)*2,root+"/activation-observed.bin");CUDA(cudaFree(x));CUDA(cudaFree(y));
    }
    {
        auto b=read(root+"/reduce-input.bin");unsigned tokens=word(b,0),hidden=word(b,4);size_t count=size_t(tokens)*hidden;
        if(!tokens||!hidden||count>1000000||b.size()!=8+count*8+tokens*80)throw std::runtime_error("reduce shape");
        auto*x=(__nv_bfloat16*)upload(b.data()+8,count*8);auto*s=(__nv_bfloat16*)upload(b.data()+8+count*8,tokens*64);auto*ids=(unsigned*)upload(b.data()+8+count*8+tokens*64,tokens*16);
        __nv_bfloat16*y;CUDA(cudaMalloc(&y,count*2));gpt_oss_expert_reduce_bf16<<<(count+255)/256,256>>>(x,s,ids,y,tokens,hidden);save(y,count*2,root+"/reduce-observed.bin");
        CUDA(cudaFree(x));CUDA(cudaFree(s));CUDA(cudaFree(ids));CUDA(cudaFree(y));
    }
    return 0;
}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}
