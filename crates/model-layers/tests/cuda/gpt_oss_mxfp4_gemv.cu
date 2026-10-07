// SPDX-License-Identifier: MIT OR Apache-2.0
// 2026-10-07: Standalone primitive test. CPU oracle uses arithmetic E2M1 decode,
// double accumulation and integer BF16 RNE, independent of device lookup/reduction.
// nvcc -arch=sm_121 -o /tmp/gpt-mxfp4-test crates/model-layers/tests/cuda/gpt_oss_mxfp4_gemv.cu
// Optional binary slice: LE u32 rows,cols; packed bytes; scale bytes. Never commit weights.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
#include "../../../../kernels/gb10/common/gpt_oss_mxfp4_gemv.cu"
#define CUDA(x) do { auto e=(x); if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));return false;} } while(0)
static uint16_t bf16(float x) { uint32_t b; memcpy(&b,&x,4); if((b&0x7fffffffU)>0x7f800000U)return (b>>16)|64; return (b+0x7fffU+((b>>16)&1))>>16; }
static float widen(uint16_t b) { uint32_t u=uint32_t(b)<<16; float f;memcpy(&f,&u,4);return f; }
static float decode(unsigned code, unsigned scale) {
    unsigned e=(code>>1)&3, m=code&1;
    double v=e?std::ldexp(1.0+double(m)/2,int(e)-1):double(m)/2;
    if(code&8)v=-v;
    return widen(bf16(float(std::ldexp(v,int(scale)-127))));
}
__global__ void probe(float* out) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<4096)out[i]=gpt_oss_unpack(i%16,i/16);
}
static bool decode_test() {
    float* out; CUDA(cudaMalloc(&out,4096*4));
    probe<<<16,256>>>(out); CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize());
    std::vector<float> got(4096); CUDA(cudaMemcpy(got.data(),out,4096*4,cudaMemcpyDeviceToHost));
    for(unsigned i=0;i<4096;++i){float want=decode(i%16,i/16); if(!(got[i]==want || (std::isnan(got[i])&&std::isnan(want)))){fprintf(stderr,"decode mismatch %u\n",i);return false;}}
    if(got[2]==0 || !std::isinf(got[255*16+2]))return false;
    CUDA(cudaFree(out)); printf("all 4096 code/scale pairs pass pinned-unpack oracle\n"); return true;
}
static uint16_t reference(const std::vector<uint8_t>& w,const std::vector<uint8_t>& s,const std::vector<uint16_t>& x,unsigned row,unsigned n,unsigned k,unsigned bad) {
    double sum=0;
    for(unsigned col=0;col<k;++col){
        size_t at=bad==3?size_t(col/2)*n+row:size_t(row)*(k/2)+col/2;
        unsigned code=(w[at]>>(((col&1)^(bad==1))*4))&15;
        unsigned group=bad==2?(col/16)%(k/32):col/32;
        sum+=double(decode(code,s[size_t(row)*(k/32)+group]))*widen(x[col]);
    }
    return bf16(float(sum));
}
static bool run(unsigned n,unsigned k,const std::vector<uint8_t>& w,const std::vector<uint8_t>& s,const char* label,bool exact,bool special=false) {
    std::vector<uint16_t> x(k),got(n),want(n);
    for(unsigned c=0;c<k;++c)x[c]=bf16(float(int(c%17)-8)/16);
    if(special){std::fill(x.begin(),x.end(),0);x[0]=bf16(1);x[1]=bf16(1.f/256);}
    uint8_t *dw,*ds;__nv_bfloat16 *dx,*dy;
    CUDA(cudaMalloc(&dw,w.size()));CUDA(cudaMalloc(&ds,s.size()));CUDA(cudaMalloc(&dx,k*2));CUDA(cudaMalloc(&dy,n*2));
    CUDA(cudaMemcpy(dw,w.data(),w.size(),cudaMemcpyHostToDevice));CUDA(cudaMemcpy(ds,s.data(),s.size(),cudaMemcpyHostToDevice));CUDA(cudaMemcpy(dx,x.data(),k*2,cudaMemcpyHostToDevice));
    gpt_oss_mxfp4_gemv_bf16<<<(n+3)/4,128>>>(dw,ds,dx,dy,n,k);
    CUDA(cudaGetLastError());CUDA(cudaDeviceSynchronize());CUDA(cudaMemcpy(got.data(),dy,n*2,cudaMemcpyDeviceToHost));
    unsigned detections[3]={0,0,0},max_ulp=0;
    for(unsigned r=0;r<n;++r){
        want[r]=reference(w,s,x,r,n,k,0);
        unsigned distance=unsigned(std::abs(int(want[r])-int(got[r])));max_ulp=std::max(max_ulp,distance);
        if(!std::isfinite(widen(want[r])) || distance>(exact?0:1)){fprintf(stderr,"%s row%u expected %04x got %04x\n",label,r,want[r],got[r]);return false;}
        for(unsigned bad=1;bad<=3;++bad)detections[bad-1]+=reference(w,s,x,r,n,k,bad)!=want[r];
    }
    if(special){
        // 2026-10-06: BMM 1+1/256 rounds to 1 BEFORE bias. A fused bias path gives 1+1/128.
        if(got[0]!=0x3f80 || bf16(widen(got[0])+1.f/256)!=0x3f80 || bf16(1.f+1.f/128)!=0x3f81)return false;
    } else if(!detections[0]||!detections[1]||!detections[2]){fprintf(stderr,"ineffective corruption control\n");return false;}
    printf("%s rows=%u K=%u max_bf16_ulp=%u controls=%u/%u/%u observed_bits=",label,n,k,max_ulp,detections[0],detections[1],detections[2]);
    for(auto v:got)printf("%04x,",v);printf("\n");
    CUDA(cudaFree(dw));CUDA(cudaFree(ds));CUDA(cudaFree(dx));CUDA(cudaFree(dy));return true;
}
int main(int argc,char** argv){
    if(!decode_test())return 1;
    for(unsigned k:{96U,2880U}){
        unsigned n=35;std::vector<uint8_t>w(size_t(n)*k/2),s(size_t(n)*k/32);
        for(size_t i=0;i<w.size();++i)w[i]=uint8_t((i*73+i/13)%256);
        for(size_t i=0;i<s.size();++i)s[i]=125+i%5;
        if(!run(n,k,w,s,"synthetic",true))return 1;
    }
    std::vector<uint8_t>w(16,0),s(1,127);w[0]=0x22;
    if(!run(1,32,w,s,"bmm-before-bias",true,true))return 1;
    for(int i=1;i<argc;++i){
        FILE*f=fopen(argv[i],"rb");if(!f)return 2;uint32_t n,k;
        if(fread(&n,4,1,f)!=1||fread(&k,4,1,f)!=1||n>128||k!=2880)return 2;
        std::vector<uint8_t>w(size_t(n)*k/2),s(size_t(n)*k/32);
        if(fread(w.data(),1,w.size(),f)!=w.size()||fread(s.data(),1,s.size(),f)!=s.size()||fgetc(f)!=EOF)return 2;
        fclose(f);if(!run(n,k,w,s,argv[i],false))return 1;
    }
    return 0;
}
