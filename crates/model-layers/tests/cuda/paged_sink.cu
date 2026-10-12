// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-07: Sink semantics and legacy bit-identity. Supply baseline source as
// baseline_paged_decode_attn.cu beside this harness, from git before the edit.
// nvcc --std=c++17 --fmad=false -arch=sm_121 -DHDIM=64 paged_sink.cu -o paged-sink
// Repeat HDIM 128/256/512 for existing compile points. FP32 attention policy only.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include "../../../../kernels/gb10/common/paged_decode_attn.cu"
namespace baseline {
#define paged_decode_attn baseline_decode
#define paged_decode_attn_splitk baseline_splitk
#define paged_decode_attn_reduce baseline_reduce
#include "baseline_paged_decode_attn.cu"
#undef paged_decode_attn
#undef paged_decode_attn_splitk
#undef paged_decode_attn_reduce
}
#define CK(call) do { auto e = (call); if(e != cudaSuccess) { \
std::fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(e)); std::exit(1); } } while(0)
template<typename T> T* upload(const std::vector<T>& v) {
    T* p; CK(cudaMalloc(&p, v.size()*sizeof(T)));
    CK(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice)); return p;
}
static float as_float(__nv_bfloat16 x) { return __bfloat162float(x); }
// 2026-10-07: Lossless BF16 bit arrays for independent offline replay.
static void array(FILE* f, const char* name, const std::vector<__nv_bfloat16>& values) {
    std::fprintf(f, "\"%s\":[", name);
    for(unsigned i=0;i<values.size();i++) std::fprintf(f,"%s%u",i?",":"",unsigned(__bfloat16_as_ushort(values[i])));
    std::fprintf(f,"]");
}
int main(int argc, char** argv) {
    if(argc>2) return 8;
    FILE* cases = nullptr;
    if(argc==2) {
        std::string path=std::string(argv[1])+"/cases-hd"+std::to_string(HDIM)+".jsonl";
        cases=std::fopen(path.c_str(),"w"); if(!cases) return 8;
    }
    const unsigned qh=64, kh=8, bs=16, blocks=9, stride=qh*HDIM+64;
    std::vector<__nv_bfloat16> k(blocks*bs*kh*HDIM),v(k.size()),sink(qh);
    for(unsigned t=0;t<blocks*bs;t++) for(unsigned h=0;h<kh;h++) for(unsigned d=0;d<HDIM;d++) {
        k[(t*kh+h)*HDIM+d]=__float2bfloat16(d==0 ? float(int(t%11)-5)/8 : 0);
        v[(t*kh+h)*HDIM+d]=__float2bfloat16(float(int(t%17)-8)+float(d%7)/8+float(h)/4);
    }
    auto dk=upload(k),dv=upload(v);
    unsigned comparisons=0, detections=0, missing_detections=0;
    for(unsigned seqs: {1u,16u,128u}) {
        std::vector<__nv_bfloat16> q(seqs*stride),old(seqs*qh*HDIM),now(old.size()),got(old.size());
        std::vector<int> tables(seqs*blocks),lens(seqs);
        for(unsigned s=0;s<seqs;s++) {
            lens[s]=129;
            for(unsigned b=0;b<blocks;b++) tables[s*blocks+b]=int(b);
            for(unsigned h=0;h<qh;h++) q[s*stride+h*HDIM]=__float2bfloat16(float(int(h%7)-3)/8);
        }
        if(argc==2 && seqs==1) {
            std::string path=std::string(argv[1])+"/inputs-hd"+std::to_string(HDIM)+".json";
            FILE* f=std::fopen(path.c_str(),"w"); if(!f) return 8;
            std::fprintf(f,"{\"schema\":1,\"head_dim\":%u,\"q_heads\":%u,\"kv_heads\":%u,\"block_size\":%u,\"max_blocks\":%u,\"q_stride\":%u,\"scale\":0.125,\"block_table\":[0,1,2,3,4,5,6,7,8],",HDIM,qh,kh,bs,blocks,stride);
            array(f,"q_bits",q); std::fprintf(f,","); array(f,"k_bits",k); std::fprintf(f,","); array(f,"v_bits",v); std::fprintf(f,"}\n");
            if(std::fclose(f)!=0) return 8;
        }
        auto dq=upload(q),dold=upload(old),dnow=upload(now),dgot=upload(got);
        auto dt=upload(tables),dl=upload(lens);
        for(unsigned length: {0u,1u,7u,128u,129u}) for(unsigned window:{0u,128u}) {
            for(auto& l:lens) l=int(length);
            CK(cudaMemcpy(dl,lens.data(),seqs*sizeof(int),cudaMemcpyHostToDevice));
            CK(cudaMemset(dold,0x55,old.size()*2)); CK(cudaMemset(dnow,0x55,now.size()*2));
            baseline::baseline_decode<<<dim3(qh,seqs),256>>>(dq,dk,dv,dold,dt,dl,blocks,qh,kh,HDIM,bs,0.125f,stride,window);
            paged_decode_attn<<<dim3(qh,seqs),256>>>(dq,dk,dv,dnow,dt,dl,blocks,qh,kh,HDIM,bs,0.125f,stride,window);
            CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(old.data(),dold,old.size()*2,cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(now.data(),dnow,now.size()*2,cudaMemcpyDeviceToHost));
            for(unsigned i=0;i<old.size();i++) if(__bfloat16_as_ushort(old[i])!=__bfloat16_as_ushort(now[i])) return 2;
            comparisons+=old.size();
            if(seqs!=1) continue;
            for(float sink_value:{-INFINITY,-10.0f,0.0f,5.0f,1000.0f,INFINITY,NAN}) {
                for(unsigned h=0;h<qh;h++) sink[h]=__float2bfloat16(sink_value+float(h%5)/8);
                auto ds=upload(sink);
                paged_decode_attn_sink<<<dim3(qh,seqs),256>>>(dq,dk,dv,dgot,dt,dl,blocks,qh,kh,HDIM,bs,0.125f,stride,window,ds);
                CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
                CK(cudaMemcpy(got.data(),dgot,got.size()*2,cudaMemcpyDeviceToHost)); CK(cudaFree(ds));
                std::vector<__nv_bfloat16> repeated(got.size());
                for(unsigned h=0;h<qh;h++) for(unsigned d=0;d<HDIM;d++) {
                    float actual=as_float(got[h*HDIM+d]);
                    const float head_sink=as_float(sink[h]);
                    if(length==0) { if(actual!=0) return 3; continue; }
                    if(std::isnan(sink_value)||sink_value==INFINITY) { if(!std::isnan(actual)) return 4; repeated[h*HDIM+d]=__float2bfloat16(NAN); continue; }
                    if(sink_value==-INFINITY && __bfloat16_as_ushort(got[h*HDIM+d])!=__bfloat16_as_ushort(old[h*HDIM+d])) return 5;
                    const unsigned start=window && length>window?length-window:0;
                    double denom=0, numer=0, maxlog=std::max(double(head_sink),1.0);
                    for(unsigned t=start;t<length;t++) {
                        double logit=as_float(q[h*HDIM])*as_float(k[(t*kh+h/(qh/kh))*HDIM])*0.125;
                        double e=std::exp(logit-maxlog);denom+=e;numer+=e*as_float(v[(t*kh+h/(qh/kh))*HDIM+d]);
                    }
                    double mass=std::exp(double(head_sink)-maxlog);
                    float expected=float(numer/(denom+mass));
                    if(std::abs(actual-expected)>0.04f) { std::fprintf(stderr,"mismatch hd%d t%u w%u s%g h%u d%u %g != %g\n",HDIM,length,window,sink_value,h,d,actual,expected);return 6; }
                    // 2026-10-07: Wrong policy adds the same sink once in each of eight warps.
                    float wrong=float(numer/(denom+8*mass));
                    repeated[h*HDIM+d]=__float2bfloat16(wrong);
                    if(std::abs(expected-wrong)>0.08) detections++;
                    if(std::abs(expected-as_float(old[h*HDIM+d]))>0.08) missing_detections++;
                }
                if(cases) {
                    std::fprintf(cases,"{\"length\":%u,\"window\":%u,",length,window);
                    array(cases,"sink_bits",sink); std::fprintf(cases,","); array(cases,"observed_bits",got); std::fprintf(cases,","); array(cases,"baseline_bits",old); std::fprintf(cases,","); array(cases,"known_bad_repeated_bits",repeated); std::fprintf(cases,"}\n");
                }
            }
        }
        CK(cudaFree(dq)); CK(cudaFree(dold)); CK(cudaFree(dnow)); CK(cudaFree(dgot)); CK(cudaFree(dt)); CK(cudaFree(dl));
    }
    CK(cudaFree(dk)); CK(cudaFree(dv));
    if(cases && std::fclose(cases)!=0) return 8;
    if(detections==0 || missing_detections==0) return 7;
    std::printf("PASS hd%d legacy_bitwise=%u per_warp_sink_detections=%u missing_sink_detections=%u\n",HDIM,comparisons,detections,missing_detections);
}
