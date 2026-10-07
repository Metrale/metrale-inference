// SPDX-License-Identifier: MIT OR Apache-2.0
// 2026-10-07: Single-frame FP32 image decoder residual operations. Checkpoint
// weights remain FP32; existing BF16 GEMMs cannot represent this operand policy.
// Diagnostic direct convolution, not a tuned convolution/performance claim.
#include <stdint.h>
#include <math.h>

extern "C" __global__ void image_vae_conv2d_f32(
    const float* x, const float* weight, const float* bias, float* y,
    uint32_t input_channels, uint32_t output_channels, uint32_t height,
    uint32_t width, uint32_t kernel) {
    const uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const uint64_t pixels=uint64_t(height)*width;
    if(i>=pixels*output_channels)return;
    const uint32_t oc=i/pixels, oy=(i%pixels)/width, ox=i%width;
    const int pad=kernel/2;
    float sum=0.0f;
    for(uint32_t ic=0;ic<input_channels;++ic)
        for(uint32_t ky=0;ky<kernel;++ky)
            for(uint32_t kx=0;kx<kernel;++kx){
                const int iy=int(oy)+int(ky)-pad, ix=int(ox)+int(kx)-pad;
                if(iy>=0&&iy<int(height)&&ix>=0&&ix<int(width))
                    sum=fmaf(x[uint64_t(ic)*pixels+uint64_t(iy)*width+ix],
                        weight[((uint64_t(oc)*input_channels+ic)*kernel+ky)*kernel+kx],sum);
            }
    y[i]=__fadd_rn(sum,bias[oc]);
}

// One block per spatial point, channels distributed over exactly 256 threads.
// Pinned FP32 VAE: L2-normalize with clamp_min(1e-12), then scale*gamma.
extern "C" __global__ void image_vae_norm_f32(
    const float* x,const float* gamma,float* y,uint32_t channels,uint32_t pixels){
    const uint32_t pixel=blockIdx.x,tid=threadIdx.x;
    __shared__ float partial[256];
    float sum=0.0f;
    for(uint32_t c=tid;c<channels;c+=256){float v=x[uint64_t(c)*pixels+pixel];sum=__fadd_rn(sum,__fmul_rn(v,v));}
    partial[tid]=sum;__syncthreads();
    for(uint32_t stride=128;stride>0;stride>>=1){if(tid<stride)partial[tid]=__fadd_rn(partial[tid],partial[tid+stride]);__syncthreads();}
    const float denominator=fmaxf(sqrtf(partial[0]),1e-12f),scale=sqrtf(float(channels));
    for(uint32_t c=tid;c<channels;c+=256){float v=__fdiv_rn(x[uint64_t(c)*pixels+pixel],denominator);y[uint64_t(c)*pixels+pixel]=__fmul_rn(__fmul_rn(v,scale),gamma[c]);}
}

extern "C" __global__ void image_vae_silu_f32(const float* x,float* y,uint32_t count){
    const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count)y[i]=x[i]/(1.0f+expf(-x[i]));
}
