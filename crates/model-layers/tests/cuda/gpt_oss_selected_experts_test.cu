// SPDX-License-Identifier: MIT OR Apache-2.0
// 2026-10-07: Same-operand grouped/serial parity harness. No timing qualification.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include "../../../../kernels/gb10/common/gpt_oss_mxfp4_gemv.cu"
#include "../../../../kernels/gb10/common/gpt_oss_expert_ops.cu"

// 2026-10-07: Independent serial bias reference preserves the incumbent BF16 add.
__global__ void serial_bias(__nv_bfloat16* values, const __nv_bfloat16* bias,
                            unsigned count) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) values[i] = __float2bfloat16_rn(
        __bfloat162float(values[i]) + __bfloat162float(bias[i]));
}

// 2026-10-07: Pointers belong to the Python Torch fixture. Inputs are never copied
// or converted; the serial branch selects identical checkpoint expert offsets.
extern "C" int selected_run(uintptr_t bp, uintptr_t sp, uintptr_t xp,
    uintptr_t ip, uintptr_t yp, uintptr_t biasp, unsigned rows, unsigned cols,
    unsigned input_stride, int grouped, int add_bias) {
    if (!rows || !cols || cols % 32 || (input_stride && input_stride != cols)) return -1;
    auto* blocks = reinterpret_cast<const unsigned char*>(bp);
    auto* scales = reinterpret_cast<const unsigned char*>(sp);
    auto* input = reinterpret_cast<const __nv_bfloat16*>(xp);
    auto* ids = reinterpret_cast<const unsigned*>(ip);
    auto* output = reinterpret_cast<__nv_bfloat16*>(yp);
    auto* bias = reinterpret_cast<const __nv_bfloat16*>(biasp);
    if (grouped) {
        gpt_oss_mxfp4_selected_bf16<<<dim3((rows+3)/4,4),128>>>(
            blocks, scales, input, ids, output, rows, cols, input_stride);
        if (add_bias) gpt_oss_selected_bias_bf16<<<(4*rows+255)/256,256>>>(
            output,bias,ids,rows);
    } else {
        unsigned selected[4];
        auto status=cudaMemcpy(selected,ids,sizeof(selected),cudaMemcpyDeviceToHost);
        if(status!=cudaSuccess) return int(status);
        for(unsigned s=0;s<4;++s) {
            if(selected[s]>=32) return -2;
            for(unsigned t=0;t<s;++t) if(selected[s]==selected[t]) return -2;
            const size_t expert=selected[s];
            gpt_oss_mxfp4_gemv_bf16<<<(rows+3)/4,128>>>(
                blocks+expert*rows*(cols/2),scales+expert*rows*(cols/32),
                input+size_t(s)*input_stride,output+size_t(s)*rows,rows,cols);
            if(add_bias) serial_bias<<<(rows+255)/256,256>>>(
                output+size_t(s)*rows,bias+expert*rows,rows);
        }
    }
    auto status=cudaGetLastError();
    if(status!=cudaSuccess) return int(status);
    return int(cudaDeviceSynchronize());
}
