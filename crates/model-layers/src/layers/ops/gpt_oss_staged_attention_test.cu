// SPDX-License-Identifier: MIT OR Apache-2.0
// Standalone Torch-CUDA operand harness; no model factory registration.
#include <cuda_runtime.h>
#include "gpt_oss_staged_attention.cu"
extern "C" int staged_run(void* q, void* k, void* v, void* output,
    void* tables, void* lengths, void* sinks, unsigned int max_blocks,
    unsigned int q_heads, unsigned int kv_heads, unsigned int block_size,
    float scale, unsigned int window, unsigned long long stream) {
    auto s = reinterpret_cast<cudaStream_t>(stream);
    gpt_oss_staged_attention_bf16<<<dim3(q_heads, 1), 256, 0, s>>>(
        static_cast<__nv_bfloat16*>(q), static_cast<__nv_bfloat16*>(k),
        static_cast<__nv_bfloat16*>(v), static_cast<__nv_bfloat16*>(output),
        static_cast<unsigned int*>(tables), static_cast<unsigned int*>(lengths),
        max_blocks, q_heads, kv_heads, 64, block_size, scale, q_heads * 64,
        window, static_cast<__nv_bfloat16*>(sinks));
    auto error = cudaGetLastError();
    return error == cudaSuccess ? cudaStreamSynchronize(s) : error;
}
