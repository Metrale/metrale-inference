#pragma once

// The MMQ engine: quantized-weight x quantized-activation GEMM tiles, their activation
// quantizers and the K-quant vector dot products. Derived from ggml's CUDA backend
// (llama.cpp ggml/src/ggml-cuda/); see THIRD_PARTY_NOTICES.md.

#include "mmq_tile.cuh"
#include "mmq_quantize.cuh"
