// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-24: Contiguous BF16 prefill flash attention at head_dim 64 for the
// llama-nemotron-embed-vl-1b-v2 text tower (MODEL.toml head_dim = 64):
// kernels/gb10/common/attn_prefill.cu instantiated at HDIM 64 as `attn_prefill_h64`, the
// mechanism the Gemma-4 targets use for HDIM 512. The embedder launches it with causal = 0
// (bidirectional) and sliding_window = 0, one launch per packed sequence.
//
// Owner: gb10 kernels.
// Invariants: head_dim must equal HDIM (64); the 64-row variant is not built.

#define HDIM 64
#define METRALE_PREFILL_ENTRY attn_prefill_h64
#define METRALE_SKIP_PREFILL_64 1
#include "../../common/attn_prefill.cu"
