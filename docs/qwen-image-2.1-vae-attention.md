# Qwen Image 2.1: wide VAE attention residual

2026-10-06. Checkpoint context: [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1), [immutable revision](https://huggingface.co/Qwen/Qwen-Image-2.1/tree/d26bb61231c349cf6b7896fa83353113880e1ba3). The evidence here uses constructed operands, not learned weights.

The original FP32 VAE needs noncausal single-head attention at head dimension 1152. The existing NLLB kernel requires head dimension equal to the thread-block size, which cannot represent 1152 threads. `image_vae_attention_f32` is an explicit diagnostic residual: 128 threads cooperate across nine channels per thread, with FP32 online softmax and separate multiply/add. No existing dispatch changes. The wrapper accepts single-frame channel-first QKV, validates geometry/pointers/non-overlap, and preserves the caller stream. This is a correctness path, not a throughput claim.

## Evidence

Spark1, CUDA 13.0, SM121, Torch 2.13.0+cu130, TF32 disabled. Six constructed cases all remain finite. Singleton, uniform attention over four pixels, and a large-logit control match FP32 SDPA MATH exactly. Random cases retain the failed exact-reference gate:

| Pixels | Different FP32 elements | Relative L2 error |
|---|---:|---:|
| 3 | 2,117 / 3,456 | 7.21e-8 |
| 17 | 16,042 / 19,584 | 1.43e-7 |
| 65 | 66,424 / 74,880 | 2.50e-7 |

An independent FP64 explicit softmax is reported separately; it does not redefine the exact gate. Wrong causal masking and returning the first value instead of attending to every pixel are detected. One scoped Rust launch/validation test and scoped Clippy pass. The same source compiles to SM90 and SM100 PTX; this is compile evidence, not runtime validation on those devices.

Kernel source SHA-256: `0414cb54fe47ca69ad6311abb672248b60baa85eee3c3b5c615db1f0bcad4bb2`.
Probe binary SHA-256: `bc1be89e6ad93b3f28212e544d0ce70cdd784845d27982982b6edbb932584ba9`.
The first probe failed before execution because `nvcc` was absent from SSH PATH; the rerun explicitly selected `/usr/local/cuda-13.0/bin/nvcc`. Both logs are retained with private raw operands, outputs and build receipt.

Reproduce with `NVCC=/path/to/nvcc python scripts/qwen_image21/vae_attention_diagnostic.py REPOSITORY NEW_OUTPUT_DIRECTORY` in the pinned reference environment. The script records failures without changing tolerances and requires a fresh output directory.

## Still open

Actual-weight full decoder comparison, native image generation, model quality, efficient large-image attention, cross-device runtime validation and certification remain open. This component does not establish native image support or commercial readiness.
