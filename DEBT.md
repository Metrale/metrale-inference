# DEBT

The register of known technical, compliance and process debt in this repository. A shortcut taken
on purpose gets a row in the same commit; a row is closed by the commit that pays it.

Columns: **id** · **what is missing** · **why deferred** · **what closes it** · **severity**
(S1 blocks customer traffic · S2 blocks a release · S3 blocks an audit claim · S4 hygiene).

## Kernels

| id | what is missing | why deferred | what closes it | severity |
|---|---|---|---|---|
| KERN-1 | Our own block-scale MMQ engine. The NVFP4 W4A4 dense-FFN prefill GEMM (`nvfp4_mmq`), the Q4_K / Q2_0 prefill GEMMs (`q4k_mmq`, `q2_0_mmq`), the Q4_K quantizer (`q4k_quantize`) and the DeepSeek-V4 K-quant MoE and Q6_K lm head (`kquant_moe`) still run on code derived from ggml/llama.cpp's CUDA MMQ (MIT). It was pruned to what the sm_121f, sm_90a and sm_100a builds of those consumers use and moved from `kernels/gb10/qwen3.6-27b/nvfp4/q4k_vendor/` to `kernels/gb10/common/mmq/` on branch `mk-mmq-own`, and it is still attributed in `THIRD_PARTY_NOTICES.md` §2. | Owner decision 2026-10-24: the rewrite follows the current kernel-cleanup work (dead-kernel removal, per-model dedupe into `common/`, the 500-line split). The derived engine is on the live dense-27B prefill path, so it cannot be dropped first. | A from-scratch MMQ written as LKB template families (CUDA-template instantiations, task-body ABI), with the same kernel entry points. It runs shadow-first behind a kill switch and must be bit-identical (or `numerics = "differs"` with a stated tolerance and a gate) against the derived kernels on every consumer and quant format. The derived files and the §2 notice entry are deleted in the commit that promotes it. | S3 |
