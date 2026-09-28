# Kernel and architecture roadmap

Which open-weight LLM architectures the engine serves, which kernels serve them and how far those
kernels have been optimised, and what it would take to serve the prominent architectures it does
not serve yet. Written 2026-09-28 against `main` at `88e14520`.

| Section | What it answers |
|---|---|
| [Methodology](#methodology) | Sources, and how "supported", "has kernels" and "optimised" are decided |
| [Shared kernel stack](#shared-kernel-stack) | The kernels every decoder family runs, with measured % of floor |
| [Table A](#table-a-supported-architectures) | Supported architectures: checkpoints, kernels per component, hardware, optimisation status, gaps |
| [Table B](#table-b-not-supported-kernels-already-exist) | Not supported, but every component maps onto kernels we already have |
| [Table C](#table-c-not-supported-kernels-missing) | Not supported, and at least one component has no kernel family |
| [Summary roadmap](#summary-roadmap) | Top 5 by prominence × closeness |
| [Sources](#sources) | One primary source per architecture |

## Methodology

### Sources

- **What the engine accepts.** `loader_for_config` in `crates/model-engine/src/factory.rs` (the
  `model_type` match, lines 58–101), the config dispatch in `crates/config/src/dispatch.rs`, the GGUF
  mapping in `crates/config/src/gguf.rs` (`arch_to_model_type`, lines 54–68), and the kernel-target
  resolver in `crates/kernels/src/resolve.rs` with its caller
  `crates/server/src/main_modules/serve_load/model_setup.rs` (lines 199–216).
- **Which kernels exist and where they are launched.** [`KERNEL-PERF.md`](KERNEL-PERF.md) and its
  inputs under [`docs/kernel-perf/`](docs/kernel-perf/) (`taxonomy.toml`, `tradeoffs.toml`,
  `measurements.toml`), plus `python3 scripts/kernel_perf.py --json` for the joined inventory: 1,329
  kernel entry points in 341 source files, 7 hardware trees, 58 compiled targets, 15 architecture
  families and 29 components.
- **Which checkpoints are served.** The `kernels/<hw>/<model>/MODEL.toml` `[[model_types]]` tables
  and the launch recipes under [`recipes/`](recipes/README.md).
- **Optimisation history.** The measured rows in
  [`docs/kernel-perf/MEASUREMENTS.md`](docs/kernel-perf/MEASUREMENTS.md); merged PRs #1, #2, #4, #14,
  #18, #25, #34 (MoE decode) and #39 (prefill); the dense concurrency-ladder campaign in
  [`bench/ladder38/RESULTS.md`](bench/ladder38/RESULTS.md); and the overnight 2026-09-27/28 prefill
  and MoE-decode experiment logs, summarised here.
- **The architectures in the wild.** Hugging Face `config.json` files and model cards, papers, and the
  supported-model lists of the major open serving engines, read 2026-09-28. One source per
  architecture is listed under [Sources](#sources).

### "Supported"

An architecture is **supported** when all three hold:

1. `loader_for_config` accepts its `model_type` (after the config dispatch has normalised it);
2. a compiled kernel target resolves for its `(model_type, hidden_size)`. Resolution tries exact
   `hidden_size` declarations first, then wildcard ones, and refuses to start when neither exists
   ("No compiled kernel target matches", `model_setup.rs:205`);
3. at least one checkpoint of it has a kernel directory, a recipe or a documented serving run.

A `model_type` that passes (1) but not (2) is listed in Table B, with the reason. Two consequences
are easy to miss:

- **Several targets pin `hidden_size`.** `gemma4` (2816, 5376), `laguna` (2048, 3072), `step3p7`
  (4096), `glm5_next` (4096), `qwen4_exp` (2560), `deepseek_v41` (5120), `kimi_k3` (7168, 1024) and
  `qwen3_6_moe` (2048, 3072, 4096, 5120) have no wildcard entry. Another size of the same family is
  refused at load until a `MODEL.toml` declares it.
- **The GGUF path maps more names than it serves.** `gguf.rs:56-57` maps GGUF `llama` and `qwen2` to
  `model_type = "mistral"`. That resolves to the `mistral-small-4` target (a wildcard), whose loader
  reads MLA tensor names (`wq_a`, `wkv_a_with_mqa`, `crates/model-arch/src/mistral_loader/loader_impl/phase_lora_qkv.rs:31,77`).
  A plain GQA Llama or Qwen2 GGUF therefore has no working load path. The `qwen3` → `qwen3_5`
  mapping is exercised by the Ternary-Bonsai-27B GGUF (a Qwen3.6 GDN hybrid, see
  [`docs/TERNARY_BONSAI_RECEIPT.md`](docs/TERNARY_BONSAI_RECEIPT.md)), not by a full-attention Qwen3,
  and `qwen3moe` → `qwen3_5_moe` and `gemma3` → `gemma4` by no documented run. Such rows are marked
  "mapped, not demonstrated".

### "Has kernels"

A component (in the 29-component taxonomy of `KERNEL-PERF.md`) **has kernels** when an equivalent
kernel family exists in the hardware tree, even if it is not yet wired for the model in question.
Examples: the DeepSeek-V3-style MLA kernels (`mla_absorbed.cu`, `mla_fused_prefill.cu`,
`paged_decode_attn_mla.cu`, ...) under `kernels/gb10/mistral-small-4/nvfp4/` and
`kernels/gb10/deepseek-v4-flash/nvfp4/`, which Mistral Small 4 and LongCat-Flash-Lite already launch,
count for any DeepSeek-V3-shaped MLA; `kernels/gb10/common/mamba2_*.cu` (runtime `n_groups` and state
size) count for any Mamba2 layer.

- **Table B** holds architectures where every component maps onto an existing kernel family. Work
  left inside a family is listed as a **kernel delta** in the "What's missing" column: a mask or score
  mode an attention family has in some kernels but not the one needed (chunked-local windows, sinks at
  a new `head_dim`, a score softcap), a new `head_dim` instantiation, a router selection mode
  (group-limited top-k), or one elementwise activation.
- **Table C** holds architectures where at least one component has no equivalent kernel family on
  any CUDA tree: a recurrence we do not implement (Mamba1, lightning attention, RWKV-7), a new
  attention algorithm, a modality tower with no kernels (audio), or a weight format with no GEMM or
  GEMV (INT4 weight-only).

Effort: **S** ≤ 1 week (loader, config mapping, `MODEL.toml`, chat template, and small kernel deltas),
**M** 1–3 weeks (a new kernel variant plus validation), **L** > 3 weeks (a new kernel family and its
prefill, decode and speculative-verify paths). These are estimates for one engineer, not measurements.

Priority weighs prominence (downloads, lab release cadence, presence in other engines' supported
lists) against hardware fit: what fits one GB10 (≈ 110 GB of weights and KV at the 0.85 utilisation
ceiling), two GB10s with expert parallelism (the `-ep2` recipes), or only the B200/B300/Hopper trees.

### "Optimised"

- **Measured % of floor.** `pct_of_floor = floor_us / time_us`, where the floor is the larger of the
  minimum DRAM traffic over the measured 249 GB/s and the useful FLOPs over the measured tensor-core
  peak of the format the kernel implements (definitions in
  [`KERNEL-PERF.md` § 6](KERNEL-PERF.md#6-of-floor-how-far-a-kernel-is-from-the-hardware-limit)).
  Measurements exist for **61 entry points on GB10 only**, in two models:
  `Qwen/Qwen3.6-35B-A3B-FP8` and `unsloth/Qwen3.8-27B-NVFP4`, at decode C=1, decode C=16, prefill 4k
  and prefill 32k. Every other kernel is **not yet measured**. A shared kernel measured in a Qwen
  regime is reported here as "measured on Qwen"; it is not a measurement for the other families that
  launch it.
- **Measured before the prefill campaign.** All 199 rows were taken with the kernels as of `e37e3cb2`,
  which is **before PR #39**. The prefill rows therefore describe the pre-#39 prefill kernels. #39
  roughly halved 32k cold TTFT on both measured models, and its new kernels are not re-measured yet.
  Where the overnight microbenchmarks give a throughput for a #39 kernel, it is quoted as
  "TFLOPS / measured peak", which is a different quantity from a `KERNEL-PERF` row.
- **Campaign labels.** "campaign #34" and "campaign #39" name merged PRs of this repository.
  "optimised overnight 2026-09-28" is the #39 prefill work. "ladder campaign 2026-08" is the dense
  C=1..128 concurrency ladder in [`bench/ladder38/RESULTS.md`](bench/ladder38/RESULTS.md).

## Shared kernel stack

Every decoder family runs these components through the shared layer code
(`crates/model-layers/src/layers/`). Paths are under `kernels/gb10/common/` unless noted; the same
files, or forks listed in [`kernels/FORKS.md`](kernels/FORKS.md), compile for `hopper`, `b200`, `b300`,
`strix` and `strix-hip`. Measured % of floor is GB10 only, on the two measured Qwen models, with the
pre-#39 kernels (see [Methodology](#optimised)).

| Component | Kernels (gb10) | Measured % of floor (Qwen3.6-35B-A3B-FP8 / Qwen3.8-27B-NVFP4) | Status |
|---|---|---|---|
| Attention, GQA/MHA | `paged_decode_attn*.cu` (bf16, fp8, nvfp4 KV; split-K; `_bf16_gqa`, `_fp8_gqa`; head_dim 128/256/512), `attn_prefill*.cu`, `prefill_paged_compute*.cuh`; TurboQuant KV variants `*_turbo{2,3,4,8}*` | decode C=1: 11 / 18 %; decode C=16: 44 / 45 %; prefill 4k/32k: 23–27 % on both | **optimised overnight 2026-09-28** (#39): the paged prefill twin reaches 63–67 TFLOPS (51–54 % of the 123.7 TFLOPS BF16 peak) in its microbenchmark; attention prefill ran at about 30 TFLOPS in the model before. Its limit is register and shared-memory pressure (250 registers, 96 of 99 KiB smem). Long-context decode (KV ≫ 600) is not measured |
| Projection GEMM/GEMV, BF16 | `dense_gemv_bf16*.cu`, `dense_gemm_bf16.cu`, `dense_gemm_tc.cu`, `dense_gemm_splitk.cu` | decode C=1: 50–100 % (median 95–96); decode C=16: 86–95 % dense, 16 % for `dense_gemv_bf16_batchm` on the MoE; prefill `dense_gemm_bf16` 1–2 % (the MTP drafter's context prefill) | decode GEMVs at floor except the batched-M GEMV at C=16. The 1–2 % prefill rows were fixed by #39, which moved the drafter prefill onto tensor cores (dense 32k drafter prefill 3.19 s → 0.29 s); not re-measured |
| Projection GEMM/GEMV, FP8 | `w8a16_gemv*.cu`, `w8a16_gemm*.cu` (pipelined m32/m64/m128), `fp8_gemm_t_blockscaled.cu`, `fp8_gemm_blockscaled_pipe.cu`, `w4a16_fp8_ldmab.cu`, `dense_gemv_fp8w*.cu` | decode (MoE only) C=1 55–95 % (median 91), C=16 20–76 % (median 71); prefill 11–49 % | **campaign #34** (canonical row tiers, skinny-M 1..16-row kernels) for decode. **#39** for prefill: the `fp8_fp8_gemm_ldmab` multistage rewrite went 28 → 172 TFLOPS (~70 % of the 243.6 TFLOPS FP8 peak) and the MoE model's FP8 projection GEMM reached 97–112 TFLOPS (40–46 %), both in microbenchmarks |
| Projection GEMM/GEMV, NVFP4 W4A16 | `w4a16_gemv*.cu`, `w4a16_gemv_tc.cu`, `w4a16_gemm.cu` (+ per-model forks) | decode C=1 59–99 % (median 89 dense); C=16 8–92 %; prefill 28–32 % | ladder campaign 2026-08; #1; decode near floor at C=1 |
| Projection GEMM/GEMV, W4A4 | `nvfp4_mmq.cu` (`kernels/gb10/qwen3.6-27b/nvfp4/`), `w4a4_gemv_mx.cu`, `w4a4_gemv_mx_ps.cu` | decode C=16 82–86 %; prefill 17–18 % for the 128-row tile (the 32-row tile reads 82–87 %) | #14, #18 (activation-reuse and persistent GEMVs); **#39** pipelined NVFP4 FFN GEMM at 272–276 TFLOPS in the engine's own PTX (~56 % of the 490.8 FP4 peak) |
| Projection, integer / K-quant | `q2_0_gemv*.cu`, `kquant_moe.cu`, `dequant_gguf_bf16.cu`; `q2_0_mmq.cu`, `q4k_mmq.cu` (qwen3.6-27b) | not measured | Q2_0, Q2_K, Q3_K, Q4_K, Q6_K, Q8_0 decode or dequantise; MLX INT8 on Metal |
| Normalization | `rms_norm*.cu` (fused residual, gated, L2), per-model `rms_norm.cu` forks | prefill 78–96 %; decode 4–15 % at C=1 (1–4 rows, launch-bound), 69 % at C=16 | at floor where bandwidth-bound; C=1 is latency, recovered only by fusion |
| Elementwise, activations | `residual_add.cu`, `moe_silu_mul.cu`, `relu_squared.cu`, gelu (`kernels/gb10/gemma-4-*/nvfp4/gelu.cu`), `glm5next_ffn.cu` (clamped SwiGLU) | residual add 97–100 %; fused SiLU·quant 57–87 % | at floor |
| RoPE | `rope.cu` (plain, proportional, YaRN, interleaved, `inv_freq` table), `rope_mrope_interleaved.cu`, fused K-norm+RoPE+cache writes | MRoPE 6 % (prefill; 0.4 % of regime time) | low priority by time share |
| KV cache | `reshape_and_cache*.cu`, `quantize_bf16_to*.cu`, TurboQuant (`wht_bf16.cu`, `tq_plus_*`) | not measured | — |
| Quantisation | `per_token_group_quant_fp8.cu`, `quant_rowwise_fp8.cu`, `quantize_bf16_to_{fp8_blockscaled,nvfp4}.cu`, `e2m1_branchless.cu`, the `nvfp4_mmq` quantizers | 21–103 % | fused into GEMM prologues where #39 did so |
| Embedding, LM head, sampling | `token_overlay.cu`, `embed_from_argmax.cu`, `argmax_bf16.cu`, `argmax_feed.cu`, per-model `logit_softcap.cu`, `embed_scale.cu` | argmax 2 % (single-CTA, 1.1 % of regime time) | latency-bound |
| Speculative decoding | MTP heads through the shared layers; `dflash2.cu` (block-diffusion drafter); GDN verify kernels (`gdn_verify_fused_*.cu`) | GDN verify 35 % (C=16) | #2, #34 (carried-state batched verify) |
| LoRA | `lora_bgmv.cu`, `moe_lora_gather_bgmv.cu`, `moe_lora_grouped_down.cu` | not measured | — |

## Table A: supported architectures

15 families, as `KERNEL-PERF.md` groups them. "Kernels" lists the family-specific components only;
every family also runs the [shared kernel stack](#shared-kernel-stack). `common/` is
`kernels/gb10/common/`, and `<dir>/` is `kernels/gb10/<dir>/nvfp4/`. "Hardware" lists the trees with a
target for the family; a tree that compiles a shared file without a target for the family is not
listed.

| Family (`model_type`) | Checkpoints we serve | Family-specific kernels (gb10) | Hardware | Optimisation status | Notable gaps |
|---|---|---|---|---|---|
| **Qwen3.x GDN hybrid, dense** (`qwen3_5`; Metal `qwen3_5_vl`) | `unsloth/Qwen3.8-27B-NVFP4` (dense flagship), `Qwen/Qwen3.6-27B-FP8`, `nvidia/Qwen3.6-27B-NVFP4`, `Kbenkhaled/Qwen3.5-27B-NVFP4`, `Qwen/Qwen3.5-0.8B`, Holo-3.1 0.8B/4B, Ornith-1.0-9B, `prism-ml/Ternary-Bonsai-27B` (GGUF Q2_0), `mlx-community/Qwen3.5-4B-MLX-8bit` (Metal) | **GDN:** `common/gated_delta_rule_*.cu` (17 WY, FLA-chunk, carried-state, persistent and register-resident variants), `common/gdn_chunk_fwd_o_mma8.cu`, `common/gdn_verify_fused_*.cu`, `common/ssm_preprocess.cu`, `common/ssm_state_norm.cu`, forks in `qwen3.6-27b/`, `qwen3.6-35b-a3b/`. **Conv:** `common/causal_conv1d.cu`. **Dense FFN:** `qwen3.6-27b/nvfp4_mmq.cu` (W4A4), `w4a16_gemm.cu`, `common/w4a4_gemv_mx*.cu`; `qwen3.6-27b/q2_0_mmq.cu`, `q4k_mmq.cu`. **Vision:** `qwen3.6-35b-a3b/vision_encoder.cu` | gb10, hopper (3.6-27B, 3.8-27B), strix and strix-hip (3.6-27B), metal (4B MLX INT8) | **Measured (Qwen3.8-27B-NVFP4, gb10):** decode C=1 median 89 % (projection GEMVs 59–99 %, GDN `wy4` 58 %); decode C=16 median 68 %; prefill 4k/32k median 36 %/29 % (pre-#39). **Campaigns:** ladder campaign 2026-08 (ahead of vLLM+MTP at C=1..128); #1, #2, #14, #18; **optimised overnight 2026-09-28** (#39): 32k cold TTFT 45.0 s → 17.3–18.0 s (vLLM 24.1 s), warm 12.4 s → 0.23–0.27 s | GDN prefill spine (`recompute_wu` → `chunk_delta_h` → `chunk_fwd_o`) was at 25–40 % of floor before #39 (#39 then made `chunk_fwd_o` 2.6× faster in its microbenchmark). It uses no TMA or mbarrier pipelining, and an external fused, warp-specialised spine measured 8.0× faster on the spine alone (Qwen3.6-35B-A3B-NVFP4, 2026-08-21). Hopper, Strix and Metal have no measured peaks, so nothing there is measured |
| **Qwen3.x GDN hybrid, MoE** (`qwen3_5_moe`, `qwen3_6_moe`, `holo3_1_moe`, `qwen3_next`) | `Qwen/Qwen3.6-35B-A3B-FP8` (MoE flagship), `nvidia/Qwen3.6-35B-A3B-NVFP4`, `Sehyo/Qwen3.5-35B-A3B-NVFP4`, `Sehyo/Qwen3.5-122B-A10B-NVFP4`, `nvidia/Qwen3.5-397B-A17B-NVFP4`, `nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4`, `Qwen/Qwen3-Coder-Next-FP8`, Holo-3.1-35B-A3B | GDN and conv as above (plus forks in `qwen3-next-80b-a3b/`, `qwen3.5-122b-a10b/`, `qwen3.6-35b-a3b/gated_delta_rule_wy17.cu`). **MoE:** `common/moe_shared_expert_fused_fp8_grouped.cu`, `moe_fp8_grouped_{sort,blend,gemm}.cu`, `moe_w8a8_grouped_gemm{,_e4m3}.cu`, `moe_topk*.cu`, `moe_permute.cu`, `moe_unpermute_blend.cu`, `moe_router_gemm*.cu`, `moe_w4a16_grouped_gemm.cu` (+ fork in `qwen3.6-35b-a3b/`). **Vision:** `qwen3.6-35b-a3b/vision_encoder.cu` | gb10; hopper and b200 (3.6-35B-A3B, Qwen3-Next); strix and strix-hip (3.6-35B-A3B) | **Measured (Qwen3.6-35B-A3B-FP8, gb10):** decode C=1 median 55 % (grouped expert GEMVs 85–97 %, router top-k 0 %, latency-bound); decode C=16 median 73 %; prefill 4k/32k median 33 %/25 % (pre-#39; W8A8 grouped expert GEMM 18–32 %). **Campaign #34:** canonical row tiers, grouped FP8 expert decode; C=1..16 at 1.52/1.29/1.19/1.06/0.94× vLLM+MTP. **Optimised overnight 2026-09-28 (#39):** 32k cold 14.42 s → 6.5–7.2 s (vLLM 9.23 s); native E4M3 grouped expert GEMM 2.8–4.0× in its microbenchmark (bit-exact k16 form shipped). Hopper: #25 (MoE prefill stream overlap at C=1) | C=16 decode is 0.94× vLLM at the default `fp8` expert tier; only the `nvfp4-gate-up` and `nvfp4` tiers pass it. Energy: 8–50 % more J/token than vLLM+MTP at C=2..16 (PR #45 data). NVFP4 MoE checkpoints run `moe_w4a16_grouped_gemm`, which is not measured; the 2026-08-21 survey put the remaining gap to a W4A4 grouped GEMM at precision (W4A4 vs W4A16), not pipelining. `qwen3_6_moe` targets pin `hidden_size` |
| **Qwen3.8-Flash-Next** (`qwen4_exp`) | `Inferact/Qwen3.8-Flash-Next-NVFP4`, `Qwen/Qwen3.8-Flash-Next` | GDN, conv, MoE and vision as above. **Sparse attention (QSA):** `qwen3.8-flash-next/qsa_indexer.cu`. **Hyper-connections:** `qwen3.8-flash-next/hyper_connection.cu`. **N-gram / PLE:** `qwen3.8-flash-next/ple.cu`. `qwen3.8-flash-next/gated_norm_sigmoid.cu` | gb10 | QSA, mHC and PLE kernels not yet measured; shared GDN/MoE measured on Qwen | gb10 only; target pins `hidden_size` 2560 |
| **Qwen3-VL MoE** (`qwen3_vl_moe`) | `ig1/Qwen3-VL-30B-A3B-Instruct-NVFP4` | Full attention and MoE from the shared stack. **Vision:** `qwen3-vl-30b-a3b/vision_encoder.cu`, `rms_norm.cu` fork | gb10 | not yet measured (the vision tower has no measured row) | dense `qwen3_vl` is not accepted (Table B) |
| **Gemma 4** (`gemma4`) | `nvidia/Gemma-4-31B-IT-NVFP4`, `bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4A16` | `gemma-4-{26b-a4b,31b}/`: `attn_prefill_512{,tc}.cu`, `paged_decode_attn{,_fp8}_512.cu` (head_dim 512 global layers), `gelu.cu`, `logit_softcap.cu`, `embed_scale.cu`, `rms_norm.cu`; 26B-A4B adds `moe_w4a16_grouped_gemm.cu`, `moe_shared_expert_fused*.cu` | gb10 | not yet measured | only hidden 2816 and 5376 are declared, so E2B/E4B/12B are refused; no audio tower |
| **Nemotron-H** (`nemotron_h`, `nemotron_h_puzzle`) | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4`, `…-Super-120B-A12B-NVFP4`, `…-Labs-3-Puzzle-75B-A9B-NVFP4` | **Mamba2:** `common/mamba2_ssd_chunk.cu`, `common/mamba2_ssm_decode.cu`. **Conv:** `common/causal_conv1d.cu`. **MoE:** `common/moe_expert_relu2_down_shared.cu`, `common/nemotron_moe_prefill.cu`, `common/relu_squared.cu`, `common/moe_topk_sigmoid.cu`; Puzzle: `nemotron-labs-3-puzzle-75b-a9b/{w4a4_gemm,moe_w4a4_grouped,moe_w4a16_grouped_gemm}.cu` | gb10; hopper and b200 (Nano, Super) | not yet measured (Mamba2, ReLU² experts and the Puzzle W4A4 kernels have no floor row) | `nemotron_h` targets pin `hidden_size`, so Nano-9B-v2 and Ultra-550B are refused (Table B) |
| **DeepSeek-V4** (`deepseek_v4`, `deepseek_v41`) | `nvidia/DeepSeek-V4-Flash-NVFP4` (two-node EP recipe), `RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8`, `deepseek-ai/DeepSeek-V4.1-Flash` (also GGUF, `deepseek41`) | `deepseek-v4-flash/`: **MLA/latent attention** `mla_*.cu`, `grouped_gemm_mla.cu`, `paged_decode_attn_{mla,fp8_mla,512}.cu`, `attn_prefill_512.cu` (sliding window + per-head sinks); **CSA/HCA** `attn_v41.cu`, `csa_compress.cu`, `prefill_attn_compressed.cu`; **mHC** `hc_v41.cu`, `hyper_connection.cu`; **Engram** `engram_v41.cu`; **MoE** `moe_v41.cu`, `kquant_moe.cu`, `moe_w4a16_grouped_gemm.cu`, `common/moe_topk_sqrtsoftplus.cu`, `common/moe_hash_route.cu` | gb10, hopper, b200 | not yet measured | V4-Flash needs two GB10s; V4-Pro (1.6T) has no target sized for it; `deepseek_v41` pins `hidden_size` 5120 |
| **Mistral Small 4** (`mistral`, from `params.json`) | `mistralai/Mistral-Small-4-119B-2603-NVFP4` | `mistral-small-4/`: `mla_absorbed.cu`, `mla_fused_prefill.cu`, `mla_prefill_attn.cu`, `paged_decode_attn_{mla,fp8_mla}.cu`, `rope.cu`; `deepseek-v4-flash/grouped_gemm_mla.cu`; MoE from `common/` | gb10 | not yet measured | the target is a wildcard for every `mistral` `model_type`, so dense Mistral checkpoints reach the MLA loader and fail (Table B) |
| **GLM-5.3-Flash** (`glm5_next`) | `LibertAIDAI/GLM-5.3-Flash-NVFP4` | **KDA:** `common/kda_{chunk,gate,layer_ops,recurrent}.cu`. **DSA sparse MLA:** `common/dsa_indexer.cu`, `glm-5.3-flash/glm5next_dsa_mla_decode.cu`, `glm5next_mla_latent_write.cu`. **mHC:** `common/glm5next_mhc.cu`. **FFN:** `common/glm5next_ffn.cu` (clamped SwiGLU). **Vision:** `glm-5.3-flash/glm_vit.cu` | gb10 (TP2/EP2 across two nodes, `scripts/launch-metrale-glm53-tp2ep2.sh`) | not yet measured | gb10 only; pins `hidden_size` 4096 |
| **Kimi K3** (`kimi_k3`; inner `kimi_linear`) | Kimi K3 (hidden 7168) on b300; `inference-optimization/Kimi-K3-0.40B` (hidden 1024 twin) on gb10 | `kimi-k3/bf16/`: `kda_decode.cu` (KDA + its short conv), `mla_decode.cu` (gated MLA); `kernels/b200/kimi-k3/bf16/dense_f32io.cu`; MXFP4 through `common/mx_block_scale.cuh` (E8M0 scales) | gb10, b200, b300 (bf16, mxfp4, nvfp4 each) | not yet measured | the full model fits only the B300 class; Kimi Linear 48B (`kimi_linear`, hidden 2304) is accepted by the factory but has no target (Table B) |
| **Laguna** (`laguna`) | `poolside/Laguna-S-2.1-NVFP4`, `poolside/Laguna-XS-2.1-NVFP4` | none of its own: sliding/full attention, per-head output gate (`common/residual_add.cu` `sigmoid_gate_mul*`), sigmoid MoE, all shared | gb10 | not yet measured | pins `hidden_size` 2048/3072; the published INT4 checkpoint needs Table C's INT4 row |
| **MiniMax-M2** (`minimax_m2`) | `lukealonso/MiniMax-M2.7-NVFP4` (two-node EP recipe), `MiniMaxAI/MiniMax-M2.7` | `minimax-m2-229b/`: `moe_w4a16_grouped_gemm.cu`, `w4a16_gemm_v{2,3}.cu`, `rms_norm.cu`; `common/moe_topk_sigmoid.cu` | gb10 | not yet measured | needs two GB10s; MiniMax-M3 is a different architecture (Table B) |
| **Step-3.7-Flash** (`step3p7`) | `stepfun-ai/Step-3.7-Flash-NVFP4` | `step3p7-flash/moe_silu_mul.cu` (SwiGLU with one constant clamp; the checkpoint's per-layer limits are dropped by the parser); the rest shared | gb10 | not yet measured | Step-3.5-Flash (`step3p5`, same text model) is not accepted (Table B) |
| **LongCat-Flash** (`longcat_flash_ngram`, `longcat_flash`) | `meituan-longcat/LongCat-Flash-Lite` | MLA from `deepseek-v4-flash/` (`mla_*.cu`, `grouped_gemm_mla.cu`, `paged_decode_attn_{mla,fp8_mla}.cu`); zero-computation experts in `common/moe_topk_softmax_bias.cu` (`moe_zero_expert_add`) | gb10 | not yet measured | the target is a wildcard, so the 560B LongCat-Flash-Chat resolves to it but does not fit one or two GB10s; LongCat-2.0 is not accepted (Table B) |
| **NLLB-200** (`m2m_100`, `nllb`) | `facebook/nllb-200-3.3B` | `common/nllb_encoder.cu`; `kernels/metal/common/nllb_encoder.metal` | gb10, metal | not measured in an NLLB regime | encoder-decoder translation only; outside the decoder families' shared engine |

## Table B: not supported, kernels already exist

**One prerequisite unlocks most of this table.** The engine has no generic loader for a plain dense
full-attention transformer. The one dense full-attention family it serves, Gemma 4 31B, goes through
the Gemma-specific loader (`crates/model-arch/src/weight_loader/gemma4.rs`: sandwich norms, head_dim
512 global layers, K = V); every other GQA attention layer sits inside a hybrid (GDN, KDA, Mamba2) or
MoE model, and `loader_for_config` has no `llama`, `qwen2`, `qwen3` or dense-`mistral` arm. The kernels
are not the obstacle: the GQA attention path (`crates/model-layers/src/layers/qwen3_attention/`)
and the dense FFN path (`crates/model-layers/src/layers/dense_ffn.rs` and its siblings) run in every Qwen3.x dense model.
Call this prerequisite **G1, a dense GQA loader**: config mapping, per-layer tensor names, QKV bias and
QK-norm options, the RoPE variants through the `inv_freq` table the `rope_forward_yarn*` kernels
already take (llama3, LongRoPE, NTK-alpha and linear scaling are all frequency tables), chat templates,
and `MODEL.toml` wildcards. G1 is **M**; the rows marked "G1 +" are **S** after it. G1 also repairs
the GGUF `llama`/`qwen2` mapping described under [Methodology](#supported).

Priority: **P1** very prominent and fits one GB10; **P2** very prominent but needs two GB10s or a
datacenter tree, or prominent and close; **P3** moderate prominence; **P4** niche or superseded.

| # | Architecture (`model_type`) | Example checkpoints | Why it is close (component → existing kernels) | What is missing | Effort | Fits | Priority |
|---|---|---|---|---|---|---|---|
| B1 | **Llama 3.1/3.2/3.3** (`llama`); Llama-shaped: Falcon3, MiniCPM5 (`llama`), Granite 4.1/4.2 (`granite`) | `meta-llama/Llama-3.3-70B-Instruct`, `Llama-3.1-8B-Instruct`, `ibm-granite/granite-4.2-8b` | GQA hd128 → shared attention; SwiGLU → dense FFN; RMSNorm; llama3 RoPE scaling → `rope_forward_yarn*` with an `inv_freq` table; FP8/NVFP4 projections | G1 + Granite µP multipliers (embedding, residual, attention and logits scaling; `embed_scale.cu` and `bf16_scaled_add` in `residual_add.cu`) | G1 + S | 8B–70B on one GB10 (NVFP4) | **P1** |
| B2 | **Qwen3 MoE** (`qwen3_moe`) | `Qwen/Qwen3-30B-A3B`, `Qwen/Qwen3-235B-A22B`, `Qwen/Qwen3-Coder-30B-A3B-Instruct`, `Qwen3-Coder-480B-A35B`; Intern-S1's text model | This is the Qwen3-VL text stack: GQA hd128 + QK-norm, 128 experts top-8 softmax, no shared expert. `Qwen3VLWeightLoader` and the `qwen3-vl-30b-a3b` target already run it | a factory arm and a flat-config dispatch (no `text_config`), plain RoPE instead of MRoPE, `MODEL.toml` `[[model_types]]`. The GGUF `qwen3moe` mapping (`gguf.rs:59`) currently sends these checkpoints to the GDN MoE loader instead: mapped, not demonstrated | **S** (no G1 needed) | 30B-A3B on one GB10; 235B on two; 480B B200 | **P1** |
| B3 | **Qwen3 dense** (`qwen3`) and **Qwen3-VL dense** (`qwen3_vl`) | `Qwen/Qwen3-8B`, `Qwen/Qwen3-32B`, `Qwen/Qwen3-VL-8B-Instruct` | GQA hd128 + per-head QK-norm (`fused_k_norm_rope_cache_write_*`); dense FFN; Qwen-ViT (`qwen3-vl-30b-a3b/vision_encoder.cu`) | G1, or the Qwen3.5 dense loader with an all-full-attention layer map: a `config.json` with `model_type` `qwen3` is parsed by the default dispatch arm and then rejected by the factory. The GGUF `qwen3` mapping already reaches the Qwen3.5 dense loader, and with no `layer_types` every layer defaults to full attention (`full_attention_interval` defaults to 1, `crates/config/src/model_config.rs:73`), so a plain Qwen3 GGUF may load today: mapped, not demonstrated | G1 + S | yes | **P1** |
| B4 | **Qwen2.5** (`qwen2`), **Seed-OSS** (`seed_oss`), Baichuan-M2 (`qwen2`) | `Qwen/Qwen2.5-7B-Instruct`, `Qwen2.5-72B`, `ByteDance-Seed/Seed-OSS-36B-Instruct` | GQA hd128 with QKV bias: bias kernels exist (`vision_add_bias`, `nllb_bias_bf16`; unlaunched `bias_add_bf16_f32`); SwiGLU; RoPE | G1 + QKV bias wiring; official AWQ/GPTQ checkpoints need Table C's INT4 row | G1 + S | up to 72B on one GB10 | **P1** |
| B5 | **gpt-oss** (`gpt_oss`) | `openai/gpt-oss-20b`, `openai/gpt-oss-120b` | GQA with alternating 128-token sliding window (the sliding-window argument of the paged kernels); YaRN; softmax top-4 of 32/128 experts; **MXFP4 experts** → E8M0 block scales in `common/mx_block_scale.cuh`, used by `moe_w4a16_grouped_gemm.cu` for the Kimi K3 MXFP4 target; per-head sinks → the sink term in `deepseek-v4-flash/attn_prefill_512.cu` and the MLA decode kernels | Kernel deltas: sinks in the head_dim-64 GQA decode and prefill kernels; a head_dim-64 prefill instantiation (prefill is compiled for 128/256/512); the clamped `swiglu-oai` activation (α 1.702, limit 7; `glm5next_swiglu_clamp` has no α); QKV and O bias (softmax over the selected top-4 equals the existing softmax-then-renormalise routing, so the router needs no change) | **M** | 20B and 120B (~63 GB) on one GB10 | **P1** |
| B6 | **GLM-4.5 / 4.6 / 4.7, GLM-4.5-Air** (`glm4_moe`) | `zai-org/GLM-4.5-Air`, `GLM-4.6`, `GLM-4.7` | GQA hd128 with QKV bias and QK-norm; partial RoPE 0.5 (`rotary_dim`); 160/128 experts top-8, sigmoid + bias (`moe_topk_sigmoid.cu`), 1 shared, first 3 dense; MTP (`crates/model-layers/src/layers/mtp_head/`); FP8 128×128 and NVFP4 projections | loader, config, `MODEL.toml`, template; MTP head wiring for this shape | S–M | Air on one GB10; 355B on two (tight) | **P1** |
| B7 | **Gemma 3** (`gemma3`, `gemma3_text`) | `google/gemma-3-27b-it`, `gemma-3-12b-it` | The Gemma 4 kernels cover 5:1 local/global with a 1024 window, QK-norm, sandwich norm, tanh-GeGLU (`gelu.cu`), embedding scale, final softcap; the dual RoPE (local θ 10k, global θ 1M with linear ×8 scaling) is an `inv_freq` table | factory arm and config mapping; head_dim 128 (27B) through the 128 kernels; SigLIP tower on the generic ViT kernels (**M**). GGUF `gemma3` is already mapped to `gemma4` (`gguf.rs:60`), and a 27B GGUF (hidden 5376) would resolve to the `gemma-4-31b` target: mapped, not demonstrated. The official QAT `q4_0` GGUF needs Table C's GGUF row | S (text) | yes | **P1** |
| B8 | **Mistral dense and Mistral 3** (`mistral` dense, `mistral3`, `ministral3`) | `mistralai/Mistral-Small-3.2-24B-Instruct-2506`, `Magistral-Small-2509`, `Devstral-Small-2507`, `Ministral-3-8B-Instruct-2512`, `Mistral-Medium-3.5-128B` | GQA hd128; SwiGLU; YaRN; sliding window (Mistral 7B v0.1); FP8 projections | G1; dispatch `mistral` on the presence of MLA fields, because today every `mistral` config reaches the MLA loader (`factory.rs:80`, wildcard target); FP8 with static per-tensor scales (a broadcast of the per-channel path); Pixtral tower on the generic ViT kernels (**M**) | G1 + S | up to 128B on one GB10 (NVFP4) | **P2** |
| B9 | **DeepSeek V3 / V3.1 / R1** (`deepseek_v3`), **Kimi K2 / K2.5 / K2.6** (`kimi_k2`, `kimi_k25`), **Mistral Large 3** (MLA, `params.json`) | `deepseek-ai/DeepSeek-V3.1`, `DeepSeek-R1`, `moonshotai/Kimi-K2-Instruct`, `Kimi-K2.6`, `mistralai/Mistral-Large-3-675B-Instruct-2512` | DeepSeek-V3-shaped MLA → the MLA kernels Mistral Small 4 and LongCat launch; sigmoid + bias router (`moe_topk_sigmoid.cu`); shared expert; first-k dense; MTP; FP8 128×128 (`fp8_gemm_t_blockscaled.cu`); Mistral Large 3's `params.json` goes through the Mistral Small 4 parser | Kernel delta: group-limited top-k (`n_group` 8, `topk_group` 4) for V3/R1, which no router kernel implements (`crates/model-arch/examples/common/glm5next_moe_run.rs:39` refuses it); K2 uses one group, so no delta. Loaders (`deepseek_v3`, `kimi_k2`); first-3-dense layers in the Mistral path; b200/b300 targets. K2.5/K2.6 add a MoonViT tower; their official INT4 weights need Table C | **M** | datacenter trees only (671B–1T) | **P2** |
| B10 | **DeepSeek V3.2** (`deepseek_v32`) | `deepseek-ai/DeepSeek-V3.2` | B9 plus DSA: the lightning indexer and top-2048 selection → `common/dsa_indexer.cu` (GLM-5.3-Flash; also compiled for b300) | as B9, plus wiring DSA to RoPE-carrying MLA | **M** | datacenter trees only | **P2** |
| B11 | **GLM-5 / 5.1 / 5.2 / 5.3** (`glm_moe_dsa`) | `zai-org/GLM-5`, `GLM-5.3` | MLA + DSA (GLM-5.3-Flash's `dsa_indexer.cu` and DSA-MLA decode); sigmoid MoE, 1 shared, first 3 dense; MTP | MLA with RoPE at qk_nope 192 / v 256 (the existing MLA kernels are shaped for 128+64/128 and 64+64): an instantiation delta; IndexShare (one indexer per 4 layers) is glue; b200/b300 targets | **M** | datacenter trees only (744B) | **P2** |
| B12 | **Llama 4 Scout / Maverick** (`llama4`) | `meta-llama/Llama-4-Scout-17B-16E-Instruct`, `Llama-4-Maverick-17B-128E-Instruct` | GQA hd128; NoPE every 4th layer; L2 QK-norm (`l2_norm_bf16`); sigmoid top-1 + shared expert; interleaved MoE/dense layers | Kernel deltas: chunked-local attention (8192-token chunks; the kernels have sliding windows but no chunk-aligned mask) and attention temperature tuning (a per-position query scale). Vision tower on the generic ViT kernels (**M**) | **M** | Scout on one GB10 (NVFP4 ~60 GB); Maverick datacenter | **P2** |
| B13 | **Gemma 4 E2B / E4B / 12B** (`gemma4`, other hidden sizes) | `google/gemma-4-E4B-it` | Same kernels as the served Gemma 4; per-layer embeddings → the PLE family (`qwen3.8-flash-next/ple.cu`); KV sharing across layers is cache bookkeeping | `MODEL.toml` sizes; PLE and KV-sharing wiring; the audio tower is Table C | S–M | yes | **P2** |
| B14 | **MiMo-V2-Flash / V2.5 / V2.6** (`mimo_v2_flash`, `mimo_v2`) | `XiaomiMiMo/MiMo-V2-Flash`, `MiMo-V2.6-Flash-RL` | SWA-128 + full attention; sinks (DeepSeek-V4 kernels); sigmoid noaux, MTP-3; MXFP4 experts (V2.6) through `mx_block_scale.cuh` | Kernel deltas: sinks and asymmetric qk 192 / v 128 head dims in the GQA kernels (the MLA prefill kernels already handle qk ≠ v); per-layer-type KV head counts. Audio tower (V2.5+) is Table C | **M** | two GB10s | **P2** |
| B15 | **Step-3.5-Flash** (`step3p5`) | `stepfun-ai/Step-3.5-Flash` | The Step-3.7-Flash text model we serve, without the vision tower (3:1 SWA-512, head-wise gate, sigmoid MoE + shared, per-layer SwiGLU clamps) | a factory arm to `Step3p7WeightLoader` and a `MODEL.toml` entry (hidden 4096 is declared for `step3p7`); MTP-3 | **S** | two GB10s (196B) | **P2** |
| B16 | **Granite 4.0-H** (`granitemoehybrid`) | `ibm-granite/granite-4.0-h-small`, `-h-tiny` | Mamba2 (`mamba2_ssd_chunk.cu`, `mamba2_ssm_decode.cu`: runtime `n_groups` and state size) + conv; NoPE GQA; softmax top-10 MoE + shared expert | loader, µP multipliers, `MODEL.toml`; Mamba2 chunk 256 (Nemotron uses its own chunk) | **M** | yes (32B-A9B) | **P2** |
| B17 | **Kimi Linear** (`kimi_linear`) | `moonshotai/Kimi-Linear-48B-A3B-Instruct` | Accepted by the factory (`factory.rs:61`) and parsed as `kimi_k3`; KDA + NoPE MLA + sigmoid MoE → the Kimi K3 kernels | no `kimi_k3` target declares hidden 2304, so serve refuses it; check the K3 path with plain (non-latent) experts and SwiGLU instead of SiTU-GLU | **S** | yes | **P3** |
| B18 | **Nemotron-H, other sizes** (`nemotron_h`) | `nvidia/NVIDIA-Nemotron-Nano-9B-v2`, `NVIDIA-Nemotron-3.5-Lightning-30B-A3B`, `NVIDIA-Nemotron-3-Ultra-550B-A55B` | Accepted by the factory; Mamba2 + ReLU² + sigmoid MoE are the served Nemotron kernels | the targets pin `hidden_size`, so Nano-9B-v2 is refused; its dense ReLU² MLP layers need the Nemotron loader's dense layer type; Ultra needs a datacenter target; Lightning's MTP layer | **S** | Nano sizes yes; Ultra no | **P3** |
| B19 | **Hunyuan-A13B, Hunyuan dense, Hy3** (`hunyuan_v1_moe`, `hunyuan_v1_dense`, `hy_v3`); Hy4-preview (`hy_v4`) | `tencent/Hunyuan-A13B-Instruct`, `tencent/Hy3`, `tencent/Hy4-preview` | GQA + QK-norm; MoE + shared; NTK-alpha RoPE as an `inv_freq` table; Hy3 sigmoid + bias, MTP. Hy4: gated MLA + DSA (GLM-5.3), sinks and identity hyper-connections (DeepSeek-V4), clamped SwiGLU (`glm5next_ffn.cu`) | loaders; Hunyuan-Large's cross-layer KV sharing is cache bookkeeping; Hy4 is **M** and datacenter-only | S–M | A13B on one GB10; Hy3 on two | **P3** |
| B20 | **ERNIE 4.5** (`ernie4_5_moe`, `ernie4_5`) | `baidu/ERNIE-4.5-21B-A3B-PT`, `ERNIE-4.5-300B-A47B-PT` | GQA; softmax + bias-correction routing (`moe_topk_softmax_bias.cu`); 2 shared experts (merged into one at load); dense first layer; MTP | loader and config; the VL variant's modality-split expert pools and 3D RoPE are **M** more | **S** | 21B on one GB10; 300B on two | **P3** |
| B21 | **Cohere Command R / A / A+** (`cohere`, `cohere2`, `cohere2_moe`) | `CohereLabs/c4ai-command-a-03-2025`, `command-a-plus-05-2026` | 3:1 SWA with NoPE global layers; interleaved RoPE (`rope_forward_yarn_interleaved`); LayerNorm → `vision_layer_norm`, `glm_vit_layernorm`, `nllb_layernorm` families; A+: sigmoid MoE + shared, official W4A4 (W4A4 GEMMs exist) | parallel attention+FFN block (glue); LayerNorm on the text path; `logit_scale` | **M** | Command A (111B) on one GB10; A+ on two | **P3** |
| B22 | **OLMo 2 / OLMo 3** (`olmo2`, `olmo3`); **OLMo-Hybrid** (`olmo_hybrid`) | `allenai/Olmo-3.1-32B-Instruct`, `allenai/Olmo-Hybrid-7B` | MHA/GQA; QK-norm over the whole projection (RMSNorm kernels); 3:1 SWA-4096; YaRN. Hybrid: GDN 3:1 + conv | G1 + post-norm ordering. Hybrid: `allow_neg_eigval` (β in [0, 2]) is a gate-kernel delta | G1 + S | yes | **P3** |
| B23 | **Phi-4 family** (`phi3`) | `microsoft/phi-4`, `Phi-4-mini-instruct`, `Phi-4-reasoning` | GQA; SwiGLU; LongRoPE as short/long `inv_freq` tables; partial rotary 0.75 (`rotary_dim`) | G1 + fused `qkv_proj`/`gate_up_proj` split at load. Phi-4-mini-flash is Table C | G1 + S | yes | **P3** |
| B24 | **Ling 2.0 / Ling 3.0** (`bailing_moe`, `bailing_moe_v3`) | `inclusionAI/Ling-flash-2.0`, `Ling-3.0-flash` | 2.0: QK-norm, partial RoPE 0.5, sigmoid MoE + shared. 3.0: MLA + KDA + short conv (Kimi K3 / GLM-5.3), head-wise gate, clamped SwiGLU | 2.0: group-limited top-k (as B9). 3.0: loader and layer map | S–M | flash sizes on one or two GB10s | **P3** |
| B25 | **LFM2 / LFM2-MoE** (`lfm2`, `lfm2_moe`) | `LiquidAI/LFM2.5-2.6B`, `LFM2.5-8B-A1B` | gated short convolution, width 3 → `causal_conv1d.cu` (runtime `d_conv` up to 8); GQA; sigmoid + bias MoE | loader and the conv block's gating | S–M | yes | **P3** |
| B26 | **Arcee Trinity** (`afmoe`) | `arcee-ai/Trinity-Mini`, `Trinity-Large-Thinking` | 3:1 SWA with NoPE global; sigmoid output gate (`sigmoid_gate_mul*`); QK-norm; sandwich norm; sigmoid MoE + shared; official NVFP4 and FP8-block | loader, µP scaling | S–M | Mini on one GB10; Large on two or datacenter | **P3** |
| B27 | **MiniMax-M3** (`minimax_m3_vl`) | `MiniMaxAI/MiniMax-M3` | GQA + per-head QK-norm; block-sparse attention (128-token blocks, top-16, 4 index heads) → the nearest family is QSA (`qwen3.8-flash-next/qsa_indexer.cu`: block pooling and an MQA indexer); sigmoid MoE + shared; MTP | Kernel deltas: the MSA block-selection rule; `swiglu-oai` (as B5); MXFP8 activations; vision tower | **M** | datacenter (428B) | **P3** |
| B28 | **LongCat-2.0** | `meituan-longcat/LongCat-2.0` | MLA + DSA indexer; zero-computation experts (`moe_zero_expert_add`); MTP-3 | 768 + 128 experts exceed the router kernels' `MAX_EXPERTS` 512 (`moe_topk_softmax_bias.cu:27`); loader | **M** | datacenter | **P3** |
| B29 | **EXAONE 4.0 / K-EXAONE** (`exaone4`, `exaone_moe`) | `LGAI-EXAONE/EXAONE-4.0-32B`, `K-EXAONE-2.0-750B-A37B` | 3:1 SWA with NoPE global; QK-norm; per-layer windows; sigmoid MoE + shared; clamped SwiGLU; MTP | G1 + post-norm; MoE variant is **M** | G1 + S | 32B yes; 750B datacenter | **P4** |
| B30 | **SmolLM3** (`smollm3`), **Apertus** (`apertus`) | `HuggingFaceTB/SmolLM3-3B`, `swiss-ai/Apertus-8B-Instruct-2509` | GQA; NoPE every 4th layer; QK-norm; llama3 RoPE | G1; Apertus: the xIELU activation (one elementwise kernel) in a non-gated MLP | G1 + S | yes | **P4** |
| B31 | **Falcon-H1** (`falcon_h1`) | `tiiuae/Falcon-H1-34B-Instruct` | Mamba2 (state 256, 2 groups) + gated RMSNorm inside the mixer; GQA; SwiGLU | attention and Mamba2 in parallel inside one layer (glue); heavy µP multipliers | **M** | yes | **P4** |
| B32 | **Mixtral** (`mixtral`) | `mistralai/Mixtral-8x7B-Instruct-v0.1`, `Mixtral-8x22B` | GQA; softmax top-2 of 8, no shared expert (`moe_topk.cu`) | G1 + MoE layer map | G1 + S | 8x7B one GB10; 8x22B two | **P4** |
| B33 | **Gemma 2** (`gemma2`) | `google/gemma-2-27b-it` | 1:1 sliding window 4096; sandwich norm; GeGLU; final softcap (`logit_softcap.cu`) | Kernel delta: tanh softcap of attention scores (50), which no attention kernel applies; `query_pre_attn_scalar` | **S** | yes | **P4** |
| B34 | **2026 MLA/linear hybrids**: Xing4.0 (`xing4_0`), AliceAI (`alice_ai`), GigaChat3.5, dots.llm1 (`dots1`), Step3 MFA (`step3_text`) | `XingChen-AGI/Xing4.0-29B-A4B`, `yandex/AliceAI-Foundation-80B-A3B-Base`, `ai-sage/GigaChat3.5-432B-A28B`, `rednote-hilab/dots.llm1.inst`, `stepfun-ai/step3` | Recombinations of families we have: MLA + manifold hyper-connections (mHC), KDA + GQA + attention residuals (Kimi K3), MLA + GDN + gated attention, MHA + QK-norm + sigmoid MoE, and MFA = low-rank shared query with one KV head (MQA) | a loader and layer map each | M each | mixed | **P4** |

## Table C: not supported, kernels missing

| # | Architecture (`model_type`) | Example checkpoints | Components without a kernel family | Kernel work needed | Effort | Priority |
|---|---|---|---|---|---|---|
| C1 | **INT4 weight-only checkpoints** (AWQ, GPTQ, compressed-tensors `pack-quantized`, group 32/128): Kimi K2-Thinking, K2.5 and K2.6 (official), Qwen2.5/Qwen3 AWQ and GPTQ, Hunyuan INT4, Laguna INT4, Arcee W4A16, Gemma 4 W4A16 | `moonshotai/Kimi-K2-Thinking`, `Qwen/Qwen2.5-72B-Instruct-AWQ` | A W4A16 INT4 GEMV/GEMM with per-group scales and zero points. The integer family has INT8 and GGUF K-quants only | an INT4 decode family (GEMV tiers, skinny-M, grouped-MoE GEMV) and a prefill GEMM, reusing the NVFP4 W4A16 structure (`w4a16_gemv*.cu`, `moe_w4a16_grouped_gemm.cu`) with an INT4 unpack and zero-point epilogue; or a load-time requantisation to NVFP4 as a disclosed precision change | **M** | **P1** (format; unlocks official Kimi K2.x weights and many community quants) |
| C2 | **GGUF quant types beyond Q2_0, Q2_K, Q3_K, Q4_K, Q6_K, Q8_0**: Q4_0 (the official Gemma QAT GGUFs), Q5_K (`Q5_K_M`), IQ-quants | `google/gemma-3-27b-it-qat-q4_0-gguf` | dequantisers and MMVQ for these block formats (`common/dequant_gguf_bf16.cu` has Q2_0, Q2_K, Q3_K, Q4_K, Q6_K, Q8_0) | per-format dequant kernels first, then native MMVQ where decode matters | S–M | **P2** |
| C3 | **Jamba 1.5 / 1.6 / 1.7, Jamba2** (`jamba`); Falcon-Mamba (`falcon_mamba`) | `ai21labs/AI21-Jamba2-Mini`, `AI21-Jamba2-3B` | **Mamba-1** selective scan (per-channel diagonal A, `dt_rank` projection, state 16). Mamba2's SSD kernels do not compute it; the only Mamba-1 scan is `kernels/metal/common/selective_scan_decode.metal`, which no engine path launches | a CUDA selective-scan family: chunked prefill scan, decode step, state cache and speculative-verify rollback | **L** | **P4** |
| C4 | **MiniMax-M1** (`minimax_m1`, `minimax`) | `MiniMaxAI/MiniMax-M1-80k` | **Lightning attention**: linear attention with a per-head scalar decay and no delta rule, 7 of every 8 layers. The GDN and KDA kernels are delta-rule recurrences | a lightning-attention family (intra-block quadratic + inter-block recurrent prefill, recurrent decode) | **L** | **P4** (superseded by MiniMax-M2, which we serve) |
| C5 | **Ring-linear-2.0, Ling-2.5-1T** (`bailing_moe_linear`, `bailing_hybrid`) | `inclusionAI/Ring-flash-linear-2.0`, `Ling-2.5-1T` | lightning attention, as C4 | the C4 family; the rest (MLA, sigmoid MoE) exists | **L** | **P4** |
| C6 | **RWKV-7** (`rwkv7`) | `RWKV/RWKV7-G1j-13.3B-20260831` | the RWKV-7 generalized delta rule (vector decay, in-context learning rate `a`, low-rank decay/value/gate projections, token shift), LayerNorm with bias. The nearest families, KDA and GDN, use a different state update | an RWKV-7 recurrence family (chunked prefill + decode) and token-shift kernels | **L** | **P4** |
| C7 | **Phi-4-mini-flash-reasoning** (`phi4flash`) | `microsoft/Phi-4-mini-flash-reasoning` | SambaY: Mamba-1 layers (C3), Gated Memory Units that reuse SSM readouts across the cross-decoder, and Differential Attention (the difference of two softmax maps) | C3's scan plus GMU and differential-attention kernels | **L** | **P4** |
| C8 | **Gemma 3n** (`gemma3n`) | `google/gemma-3n-E4B-it` | AltUp (4 predicted streams), LAuReL low-rank residual, MatFormer nesting, top-k activation sparsity, audio tower | AltUp predict/correct and sparsity kernels; the audio tower (C9). Per-layer embeddings and KV sharing are in reach (B13) | **M–L** | **P3** |
| C9 | **Audio towers**: Qwen3-Omni (`qwen3_omni_moe`: audio encoder, talker, code predictor, code2wav vocoder), Gemma 4 E2B/E4B audio, MiMo-V2.5+ audio | `Qwen/Qwen3-Omni-30B-A3B-Instruct` | no audio encoder, talker or vocoder kernels (convolutional front-end, audio transformer encoder, waveform decoder) | an audio component: front-end convolutions and mel features, then the encoder on the shared attention and GEMM kernels; the talker and vocoder for speech output | **L** | **P3** |
| C10 | **Diffusion LMs**: DiffusionGemma | `google/diffusiongemma-26B-A4B-it` | bidirectional block attention and an iterative denoising sampler. `recipes/diffusion-gemma/` is a vLLM reference recipe, not an engine path; the engine's block-diffusion kernel (`common/dflash2.cu`) is a speculative drafter, not a decoder | a non-causal block-attention mode and a denoising scheduler | **L** | **P4** |
| C11 | **ZAYA1** (`zaya`) | `Zyphra/ZAYA1-8B` | compressed convolutional attention | a new attention family | **L** | **P4** |

## Summary roadmap

Ranked by prominence × closeness. Closeness is the Table B effort; prominence is download counts, how
often the lab ships, and how many other engines list the architecture.

| Rank | Work item | Unlocks | Why | Effort |
|---|---|---|---|---|
| 1 | **G1, the dense GQA loader**, then Llama 3.x, Qwen3 dense, Qwen2.5 and dense Mistral | B1, B3, B4, B8; then B22, B23, B29, B30, B32 | The most-used open architectures, and the only prominent class with no load path at all. Their kernels (shared GQA attention, dense FFN, BF16/FP8/NVFP4 projections) run in every Qwen3.x dense model today, with decode GEMVs measured at 89–100 % of floor. G1 also fixes the GGUF `llama`/`qwen2` route into the MLA loader | M, then S per architecture |
| 2 | **Qwen3 MoE** (`qwen3_moe`) through the Qwen3-VL text stack | B2 | Qwen3-30B-A3B and Qwen3-Coder-30B-A3B fit one GB10; the loader and target already exist for the same text stack, and the MoE kernels are the ones campaigns #34 and #39 optimised | S |
| 3 | **gpt-oss** | B5 (and the `swiglu-oai` delta that B27 also needs) | Very widely deployed; both sizes fit one GB10; MXFP4 expert decode already exists for Kimi K3. Kernel deltas are sinks and a prefill instantiation at head_dim 64, plus one activation | M |
| 4 | **GLM-4.5 / 4.6 / 4.7 and GLM-4.5-Air** | B6 (and B9's router pieces) | Every component is an existing kernel; Air fits one GB10; GLM-4.x checkpoints ship in FP8 and NVFP4 | S–M |
| 5 | **Gemma 3**, then the other Gemma 4 sizes | B7, B13 | The Gemma 4 kernels cover Gemma 3's attention pattern, norms and activations; the GGUF path already maps `gemma3` to them. Text is S; the SigLIP tower is M | S (text) |

Next in line: INT4 weight-only decode (C1), which unlocks the official Kimi K2.x weights and many
community checkpoints; DeepSeek V3.x and Kimi K2 (B9, B10), which are very prominent but need a
datacenter tree; Llama 4 Scout (B12); Granite 4.0-H (B16).

Measurement debt is as large as the coverage debt: 12 of the 14 decoder families have no measured
kernel of their own, and the prefill rows of the other two predate #39. Re-measuring the Qwen prefill
regimes on current `main`, then adding one decode regime for Nemotron-H (Mamba2), DeepSeek-V4 (MLA,
CSA) and GLM-5.3-Flash (KDA, DSA), would put a number on every family-specific component family.

## Sources

Configs are `https://huggingface.co/<id>/blob/main/config.json` unless another file is named; gated
repositories and repositories without a `config.json` link to the model page. Read 2026-09-28.

**Supported families.**
Qwen3.x dense: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json) ·
Qwen3.x MoE: [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/config.json),
[Qwen/Qwen3-Next-80B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/blob/main/config.json) ·
Qwen3.8-Flash-Next: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json) ·
Qwen3-VL: [Qwen/Qwen3-VL-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct/blob/main/config.json) ·
Gemma 4: [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it/blob/main/config.json) ·
Nemotron-H: [nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/blob/main/config.json) ·
DeepSeek-V4: [deepseek-ai/DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/config.json),
[deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) ·
Mistral Small 4: [mistralai/Mistral-Small-4-119B-2603](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603) ·
GLM-5.3-Flash: [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/config.json) ·
Kimi K3: [moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) ·
Laguna: [poolside/Laguna-S-2.1](https://huggingface.co/poolside/Laguna-S-2.1/blob/main/config.json) ·
MiniMax-M2: [MiniMaxAI/MiniMax-M2.7](https://huggingface.co/MiniMaxAI/MiniMax-M2.7/blob/main/config.json) ·
Step-3.7-Flash: [stepfun-ai/Step-3.7-Flash](https://huggingface.co/stepfun-ai/Step-3.7-Flash/blob/main/config.json) ·
LongCat: [meituan-longcat/LongCat-Flash-Lite](https://huggingface.co/meituan-longcat/LongCat-Flash-Lite/blob/main/config.json) ·
NLLB: [facebook/nllb-200-3.3B](https://huggingface.co/facebook/nllb-200-3.3B/blob/main/config.json).

**Table B.**
B1 [meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct),
[ibm-granite/granite-4.2-8b](https://huggingface.co/ibm-granite/granite-4.2-8b/blob/main/config.json) ·
B2 [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json) ·
B3 [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B/blob/main/config.json) ·
B4 [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json),
[ByteDance-Seed/Seed-OSS-36B-Instruct](https://huggingface.co/ByteDance-Seed/Seed-OSS-36B-Instruct/blob/main/config.json) ·
B5 [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b/blob/main/config.json) ·
B6 [zai-org/GLM-4.6](https://huggingface.co/zai-org/GLM-4.6/blob/main/config.json) ·
B7 [google/gemma-3-27b-it](https://huggingface.co/google/gemma-3-27b-it) ·
B8 [mistralai/Mistral-Small-3.2-24B-Instruct-2506](https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506/blob/main/config.json),
[mistralai/Ministral-3-8B-Instruct-2512](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512/blob/main/config.json) ·
B9 [deepseek-ai/DeepSeek-V3.1](https://huggingface.co/deepseek-ai/DeepSeek-V3.1/blob/main/config.json),
[moonshotai/Kimi-K2-Instruct](https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/config.json),
[Mistral-Large-3 `params.json`](https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512/blob/main/params.json) ·
B10 [deepseek-ai/DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/config.json) ·
B11 [zai-org/GLM-5](https://huggingface.co/zai-org/GLM-5/blob/main/config.json) ·
B12 [meta-llama/Llama-4-Scout-17B-16E-Instruct](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct) ·
B13 [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it/blob/main/config.json) ·
B14 [XiaomiMiMo/MiMo-V2-Flash](https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash/blob/main/config.json) ·
B15 [stepfun-ai/Step-3.5-Flash](https://huggingface.co/stepfun-ai/Step-3.5-Flash/blob/main/config.json) ·
B16 [ibm-granite/granite-4.0-h-small](https://huggingface.co/ibm-granite/granite-4.0-h-small/blob/main/config.json) ·
B17 [moonshotai/Kimi-Linear-48B-A3B-Instruct](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/blob/main/config.json) ·
B18 [nvidia/NVIDIA-Nemotron-Nano-9B-v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2/blob/main/config.json) ·
B19 [tencent/Hunyuan-A13B-Instruct](https://huggingface.co/tencent/Hunyuan-A13B-Instruct/blob/main/config.json),
[tencent/Hy3](https://huggingface.co/tencent/Hy3/blob/main/config.json) ·
B20 [baidu/ERNIE-4.5-21B-A3B-PT](https://huggingface.co/baidu/ERNIE-4.5-21B-A3B-PT/blob/main/config.json) ·
B21 [CohereLabs/command-a-plus-05-2026-bf16](https://huggingface.co/CohereLabs/command-a-plus-05-2026-bf16/blob/main/config.json) ·
B22 [allenai/Olmo-3.1-32B-Instruct](https://huggingface.co/allenai/Olmo-3.1-32B-Instruct/blob/main/config.json),
[allenai/Olmo-Hybrid-7B](https://huggingface.co/allenai/Olmo-Hybrid-7B/blob/main/config.json) ·
B23 [microsoft/phi-4](https://huggingface.co/microsoft/phi-4/blob/main/config.json) ·
B24 [inclusionAI/Ling-flash-2.0](https://huggingface.co/inclusionAI/Ling-flash-2.0/blob/main/config.json),
[inclusionAI/Ling-3.0-flash](https://huggingface.co/inclusionAI/Ling-3.0-flash/blob/main/config.json) ·
B25 [LiquidAI/LFM2-8B-A1B](https://huggingface.co/LiquidAI/LFM2-8B-A1B/blob/main/config.json) ·
B26 [arcee-ai/Trinity-Mini](https://huggingface.co/arcee-ai/Trinity-Mini/blob/main/config.json) ·
B27 [MiniMaxAI/MiniMax-M3](https://huggingface.co/MiniMaxAI/MiniMax-M3) ·
B28 [meituan-longcat/LongCat-2.0](https://huggingface.co/meituan-longcat/LongCat-2.0) ·
B29 [LGAI-EXAONE/EXAONE-4.0-32B](https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-32B/blob/main/config.json) ·
B30 [HuggingFaceTB/SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B/blob/main/config.json),
[swiss-ai/Apertus-8B-Instruct-2509](https://huggingface.co/swiss-ai/Apertus-8B-Instruct-2509/blob/main/config.json) ·
B31 [tiiuae/Falcon-H1-34B-Instruct](https://huggingface.co/tiiuae/Falcon-H1-34B-Instruct/blob/main/config.json) ·
B32 [mistralai/Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/blob/main/config.json) ·
B33 [google/gemma-2-27b-it](https://huggingface.co/google/gemma-2-27b-it) ·
B34 [XingChen-AGI/Xing4.0-29B-A4B](https://huggingface.co/XingChen-AGI/Xing4.0-29B-A4B),
[yandex/AliceAI-Foundation-80B-A3B-Base](https://huggingface.co/yandex/AliceAI-Foundation-80B-A3B-Base),
[ai-sage/GigaChat3.5-432B-A28B](https://huggingface.co/ai-sage/GigaChat3.5-432B-A28B),
[rednote-hilab/dots.llm1.inst](https://huggingface.co/rednote-hilab/dots.llm1.inst/blob/main/config.json),
[stepfun-ai/step3](https://huggingface.co/stepfun-ai/step3/blob/main/config.json).

**Table C.**
C1 [moonshotai/Kimi-K2-Thinking](https://huggingface.co/moonshotai/Kimi-K2-Thinking/blob/main/config.json) ·
C2 [google/gemma-3-27b-it-qat-q4_0-gguf](https://huggingface.co/google/gemma-3-27b-it-qat-q4_0-gguf) ·
C3 [ai21labs/AI21-Jamba2-Mini](https://huggingface.co/ai21labs/AI21-Jamba2-Mini/blob/main/config.json) ·
C4 [MiniMaxAI/MiniMax-M1-80k](https://huggingface.co/MiniMaxAI/MiniMax-M1-80k/blob/main/config.json) ·
C5 [inclusionAI/Ring-flash-linear-2.0](https://huggingface.co/inclusionAI/Ring-flash-linear-2.0/blob/main/config.json) ·
C6 [RWKV/RWKV7-G1j-13.3B-20260831](https://huggingface.co/RWKV/RWKV7-G1j-13.3B-20260831) ·
C7 [microsoft/Phi-4-mini-flash-reasoning](https://huggingface.co/microsoft/Phi-4-mini-flash-reasoning/blob/main/config.json) ·
C8 [google/gemma-3n-E4B-it](https://huggingface.co/google/gemma-3n-E4B-it) ·
C9 [Qwen/Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct/blob/main/config.json) ·
C10 [google/diffusiongemma-26B-A4B-it](https://huggingface.co/google/diffusiongemma-26B-A4B-it) ·
C11 [Zyphra/ZAYA1-8B](https://huggingface.co/Zyphra/ZAYA1-8B).

**Prominence.** Hugging Face text-generation listings sorted by trending score and by downloads
(`https://huggingface.co/models?pipeline_tag=text-generation&sort=trending`), the vLLM supported-model
list (`https://docs.vllm.ai/en/latest/models/supported_models.html`), the SGLang model directory and
the llama.cpp architecture list, all read 2026-09-28.
