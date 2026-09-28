<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# Routing audit: the golden plans against today's dispatch code

2026-09-28. Milestone 0 of the auto-fuser. Every kernel group in `kernels/circuits/plans/` comes
from one rule in `kernels/gb10/common/FUSIONS.toml`. This file cites, for every rule, the
dispatch code that makes today's choice, and lists the routing the circuit cannot express yet.

## How the rules were checked

- Four read-only traces of the code at `fc670865`: dense M=1 decode, dense multi-sequence
  decode, MoE decode across all three row regimes, and the MTP verify and draft paths. A fifth
  trace inventoried the levers. Each rule below carries the citations from those traces.
- Spot checks against the source. A citation that named a path or line that does not exist was
  corrected. `crates/circuit/tests/circuits.rs` fails when a cited file is missing or shorter
  than the cited line.
- Kernel names are the (module, function) pairs the engine looks up, with KERNEL.toml
  `[modules]` renames applied. The tests check every rule's kernel against the sources its
  golden target compiles.
- Launch counts agree with the traces. For example, the dense decode is 818 launches: 48 GDN
  layers × 12, plus 16 attention layers × 15, plus the final norm and the head.

## The golden matrix

| Instance (recipe) | Model | Policy |
|---|---|---|
| `qwen3.8/qwen3.8-27b-nvfp4-unsloth` | dense Qwen3.8-27B | by_rows tiers, bf16 KV, bf16 head, f32 GDN state, batched recurrence off, gemv_sw on, tc8 on, split SiLU on, no opt-in levers |
| `qwen3.6/qwen3.6-35b-a3b-fp8-bf16head` | Qwen3.6-35B-A3B MoE | as above, except canonical row tiers |

Each instance has one plan per mode and row count:

- **decode** n1;
- **multi_seq** at each rung of `padded_batch_n`: 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128;
- **verify** K = 2, 3, 4;
- **draft** n1, the serial MTP head.

That is 32 plan files, plus 6 display snapshots (decode n1 of each model at Unicode 80/120 and
ASCII 80). Every policy value cites its source in `kernels/circuits/INSTANCES.toml`.

The dense agentic recipe runs `max_batch_size: 1`, so its multi_seq plans describe what the
engine would run at those widths under this policy, not a width that recipe reaches.

## Rules and their dispatch sites

Abbreviations as in FUSIONS.toml:

- `ml/` = crates/model-layers/src/layers/
- `me/` = crates/model-engine/src/model/trait_impl/
- `mm/` = crates/model-engine/src/model/
- `k/` = kernels/gb10/common/

`crates/circuit/tests/circuits.rs` checks that this table lists every rule with its numerics
class and exact citation.

| Rule | Numerics | Dispatch site(s) |
|---|---|---|
| `embed_row_copy` | reference | me/decode_a2.rs:241-254 (one D2D copy per row, padding rows memset); ml/mtp_head/forward.rs:67-68 |
| `input_norm_residual` | reference | ml/qwen3_ssm/trait_decode.rs:38-49; ml/qwen3_attention/trait_impl/decode_inner.rs:76-87; ml/qwen3_attention/trait_impl/multi_seq/mod.rs:58-69; ml/qwen3_ssm/trait_decode_multi_seq/ssm_batched.rs:170-183; ml/qwen3_ssm/trait_decode_batched.rs:193 |
| `draft_rms_norm` | reference | ml/mtp_head/forward.rs:71-81,88-98,137-147 (the residual copy is a D2D copy at :133) |
| `residual_add_post_norm` | reference | k/rms_norm.cu:379-382 (FP32 sum of squares before the BF16 rounding: not the unfused chain); ml/qwen3_ssm/trait_decode.rs:117-129; ml/qwen3_attention/trait_impl/decode_inner.rs:241-253; ml/qwen3_ssm/trait_decode_multi_seq/ssm_batched.rs:247-261; ml/qwen3_attention/trait_impl/multi_seq/ffn.rs:131-143; ml/qwen3_ssm/trait_decode_batched.rs:284; ml/mtp_head/forward.rs:312-324 |
| `ffn_residual_add` | reference | ml/qwen3_ssm/trait_decode.rs:143-150; ml/qwen3_attention/trait_impl/decode_inner.rs:384-391; ml/qwen3_attention/trait_impl/multi_seq/ffn.rs:147-154; ml/qwen3_ssm/trait_decode_multi_seq.rs:212,235 (after the prefill and km arms); ml/qwen3_ssm/trait_decode_batched.rs:323-334; ml/mtp_head/forward.rs:343 |
| `ffn_residual_add_per_row_gdn_k2k3` | reference | ml/qwen3_ssm/trait_decode_multi_seq.rs:178-196 (the n == 2 | 3 arm adds each row's FFN output in a loop, :192) |
| `final_norm` | reference | me/decode_a3.rs:102-105; mm/impl_a3_norm.rs:23-51; me/verify_c2.rs:341; ml/mtp_head/forward.rs:346-356 |
| `gdn_ba_gates_gemv_per_row` | reference | ml/qwen3_ssm/ssm_forward.rs:177-192; ml/qwen3_ssm/trait_decode_multi_seq/ssm_batched_recurrent/per_seq.rs:65-78 (per-sequence loop when the batched recurrence is off, HARDWARE.toml:118) |
| `gdn_ba_gates_gemm_verify` | reference | ml/qwen3_ssm/trait_decode_batched/gates_norm.rs:37-59 (kill switch METRALE_NO_BATCHED_BA_GATES) |
| `gdn_conv_l2_f32_per_row` | reference | ml/qwen3_ssm/ssm_forward.rs:209-229; ml/qwen3_ssm/trait_decode_multi_seq/ssm_batched_recurrent/per_seq.rs:99-113 |
| `gdn_conv_l2_bf16_verify` | reference | ml/qwen3_ssm/trait_decode_batched_conv_gdn.rs:105-110 (BF16 conv rows, not bitwise equal to decode's FP32 conv), :114-331 |
| `gdn_recurrence_f32_per_row` | reference | ml/qwen3_ssm/ssm_forward.rs:262,326-343; ml/qwen3_ssm/trait_decode_multi_seq/ssm_batched_recurrent/per_seq.rs:195-212 |
| `gdn_recurrence_f32_fused_norm` | differs (gdn_fused_norm) | ml/qwen3_ssm/ssm_forward.rs:280-303 (--gdn-fused-norm, default off: crates/server/src/cli/serve_args.rs:197-206); k/gated_delta_rule.cu:940-942,1051 (the fused kernel clamps the state norm, the unfused one does not) |
| `gdn_recurrence_wy2_verify` | reference | ml/qwen3_ssm/trait_decode_batched_conv_gdn.rs:114-331; ml/qwen3_ssm/trait_decode_batched_conv_gdn/wy_select.rs:156-163 |
| `gdn_recurrence_wy3_verify` | reference | ml/qwen3_ssm/trait_decode_batched_conv_gdn.rs:114-331; ml/qwen3_ssm/trait_decode_batched_conv_gdn/wy_select.rs:156-163 |
| `gdn_recurrence_wy4_verify` | reference | ml/qwen3_ssm/trait_decode_batched_conv_gdn.rs:114-331; ml/qwen3_ssm/trait_decode_batched_conv_gdn/wy_select.rs:156-163 |
| `gdn_fused_conv_norm_k2_verify` | differs (gdn_fused_verify) | ml/qwen3_ssm/trait_decode_batched_conv_gdn.rs:57-67 (METRALE_GDN_FUSED_VERIFY=1; checked at cos >= 0.99999, not bitwise) |
| `gdn_out_norm_f32_per_row` | reference | ml/qwen3_ssm/ssm_forward.rs:350-368; ml/qwen3_ssm/trait_decode_multi_seq/ssm_batched_recurrent/per_seq.rs:222-236 |
| `gdn_out_norm_prefill_verify` | reference | ml/qwen3_ssm/trait_decode_batched/gates_norm.rs:138-157 (kill switch METRALE_NO_BATCHED_GDN_NORM) |
| `deinterleave_qg` | reference | ml/qwen3_attention/decode/attention_forward/q_proj.rs:65-74; ml/qwen3_attention/trait_impl/multi_seq/qkv/batch.rs:266-283; ml/qwen3_attention/trait_impl/multi_seq/qkv_fp8_batch.rs:164-175; ml/mtp_head/forward.rs:171-183 |
| `qk_norm_rows` | reference | ml/qwen3_attention/decode/attention_forward.rs:140-151,174-186; ml/mtp_head/forward.rs:215-236 |
| `qk_norm_strided` | reference | ml/qwen3_attention/trait_impl/multi_seq/qkv.rs:222-271 (bit-identical to rms_norm per row) |
| `rope_mrope_interleaved` | reference | ml/qwen3_attention/decode/attention_forward/rope.rs:70-88 |
| `rope_strided` | reference | ml/qwen3_attention/trait_impl/multi_seq/attn.rs:48-87 (plain RoPE; bit-identical to MRoPE while pos_t == pos_h == pos_w, k/rope_mrope_interleaved.cu:15-17) |
| `rope_plain_draft` | reference | ml/mtp_head/forward.rs:264-277 |
| `kv_write_bf16` | reference | ml/qwen3_attention/decode/write_kv_cache.rs:428-444; ml/qwen3_attention/trait_impl/multi_seq/attn.rs:159-176; ml/mtp_head/forward/attend.rs:36-73 (the draft KV is BF16 under mtp_quantization bf16, ml/mtp_head/new.rs:228-236) |
| `paged_attention_bf16` | reference | ml/qwen3_attention/decode/run_paged_decode/bf16_fp8.rs:125-149; ml/qwen3_attention/decode/splitk_dispatch.rs:241-251 (no BF16 split-K pair on GB10); ml/qwen3_attention/trait_impl/multi_seq/attn.rs:231-306 |
| `sigmoid_gate_mul` | reference | ml/qwen3_attention/decode/attention_forward.rs:421-432; ml/mtp_head/forward.rs:298-306 |
| `sigmoid_gate_mul_batched` | reference | ml/qwen3_attention/trait_impl/multi_seq/attn/o_proj.rs:59-81 |
| `w4a16_gemv_sw_1row` | reference | ml/qwen3_ssm/ssm_forward.rs:86-101,386-416; ml/qwen3_attention/decode/attention_forward_oproj.rs:80-91; ml/dense_ffn_decode.rs:365-376 (gemv_sw on unless METRALE_NO_GEMV_SW=1, ml/ops/model_levers_resolve.rs:53; bit-identical to w4a16_gemv, k/w4a16_gemv_fused.cu:305-307) |
| `w4a16_gemv_qg_1row` | reference | ml/qwen3_attention/decode/attention_forward/q_proj.rs:75-113 (K walk differs from w4a16_gemv, k/w4a16_gemv.cu:1447-1449) |
| `w4a16_gemv_dual_kv_1row` | reference | ml/qwen3_attention/decode/attention_forward_kv.rs:76-93 (always the non-_sw dual) |
| `w4a16_gemv_dual_sw_gate_up_1row` | reference | ml/dense_ffn_decode.rs:314-341 |
| `silu_mul_split_1row` | reference | ml/dense_ffn_decode.rs:350-364 (decode_split_silu = true, kernels/gb10/HARDWARE.toml:164, kill switch METRALE_NO_DECODE_SPLIT_SILU) |
| `silu_input_down_1row` | differs (decode_fused_silu) | ml/dense_ffn_decode.rs:386-412; ml/dense_ffn_fp8_down.rs:28-31 (keeps the SiLU product in FP32 where moe_silu_mul rounds it to BF16) |
| `silu_mul_rows` | reference | ml/dense_ffn_decode_batch.rs:89-121,207-273 (forward_k2/k3/km); ml/mtp_head/moe_forward.rs:50-63 |
| `w4a16_tc8_multi_seq` | reference | ml/qwen3_ssm/trait_decode_multi_seq/ssm_batched_proj.rs:156-195,329-346; ml/qwen3_attention/trait_impl/multi_seq/qkv/batch.rs:254-313; ml/dense_ffn_decode_batch.rs:207-273; ml/ops/quant_dispatch.rs:183-205; ml/ops/gemv_tc.rs:56-74,137-148 (tc8 rounds differently from the CUDA-core tiers, k/w4a16_gemv_tc.cu:26-29) |
| `w4a16_tc8_verify_k4` | reference | ml/qwen3_ssm/trait_decode_batched/qkvz_proj.rs:94; ml/qwen3_ssm/trait_decode_batched/out_proj.rs:91-107; ml/qwen3_attention/trait_impl/multi_seq/qkv/batch.rs:266-313; ml/qwen3_attention/trait_impl/multi_seq/attn/o_proj.rs:326-362; ml/dense_ffn_decode_batch.rs:207-273 |
| `w4a16_tc8_gate_up_km` | reference | ml/dense_ffn_decode_batch.rs:207-273 (try_forward_km: gate and up are two launches); arms: ml/qwen3_attention/trait_impl/multi_seq/ffn.rs:178-208 (4..=ffn_proj_max_rows() = 8), ml/qwen3_ssm/trait_decode_multi_seq.rs:211-230, ml/qwen3_ssm/trait_decode_batched.rs:348-365 |
| `w4a16_batch2_verify` | reference | ml/qwen3_ssm/trait_decode_batched/qkvz_proj.rs:240-251; ml/qwen3_ssm/trait_decode_batched/out_proj.rs:191-201; ml/qwen3_attention/trait_impl/multi_seq/attn/o_proj.rs:337-362; ml/dense_ffn_decode_batch.rs:89-121 (bit-identical per row to w4a16_gemv, k/w4a16_gemv.cu:454) |
| `w4a16_batch2_ms_o_down` | reference | ml/qwen3_attention/trait_impl/multi_seq/attn/o_proj.rs:337-362; ml/dense_ffn_decode_batch.rs:89-121 (forward_k2 down) |
| `w4a16_batch3_verify` | reference | ml/qwen3_ssm/trait_decode_batched/qkvz_proj.rs:214-225; ml/qwen3_ssm/trait_decode_batched/out_proj.rs:180-190; ml/qwen3_attention/trait_impl/multi_seq/attn/o_proj.rs:326-336; ml/dense_ffn_decode_batch.rs:123-184 |
| `w4a16_dual_batch2_gate_up` | reference | ml/dense_ffn_decode_batch.rs:89-121 (forward_k2); arms ml/qwen3_attention/trait_impl/multi_seq/ffn.rs:153-176, ml/qwen3_ssm/trait_decode_multi_seq.rs:178-196, ml/qwen3_ssm/trait_decode_batched.rs:337-347 |
| `w4a16_dual_batch3_gate_up` | reference | ml/dense_ffn_decode_batch.rs:123-184 (forward_k3); arm ml/qwen3_ssm/trait_decode_batched.rs:326-336 |
| `w4a16_qg_batch2` | reference | ml/qwen3_attention/trait_impl/multi_seq/qkv/batch.rs:44-159 (matches w4a16_gemv_qg bit for bit, k/w4a16_gemv.cu:1656-1658) |
| `w4a16_qg_batch3` | reference | ml/qwen3_attention/trait_impl/multi_seq/qkv/batch.rs:44-159 |
| `w4a16_dual_batch2_kv` | reference | ml/qwen3_attention/trait_impl/multi_seq/qkv/batch.rs:165-192 (w4a16_gemv_qg's K walk, not w4a16_gemv_dual's, k/w4a16_gemv.cu:1788; the copies into qkv_buf are D2D) |
| `w4a16_dual_batch3_kv` | reference | ml/qwen3_attention/trait_impl/multi_seq/qkv/batch.rs:165-192 |
| `w4a16_gemm_t_p3_rows` | reference | ml/qwen3_ssm/trait_decode_multi_seq/ssm_batched_proj.rs:156-195; ml/qwen3_attention/trait_impl/multi_seq/qkv.rs:329-428; ml/qwen3_ssm/kernel_select.rs:29-97 (E4M3 activations, kernels/gb10/qwen3.6-27b/nvfp4/w4a16_gemm.cu:344-347) |
| `w4a16_gemm_t_k64_n64_p3_rows` | reference | ml/qwen3_ssm/trait_decode_multi_seq/ssm_batched_proj.rs:329-346; ml/qwen3_attention/trait_impl/multi_seq/attn/o_proj.rs:337-362; ml/mod.rs:140-177 (K = 6144 takes the k64 tile; 40 * ceil(m / 64) <= 64 picks n64) |
| `w4a16_gemm_t_m128_rows` | reference | ml/qwen3_ssm/trait_decode_multi_seq/ssm_batched_proj.rs:156-195 (m >= 65 unless METRALE_NO_SSM_M128); ml/qwen3_attention/trait_impl/multi_seq/qkv.rs:329-428; ml/qwen3_attention/trait_impl/multi_seq/attn/o_proj.rs:337-362 |
| `w4a16_gemm_t_k64_p3_gdn_out_wide` | reference | ml/qwen3_ssm/trait_decode_multi_seq/ssm_batched_proj.rs:329-346; ml/qwen3_ssm/kernel_select.rs:80-97 (m128 refused: 40 CTAs < 48 SMs; same bits as k64_n64_p3, kernels/gb10/qwen3.6-27b/nvfp4/w4a16_gemm.cu:1629-1631) |
| `w4a4_downcast_mx8` | differs (w4a4_downcast) | ml/ops/w4a4_proj.rs:58,225-247,338 (--w4a4-downcast, default off: crates/server/src/cli/serve_args.rs:250-260; an accuracy change, w4a4_proj.rs:6-10) |
| `ffn_mmq16_gate_up_gdn` | reference | ml/qwen3_ssm/trait_decode_multi_seq.rs:196-210 (n >= METRALE_SSM_FFN_PREFILL_MIN_N, default 5, ahead of the km arm: :24-40); ml/dense_ffn_prefill_nvfp4.rs:51-55 |
| `ffn_mmq16_act_down_gdn` | reference | ml/dense_ffn_prefill_nvfp4.rs:272-286,330-372 (the tile path leaves the down scale to metrale_nvfp4_scale_bf16) |
| `ffn_mmq16_gate_up` | reference | ml/qwen3_attention/trait_impl/multi_seq/ffn.rs:233-260 (a dense FFN above the km band); ml/dense_ffn_prefill_nvfp4.rs:51-55 |
| `ffn_mmq16_act_down` | reference | ml/dense_ffn_prefill_nvfp4.rs:272-286,330-372 |
| `ffn_mmq32_gate_up` | reference | ml/dense_ffn_prefill_nvfp4.rs:56-60 |
| `ffn_mmq32_act_down` | reference | ml/dense_ffn_prefill_nvfp4.rs:56-60,272-286,330-372 |
| `ffn_mmq64_gate_up` | reference | ml/dense_ffn_prefill_nvfp4.rs:61-66 (METRALE_NO_MMQ_TILE64 unset, ml/dense_ffn.rs:250-253) |
| `ffn_mmq64_act_down` | reference | ml/dense_ffn_prefill_nvfp4.rs:61-66,272-286,330-372 |
| `ffn_mmq_pipe_gate_up` | reference | ml/dense_ffn_prefill_nvfp4.rs:67-68; ml/ops/nvfp4_mmq.rs:175-183 (the 128 tile with K % 256 == 0 runs metrale_nvfp4_gemm_pipe) |
| `ffn_mmq_pipe_act_down` | reference | ml/ops/nvfp4_mmq.rs:175-183 (the pipe applies the down scale in its store, so no metrale_nvfp4_scale_bf16) |
| `w8a16_m32` | reference | ml/qwen3_ssm/row_tier_proj.rs:64-117; ml/qwen3_attention/decode/attention_forward/q_proj.rs:49-61; ml/qwen3_attention/decode/attention_forward_kv.rs:45-72; ml/ops/gemm_quant_w8a16.rs:272-274; ml/qwen3_attention/trait_impl/multi_seq/qkv_fp8_batch.rs:200-212; ml/qwen3_attention/trait_impl/multi_seq/attn/o_proj.rs:231-244; ml/ops/w8a16_gemm_pipelined_m32.rs:287-313 |
| `w8a16_m64` | reference | ml/qwen3_attention/trait_impl/multi_seq/qkv_fp8_batch.rs:200-212,253-258 (canonical chunk 64); ml/qwen3_attention/trait_impl/multi_seq/attn/o_proj.rs:231-244; ml/ops/w8a16_gemm_pipelined_m32.rs:314-326 |
| `w8a16_m64_attn_chunked` | reference | ml/qwen3_attention/trait_impl/multi_seq/qkv_fp8_batch.rs:253-285 (64-row chunks); ml/qwen3_attention/trait_impl/multi_seq/attn/o_proj.rs:231-244,275-300 (step 64) |
| `w8a16_full_gdn_wide` | reference | ml/qwen3_ssm/trait_decode_multi_seq/ssm_batched_proj.rs:89-155,233-299; ml/ops/w8a16_gemm_pipelined_m32.rs:287-340 (w8a16_gemm_pipelined_by_m above 64 rows) |
| `moe_router_gemv` | reference | ml/moe/forward/route.rs:36 |
| `moe_topk_softmax` | reference | ml/moe/forward/route.rs:117-128 (ties go to the lower expert index) |
| `moe_gate_up_shared_fp8` | reference | ml/moe/forward.rs:231-258 (grid y 0..7 routed experts, y = 8 the shared expert); ml/moe/init.rs:431-434 |
| `moe_silu_down_shared_fp8` | reference | ml/moe/forward.rs:259-300; ml/moe/init.rs:435-438 |
| `moe_weighted_sum_blend` | reference | ml/moe/forward.rs:442-472 (sum of w * out plus sigmoid(x . seg) * shared, rounded to BF16 once); the EP reduce is a no-op without EP, ml/moe/forward/ep_reduce.rs:12-50 |
| `moe_grouped_router_rows` | reference | ml/moe/forward_fp8_grouped_router.rs:95-137 (the per-row router repeats the one-row router's bits, :6-12); arms ml/qwen3_attention/trait_impl/multi_seq/ffn.rs:94-130, ml/qwen3_ssm/trait_decode_multi_seq.rs:154-170, ml/qwen3_ssm/trait_decode_batched.rs:304-320 |
| `moe_grouped_topk_sort` | reference | ml/moe/forward_fp8_grouped_decode.rs:258-289 (FP8_GROUPED_DECODE_MAX_ROWS = 64, :28) |
| `moe_gate_up_act_grouped` | reference | ml/moe/forward_fp8_grouped_decode.rs:290-311; k/moe_shared_expert_fused_fp8_grouped.cu:17-22 (writes the FP32 SiLU product; per row equal to moe_shared_expert_fused_fp8.cu bit for bit) |
| `moe_down_act_grouped` | reference | ml/moe/forward_fp8_grouped_decode.rs:312-330 |
| `moe_blend_grouped` | reference | ml/moe/forward_fp8_grouped_decode.rs:331-347 |
| `moe_prefill_fp8_w8a8_wide` | reference | ml/qwen3_attention/trait_impl/multi_seq/ffn.rs:233-260 and ml/qwen3_ssm/trait_decode_multi_seq.rs:196-210 (forward_prefill above the grouped cap); ml/moe/forward_prefill_fp8.rs:113-165 (shared W8A8 first, input quantized once), :166-265 (router: dense_gemm_router below MOE_ROUTER_RT_MIN_ROWS = 1024, ml/moe/helpers_c.rs:246-283), :313-395; ml/moe/forward_prefill_fp8/gate_up.rs:57-160 and down.rs:40-140 (ctx.decode_step declines the adaptive and E4M3 arms, ml/moe/adaptive_fp8.rs:105-116); ml/moe/forward_prefill_fp8/combine.rs:40-60 (W8A8 activations: not row-invariant with paths A and B) |
| `lm_head_bf16_gemv` | reference | mm/impl_a3_lm_head.rs:428-438 (use_fp32_logits is false, mm/impl_a1/kernels.rs:200) |
| `lm_head_bf16_batchm` | reference | me/lm_head_batched.rs:133-162 (m <= lm_head_batchm_max = 8, kernels/gb10/HARDWARE.toml:115); mm/impl_a3_lm_head.rs:177-229 (verify); per row bit-identical to dense_gemv_bf16, k/dense_gemv_bf16_batchm.cu:16 |
| `lm_head_bf16_gemm` | reference | me/lm_head_batched.rs:133-162 |
| `draft_lm_head_nvfp4` | reference | ml/mtp_head/forward.rs:366-377 (the draft head is NVFP4 whatever the target head: a draft-only copy when the target head is BF16, mm/impl_a1/ssm_setup.rs:48-50) |
| `draft_argmax` | reference | ml/mtp_head/forward.rs:437-455 |
| `draft_concat` | reference | ml/mtp_head/forward.rs:101-109 |
| `draft_dense_gemv_bf16` | reference | ml/mtp_head/forward.rs:123,171-183,209-212,309; ml/mtp_head.rs:330-339 (mtp_quantization bf16: every projection is ProjectionWeight::Bf16) |
| `draft_dense_gate_up_bf16` | reference | ml/mtp_head/moe_forward.rs:30-64 (dense_ffn_forward_generic: gate and up are two GEMVs) |

## What the audit found in today's routing

These are facts about the code, found while encoding it. They are not changes.

1. **The dense checkpoint's FP8 layers run as NVFP4.**
   - The per-channel FP8 attention q/k/v/o, linear_attn in_proj_qkv/in_proj_z/out_proj, and the MLPs of layers 56-63 fail the native-FP8 test.
   - They are requantized to NVFP4 at load (crates/model-arch/src/weight_loader/qwen35_dense.rs:81-103).
   - Decode runs W4A16 everywhere. That contradicts "follow the checkpoint's precision"; perf/weight-quantization addresses it.
2. **Several default routes already run below the declared activation precision:**
   - The dense multi-sequence FFN takes the NVFP4 MMQ path (W4A4) with no flag: at 8 rows or more in GDN layers, and 12 or more in attention layers.
   - The tile GEMMs at 9-64 rows and above round activations to E4M3 (kernels/gb10/qwen3.6-27b/nvfp4/w4a16_gemm.cu:344-347).
3. **Canonical MoE tiers are not row-invariant above 64 rows.**
   - Batched decode at 96/128 takes the W8A8 `forward_prefill_fp8` path.
   - In batched verify at R > 64, the GDN-layer FFN loops path A per row while the attention-layer FFN takes W8A8, so two layers of one step use different MoE arithmetic.
4. **Probable defect: FP8 KV scales are never loaded for the dense checkpoint.**
   - `load_kv_scales` looks up `{p}.k_proj.k_scale`; the checkpoint ships `self_attn.k_scale`.
   - Not exercised by the golden policy (bf16 KV). It matters for the throughput recipe's fp8 KV.
5. **The GDN batched recurrence almost never engages at a padded width.** Padding rows share one dummy slot, so the contiguity check fails (ml/qwen3_ssm/trait_decode_multi_seq/ssm_batched_recurrent.rs:93-149).
6. **`gdn_fused_norm` is not only a fusion.** The fused kernel clamps the state's Frobenius norm and the unfused one does not (k/gated_delta_rule.cu:940-942, 1051). The rule is therefore `differs`, not `bit_identical`.
7. **The dense throughput recipe's K ladder disagrees with its BENCH pin.** The recipe prose says `1:3,2:1,4:2,8:2,16:1`; BENCH.toml says `1:3,2:2,4:1,8:1,16:1`, and the BENCH pin wins.

## Open items: routing the circuit cannot express yet

**Live state.** These are decided per step at run time. A plan is keyed by (mode, rows) only.

- MTP gate `auto` (the dense agentic recipe) switches between verify and plain decode on measured throughput.
- The K ladder, adaptive rung, D-Cut row pruning and the per-slot capacity clamp choose the verify width.
- Graph borrowing replays a wider captured graph, so kernels run at the captured width.
- GDN carry is eager or lazy (8 sequences or more). Per-sequence fallback happens on non-contiguous slots or null WY tables.
- The batched-recurrence contiguity check (item 5 above).
- Mixed prefill+decode steps, preemption, grammar truncation of drafts.
- FP8 KV calibration suppresses graphs until its window freezes.

**Modes not modelled.**

- Batched verify (`verify_e`, R = sum of ks over sequences).
- The batched draft propose for two or more sequences.
- Prefill.

**Env-tunable thresholds baked into rule row ranges at their defaults.** Each is a numeric lever the policy does not carry:

- `METRALE_SSM_FFN_PREFILL_MIN_N` (5);
- `METRALE_SSM_TC_PROJ` (9);
- `ffn_proj_max_rows` (8; 32 under w4a4);
- the grouped-MoE cap (64);
- `lm_head_batchm_max` (8, HARDWARE.toml);
- `MOE_ROUTER_RT_MIN_ROWS` (1024);
- `METRALE_NO_SSM_M128`.

**Kill switches not modelled.** Most `METRALE_NO_*` levers are absent. The policy states only the settings a rule reads:

- `row_tiers`, `kv_cache_dtype`, `lm_head_dtype`, `ssm_h_dtype`, `ssm_batched_recurrent`;
- `gemv_sw`, `w4a16_tc`, `decode_split_silu`.

**The W8A8 MoE path above 64 rows is one opaque group.** `moe_prefill_fp8_w8a8_wide` pins today's 15-launch sequence. Inside it, the engine picks between arms that the rule does not model:

- adaptive and E4M3 arms, both declined under `decode_step`;
- pm4 versus grouped GEMM, by handle presence;
- the shared expert as W8A8, W8A16 or an installed BF16 copy;
- side-stream overlap for the shared expert.

**Copies are not launches.** These D2D copies are not kernels and are not counted:

- embedding rows;
- the draft head's residual copy;
- the 3n `qkv_buf` copies of the 2/3-row attention batch.

**Other gaps.**

- **EP reduce** is omitted: EP is not used by either recipe.
- **Recipes without an instance.** `qwen3.8-27b-nvfp4-throughput` and `qwen3.6-35b-a3b-fp8-nvfp4head` need rules the golden policies never reach, and `met circuit show` refuses them by name:
  - FP8 KV, the NVFP4 head (tile GEMM under canonical tiers);
  - f16-pool state and the batched recurrence;
  - W4A4 at 9-64 rows (mx16/mx32/mx64 and the `_ps`/`_nt` twins).
- **Only four `differs` levers are modelled:** `gdn_fused_norm`, `decode_fused_silu`, `gdn_fused_verify` (K=2) and `w4a4_downcast` (4-8 rows, mx8).
- **Sampling is outside the main circuit.** The verify's per-row argmax is part of today's verify forward but not of the circuit.
- **Strided RoPE equals MRoPE only for text.** The strided kernel matches the interleaved MRoPE only while the three positions are equal. An image sequence's multi-row step should be checked.
