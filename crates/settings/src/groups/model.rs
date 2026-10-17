// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The `model` lever group: the `model.*` levers a `met` process reads at run time.
//!
//! Owner: settings.
//! Invariants: rows in env-name order.

lever_group! {
    /// 2026-10-17: The `model` group, resolved by `crate::resolve()`.
    pub struct ModelSettings { group: "model", class: Runtime }
    ds41_candidate_source: Int = "built-in", None, "METRALE_DS41_CANDIDATE_SOURCE", "crates/model-arch/src/weight_loader/deepseek_v41/load_layers.rs", "Read in `crates/model-arch/src/weight_loader/deepseek_v41/load_layers.rs`.";
    ds41_evict_random_pct: Text = "unset", None, "METRALE_DS41_EVICT_RANDOM_PCT", "crates/model-weights/src/legacy_env.rs", "The victim is drawn from the oldest this-many percent (0 = strict LRU).";
    ds41_kquant_resident: Text = "unset", None, "METRALE_DS41_KQUANT_RESIDENT", "crates/model-weights/src/legacy_env.rs", "DeepSeek-V4.1 attention projections and shared expert stay Q2_K / Q3_K (ggml types 10 / 11) on the device; METRALE_DS41_KQUANT_RESIDENT=0 restores the bf16 expansion.";
    ds41_q6k_head: Text = "unset", None, "METRALE_DS41_Q6K_HEAD", "crates/model-weights/src/legacy_env.rs", "The DeepSeek-V4.1 tensors that stay K-quant on the device, by ggml type id and HF name: the attention projections and shared expert as Q2_K / Q3_K (10 / 11; `METRALE_DS41_KQUANT_RESIDENT=0`...";
    ds41_reader_pool: Switch = "off", None, "METRALE_DS41_READER_POOL", "crates/model-arch/src/weight_loader/deepseek_v41/load_layers.rs", "`METRALE_DS41_READER_POOL=1`: the persistent reader pool (misses and predicted experts in flight together; what `METRALE_DS41_PREFETCH_K` needs). `1` turns it on.";
    ds41_router_staged: Switch = "on", None, "METRALE_DS41_ROUTER_STAGED", "crates/model-arch/src/moe_v41/route.rs", "The staged single-token router GEMV; `METRALE_DS41_ROUTER_STAGED=0` selects the products form. `0` turns it off.";
    ds41_shared_side: Switch = "on", None, "METRALE_DS41_SHARED_SIDE", "crates/model-arch/src/moe_v41.rs", "At one token the shared expert runs on a side stream while the main stream runs the router; the accumulation order does not change. `0` turns it off.";
    gguf_native_q2: Switch = "off", None, "METRALE_GGUF_NATIVE_Q2", "crates/gpu-runtime/src/legacy_env.rs", "Both scratch buffers are env-gated — 0 (→ NULL) unless the respective `METRALE_GGUF_NATIVE_Q2*` flag is set, so non-Q2 models pay nothing. `1` turns it on.";
    glm_arena_lean: Switch = "off", None, "METRALE_GLM_ARENA_LEAN", "crates/model-arch/src/glm5next_layer/levers.rs", "`1` sizes the buffer arena of a `glm5_next` model for the regions its layer stack reads (hidden_states, residual, norm_output, moe_output, logits, token_ids, scratch, the hc streams/post/comb, LoRA slots) and shrinks every other architecture's region to 256 B (qkv/attn outputs, ssm_*, expert_*_out, fp8_act, gate_logits, ...): -2.8 GB at 8208 prefill rows, -5.6 GB at 16400 on GLM-5.3. Refused at load on another model type. `0` or unset keeps the full arena.";
    glm_cublas_proj: Switch = "on", None, "METRALE_GLM_CUBLAS_PROJ", "crates/model-arch/src/glm5next_layer/levers.rs", "`METRALE_GLM_CUBLAS_PROJ=0` is the kill-switch. `0` turns it off.";
    glm_dsa_ws_shared: Switch = "on", None, "METRALE_GLM_DSA_WS_SHARED", "crates/model-arch/src/glm5next_dsa/layer/workspace.rs", "ONE DSA scratch workspace for every DSA layer of the GLM text stack (as the KDA and MLP workspaces); `METRALE_GLM_DSA_WS_SHARED=0` restores one per layer (11 x 0.67 GB at 8208 prefill rows on GLM-5.3). `0` turns it off.";
    glm_mlp_ws_shared: Switch = "on", None, "METRALE_GLM_MLP_WS_SHARED", "crates/model-arch/src/glm5next_mlp/forward.rs", "ONE MLP scratch for the whole layer stack; `METRALE_GLM_MLP_WS_SHARED=0` restores one per layer. `0` turns it off.";
    glm_vision: Text = "unset", None, "METRALE_GLM_VISION", "crates/config/src/legacy_env.rs", "Read in `crates/config/src/legacy_env.rs`.";
    glm_vision_check: Text = "unset", None, "METRALE_GLM_VISION_CHECK", "crates/model-engine/src/model/trait_impl/prefill_a/vision_sync.rs", "Read in `crates/model-engine/src/model/trait_impl/prefill_a/vision_sync.rs`.";
    lora_allow_partial: Switch = "off", None, "METRALE_LORA_ALLOW_PARTIAL", "crates/config/src/legacy_env.rs", "`METRALE_LORA_ALLOW_PARTIAL=1` — load an adapter naming target modules Metrale Engine cannot apply, skipping those and applying the rest. `1` or `true` turns it on.";
    lora_eager: Switch = "off", None, "METRALE_LORA_EAGER", "crates/model-layers/src/layers/ops/model_levers_resolve.rs", "Sets `lora_eager` (read in `crates/model-layers/src/layers/ops/model_levers_resolve.rs`). `1` or `true` turns it on.";
    lora_no_apply: Switch = "off", None, "METRALE_LORA_NO_APPLY", "crates/model-layers/src/layers/ops/lora_delta.rs", "Keep the adapter resident but skip every `apply_lora_delta`: separates the cost of applying an adapter from the cost of having one loaded. `1` turns it on.";
    lora_rotate: Switch = "off", None, "METRALE_LORA_ROTATE", "crates/model-layers/src/layers/ops/model_levers_resolve.rs", "Sets `lora_rotate` (read in `crates/model-layers/src/layers/ops/model_levers_resolve.rs`). `1` or `true` turns it on.";
    model_id: Text = "unset", None, "METRALE_MODEL_ID", "crates/model-engine/src/model/ssm_tier/fingerprint.rs", "`METRALE_MODEL_ID` (optional) is an extra salt for the one case geometry cannot distinguish: a fine-tune with byte-identical config — unset (the common case) it contributes an empty record and...";
    ple_serial_fault: Switch = "off", None, "METRALE_PLE_SERIAL_FAULT", "crates/storage/src/links.rs", "`METRALE_PLE_SERIAL_FAULT=1` faults every batch serially. `1` turns it on.";
    qwen4exp_decode_prof: Switch = "off", None, "METRALE_QWEN4EXP_DECODE_PROF", "crates/model-layers/src/layers/qwen3_ssm/trait_decode_hc.rs", "METRALE_QWEN4EXP_DECODE_PROF=1: per-stage wall clock, first tokens only (the counter caps the log volume). `1` turns it on.";
    qwen4exp_no_ple: Switch = "off", None, "METRALE_QWEN4EXP_NO_PLE", "crates/model-arch/src/weight_loader/qwen4_exp.rs", "With PLE disabled for bisection, skip the 21 MB arena and the 128-shard open entirely rather than building what we will not run. `1` turns it on.";
    vision_max_pixels: Int = "0", "--vision-max-pixels", "METRALE_VISION_MAX_PIXELS", "crates/server/src/main_modules/serve_phases/load_levers.rs", "The pixel cap on one image; `0` leaves the model's own cap.";
    vision_posinterp: Switch = "on", None, "METRALE_VISION_POSINTERP", "crates/model-layers/src/layers/vision_encoder/enc_impl/forward.rs", "1. `0` turns it off.";
    weight_dual_rail: Switch = "off", None, "METRALE_WEIGHT_DUAL_RAIL", "crates/model-weights/src/legacy_env.rs", "Rail 0 defaults to the shared expert CX7 link; dual-rail is opt-in (METRALE_WEIGHT_DUAL_RAIL=1). `1` turns it on.";
    weight_rail2_dev: Text = "unset", None, "METRALE_WEIGHT_RAIL2_DEV", "crates/model-weights/src/legacy_env.rs", "Read in `crates/model-weights/src/legacy_env.rs`.";
    weight_rail2_gid: Text = "unset", None, "METRALE_WEIGHT_RAIL2_GID", "crates/model-weights/src/legacy_env.rs", "Read in `crates/model-weights/src/legacy_env.rs`.";
}
