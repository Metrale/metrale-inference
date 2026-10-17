// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Construction of the DFlash drafter head from its loaded weights: kernel
//! handles, the paged BF16 drafter KV pool, scratch, the RoPE table, the fused K/V
//! weight and, by default, FP8 weight copies.
//!
//! Owner: model-arch (DFlash drafter).
//! Invariants: none beyond the types.

mod kernel_handles;
mod quant_weights;
mod rope_table;
mod scratch;
mod startup_validation;

use anyhow::Result;
use metrale_cache::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use parking_lot::Mutex;

use super::{BlockDiffusionDraftHead, DflashLayer, DflashQuantization};
use crate::weight_loader::DflashWeights;

impl BlockDiffusionDraftHead {
    pub fn from_weights(
        weights: DflashWeights,
        embed_tokens_shared: DevicePtr,
        lm_head_shared: DevicePtr,
        lm_head_nvfp4: Option<metrale_model_layers::weight_map::QuantizedWeight>,
        lm_head_native_fp8: Option<(metrale_model_layers::weight_map::Fp8DenseWeight, usize)>,
        target_hidden_size: usize,
        gamma: Option<usize>,
        window_size: Option<usize>,
        gpu: &dyn GpuBackend,
        max_seq_len: usize,
        // 2026-09-25: The widest cross-sequence batch one forward serves. The
        // gamma-sized scratch holds this many bands of gamma rows, and the drafter KV
        // pool holds this many sequences' blocks.
        max_batch_size: usize,
        // 2026-10-10: This rank's part of a tensor-parallel drafter: `weights` and
        // `lm_head_shared` are already cut to it (`tp::shard_weights`).
        tp: Option<super::tp::DflashTp>,
        // 2026-10-10: `--draft-weight-quantization`: the copies of the GEMM weights to build.
        weight_quant: super::DraftWeightQuantization,
    ) -> Result<Self> {
        let nb = max_batch_size.max(1);
        let target_layer_ids = weights
            .config
            .dflash_config
            .as_ref()
            .map(|c| c.target_layer_ids.clone())
            .unwrap_or_default();
        let mask_token_id = weights
            .config
            .dflash_config
            .as_ref()
            .map(|c| c.mask_token_id)
            .unwrap_or(0);

        if target_layer_ids.is_empty() {
            anyhow::bail!(
                "DFlash drafter config.json has no `dflash_config.target_layer_ids` — \
                 cannot determine which target hidden states to capture"
            );
        }

        let _ = target_hidden_size;

        let num_layers = weights.config.num_hidden_layers;
        let hidden_size = weights.config.hidden_size;
        let intermediate_size = weights.config.intermediate_size;
        let num_q_heads = weights.config.num_attention_heads;
        let num_kv_heads = weights.config.num_key_value_heads;
        let head_dim = weights.config.head_dim;
        let vocab_size = weights.config.vocab_size;
        let levers = super::levers::DFlashLevers::from_env();
        // Resolve handles and reject unsupported TP combinations before allocating the KV pool
        // or the head's scratch buffers.
        let kernels = kernel_handles::load_kernels(gpu)?;
        let dflash2_cfg = weights.config.dflash_config.as_ref();
        let dflash2_selector_active = weights.selector_pred.is_some()
            && weights.selector_succ.is_some()
            && weights.selector_hidden_proj.is_some()
            && dflash2_cfg.is_some_and(|c| {
                c.selector_top_k == 16
                    && c.selector_rank > 0
                    && c.conv_kernel_size == 2
                    && c.conv_group_size > 0
            })
            && levers.dflash2
            && kernels.dflash2_conv2.0 != 0
            && kernels.dflash2_topk16.0 != 0
            && kernels.dflash2_selector_walk.0 != 0;
        startup_validation::validate_tp_startup(
            tp.is_some(),
            levers.topk16_split,
            dflash2_selector_active,
            kernels.dflash2_topk16_part.0 != 0,
            kernels.dflash2_topk16_merge.0 != 0,
            vocab_size,
        )?;
        // 2026-09-25: `gamma` is `--dflash-gamma` when given, else `default_dflash_gamma`
        // of the drafter's `effective_block_size()` (`dflash_config.block_size`, else the
        // top-level `block_size`, which defaults to 16). 2026-09-30: one resolver
        // (`resolve_dflash_gamma`) for this, the factory's pools and the server's reserve. The
        // serve reads the head's gamma back through `block_gamma()`.
        let gamma_val = metrale_model_layers::layers::qwen3_ssm::resolve_dflash_gamma(
            gamma,
            Some(weights.config.effective_block_size()),
        );

        // 2026-09-25: The drafter's paged KV cache: one cache for all drafter layers,
        // in 16-token blocks.
        let block_size = 16;
        let kv_config = KvCacheConfig {
            block_size,
            num_kv_heads,
            head_dim,
            num_layers,
            // 2026-09-25: BF16, the type the drafter's paged attention
            // (`attn_prefill_paged_indirect`) reads and `reshape_and_cache_flash` writes.
            dtype: KvCacheDtype::Bf16,
            layer_dtypes: vec![],
            layer_dims: vec![],
            cache_blocks_per_seq: None,
        };
        // 2026-09-25: Each sequence's lazy block allocation in propose.rs takes
        // `ceil((max_ctx_len + gamma + 1) / 16)` blocks, and `max_ctx_len <=
        // max_seq_len`, so `per_seq_blocks` covers one sequence. The pool holds
        // `max_batch_size` of them plus one spare block.
        // provenance-id: 526f6e616c6420522e205374657369616b
        let per_seq_blocks = (max_seq_len + gamma_val + 1).div_ceil(block_size);
        let num_blocks = per_seq_blocks * max_batch_size.max(1) + 1;
        tracing::info!(
            "DFlash drafter paged KV pool: {} blocks ({} per-seq x max_batch_size {})",
            num_blocks,
            per_seq_blocks,
            max_batch_size.max(1)
        );
        let kv_cache = PagedKvCache::new(kv_config, num_blocks, gpu)?;

        // 2026-09-25: Scratch, allocated once. The row buffers hold `rows_max` rows: the
        // non-paged path's `ctx_window` ctx rows plus `gamma` block rows, or `nb` bands of
        // `gamma` rows for a batched propose, whichever is more.
        let bf16 = 2usize;
        let g = gamma_val;
        // 2026-09-25: `ctx_window` bounds the ctx rows the non-paged path attends to and
        // the rows one ctx precompute call takes. METRALE_DFLASH_CTX_WINDOW (default
        // 4096), read here at construction. The row scratch grows linearly with it.
        let ctx_window: usize = std::env::var("METRALE_DFLASH_CTX_WINDOW")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4096);
        tracing::info!(
            "DFlash ctx_window = {} (set METRALE_DFLASH_CTX_WINDOW to override; \
             drafter trained on full captured prefix — larger is better, \
             scratch grows linearly)",
            ctx_window
        );
        let n_attn = g + ctx_window;
        let rows_max = n_attn.max(nb * gamma_val);
        let q_dim = num_q_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let scratch = scratch::alloc_scratch(
            gpu,
            &weights,
            &scratch::ScratchDims {
                nb,
                g,
                rows_max,
                hidden_size,
                q_dim,
                kv_dim,
                intermediate_size,
                ctx_window,
                num_layers,
                num_kv_heads,
                head_dim,
                vocab_size,
                target_layer_ids: &target_layer_ids,
                target_hidden_size,
                max_seq_len,
                gamma_val,
                tp,
            },
        )?;

        // 2026-09-25: The RoPE inverse-frequency table, from the drafter config's
        // `rope_scaling`: absent gives plain RoPE (`1 / theta^(2j / dim)`),
        // `rope_type = "yarn"` the YaRN-blended table, and any other value plain RoPE
        // with a warning. RoPE rotates the whole head (`rotary_dim = head_dim`).
        let rope_theta = weights
            .config
            .effective_rope_theta()
            .map_err(|e| anyhow::anyhow!(e))?;
        let rotary_dim = head_dim;
        let inv_freq_table = rope_table::rope_inv_freq_table(&weights, rope_theta, rotary_dim);

        let inv_freq_bytes: Vec<u8> = inv_freq_table
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let yarn_inv_freq = gpu.alloc(inv_freq_bytes.len())?;
        gpu.copy_h2d(&inv_freq_bytes, yarn_inv_freq)?;

        // 2026-09-25: The fused K/V weight `[L * 2 * kv_dim, hidden]` BF16, laid out
        // `[K_0, V_0, K_1, V_1, ...]` and copied device to device from each layer's
        // `k_proj` and `v_proj`, so `precompute_ctx_kv` computes every layer's ctx K/V in
        // one GEMM. `kv_dim_bytes` is the size of one layer's K (or V) weight.
        let kv_dim_bytes = num_kv_heads * head_dim * hidden_size * bf16;
        let fused_total_bytes = num_layers * 2 * kv_dim_bytes;
        let fused_kv_weight = gpu.alloc(fused_total_bytes)?;
        for (l, layer) in weights.layers.iter().enumerate() {
            let layer_base = l * 2 * kv_dim_bytes;
            gpu.copy_d2d(
                layer.k_proj.weight,
                fused_kv_weight.offset(layer_base),
                kv_dim_bytes,
            )?;
            gpu.copy_d2d(
                layer.v_proj.weight,
                fused_kv_weight.offset(layer_base + kv_dim_bytes),
                kv_dim_bytes,
            )?;
        }
        tracing::info!(
            "DFlash fused_kv_weight: {} bytes ({} layers × 2 × kv_dim × h × bf16), \
             layout [K0,V0,K1,V1,…] to match vLLM precompute_and_store_context_kv",
            fused_total_bytes,
            num_layers,
        );

        let lm_head_whole = weights.tp_lm_head;
        let mut head = Self {
            // 2026-09-25: The drafter's `DFlashLevers`, resolved from the environment once,
            // here.
            levers,
            num_layers,
            hidden_size,
            intermediate_size,
            num_q_heads,
            num_kv_heads,
            head_dim,
            vocab_size,
            draft_vocab_size: weights.config.draft_vocab_size.unwrap_or(vocab_size),
            gamma: gamma_val,
            block_gamma: std::sync::atomic::AtomicUsize::new(gamma_val),
            max_batch: nb,
            mask_token_id,
            window_size,
            target_layer_ids,
            target_hidden_size,

            embed_tokens_shared,
            lm_head_shared,
            lm_head_nvfp4,
            lm_head_q: None,
            hidden_norm: weights.hidden_norm,
            norm: weights.norm,
            fc: weights.fc,
            draft_id_to_target_id: None,
            layers: weights
                .layers
                .into_iter()
                .map(|l| DflashLayer {
                    input_layernorm: l.input_layernorm,
                    post_attention_layernorm: l.post_attention_layernorm,
                    q_proj: l.q_proj,
                    k_proj: l.k_proj,
                    v_proj: l.v_proj,
                    o_proj: l.o_proj,
                    q_norm: l.q_norm,
                    k_norm: l.k_norm,
                    gate_proj: l.gate_proj,
                    up_proj: l.up_proj,
                    down_proj: l.down_proj,
                    // 2026-09-25: Filled below when the FP8 drafter path is on.
                    q_proj_q: None,
                    k_proj_q: None,
                    v_proj_q: None,
                    o_proj_q: None,
                    gate_proj_q: None,
                    up_proj_q: None,
                    down_proj_q: None,
                    attention_conv_base: l.attention_conv_base,
                    attention_conv_proj: l.attention_conv_proj,
                    mlp_conv_base: l.mlp_conv_base,
                    mlp_conv_proj: l.mlp_conv_proj,
                    attention_conv_proj_q: None,
                    mlp_conv_proj_q: None,
                    tp_src: l.tp_src,
                })
                .collect(),
            fused_kv_weight: Some(fused_kv_weight),
            kv_cache: Mutex::new(kv_cache),
            scratch,
            kernels,
            max_seq_len,
            yarn_inv_freq,
            rope_theta,
            rotary_dim,
            rms_norm_eps: weights
                .config
                .effective_rms_norm_eps()
                .map_err(|e| anyhow::anyhow!(e))?,
            ctx_window,
            propose_graphs: parking_lot::Mutex::new(super::ProposeGraphs::default()),
            suppress_graphs: std::sync::atomic::AtomicBool::new(false),
            propose_warmup_count: std::sync::atomic::AtomicUsize::new(0),
            quant: DflashQuantization::Bf16,
            // 2026-09-25: `markov_rank` is 0 when the checkpoint has no `markov_w1`.
            markov_rank: if weights.markov_w1.is_some() {
                weights.config.markov_rank
            } else {
                0
            },
            markov_w1: weights.markov_w1,
            markov_w2: weights.markov_w2,
            confidence_proj: weights.confidence_proj,
            confidence_bias: weights.confidence_bias,
            confidence_with_markov: weights.config.confidence_head_with_markov,
            shifted_rows: weights
                .config
                .dflash_config
                .as_ref()
                .and_then(|c| c.projector_type.as_deref())
                == Some("dspark"),
            conv_kernel_size: weights
                .config
                .dflash_config
                .as_ref()
                .map(|c| c.conv_kernel_size)
                .unwrap_or(0),
            conv_group_size: weights
                .config
                .dflash_config
                .as_ref()
                .map(|c| c.conv_group_size)
                .unwrap_or(0),
            selector_rank: weights
                .config
                .dflash_config
                .as_ref()
                .map(|c| c.selector_rank)
                .unwrap_or(0),
            selector_top_k: weights
                .config
                .dflash_config
                .as_ref()
                .map(|c| c.selector_top_k)
                .unwrap_or(0),
            selector_pred: weights.selector_pred,
            selector_succ: weights.selector_succ,
            selector_hidden_proj: weights.selector_hidden_proj,
            aux_q: None,
            tp,
        };
        if head.selector_pred.is_some() {
            tracing::info!(
                "DFlash2 armed: conv k={} group={} selector rank={} top_k={} \
                 (kernels present: conv={} topk={} walk={})",
                head.conv_kernel_size,
                head.conv_group_size,
                head.selector_rank,
                head.selector_top_k,
                head.kernels.dflash2_conv2.0 != 0,
                head.kernels.dflash2_topk16.0 != 0,
                head.kernels.dflash2_selector_walk.0 != 0,
            );
        }
        if head.shifted_rows {
            tracing::info!(
                "DSpark drafter: SpecForge shifted row convention active \
                 (projector_type=dspark) — draft vector rotates right by 1"
            );
        }

        tracing::info!(
            "BlockDiffusionDraftHead loaded: {} layers, hidden={}, intermediate={}, \
             GQA {}/{}, head_dim={}, γ={}, vocab={}, mask_token_id={}, target_layers={:?}",
            head.num_layers,
            head.hidden_size,
            head.intermediate_size,
            head.num_q_heads,
            head.num_kv_heads,
            head.head_dim,
            head.gamma,
            head.vocab_size,
            head.mask_token_id,
            head.target_layer_ids,
        );

        // 2026-09-25: The quantized drafter weights (`--draft-weight-quantization`): each
        // layer's seven dense GEMM weights (q/k/v/o/gate/up/down) and the lm_head, in FP8 E4M3
        // with one f32 scale per row, or (2026-10-10) NVFP4; `quant` follows. The FP8 tiers need
        // the row-scaled FP8 GEMM and its m16 variant; without them the drafter stays on BF16
        // with a warning, as before the flag. The NVFP4 tier needs its quantizers and fails
        // without them.
        let fp8_kernels_present = head.kernels.fp8_gemm_n128_row_scaled.0 != 0
            && head.kernels.fp8_gemm_n128_row_scaled_m16.0 != 0;
        let nvfp4 = weight_quant == super::DraftWeightQuantization::Nvfp4;
        anyhow::ensure!(
            !nvfp4
                || (head.kernels.nvfp4_global_absmax.0 != 0
                    && head.kernels.quantize_bf16_to_nvfp4.0 != 0),
            "--draft-weight-quantization nvfp4: this target's kernels have no \
             quantize_nvfp4 module"
        );
        let quantize = weight_quant.quantizes_layers() && (nvfp4 || fp8_kernels_present);
        if weight_quant.quantizes_layers() && !quantize {
            tracing::warn!(
                "--draft-weight-quantization {} but the fp8_gemm_t_row_scaled(_m16) kernels are \
                 in neither this target's w4a16 nor its dflash_fp8_gemm PTX module — staying on \
                 the BF16 drafter path.",
                weight_quant.name()
            );
        }
        let qz = super::draft_weight::DraftQuantizer {
            gpu,
            nvfp4,
            fp8_k: head.kernels.quantize_bf16_to_fp8,
            absmax_k: head.kernels.nvfp4_global_absmax,
            nvfp4_k: head.kernels.quantize_bf16_to_nvfp4,
            stream: 0,
        };
        if quantize {
            quant_weights::quantize_drafter(
                &mut head,
                &qz,
                q_dim,
                kv_dim,
                lm_head_native_fp8,
                lm_head_whole,
            )?;
        }
        // 2026-10-10: The auxiliary GEMMs' copies, in the layer copies' format;
        // `quantize_aux` refuses a head whose layers are not.
        if weight_quant.quantizes_aux() {
            head.aux_q = Some(head.quantize_aux(&qz)?);
        }

        Ok(head)
    }

    /// 2026-09-25: Errors when `target_hidden_size` differs from the value the head was
    /// built with, which sets the `fc` input width
    /// (`target_layer_ids.len() * target_hidden_size`). Nothing calls it.
    pub fn validate_against_target(&self, target_hidden_size: usize) -> Result<()> {
        if self.target_hidden_size != target_hidden_size {
            anyhow::bail!(
                "DFlash drafter target_hidden_size mismatch: drafter expects {}, target is {}",
                self.target_hidden_size,
                target_hidden_size
            );
        }
        Ok(())
    }
}
