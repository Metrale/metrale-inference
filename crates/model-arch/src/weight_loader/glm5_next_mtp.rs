// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Load GLM-5.3's MTP block, `model.language_model.layers.{num_hidden_layers}`,
//! as the body of the draft proposer.
//!
//! Owner: model-arch weight loader (GLM-5.3).
//! Invariants:
//! - The block is one DSA mixer and one routed MoE with `mhc: None`, the plain-residual path
//!   of [`crate::glm5next_layer::Glm5NextLayer`].
//! - Its DSA layer uses `attn_layer_idx = 0` of the drafter's own one-layer KV pool
//!   (`glm5next_mtp_head`), never the target's.
//! - `Glm5NextWeightLoader::prune_after_load` keeps this layer's tensors: `is_reuploaded`
//!   matches only layer indices below `num_hidden_layers`.

use anyhow::{Context, Result};
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_weights::weights::WeightStore;

use crate::glm5next_dsa::Glm5NextDsaKernels;
use crate::glm5next_dsa::build::build_dsa_weights;
use crate::glm5next_dsa::layer::{Glm5NextDsaLayer, Glm5NextDsaLayerKernels, Glm5NextDsaWorkspace};
use crate::glm5next_layer::{Glm5NextLayer, Glm5NextMixer, Glm5NextMlpSite};
use crate::glm5next_mlp::{Glm5NextMlpConfig, Glm5NextMlpKernels, build as mlp_build};
use metrale_model_layers::weight_map::DenseWeight;

/// 2026-09-25: The MTP block plus the four tensors around it.
pub struct Glm5NextMtpModule {
    /// 2026-09-25: The block as a `mhc: None` layer: DSA mixer, routed MoE, plain residual.
    pub layer: Glm5NextLayer,
    /// 2026-09-25: `[hidden, 2 * hidden]`: projects `concat(enorm(embed), hnorm(hidden))`
    /// back to `hidden` (`glm5next_mtp_head.rs`).
    pub eh_proj: DenseWeight,
    /// 2026-09-25: RMSNorm weights for the embedding half and the target-hidden half of the
    /// concat.
    pub enorm: DevicePtr,
    pub hnorm: DevicePtr,
    /// 2026-09-25: `shared_head.norm`, the final norm before the target's `lm_head`.
    pub final_norm: DevicePtr,
}

/// 2026-09-25: Build the MTP block, or `None` when the checkpoint has no
/// `layers.{num_hidden_layers}`.
///
/// Runs on every rank: `build_moe` binds only this rank's `local_expert_range()`, so each
/// rank holds its own share of the routed experts, as in the text stack. 2026-10-09: Under
/// `--moe-expert-layout tp`, its slice of every expert.
pub fn load_glm5next_mtp_module(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    lc: metrale_model_layers::load_context::LoadContext<'_>,
) -> Result<Option<Glm5NextMtpModule>> {
    let idx = config.num_hidden_layers;
    let prefix = format!("model.language_model.layers.{idx}.");
    if !store.names().any(|n| n.starts_with(&prefix)) {
        return Ok(None);
    }
    let src = super::glm5_next_load::layer_source(gpu, store, idx)
        .with_context(|| format!("glm5_next MTP: collecting layer {idx}"))?;
    let load = |n: &str| src.f32(n);

    // 2026-10-13: The drafter block's context is capped (`METRALE_GLM_MTP_CONTEXT_CAP`), so its
    // indexer cache, workspace and ceiling grids are sized at the cap, not `--max-seq-len`.
    // 2026-10-17: The GLM levers, resolved from the same settings as the text stack's.
    let levers = crate::glm5next_layer::GlmLevers::from_settings(lc.settings)?;
    levers.dense.check_tiers(
        crate::glm5next_w4a16_dense::enabled(),
        crate::glm5next_fp8_dense::enabled(),
    )?;
    let mtp = crate::glm5next_mtp_head::MtpLevers::from_settings(lc.settings)?;
    let dsa_cfg = crate::glm5next_mtp_head::drafter_dsa_config(config, mtp.context_cap)?;
    let mlp_cfg = Glm5NextMlpConfig::from_config(config)?;
    let dsa_plan = crate::glm5next_dsa::tp::DsaTpPlan::from_config(config, &dsa_cfg)?;
    let dsa_layer_kernels = Glm5NextDsaLayerKernels::resolve(gpu)?;
    let dsa_kernels = Glm5NextDsaKernels::resolve_for(gpu, lc.settings)?;
    let drafter_format = crate::glm5next_dsa::fp8_rows::IndexerRowFormat::drafter(
        lc.settings.kernels.glm_dsa_pool_keys,
        dsa_kernels.rows.pool_keys_ready(),
    )?;
    let mlp_kernels = Glm5NextMlpKernels::resolve(gpu, levers)?;

    let mut mixer = Glm5NextMixer::Dsa(Box::new(Glm5NextDsaLayer {
        persist_bt: !levers.dsa_alloc_per_step,
        levers,
        cfg: dsa_cfg,
        weights: build_dsa_weights(
            gpu,
            &dsa_cfg,
            &dsa_plan,
            &load,
            crate::mla_absorb::AbsorbThreads::from_settings(lc.settings),
        )?,
        kernels: dsa_layer_kernels,
        select_kernels: dsa_kernels,
        decode_kernel: crate::glm5next_dsa::attend::Glm5NextDsaDecodeKernel::resolve(
            gpu,
            levers.dsa.decode_hb,
        )?,
        dsa_owned: std::sync::Arc::new(crate::glm5next_dsa::levers::DsaOwnedLevers::from_settings(
            lc.settings,
        )?),
        // 2026-09-25: Single-row workspace: the drafter runs this layer one row at a time.
        // 2026-10-11: One row per sequence of the widest batched propose under its lever.
        // 2026-10-11: And a pass of drafter context rows (`ctx_rows`).
        workspace: std::sync::Arc::new(Glm5NextDsaWorkspace::new(
            gpu,
            &dsa_cfg,
            mtp.dsa_rows(),
            &levers,
        )?),
        layer_idx: idx,
        // 2026-09-25: Sole consumer of its own one-layer KV pool.
        attn_layer_idx: 0,
        rms_eps: config.rms_norm_eps as f32,
        kv_scale: 1.0,
        // 2026-10-09: The drafter's one-layer pool is sized apart from the text stack's; its
        // indexer stays flat. 2026-10-21: With persisted pool keys under
        // `METRALE_GLM_DSA_POOL_KEYS` (`IndexerRowFormat::drafter`).
        indexer_cache: crate::glm5next_dsa::paged::IndexerCache::Flat,
        indexer_format: drafter_format,
    }));

    let expert = super::glm5_next_load::expert_binder_at(gpu, store, idx, &mlp_cfg);
    // 2026-10-08: The MTP layer's plan, as for a text layer; this checkpoint declares the layer
    // 16-bit, so its experts (quantized at load) run W4A16. One row: its workspace is one row.
    let precision = |has_scales: bool| {
        super::glm5_next_load::mtp_expert_precision(config, &mlp_kernels, idx, &mlp_cfg, has_scales)
    };
    let mut mlp = Glm5NextMlpSite::Moe(Box::new(mlp_build::build_moe(
        gpu,
        &mlp_cfg,
        config.shared_expert_intermediate_size,
        &load,
        &expert,
        &precision,
        mtp.block_rows(),
    )?));
    // 2026-10-11: The drafter's dense projections at the serve's tiers, as a text layer's. They
    // were left BF16: at TP=3 the drafter's DSA q_absorb and o_absorb alone read ~0.54 ms per
    // propose in BF16. Draft tokens only; the target verifies every draft.
    super::glm5_next_load::register_mtp_dense_tiers(
        gpu,
        &mut mixer,
        &mut mlp,
        &mlp_cfg,
        &levers.dense,
        idx,
    )
    .with_context(|| format!("glm5_next MTP: dense tiers of layer {idx}"))?;

    let up =
        |n: &str| -> Result<DevicePtr> { super::glm5_next_load::upload_bf16(gpu, &src.f32(n)?) };
    Ok(Some(Glm5NextMtpModule {
        layer: Glm5NextLayer {
            layer_idx: idx,
            mixer,
            mlp,
            mlp_cfg,
            mlp_kernels,
            // 2026-09-25: Its own one-row workspace, not the text stack's shared one, which
            // is sized for a prefill sub-chunk.
            // 2026-10-11: As many rows as the DSA workspace above.
            mlp_ws: std::sync::Arc::new(crate::glm5next_mlp::forward::Glm5NextMlpWorkspace::new(
                gpu,
                &mlp_cfg,
                mtp.block_rows(),
                &levers.mlp,
            )?),
            // 2026-09-25: No hyper-connection: nothing here binds `hc_*` tensors, and
            // `mhc: None` selects the plain residual path.
            mhc: None,
            input_norm: up("input_layernorm.weight")?,
            post_attn_norm: up("post_attention_layernorm.weight")?,
            rms_norm_k: gpu.kernel("rms_norm_vanilla", "rms_norm_vanilla")?,
            add_k: metrale_model_layers::layers::try_kernel(gpu, "bf16_add", "bf16_add_inplace"),
            rms_eps: config.rms_norm_eps as f32,
            hidden: config.hidden_size,
            // 2026-09-25: Row-parallel `o_proj`, as in the DSA text layers.
            mixer_all_reduce: dsa_plan.needs_output_all_reduce(),
            // 2026-09-25: No mHC highway to expand or collapse.
            is_first: false,
            is_last: false,
            // 2026-10-10: The MTP block has no mHC reduce to hide a warm under.
            l2_warm: None,
            keep_slot_graphs: lc.levers.glm_keep_slot_graphs,
            dump: crate::glm5next_layer_dump::config(lc.settings)?.map(std::sync::Arc::new),
            levers,
        },
        eh_proj: DenseWeight {
            weight: store.get(&format!("{prefix}eh_proj.weight"))?.ptr,
        },
        enorm: up("enorm.weight")?,
        hnorm: up("hnorm.weight")?,
        final_norm: up("shared_head.norm.weight")?,
    }))
}
