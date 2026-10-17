// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The proposers `build_model` installs after
//! `TransformerModel::new` (DeepSeek-V4 MTP, GLM-5.3 MTP, DFlash), and the
//! allocation report it logs last.
//!
//! Owner: metrale-model-engine.
//! Invariants:
//! - A DeepSeek-V4 or GLM-5.3 proposer build error is logged and not returned;
//!   a DFlash drafter error is returned.

use anyhow::{Context, Result};
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_arch::weight_loader::deepseek_v4::mtp::DeepseekV4MtpModule;
use metrale_model_arch::weight_loader::glm5_next_mtp::Glm5NextMtpModule;
use metrale_model_arch::weight_loader::load_dflash_weights;
use metrale_model_layers::weight_map::{DenseWeight, Fp8DenseWeight, QuantizedWeight};

use crate::factory::DflashBuildArgs;
use crate::model::TransformerModel;

/// 2026-09-26: Installs the DeepSeek-V4 MTP proposer when the module loaded.
pub(super) fn install_v4_mtp_proposer(
    model: &mut TransformerModel,
    v4_mtp_module: Option<DeepseekV4MtpModule>,
    v4_mtp_embed: DenseWeight,
    v4_mtp_lm_head: DenseWeight,
    mtp_vocab_size: u32,
    max_seq_len: usize,
) {
    if let Some(v4_module) = v4_mtp_module {
        match metrale_model_arch::deepseek_v4_mtp::DeepseekV4MtpHead::new(
            v4_module,
            v4_mtp_embed,
            v4_mtp_lm_head,
            model.config_ref(),
            model.gpu_backend(),
            mtp_vocab_size,
            max_seq_len,
        ) {
            Ok(head) => {
                model.set_dflash_proposer(std::sync::Arc::new(head));
                tracing::info!(target: "metrale_model_engine::factory::build", "DeepSeek-V4 MTP speculative decoding: ENABLED (single-module)");
            }
            Err(e) => {
                tracing::warn!(target: "metrale_model_engine::factory::build", "Failed to build DeepSeek-V4 MTP proposer: {e:#}. Speculative decoding disabled."
                )
            }
        }
    }
}

/// 2026-09-26: Installs the GLM-5.3 MTP proposer when the module loaded.
pub(super) fn install_glm_mtp_proposer(
    model: &mut TransformerModel,
    glm_mtp_module: Option<Glm5NextMtpModule>,
    glm_mtp_embed: DenseWeight,
    glm_mtp_lm_head: DenseWeight,
    // 2026-10-11: The target's final norm, for the drafter's `METRALE_MTP_TARGET_POSTNORM` feed.
    glm_mtp_target_final_norm: DenseWeight,
    max_seq_len: usize,
    max_batch_size: usize,
) {
    if let Some(m) = glm_mtp_module {
        match metrale_model_arch::glm5next_mtp_head::Glm5NextMtpHead::new(
            m,
            glm_mtp_embed,
            glm_mtp_lm_head,
            glm_mtp_target_final_norm,
            model.config_ref(),
            model.gpu_backend(),
            max_seq_len,
            max_batch_size,
        ) {
            Ok(head) => {
                model.set_dflash_proposer(std::sync::Arc::new(head));
                tracing::info!(target: "metrale_model_engine::factory::build", "GLM-5.3 MTP speculative decoding: ENABLED");
                // 2026-10-11: The proposer is installed after construction, which built no
                // batched-verify stash, so `can_batch_verify` refused every MTP batch and each
                // sequence verified alone at C>=2. A failure keeps the per-sequence verify.
                if let Err(e) = model.ensure_verify_hidden_stash() {
                    tracing::warn!(target: "metrale_model_engine::factory::build", "GLM-5.3 MTP: no batched-verify stash ({e:#}); MTP sequences verify one at a time");
                }
            }
            Err(e) => {
                tracing::warn!(target: "metrale_model_engine::factory::build", "Failed to build GLM-5.3 MTP proposer: {e:#}. Speculative decoding disabled."
                )
            }
        }
    }
}

/// 2026-09-26: Installs the DFlash drafter when `dflash_args` is given and its
/// store has DFlash weights.
pub(super) fn install_dflash_drafter(
    model: &mut TransformerModel,
    dflash_args: Option<DflashBuildArgs<'_>>,
    target_embed_for_dflash: DevicePtr,
    target_lm_head_for_dflash: DevicePtr,
    target_lm_head_nvfp4_for_dflash: Option<QuantizedWeight>,
    target_lm_head_native_fp8_for_dflash: Option<(Fp8DenseWeight, usize)>,
    target_hidden_for_dflash: usize,
    max_seq_len: usize,
    max_batch_size: usize,
    settings: &metrale_settings::InferenceServerSettings,
) -> Result<()> {
    if let Some(args) = dflash_args {
        let weights = load_dflash_weights(
            args.drafter_store,
            &args.drafter_config,
            model.gpu_backend(),
            1,
        )?;
        if let Some(weights) = weights {
            let (weights, lm_head, lm_nvfp4, lm_fp8, tp) = dflash_tp_split(
                model,
                args.tp_size,
                weights,
                target_lm_head_for_dflash,
                target_lm_head_nvfp4_for_dflash,
                target_lm_head_native_fp8_for_dflash,
            )?;
            let head = metrale_model_arch::dflash_head::BlockDiffusionDraftHead::from_weights(
                weights,
                target_embed_for_dflash,
                lm_head,
                lm_nvfp4,
                lm_fp8,
                target_hidden_for_dflash,
                args.gamma,
                // 2026-10-09: The trained window when the flag states it, else none
                // (`DflashConfig::attention_window`); every other value was ignored before.
                args.drafter_config.attention_window(args.window_size),
                model.gpu_backend(),
                max_seq_len,
                max_batch_size,
                tp,
                args.weight_quant,
                settings,
            )?;
            // 2026-10-10: The tensor-parallel propose runs the paged DFlash2 path only.
            anyhow::ensure!(
                tp.is_none() || (head.dflash2_active() && head.levers.option_b),
                "--draft-tp-size above 1 needs a DFlash2 drafter (conv + selector) on the paged \
                 cache (METRALE_DFLASH_OPTION_B not 0)"
            );
            model.set_dflash_proposer(std::sync::Arc::new(head));
            model.ensure_verify_hidden_stash()?;
            tracing::info!(target: "metrale_model_engine::factory::build", "DFlash drafter installed as the active proposer");
        } else {
            tracing::warn!(target: "metrale_model_engine::factory::build", "DFlash drafter store had no fc.weight — proposer not installed; \
                 falling back to whatever proposer (if any) the target's MTP path built"
            );
        }
    }
    Ok(())
}

/// 2026-10-10: The drafter inputs of this rank: with `--draft-tp-size` above 1 (`tp_size`, the
/// world size), the weights cut to this rank's part (`dflash_head::tp`), the LM head pointer at its
/// first vocab row and no NVFP4 or native FP8 head share (the head's copy is quantized from those
/// rows); otherwise everything unchanged. A target whose LM head is NVFP4 is refused under the
/// lever: the row offset assumes BF16 rows.
#[allow(clippy::type_complexity)]
fn dflash_tp_split(
    model: &TransformerModel,
    tp_size: usize,
    weights: metrale_model_arch::weight_loader::DflashWeights,
    lm_head: DevicePtr,
    lm_nvfp4: Option<QuantizedWeight>,
    lm_fp8: Option<(Fp8DenseWeight, usize)>,
) -> Result<(
    metrale_model_arch::weight_loader::DflashWeights,
    DevicePtr,
    Option<QuantizedWeight>,
    Option<(Fp8DenseWeight, usize)>,
    Option<metrale_model_arch::dflash_head::tp::DflashTp>,
)> {
    // 2026-10-10: Every rank must split the same way, or rank 0's collectives would have no
    // partner: the ranks agree on the size before anything is cut.
    if let Some(comm) = model.comm_ref() {
        crate::rank_agree::assert_ranks_agree(
            model.gpu_backend(),
            comm,
            &[("--draft-tp-size", tp_size as u64)],
        )?;
    }
    if tp_size <= 1 {
        return Ok((weights, lm_head, lm_nvfp4, lm_fp8, None));
    }
    let (rank, world) = model
        .comm_rank_world()
        .context("--draft-tp-size above 1 needs a multi-rank serve")?;
    anyhow::ensure!(
        world == tp_size,
        "--draft-tp-size {tp_size} on a {world}-rank communicator: it must be the world size"
    );
    anyhow::ensure!(
        lm_nvfp4.is_none(),
        "--draft-tp-size {tp_size} needs the target's BF16 LM head; this target's is NVFP4"
    );
    let c = &weights.config;
    let plan = metrale_model_arch::dflash_head::tp::plan(
        c.num_attention_heads,
        c.num_key_value_heads,
        c.intermediate_size,
        c.vocab_size,
        world,
        rank,
    )?;
    tracing::info!(
        target: "metrale_model_engine::factory::build",
        "DFlash TP: rank {rank}/{world}: q heads {:?}, KV heads {:?}, MLP {:?}, vocab {:?}",
        plan.q_heads,
        plan.kv_heads,
        plan.inter,
        plan.vocab
    );
    let (weights, lm_head) = metrale_model_arch::dflash_head::tp::shard_weights(
        weights,
        &plan,
        lm_head,
        model.gpu_backend(),
    )?;
    Ok((
        weights,
        lm_head,
        None,
        None,
        Some(metrale_model_arch::dflash_head::tp::DflashTp {
            rank,
            world,
            vocab_base: plan.vocab.0,
        }),
    ))
}

/// 2026-09-26: Logs the backend's allocation report and whether tracked-live
/// bytes stayed within `total_budget`.
pub(super) fn log_alloc_report(
    model: &TransformerModel,
    total_budget: usize,
    gpu_memory_utilization: f64,
    total_mem: usize,
    gib: impl Fn(usize) -> f64,
) {
    // 2026-09-25: The backend's allocation report, logged at INFO once, after
    // every build allocation.
    if let Some(report) = model.gpu_backend().alloc_report(12, 64) {
        for line in report.lines() {
            tracing::info!(target: "metrale_model_engine::factory::build", "{line}");
        }
    }
    // 2026-10-02: A diagnostic: its failure is logged, never fatal to the serve.
    if let Err(e) = super::alloc_digest::write_if_requested(model.gpu_backend()) {
        tracing::warn!("METRALE_DEBUG_ALLOC_DIGEST: {e:#}");
    }
    // 2026-09-25: Tracked-live bytes above the util budget mean an allocation
    // the sizing above did not reserve for. Logged as a warning; the build
    // still succeeds.
    if let Some(live) = model.gpu_backend().live_bytes() {
        if live > total_budget {
            tracing::warn!(target: "metrale_model_engine::factory::build", "util pledge exceeded: {:.1} GB tracked live vs {:.1} GB pledged \
                 (--gpu-memory-utilization {:.0}% of {:.1} GB) — an allocation \
                 family above is missing from the preflight reserve",
                gib(live),
                gib(total_budget),
                gpu_memory_utilization * 100.0,
                gib(total_mem),
            );
        } else {
            tracing::info!(target: "metrale_model_engine::factory::build", "util pledge honored: {:.1} GB tracked live within the {:.1} GB \
                 budget ({:.1} GB pledge headroom)",
                gib(live),
                gib(total_budget),
                gib(total_budget - live),
            );
        }
    }
}
