// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The NLLB / M2M-100 branch of `build_model`, moved out of `build.rs` unchanged.
//!
//! Owner: metrale-model-engine.
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_weights::weights::WeightStore;

use super::slots;

/// 2026-10-17: True for the encoder-decoder model types [`build`] serves.
pub(super) fn is_nllb(config: &ModelConfig) -> bool {
    matches!(config.model_type.as_str(), "m2m_100" | "nllb")
}

/// 2026-09-25: NLLB / M2M-100 is an encoder-decoder model, served by `NllbGpuModel` from the
/// same `store`. `build_model` returns this before `loader_for_config`, so the decoder-only
/// loader never runs for it.
pub(super) fn build(
    config: &ModelConfig,
    store: &WeightStore,
    gpu: Box<dyn GpuBackend>,
    nllb_lang: Option<(u32, u32)>,
    max_seq_len: usize,
    request: &slots::SlotRequest,
    nllb_lora_dir: Option<&std::path::Path>,
    settings: &metrale_settings::InferenceServerSettings,
) -> Result<slots::BuiltModel> {
    let (src, tgt) = nllb_lang.ok_or_else(|| {
        anyhow::anyhow!(
            "NLLB serving requires --src-lang and --tgt-lang (translation language pair)"
        )
    })?;
    let lang = crate::model::nllb::NllbLang {
        src_lang_id: src,
        tgt_lang_id: tgt,
        decoder_start_id: config.eos_token_id,
        eos_id: config.eos_token_id,
        pad_id: 1,
    };
    let nllb_slots = match request {
        slots::SlotRequest::Count(n) => *n,
        slots::SlotRequest::Auto => anyhow::bail!(
            "--max-batch-size auto balances recurrent-state slots against the KV pool; \
             an encoder-decoder model has neither. Pass a count."
        ),
    };
    let model = crate::model::nllb::NllbGpuModel::new(
        config,
        store,
        gpu,
        lang,
        max_seq_len,
        nllb_slots,
        nllb_lora_dir,
        metrale_model_layers::layers::ops::OpsLevers::from_settings(settings)?.lora,
        crate::model::nllb::NllbLevers::from_settings(settings),
    )?;
    Ok(slots::BuiltModel {
        max_batch_size: nllb_slots,
        model: Box::new(model),
        drafter_weights_sha256: None,
    })
}
