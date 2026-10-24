// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-24: The embedding-model branch of `build_model`: a config whose capabilities say it
//! does not generate is served by `EmbedGpuModel` from the same store, before the decoder-only
//! loaders run (as the NLLB branch).
//!
//! Owner: metrale-model-engine.
//! Invariants: none beyond the types.
#![cfg(feature = "cuda")]

use anyhow::Result;
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_weights::weights::WeightStore;

use super::slots;

/// 2026-10-24: True for a config that pools inputs into embeddings.
pub(super) fn is_embedding(config: &ModelConfig) -> bool {
    !config.capabilities().generates
}

/// 2026-10-24: Build the embedding model. It holds no sequences, so the slot request is
/// refused only in the form that needs a KV pool to resolve (`auto`).
pub(super) fn build(
    config: &ModelConfig,
    store: WeightStore,
    gpu: Box<dyn GpuBackend>,
    max_batch_tokens: usize,
    request: &slots::SlotRequest,
) -> Result<slots::BuiltModel> {
    let max_batch_size = match request {
        slots::SlotRequest::Count(n) => *n,
        slots::SlotRequest::Auto => anyhow::bail!(
            "--max-batch-size auto balances sequence slots against the KV pool; an embedding \
             model has neither. Pass a count."
        ),
    };
    let model = crate::model::embed::EmbedGpuModel::new(config, store, gpu, max_batch_tokens)?;
    Ok(slots::BuiltModel {
        max_batch_size,
        model: Box::new(model),
        drafter_weights_sha256: None,
    })
}
