// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Marker loader for a model this decoder-only loader interface cannot build:
//! NLLB / M2M-100 (`model_type` `m2m_100` or `nllb`), an encoder-decoder, and (2026-10-24)
//! an embedding model (`llama_nemotron_vl`), a bidirectional encoder that pools.
//!
//! In a `cuda` build, model-engine's `build_model` (`factory/build.rs`) serves these model
//! types with their dedicated runtime (`NllbGpuModel`, `EmbedGpuModel`) and returns before
//! `loader_for_config`'s loader is used; serve still asks the loader for its declarations
//! (TP support, expert tables), which keep their defaults here.
//!
//! Owner: model-arch weight loader.
//! Invariants:
//! - `load_layers`, `load_embedding`, `load_final_norm` and `load_lm_head` always return an
//!   error; `load_mtp_weights` returns `Ok(None)` and `tp_support` is `Unsupported`.

use anyhow::{Result, bail};
use metrale_cache::kv_cache::KvCacheDtype;
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_weights::weights::WeightStore;

use crate::weight_loader::ModelWeightLoader;
use metrale_model_layers::layer::TransformerLayer;
use metrale_model_layers::weight_map::{DenseWeight, MtpWeights};

/// 2026-10-24: Which model the marker stands for, and the runtime that serves it.
pub struct DedicatedRuntimeLoader {
    family: &'static str,
    kind: &'static str,
    runtime: &'static str,
}

impl DedicatedRuntimeLoader {
    pub const NLLB: Self = Self {
        family: "NLLB / m2m_100",
        kind: "encoder-decoder",
        runtime: "metrale_model_engine::model::nllb::NllbGpuModel",
    };
    pub const EMBEDDING: Self = Self {
        family: "llama_nemotron_vl",
        kind: "embedding",
        runtime: "metrale_model_engine::model::embed::EmbedGpuModel",
    };

    fn unsupported(&self) -> anyhow::Error {
        anyhow::anyhow!(
            "{} is served by the dedicated GPU {} runtime ({}), which build_model selects before this loader; the generic decoder-only ModelWeightLoader pipeline cannot serve it. Reaching this loader means the dedicated serve path was bypassed.",
            self.family,
            self.kind,
            self.runtime
        )
    }
}

impl ModelWeightLoader for DedicatedRuntimeLoader {
    fn tp_support(&self) -> metrale_config::TpSupport {
        metrale_config::TpSupport::Unsupported
    }

    fn load_layers(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
        _layer_kv_dtypes: &[KvCacheDtype],
        _lc: metrale_model_layers::load_context::LoadContext<'_>,
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        bail!(self.unsupported())
    }

    fn load_embedding(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        bail!(self.unsupported())
    }

    fn load_final_norm(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        bail!(self.unsupported())
    }

    fn load_lm_head(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        bail!(self.unsupported())
    }

    fn load_mtp_weights(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<Option<MtpWeights>> {
        Ok(None)
    }
}
