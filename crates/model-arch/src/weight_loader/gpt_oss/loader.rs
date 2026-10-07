// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Explicit C1 GPT loader, intentionally not registered in factory.
use super::{GptOssCheckpoint, runtime::GptOssLayer};
use crate::weight_loader::ModelWeightLoader;
use anyhow::{Result, ensure};
use metrale_cache::kv_cache::KvCacheDtype;
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_layers::{
    layer::TransformerLayer,
    weight_map::{DenseWeight, MtpWeights},
};
use metrale_model_weights::weights::WeightStore;

/// 2026-10-07: Native eager single-device route; support remains unregistered.
pub struct GptOssWeightLoader {
    pub chunk_prefill: bool,
}
impl ModelWeightLoader for GptOssWeightLoader {
    fn supports_tp(&self) -> bool {
        false
    }
    fn load_layers(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        ensure!(
            dtypes.len() == 24 && dtypes.iter().all(|d| *d == KvCacheDtype::Bf16),
            "GPT loader requires24 BF16 KV layers"
        );
        let checkpoint = GptOssCheckpoint::bind(store, config)?;
        checkpoint
            .layers
            .iter()
            .enumerate()
            .map(|(index, weights)| {
                let mut layer = GptOssLayer::new(weights, config, index, gpu)?;
                layer.set_chunk_prefill(self.chunk_prefill, gpu)?;
                Ok(Box::new(layer) as Box<dyn TransformerLayer>)
            })
            .collect()
    }
    fn load_embedding(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        Ok(DenseWeight {
            weight: GptOssCheckpoint::bind(store, config)?.embedding.ptr(),
        })
    }
    fn load_final_norm(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        Ok(DenseWeight {
            weight: GptOssCheckpoint::bind(store, config)?.final_norm.ptr(),
        })
    }
    fn load_lm_head(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        Ok(DenseWeight {
            weight: GptOssCheckpoint::bind(store, config)?.head.ptr(),
        })
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
