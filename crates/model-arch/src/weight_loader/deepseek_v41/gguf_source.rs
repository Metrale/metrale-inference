// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-10-19: [`V41Source`] over a GGUF checkpoint: every per-layer tensor is already in the
//! weight store, whole (the GGUF path serves one rank).
//!
//! Owner: model-arch weight loader.
//! Invariants: none beyond the types.

use std::sync::Arc;

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::layers::ops::ResidentMat;
use metrale_model_weights::weights::WeightStore;
use metrale_model_weights::weights::expert_stream::ShardFiles;

use super::source::{EngramWkv, Format, T, V41Source, name};
use super::{bf16_ptr, download_f32, resident_mat};

pub struct GgufSource<'a> {
    pub store: &'a WeightStore,
    pub files: Arc<ShardFiles>,
    /// 2026-10-19: `METRALE_DS41_ENGRAM_Q2K`.
    pub engram_q2k: bool,
}

impl GgufSource<'_> {
    fn n(&self, l: usize, t: T) -> String {
        name(Format::Gguf, l, t)
    }
}

impl V41Source for GgufSource<'_> {
    fn has(&self, l: usize, t: T) -> bool {
        self.store.contains(&self.n(l, t))
    }

    fn mat(&self, _gpu: &dyn GpuBackend, l: usize, t: T) -> Result<ResidentMat> {
        resident_mat(self.store, &self.n(l, t))
    }

    fn bf16(&self, _gpu: &dyn GpuBackend, l: usize, t: T) -> Result<DevicePtr> {
        bf16_ptr(self.store, &self.n(l, t))
    }

    fn host_f32(&self, gpu: &dyn GpuBackend, l: usize, t: T) -> Result<Vec<f32>> {
        download_f32(gpu, self.store, &self.n(l, t))
    }

    fn engram_wkv(&self, gpu: &dyn GpuBackend, l: usize) -> Result<EngramWkv> {
        Ok(EngramWkv {
            wkv: ResidentMat::Bf16(self.bf16(gpu, l, T::EngramWkv)?),
            wkv_q2k: super::engram_q2k::engram_wkv_q2k(gpu, &self.files, l, self.engram_q2k)?,
        })
    }
}
