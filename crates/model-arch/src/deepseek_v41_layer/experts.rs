// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-10-19: Where a DeepSeek-V4.1 runtime's routed experts and engram rows come from, chosen
//! once at load by the checkpoint format:
//!
//! - GGUF: experts streamed through the expert cache (`ExpertLru` over `ExpertArena`, located by
//!   `ExpertSliceMap`), Q2_K engram rows (`EngramRowReader`), dequantized on the device.
//! - Safetensors: every routed expert resident in device memory, sliced to this rank's
//!   intermediate columns ([`ResidentMoe`], the tensor-parallel path), and E4M3 engram rows with
//!   their E8M0 scales (`Fp8EngramRows`), decoded on the host. An E4M3 value times a power of
//!   two is exact in bf16, so the decode is exact.
//!
//! The captured-step paths (`graph.rs`, `step_seg*.rs`) and the routing-prediction trace drive
//! the expert cache, so they ask for [`V41Runtime::streamed`] and fail on the resident path;
//! the loader refuses those levers there before the first step.
//!
//! Owner: model-arch, DeepSeek-V4.1.
//! Invariants:
//! - The FP8 row decode refuses the E8M0 NaN code (`deepseek_v41_ref::mat::e8m0`).

use std::sync::{Arc, Mutex};

use anyhow::{Result, bail, ensure};
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_weights::weights::Fp8EngramRows;
use metrale_model_weights::weights::expert_stream::{
    EngramRowReader, ExpertArena, ExpertLru, ExpertSliceMap,
};

use super::V41Runtime;
use crate::deepseek_v41_ref::mat::{e4m3, e8m0};
use crate::engram_v41::ENGRAM_ROW_BYTES;
use crate::moe_v41::resident::{ResidentExperts, ResidentMoe};

/// 2026-10-19: The GGUF expert cache.
pub struct StreamedExperts {
    pub lru: Mutex<ExpertLru>,
    pub arena: ExpertArena,
    pub slices: Arc<ExpertSliceMap>,
}

/// 2026-10-19: Every layer's resident routed experts (`layers[layer]`) and the shared
/// tensor-parallel MoE scratch.
pub struct ResidentSet {
    pub moe: ResidentMoe,
    pub layers: Vec<ResidentExperts>,
}

pub enum V41Experts {
    Streamed(StreamedExperts),
    Resident(ResidentSet),
}

pub enum V41EngramRows {
    Q2k(EngramRowReader),
    Fp8(Fp8EngramRows),
}

impl V41Runtime {
    /// 2026-10-19: The expert cache, for the paths that drive it. Fails on the resident path.
    pub fn streamed(&self) -> Result<&StreamedExperts> {
        match &self.experts {
            V41Experts::Streamed(s) => Ok(s),
            V41Experts::Resident(_) => {
                bail!(
                    "deepseek-v4.1: this path drives the GGUF expert cache; the experts are resident"
                )
            }
        }
    }

    /// 2026-10-19: Read rows `row_ids` of engram layer `layer` and fill that layer's row
    /// workspace with them, enqueued on `stream`.
    pub fn fill_engram_rows(
        &self,
        gpu: &dyn GpuBackend,
        layer: usize,
        row_ids: &[u64],
        stream: u64,
    ) -> Result<()> {
        let engram = self.engram.lock().unwrap();
        let n = row_ids.len();
        match &self.rows {
            V41EngramRows::Q2k(rd) => {
                let mut raw = vec![0u8; n * ENGRAM_ROW_BYTES];
                rd.read_rows(layer, row_ids, &mut raw)?;
                engram.rows_from_q2k(gpu, Some(layer), &raw, n, stream)
            }
            V41EngramRows::Fp8(rd) => {
                let t = rd
                    .table(layer)
                    .ok_or_else(|| anyhow::anyhow!("layer {layer} has no engram table"))?;
                ensure!(
                    t.head_dim == engram.head_dim,
                    "engram layer {layer}: table head_dim {} for a {}-wide engram",
                    t.head_dim,
                    engram.head_dim
                );
                let mut codes = vec![0u8; n * t.head_dim];
                let mut scales = vec![0u8; n * t.scale_cols];
                rd.read_rows(layer, row_ids, &mut codes, &mut scales)?;
                let bits = decode_fp8_rows(&codes, &scales, t.head_dim, t.scale_cols);
                engram.rows_from_bf16(gpu, Some(layer), &bits, n, stream)
            }
        }
    }
}

/// 2026-10-19: E4M3 `codes` (`[rows, head_dim]`) times their E8M0 `scales`
/// (`[rows, scale_cols]`, one per `head_dim / scale_cols` values) as bf16 bits, rounded to
/// nearest even (exact for every finite scale whose product is a bf16 normal).
pub fn decode_fp8_rows(
    codes: &[u8],
    scales: &[u8],
    head_dim: usize,
    scale_cols: usize,
) -> Vec<u16> {
    let blk = head_dim / scale_cols;
    codes
        .iter()
        .enumerate()
        .map(|(i, &c)| {
            let (r, d) = (i / head_dim, i % head_dim);
            let v = e4m3(c) * e8m0(scales[r * scale_cols + d / blk]);
            half::bf16::from_f32(v).to_bits()
        })
        .collect()
}

#[cfg(test)]
#[path = "experts_tests.rs"]
mod tests;
