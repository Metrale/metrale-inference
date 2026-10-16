// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: The model side of `METRALE_GLM_PREFILL_CELL_GRID` (`glm5next_prefill_cell`):
//! the clamps of the grid serve's chunk ends the cell serve has no chunk end for, and the
//! read-only SSM state norm readback (`METRALE_GLM_SSM_NORM_READBACK=1`).
//!
//! - On a prefix hit restoring the state at `u`, before the first computed row, the restored
//!   state takes `restored_clamps` more clamps (the grid serve's fully cached chunks clamp it
//!   once each).
//! - Rank 0 clamps after its chunk 0 when that chunk ends past the grid's first point, where the
//!   grid serve's chunk 1 ends (`rank0_first_chunk_clamp`).
//! - The readback, before every clamp, copies each SSM layer's FP32 state to the host and
//!   prints the largest per-head Frobenius norm ([`norm_max`]); it never writes the device.
//!
//! Owner: model-engine.
//! Invariants:
//! - Lever unset, no extra clamp runs. Readback unset, no copy runs.
//! - Every extra clamp is the scheduler's own call (`normalize_ssm_states`), on its stream.

use std::sync::OnceLock;

use anyhow::Result;
use metrale_model_arch::glm5next_prefill_cell as cell;

use crate::model::types::TransformerModel;
use crate::traits::{ModelForward, SequenceState};

/// 2026-10-15: The readback lever's name.
pub const SSM_NORM_READBACK_ENV: &str = "METRALE_GLM_SSM_NORM_READBACK";

/// 2026-10-15: The clamp's per-head bound, `MAX_NORM` of `ssm_state_norm.cu`.
const CLAMP_NORM: f64 = 200.0;

/// 2026-10-15: [`SSM_NORM_READBACK_ENV`], read once: unset or `0` off, `1` on, else an error.
pub fn ssm_norm_readback() -> Result<bool> {
    static V: OnceLock<Result<bool, String>> = OnceLock::new();
    V.get_or_init(
        || match std::env::var(SSM_NORM_READBACK_ENV).ok().as_deref() {
            None | Some("0") => Ok(false),
            Some("1") => Ok(true),
            Some(v) => Err(format!("{SSM_NORM_READBACK_ENV}={v:?}: expected 0 or 1")),
        },
    )
    .clone()
    .map_err(anyhow::Error::msg)
}

/// 2026-10-15: The largest per-head Frobenius norm over every layer, where it is, how many
/// (layer, head) blocks exceed the clamp bound, and each layer's largest.
#[derive(Clone, Debug, PartialEq)]
pub struct NormMax {
    pub max: f64,
    pub layer: usize,
    pub head: usize,
    pub over: usize,
    pub per_layer: Vec<f64>,
}

/// 2026-10-15: [`NormMax`] of `layers`, each `heads` blocks of `k * v` FP32 values (the
/// clamp's `[heads, k, v]` layout); the norm in f64. Errors on a layer of another size. Pure.
pub fn norm_max(layers: &[Vec<f32>], heads: usize, k: usize, v: usize) -> Result<NormMax> {
    let block = k * v;
    let mut out = NormMax {
        max: 0.0,
        layer: 0,
        head: 0,
        over: 0,
        per_layer: Vec::with_capacity(layers.len()),
    };
    for (l, h) in layers.iter().enumerate() {
        anyhow::ensure!(
            h.len() == heads * block,
            "SSM layer {l}: {} values, expected {heads} x {k} x {v}",
            h.len()
        );
        let mut layer_max = 0.0f64;
        for (hd, b) in h.chunks_exact(block).enumerate() {
            let n = b
                .iter()
                .map(|&x| f64::from(x) * f64::from(x))
                .sum::<f64>()
                .sqrt();
            if n > CLAMP_NORM {
                out.over += 1;
            }
            if n > out.max {
                (out.max, out.layer, out.head) = (n, l, hd);
            }
            layer_max = layer_max.max(n);
        }
        out.per_layer.push(layer_max);
    }
    Ok(out)
}

/// 2026-10-15: The readback line.
pub fn norm_line(rank: usize, seq_len: usize, m: &NormMax) -> String {
    let per: Vec<String> = m.per_layer.iter().map(|x| format!("{x:.3}")).collect();
    format!(
        "SSM_NORM_MAX rank={rank} seq_len={seq_len} max={:.3} layer={} head={} over200={} per_layer={}",
        m.max,
        m.layer,
        m.head,
        m.over,
        per.join(",")
    )
}

impl TransformerModel {
    fn cell_rank(&self) -> usize {
        self.comm_ref().map_or(0, |c| c.rank())
    }

    /// 2026-10-15: Before the first computed row from `compute_start`: when the pass resumes a
    /// state restored at `compute_start`, the clamps the grid serve's cached chunks would have
    /// run on it beyond this serve's (`restored_clamps`).
    pub(super) fn cell_restored_clamps(
        &self,
        seq: &SequenceState,
        compute_start: usize,
        stream: u64,
    ) -> Result<()> {
        let Some(grid) = cell::cell_grid()? else {
            return Ok(());
        };
        if seq.marconi_skip_to == 0 || compute_start != seq.marconi_skip_to {
            return Ok(());
        }
        let rank = self.cell_rank();
        let walked = &seq.cell_chunk_ends;
        let k = cell::restored_clamps(grid, walked, rank == 0, compute_start)
            .map_err(anyhow::Error::msg)?;
        tracing::info!(
            "GLM prefill cell grid: rank {rank} restored state at {compute_start}: {k} extra \
             clamps (walked chunk ends {walked:?}, grid first {} step {})",
            grid.first,
            grid.step
        );
        for _ in 0..k {
            self.normalize_ssm_states(seq, stream)?;
        }
        Ok(())
    }

    /// 2026-10-16: Before a prefill chunk from `chunk_start`: chunk 0 clears the chunk ends this
    /// request walked (a preempted request re-enters chunk 0), so [`Self::cell_restored_clamps`]
    /// counts this attempt's only.
    pub(super) fn cell_walk_start(
        &self,
        seq: &mut SequenceState,
        chunk_start: usize,
    ) -> Result<()> {
        if chunk_start == 0 && cell::cell_grid()?.is_some() {
            seq.cell_chunk_ends.clear();
        }
        Ok(())
    }

    /// 2026-10-16: After a prefill chunk: its end, walked.
    pub(super) fn cell_walk_end(&self, seq: &mut SequenceState, chunk_end: usize) -> Result<()> {
        if cell::cell_grid()?.is_some() {
            seq.cell_chunk_ends.push(chunk_end);
        }
        Ok(())
    }

    /// 2026-10-15: After rank 0's chunk 0 (`[0, chunk_len)`): the clamp the grid serve runs at
    /// its chunk 1's end when chunk 0 ends past the grid's first point.
    pub(super) fn cell_first_chunk_clamp(
        &self,
        seq: &SequenceState,
        chunk_start: usize,
        chunk_len: usize,
        stream: u64,
    ) -> Result<()> {
        let Some(grid) = cell::cell_grid()? else {
            return Ok(());
        };
        if chunk_start != 0
            || self.cell_rank() != 0
            || !cell::rank0_first_chunk_clamp(grid, chunk_len)
        {
            return Ok(());
        }
        self.normalize_ssm_states(seq, stream)
    }

    /// 2026-10-15: Under [`SSM_NORM_READBACK_ENV`], before a clamp on `stream`: every SSM
    /// layer's FP32 state of `seq`'s slot to the host and one [`norm_line`]. An FP16 state is
    /// skipped with a line saying so.
    pub(super) fn ssm_norm_readback_line(&self, seq: &SequenceState, stream: u64) -> Result<()> {
        if !ssm_norm_readback()? {
            return Ok(());
        }
        let rank = self.cell_rank();
        if self.seq_ssm_h_is_f16(seq) {
            tracing::info!(
                "SSM_NORM_MAX rank={rank} seq_len={} skipped=f16",
                seq.seq_len
            );
            return Ok(());
        }
        let (heads, k, v) = self.config.ssm_state_norm_dims();
        let n = heads * k * v;
        let mut layers = Vec::with_capacity(self.ssm_pool.num_ssm_layers);
        for i in 0..self.ssm_pool.num_ssm_layers {
            let mut bytes = vec![0u8; n * 4];
            self.gpu.copy_d2h_on_stream(
                self.ssm_pool.h_state(i, seq.slot_idx),
                &mut bytes,
                stream,
            )?;
            layers.push(
                bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect::<Vec<f32>>(),
            );
        }
        let m = norm_max(&layers, heads, k, v)?;
        tracing::info!("{}", norm_line(rank, seq.seq_len, &m));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(vals: &[f32], k: usize, v: usize) -> Vec<f32> {
        let mut b = vec![0.0f32; k * v];
        b[..vals.len()].copy_from_slice(vals);
        b
    }

    /// 2026-10-15: The norm is per (layer, head) block over its `k * v` values; the largest is
    /// located; only blocks strictly above 200 count, as the kernel scales only those.
    #[test]
    fn finds_the_largest_block_and_counts_over_bound() {
        let (k, v) = (2, 2);
        let l0 = [block(&[3.0, 4.0], k, v), block(&[200.0], k, v)].concat();
        let l1 = [block(&[0.0], k, v), block(&[120.0, 160.0, 1.0], k, v)].concat();
        let m = norm_max(&[l0, l1], 2, k, v).unwrap();
        assert_eq!((m.layer, m.head, m.over), (1, 1, 1));
        assert!((m.max - (120.0f64 * 120.0 + 160.0 * 160.0 + 1.0).sqrt()).abs() < 1e-9);
        assert_eq!(m.per_layer[0], 200.0);
        assert!(norm_line(2, 33570, &m).starts_with(
            "SSM_NORM_MAX rank=2 seq_len=33570 max=200.002 layer=1 head=1 over200=1 per_layer=200.000,200.002"
        ));
    }

    /// 2026-10-15: Negative control: a head boundary off by one value moves a value into the
    /// wrong block, which changes the answer; a layer of the wrong size is refused.
    #[test]
    fn block_boundaries_matter_and_sizes_are_checked() {
        let l = [block(&[0.0, 0.0, 0.0, 150.0], 2, 2), block(&[150.0], 2, 2)].concat();
        assert_eq!(norm_max(std::slice::from_ref(&l), 2, 2, 2).unwrap().over, 0);
        let wrong = norm_max(&[l[1..].iter().copied().chain([0.0]).collect()], 2, 2, 2).unwrap();
        assert_eq!(wrong.over, 1);
        assert!(norm_max(&[vec![0.0; 7]], 2, 2, 2).is_err());
    }
}
