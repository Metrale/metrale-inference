// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: `--profile` under `--forward circuit` (LIFECYCLE-DESIGN.md 15.10): the decode
//! program run eagerly with its timing events (`circuit_exec::profile`), reported in the lines
//! legacy's `decode_profiled` logs: per layer its op and FFN lines and the DIAG readback of the
//! residual stream, then the top-5 logits and the step line. The DIAG lines are formatted here
//! for both forwards.
//!
//! Owner: model-engine (FEATURES workstream).
//! Invariants:
//! - The profiled step runs the decode program the unprofiled step runs, so its logits are the
//!   circuit's; only the timing and the readbacks are added.
//! - Like `decode_profiled`, it pushes the token and advances `seq_len` only after the step
//!   succeeds.

use anyhow::{Context, Result};
use metrale_config::LayerType;
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_layers::circuit_exec::{CircuitExec, StepEnv};
use metrale_model_layers::layer::ForwardContext;

use super::super::types::TransformerModel;
use super::impl_circuit_run::gdn_states;
use crate::traits::SequenceState;

/// 2026-10-03: The DIAG line of the residual stream after the embedding.
pub(crate) fn diag_embed_line(tok: usize, norm: f32, vals: &[f32]) -> String {
    format!(
        "DIAG tok={tok} after_embed (FP32): norm={norm:.4} vals={:.4?}",
        &vals[..4]
    )
}

/// 2026-10-03: The DIAG line of the residual stream after layer `layer`.
pub(crate) fn diag_layer_line(
    tok: usize,
    layer: usize,
    kind: LayerType,
    norm: f32,
    vals: &[f32],
) -> String {
    format!(
        "DIAG tok={tok} after_L{layer} ({kind:?}) [FP32]: norm={norm:.4} vals={:.4?}",
        &vals[..4]
    )
}

/// 2026-10-03: The DIAG line of the five largest BF16 logits, largest first; ties keep the
/// lower index first (a stable sort, as legacy's).
pub(crate) fn diag_top5_line(tok: usize, logits_bf16: &[u8]) -> String {
    let mut indexed: Vec<(usize, f32)> = logits_bf16
        .chunks_exact(2)
        .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
        .enumerate()
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    format!(
        "DIAG tok={tok} top5_logits: {:?}",
        &indexed[..5.min(indexed.len())]
    )
}

impl TransformerModel {
    /// 2026-10-03: The profiled decode step of `token` through `exec`'s decode program, after
    /// the embedding and metadata upload `decode_dispatch_with` made; the logits pointer.
    pub(super) fn circuit_profiled(
        &self,
        exec: &CircuitExec,
        token: u32,
        hidden: DevicePtr,
        seq: &mut SequenceState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        let profile = exec
            .profile
            .as_ref()
            .context("--profile with a circuit built without its timing events")?;
        let meta = ctx
            .attn_metadata
            .context("circuit decode needs the step's attention metadata")?;
        let gdn = gdn_states(self.layers.len(), &[&seq.layer_states])?;
        let tok = seq.seq_len;
        self.gpu.synchronize(stream)?;
        let (vals, norm) = self.readback_f32(hidden, 8)?;
        tracing::info!("{}", diag_embed_line(tok, norm, &vals));
        let mut diag = vec![String::new(); self.layers.len()];
        let times = profile.run(
            &exec.decode,
            &StepEnv {
                gpu: self.gpu.as_ref(),
                stream,
                gdn: &gdn,
                max_blocks_per_seq: meta.max_blocks_per_seq,
                prefill: None,
            },
            &mut |l| {
                self.gpu.synchronize(stream)?;
                let (vals, norm) = self.readback_f32(hidden, 8)?;
                diag[l] = diag_layer_line(tok, l, self.config.layer_type(l), norm, &vals);
                Ok(())
            },
        )?;
        let (per_layer, line) = profile.report(&times, tok);
        for (lines, d) in per_layer.iter().zip(&diag) {
            for x in lines {
                tracing::info!("{x}");
            }
            tracing::info!("{d}");
        }
        let mut logits = vec![0u8; self.config.vocab_size * 2];
        self.gpu.copy_d2h(self.buffers.logits(), &mut logits)?;
        tracing::info!("{}", diag_top5_line(tok, &logits));
        tracing::info!("{line}");
        seq.tokens.push(token);
        seq.seq_len += 1;
        Ok(self.decode_logits_ptr())
    }
}

#[cfg(test)]
#[path = "impl_circuit_profile_tests.rs"]
mod impl_circuit_profile_tests;
