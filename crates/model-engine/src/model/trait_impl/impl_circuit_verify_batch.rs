// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: The batched MTP verify's body under `--forward circuit`: the program compiled for
//! the batch's row table runs in place of `run_verify_layers`, the final norm, the lm_head and
//! the argmax of `decode_verify_batched_dispatch`. Everything around the body stays the legacy
//! forward's: embeds, the WY tables, the carried-state protocol's host side, the metadata, the
//! graph key and capture, and the argmax readback.
//!
//! Owner: model-engine speculative decoding.
//! Invariants:
//! - The row table's runs are the ones the legacy GDN layers form (`batched_conv_gdn_route`),
//!   and a run is contiguous exactly when every GDN layer would run it batched
//!   (`gdn_verify_run_batched`, the legacy layers' own decision); layers that disagree are an
//!   error.
//! - The circuit models the carried-state verify, and the verify without WY tables (which
//!   neither carries nor batches a run); any other verify is refused before anything launches.

use anyhow::{Context, Result, ensure};
use metrale_circuit::{RowTable, VerifyRun};
use metrale_config::LayerType;
use metrale_gpu_runtime::gpu::DevicePtr;
use std::sync::Arc;

use metrale_model_layers::circuit_exec::verify_batch::VerifyBatchProgram;
use metrale_model_layers::circuit_exec::{CircuitExec, GdnState};
use metrale_model_layers::layer::{AttnMetadataDev, LayerState};

use super::super::types::TransformerModel;
use crate::traits::SequenceState;

impl TransformerModel {
    /// 2026-09-30: The row table of a batched verify of `seqs` at widths `ks`.
    pub(super) fn verify_batch_table(
        &self,
        seqs: &mut [&mut SequenceState],
        ks: &[usize],
        wy_tables: DevicePtr,
        carried: bool,
    ) -> Result<RowTable> {
        let gdn: Vec<usize> = (0..self.layers.len())
            .filter(|&i| self.config.layer_type(i) == LayerType::LinearAttention)
            .collect();
        let mut runs = Vec::new();
        let mut g0 = 0;
        while g0 < ks.len() {
            let g1 = (g0..ks.len())
                .find(|&i| ks[i] != ks[g0])
                .unwrap_or(ks.len());
            let mut verdict: Option<bool> = None;
            for &l in &gdn {
                let states: Vec<&mut (dyn LayerState + 'static)> = seqs[g0..g1]
                    .iter_mut()
                    .map(|s| s.layer_states[l].as_mut())
                    .collect();
                let batched = self.layers[l]
                    .gdn_verify_run_batched(&states, ks[g0], self.levers.gdn_wyn, wy_tables)?
                    .with_context(|| format!("GDN layer {l} has no batched verify"))?;
                match verdict {
                    None => verdict = Some(batched),
                    Some(v) => ensure!(
                        v == batched,
                        "GDN layers disagree on running sequences {g0}..{g1} batched"
                    ),
                }
            }
            runs.push(VerifyRun {
                k: ks[g0] as u64,
                n: (g1 - g0) as u64,
                contiguous: verdict.unwrap_or(false),
            });
            g0 = g1;
        }
        Ok(RowTable { runs, carried })
    }

    /// 2026-09-30: The program for this batch and the GDN state its rows read, to run in place of
    /// the legacy verify body. `carry` is `gdn_carry_begin`'s verdict and `meta` the staged
    /// metadata. Host work only, so it runs before any capture.
    pub(super) fn circuit_verify_batch_prepare(
        &self,
        exec: &CircuitExec,
        seqs: &mut [&mut SequenceState],
        ks: &[usize],
        wy_tables: DevicePtr,
        carry: bool,
        meta: &AttnMetadataDev,
        argmax_out: DevicePtr,
    ) -> Result<(Arc<VerifyBatchProgram>, Vec<Vec<GdnState>>)> {
        // 2026-09-30: A verify with its WY tables staged carries (`gdn_carry_begin`); one that
        // stages them and does not carry would run the tables' parent WY or write-on-accept arms,
        // which the circuit does not model.
        ensure!(
            carry || wy_tables.is_null(),
            "a batched verify with WY tables that does not carry is not modelled \
             (METRALE_NO_GDN_CARRY, or no carry binding)"
        );
        let vb = exec
            .verify_batch
            .as_ref()
            .context("this circuit executor compiles no batched verify")?;
        ensure!(
            meta.positions == vb.fixed_meta().positions && meta.seq_slot.is_null(),
            "the batched verify's metadata is not where the circuit reads it"
        );
        ensure!(
            argmax_out == vb.fixed_tokens(),
            "the batched verify reads its tokens back from where the circuit does not write them"
        );
        let table = self.verify_batch_table(seqs, ks, wy_tables, carry)?;
        let entry = vb.program(self.gpu.as_ref(), &table)?;
        let rows: Vec<&[Box<dyn LayerState>]> =
            seqs.iter().map(|s| s.layer_states.as_slice()).collect();
        let gdn = super::impl_circuit_run::gdn_states(self.layers.len(), &rows)?;
        Ok((entry, gdn))
    }
}
