// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-29: Running the circuit executor's programs in place of the legacy layer loops: one
//! sequence's decode (`decode_forward_body`), a multi-sequence step
//! (`decode_batch_compute_main_with`) and an MTP verify (`decode_verify_graphed{,_k3,_k4}`).
//! Each hands the program the GatedDeltaNet state of every row's sequence.
//!
//! Owner: model-engine (decode).
//! Invariants:
//! - A program sees only the states its step reads: one row per sequence, in row order, and in
//!   a verify the sequence's rollback slots (`h_state_intermediates`,
//!   `conv_state_intermediates`).
//! - An FP16 h state is refused: the plans run the FP32 kernels.

use anyhow::{Context, Result, ensure};
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_layers::circuit_exec::program::MAX_VERIFY_STEPS;
use metrale_model_layers::circuit_exec::{CircuitExec, GdnState, Program, StepEnv};
use metrale_model_layers::layer::{ForwardContext, LayerState, SsmLayerState};

use super::super::types::TransformerModel;
use crate::traits::SequenceState;

/// 2026-09-29: Per layer, each row's GDN state; row `i` is `rows[i]`'s sequence. A layer
/// without one gets no rows.
fn gdn_states(layers: usize, rows: &[&[Box<dyn LayerState>]]) -> Result<Vec<Vec<GdnState>>> {
    let mut gdn = vec![Vec::new(); layers];
    for (row, seq) in rows.iter().enumerate() {
        for (layer, st) in seq.iter().enumerate() {
            let Some(s) = st.as_any().downcast_ref::<SsmLayerState>() else {
                continue;
            };
            ensure!(
                !s.h_is_f16,
                "row {row} layer {layer}: FP16 h state under an FP32-state circuit plan"
            );
            let slot = |v: &[DevicePtr], t: usize| v.get(t).copied().unwrap_or(DevicePtr::NULL);
            gdn[layer].push(GdnState {
                h: s.h_state,
                conv: s.conv_state,
                h_steps: std::array::from_fn::<_, MAX_VERIFY_STEPS, _>(|t| {
                    slot(&s.h_state_intermediates, t)
                }),
                conv_steps: std::array::from_fn::<_, MAX_VERIFY_STEPS, _>(|t| {
                    slot(&s.conv_state_intermediates, t)
                }),
            });
        }
    }
    Ok(gdn)
}

impl TransformerModel {
    fn run_program(
        &self,
        program: &Program,
        gdn: &[Vec<GdnState>],
        max_blocks_per_seq: u32,
        stream: u64,
    ) -> Result<()> {
        program.run(&StepEnv {
            gpu: self.gpu.as_ref(),
            stream,
            gdn,
            max_blocks_per_seq,
        })
    }

    /// 2026-09-28: Run `exec`'s decode program for `seq` in place of the layer loops, the final
    /// norm and the lm_head.
    pub(super) fn circuit_forward_body(
        &self,
        exec: &CircuitExec,
        seq: &SequenceState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let meta = ctx
            .attn_metadata
            .context("circuit decode needs the step's attention metadata")?;
        let gdn = gdn_states(self.layers.len(), &[&seq.layer_states])?;
        self.run_program(&exec.decode, &gdn, meta.max_blocks_per_seq, stream)
    }

    /// 2026-09-28: Run `exec`'s program for `padded_n` rows in place of the layer loops, the
    /// final norm and the lm_head of a multi-sequence step; row `i` is `states[i]`'s sequence,
    /// padding rows included.
    pub(super) fn circuit_multi_seq_body(
        &self,
        exec: &CircuitExec,
        states: &[Vec<Box<dyn LayerState>>],
        padded_n: usize,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            states.len() == padded_n,
            "{} states for {padded_n} rows",
            states.len()
        );
        let rows: Vec<&[Box<dyn LayerState>]> = states.iter().map(Vec::as_slice).collect();
        let gdn = gdn_states(self.layers.len(), &rows)?;
        // 2026-09-30: The arm this step's slots select (a runtime route's, or the primary);
        // under a graph capture the choice is baked with the slots the graph is keyed by.
        let program = exec.multi_seq_step(padded_n as u64, &gdn)?;
        self.run_program(program, &gdn, self.max_blocks_per_seq, stream)
    }

    /// 2026-09-29: Run `exec`'s verify program for `k` rows of `seq` in place of the layer
    /// loop, the final norm, the lm_head and the argmaxes of an MTP verify.
    pub(super) fn circuit_verify_body(
        &self,
        exec: &CircuitExec,
        seq: &SequenceState,
        k: usize,
        stream: u64,
    ) -> Result<()> {
        let program = exec
            .verify_program(k as u64)
            .with_context(|| format!("no circuit verify program was compiled for K={k}"))?;
        let gdn = gdn_states(self.layers.len(), &[&seq.layer_states])?;
        self.run_program(program, &gdn, self.max_blocks_per_seq, stream)
    }
}
