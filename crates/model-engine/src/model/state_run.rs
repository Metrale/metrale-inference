// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The circuit's state programs, run (M5, LIFECYCLE-DESIGN.md sections 3.5 and
//! 15.3). Under `--forward circuit` the copies between steps (checkpoint, rollback, commit, slot
//! zero, the decode-rollback ring and the prefix cache) are the bound programs'
//! (`circuit_exec::state_bind`): the nodes decide which bytes move and how many, and this file
//! only resolves each node's places to the pools' addresses. Without a circuit the pools' own
//! loops run, until the flip deletes them.
//!
//! Owner: model-engine.
//! Invariants:
//! - A node's address is the pool's own accessor for its place, so a program reaches exactly
//!   the slots the legacy loop reaches.
//! - The programs are installed and removed with the executor (`set_forward`), on both pools at
//!   once.

use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use metrale_circuit::state_ops::StatePlace;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::circuit_exec::state_bind::{
    BoundStateNode, Conversion, StatePart, StatePrograms,
};
use parking_lot::RwLock;

use super::ssm_batched_copy::{StateCopy, run_ssm_state_copies};
use super::ssm_pool::SsmStatePool;
use super::ssm_snapshot::SsmSnapshotPool;

/// 2026-10-03: The bound state programs a pool runs; `None` under the legacy forward.
pub(crate) type ProgramsCell = RwLock<Option<Arc<StatePrograms>>>;

/// 2026-10-03: Which parts of a program to run: a commit whose h the write-on-accept fold
/// already placed copies only conv.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Parts {
    pub h: bool,
    pub conv: bool,
}

impl Parts {
    pub(crate) const ALL: Parts = Parts {
        h: true,
        conv: true,
    };

    fn has(self, part: StatePart) -> bool {
        match part {
            StatePart::H => self.h,
            StatePart::Conv => self.conv,
        }
    }
}

/// 2026-10-03: The places of one sequence's run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Places {
    /// 2026-10-03: The sequence's pool slot.
    pub slot: usize,
    /// 2026-10-03: The verify intermediate of the last accepted row (`num_accepted - 1`).
    pub step: Option<usize>,
    /// 2026-10-03: The cache slot: a ring flat index or a prefix-cache slot.
    pub cache: Option<usize>,
}

impl SsmStatePool {
    /// 2026-10-03: The address of `part` of SSM layer `layer` at a pool place.
    fn place_ptr(
        &self,
        place: StatePlace,
        part: StatePart,
        layer: usize,
        p: Places,
    ) -> Result<DevicePtr> {
        Ok(match (place, part) {
            (StatePlace::Live, StatePart::H) => self.h_state(layer, p.slot),
            (StatePlace::Live, StatePart::Conv) => self.conv_state(layer, p.slot),
            (StatePlace::Checkpoint, _) => {
                ensure!(
                    self.has_mtp,
                    "no verify checkpoint pool (the serve does not speculate)"
                );
                match part {
                    StatePart::H => self.h_checkpoint(layer, p.slot),
                    StatePart::Conv => self.conv_checkpoint(layer, p.slot),
                }
            }
            (StatePlace::AcceptedStep, _) => {
                let t = p.step.context("a commit names no accepted row")?;
                match part {
                    StatePart::H => {
                        ensure!(
                            t < self.h_inter_count(p.slot),
                            "slot {} keeps {} h intermediates; row {t} was accepted",
                            p.slot,
                            self.h_inter_count(p.slot)
                        );
                        self.h_intermediate(layer, p.slot, t)
                    }
                    StatePart::Conv => self.conv_intermediate(layer, p.slot, t),
                }
            }
            (StatePlace::Ring | StatePlace::Prefix, _) => {
                anyhow::bail!("{place:?} is a snapshot place, not a pool place")
            }
        })
    }

    /// 2026-10-03: Run `id`'s copies between pool places for one sequence on `stream`, h then
    /// conv, through the batched copy (`run_ssm_state_copies`), as the legacy verify sites do.
    pub(crate) fn run_copies(
        &self,
        nodes: &[BoundStateNode],
        p: Places,
        parts: Parts,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let (mut h, mut conv) = (Vec::new(), Vec::new());
        for n in nodes.iter().filter(|n| parts.has(n.part)) {
            let metrale_circuit::state_ops::StateOp::Copy { from, to } = n.op else {
                anyhow::bail!("{:?} is not a pool copy", n.op);
            };
            let copy = StateCopy {
                src: self.place_ptr(from, n.part, n.ssm_layer, p)?,
                dst: self.place_ptr(to, n.part, n.ssm_layer, p)?,
                bytes: n.bytes,
            };
            match n.part {
                StatePart::H => h.push(copy),
                StatePart::Conv => conv.push(copy),
            }
        }
        run_ssm_state_copies(gpu, &h, &conv, stream)
    }

    /// 2026-10-03: Zero slot `slot`'s live state on `stream`, node by node.
    pub(crate) fn run_zero(
        &self,
        nodes: &[BoundStateNode],
        slot: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let p = Places {
            slot,
            step: None,
            cache: None,
        };
        for n in nodes {
            let dst = self.place_ptr(StatePlace::Live, n.part, n.ssm_layer, p)?;
            gpu.memset_async(dst, 0, n.bytes, stream)?;
        }
        Ok(())
    }
}

impl SsmSnapshotPool {
    /// 2026-10-03: The address of `part` of SSM layer `layer` in a ring or prefix slot: the
    /// snapshot regions are strided by the FP32 widths.
    fn snapshot_ptr(
        &self,
        place: StatePlace,
        part: StatePart,
        layer: usize,
        slot: usize,
    ) -> Result<DevicePtr> {
        let (h, conv) = match place {
            StatePlace::Ring => (&self.decode_h_snapshots, &self.decode_conv_snapshots),
            StatePlace::Prefix => (&self.h_snapshots, &self.conv_snapshots),
            other => anyhow::bail!("{other:?} is a pool place, not a snapshot place"),
        };
        Ok(match part {
            StatePart::H => h[layer].offset(slot * self.h_bytes),
            StatePart::Conv => conv[layer].offset(slot * self.conv_bytes),
        })
    }

    /// 2026-10-03: Run a ring or prefix program between `pool` slot `p.slot` and snapshot slot
    /// `p.cache` on `stream`, node by node in layer order (h then conv per layer, as the legacy
    /// loops issue them), converting where the node says.
    pub(crate) fn run_snapshot(
        &self,
        nodes: &[BoundStateNode],
        pool: &SsmStatePool,
        p: Places,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let cache = p
            .cache
            .context("a snapshot program names no snapshot slot")?;
        for n in nodes {
            let loaded = match n.conversion {
                Some(Conversion::Widen) => self.h_f16_to_f32_k.0 != 0,
                Some(Conversion::Narrow) => self.h_f32_to_f16_k.0 != 0,
                None => true,
            };
            ensure!(
                loaded,
                "the {:?} h-state kernel is not loaded on this target",
                n.conversion
            );
        }
        let at = |place: StatePlace, n: &BoundStateNode| -> Result<DevicePtr> {
            match place {
                StatePlace::Ring | StatePlace::Prefix => {
                    self.snapshot_ptr(place, n.part, n.ssm_layer, cache)
                }
                pool_place => pool.place_ptr(pool_place, n.part, n.ssm_layer, p),
            }
        };
        for n in nodes {
            let (from, to) = match n.op {
                metrale_circuit::state_ops::StateOp::Copy { from, to }
                | metrale_circuit::state_ops::StateOp::Convert { from, to } => (from, to),
                other => anyhow::bail!("{other:?} is not a snapshot copy"),
            };
            let (src, dst) = (at(from, n)?, at(to, n)?);
            match n.conversion {
                None => gpu.copy_d2d_async(src, dst, n.bytes, stream)?,
                Some(Conversion::Widen) => {
                    metrale_model_layers::layers::ops::ssm_h_state_f16_to_f32(
                        gpu,
                        self.h_f16_to_f32_k,
                        src,
                        dst,
                        (n.bytes / 4) as u64,
                        stream,
                    )?
                }
                Some(Conversion::Narrow) => {
                    metrale_model_layers::layers::ops::ssm_h_state_f32_to_f16(
                        gpu,
                        self.h_f32_to_f16_k,
                        src,
                        dst,
                        (n.bytes / 2) as u64,
                        stream,
                    )?
                }
            }
        }
        Ok(())
    }
}

impl super::types::TransformerModel {
    /// 2026-10-03: Run state program `id` for pool slot `slot` (the accepted row `step` for a
    /// commit) on `stream`, when the circuit's programs are installed; `false` under the legacy
    /// forward, whose caller then runs its own loop.
    pub(super) fn run_state_program(
        &self,
        id: metrale_circuit::state_ops::StateProgramId,
        slot: usize,
        step: Option<usize>,
        parts: Parts,
        stream: u64,
    ) -> Result<bool> {
        let guard = self.ssm_pool.programs.read();
        let Some(p) = guard.as_ref() else {
            return Ok(false);
        };
        let places = Places {
            slot,
            step,
            cache: None,
        };
        self.ssm_pool
            .run_copies(p.nodes(id)?, places, parts, self.gpu.as_ref(), stream)?;
        Ok(true)
    }

    /// 2026-10-03: Install (or, with `None`, remove) the bound state programs on both pools.
    pub(super) fn install_state_programs(&self, programs: Option<Arc<StatePrograms>>) {
        *self.ssm_pool.programs.write() = programs.clone();
        *self.ssm_snapshots.programs.write() = programs;
    }
}

#[cfg(test)]
#[path = "state_run_tests.rs"]
mod state_run_tests;
