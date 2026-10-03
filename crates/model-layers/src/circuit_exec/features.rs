// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The FEATURES workstream's part of an executor build (LIFECYCLE-DESIGN.md 15.10):
//! the tensor-parallel rank's shape, the circuit overlays (LoRA adapters, TP reduces), the decode
//! program's timing events under `--profile` (`profile`) and the sequence swap runner (`swap`),
//! made once the decode program is compiled and freed with the executor. One call per hook, so
//! the core build carries one line each.
//!
//! Owner: model-layers circuit executor (FEATURES workstream).
//! Invariants:
//! - A failure frees the workspace the build allocated, as the build's own failures do.

use anyhow::{Context, Result};
use metrale_circuit::{Circuit, FusionPlan};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::Boot;
use super::profile::DecodeProfile;
use super::program::Program;
use super::swap::{self, SwapRunner};
use metrale_circuit::parallel::Parallel;

/// 2026-10-03: The profile and the swap runner `b` asks for, over the decode `program` compiled
/// from `plan` of `circuit`.
pub(super) fn build(
    b: &Boot<'_>,
    (circuit, plan, program): (&Circuit, &FusionPlan, &Program),
    workspace: DevicePtr,
) -> Result<(Option<DecodeProfile>, Option<SwapRunner>)> {
    let made = (|| -> Result<(Option<DecodeProfile>, Option<SwapRunner>)> {
        let profile = b
            .profile
            .then(|| DecodeProfile::new(b.gpu, circuit, plan, program))
            .transpose()?;
        let runner = match b
            .swap
            .clone()
            .map(|s| swap::build(b.gpu, circuit, s))
            .transpose()
        {
            Ok(r) => r,
            Err(e) => {
                if let Some(p) = profile {
                    p.free(b.gpu).ok();
                }
                return Err(e);
            }
        };
        Ok((profile, runner))
    })();
    if made.is_err() {
        b.gpu.free(workspace).ok();
    }
    made
}

/// 2026-10-03: The shape this process plans at: its tensor-parallel rank's share, under TP.
pub(super) fn rank_shape(b: &Boot<'_>, shape: &mut metrale_circuit::ArchShape) -> Result<()> {
    if let Some(Parallel::Tensor(tp)) = b.parallel {
        *shape = metrale_circuit::parallel::rank_shape(shape, tp)
            .context("the tensor-parallel rank's shape")?;
    }
    Ok(())
}

/// 2026-10-03: The overlays of the served circuit: the LoRA adapters' nodes, the
/// tensor-parallel reduces or the expert-parallel split. Adapters across ranks are refused
/// (legacy folds the out_proj adapter after the reduce; no overlay order describes that yet).
pub(super) fn overlay(b: &Boot<'_>, circuit: &mut Circuit) -> Result<()> {
    anyhow::ensure!(
        b.lora.is_none() || b.parallel.is_none(),
        "LoRA adapters across tensor- or expert-parallel ranks"
    );
    if let Some(l) = &b.lora {
        *circuit = metrale_circuit::lora::adapt(circuit, &l.spec)
            .context("adapting the circuit to the LoRA pool")?;
    }
    match b.parallel {
        Some(Parallel::Tensor(tp)) => {
            *circuit = metrale_circuit::parallel::with_reduces(circuit, tp)
                .context("the tensor-parallel reduces")?;
        }
        Some(Parallel::Expert(ep)) => {
            *circuit = metrale_circuit::parallel::with_expert_reduces(circuit, ep)
                .context("the expert-parallel reduces")?;
        }
        None => {}
    }
    Ok(())
}

/// 2026-10-03: Free what [`build`] made.
pub(super) fn free(
    gpu: &dyn GpuBackend,
    profile: Option<DecodeProfile>,
    swap: Option<SwapRunner>,
) -> Result<()> {
    if let Some(p) = profile {
        p.free(gpu)?;
    }
    if let Some(s) = swap {
        s.free(gpu)?;
    }
    Ok(())
}
