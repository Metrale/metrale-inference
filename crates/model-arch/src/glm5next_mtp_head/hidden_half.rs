// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: The target-hidden half of the drafter's `eh_proj` input, `concat[h..2h]`.
//!
//! Owner: model-arch (GLM-5.3 MTP drafter).
//! Invariants:
//! - Only a target row (a propose's first draft, a drafter context row) can take the target's
//!   final norm; a chain draft reads the drafter's own `shared_head.norm` output, which vLLM
//!   recycles as is (`glm5next/common/mtp.py`: "the post-norm result feeds both draft logits
//!   and the next recycled hidden state").
//! - Drafter-only: the target verifies every draft, so this changes acceptance, never an
//!   emitted token.
//!
//! The reference GLM MTP layer (vLLM `Glm5NextMultiTokenPredictorLayer`) is fed the target
//! model's OUTPUT hidden, `model.norm(hc_contract(streams))`: vLLM's runner hands the MTP
//! proposer the model's returned hidden states, and `Glm5NextModel.forward` returns them after
//! `self.norm`. Metrale saves `hidden_states` rows before the final norm (`save_hidden_for_mtp`,
//! the prefill capture), which is the Qwen MTP convention (`pre_fc_norm_hidden`). `hnorm` is an
//! RMSNorm, so the final norm's scale cancels and only its weight survives:
//! `hnorm(norm(v)) = hnorm(v * w_final)`. GLM-5.3's `w_final` is far from uniform (mean 1.42,
//! sd 0.17, one channel at -0.015), so the two feeds differ.

use super::*;

/// 2026-10-11: Whether a row takes the target's final norm before `hnorm`.
pub(super) fn takes_target_norm(lever: bool, target_row: bool) -> bool {
    lever && target_row
}

/// 2026-10-11: One row, `out = x * w / rms(x)`: the drafter's single-row norm launch.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_rms_norm(
    gpu: &dyn GpuBackend,
    rms_norm_k: KernelHandle,
    x: DevicePtr,
    w: DevicePtr,
    out: DevicePtr,
    n: usize,
    eps: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, rms_norm_k)
        .grid([1, 1, 1])
        .block([(n.min(1024)) as u32, 1, 1])
        .arg_ptr(x)
        .arg_ptr(w)
        .arg_ptr(out)
        .arg_u32(n as u32)
        .arg_f32(eps)
        .launch(stream)
}

/// 2026-10-11: `out = hnorm(src)`, where `src` is `row`, or `target_norm(row)` staged in
/// `scratch` when `target_norm` is given. `scratch` must not alias `row` or `out`. Both norms
/// are `rms_norm_k` (`x * w / rms`, the kernel the target's final norm runs for GLM).
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_hidden_half(
    gpu: &dyn GpuBackend,
    rms_norm_k: KernelHandle,
    eps: f32,
    hidden: usize,
    row: DevicePtr,
    target_norm: Option<DevicePtr>,
    hnorm: DevicePtr,
    scratch: DevicePtr,
    out: DevicePtr,
    stream: u64,
) -> Result<()> {
    let norm = |x, w, o| launch_rms_norm(gpu, rms_norm_k, x, w, o, hidden, eps, stream);
    let src = match target_norm {
        Some(w) => {
            norm(row, w, scratch)?;
            scratch
        }
        None => row,
    };
    norm(src, hnorm, out)
}

impl Glm5NextMtpHead {
    /// 2026-10-11: `concat[h..2h] = hnorm(row)`, or `hnorm(target_final_norm(row))` for a target
    /// row under `ModelLevers::mtp_target_postnorm` (`METRALE_MTP_TARGET_POSTNORM=1`).
    pub(super) fn hidden_half(
        &self,
        ctx: &ForwardContext,
        row: DevicePtr,
        target_row: bool,
        scratch: DevicePtr,
        concat: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let target_norm = takes_target_norm(ctx.levers.mtp_target_postnorm, target_row)
            .then_some(self.target_final_norm);
        launch_hidden_half(
            ctx.gpu,
            self.rms_norm_k,
            self.module.layer.rms_eps,
            self.hidden,
            row,
            target_norm,
            self.module.hnorm,
            scratch,
            concat.offset(self.hidden * 2),
            stream,
        )
    }
}

#[cfg(test)]
#[path = "hidden_half_tests.rs"]
mod tests;
