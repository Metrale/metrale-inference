// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: Each draft's top-1 log-probability, for `--draft-confidence-stop` and D-Cut.
//!
//! Over the whole vocab the argmax kernel returns it (`argmax_bf16_batch_lp`). On a
//! vocab-sharded head each rank has only its shard's log-softmax, so the device exchange
//! carries each rank's shard log-sum-exp beside its (max, argmax) lanes and the pick computes
//! the winner's log-probability over the whole vocab (`head_xchg::pack_lp` / `pick_lp`).
//!
//! Owner: model-arch (GLM-5.3 MTP drafter).
//! Invariants:
//! - A draft without a log-probability (a sharded head on the host exchange, or a build without
//!   the kernels) reads as NaN, which never stops a chain (`draft_stop::chain_continues` is
//!   false only for a number below the threshold; the callers skip NaN).

use super::*;

/// 2026-10-11: The log-probability kernels; a zero handle where the build lacks one.
pub(super) struct LpKernels {
    pub(super) argmax_lp: KernelHandle,
    pub(super) pack_lp: KernelHandle,
    pub(super) pick_lp: KernelHandle,
}

impl LpKernels {
    pub(super) fn resolve(gpu: &dyn GpuBackend) -> Self {
        let k = |m, f| metrale_model_layers::layers::try_kernel(gpu, m, f);
        Self {
            argmax_lp: k("argmax", "argmax_bf16_batch_lp"),
            pack_lp: k("glm5next_mtp_xchg", "glm5next_mtp_xchg_pack_lp"),
            pick_lp: k("glm5next_mtp_xchg", "glm5next_mtp_xchg_pick_lp"),
        }
    }
}

/// 2026-10-11: Whether a propose computes log-probabilities: asked for (`wanted`), the argmax
/// kernel present, and on a sharded head the device exchange with its log-probability kernels.
pub(super) fn lp_computable(wanted: bool, k: &LpKernels, sharded: bool, device_xchg: bool) -> bool {
    wanted
        && k.argmax_lp.0 != 0
        && (!sharded || (device_xchg && k.pack_lp.0 != 0 && k.pick_lp.0 != 0))
}

impl Glm5NextMtpHead {
    /// 2026-10-11: Whether this propose's drafts carry log-probabilities (`lp_computable`), for a
    /// caller that wants them (`wanted`).
    pub(super) fn lp_ready(&self, ctx: &ForwardContext, wanted: bool) -> bool {
        let (_, _, sharded) = self.head_range(ctx);
        lp_computable(wanted, &self.lp_k, sharded, self.device_xchg(ctx).is_some())
    }

    /// 2026-10-11: The per-sequence propose wants log-probabilities only for the confidence stop.
    pub(super) fn single_lp_ready(&self, ctx: &ForwardContext) -> bool {
        self.lp_ready(
            ctx,
            metrale_model_layers::speculative::draft_stop::draft_stop_logprob().is_some(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn k(a: u64, p: u64, q: u64) -> LpKernels {
        LpKernels {
            argmax_lp: KernelHandle(a),
            pack_lp: KernelHandle(p),
            pick_lp: KernelHandle(q),
        }
    }

    /// 2026-10-11: Log-probabilities only when asked for and computable: an unsharded head needs
    /// the argmax kernel, a sharded one also the device exchange and both exchange kernels.
    #[test]
    fn log_probabilities_need_the_request_and_every_kernel_their_path_uses() {
        assert!(lp_computable(true, &k(1, 0, 0), false, false));
        assert!(!lp_computable(false, &k(1, 1, 1), false, true));
        assert!(!lp_computable(true, &k(0, 1, 1), false, true));
        assert!(lp_computable(true, &k(1, 1, 1), true, true));
        assert!(!lp_computable(true, &k(1, 1, 1), true, false));
        assert!(!lp_computable(true, &k(1, 0, 1), true, true));
        assert!(!lp_computable(true, &k(1, 1, 0), true, true));
    }
}
