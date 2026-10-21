// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: The draft head sweep of one row (moved from `forward_body`), and the screened
//! draft head (`METRALE_GLM_MTP_SCREEN_HEAD=1`).
//!
//! The screened draft head reads the target's INT8 screen copy of the same `lm_head` rows
//! (model-engine `screen_head.rs`; the MTP layer has no head of its own) instead of the drafter's
//! FP8 copy: the screen keys this rank's shard at local indices and `screen_head_local` leaves
//! the shard argmax and its logit where the BF16 sweep and `argmax_bf16` would, so the exchange
//! is unchanged and the draft is the BF16 head's, bit for bit (`screen_head_gate`'s drafter
//! check). The FP8 head's drafts differ from it at near ties; a draft changes acceptance, never
//! an emitted token. The drafter's shard is then the screen's (`init`), so one copy serves both.
//!
//! Owner: model-arch (GLM-5.3 MTP drafter).
//! Invariants:
//! - The screened sweep takes a sharded call of at most `SCREEN_MAX_M` rows without the draft
//!   log-probability; every other call sweeps the FP8 copy or the BF16 head as before.
//! - Its state and bound scratch are its own, not the target verify's, so an early-launched
//!   drafter graph never shares them with a verify in flight.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_model_layers::layer::ForwardContext;
use metrale_model_layers::layers::ops::{
    self, SCREEN_MAX_M, SCREEN_STATE_BYTES, ScreenKernels, ScreenShard,
};
use metrale_model_layers::weight_map::DenseWeight;

use super::{Glm5NextMtpHead, Glm5NextMtpProposerState};

/// 2026-10-21: The screened draft head: the target screen's kernels and copy, this drafter's own
/// scratch.
pub struct DraftScreen {
    kernels: ScreenKernels,
    local: KernelHandle,
    /// 2026-10-21: The target's shard copy keyed at local indices (`base` 0) over this drafter's
    /// own `upper` and `state`.
    shard: ScreenShard,
    kappa: f32,
    /// 2026-10-21: The shard's first global row (the drafter's `head_v0`).
    begin: usize,
}

impl DraftScreen {
    /// 2026-10-21: Over the target screen `(kernels, shard, kappa)`; `local` is
    /// `screen_head_local`. Allocates this drafter's bound scratch and state block.
    pub fn new(
        gpu: &dyn GpuBackend,
        (kernels, shard, kappa): (ScreenKernels, ScreenShard, f32),
        local: KernelHandle,
    ) -> Result<Self> {
        anyhow::ensure!(
            kernels.loaded() && local.0 != 0,
            "screened draft head: the screen kernels are not in this build"
        );
        let rows = shard.rows as usize;
        Ok(Self {
            kernels,
            local,
            shard: ScreenShard {
                upper: gpu.alloc(SCREEN_MAX_M as usize * rows * 4)?,
                state: gpu.alloc(SCREEN_STATE_BYTES)?,
                base: 0,
                ..shard
            },
            kappa,
            begin: shard.base as usize,
        })
    }

    /// 2026-10-21: `(head_v0, head_n)`: the drafter sweeps the screen's shard.
    pub(super) fn range(&self) -> (usize, usize) {
        (self.begin, self.shard.rows as usize)
    }

    /// 2026-10-21: Rows `x` (`m`, BF16) into `arg[0..m]` and their logits (row stride
    /// `row_stride`), as the BF16 sweep and the argmax leave them.
    pub(super) fn sweep(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        m: usize,
        arg: DevicePtr,
        logits: DevicePtr,
        row_stride: usize,
        stream: u64,
    ) -> Result<()> {
        let m = m as u32;
        ops::screen_head_keys(
            gpu,
            &self.kernels,
            &self.shard,
            x,
            m,
            DevicePtr::NULL,
            self.kappa,
            stream,
        )?;
        ops::screen_head_local(
            gpu,
            self.local,
            &self.shard,
            x,
            m,
            arg,
            logits,
            u32::try_from(row_stride)?,
            stream,
        )
    }
}

impl Glm5NextMtpHead {
    /// 2026-10-21: The screened draft head for a call of `m` rows over `n` head rows, when it
    /// takes it (module invariants).
    pub(super) fn draft_screen(
        &self,
        sharded: bool,
        n: usize,
        m: usize,
        lp: bool,
    ) -> Option<&DraftScreen> {
        self.screen
            .as_ref()
            .filter(|s| sharded && !lp && m <= SCREEN_MAX_M as usize && n == s.range().1)
    }

    /// 2026-10-21: `shared_head.norm`'s output row `st.x` through the draft head into `st.arg`
    /// (and the logits the exchange reads); returns `(v0, lp)` for the exchange.
    pub(super) fn head_sweep_one(
        &self,
        gpu: &dyn GpuBackend,
        st: &mut Glm5NextMtpProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<(usize, Option<DevicePtr>)> {
        let h = self.hidden;
        let (n, v0, sharded) = self.head_range(ctx);
        // 2026-10-11: Under the confidence stop the argmax also writes the draft's
        // log-probability, at `arg + 4` (`draft_lp`).
        let lp = self.single_lp_ready(ctx).then_some(st.arg.offset(4));
        if let Some(s) = self.draft_screen(sharded, n, 1, lp.is_some()) {
            s.sweep(gpu, st.x, 1, st.arg, st.logits, n, stream)?;
            return Ok((v0, lp));
        }
        // 2026-09-25: The FP8 copy covers only `[head_v0, head_v0 + head_n)`, so it serves the
        // sharded sweep alone; an unsharded sweep reads the BF16 head.
        match self.head_fp8.filter(|_| sharded && n == self.head_n) {
            Some(q) => ops::dense_gemv_fp8w(
                gpu,
                self.gemv_fp8w_k,
                st.x,
                &q,
                st.logits,
                n as u32,
                h as u32,
                stream,
            )?,
            None => ops::dense_gemv(
                gpu,
                self.gemv_k,
                st.x,
                &DenseWeight {
                    weight: self.lm_head.weight.offset(v0 * h * 2),
                },
                st.logits,
                n as u32,
                h as u32,
                stream,
            )?,
        }
        match lp {
            Some(lp) => ops::argmax_bf16_batch_lp(
                gpu,
                self.lp_k.argmax_lp,
                st.logits,
                st.arg,
                lp,
                n as u32,
                1,
                n as u32,
                stream,
            )?,
            None => ops::argmax_bf16(gpu, self.argmax_k, st.logits, st.arg, n as u32, stream)?,
        }
        Ok((v0, lp))
    }
}
