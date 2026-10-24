// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: Where a draft's id ends after the drafter forward, and its readback: split out of
//! `glm5next_mtp_head.rs` when `propose_device` (`METRALE_GLM_MTP_DEVICE_STEP`) began leaving the
//! id on the device.
//!
//! Owner: model-arch (GLM-5.3 MTP drafter).
//! Invariants: the readback is the one `forward_one` always did; only its caller changed.

use super::*;

impl Glm5NextMtpHead {
    /// 2026-10-11: Whether a draft's id ends on the device, in `st.arg`, on every rank: the
    /// device exchange runs (or the head is not split), and the head is not skipped.
    pub(super) fn draft_stays_on_device(&self, ctx: &ForwardContext) -> bool {
        let (_, v0, sharded) = self.head_range(ctx);
        // 2026-10-11: An unsplit head leaves the global id (`v0` is 0); a split one leaves a
        // local id unless the device exchange turns it global.
        !self.levers.skip_head()
            && if sharded {
                self.device_xchg(ctx).is_some()
            } else {
                v0 == 0
            }
    }

    /// 2026-10-11: The readback of the draft [`Self::draft_on_device`] computed: the id and,
    /// when this propose computes one, its log-probability (the lane after the id in `st.arg`),
    /// in one read after the device exchange, else through the host exchange (no log-probability).
    pub(super) fn read_draft(
        &self,
        st: &mut Glm5NextMtpProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<(u32, f32)> {
        let gpu = ctx.gpu;
        // 2026-09-25: Timing arm `METRALE_GLM_MTP_SKIP=head`: the head is skipped and the
        // draft is token 0, so `propose` times the block alone. `=block` skips the block instead.
        if self.levers.skip_head() {
            return Ok((0, f32::NAN));
        }
        let (_, v0, sharded) = self.head_range(ctx);
        // 2026-10-11: The id, and the log-probability after it when this propose computes one.
        let with_lp = self.single_lp_ready(ctx);
        let mut out = [0u8; 8];
        let read = if with_lp { 8 } else { 4 };
        let lp_of = |out: &[u8; 8]| {
            if with_lp {
                f32::from_le_bytes([out[4], out[5], out[6], out[7]])
            } else {
                f32::NAN
            }
        };
        if self.device_xchg(ctx).is_some() {
            // 2026-10-11: The device exchange left the global id in `st.arg`: one read.
            gpu.copy_d2h(st.arg, &mut out[..read])?;
            return Ok((
                u32::from_le_bytes([out[0], out[1], out[2], out[3]]),
                lp_of(&out),
            ));
        }
        gpu.synchronize(stream)?;
        gpu.copy_d2h(st.arg, &mut out[..read])?;
        let local = u32::from_le_bytes([out[0], out[1], out[2], out[3]]) as usize;
        let Some(comm) = ctx.comm.filter(|_| sharded) else {
            return Ok(((v0 + local) as u32, lp_of(&out)));
        };
        // 2026-09-25: Exchange (max, argmax) through one all-reduce of BF16 lanes: each rank's
        // max logit, then three base-256 digits of each rank's global index (`head_xchg` has
        // the layout and why a SUM all-reduce delivers it untouched). The logit lane is the
        // head's own BF16 output, so it round-trips bit for bit.
        let mut lb = [0u8; 2];
        gpu.copy_d2h(st.logits.offset(local * 2), &mut lb)?;
        let mut pack = head_xchg::pack(self.head_world, self.head_rank, lb, v0 + local)?;
        gpu.copy_h2d(&pack, st.head_xchg)?;
        comm.all_reduce_async(st.head_xchg.0, pack.len(), stream)?;
        gpu.synchronize(stream)?;
        gpu.copy_d2h(st.head_xchg, &mut pack)?;
        // 2026-09-25: A tie goes to the lower rank, identically on every rank, so the ranks'
        // drafter KV streams cannot diverge on a tie. 2026-10-11: No log-probability on the host
        // exchange.
        Ok((head_xchg::pick(self.head_world, &pack)?, f32::NAN))
    }
}
