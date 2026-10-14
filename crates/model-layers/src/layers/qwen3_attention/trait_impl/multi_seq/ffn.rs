// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Multi-sequence decode FFN: residual add, post-attention norm and
//! the MoE or dense FFN for all n rows, then the residual add of its output.
//!
//! Owner: model-layers (qwen3 attention).
//! Invariants:
//! - A layer with a shortcut MoE carry and no MLA fails before any launch
//!   (the `ensure!` at the top of `ms_phase_ffn`).
//!
//! `ms_phase_ffn` picks its arm with `select_attn_arm` (`layers/ffn_rows_select.rs`,
//! which states the ladder's order) and runs it with `ffn_rows`; a LongCat
//! shortcut-MoE layer keeps its own one-row-at-a-time loop.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::ctx::MultiSeqCtx;
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::layers::qwen3_attention::Qwen3AttentionLayer;

/// 2026-09-25: The pairwise MoE decode arm is on unless
/// `METRALE_MOE_PAIRWISE_DECODE=0`, read once per process.
fn pairwise_moe_decode_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("METRALE_MOE_PAIRWISE_DECODE").as_deref() != Ok("0"))
}

/// 2026-09-25: The opt-in grouped routed MoE arm (`forward_prefill`),
/// `METRALE_MOE_GROUPED_ROUTED_DECODE=1`, read once per process. It applies from
/// `grouped_routed_decode_min()` rows: `METRALE_MOE_GROUPED_ROUTED_DECODE_MIN`,
/// or 2 when unset or unparsable.
fn grouped_routed_decode_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("METRALE_MOE_GROUPED_ROUTED_DECODE").as_deref() == Ok("1"))
}
fn grouped_routed_decode_min() -> usize {
    use std::sync::OnceLock;
    static M: OnceLock<usize> = OnceLock::new();
    *M.get_or_init(|| {
        std::env::var("METRALE_MOE_GROUPED_ROUTED_DECODE_MIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2)
    })
}

impl Qwen3AttentionLayer {
    pub(super) fn ms_phase_ffn(&self, c: &MultiSeqCtx<'_>, o_out: DevicePtr) -> Result<()> {
        // 2026-09-25: Only the one-row-at-a-time loop at the end runs the shortcut
        // MoE (`shortcut_carry_out` / `shortcut_carry_in`, set by the LongCat
        // loader); the other arms would skip it without an error. That loop is
        // forced when `mla` is set, so a shortcut layer without MLA is refused.
        anyhow::ensure!(
            (self.shortcut_carry_out.is_none() && self.shortcut_carry_in.is_none())
                || self.mla.is_some(),
            "shortcut-MoE model reached the batched FFN ladder, which does not              implement the shortcut; only the per-token branch does"
        );
        let MultiSeqCtx {
            fwd,
            n,
            stream,
            h,
            eps,
            bf16,
            hidden,
            residual,
            ..
        } = *c;

        if self.ffn.is_none() {
            ops::residual_add(
                fwd.gpu,
                self.residual_add_k,
                hidden,
                o_out,
                (n * h) as u32,
                stream,
            )?;
            return Ok(());
        }
        // 2026-10-13: The arm from `select_attn_arm` over this layer's
        // capabilities, run by `ffn_rows` with the residual add as the row sink
        // (`layers/ffn_rows.rs`); the highway path (`mod.rs`) runs the same
        // ladder with `hc_post_rows`. The pre-FFN norm keeps its old placement:
        // one launch over the n rows for the batched arms, one per row for the
        // final branch's arms.
        let arm = crate::layers::select_attn_arm(&self.attn_ffn_caps(n, fwd));
        let normed2 = fwd.buffers.norm_output();
        let residual_elem = 2usize;
        match arm {
            crate::layers::FfnArm::PerRow | crate::layers::FfnArm::Batched => {
                for i in 0..n {
                    ops::residual_add_rms_norm(
                        fwd.gpu,
                        self.residual_add_rms_norm_k,
                        hidden.offset(i * h * residual_elem),
                        o_out.offset(i * h * bf16),
                        &self.post_attn_norm,
                        normed2.offset(i * h * bf16),
                        residual.offset(i * h * residual_elem),
                        1,
                        h as u32,
                        eps,
                        stream,
                    )?;
                }
            }
            _ => ops::residual_add_rms_norm(
                fwd.gpu,
                self.residual_add_rms_norm_k,
                hidden,
                o_out,
                &self.post_attn_norm,
                normed2,
                residual,
                n as u32,
                h as u32,
                eps,
                stream,
            )?,
        }
        if arm == crate::layers::FfnArm::PerRow
            && (self.shortcut_carry_out.is_some() || self.shortcut_carry_in.is_some())
        {
            return self.ms_ffn_rows_shortcut(c, normed2);
        }
        let mut residual_add = |rows: DevicePtr, row0: usize, m: usize| {
            ops::residual_add(
                fwd.gpu,
                self.residual_add_k,
                hidden.offset(row0 * h * residual_elem),
                rows,
                (m * h) as u32,
                stream,
            )
        };
        crate::layers::ffn_rows(&self.ffn, arm, normed2, n, fwd, stream, &mut residual_add)
    }

    /// 2026-10-13: The attention ladder's capabilities for `n` rows
    /// (`select_attn_arm` orders them). Every predicate is evaluated here, once
    /// per step; the `!force_seq` guard of the batched arms is applied in the
    /// selector.
    pub(super) fn attn_ffn_caps(
        &self,
        n: usize,
        fwd: &ForwardContext,
    ) -> crate::layers::LadderCaps {
        use crate::layers::moe::GroupedRouting;
        let force_seq = self.mla.is_some();
        crate::layers::LadderCaps {
            n,
            force_seq,
            use_grouped: n >= grouped_routed_decode_min()
                && grouped_routed_decode_enabled()
                && self.ffn.moe_grouped_decode_ok(),
            dense_fixed: self.ffn.dense_fixed_ok(n, fwd),
            nvfp4_grouped: self.ffn.nvfp4_grouped_ok(n, fwd),
            fp8_row_invariant: n >= 2
                && crate::layers::row_invariant()
                && self
                    .ffn
                    .fp8_grouped_routing_ok(n, GroupedRouting::PerRow, fwd),
            is_dense: self.ffn.is_dense(),
            ssm_dense_prefill: false,
            // 2026-09-25: One `forward_km` call over all n rows, for n within the
            // dense FFN's narrow decode edge.
            km_chunk: ((4..=self.ffn.narrow_rows() as usize).contains(&n)
                && self.ffn.can_forward_km(n as u32))
            .then_some(n),
            fp8_grouped: self.ffn.fp8_grouped_decode_ok(n, fwd),
            moe_grouped: crate::layers::moe_grouped_decode_for(n),
            atomic_c4: false,
            token_major: false,
            pairwise: pairwise_moe_decode_enabled(),
            batched: std::env::var("METRALE_MOE_BATCHED_DECODE").ok().as_deref() == Some("1"),
        }
    }

    /// 2026-10-13: The one-row-at-a-time FFN of a LongCat shortcut-MoE layer
    /// (`shortcut_carry_out` / `shortcut_carry_in`), the only arm that runs the
    /// shortcut: the producer's output is copied into the carry before `self.ffn`
    /// runs, because both write `moe_output`; the consumer adds the paired
    /// sublayer's stashed row after the residual add. `normed_base` holds the
    /// normed rows.
    fn ms_ffn_rows_shortcut(&self, c: &MultiSeqCtx<'_>, normed_base: DevicePtr) -> Result<()> {
        let MultiSeqCtx {
            fwd,
            n,
            stream,
            h,
            bf16,
            hidden,
            ..
        } = *c;
        let residual_elem = 2usize;
        for i in 0..n {
            let hidden_i = hidden.offset(i * h * residual_elem);
            let normed2_i = normed_base.offset(i * h * bf16);
            if let (Some(moe_ffn), Some((carry, cap))) = (&self.moe_ffn, self.shortcut_carry_out) {
                anyhow::ensure!(n <= cap, "shortcut carry capacity {cap} < decode batch {n}");
                let sc_out = moe_ffn.forward(normed2_i, fwd, stream)?;
                if let crate::layers::FfnComponent::Moe(m) = moe_ffn {
                    m.apply_zero_expert(sc_out, normed2_i, 1, fwd, stream)?;
                }
                fwd.gpu
                    .copy_d2d_async(sc_out, carry.offset(i * h * bf16), h * bf16, stream)?;
            }
            let moe_out = self.ffn.forward(normed2_i, fwd, stream)?;
            ops::residual_add(
                fwd.gpu,
                self.residual_add_k,
                hidden_i,
                moe_out,
                h as u32,
                stream,
            )?;
            if let Some((carry, _cap)) = self.shortcut_carry_in {
                ops::residual_add(
                    fwd.gpu,
                    self.residual_add_k,
                    hidden_i,
                    carry.offset(i * h * bf16),
                    h as u32,
                    stream,
                )?;
            }
        }
        Ok(())
    }
}
