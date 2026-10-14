// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Multi-sequence decode for the GDN layer
//! (`decode_multi_seq_inner`): the SSM mixer, then the FFN dispatched by
//! batch width.
//!
//! Owner: model-layers, GDN/SSM layer (`qwen3_ssm`).
//! Invariants: none beyond the types.

use super::*;

mod hc;
mod ssm_batched;
mod ssm_batched_proj;
mod ssm_batched_recurrent;

/// 2026-09-25: Smallest batch that takes the batched dense FFN
/// (`forward_prefill`): `METRALE_SSM_FFN_PREFILL_MIN_N`, default 5. A value
/// below 2 or one that does not parse gives 5. Read once per process.
///
/// Measured 2026-07-27 on GB10, Qwen3.6-27B NVFP4, decode tok/s, 2 reps per
/// cell: MIN_N=9 gave C=4 37.7, C=8 53.4; MIN_N=5 gave 37.8, 57.8; MIN_N=4
/// gave 36.2, 57.8.
fn ssm_ffn_prefill_min_n() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("METRALE_SSM_FFN_PREFILL_MIN_N")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&v| v >= 2)
            .unwrap_or(5)
    })
}

/// 2026-09-25: The batched dense FFN arm is on unless
/// `METRALE_NO_SSM_FFN_PREFILL` is exactly `1`. Read once per process.
fn ssm_ffn_prefill_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("METRALE_NO_SSM_FFN_PREFILL").as_deref() != Ok("1"))
}

impl Qwen3SsmLayer {
    #[allow(clippy::too_many_arguments)]
    /// 2026-09-25: Multi-sequence decode for a GDN layer.
    ///
    /// The mixer carries per-sequence recurrent state. It runs through
    /// `try_decode_multi_seq_ssm_batched` when that accepts the layer, else as
    /// a per-sequence loop over `ssm_forward`. Either way row `i` of
    /// `norm_output` ends up holding sequence `i`'s FFN input, and the FFN
    /// then runs over all `n` rows, dispatched by `n` and the FFN kind.
    ///
    /// In the per-sequence loop, `ssm_forward` writes no `norm_output` row and
    /// returns its output in `moe_output`, which the same iteration's
    /// `residual_add_rms_norm` consumes before the next sequence runs.
    pub(super) fn decode_multi_seq_inner<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_seqs: usize,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        _kv_cache: &mut PagedKvCache,
        _seq_lens: &[usize],
        _block_tables: &[Vec<u32>],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let bf16 = 2usize;
        let eps = ctx.config.rms_norm_eps as f32;
        let n = num_seqs;
        let ssm_ms_profile =
            crate::layers::ops::ModelLevers::get().ssm_ms_profile && !ctx.graph_capture;
        let phase_a_t0 = if ssm_ms_profile {
            ctx.gpu.synchronize(stream).ok();
            Some(std::time::Instant::now())
        } else {
            None
        };

        let residual_elem = 2usize;

        // 2026-09-25: Mixer. The batched path runs QKVZ and out_proj as one
        // GEMM each over all n rows; the per-sequence loop below reads those
        // weights once per sequence.
        if !self
            .try_decode_multi_seq_ssm_batched(hidden, residual, n, states, false, ctx, stream)?
        {
            for i in 0..n {
                let hidden_i = hidden.offset(i * h * residual_elem);
                let residual_i = residual.offset(i * h * residual_elem);
                let normed_i = ctx.buffers.norm_output().offset(i * h * bf16);

                let ssm_state = states[i]
                    .as_any_mut()
                    .downcast_mut::<SsmLayerState>()
                    .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState for seq {i}"))?;

                ops::rms_norm_residual(
                    ctx.gpu,
                    self.rms_norm_residual_k,
                    hidden_i,
                    &self.input_norm,
                    normed_i,
                    residual_i,
                    1,
                    h as u32,
                    eps,
                    stream,
                )?;

                let ssm_out = self.ssm_forward(normed_i, ssm_state, ctx, stream, false)?;

                ops::residual_add_rms_norm(
                    ctx.gpu,
                    self.residual_add_rms_norm_k,
                    hidden_i,
                    ssm_out,
                    &self.post_attn_norm,
                    normed_i,
                    residual_i,
                    1,
                    h as u32,
                    eps,
                    stream,
                )?;
            }
        }
        let phase_a_us = if let Some(t0) = phase_a_t0 {
            ctx.gpu.synchronize(stream).ok();
            t0.elapsed().as_micros()
        } else {
            0
        };
        let phase_b_t0 = if ssm_ms_profile {
            Some(std::time::Instant::now())
        } else {
            None
        };

        // 2026-10-13: The FFN ladder: the arm from `select_ssm_arm` over this
        // layer's capabilities, run by `ffn_rows` with the residual add as the
        // row sink (`layers/ffn_rows.rs`). The highway path (`hc.rs`) runs the
        // same ladder with `hc_post_rows` as the sink.
        let normed_base = ctx.buffers.norm_output();
        let arm = crate::layers::select_ssm_arm(&self.ssm_ffn_caps(n, ctx));
        let mut residual_add = |rows: DevicePtr, row0: usize, m: usize| {
            ops::residual_add(
                ctx.gpu,
                self.residual_add_k,
                hidden.offset(row0 * h * residual_elem),
                rows,
                (m * h) as u32,
                stream,
            )
        };
        crate::layers::ffn_rows(
            &self.ffn,
            arm,
            normed_base,
            n,
            ctx,
            stream,
            &mut residual_add,
        )?;
        if let Some(t0) = phase_b_t0 {
            ctx.gpu.synchronize(stream).ok();
            tracing::info!(
                "METRALE_SSM_MS_PROFILE n={n}: mixer={}us moe_residual={}us",
                phase_a_us,
                t0.elapsed().as_micros(),
            );
        }

        Ok(())
    }

    /// 2026-10-13: The GDN ladder's capabilities for `n` rows (`select_ssm_arm`
    /// orders them). Every predicate is evaluated here, once per step; none has
    /// a side effect beyond its own once-per-process logging.
    pub(super) fn ssm_ffn_caps(&self, n: usize, ctx: &ForwardContext) -> crate::layers::LadderCaps {
        use crate::layers::moe::GroupedRouting;
        crate::layers::LadderCaps {
            n,
            force_seq: false,
            use_grouped: false,
            dense_fixed: self.ffn.dense_fixed_ok(n, ctx),
            nvfp4_grouped: self.ffn.nvfp4_grouped_ok(n, ctx),
            fp8_row_invariant: n >= 2
                && crate::layers::row_invariant()
                && self
                    .ffn
                    .fp8_grouped_routing_ok(n, GroupedRouting::PerRow, ctx),
            is_dense: self.ffn.is_dense(),
            ssm_dense_prefill: n >= ssm_ffn_prefill_min_n()
                && self.ffn.is_dense()
                && ssm_ffn_prefill_enabled(),
            // 2026-09-25: Chunks of at most 8 rows, reading the weights ceil(n/8) times.
            km_chunk: (n >= 4 && self.ffn.can_forward_km(8.min(n) as u32)).then_some(8),
            fp8_grouped: self.ffn.fp8_grouped_decode_ok(n, ctx),
            moe_grouped: crate::layers::moe_grouped_decode_for(n),
            atomic_c4: n == 4
                && std::env::var("METRALE_MOE_ATOMIC_C4_DECODE")
                    .ok()
                    .as_deref()
                    == Some("1"),
            token_major: !crate::layers::ops::ModelLevers::get().moe_legacy_pertoken_decode,
            pairwise: false,
            batched: std::env::var("METRALE_MOE_BATCHED_DECODE").ok().as_deref() == Some("1"),
        }
    }
}
