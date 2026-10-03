// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The exact verify arm for one run of a batched MTP verify
//! (`decode_batched_conv_gdn_multi_exact`, under `--exact-verify`): per row position, two
//! launches over all n sequences.
//!
//! Owner: model-layers (Qwen3 SSM layer).
//! Invariants:
//! - It launches only when n >= 2, `--gdn-fused-norm` is on, both strided kernels are
//!   linked, and the pointer checks pass: conv and h states on consecutive slots, h
//!   intermediates `h_bytes` apart within a sequence and evenly spaced across sequences
//!   without overlap. Otherwise it returns `Ok(false)` having launched nothing, and each
//!   sequence runs the per-row exact arm.
//!
//! The launches per position are `causal_conv1d_update_l2norm_f32_strided` (the FP32
//! conv with per-sequence row strides, also used by the batched-recurrent decode) and
//! `gated_delta_rule_decode_f32_strided_norm_snap` (the strided fused GDN + norm, which
//! writes the h intermediate). Conv intermediates are copied per sequence.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::trait_decode_batched_conv_gdn::ConvGdnArgs;
use super::{Qwen3SsmLayer, SsmLayerState};
use crate::layer::LayerState;
use crate::layers::ops;

/// 2026-09-25: A sequence's conv state and conv intermediates, collected before any
/// launch.
struct ExactMultiSeq {
    conv_state: DevicePtr,
    conv_inter: Vec<DevicePtr>,
}

impl Qwen3SsmLayer {
    /// 2026-09-25: Try the batched exact verify for one run (see the module header).
    /// `args` holds the run's buffers, offset to its first row, with
    /// `args.num_tokens = k`.
    pub(super) fn decode_batched_conv_gdn_multi_exact(
        &self,
        states: &mut [&mut (dyn LayerState + 'static)],
        ctx: &crate::layer::ForwardContext,
        args: &ConvGdnArgs,
    ) -> Result<bool> {
        let n = states.len();
        let kk = args.num_tokens;
        let conv_bytes = self.conv_state_bytes;
        let h_bytes = self.h_state_bytes;

        if n < 2
            || !crate::layers::qwen3_ssm::gdn_fused_norm_enabled()
            || self.conv1d_l2norm_f32_strided_k.0 == 0
            || self.gdn_f32_strided_norm_snap_k.0 == 0
        {
            return Ok(false);
        }

        // 2026-09-25: The strided kernels place sequence b's conv and h state at
        // `b * conv_bytes` and `b * h_bytes` from the first, so the states must sit
        // there; the h intermediates need a uniform stride across sequences.
        let mut seqs: Vec<ExactMultiSeq> = Vec::with_capacity(n);
        let mut conv_base = DevicePtr::NULL;
        let mut h_base = DevicePtr::NULL;
        let mut h_inter_base = DevicePtr::NULL;
        let mut h_inter_seq_stride = 0u64;
        for (i, state) in states.iter().enumerate() {
            let Some(st) = state.as_any().downcast_ref::<SsmLayerState>() else {
                return Ok(false);
            };
            // 2026-09-25: The GDN launch writes h intermediates 0..k-2 and none for the
            // last row.
            if st.conv_state_intermediates.len() < kk || st.h_state_intermediates.len() < kk - 1 {
                return Ok(false);
            }
            let hi0 = st.h_state_intermediates[0];
            for t in 1..kk - 1 {
                if st.h_state_intermediates[t].0 != hi0.0 + (t * h_bytes) as u64 {
                    return Ok(false);
                }
            }
            if i == 0 {
                conv_base = st.conv_state;
                h_base = st.h_state;
                h_inter_base = hi0;
            } else {
                if st.conv_state.0 != conv_base.0 + (i * conv_bytes) as u64
                    || st.h_state.0 != h_base.0 + (i * h_bytes) as u64
                {
                    return Ok(false);
                }
                if i == 1 {
                    h_inter_seq_stride = hi0.0.wrapping_sub(h_inter_base.0);
                    // 2026-09-25: A sequence's k-1 h intermediates must not overlap the next
                    // sequence's.
                    if h_inter_seq_stride < ((kk - 1) * h_bytes) as u64 {
                        return Ok(false);
                    }
                } else if hi0.0 != h_inter_base.0 + (i as u64) * h_inter_seq_stride {
                    return Ok(false);
                }
            }
            seqs.push(ExactMultiSeq {
                conv_state: st.conv_state,
                conv_inter: st.conv_state_intermediates[..kk].to_vec(),
            });
        }

        let ConvGdnArgs {
            deinterleaved,
            gates_buf,
            normed_out,
            qkvz_size,
            conv_dim,
            key_dim,
            value_dim,
            d_conv,
            qk_ch,
            nk,
            nv,
            kd,
            vd,
            bf16,
            fp32,
            stream,
            ..
        } = *args;
        let eps = ctx.config.rms_norm_eps as f32;
        // 2026-09-25: FP32 conv rows, one per sequence `qkvz_size` FP32 apart, reused at
        // every position: each position's GDN launch reads them on the same stream before
        // the next conv launch writes them.
        let conv_scratch = ctx.buffers.ssm_conv_out_f32();

        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| {
            tracing::info!(
                "EXACT batched MTP verify ENGAGED (#435, opt-in --exact-verify): \
                 per-token strided conv_f32 + strided fused-norm snap at batch=n \
                 (2 launches + n conv d2d per position); omit the flag for the \
                 default WY arms"
            );
        });

        for t in 0..kk {
            ops::conv1d_update_l2norm_strided(
                ctx.gpu,
                self.conv1d_l2norm_f32_strided_k,
                conv_base,
                deinterleaved.offset(t * qkvz_size * bf16),
                &self.ssm.conv1d,
                conv_scratch,
                conv_dim as u32,
                d_conv as u32,
                n as u32,
                qk_ch,
                kd as u32,
                1e-6,
                (kk * qkvz_size) as u32,
                qkvz_size as u32,
                stream,
            )?;

            let snapshot = t + 1 < kk;
            let (h_inter_t, h_inter_stride_elems) = if snapshot {
                (h_inter_base.offset(t * h_bytes), h_inter_seq_stride / 4)
            } else {
                // 2026-09-25: No intermediate for the last row; a null base skips the
                // stores.
                (DevicePtr::NULL, 0)
            };
            let gate_t = gates_buf.offset(t * nv * 2 * fp32);
            ops::gdn_decode_f32_strided_norm_snap(
                ctx.gpu,
                self.gdn_f32_strided_norm_snap_k,
                h_base,
                conv_scratch,
                conv_scratch.offset(key_dim * fp32),
                conv_scratch.offset(key_dim * 2 * fp32),
                gate_t,
                gate_t.offset(nv * fp32),
                deinterleaved.offset(t * qkvz_size * bf16 + conv_dim * bf16),
                self.ssm.norm.weight,
                normed_out.offset(t * value_dim * bf16),
                h_inter_t,
                h_inter_stride_elems,
                n as u32,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                qkvz_size as u32,
                qkvz_size as u32,
                (kk * nv * 2) as u32,
                (kk * qkvz_size) as u32,
                (kk * value_dim) as u32,
                eps,
                stream,
            )?;

            // 2026-09-25: Conv intermediates, one copy per sequence, none for the last row.
            if snapshot {
                for seq in &seqs {
                    ctx.gpu.copy_d2d_async(
                        seq.conv_state,
                        seq.conv_inter[t],
                        conv_bytes,
                        stream,
                    )?;
                }
            }
        }

        Ok(true)
    }

    /// 2026-10-01: Whether the carried exact verify can run at width `kk` (see
    /// [`Self::decode_batched_conv_gdn_multi_exact_carry`]): the carry engages
    /// (`carry_now`), the chain and FP32 conv twins are linked, and the decode in force is
    /// the unfused one, whose output norm is `gated_rms_norm_f32_strided`.
    pub(super) fn exact_carry_ready(&self, kk: usize) -> bool {
        self.carry_now(true, kk)
            && self.carry.exact_kernel_for(kk, false).0 != 0
            && self.carry.exact_kernel_for(kk, true).0 != 0
            && self.carry.kernels.conv_f32.0 != 0
            && self.gated_rms_norm_f32_strided_k.0 != 0
            && !crate::layers::qwen3_ssm::gdn_fused_norm_enabled()
    }

    /// 2026-10-03: Row 0's conv state when the carried exact arm can take the run: at least two
    /// sequences, WY tables staged, and every conv state on consecutive slots. `None` is
    /// [`Self::decode_batched_conv_gdn_multi_exact_carry`]'s decline; the circuit's run verdict
    /// (`gdn_verify_run_batched`) reads the same check.
    pub(super) fn exact_carry_conv_base(
        &self,
        states: &[&mut (dyn LayerState + 'static)],
        h_table: DevicePtr,
    ) -> Option<DevicePtr> {
        if states.len() < 2 || h_table.is_null() {
            return None;
        }
        let mut conv_base = DevicePtr::NULL;
        for (i, state) in states.iter().enumerate() {
            let st = state.as_any().downcast_ref::<SsmLayerState>()?;
            if i == 0 {
                conv_base = st.conv_state;
            } else if st.conv_state.0 != conv_base.0 + (i * self.conv_state_bytes) as u64 {
                return None;
            }
        }
        Some(conv_base)
    }

    /// 2026-10-01: The exact verify for one run, carried: three launches for all n sequences
    /// and K rows, `gdn_carry_conv_f32` (FP32 conv rows `b * K + t`, `qkvz_size` floats
    /// apart), `gdn_exact_carry{K}` (the strided decode's per-token chain, H read once, the
    /// accepted rows left pending in the stash instead of written as h intermediates), and
    /// one `gated_rms_norm_f32_strided` over the n * K rows. Rows are the bits of the
    /// per-token exact arm: each twin runs its parent's arithmetic in its parent's order.
    /// `Ok(false)` with nothing launched when the conv states are not on consecutive slots;
    /// the caller then folds the run and each sequence takes the per-row exact arm.
    pub(super) fn decode_batched_conv_gdn_multi_exact_carry(
        &self,
        states: &mut [&mut (dyn LayerState + 'static)],
        h_table: DevicePtr,
        slot_tab: DevicePtr,
        flag: DevicePtr,
        args: &ConvGdnArgs,
        ctx: &crate::layer::ForwardContext,
    ) -> Result<bool> {
        let n = states.len();
        let kk = args.num_tokens;
        let Some(binding) = self.carry.binding.get().copied() else {
            return Ok(false);
        };
        let Some(conv_base) = self.exact_carry_conv_base(states, h_table) else {
            return Ok(false);
        };
        let ConvGdnArgs {
            deinterleaved,
            gates_buf,
            normed_out,
            qkvz_size,
            conv_dim,
            key_dim,
            value_dim,
            d_conv,
            qk_ch,
            nk,
            nv,
            kd,
            vd,
            bf16,
            fp32,
            stream,
            ..
        } = *args;
        let lazy = super::carry::carry_lazy(n);
        let rows = ctx.buffers.ssm_conv_out_f32();
        ops::gdn_carry_conv_f32(
            ctx.gpu,
            self.carry.kernels.conv_f32,
            conv_base,
            deinterleaved,
            &self.ssm.conv1d,
            rows,
            binding.conv_stash,
            slot_tab,
            binding.pend,
            binding.conv_seq_elems as u32,
            kk as u32,
            conv_dim as u32,
            d_conv as u32,
            qk_ch,
            kd as u32,
            1e-6,
            qkvz_size as u32,
            qkvz_size as u32,
            n as u32,
            lazy,
            stream,
        )?;
        // 2026-10-01: The GDN output goes to each row's `value_dim` tail, as in the strided arm.
        let gdn_out = rows.offset(conv_dim * fp32);
        ops::gdn_exact_carry(
            ctx.gpu,
            self.carry.exact_kernel_for(kk, lazy),
            h_table,
            rows,
            rows.offset(key_dim * fp32),
            rows.offset(key_dim * 2 * fp32),
            gates_buf,
            gates_buf.offset(nv * fp32),
            gdn_out,
            binding.stash,
            slot_tab,
            binding.pend,
            binding.seq_floats as u32,
            n as u32,
            nk as u32,
            nv as u32,
            kd as u32,
            [
                qkvz_size as u32,
                qkvz_size as u32,
                (nv * 2) as u32,
                qkvz_size as u32,
            ],
            flag,
            stream,
        )?;
        ops::gated_rms_norm_strided(
            ctx.gpu,
            self.gated_rms_norm_f32_strided_k,
            gdn_out,
            deinterleaved.offset(conv_dim * bf16),
            &self.ssm.norm,
            normed_out,
            nv as u32,
            (n * kk) as u32,
            vd as u32,
            vd as u32,
            ctx.config.rms_norm_eps as f32,
            vd as u32,
            qkvz_size as u32,
            qkvz_size as u32,
            value_dim as u32,
            stream,
        )?;
        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| {
            tracing::info!(
                "EXACT batched MTP verify CARRIED (n={n}, k={kk}): per-token chain with the \
                 state read once and accepted rows stashed, no h intermediates written"
            );
        });
        Ok(true)
    }
}
