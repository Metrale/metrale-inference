// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: The exact MTP verify on an FP16 h-state: the FP32 conv chain
//! (`gdn_conv_chain_f32`, or `gdn_conv_chain_f32_batched` for a run of sequences), then
//! `gdn_exact_chain_f16_{K}`, which runs the FP16 fused-norm decode's chain for every row with the
//! state in registers and writes the FP16 intermediates, the final state and the normed rows.
//!
//! Owner: model-layers (Qwen3 SSM layer).
//! Invariants:
//! - Rows are the decode's bits: each twin runs its parent's arithmetic in its parent's order
//!   (`exact_chain_f16_microtest`), and the FP16 decode always runs the fused norm, as the twin
//!   does.
//! - The single-sequence form refuses (an error, not a fallback) when its twins are absent; the
//!   boot audit already refuses a kernel set without them (`carry.rs`). The run form declines
//!   (`Ok(false)`, nothing launched) when the layout checks fail, and each sequence then takes
//!   the single-sequence form.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::trait_decode_batched_conv_gdn::ConvGdnArgs;
use super::trait_decode_batched_conv_gdn_exact::exact_row;
use super::{Qwen3SsmLayer, SsmLayerState};
use crate::layer::{LayerState, VERIFY_WY_TABLE_STRIDE_BYTES};
use crate::layers::ops;

impl Qwen3SsmLayer {
    /// 2026-10-01: The FP16 chain kernel for `kk` rows, or an error naming what is missing.
    fn chain_f16_kernel(&self, kk: usize) -> Result<metrale_gpu_runtime::gpu::KernelHandle> {
        let k = &self.carry.kernels;
        let h = match kk {
            2..=4 => k.chain_f16[kk - 2],
            _ => metrale_gpu_runtime::gpu::KernelHandle(0),
        };
        if h.0 == 0 {
            anyhow::bail!(
                "exact MTP verify on an FP16 h-state: no gdn_exact_chain_f16_{kk} in this kernel \
                 set (K = 2..4 only). Run --ssm-h-dtype f32, or drop the fixed GDN format and \
                 --exact-verify"
            );
        }
        Ok(h)
    }

    /// 2026-10-01: The single-sequence form (see the module header). Afterwards `h_state`,
    /// `conv_state`, intermediates 0..K-2 and the normed rows hold what K FP16 decode steps
    /// write.
    pub(super) fn decode_batched_conv_gdn_exact_f16(
        &self,
        ssm_state: &mut SsmLayerState,
        ctx: &crate::layer::ForwardContext,
        args: &ConvGdnArgs,
    ) -> Result<()> {
        let kk = args.num_tokens;
        let gdn = self.chain_f16_kernel(kk)?;
        let conv = self.carry.kernels.conv_chain;
        if conv.0 == 0
            || ssm_state.h_state_intermediates.len() + 1 < kk
            || ssm_state.conv_state_intermediates.len() + 1 < kk
        {
            anyhow::bail!(
                "exact MTP verify on an FP16 h-state: gdn_conv_chain_f32 missing or fewer than \
                 {} intermediates (h={}, conv={})",
                kk - 1,
                ssm_state.h_state_intermediates.len(),
                ssm_state.conv_state_intermediates.len()
            );
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
            fp32,
            stream,
            ..
        } = *args;
        // 2026-10-01: Intermediates rows 0..K-2 write; a row that has none gets a valid pointer
        // the kernels never write through.
        let inter = |v: &[DevicePtr], fallback: DevicePtr| -> [DevicePtr; 3] {
            std::array::from_fn(|t| if t + 1 < kk { v[t] } else { fallback })
        };
        let rows = ctx.buffers.ssm_conv_out_f32();
        let row0 = exact_row(0, kk, qkvz_size, conv_dim, value_dim, nv);
        ops::gdn_conv_chain_f32(
            ctx.gpu,
            conv,
            ssm_state.conv_state,
            deinterleaved,
            &self.ssm.conv1d,
            rows,
            inter(&ssm_state.conv_state_intermediates, ssm_state.conv_state),
            kk as u32,
            conv_dim as u32,
            d_conv as u32,
            qk_ch,
            kd as u32,
            1e-6,
            qkvz_size as u32,
            qkvz_size as u32,
            stream,
        )?;
        let [i0, i1, i2] = inter(&ssm_state.h_state_intermediates, ssm_state.h_state);
        ops::gdn_exact_chain_f16(
            ctx.gpu,
            gdn,
            ops::F16ChainStates::Single([ssm_state.h_state, i0, i1, i2]),
            rows,
            rows.offset(key_dim * fp32),
            rows.offset(key_dim * 2 * fp32),
            gates_buf.offset(row0.gate),
            gates_buf.offset(row0.beta),
            deinterleaved.offset(row0.z),
            self.ssm.norm.weight,
            normed_out,
            1,
            [nk as u32, nv as u32, kd as u32],
            [
                qkvz_size as u32,
                qkvz_size as u32,
                (nv * 2) as u32,
                qkvz_size as u32,
                value_dim as u32,
            ],
            ctx.config.rms_norm_eps as f32,
            stream,
        )?;
        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| {
            tracing::info!(
                "EXACT MTP verify on the FP16 h-state (k={kk}): FP32 conv chain + \
                 gdn_exact_chain_f16, the FP16 fused-norm decode's bits"
            );
        });
        Ok(())
    }

    /// 2026-10-01: The run form for n >= 2 sequences with equal K: one batched conv chain and one
    /// FP16 chain over the n * K rows (row `b * K + t`), the states through the run's WY tables
    /// (`wy_tables`: h, then intermediate t at `(t + 1) * VERIFY_WY_TABLE_STRIDE_BYTES`). Needs
    /// the conv states on consecutive slots and the conv intermediates `conv_bytes` apart within
    /// a sequence and evenly spaced across sequences; otherwise `Ok(false)`.
    pub(super) fn decode_batched_conv_gdn_multi_exact_f16(
        &self,
        states: &mut [&mut (dyn LayerState + 'static)],
        wy_tables: DevicePtr,
        ctx: &crate::layer::ForwardContext,
        args: &ConvGdnArgs,
    ) -> Result<bool> {
        let n = states.len();
        let kk = args.num_tokens;
        let conv_bytes = self.conv_state_bytes;
        let conv_k = self.carry.kernels.conv_chain_batched;
        if n < 2 || wy_tables.is_null() || conv_k.0 == 0 {
            return Ok(false);
        }
        let gdn = self.chain_f16_kernel(kk)?;
        let mut conv_base = DevicePtr::NULL;
        let mut inter_base = DevicePtr::NULL;
        let mut inter_seq_stride = 0u64;
        for (i, state) in states.iter().enumerate() {
            let Some(st) = state.as_any().downcast_ref::<SsmLayerState>() else {
                return Ok(false);
            };
            if st.conv_state_intermediates.len() + 1 < kk || st.h_state_intermediates.len() + 1 < kk
            {
                return Ok(false);
            }
            let i0 = st.conv_state_intermediates[0];
            for t in 1..kk - 1 {
                if st.conv_state_intermediates[t].0 != i0.0 + (t * conv_bytes) as u64 {
                    return Ok(false);
                }
            }
            if i == 0 {
                conv_base = st.conv_state;
                inter_base = i0;
            } else {
                if st.conv_state.0 != conv_base.0 + (i * conv_bytes) as u64 {
                    return Ok(false);
                }
                if i == 1 {
                    inter_seq_stride = i0.0.wrapping_sub(inter_base.0);
                    // 2026-10-01: The next sequence's intermediates must not overlap this one's.
                    if inter_seq_stride < ((kk - 1) * conv_bytes) as u64 {
                        return Ok(false);
                    }
                } else if i0.0 != inter_base.0 + (i as u64) * inter_seq_stride {
                    return Ok(false);
                }
            }
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
            bf16,
            fp32,
            stream,
            ..
        } = *args;
        let rows = ctx.buffers.ssm_conv_out_f32();
        ops::gdn_conv_chain_f32_batched(
            ctx.gpu,
            conv_k,
            conv_base,
            deinterleaved,
            &self.ssm.conv1d,
            rows,
            inter_base,
            n as u32,
            kk as u32,
            [conv_dim as u32, d_conv as u32, qk_ch, kd as u32],
            1e-6,
            [qkvz_size as u32, qkvz_size as u32],
            [(conv_bytes / 4) as u64, inter_seq_stride / 4],
            stream,
        )?;
        let table = |t: usize| wy_tables.offset(t * VERIFY_WY_TABLE_STRIDE_BYTES);
        ops::gdn_exact_chain_f16(
            ctx.gpu,
            gdn,
            ops::F16ChainStates::Tables([table(0), table(1), table(2), table(3)]),
            rows,
            rows.offset(key_dim * fp32),
            rows.offset(key_dim * 2 * fp32),
            gates_buf,
            gates_buf.offset(nv * fp32),
            deinterleaved.offset(conv_dim * bf16),
            self.ssm.norm.weight,
            normed_out,
            n as u32,
            [nk as u32, nv as u32, kd as u32],
            [
                qkvz_size as u32,
                qkvz_size as u32,
                (nv * 2) as u32,
                qkvz_size as u32,
                value_dim as u32,
            ],
            ctx.config.rms_norm_eps as f32,
            stream,
        )?;
        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| {
            tracing::info!(
                "EXACT batched MTP verify on the FP16 h-state (n={n}, k={kk}): batched FP32 conv \
                 chain + gdn_exact_chain_f16 over the run's WY tables"
            );
        });
        Ok(true)
    }
}
