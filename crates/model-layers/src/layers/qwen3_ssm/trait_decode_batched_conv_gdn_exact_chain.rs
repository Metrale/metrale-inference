// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: The per-row exact verify of one sequence in three launches
//! (`decode_batched_conv_gdn_exact_chain`): `gdn_conv_chain_f32` (the FP32 conv for every row,
//! its conv intermediates written inline), `gdn_exact_chain{K}` (the strided decode's chain with
//! the state read once, its h intermediates written inline) and one FP32-input gated norm over
//! the K rows.
//!
//! Owner: model-layers (Qwen3 SSM layer).
//! Invariants:
//! - It runs only where the per-row arm runs the FP32 conv, the FP32 GDN and the unfused norm
//!   (no `--gdn-fused-norm`), for K = 2..4 with the twins linked; otherwise it returns `false`
//!   having launched nothing and the per-row arm runs.
//! - Afterwards `h_state`, `conv_state`, intermediates 0..K-2, the FP32 conv rows and the normed
//!   rows hold what the per-row arm writes: each twin runs its parent's arithmetic in its
//!   parent's order (`exact_carry_microtest`), and `gated_rms_norm_f32_input_strided` runs
//!   `gated_rms_norm_f32_input`'s per-row code.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::trait_decode_batched_conv_gdn::ConvGdnArgs;
use super::trait_decode_batched_conv_gdn_exact::exact_row;
use super::{Qwen3SsmLayer, SsmLayerState};
use crate::layer::ForwardContext;
use crate::layers::ops;

impl Qwen3SsmLayer {
    /// 2026-10-01: See the module header. `true` when it ran.
    pub(super) fn decode_batched_conv_gdn_exact_chain(
        &self,
        ssm_state: &mut SsmLayerState,
        ctx: &ForwardContext,
        args: &ConvGdnArgs,
    ) -> Result<bool> {
        let kk = args.num_tokens;
        let k = &self.carry.kernels;
        let gdn = match kk {
            2..=4 => k.chain[kk - 2],
            _ => return Ok(false),
        };
        if gdn.0 == 0
            || k.conv_chain.0 == 0
            || self.conv1d_l2norm_f32_k.0 == 0
            || self.gdn_f32_k.0 == 0
            || self.gated_rms_norm_f32_k.0 == 0
            || self.gated_rms_norm_f32_strided_k.0 == 0
            || crate::layers::qwen3_ssm::gdn_fused_norm_enabled()
            || ssm_state.h_state_intermediates.len() + 1 < kk
            || ssm_state.conv_state_intermediates.len() + 1 < kk
        {
            return Ok(false);
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
            fp32,
            stream,
            ..
        } = *args;
        // 2026-10-01: The intermediates rows 0..K-2 write; a row that has none gets a valid
        // pointer the kernels never write through.
        let inter = |v: &[DevicePtr], fallback: DevicePtr| -> [DevicePtr; 3] {
            std::array::from_fn(|t| if t + 1 < kk { v[t] } else { fallback })
        };
        let rows = ctx.buffers.ssm_conv_out_f32();
        let row0 = exact_row(0, kk, qkvz_size, conv_dim, value_dim, nv);
        ops::gdn_conv_chain_f32(
            ctx.gpu,
            k.conv_chain,
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
        let gdn_out = rows.offset(row0.gdn_out_f32);
        ops::gdn_exact_chain(
            ctx.gpu,
            gdn,
            ssm_state.h_state,
            rows,
            rows.offset(key_dim * fp32),
            rows.offset(key_dim * 2 * fp32),
            gates_buf.offset(row0.gate),
            gates_buf.offset(row0.beta),
            gdn_out,
            inter(&ssm_state.h_state_intermediates, ssm_state.h_state),
            nk as u32,
            nv as u32,
            kd as u32,
            [
                qkvz_size as u32,
                qkvz_size as u32,
                (nv * 2) as u32,
                qkvz_size as u32,
            ],
            stream,
        )?;
        ops::gated_rms_norm_strided(
            ctx.gpu,
            self.gated_rms_norm_f32_strided_k,
            gdn_out,
            deinterleaved.offset(row0.z),
            &self.ssm.norm,
            normed_out,
            nv as u32,
            kk as u32,
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
                "EXACT single-sequence MTP verify CHAINED (k={kk}): one conv, one GDN and one \
                 norm launch for every row, the state read once"
            );
        });
        Ok(true)
    }
}
