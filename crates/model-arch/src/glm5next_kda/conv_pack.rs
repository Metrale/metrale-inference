// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-13: `METRALE_GLM_KDA_CONV_PACK=1`: the q/k/v pack (`kda_pack_qkv_bf16`) and the
//! per-sequence conv (`causal_conv1d_update_l2norm_tokens`) of rows `0..k` of one sequence in
//! one launch, `causal_conv1d_update_l2norm_tokens_parts` (`kda_conv_tokens.cu`).
//!
//! Owner: model-arch (GLM-5.3-Flash KDA).
//! Invariants:
//! - Every byte of `qkv_proj` and `conv_out` is the pack-then-conv pair's: the kernel stores
//!   the pack's copy of each input and runs the conv body's expressions in its order (its
//!   comment has the argument); `causal_conv1d_window_advance` then reads the `qkv_proj` it
//!   wrote, as it read the pack's.
//! - The grid comes from the SM count and the channel count, never from the row count, and a
//!   row's bytes do not depend on the grid.
//! - `front_end` skips the pack exactly when it is handed [`Pack::InConv`], and only callers
//!   that run [`Glm5NextKdaLayer::conv_tokens`] next, with nothing reading `qkv_proj` between,
//!   hand it that ([`Glm5NextKdaLayer::pack_for`]).

use super::*;

/// 2026-10-13: Blocks per SM the fused launch aims for: ptxas reports 48 registers and 512 B of
/// shared memory per 256-thread block on sm_121, so five fit; four leaves room for the stream's
/// neighbours.
const CONV_PACK_BLOCKS_PER_SM: u32 = 4;

/// 2026-10-13: Where a forward's q/k/v pack runs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Pack {
    /// `front_end` launches `kda_pack_qkv_bf16`.
    InFrontEnd,
    /// `conv_tokens` launches the fused kernel, which writes `qkv_proj` as the pack does.
    InConv,
}

/// 2026-10-13: Grid y of the fused launch: enough row spans that `grid_x` channel blocks fill
/// `sms` SMs at [`CONV_PACK_BLOCKS_PER_SM`]; at least one.
pub(crate) fn conv_pack_spans(sms: u32, grid_x: u32) -> u32 {
    (sms * CONV_PACK_BLOCKS_PER_SM)
        .div_ceil(grid_x.max(1))
        .max(1)
}

impl Glm5NextKdaLayer {
    /// 2026-10-13: The [`Pack`] for a forward whose conv is `conv_tokens` when
    /// `conv_tokens_follows`, else the per-row or chunked-prefill conv (which read the pack).
    /// Refuses the lever without the fused kernel or at a conv width it does not take.
    pub(super) fn pack_for(&self, conv_tokens_follows: bool) -> Result<Pack> {
        if !conv_tokens_follows || !self.levers.kda.conv_pack {
            return Ok(Pack::InFrontEnd);
        }
        if self.kernels.seq.conv_tokens_parts.0 == 0 || !(1..=4).contains(&self.cfg.conv_kernel) {
            bail!(
                "METRALE_GLM_KDA_CONV_PACK=1: needs causal_conv1d_update_l2norm_tokens_parts and \
                 a conv width of 1..=4 (width {})",
                self.cfg.conv_kernel
            );
        }
        Ok(Pack::InConv)
    }

    /// 2026-10-13: `front_end`'s q/k/v pack of rows `0..t` into `qkv_proj`, skipped under
    /// [`Pack::InConv`] (the conv that follows writes `qkv_proj` itself).
    pub(super) fn pack_qkv(
        &self,
        gpu: &dyn GpuBackend,
        t: usize,
        ws: &Glm5NextKdaWorkspace,
        pack: Pack,
        stream: u64,
    ) -> Result<()> {
        if pack == Pack::InConv {
            return Ok(());
        }
        let qkv = self.cfg.qkv_dim();
        KernelLaunch::new(gpu, self.kernels.pack)
            .grid([div_ceil(qkv as u32, 256), t as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(ws.qkv_parts)
            .arg_ptr(ws.qkv_parts.offset(t * qkv * 2))
            .arg_ptr(ws.qkv_parts.offset(2 * t * qkv * 2))
            .arg_ptr(ws.qkv_proj)
            .arg_u32(t as u32)
            .arg_u32(qkv as u32)
            .launch(stream)
    }

    /// 2026-10-13: The fused pack and conv of workspace rows `0..k`: reads the three
    /// `qkv_parts` projections, writes `qkv_proj` and `conv_out`. `conv_tokens` advances the
    /// window after it.
    pub(super) fn conv_tokens_parts(
        &self,
        gpu: &dyn GpuBackend,
        k: usize,
        state: &KdaSeqState,
        ws: &Glm5NextKdaWorkspace,
        stream: u64,
    ) -> Result<()> {
        let c = &self.cfg;
        let (qkv, cd) = (c.qkv_dim(), c.conv_dim());
        let grid_x = div_ceil(cd as u32, 256);
        KernelLaunch::new(gpu, self.kernels.seq.conv_tokens_parts)
            .grid([grid_x, conv_pack_spans(gpu.sm_count()?, grid_x), 1])
            .block([256, 1, 1])
            .arg_ptr(state.conv)
            .arg_ptr(ws.qkv_parts)
            .arg_ptr(ws.qkv_parts.offset(k * qkv * 2))
            .arg_ptr(ws.qkv_parts.offset(2 * k * qkv * 2))
            .arg_ptr(self.weights.conv.weight)
            .arg_ptr(DevicePtr::NULL)
            .arg_ptr(ws.qkv_proj)
            .arg_ptr(ws.conv_out)
            .arg_u32(cd as u32)
            .arg_u32(qkv as u32)
            .arg_u32(k as u32)
            .arg_u32(c.conv_kernel as u32)
            .arg_u32(c.qk_channels() as u32)
            .arg_u32(c.head_dim as u32)
            .arg_f32(c.l2_eps)
            .launch(stream)
    }
}

#[cfg(test)]
#[path = "conv_pack_tests.rs"]
mod tests;
