// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: `METRALE_GLM_KDA_CHUNK64_CONV=1`: the 64-token chunked scan (`chunk64.rs`) runs
//! the KDA conv inside its prepare: `kda_chunk64_prepare_conv`
//! (`kernels/gb10/common/kda_chunk64_prepare.cu`, the tile in `kda_chunk64_conv.cuh`) convolves
//! each (chunk, head) tile's q/k/v rows from the projections, writes them to a slab-local stage
//! (`conv_out` rows `0 .. CHUNK64_SLAB * 64`, reused every slab, so it stays in L2) and its k
//! straight into the prepare's shared K tile, instead of one `conv_tokens` launch writing every
//! conv row (and, under `METRALE_GLM_KDA_CONV_PACK=1`, the packed rows) to DRAM for the
//! prepare to read back. `causal_conv1d_window_advance_parts` advances the conv window after
//! the last slab, from the projections where they lie.
//!
//! Owner: model-arch (GLM-5.3-Flash KDA).
//! Invariants:
//! - Every byte of `core`, the recurrent state, the conv state and the slab factors equals the
//!   unfused path's (`conv_tokens` then `kda_chunk64_prepare`): the tile runs
//!   `causal_conv1d_update_l2norm_tokens_parts`'s expressions in its order (the kernel comment
//!   has the argument); `examples/glm5next_kda_chunk64_conv_parity.rs` compares them on a GPU.
//! - The prepares read the conv window before it moves: the advance is the last launch.
//! - Under [`Pack::InConv`] nothing writes `qkv_proj` (the pack is skipped and the fused conv
//!   does not store it), so a verify record of such a forward is refused
//!   ([`Glm5NextKdaLayer::chunk64_conv_skips_pack`]).
//! - Refused (before any launch) without both kernels, or at a shape the tile does not take:
//!   conv width 4, head_dim 128, q|k channels twice the q channels, q channels a multiple of
//!   256 (whole heads per 256-channel block of the parts kernel, which the tile mirrors).

use super::*;

/// 2026-10-14: The conv width `kda_chunk64_prepare_conv` is instantiated for.
pub(crate) const CHUNK64_CONV_WIDTH: usize = 4;

/// 2026-10-14: `METRALE_GLM_KDA_CHUNK64_CONV`, read once: `1` fuses, unset or `0` does not.
pub(crate) fn kda_chunk64_conv() -> Result<bool> {
    static V: std::sync::OnceLock<Result<bool, String>> = std::sync::OnceLock::new();
    V.get_or_init(|| {
        parse_chunk64_conv(
            std::env::var("METRALE_GLM_KDA_CHUNK64_CONV")
                .ok()
                .as_deref(),
        )
    })
    .clone()
    .map_err(anyhow::Error::msg)
}

/// 2026-10-14: [`kda_chunk64_conv`]'s parse, pure.
pub(crate) fn parse_chunk64_conv(v: Option<&str>) -> Result<bool, String> {
    match v {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(s) => Err(format!(
            "METRALE_GLM_KDA_CHUNK64_CONV={s:?}: expected 0 or 1"
        )),
    }
}

/// 2026-10-14: The shapes the fused tile takes, pure: `None` when it takes them, else why not.
pub(crate) fn chunk64_conv_shape_refusal(
    d_conv: usize,
    head_dim: usize,
    qk_channels: usize,
    part: usize,
) -> Option<String> {
    if d_conv != CHUNK64_CONV_WIDTH {
        return Some(format!("conv width {d_conv}, not {CHUNK64_CONV_WIDTH}"));
    }
    if head_dim != CHUNK64_D {
        return Some(format!("head_dim {head_dim}, not {CHUNK64_D}"));
    }
    if qk_channels != 2 * part || part == 0 || !part.is_multiple_of(256) {
        return Some(format!(
            "q|k channels {qk_channels} over q channels {part}: needs twice, a multiple of 256"
        ));
    }
    None
}

impl Glm5NextKdaLayer {
    /// 2026-10-14: Whether the 64-token scan takes the fused conv: the lever is set and the
    /// kernels and shape allow it. The lever set where they do not is an error.
    pub(crate) fn chunk64_conv_on(&self) -> Result<bool> {
        if !kda_chunk64_conv()? {
            return Ok(false);
        }
        self.chunk64_conv_check()?;
        Ok(true)
    }

    /// 2026-10-14: The refusals of the fused path, before any launch.
    pub(crate) fn chunk64_conv_check(&self) -> Result<()> {
        let (s, c) = (&self.kernels.seq, &self.cfg);
        if s.chunk64_prepare_conv.0 == 0 || s.conv_window_parts.0 == 0 {
            bail!(
                "METRALE_GLM_KDA_CHUNK64_CONV=1: needs kda_chunk64_prepare_conv and \
                 causal_conv1d_window_advance_parts"
            );
        }
        match chunk64_conv_shape_refusal(c.conv_kernel, c.head_dim, c.qk_channels(), c.qkv_dim()) {
            Some(why) => bail!("METRALE_GLM_KDA_CHUNK64_CONV=1: {why}"),
            None => Ok(()),
        }
    }

    /// 2026-10-14: Whether a forward of `k` rows of one sequence leaves `qkv_proj` unwritten:
    /// the 64-token scan applies with the fused conv, and the pack would have been the conv's.
    /// `record_verify_rows` refuses such a forward (it copies `qkv_proj`).
    pub(crate) fn chunk64_conv_skips_pack(&self, k: usize) -> Result<bool> {
        Ok(kda_chunk64_conv()?
            && self.chunk64_applies(k, kda_chunk64_min_rows()?)
            && self.pack_for(true)? == Pack::InConv)
    }

    /// 2026-10-14: The q/k/v projections of rows `0..k` and their row stride: the three parts
    /// (`qkv_parts`, `[k, qkv]` each) when the pack is the conv's, else the packed rows.
    fn chunk64_conv_inputs(
        &self,
        k: usize,
        ws: &Glm5NextKdaWorkspace,
        pack: Pack,
    ) -> (DevicePtr, DevicePtr, DevicePtr, usize) {
        let (qkv, cd) = (self.cfg.qkv_dim(), self.cfg.conv_dim());
        match pack {
            Pack::InConv => (
                ws.qkv_parts,
                ws.qkv_parts.offset(k * qkv * 2),
                ws.qkv_parts.offset(2 * k * qkv * 2),
                qkv,
            ),
            Pack::InFrontEnd => (
                ws.qkv_proj,
                ws.qkv_proj.offset(qkv * 2),
                ws.qkv_proj.offset(2 * qkv * 2),
                cd,
            ),
        }
    }

    /// 2026-10-14: One slab's fused prepare: chunks `c0 .. c0 + n` of rows `0..k`, the stage at
    /// `conv_out` row 0, the factors in the scratch `kda_chunk64_prepare` writes.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn chunk64_prepare_conv(
        &self,
        gpu: &dyn GpuBackend,
        k: usize,
        c0: usize,
        n: usize,
        state: &KdaSeqState,
        ws: &Glm5NextKdaWorkspace,
        pack: Pack,
        stream: u64,
    ) -> Result<()> {
        let c = &self.cfg;
        let (qkv, cd, h) = (c.qkv_dim(), c.conv_dim(), c.heads);
        let (xq, xk, xv, in_stride) = self.chunk64_conv_inputs(k, ws, pack);
        KernelLaunch::new(gpu, self.kernels.seq.chunk64_prepare_conv)
            .grid([n as u32, h as u32, 1])
            .block([CHUNK64_PREP_THREADS, 1, 1])
            .shared_mem(CHUNK64_PREP_SMEM)
            .arg_ptr(xq)
            .arg_ptr(xk)
            .arg_ptr(xv)
            .arg_u32(in_stride as u32)
            .arg_ptr(state.conv)
            .arg_ptr(self.weights.conv.weight)
            .arg_ptr(DevicePtr::NULL)
            .arg_u32(c.conv_kernel as u32)
            .arg_f32(c.l2_eps)
            .arg_ptr(ws.conv_out)
            .arg_ptr(ws.gate)
            .arg_ptr(ws.beta)
            .arg_ptr(ws.chunk_u)
            .arg_ptr(ws.chunk_w)
            .arg_ptr(ws.q_f32)
            .arg_ptr(ws.k_f32)
            .arg_ptr(ws.v_f32)
            .arg_ptr(ws.chunk_gc)
            .arg_u32(h as u32)
            .arg_u32(k as u32)
            .arg_u32(c0 as u32)
            .arg_u32(cd as u32)
            .arg_u32(qkv as u32)
            .arg_f32(1.0 / (c.head_dim as f32).sqrt())
            .launch(stream)
    }

    /// 2026-10-14: The conv window past rows `0..k`, read from the projections the fused
    /// prepares read; launched after the last slab.
    pub(crate) fn chunk64_conv_window(
        &self,
        gpu: &dyn GpuBackend,
        k: usize,
        state: &KdaSeqState,
        ws: &Glm5NextKdaWorkspace,
        pack: Pack,
        stream: u64,
    ) -> Result<()> {
        let c = &self.cfg;
        let cd = c.conv_dim();
        let (xq, xk, xv, in_stride) = self.chunk64_conv_inputs(k, ws, pack);
        KernelLaunch::new(gpu, self.kernels.seq.conv_window_parts)
            .grid([div_ceil(cd as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(state.conv)
            .arg_ptr(xq)
            .arg_ptr(xk)
            .arg_ptr(xv)
            .arg_u32(cd as u32)
            .arg_u32(c.qkv_dim() as u32)
            .arg_u32(in_stride as u32)
            .arg_u32(c.conv_kernel as u32)
            .arg_u32(k as u32)
            .launch(stream)
    }
}

#[cfg(test)]
#[path = "chunk64_conv_tests.rs"]
mod tests;
