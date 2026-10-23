// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The building blocks of `Glm5NextLayer`: norm, mixer, MLP, all-reduce, the MTP
//! block's plain residual path, and the state accessors.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use super::*;
use crate::glm5next_layer_dump::tap as dump;

mod ab_levers;
mod drafter;
mod forward;
mod forward_pieces;
mod forward_sp;
mod forward_sp_pipe;
mod mhc_slice;
mod mix;
mod mhc_site;
mod mlp;
pub(crate) use mhc_site::{LayerIo, MhcSite};
mod multi_seq;
mod prefill_capture;
pub use multi_seq::{GroupSpan, group_spans};
mod prefill_spans;
pub use prefill_spans::{SpanPiece, prefill_span_groups};
mod reduce_post;
mod replay;
mod verify_multi;

impl Glm5NextLayer {
    /// 2026-10-13: This layer as the layer dump sees it (`glm5next_layer_dump`); `topk` when the
    /// launch's rows own the DSA selection output (not a multi-piece mixer call).
    /// 2026-10-22: `moe_ids` on a routed MLP: its top-k runs once over every row of the launch,
    /// pieces or not (`forward_moe_pieces`), so the launch's rows always own it.
    pub(in crate::glm5next_layer) fn dump_layer(
        &self,
        mhc: &Glm5NextMhc,
        topk: bool,
    ) -> dump::Layer {
        let topk = match (&self.mixer, topk) {
            (Glm5NextMixer::Dsa(l), true) => {
                Some((l.workspace.selection_tokens(), l.cfg.out_width()))
            }
            _ => None,
        };
        dump::Layer {
            idx: self.layer_idx,
            hidden: self.hidden,
            hc: mhc.hc_mult,
            topk,
            moe_ids: matches!(self.mlp, Glm5NextMlpSite::Moe(_))
                .then(|| (self.mlp_ws.ids(), self.mlp_cfg.top_k)),
        }
    }

    /// 2026-10-13: The summed mixer output after the attention site's post-mix, when the layer
    /// all-reduces it: the reduce sums in place, and the fused post-mix writes the sum back.
    pub(in crate::glm5next_layer) fn dump_mixer_red(
        &self,
        tap: Option<&dump::Tap<'_>>,
        attn_out: DevicePtr,
    ) -> Result<()> {
        if !self.mixer_all_reduce {
            return Ok(());
        }
        dump::put(tap, dump::Point::MixerRed, attn_out)
    }

    /// 2026-10-13: After the FFN site's post-mix: the summed MLP output (when the MLP reduces)
    /// and the layer's output streams.
    pub(in crate::glm5next_layer) fn dump_layer_out(
        &self,
        tap: Option<&dump::Tap<'_>>,
        ffn_out: DevicePtr,
        streams: DevicePtr,
    ) -> Result<()> {
        if self.mlp_cfg.needs_all_reduce() {
            dump::put(tap, dump::Point::MlpRed, ffn_out)?;
        }
        dump::put(tap, dump::Point::StreamsOut, streams)
    }

    /// 2026-09-25: Run the mixer on `normed`; returns the pointer that holds its output.
    #[allow(clippy::too_many_arguments)]
    fn mixer_forward(
        &self,
        normed: DevicePtr,
        residual: DevicePtr,
        st: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_offloaded: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        match &self.mixer {
            Glm5NextMixer::Kda { layer, ws, .. } => {
                let ssm = self.kda_state(st)?;
                let kda = KdaSeqState {
                    conv: ssm.conv_state,
                    recurrent: ssm.h_state,
                };
                let t = self.levers.profile.start();
                layer.decode(ctx.gpu, normed, &kda, ws, stream)?;
                profile::end(profile::KDA, t, ctx.gpu, stream);
                Ok(ws.final_out)
            }
            Glm5NextMixer::Dsa(layer) => {
                let dsa: &mut Glm5NextDsaState = self.dsa_state(st)?;
                layer.decode(
                    normed,
                    residual,
                    dsa,
                    kv_cache,
                    seq_len,
                    block_table,
                    disk_block_ids,
                    disk_offloaded,
                    ctx,
                    stream,
                )?;
                // 2026-09-25: DSA's `decode` writes its output projection over its input buffer.
                Ok(normed)
            }
        }
    }

    /// 2026-09-25: While profiling, a 2-byte all-reduce timed into bucket `bar`, then the
    /// route-trace line for `site`. No-op when profiling is off or there is no communicator.
    fn reduce_probe(&self, bar: usize, site: &str, ctx: &ForwardContext, stream: u64) {
        if !self.levers.profile.on() {
            return;
        }
        let Some(comm) = ctx.comm else { return };
        let t = self.levers.profile.start_hot();
        let p = profile::probe_buf(ctx.gpu);
        if p != 0 {
            let _ = comm.all_reduce_async(p, 2, stream);
        }
        let us = profile::end_us(bar, t, ctx.gpu, stream);
        profile::trace_bar(
            self.levers.profile,
            site,
            self.layer_idx,
            matches!(self.mlp, Glm5NextMlpSite::Moe(_)),
            us,
        );
    }

    /// 2026-09-25: `dst += src` over `n` BF16 elements (`bf16_add_inplace`); errors when the
    /// target lacks the kernel.
    fn add_inplace(
        &self,
        gpu: &dyn GpuBackend,
        dst: DevicePtr,
        src: DevicePtr,
        n: usize,
        stream: u64,
    ) -> Result<()> {
        if self.add_k.0 == 0 {
            bail!(
                "GLM layer {}: bf16_add_inplace is not loaded on this target; the MTP layer's \
                 plain residual path needs it",
                self.layer_idx
            );
        }
        KernelLaunch::new(gpu, self.add_k)
            .grid([(n as u32).div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(dst)
            .arg_ptr(src)
            .arg_i32(n as i32)
            .launch(stream)
    }

    /// 2026-09-25: One token through a block with no hyper-connection (`mhc: None`, the MTP
    /// block): `x += mixer(norm(x))`, then `x += mlp(norm(x))`, each all-reduce before its add.
    #[allow(clippy::too_many_arguments)]
    fn forward_one_plain(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        st: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_offloaded: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let gpu = ctx.gpu;
        let h = self.hidden;
        let normed = ctx.buffers.norm_output();
        let ffn_out = ctx.buffers.moe_output();

        self.norm(gpu, hidden, self.input_norm, normed, 1, stream)?;
        let attn_out = self.mixer_forward(
            normed,
            residual,
            st,
            kv_cache,
            seq_len,
            block_table,
            disk_block_ids,
            disk_offloaded,
            ctx,
            stream,
        )?;
        if self.mixer_all_reduce {
            self.reduce_partial(attn_out, 1, ctx, stream)?;
        }
        self.add_inplace(gpu, hidden, attn_out, h, stream)?;

        self.norm(gpu, hidden, self.post_attn_norm, normed, 1, stream)?;
        // 2026-10-10: No post-mix here to take the sum, so the all-reduce stays whole.
        self.mlp_forward(normed, ffn_out, 1, &[1], false, ctx, stream, None)?;
        self.add_inplace(gpu, hidden, ffn_out, h, stream)
    }

    /// 2026-09-25: This KDA layer's state as the SSM pool's `SsmLayerState`. Errors when the
    /// state is another type, or when its recurrent state is FP16: `h_is_f16` is set, or an FP32
    /// staging blob is attached, which only an FP16-sized pool has.
    pub(super) fn kda_state<'a>(
        &self,
        state: &'a mut dyn LayerState,
    ) -> Result<&'a mut SsmLayerState> {
        let st = state
            .as_any_mut()
            .downcast_mut::<SsmLayerState>()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "GLM layer {}: a KDA mixer was handed state that is not an SsmLayerState",
                    self.layer_idx
                )
            })?;
        if st.h_is_f16 || st.h_prefill_stage.is_some() {
            bail!(
                "GLM layer {}: KDA recurrent state is FP32-only; --ssm-h-dtype f16 narrowed it",
                self.layer_idx
            );
        }
        Ok(st)
    }

    /// 2026-09-25: This DSA layer's `Glm5NextDsaState`; errors on any other state type.
    pub(super) fn dsa_state<'a>(
        &self,
        state: &'a mut dyn LayerState,
    ) -> Result<&'a mut Glm5NextDsaState> {
        state
            .as_any_mut()
            .downcast_mut::<Glm5NextDsaState>()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "GLM layer {}: a DSA mixer was handed state that is not a Glm5NextDsaState",
                    self.layer_idx
                )
            })
    }
}
