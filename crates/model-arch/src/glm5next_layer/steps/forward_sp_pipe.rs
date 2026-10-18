// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: `METRALE_GLM_SP_PIPELINE=1`: a run of SP-mHC prefill sub-chunks of one segment
//! through a layer pipelined (`glm5next_sp_pipeline`). [`Glm5NextLayer::forward_sp_pipeline`]
//! runs the enqueues of `glm5next_sp_pipeline::program` on the caller's stream and the SP comm
//! stream: each sub-chunk's sections by `glm5next_sp_overlap::run_section` deferred into the
//! program's events, its mixer by `mix`, its MLP by `mlp_compute`, and its reduce-scatters
//! whole-shard on the comm stream.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Taken only from `prefill_segments` for a leading run of two or more sub-chunks of one
//!   segment that each take the SP body (`sp_takes`); every rank decides from the segment's
//!   rows, the world size and the levers alone, so all ranks run the same program and their
//!   collectives pair.
//! - Per sub-chunk, the launches of `forward_rows_sp` (and `forward_k`'s mixer) with the same
//!   arguments, through the same functions (`sp_head_piece`, `sp_mid_piece`, `sp_tail_piece`,
//!   `mix`): only the normed rows' and the MLP partials' buffers and the host order across
//!   sub-chunks differ (the module header of `glm5next_sp_pipeline`).
//! - Leaves the comm stream joined into the caller's stream.

use metrale_comm::PeerPartials;

use super::forward_sp::SpRows;
use super::*;
use crate::glm5next_sp_mhc as sp;
use crate::glm5next_sp_overlap::{self as ov, On};
use crate::glm5next_sp_pipeline::{self as pipe, Prim, RsKind, Stage};

/// 2026-10-14: One sub-chunk of the run: its rows, its views and pieces, and what its earlier
/// stages left for its later ones.
struct Sub {
    k: usize,
    hidden: DevicePtr,
    seq_len: usize,
    normed: DevicePtr,
    rows: SpRows,
    shards: Vec<(usize, usize)>,
    pieces: Vec<Vec<(usize, usize)>>,
    attn_out: Option<DevicePtr>,
    /// 2026-10-14: The received partials of each reduce-scatter, by `pipe::recv_slot`.
    received: [Option<PeerPartials>; 2],
}

impl Glm5NextLayer {
    /// 2026-10-14: The sub-chunks `ks` of a segment's leading rows through this layer, pipelined:
    /// row 0 at `hidden`, position `seq_len` and highway slot `slot_base`; sub-chunk `s` at the
    /// rows after those of the sub-chunks before it, exactly as `prefill_segments` hands them
    /// to `forward_k` one by one.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::glm5next_layer) fn forward_sp_pipeline(
        &self,
        hidden: DevicePtr,
        ks: &[usize],
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
        slot_base: usize,
    ) -> Result<()> {
        let Some(mhc) = self.mhc.as_ref() else {
            bail!("GLM layer {}: no hyper-connection bound", self.layer_idx);
        };
        let Some(comm) = ctx.comm else {
            bail!(
                "GLM layer {}: SP pipeline without a communicator",
                self.layer_idx
            );
        };
        anyhow::ensure!(
            ks.len() >= pipe::MIN_SUBS && mhc.kernels.hc_post_rank_sum.0 != 0,
            "GLM layer {}: SP pipeline over {} sub-chunks (needs {}) or without \
             glm5next_hc_post_rank_sum on this target",
            self.layer_idx,
            ks.len(),
            pipe::MIN_SUBS
        );
        let (gpu, h, hc) = (ctx.gpu, self.hidden, mhc.hc_mult);
        let (row_bytes, world) = (h * 2, comm.world_size());
        let (ws, pw, ovs) = (
            sp::workspace_get()?,
            pipe::workspace_get()?,
            ov::streams_get()?,
        );
        let overlap_on = self.levers.mhc.sp_overlap;
        let post_pre = self.levers.mhc.post_pre_fused;
        let reduce = self.mlp_cfg.needs_all_reduce();
        let ffn_out = ctx.buffers.moe_output();
        let mut subs = Vec::with_capacity(ks.len());
        let mut r = 0;
        for (s, &k) in ks.iter().enumerate() {
            let normed = pw.normed_at(ctx.buffers.norm_output(), s, k * row_bytes)?;
            let sub_hidden = hidden.offset(r * row_bytes);
            let shards = sp::shards(k, world);
            let pieces = ov::pieces(&shards, ov::piece_count(overlap_on, k, world));
            subs.push(Sub {
                k,
                hidden: sub_hidden,
                seq_len: seq_len + r,
                normed,
                rows: SpRows::new(ctx, hc, h, slot_base + r, sub_hidden, normed),
                shards,
                pieces,
                attn_out: None,
                received: [None; 2],
            });
            r += k;
        }
        let on = |o: On| match o {
            On::Main => stream,
            On::Comm => ovs.comm,
        };
        for prim in pipe::program(ks.len(), reduce, self.is_last) {
            match prim {
                Prim::Record(ev, o) => gpu.record_event(pw.event(ev), on(o))?,
                Prim::Wait(o, ev) => gpu.stream_wait_event(on(o), pw.event(ev))?,
                Prim::Mixer(s) => {
                    let sub = &subs[s];
                    let t = self.levers.profile.start();
                    let out = self.mix(
                        sub.normed,
                        sub.k,
                        state,
                        kv_cache,
                        sub.seq_len,
                        block_table,
                        ctx,
                        stream,
                        false,
                        true,
                        &[],
                    )?;
                    profile::end(profile::KDA, t, gpu, stream);
                    subs[s].attn_out = Some(out);
                }
                Prim::Mlp(s) => {
                    let sub = &subs[s];
                    self.mlp_compute(sub.normed, ffn_out, sub.k, &[sub.k], ctx, stream)?;
                }
                Prim::Rs(kind, s) => {
                    let sub = &subs[s];
                    let (src, bucket) = match kind {
                        RsKind::Mixer => (
                            sub.attn_out
                                .context("SP pipeline: a reduce-scatter before its mixer")?,
                            profile::REDUCE_ATTN,
                        ),
                        RsKind::Mlp => (ffn_out, profile::REDUCE_MLP),
                    };
                    let t = self.levers.profile.start_hot();
                    let got = sp::reduce_scatter_rows(
                        comm,
                        &pw.recv_for(&ws, kind)?,
                        src,
                        &sub.shards,
                        0,
                        row_bytes,
                        ovs.comm,
                    )?;
                    profile::end(bucket, t, gpu, ovs.comm);
                    subs[s].received[pipe::recv_slot(kind)] = Some(got);
                }
                Prim::Section { stage, sub, done } => {
                    let sub = &subs[sub];
                    let (received, ag) = match stage {
                        Stage::Head => (None, Some((sub.normed, Some(profile::REDUCE_ATTN_ENQ)))),
                        Stage::Mid => (
                            sub.received[pipe::recv_slot(RsKind::Mixer)],
                            Some((sub.normed, Some(profile::REDUCE_MLP_ENQ))),
                        ),
                        Stage::Tail => (
                            sub.received[pipe::recv_slot(RsKind::Mlp)],
                            self.is_last.then_some((sub.hidden, None)),
                        ),
                        Stage::Mixer | Stage::Mlp => bail!("SP pipeline: {stage:?} as a section"),
                    };
                    anyhow::ensure!(
                        ag.is_some() == done.is_some(),
                        "SP pipeline: a {stage:?} section's all-gather and its event disagree"
                    );
                    if stage == Stage::Head
                        && self.is_first
                        && let Some(l2) = &self.l2_warm
                    {
                        l2.reset();
                    }
                    let attn_out = sub.attn_out;
                    ov::run_section(
                        gpu,
                        comm,
                        &ws,
                        Some(&ovs),
                        stream,
                        &ov::Section {
                            pieces: &sub.pieces,
                            rs: None,
                            ag,
                            row_bytes,
                            received,
                            defer: done.map(|d| pw.event(d)),
                            prof: self.levers.profile,
                        },
                        &mut |a, b, partials| match stage {
                            Stage::Head => self.sp_head_piece(gpu, mhc, &sub.rows, a, b, stream),
                            Stage::Mid => self.sp_mid_piece(
                                gpu,
                                mhc,
                                &sub.rows,
                                a,
                                b,
                                attn_out.context("SP pipeline: a post-mix before its mixer")?,
                                partials,
                                post_pre,
                                None,
                                stream,
                            ),
                            _ => self.sp_tail_piece(
                                gpu, mhc, &sub.rows, a, b, ffn_out, partials, reduce, false, stream,
                            ),
                        },
                    )?;
                    if stage == Stage::Tail && self.is_last {
                        if let Some(l2) = &self.l2_warm {
                            l2.join(gpu, stream)?;
                        }
                        profile::step(self.levers.profile);
                    }
                }
            }
        }
        Ok(())
    }
}
