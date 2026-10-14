// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: `forward_prefill_spans`, the GLM layer's multi-sequence prefill pass
//! (`LayerSplitPrefill::prefill_spans`): several sequences' prompt chunks through the layer in
//! row groups that share the mHC, norm and MLP launches, with each sequence's mixer run as its
//! own `prefill` runs it.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Each sequence is cut into the pieces its own `prefill` runs: `prefill_rows()` rows from its
//!   first row on, the last piece shorter. A piece is never split across groups, so its mixer
//!   call (KDA `decode_k` or the chunked scan, DSA `decode_k` with `is_prefill`) has the same
//!   rows, position and state as in the single-sequence prefill.
//! - Groups hold whole pieces in pass order, at most `prefill_rows()` rows, so a sequence's
//!   pieces step its state in order and every group fits the mHC `mix` scratch and the MLP
//!   workspace the single-sequence prefill fits.
//! - Pieces share a group only where the layer's MLP computes a row the same whatever its
//!   batch-mates (`mlp_share_class`): a routed site only on a grouped GEMM (W4A16, or W4A4
//!   above 16 rows), the same one for the piece's own row count, the group's and every
//!   batch-mate's. Any other piece runs as a group of its own.
//! - Row `r` of the pass uses highway slot `ctx.hc_row_offset + r`.
//!
//! # Which launches a row shares with other sequences
//!
//! The group-wide ones: `hc_pre`/`hc_post`/`hc_head_mean` and the norms (one block per token,
//! nothing shared across tokens; `glm5next_c16_kernel_bench` checks the rows `hc_mix` against
//! the per-row one byte for byte), and in a routed MLP the top-k, the sort, the grouped expert
//! GEMM and the combine, each of which computes a row from its own inputs. The launches whose
//! arithmetic depends on the row count stay per piece: the router GEMM and the shared expert
//! (`forward_moe_pieces`), a dense MLP site, and the mixer and MLP all-reduces (so each
//! collective has the size the piece's own pass gives it). So a row's bits equal the
//! single-sequence prefill's; the routed site is measured byte for byte by
//! `examples/glm5next_prefill_spans_mlp_gate.rs`.

use super::prefill_capture::{CapturePoint, capture_points, kda_ordinal};
use super::*;
use crate::glm5next_layer::levers::kda_chunk_prefill;
use metrale_model_layers::layer::PrefillSpan;

/// 2026-10-14: The MLP path a group of pieces shares (`prefill_span_groups`' class).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MlpShare {
    /// A dense site: it runs per piece.
    Dense,
    /// A routed site on this shared path.
    Routed(crate::glm5next_mlp::forward::ExpertRoute),
}

/// 2026-10-09: One piece of a multi-sequence prefill pass: rows `t0..t0 + rows` of sequence
/// `seq`, at row `row` of the pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpanPiece {
    pub seq: usize,
    pub t0: usize,
    pub row: usize,
    pub rows: usize,
}

/// 2026-10-09: The row groups of a pass whose sequence `s` has `rows[s]` rows: every sequence
/// cut at multiples of `cap` from its first row, the pieces packed in order into groups of at
/// most `cap` rows. A piece shares a group only when `share` gives its own row count a class
/// and gives the group's total with it the same class; any other piece gets a group of its own,
/// so its launches are those of its own prefill. A `cap` of 0 is treated as 1; a sequence with
/// no rows has no piece.
///
/// 2026-10-14: `share` names a class (the MLP path a launch of that many rows takes), not a
/// yes/no: two shareable paths that compute a row differently must never meet in one group.
///
/// 2026-10-11: `cuts[s]` (ascending, inside the sequence; a missing entry is none) are the
/// sequence's in-pass capture points: the sequence is cut there too and its sub-chunk grid
/// restarts at each, as its own prefill splits the pass there (`steps/prefill_capture.rs`).
pub fn prefill_span_groups<C: PartialEq>(
    rows: &[usize],
    cuts: &[Vec<usize>],
    cap: usize,
    share: impl Fn(usize) -> Option<C>,
) -> Vec<Vec<SpanPiece>> {
    let cap = cap.max(1);
    let mut groups = Vec::new();
    let mut cur: Vec<SpanPiece> = Vec::new();
    let mut cur_rows = 0usize;
    let mut cur_class: Option<C> = None;
    let mut row = 0usize;
    for (seq, &n) in rows.iter().enumerate() {
        let seq_cuts = cuts.get(seq).map_or(&[][..], Vec::as_slice);
        let mut t0 = 0usize;
        while t0 < n {
            let bound = seq_cuts
                .iter()
                .copied()
                .find(|&c| c > t0 && c < n)
                .unwrap_or(n);
            let len = cap.min(bound - t0);
            let piece = SpanPiece {
                seq,
                t0,
                row,
                rows: len,
            };
            row += len;
            t0 += len;
            let Some(class) = share(len) else {
                if !cur.is_empty() {
                    groups.push(std::mem::take(&mut cur));
                    cur_rows = 0;
                }
                groups.push(vec![piece]);
                continue;
            };
            // 2026-10-14: A piece joins only a group of its own class whose total keeps it.
            if !cur.is_empty()
                && (cur_rows + len > cap
                    || cur_class.as_ref() != Some(&class)
                    || share(cur_rows + len).as_ref() != Some(&class))
            {
                groups.push(std::mem::take(&mut cur));
                cur_rows = 0;
            }
            cur.push(piece);
            cur_rows += len;
            cur_class = Some(class);
        }
    }
    if !cur.is_empty() {
        groups.push(cur);
    }
    groups
}

impl Glm5NextLayer {
    /// 2026-10-09: The multi-sequence prefill pass (module doc). Errors before any launch on
    /// the MTP block, an empty pass, a sequence with no rows, or rows past the highway.
    pub(in crate::glm5next_layer) fn forward_prefill_spans(
        &self,
        hidden: DevicePtr,
        spans: &mut [PrefillSpan<'_>],
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let Some(mhc) = self.mhc.as_ref() else {
            bail!(
                "GLM layer {}: the multi-sequence prefill needs the hyper-connection; this is \
                 the MTP block",
                self.layer_idx
            );
        };
        if spans.is_empty() || spans.iter().any(|s| s.rows == 0) {
            bail!(
                "GLM layer {}: a multi-sequence prefill needs at least one row per sequence, \
                 got {:?}",
                self.layer_idx,
                spans.iter().map(|s| s.rows).collect::<Vec<_>>()
            );
        }
        let rows: Vec<usize> = spans.iter().map(|s| s.rows).collect();
        let total: usize = rows.iter().sum();
        let cap = ctx.buffers.max_batch_tokens();
        if ctx.hc_row_offset + total > cap {
            bail!(
                "GLM layer {}: prefill rows {}..{} pass the {cap}-slot mHC highway",
                self.layer_idx,
                ctx.hc_row_offset,
                ctx.hc_row_offset + total
            );
        }
        let row_bytes = self.hidden * 2;
        let share = |r: usize| self.mlp_share_class(r);
        // 2026-10-11: A sequence with an in-pass capture is cut at its points, and a KDA layer
        // copies its state after the piece that ends at each (`steps/prefill_capture.rs`).
        let points: Vec<Vec<(usize, CapturePoint)>> = spans
            .iter()
            .map(|s| capture_points(s.rows, s.capture))
            .collect();
        let ordinals: Vec<Option<usize>> = spans
            .iter()
            .map(|s| kda_ordinal(&self.mixer, s.capture))
            .collect();
        let cuts: Vec<Vec<usize>> = points
            .iter()
            .map(|p| p.iter().map(|&(at, _)| at).collect())
            .collect();
        for (chunk, group) in prefill_span_groups(&rows, &cuts, prefill_rows().min(cap), share)
            .into_iter()
            .enumerate()
        {
            let base = group[0].row;
            let m: usize = group.iter().map(|p| p.rows).sum();
            let slot_base = ctx.hc_row_offset + base;
            let x = hidden.offset(base * row_bytes);
            // 2026-10-09: The MLP keeps a launch per piece where its arithmetic depends on the
            // row count (`mlp_forward`).
            let pieces: Vec<usize> = group.iter().map(|p| p.rows).collect();
            // 2026-10-13: The layer dump; the DSA selection holds the last piece's rows only.
            let call = dump::Call {
                path: dump::Path::SpanPrefill,
                chunk,
                last_pos: None,
            };
            let tap = dump::begin(ctx, stream, self.dump_layer(mhc, false), call, None, || {
                group
                    .iter()
                    .map(|p| dump::Window {
                        row0: p.row - base,
                        rows: p.rows,
                        seq: p.seq,
                        first_pos: spans[p.seq].seq_len_start + p.t0,
                        last_pos: Some(spans[p.seq].seq_len_start + spans[p.seq].rows - 1),
                    })
                    .collect()
            })?;
            let spans = &mut *spans;
            let kv = &mut *kv_cache;
            self.forward_rows_with(
                mhc,
                x,
                m,
                slot_base,
                ctx,
                stream,
                &pieces,
                tap.as_ref(),
                |normed| {
                    for p in &group {
                        let span = &mut spans[p.seq];
                        self.prefill_piece_mixer(
                            normed.offset((p.row - base) * row_bytes),
                            p.rows,
                            &mut *span.state,
                            kv,
                            span.seq_len_start + p.t0,
                            span.block_table,
                            ctx,
                            stream,
                        )?;
                        let end = p.t0 + p.rows;
                        if let (Some(&(_, which)), Some(idx), Some(c)) = (
                            points[p.seq].iter().find(|&&(at, _)| at == end),
                            ordinals[p.seq],
                            span.capture,
                        ) {
                            self.capture_kda_state(
                                &mut *span.state,
                                c,
                                which,
                                idx,
                                ctx.gpu,
                                stream,
                            )?;
                        }
                    }
                    Ok(normed)
                },
            )?;
        }
        Ok(())
    }

    /// 2026-10-09: Whether pieces whose rows add up to `rows` may share this layer's MLP
    /// launches with a row's bits unchanged: always for a dense site (it runs per piece); for a
    /// routed site only on a path whose rows do not depend on each other
    /// (`examples/glm5next_prefill_spans_mlp_gate.rs`).
    ///
    /// 2026-10-14: As a class, so a group never mixes two shareable paths: the routed site's
    /// `shared_route` (the grouped W4A16 GEMM, or the grouped W4A4 GEMM the declared serve
    /// runs above 16 rows).
    fn mlp_share_class(&self, rows: usize) -> Option<MlpShare> {
        match &self.mlp {
            Glm5NextMlpSite::Dense(_) => Some(MlpShare::Dense),
            Glm5NextMlpSite::Moe(w) => crate::glm5next_mlp::forward::shared_route(
                &self.mlp_kernels,
                &self.mlp_cfg,
                w,
                &self.mlp_ws,
                rows,
            )
            .map(MlpShare::Routed),
        }
    }

    /// 2026-10-09: One piece's mixer, with the output left over its `k` input rows at `x`:
    /// the call `forward_k` makes for a prefill sub-chunk of these rows. KDA leaves its output
    /// in `ws.final_out`, which the next piece reuses, so it is copied back to `x`; DSA writes
    /// over `x` itself.
    #[allow(clippy::too_many_arguments)]
    fn prefill_piece_mixer(
        &self,
        x: DevicePtr,
        k: usize,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        match &self.mixer {
            Glm5NextMixer::Kda { layer, ws, .. } => {
                let st = self.kda_state(state)?;
                let kda = KdaSeqState {
                    conv: st.conv_state,
                    recurrent: st.h_state,
                };
                if k > 1 && kda_chunk_prefill() {
                    layer.prefill(ctx.gpu, x, k, &kda, ws, stream)?;
                } else {
                    layer.decode_k(ctx.gpu, x, k, &kda, ws, &[], stream)?;
                }
                ctx.gpu
                    .copy_d2d_async(ws.final_out, x, k * self.hidden * 2, stream)
            }
            Glm5NextMixer::Dsa(layer) => layer.decode_k(
                x,
                k,
                state,
                kv_cache,
                seq_len,
                block_table,
                ctx,
                stream,
                // 2026-10-09: A prefill piece, as `Glm5NextLayer::prefill` passes it.
                true,
            ),
        }
    }
}

/// 2026-10-09: Every rank runs the same row groups (they follow from the span rows alone) and
/// issues one all-reduce per mixer and MLP site per group, so the pass runs on a multi-rank
/// serve from the spans rank 0 announces.
impl metrale_model_layers::layer::LayerSplitPrefill for Glm5NextLayer {
    fn prefill_spans(
        &self,
        hidden: DevicePtr,
        spans: &mut [PrefillSpan<'_>],
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.forward_prefill_spans(hidden, spans, kv_cache, ctx, stream)
    }
}

#[cfg(test)]
#[path = "prefill_spans_tests.rs"]
mod tests;
