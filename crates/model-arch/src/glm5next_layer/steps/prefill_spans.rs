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
//!   batch-mates (`mlp_shares_rows`): a routed site only on the grouped GEMM, both for the
//!   piece's own row count and for the group's. Any other piece runs as a group of its own.
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
/// most `cap` rows. A piece shares a group only when `share` holds for its own row count and
/// for the group's total with it; any other piece gets a group of its own, so its launches are
/// those of its own prefill. A `cap` of 0 is treated as 1; a sequence with no rows has no piece.
///
/// 2026-10-11: `cuts[s]` (ascending, inside the sequence; a missing entry is none) are the
/// sequence's in-pass capture points: the sequence is cut there too and its sub-chunk grid
/// restarts at each, as its own prefill splits the pass there (`steps/prefill_capture.rs`).
pub fn prefill_span_groups(
    rows: &[usize],
    cuts: &[Vec<usize>],
    cap: usize,
    share: impl Fn(usize) -> bool,
) -> Vec<Vec<SpanPiece>> {
    let cap = cap.max(1);
    let mut groups = Vec::new();
    let mut cur: Vec<SpanPiece> = Vec::new();
    let mut cur_rows = 0usize;
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
            if !share(len) {
                if !cur.is_empty() {
                    groups.push(std::mem::take(&mut cur));
                    cur_rows = 0;
                }
                groups.push(vec![piece]);
                continue;
            }
            if !cur.is_empty() && (cur_rows + len > cap || !share(cur_rows + len)) {
                groups.push(std::mem::take(&mut cur));
                cur_rows = 0;
            }
            cur.push(piece);
            cur_rows += len;
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
        let share = |r: usize| self.mlp_shares_rows(r);
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
    /// routed site only on the grouped GEMM, whose rows do not depend on each other
    /// (`examples/glm5next_prefill_spans_mlp_gate.rs`). The W4A4, row-union and per-row paths
    /// are left to one piece per group.
    fn mlp_shares_rows(&self, rows: usize) -> bool {
        match &self.mlp {
            Glm5NextMlpSite::Dense(_) => true,
            Glm5NextMlpSite::Moe(w) => {
                crate::glm5next_mlp::forward::expert_route(
                    &self.mlp_kernels,
                    &self.mlp_cfg,
                    w,
                    &self.mlp_ws,
                    rows,
                ) == crate::glm5next_mlp::forward::ExpertRoute::Grouped
            }
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
        // 2026-10-14: The prompt pass's dense tier (`glm5next_phase`).
        let _phase = crate::glm5next_phase::enter_prefill();
        self.forward_prefill_spans(hidden, spans, kv_cache, ctx, stream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn piece(seq: usize, t0: usize, row: usize, rows: usize) -> SpanPiece {
        SpanPiece { seq, t0, row, rows }
    }

    /// 2026-10-09: Two ladder prompts share a 512-row group; the third starts the next one,
    /// since the three do not fit together.
    #[test]
    fn short_prompts_pack_whole_into_groups() {
        assert_eq!(
            prefill_span_groups(&[198, 198, 197], &[], 512, |_| true),
            vec![
                vec![piece(0, 0, 0, 198), piece(1, 0, 198, 198)],
                vec![piece(2, 0, 396, 197)],
            ]
        );
    }

    /// 2026-10-09: A sequence longer than the cap is cut where its own prefill cuts it (every
    /// `cap` rows from its first row); a piece that does not fit the open group starts the next
    /// one instead of being split.
    #[test]
    fn a_long_sequence_keeps_its_own_sub_chunks() {
        assert_eq!(
            prefill_span_groups(&[10, 3, 1], &[], 4, |_| true),
            vec![
                vec![piece(0, 0, 0, 4)],
                vec![piece(0, 4, 4, 4)],
                vec![piece(0, 8, 8, 2)],
                vec![piece(1, 0, 10, 3), piece(2, 0, 13, 1)],
            ]
        );
    }

    /// 2026-10-09: For assorted shapes: every row lands in exactly one piece, in pass order;
    /// groups are contiguous and at most `cap` rows; each sequence's pieces are exactly the
    /// sub-chunks its single-sequence prefill runs (`t0` a multiple of `cap`, `cap` rows but
    /// the last).
    #[test]
    fn pieces_cover_every_row_once_as_the_single_sequence_sub_chunks() {
        for (rows, cap) in [
            (vec![198usize, 198, 198, 198, 197], 512usize),
            (vec![1, 1, 1], 2),
            (vec![513, 7, 1024, 1], 512),
            (vec![5], 1),
            (vec![3, 0, 4], 0),
        ] {
            let groups = prefill_span_groups(&rows, &[], cap, |_| true);
            let cap = cap.max(1);
            let mut next_row = 0usize;
            let mut next_t = vec![0usize; rows.len()];
            for g in &groups {
                assert!(!g.is_empty());
                assert!(
                    g.iter().map(|p| p.rows).sum::<usize>() <= cap,
                    "{rows:?} cap {cap}"
                );
                for p in g {
                    assert_eq!(p.row, next_row, "{rows:?} cap {cap}: rows out of order");
                    assert_eq!(
                        p.t0, next_t[p.seq],
                        "{rows:?} cap {cap}: sequence rows skipped"
                    );
                    assert_eq!(
                        p.t0 % cap,
                        0,
                        "{rows:?} cap {cap}: piece off the sub-chunk grid"
                    );
                    assert_eq!(p.rows, cap.min(rows[p.seq] - p.t0), "{rows:?} cap {cap}");
                    next_row += p.rows;
                    next_t[p.seq] += p.rows;
                }
            }
            assert_eq!(next_row, rows.iter().sum::<usize>());
            assert_eq!(next_t, rows);
        }
    }

    /// 2026-10-11: A sequence with capture points is cut there as well, and its sub-chunk grid
    /// restarts at each point, as its own prefill splits the pass there; for assorted shapes
    /// every sequence's pieces are exactly the sub-chunks of separate passes ending and starting
    /// at its points. Points outside the sequence cut nothing.
    #[test]
    fn capture_points_cut_a_sequence_into_separate_passes_sub_chunks() {
        assert_eq!(
            prefill_span_groups(&[10, 6], &[vec![6], vec![]], 4, |_| true),
            vec![
                vec![piece(0, 0, 0, 4)],
                vec![piece(0, 4, 4, 2)],
                vec![piece(0, 6, 6, 4)],
                vec![piece(1, 0, 10, 4)],
                vec![piece(1, 4, 14, 2)],
            ]
        );
        for (rows, cuts, cap) in [
            (
                vec![4000usize, 300],
                vec![vec![3968usize, 3984], vec![]],
                2048usize,
            ),
            (vec![50, 50], vec![vec![16], vec![32, 48]], 16),
            (vec![7, 9], vec![vec![0, 7, 9], vec![1]], 4),
            (vec![2048, 17], vec![vec![2048], vec![16]], 2048),
        ] {
            let groups = prefill_span_groups(&rows, &cuts, cap, |_| true);
            for (s, &n) in rows.iter().enumerate() {
                let mut want = Vec::new();
                let mut starts: Vec<usize> = cuts[s]
                    .iter()
                    .copied()
                    .filter(|&c| c > 0 && c < n)
                    .collect();
                starts.insert(0, 0);
                starts.push(n);
                for w in starts.windows(2) {
                    let mut t = w[0];
                    while t < w[1] {
                        let k = cap.min(w[1] - t);
                        want.push((t, k));
                        t += k;
                    }
                }
                let got: Vec<(usize, usize)> = groups
                    .iter()
                    .flatten()
                    .filter(|p| p.seq == s)
                    .map(|p| (p.t0, p.rows))
                    .collect();
                assert_eq!(got, want, "{rows:?} {cuts:?} cap {cap}: sequence {s}");
            }
        }
    }

    /// 2026-10-09: The serve's routed site shares only from 17 rows (the grouped GEMM): a
    /// shorter piece runs alone, and a piece may not join a group whose total the predicate
    /// refuses.
    #[test]
    fn a_piece_the_predicate_refuses_runs_alone() {
        let grouped = |r: usize| r >= 17;
        assert_eq!(
            prefill_span_groups(&[77, 9, 77, 77, 16, 20], &[], 512, grouped),
            vec![
                vec![piece(0, 0, 0, 77)],
                vec![piece(1, 0, 77, 9)],
                vec![piece(2, 0, 86, 77), piece(3, 0, 163, 77)],
                vec![piece(4, 0, 240, 16)],
                vec![piece(5, 0, 256, 20)],
            ]
        );
        let below_300 = |r: usize| r < 300;
        assert_eq!(
            prefill_span_groups(&[200, 150, 50], &[], 512, below_300),
            vec![
                vec![piece(0, 0, 0, 200)],
                vec![piece(1, 0, 200, 150), piece(2, 0, 350, 50)],
            ]
        );
    }
}
