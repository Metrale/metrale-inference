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
//! - Row `r` of the pass uses highway slot `ctx.hc_row_offset + r`.
//!
//! # Which launches a row shares with other sequences
//!
//! The group-wide ones: `hc_pre`/`hc_post`/`hc_head_mean` and the norms (one block per token,
//! nothing shared across tokens), the mixer and MLP all-reduces, and in a routed MLP the
//! top-k, the sort, the expert GEMMs, the shared expert and the combine, each of which computes
//! a row from its own inputs. The launches whose arithmetic depends on the row count stay per
//! piece: the router GEMM (`forward_moe_pieces`) and a dense MLP site. So on one GPU a row's
//! bits equal the single-sequence prefill's, measured byte for byte on the routed site by
//! `examples/glm5next_prefill_spans_mlp_gate.rs`, with one exception: the routed experts pick
//! their kernel by the group's row count (W4A4 up to 16 rows, the row-union GEMV below
//! `METRALE_GLM_MOE_PREFILL_GEMM_MIN_ROWS`, the grouped GEMM from there), so a prompt whose own
//! row count picks another kernel than its group's (16 tokens or fewer, or below that floor) is
//! not byte-identical. Across three ranks the all-reduce is a sum of `rows * hidden` elements
//! whose per-element order NCCL may pick by message size.

use super::forward::kda_chunk_prefill;
use super::*;
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
/// most `cap` rows. A `cap` of 0 is treated as 1; a sequence with no rows has no piece.
pub fn prefill_span_groups(rows: &[usize], cap: usize) -> Vec<Vec<SpanPiece>> {
    let cap = cap.max(1);
    let mut groups = Vec::new();
    let mut cur: Vec<SpanPiece> = Vec::new();
    let mut cur_rows = 0usize;
    let mut row = 0usize;
    for (seq, &n) in rows.iter().enumerate() {
        let mut t0 = 0usize;
        while t0 < n {
            let len = cap.min(n - t0);
            if cur_rows + len > cap {
                groups.push(std::mem::take(&mut cur));
                cur_rows = 0;
            }
            cur.push(SpanPiece {
                seq,
                t0,
                row,
                rows: len,
            });
            cur_rows += len;
            row += len;
            t0 += len;
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
        for group in prefill_span_groups(&rows, prefill_rows().min(cap)) {
            let base = group[0].row;
            let m: usize = group.iter().map(|p| p.rows).sum();
            let slot_base = ctx.hc_row_offset + base;
            let x = hidden.offset(base * row_bytes);
            // 2026-10-09: The MLP keeps a launch per piece where its arithmetic depends on the
            // row count (`mlp_forward`).
            let pieces: Vec<usize> = group.iter().map(|p| p.rows).collect();
            let spans = &mut *spans;
            let kv = &mut *kv_cache;
            self.forward_rows_with(mhc, x, m, slot_base, ctx, stream, &pieces, |normed| {
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
                }
                Ok(normed)
            })?;
        }
        Ok(())
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
            prefill_span_groups(&[198, 198, 197], 512),
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
            prefill_span_groups(&[10, 3, 1], 4),
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
            let groups = prefill_span_groups(&rows, cap);
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
}
