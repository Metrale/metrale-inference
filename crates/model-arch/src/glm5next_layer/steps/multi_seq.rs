// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: `forward_multi`, the batched multi-sequence decode body: one token for each of
//! `n` sequences through the layer, the rows in groups of at most `multi_seq_chunk_rows()`.
//! 2026-10-09: Both it and the batched speculative verify (`verify_multi.rs`, `ks[i]` rows for
//! sequence `i`) run through `forward_spans`; the decode is the case of one row per sequence.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Row `r` uses highway slot `ctx.hc_row_offset + r`, the state of the sequence whose rows
//!   hold it, that sequence's position for it and metadata row `r`; no row reads another
//!   sequence's state, and a sequence's rows step its state in row order.
//! - A group is at most `DENSE_GEMV_BATCHM_MAX_M` rows (`multi_seq_chunk_rows`), the width up
//!   to which every projection gives each row the M = 1 GEMV's bits. A sequence's rows may
//!   straddle two groups; the second continues from the state the first left.
//!
//! # Why a row's output equals its sequence decoded alone (`forward_one`)
//!
//! `forward_one` and this path launch the same kernels per row; the only differences are the
//! row counts of the launches that span rows:
//! - mHC (`hc_expand`, `hc_mix`, `hc_finish`, `hc_post`, `hc_head`) and `rms_norm_vanilla` run
//!   one block (or one block row) per token and share nothing across tokens.
//! - The KDA and DSA projections, the shared expert and the dense MLP run the batched GEMV,
//!   whose rows are bit-identical to the M = 1 GEMV up to `DENSE_GEMV_BATCHM_MAX_M`.
//! - The router logits stay one M = 1 GEMV per row; the top-k and the combine run one block
//!   per row.
//! - The routed experts take the row union (`w4a16_gemv_sw_moe_batchm_m<R>`), whose per
//!   (row, slot) arithmetic is `w4a16_gemv_sw_moe`'s, as
//!   `examples/glm5next_moe_row_batch_microtest.rs` checks byte for byte.
//! - The KDA conv and recurrence and the DSA latent write, indexer and selection are the
//!   single-row launches, on the row's own state.
//! - The all-reduce sums `rows * hidden` elements across ranks. Single-GPU there is none; over
//!   two ranks a sum of two is order-free; over three, NCCL's reduction order for an element
//!   may depend on the message size and the element's offset, so identity across batch
//!   compositions is not promised there (the per-sequence path has the same exposure).
//!
//! One lever breaks the identity: `METRALE_GLM_MOE_PREFILL_GEMM_MIN_ROWS` set at or below the
//! group width sends a group's routed experts to the grouped tensor-core GEMM (the default,
//! 128, is above it). It acts on `forward_k` the same way.

use super::*;
use crate::glm5next_dsa::layer::DsaRowSpan;
use crate::glm5next_kda::KdaVerifyRecord;

/// 2026-10-09: One sequence's rows inside one row group of `forward_spans`: group rows
/// `row0..row0 + rows` are the sequence's rows `t0..t0 + rows`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::glm5next_layer) struct GroupSpan {
    pub seq: usize,
    pub row0: usize,
    pub t0: usize,
    pub rows: usize,
}

/// 2026-10-09: The spans of the group of `m` rows starting at row `base` of a pass whose rows
/// are sequence-major with `ks[s]` rows for sequence `s`, in row order. Sequences with no row in
/// the group are absent, so the spans' sequences are consecutive.
pub(in crate::glm5next_layer) fn group_spans(
    ks: &[usize],
    base: usize,
    m: usize,
) -> Vec<GroupSpan> {
    let mut spans = Vec::new();
    let mut off = 0usize;
    for (seq, &k) in ks.iter().enumerate() {
        let (lo, hi) = (off.max(base), (off + k).min(base + m));
        if lo < hi {
            spans.push(GroupSpan {
                seq,
                row0: lo - base,
                t0: lo - off,
                rows: hi - lo,
            });
        }
        off += k;
    }
    spans
}

/// 2026-10-09: What a KDA layer keeps of a sequence's verify rows besides the stepped state:
/// nothing (a decode), the state after each row `t < k - 1` (snapshot rollback), or each such
/// row's recurrent inputs (replay rollback).
pub(in crate::glm5next_layer) enum KdaRowKeep {
    Nothing,
    Snapshots(Vec<(DevicePtr, DevicePtr)>),
    Record {
        record: KdaVerifyRecord,
        rows: usize,
    },
}

impl Glm5NextLayer {
    /// 2026-10-08: One decode token for each of `num_seqs` sequences, in groups of
    /// `multi_seq_chunk_rows()` rows, each group through `forward_rows_with` with a per-row
    /// mixer. A DSA layer reads each row's position, KV slot, `seq_len` and block table from
    /// `ctx.attn_metadata`, which must hold at least `num_seqs` rows; a KDA layer reads each
    /// row's pool state. Errors on a layer without a hyper-connection.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::glm5next_layer) fn forward_multi<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        num_seqs: usize,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        kv_cache: &mut PagedKvCache,
        seq_lens: &[usize],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if states.len() < num_seqs || seq_lens.len() < num_seqs {
            bail!(
                "GLM layer {}: a {num_seqs}-row batched decode got {} states and {} seq_lens",
                self.layer_idx,
                states.len(),
                seq_lens.len()
            );
        }
        self.forward_spans(
            hidden,
            &vec![1; num_seqs],
            &seq_lens[..num_seqs],
            &mut states[..num_seqs],
            kv_cache,
            None,
            ctx,
            stream,
        )
    }

    /// 2026-10-09: `ks[s]` rows for each sequence `s`, sequence-major in `hidden`: row `t` of
    /// sequence `s` is its token at position `seq_lens[s] + t`. The rows run in groups of
    /// `multi_seq_chunk_rows()` through `forward_rows_with`. A KDA layer steps each sequence's
    /// state through its rows in order and keeps what `keep[s]` asks (`None`: nothing); a DSA
    /// layer runs `decode_spans` over the group's sequences. Errors on a layer without a
    /// hyper-connection or when the rows pass the highway.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::glm5next_layer) fn forward_spans(
        &self,
        hidden: DevicePtr,
        ks: &[usize],
        seq_lens: &[usize],
        states: &mut [&mut (dyn LayerState + 'static)],
        kv_cache: &mut PagedKvCache,
        keep: Option<&[KdaRowKeep]>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let Some(mhc) = self.mhc.as_ref() else {
            bail!(
                "GLM layer {}: the batched decode needs the hyper-connection; this is the MTP \
                 block",
                self.layer_idx
            );
        };
        let n = ks.len();
        if states.len() != n || seq_lens.len() != n || keep.is_some_and(|k| k.len() != n) {
            bail!(
                "GLM layer {}: {n} sequences with {} states, {} seq_lens and {:?} keep plans",
                self.layer_idx,
                states.len(),
                seq_lens.len(),
                keep.map(<[KdaRowKeep]>::len)
            );
        }
        let total: usize = ks.iter().sum();
        let cap = ctx.buffers.max_batch_tokens();
        if ctx.hc_row_offset + total > cap {
            bail!(
                "GLM layer {}: rows {}..{} pass the {cap}-slot mHC highway",
                self.layer_idx,
                ctx.hc_row_offset,
                ctx.hc_row_offset + total
            );
        }
        let meta = match &self.mixer {
            Glm5NextMixer::Dsa(_) => Some(ctx.attn_metadata.as_ref().ok_or_else(|| {
                anyhow::anyhow!(
                    "GLM layer {}: a batched DSA decode reads each row's position, KV slot and \
                     block table from attn_metadata, and none was uploaded",
                    self.layer_idx
                )
            })?),
            Glm5NextMixer::Kda { .. } => None,
        };
        let kda: Vec<KdaSeqState> = match &self.mixer {
            Glm5NextMixer::Kda { .. } => states
                .iter_mut()
                .map(|s| {
                    let st = self.kda_state(&mut **s)?;
                    Ok(KdaSeqState {
                        conv: st.conv_state,
                        recurrent: st.h_state,
                    })
                })
                .collect::<Result<_>>()?,
            Glm5NextMixer::Dsa(_) => Vec::new(),
        };
        let row_bytes = self.hidden * 2;
        for (base, m) in multi_seq_chunks(total, multi_seq_chunk_rows()) {
            let spans = group_spans(ks, base, m);
            let x = hidden.offset(base * row_bytes);
            let slot_base = ctx.hc_row_offset + base;
            match (&self.mixer, meta) {
                (Glm5NextMixer::Kda { layer, ws, .. }, _) => {
                    let row_states: Vec<KdaSeqState> = spans
                        .iter()
                        .flat_map(|sp| std::iter::repeat_n(kda[sp.seq], sp.rows))
                        .collect();
                    self.forward_rows_with(mhc, x, m, slot_base, ctx, stream, |normed| {
                        layer.decode_rows_then(
                            ctx.gpu,
                            normed,
                            &row_states,
                            ws,
                            stream,
                            |row| {
                                self.kda_after_row(
                                    layer,
                                    &spans,
                                    keep,
                                    &row_states,
                                    row,
                                    ctx,
                                    stream,
                                )
                            },
                        )?;
                        if let Some(keep) = keep {
                            for sp in &spans {
                                if let KdaRowKeep::Record { record, rows } = &keep[sp.seq] {
                                    let end = (sp.t0 + sp.rows).min(*rows);
                                    if end > sp.t0 {
                                        layer.record_verify_rows_at(
                                            ctx.gpu,
                                            ws,
                                            sp.row0,
                                            sp.t0,
                                            end - sp.t0,
                                            record,
                                            stream,
                                        )?;
                                    }
                                }
                            }
                        }
                        Ok(ws.final_out)
                    })?;
                }
                (Glm5NextMixer::Dsa(layer), Some(meta)) => {
                    let (s_lo, s_hi) = match (spans.first(), spans.last()) {
                        (Some(a), Some(b)) => (a.seq, b.seq),
                        _ => bail!("GLM layer {}: an empty row group", self.layer_idx),
                    };
                    let dsa_spans: Vec<DsaRowSpan> = spans
                        .iter()
                        .map(|sp| DsaRowSpan {
                            first_pos: seq_lens[sp.seq] + sp.t0,
                            rows: sp.rows,
                        })
                        .collect();
                    let group = &mut states[s_lo..=s_hi];
                    self.forward_rows_with(mhc, x, m, slot_base, ctx, stream, |normed| {
                        // 2026-10-08: Writes its output projection over its input buffer, as
                        // `decode_k` does.
                        layer.decode_spans(
                            normed, group, &dsa_spans, kv_cache, meta, base, ctx, stream,
                        )?;
                        Ok(normed)
                    })?;
                }
                (Glm5NextMixer::Dsa(_), None) => {
                    bail!("GLM layer {}: DSA mixer without metadata", self.layer_idx)
                }
            }
        }
        Ok(())
    }

    /// 2026-10-09: After group row `row`'s recurrent step: the snapshot its sequence's plan
    /// asks for (the state after its row `t`, for `t` below the snapshot count), else nothing.
    #[allow(clippy::too_many_arguments)]
    fn kda_after_row(
        &self,
        layer: &Glm5NextKdaLayer,
        spans: &[GroupSpan],
        keep: Option<&[KdaRowKeep]>,
        row_states: &[KdaSeqState],
        row: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let Some(keep) = keep else {
            return Ok(());
        };
        let Some(sp) = spans
            .iter()
            .find(|sp| (sp.row0..sp.row0 + sp.rows).contains(&row))
        else {
            bail!(
                "GLM layer {}: group row {row} is in no span",
                self.layer_idx
            );
        };
        if let KdaRowKeep::Snapshots(snaps) = &keep[sp.seq]
            && let Some(dst) = snaps.get(sp.t0 + row - sp.row0)
        {
            layer.snapshot_state(ctx.gpu, &row_states[row], *dst, stream)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span(seq: usize, row0: usize, t0: usize, rows: usize) -> GroupSpan {
        GroupSpan {
            seq,
            row0,
            t0,
            rows,
        }
    }

    /// 2026-10-09: The batched decode (one row per sequence) gives every sequence one span at
    /// its own row.
    #[test]
    fn one_row_per_sequence_is_one_span_per_row() {
        assert_eq!(
            group_spans(&[1, 1, 1], 0, 3),
            vec![span(0, 0, 0, 1), span(1, 1, 0, 1), span(2, 2, 0, 1)]
        );
        assert_eq!(group_spans(&[1; 20], 16, 4)[0], span(16, 0, 0, 1));
    }

    /// 2026-10-09: γ = 4 (k = 5) over 16-row groups: the fourth sequence straddles the
    /// groups, its row 0 in the first and rows 1..5 in the second, so the second group
    /// continues its state and its record at row 1.
    #[test]
    fn a_sequence_straddling_two_groups_continues_at_its_next_row() {
        let ks = [5, 5, 5, 5];
        let groups = multi_seq_chunks(20, 16);
        assert_eq!(groups, vec![(0, 16), (16, 4)]);
        assert_eq!(
            group_spans(&ks, 0, 16),
            vec![
                span(0, 0, 0, 5),
                span(1, 5, 0, 5),
                span(2, 10, 0, 5),
                span(3, 15, 0, 1)
            ]
        );
        assert_eq!(group_spans(&ks, 16, 4), vec![span(3, 0, 1, 4)]);
    }

    /// 2026-10-09: Every row of the pass lands in exactly one span, in row order, for uneven
    /// widths too.
    #[test]
    fn the_spans_cover_every_row_once_in_order() {
        let ks = [8, 3, 8, 1, 8];
        let total: usize = ks.iter().sum();
        let mut seen = Vec::new();
        for (base, m) in multi_seq_chunks(total, 16) {
            let spans = group_spans(&ks, base, m);
            assert_eq!(spans.iter().map(|s| s.rows).sum::<usize>(), m);
            for sp in spans {
                for t in sp.t0..sp.t0 + sp.rows {
                    seen.push((sp.seq, t));
                }
            }
        }
        let expect: Vec<(usize, usize)> = ks
            .iter()
            .enumerate()
            .flat_map(|(s, &k)| (0..k).map(move |t| (s, t)))
            .collect();
        assert_eq!(seen, expect);
    }
}
