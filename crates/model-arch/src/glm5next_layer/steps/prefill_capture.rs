// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: A GLM layer's prefill under an in-pass SSM capture (`ForwardContext::
//! midchunk_capture`, the prefix cache's tail split): the pass's rows run as consecutive
//! segments split at the capture points, each segment in its own sub-chunks from its first
//! row on, and after a segment that ends at a capture point a KDA layer copies its recurrent
//! and conv state into the plan's slot.
//!
//! Why the bits are those of separate passes. Without the capture the model runs the prompt
//! tail as passes of its own (`prefill_chunk_dispatch`'s two-pass split), and every GLM launch
//! is per sub-chunk of a pass: projections, MoE, mHC, all-reduces, the KDA walk and the DSA
//! rows. Splitting the rows of one pass at the same points gives every layer the same sub-chunks
//! with the same inputs (the KV, indexer and KDA state the earlier rows of the layer left), so
//! every row and every captured state carries the separate passes' bytes, and a later request
//! that restores a captured state and replays the rows after it gets the same bits too. The
//! layer therefore answers `splits_pass_at_capture`, and the model may capture in-pass whatever
//! the row-count dependence of the GLM kernels and on a multi-rank serve.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Without a capture the rows run exactly as before: one segment.
//! - The rows equal the separate passes' bytes only for passes ending and starting at the
//!   SAME points: the model therefore plans one point, the tail split point, for such layers
//!   (`prepare_cut_capture`), the split `--no-ssm-tail-midchunk` runs as two passes.
//! - 2026-10-14: A separate pass of ONE row after position 0 runs each layer's decode path, so
//!   a one-row segment carries its bytes only under `METRALE_ONE_ROW_TAIL_DECODE=1`, which runs
//!   it through `forward_one` as `decode` does; by default it is a one-row `forward_k` prefill
//!   sub-chunk, as before.
//! - 2026-10-14: Under `--prefill-tail-mode merge` (and more than one row per sub-chunk) the rows
//!   run in sub-chunks from row 0 instead (`glm5next_tail_merge::merge_plan`), a capture point cutting only the KDA
//!   conv and recurrence of the sub-chunk holding it (`glm5next_kda/cut.rs`). The captured states
//!   are the split's bits only while each sub-chunk and the split's segment share a GEMM row
//!   class (W4A16 256 rows, FP8 1024); the rows after the last point are not the separate
//!   pass's bits (disclosure PREFILL-TAILMERGE-1).

use super::*;
use crate::glm5next_kda::cut::KdaCut;
use metrale_model_layers::layer::{MidchunkCapture, split_segment_takes_decode_path};

/// 2026-10-11: Which slot a capture point fills.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CapturePoint {
    /// 2026-10-11: `cap_local_early` (`h_dsts_early`, `conv_dsts_early`).
    Early,
    /// 2026-10-11: `cap_local` (`h_dsts`, `conv_dsts`).
    Main,
}

/// 2026-10-11: The `(start, end, capture after end)` segments of a pass of `n` rows split at the
/// capture points strictly inside it (`early < main` when both exist), in row order; one
/// `(0, n, None)` segment without a capture. Pure.
pub(crate) fn capture_segments(
    n: usize,
    early: Option<usize>,
    main: Option<usize>,
) -> Vec<(usize, usize, Option<CapturePoint>)> {
    let mut points: Vec<(usize, CapturePoint)> = [
        early.map(|p| (p, CapturePoint::Early)),
        main.map(|p| (p, CapturePoint::Main)),
    ]
    .into_iter()
    .flatten()
    .filter(|&(p, _)| p > 0 && p < n)
    .collect();
    points.sort_by_key(|&(p, _)| p);
    points.dedup_by_key(|&mut (p, _)| p);
    let mut out = Vec::with_capacity(points.len() + 1);
    let mut start = 0;
    for (p, which) in points {
        out.push((start, p, Some(which)));
        start = p;
    }
    out.push((start, n, None));
    out
}

/// 2026-10-11: The capture points strictly inside a pass of `n` rows under `cap`, in row order,
/// each with its slot: the ends of `capture_segments` that capture. None without a capture.
pub(crate) fn capture_points(
    n: usize,
    cap: Option<&MidchunkCapture<'_>>,
) -> Vec<(usize, CapturePoint)> {
    let Some(c) = cap else {
        return Vec::new();
    };
    capture_segments(n, c.cap_local_early, Some(c.cap_local))
        .into_iter()
        .filter_map(|(_, end, after)| after.map(|w| (end, w)))
        .collect()
}

/// 2026-10-11: The SSM ordinal a layer takes from `cap`'s counter for its pass: one per KDA
/// layer, in model order; none for a DSA layer or without a capture.
pub(crate) fn kda_ordinal(
    mixer: &Glm5NextMixer,
    cap: Option<&MidchunkCapture<'_>>,
) -> Option<usize> {
    match (cap, mixer) {
        (Some(c), Glm5NextMixer::Kda { .. }) => Some(
            c.ssm_layer_counter
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        ),
        _ => None,
    }
}

impl Glm5NextLayer {
    /// 2026-10-11: The rows of a text layer's prefill pass, `rows` per sub-chunk (`forward_k`,
    /// or the per-token walk at one row), split at the capture points of
    /// `ctx.midchunk_capture` (`capture_segments`). A KDA layer takes its ordinal from the
    /// capture's counter, as every SSM layer of the pass does, and copies its state after each
    /// segment that ends at a capture point into that point's destination for its ordinal;
    /// an empty destination list (a rank that reserved no slot) copies nothing but splits the
    /// same, so every rank runs the same launches.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::glm5next_layer) fn prefill_segments(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        rows: usize,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len_start: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let slot0 = ctx.hc_row_offset;
        let cap = ctx.midchunk_capture.as_ref();
        let ordinal = kda_ordinal(&self.mixer, cap);
        // 2026-10-14: `--prefill-tail-mode merge`: sub-chunks from row 0 with KDA cuts at the
        // capture points (`glm5next_tail_merge::merge_plan`), no capture after a segment; else the split segments.
        let merge = rows > 1
            && cap.is_some()
            && metrale_gpu_runtime::prefill_tail_mode::prefill_tail_mode()
                == metrale_gpu_runtime::prefill_tail_mode::PrefillTailMode::Merge;
        let segments: Vec<(usize, usize, Option<CapturePoint>, Vec<KdaCut>)> = if merge {
            let mut v = Vec::new();
            for (s, e, cuts) in crate::glm5next_tail_merge::merge_plan(
                num_tokens,
                rows,
                &capture_points(num_tokens, cap),
            ) {
                let cuts = cuts
                    .into_iter()
                    .map(|(row, which)| {
                        let dst = match (ordinal, cap) {
                            (Some(idx), Some(c)) => self.capture_dsts(c, which, idx)?,
                            _ => None,
                        };
                        Ok(KdaCut { row, dst })
                    })
                    .collect::<Result<Vec<KdaCut>>>()?;
                v.push((s, e, None, cuts));
            }
            v
        } else {
            capture_segments(
                num_tokens,
                cap.and_then(|c| c.cap_local_early),
                cap.map(|c| c.cap_local),
            )
            .into_iter()
            .map(|(s, e, after)| (s, e, after, Vec::new()))
            .collect()
        };
        // 2026-10-13: The layer dump's sub-chunk index and the pass's last position.
        let mut chunk = 0usize;
        let last_pos = (num_tokens > 0).then(|| seq_len_start + num_tokens - 1);
        let split = !merge && segments.len() > 1;
        // 2026-10-14: `METRALE_GLM_PREFILL_PIECE_ROWS=P`: a wider sub-chunk runs as `P`-row
        // pieces (`steps/forward_pieces.rs`); not beside the SP pipeline, which plans its own
        // sub-chunks, and neither under the merged tail (one `forward_k` per merged sub-chunk).
        let piece = crate::glm5next_prefill_piece::piece_rows()?;
        let pipeline = self.levers.mhc.sp_pipeline;
        if piece.is_some() && pipeline {
            bail!(
                "GLM layer {}: {} and METRALE_GLM_SP_PIPELINE=1 cannot both be set",
                self.layer_idx,
                crate::glm5next_prefill_piece::PIECE_ROWS_ENV
            );
        }
        if merge && (piece.is_some() || pipeline) {
            bail!(
                "GLM layer {}: --prefill-tail-mode merge (the default) runs neither {} nor \
                 METRALE_GLM_SP_PIPELINE=1; pass --prefill-tail-mode split beside them",
                self.layer_idx,
                crate::glm5next_prefill_piece::PIECE_ROWS_ENV
            );
        }
        for (start, end, after, cuts) in segments {
            // 2026-10-14: Under `METRALE_ONE_ROW_TAIL_DECODE=1` a one-row segment after position
            // 0 runs `decode`'s walk (`forward_one`, its dump call), the path a warm one-row
            // replay of the row takes, on the row's own highway slot (`layer/one_row_path.rs`).
            // The per-token walk (`rows == 1`) already runs `forward_one`.
            let tail_decode = rows > 1
                && split_segment_takes_decode_path(
                    end - start,
                    seq_len_start + start,
                    split,
                    ctx.levers.one_row_tail_decode,
                    ctx.levers.one_row_prefill,
                );
            let mut t = start;
            // 2026-10-14: `METRALE_GLM_SP_PIPELINE=1`: the segment's leading run of SP-mHC
            // sub-chunks pipelined through the layer (`steps/forward_sp_pipe.rs`); the rows after
            // it run below as before.
            if rows > 1 && !tail_decode && pipeline {
                let ks = crate::glm5next_sp_pipeline::sub_chunks(end - start, rows);
                let takes = ks
                    .iter()
                    .map(|&k| self.sp_takes(ctx, k, true))
                    .collect::<Result<Vec<bool>>>()?;
                let p = crate::glm5next_sp_pipeline::pipelined(&takes);
                if p > 0 {
                    self.forward_sp_pipeline(
                        hidden.offset(t * self.hidden * 2),
                        &ks[..p],
                        state,
                        kv_cache,
                        seq_len_start + t,
                        block_table,
                        ctx,
                        stream,
                        slot0 + t,
                    )?;
                    t += ks[..p].iter().sum::<usize>();
                    chunk += p;
                }
            }
            while t < end {
                let k = rows.min(end - t);
                if let Some(p) = piece.filter(|&p| rows > 1 && !tail_decode && k > p) {
                    self.forward_k_pieces(
                        hidden.offset(t * self.hidden * 2),
                        k,
                        p,
                        state,
                        kv_cache,
                        seq_len_start + t,
                        block_table,
                        ctx,
                        stream,
                        slot0 + t,
                        &mut chunk,
                        last_pos,
                    )?;
                    t += k;
                    continue;
                }
                let call = if tail_decode {
                    dump::Call::step(dump::Path::Decode)
                } else {
                    dump::Call {
                        path: dump::Path::Prefill,
                        chunk,
                        last_pos,
                    }
                };
                chunk += 1;
                if !cuts.is_empty() && (t, k) != (start, end - start) {
                    bail!(
                        "GLM layer {}: a merged sub-chunk [{start}, {end}) must run as one call",
                        self.layer_idx
                    );
                }
                if rows > 1 && !tail_decode {
                    self.forward_k(
                        hidden.offset(t * self.hidden * 2),
                        k,
                        state,
                        kv_cache,
                        seq_len_start + t,
                        block_table,
                        ctx,
                        stream,
                        false,
                        slot0 + t,
                        // 2026-09-25: `is_prefill` is true only for this caller.
                        // This IS the prefill sub-chunk caller.
                        true,
                        call,
                        &cuts,
                    )?;
                    t += k;
                } else {
                    let off = t * self.hidden * 2;
                    self.forward_one(
                        hidden.offset(off),
                        residual.offset(off),
                        slot0 + t,
                        state,
                        kv_cache,
                        seq_len_start + t,
                        block_table,
                        disk_block_ids,
                        disk_last_offloaded_per_layer,
                        ctx,
                        stream,
                        call,
                    )?;
                    t += 1;
                }
            }
            if let (Some(which), Some(idx), Some(c)) = (after, ordinal, cap) {
                self.capture_kda_state(state, c, which, idx, ctx.gpu, stream)?;
            }
        }
        Ok(())
    }

    /// 2026-10-11: Copy this KDA layer's recurrent and conv state into capture point `which`'s
    /// destinations for SSM ordinal `idx`; nothing when that list has no entry `idx`. Errors
    /// when the plan's per-layer sizes are not this layer's state sizes.
    pub(in crate::glm5next_layer) fn capture_kda_state(
        &self,
        state: &mut dyn LayerState,
        cap: &MidchunkCapture<'_>,
        which: CapturePoint,
        idx: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let Some((h_dst, conv_dst)) = self.capture_dsts(cap, which, idx)? else {
            return Ok(());
        };
        let st = self.kda_state(state)?;
        gpu.copy_d2d_async(st.h_state, h_dst, cap.h_bytes, stream)?;
        gpu.copy_d2d_async(st.conv_state, conv_dst, cap.conv_bytes, stream)
    }

    /// 2026-10-14: Capture point `which`'s `(h_dst, conv_dst)` for SSM ordinal `idx` on this KDA
    /// layer; `None` for a DSA layer or when that list has no entry `idx`. Errors when the plan's
    /// per-layer sizes are not this layer's state sizes.
    fn capture_dsts(
        &self,
        cap: &MidchunkCapture<'_>,
        which: CapturePoint,
        idx: usize,
    ) -> Result<Option<(DevicePtr, DevicePtr)>> {
        let Glm5NextMixer::Kda { cfg, .. } = &self.mixer else {
            return Ok(None);
        };
        let (h_dsts, conv_dsts) = match which {
            CapturePoint::Early => (cap.h_dsts_early, cap.conv_dsts_early),
            CapturePoint::Main => (cap.h_dsts, cap.conv_dsts),
        };
        let (Some(&h_dst), Some(&conv_dst)) = (h_dsts.get(idx), conv_dsts.get(idx)) else {
            return Ok(None);
        };
        let (h_bytes, conv_bytes) = (cfg.recurrent_state_elems() * 4, cfg.conv_state_elems() * 4);
        if (cap.h_bytes, cap.conv_bytes) != (h_bytes, conv_bytes) {
            bail!(
                "GLM layer {}: the capture plan holds {} + {} bytes per SSM layer, the KDA \
                 state is {h_bytes} + {conv_bytes}",
                self.layer_idx,
                cap.h_bytes,
                cap.conv_bytes
            );
        }
        Ok(Some((h_dst, conv_dst)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-10-11: No capture, or capture points outside the pass: one segment. Points inside:
    /// consecutive segments ending at each point in row order, each tagged with its slot.
    #[test]
    fn segments_split_at_the_points_inside_the_pass() {
        use CapturePoint::{Early, Main};
        assert_eq!(capture_segments(40, None, None), vec![(0, 40, None)]);
        assert_eq!(capture_segments(40, None, Some(0)), vec![(0, 40, None)]);
        assert_eq!(capture_segments(40, None, Some(40)), vec![(0, 40, None)]);
        assert_eq!(
            capture_segments(40, None, Some(32)),
            vec![(0, 32, Some(Main)), (32, 40, None)]
        );
        assert_eq!(
            capture_segments(4000, Some(3968), Some(3984)),
            vec![
                (0, 3968, Some(Early)),
                (3968, 3984, Some(Main)),
                (3984, 4000, None)
            ]
        );
        assert_eq!(
            capture_segments(40, Some(50), Some(32)),
            vec![(0, 32, Some(Main)), (32, 40, None)]
        );
    }
}
