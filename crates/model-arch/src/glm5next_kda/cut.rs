// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: `--prefill-tail-mode merge`: a prefill sub-chunk of one sequence whose KDA conv
//! and recurrence are cut at given rows, to take the state there (the prefix cache's in-pass
//! capture), while every other launch of the layer covers all of the sub-chunk's rows.
//!
//! Under `split` the layer runs the rows before and after a capture point as separate
//! `forward_k` calls, so each projection, MoE and all-reduce launch runs twice for a pass that
//! ends a few rows past the point: for a short prompt the second call re-reads every layer's
//! weights for at most one block of rows. Here the projections, gates, o-norm and o-proj run
//! once over the `k` rows (`front_end`, `back_end`), and only the stateful part is cut:
//!
//! - the conv runs once over all `k` rows (`conv_tokens`); each row's conv output is a function
//!   of the rows and the window before the sub-chunk only, so it is the per-segment conv's;
//!   a cut's conv state is the window copied before the conv and advanced past the cut's rows
//!   (`causal_conv1d_window_advance`, which only moves inputs), as `stateful_tokens_snap`
//!   takes a verify's conv intermediates;
//! - the recurrence runs as one part per span between cuts, each part with the launch the
//!   split segment of that many rows takes (`decode_k_modes`): the 64-token scan where its
//!   floor is reached, else the per-sequence token kernel (whose bits are the per-row walk's);
//!   a cut's recurrent state is copied after the part ending there.
//!
//! So with the same q/k/v/gate/beta rows the captured states and every row's core output
//! are the split segments' bits. The rows themselves come from projections over `k` rows instead
//! of the segments' row counts, which may differ in low bits (disclosure PREFILL-TAILMERGE-1).
//!
//! Owner: model-arch (GLM-5.3-Flash KDA).
//! Invariants:
//! - Only the per-sequence token kernels and the 64-token scan: refused under the 16-token
//!   scan (`METRALE_GLM_KDA_CHUNK_SCAN`) or the chunked prefill (`METRALE_GLM_KDA_CHUNK_PREFILL`).
//! - Cuts are strictly increasing rows in `1..=k`.
//! - The gate runs in the front end and the o_norm in the back end ([`GateAt::Kernel`],
//!   [`NormAt::Back`]) and the conv in its own launch, whatever `METRALE_GLM_KDA_GATE_IN_PREPARE`,
//!   `METRALE_GLM_KDA_ONORM_IN_SCAN` and `METRALE_GLM_KDA_CHUNK64_CONV` say: a part on the token
//!   kernels reads the gate rows and its own core rows, and those levers' bytes are the separate
//!   launches' by their own invariants.

use super::decode::RowIo;
use super::*;

/// 2026-10-14: One cut of a [`Glm5NextKdaLayer::prefill_cut`]: the state after the sub-chunk's
/// first `row` rows goes to `dst` = `(h_dst, conv_dst)`; `None` cuts the recurrence the same
/// way and copies nothing (a rank or ordinal that reserved no slot runs the same launches).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KdaCut {
    pub row: usize,
    pub dst: Option<(DevicePtr, DevicePtr)>,
}

/// 2026-10-14: The recurrence parts of `k` rows cut at `rows` (strictly increasing, each in
/// `1..=k`): `(start, end, index of the cut at end)` in row order, the last part ending at `k`
/// (with no cut index unless the last cut is at `k`). Errors on cuts out of order or range.
/// Pure.
pub fn cut_parts(k: usize, rows: &[usize]) -> Result<Vec<(usize, usize, Option<usize>)>, String> {
    let mut out = Vec::with_capacity(rows.len() + 1);
    let mut start = 0;
    for (i, &r) in rows.iter().enumerate() {
        if r <= start || r > k {
            return Err(format!(
                "KDA cut {i} at row {r}: cuts must be strictly increasing rows in 1..={k} \
                 (previous {start})"
            ));
        }
        out.push((start, r, Some(i)));
        start = r;
    }
    if start < k {
        out.push((start, k, None));
    }
    Ok(out)
}

impl Glm5NextKdaLayer {
    /// 2026-10-14: `k` rows of one sequence's prefill sub-chunk with the conv and recurrence
    /// cut at `cuts` (module docs), the state advanced past all `k` rows. Refuses a config the
    /// per-sequence token kernels cannot run, a chunked-scan lever this path does not
    /// reproduce, or invalid cuts, before any launch.
    #[allow(clippy::too_many_arguments)]
    pub fn prefill_cut(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        k: usize,
        state: &KdaSeqState,
        ws: &Glm5NextKdaWorkspace,
        cuts: &[KdaCut],
        stream: u64,
    ) -> Result<()> {
        let rows: Vec<usize> = cuts.iter().map(|c| c.row).collect();
        let parts = cut_parts(k, &rows).map_err(anyhow::Error::msg)?;
        if !self.seq_tokens_ready() {
            bail!("--prefill-tail-mode merge: the per-sequence token kernels cannot run here");
        }
        if super::chunk16::kda_chunk_scan_min_rows()?.is_some()
            || crate::glm5next_layer::levers::kda_chunk_prefill()
        {
            bail!(
                "--prefill-tail-mode merge: not beside METRALE_GLM_KDA_CHUNK_SCAN or \
                 METRALE_GLM_KDA_CHUNK_PREFILL"
            );
        }
        let chunk64_min = super::chunk64::kda_chunk64_min_rows()?;
        // 2026-10-14: Once per process: the proof in the serve log that the mode reached here.
        static APPLIED: std::sync::Once = std::sync::Once::new();
        APPLIED.call_once(|| {
            tracing::info!(
                "GLM KDA: --prefill-tail-mode merge applied at {}:{}: {k} rows cut at {rows:?}",
                file!(),
                line!()
            );
        });
        let pack = self.pack_for(true)?;
        let c = &self.cfg;
        let (cd, h_bytes, conv_bytes) = (
            c.conv_dim(),
            c.recurrent_state_elems() * 4,
            c.conv_state_elems() * 4,
        );
        self.rows_with(
            gpu,
            hidden,
            k,
            ws,
            pack,
            GateAt::Kernel,
            NormAt::Back,
            stream,
            |_| Ok(()),
            || {
                for (_, conv_dst) in cuts.iter().filter_map(|c| c.dst) {
                    gpu.copy_d2d_async(state.conv, conv_dst, conv_bytes, stream)?;
                }
                self.conv_tokens(gpu, k, state, ws, pack, stream)?;
                let rows0 = RowIo::workspace(c, ws, 0);
                for cut in cuts {
                    let Some((_, conv_dst)) = cut.dst else {
                        continue;
                    };
                    KernelLaunch::new(gpu, self.kernels.seq.conv_window)
                        .grid([div_ceil(cd as u32, 256), 1, 1])
                        .block([256, 1, 1])
                        .arg_ptr(conv_dst)
                        .arg_ptr(rows0.qkv_in)
                        .arg_u32(cd as u32)
                        .arg_u32(c.conv_kernel as u32)
                        .arg_u32(cut.row as u32)
                        .launch(stream)?;
                }
                for &(a, b, at) in &parts {
                    if self.chunk64_applies(b - a, chunk64_min) {
                        self.scan_chunk64_rows(
                            gpu,
                            a,
                            b - a,
                            state,
                            ws,
                            GateAt::Kernel,
                            NormAt::Back,
                            stream,
                        )?;
                    } else {
                        self.recurrence_tokens_rows(gpu, a, b - a, state, ws, stream)?;
                    }
                    if let Some((h_dst, _)) = at.and_then(|i| cuts[i].dst) {
                        gpu.copy_d2d_async(state.recurrent, h_dst, h_bytes, stream)?;
                    }
                }
                Ok(())
            },
        )
    }
}

#[cfg(test)]
#[path = "cut_tests.rs"]
mod launch_tests;

#[cfg(test)]
mod tests {
    use super::cut_parts;

    /// 2026-10-14: The tail split of a 256-row pass at 240: two parts, the first ending at the
    /// cut; a cut at `k` ends the last part; no cut is one part.
    #[test]
    fn parts_end_at_each_cut_in_row_order() {
        assert_eq!(
            cut_parts(256, &[240]).unwrap(),
            vec![(0, 240, Some(0)), (240, 256, None)]
        );
        assert_eq!(
            cut_parts(4000, &[3968, 3984]).unwrap(),
            vec![
                (0, 3968, Some(0)),
                (3968, 3984, Some(1)),
                (3984, 4000, None)
            ]
        );
        assert_eq!(cut_parts(64, &[64]).unwrap(), vec![(0, 64, Some(0))]);
        assert_eq!(cut_parts(64, &[]).unwrap(), vec![(0, 64, None)]);
    }

    /// 2026-10-14: A cut at row 0, past `k`, repeated or out of order is refused, so no part is
    /// empty and the recurrence never runs rows twice or backwards.
    #[test]
    fn invalid_cuts_are_refused() {
        assert!(cut_parts(256, &[0]).is_err());
        assert!(cut_parts(256, &[257]).is_err());
        assert!(cut_parts(256, &[16, 16]).is_err());
        assert!(cut_parts(256, &[32, 16]).is_err());
    }
}
