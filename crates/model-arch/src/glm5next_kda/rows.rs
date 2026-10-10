// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The KDA stateful steps of several sequences at once: `decode_seq_rows` runs row
//! `t` of every sequence as one conv launch and one recurrent launch (per 16 rows), for `t` in
//! order, so a batched decode (one row each) and a batched verify (`k` rows each) step all
//! sequences together instead of row by row.
//!
//! Owner: model-arch (GLM-5.3-Flash KDA).
//! Invariants:
//! - A sequence's rows step in row order, each on that sequence's state: row `t + 1`'s launch
//!   follows row `t`'s and the `after_row` of row `t`.
//! - A launch never holds two rows of one sequence (they share a state).
//!
//! # Why a row's bits do not change
//!
//! Block (x, r) of `causal_conv1d_update_l2norm_rows` is `causal_conv1d_update_l2norm`'s block
//! (x, 0) on row r's input, output and window, and block (h, v-slice, r) of
//! `kda_recurrent_decode_bf16_smem_rows` (or `_rows_reg`) is the single-row kernel's block
//! (h, v-slice) on row r. Rows of different sequences share nothing, so neither the grouping
//! nor the order across sequences reaches a row's arithmetic.

use super::decode::RowIo;
use super::*;

/// 2026-10-09: One row of a rows launch: its workspace row and its sequence's state.
#[derive(Clone, Copy, Debug)]
pub(super) struct RowStep {
    pub(super) row: usize,
    pub(super) state: KdaSeqState,
}

/// 2026-10-09: The rows each launch of `decode_seq_rows` takes, in launch order: for `t` from
/// 0, the row `t` of every sequence with more than `t` rows (`seqs[s]` = (state, rows), rows
/// laid out sequence-major from workspace row 0), in groups of at most `KDA_ROWS_MAX`.
///
/// Entries that share a state (the padding rows of a batched decode all step the pool's dummy
/// slot) never share a launch: the `p`-th entry of a state runs in phase `p`, after every row
/// of phase `p - 1`, so a shared state steps through its entries one after another in entry
/// order, as the row-by-row path stepped it.
pub(super) fn t_major_launches(seqs: &[(KdaSeqState, usize)]) -> Vec<Vec<RowStep>> {
    let mut row0 = Vec::with_capacity(seqs.len());
    let mut phase = Vec::with_capacity(seqs.len());
    let mut next = 0usize;
    for (i, &(state, k)) in seqs.iter().enumerate() {
        row0.push(next);
        next += k;
        phase.push(
            seqs[..i]
                .iter()
                .filter(|(s, _)| s.recurrent.0 == state.recurrent.0)
                .count(),
        );
    }
    let phases = phase.iter().copied().max().map_or(0, |p| p + 1);
    let mut out = Vec::new();
    for p in 0..phases {
        let max_k = seqs
            .iter()
            .zip(&phase)
            .filter(|(_, ph)| **ph == p)
            .map(|(&(_, k), _)| k)
            .max()
            .unwrap_or(0);
        for t in 0..max_k {
            let rows: Vec<RowStep> = seqs
                .iter()
                .zip(row0.iter().zip(&phase))
                .filter(|((_, k), (_, ph))| **ph == p && *k > t)
                .map(|((state, _), (r0, _))| RowStep {
                    row: r0 + t,
                    state: *state,
                })
                .collect();
            out.extend(rows.chunks(KDA_ROWS_MAX).map(<[RowStep]>::to_vec));
        }
    }
    out
}

impl Glm5NextKdaLayer {
    /// 2026-10-09: Whether this target has the rows kernels `decode_seq_rows` launches.
    fn rows_kernels_ready(&self) -> bool {
        self.kernels.conv_decode_rows.0 != 0
            && self.kernels.recurrent_smem_rows.0 != 0
            && self.smem_geometry().is_some()
    }

    /// 2026-10-09: Every sequence's rows through the KDA block: `seqs[s]` is (state, rows),
    /// the rows of sequence `s` contiguous in `hidden` in order, sequences one after another.
    /// The projections run once over all rows; then row `t` of every sequence steps in one
    /// conv and one recurrent launch per 16 rows, and `after_row(row)` runs for each of those
    /// rows before row `t + 1`. Without the rows kernels, or unless `METRALE_GLM_KDA_SEQ_ROWS`
    /// asks for them (`kda_seq_rows`; 2026-10-09: off by default), it steps row by row
    /// (`decode_rows_then`), with the same bits.
    pub fn decode_seq_rows(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        seqs: &[(KdaSeqState, usize)],
        ws: &Glm5NextKdaWorkspace,
        stream: u64,
        after_row: impl FnMut(usize) -> Result<()>,
    ) -> Result<()> {
        // 2026-10-09: A group with one row per sequence is a batched decode.
        let decode = seqs.iter().all(|&(_, k)| k <= 1);
        let use_rows = match kda_seq_rows() {
            KdaSeqRows::All => true,
            KdaSeqRows::DecodeOnly => decode,
            KdaSeqRows::Off => false,
        };
        self.decode_seq_rows_with(gpu, hidden, seqs, ws, stream, after_row, use_rows)
    }

    /// 2026-10-09: [`Self::decode_seq_rows`] with the rows-kernel choice explicit.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_seq_rows_with(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        seqs: &[(KdaSeqState, usize)],
        ws: &Glm5NextKdaWorkspace,
        stream: u64,
        mut after_row: impl FnMut(usize) -> Result<()>,
        // 2026-10-09: Whether to use the rows kernels; `decode_seq_rows` passes the lever.
        use_rows: bool,
    ) -> Result<()> {
        let total: usize = seqs.iter().map(|&(_, k)| k).sum();
        let multi = seqs.iter().filter(|&&(_, k)| k > 0).count() > 1;
        if !multi || !self.rows_kernels_ready() || !use_rows {
            let states: Vec<KdaSeqState> = seqs
                .iter()
                .flat_map(|&(s, k)| std::iter::repeat_n(s, k))
                .collect();
            return self.decode_rows_then(gpu, hidden, &states, ws, stream, after_row);
        }
        self.rows_with(
            gpu,
            hidden,
            total,
            ws,
            stream,
            |_| Ok(()),
            || {
                for launch in t_major_launches(seqs) {
                    self.conv_rows(gpu, &launch, ws, stream)?;
                    self.recurrent_rows(gpu, &launch, ws, stream)?;
                    for step in &launch {
                        after_row(step.row)?;
                    }
                }
                Ok(())
            },
        )
    }

    /// 2026-10-09: `causal_conv1d_update_l2norm_rows` over `rows` (at most `KDA_ROWS_MAX`).
    fn conv_rows(
        &self,
        gpu: &dyn GpuBackend,
        rows: &[RowStep],
        ws: &Glm5NextKdaWorkspace,
        stream: u64,
    ) -> Result<()> {
        check_launch(rows)?;
        let c = &self.cfg;
        let mut launch = KernelLaunch::new(gpu, self.kernels.conv_decode_rows)
            .grid([div_ceil(c.conv_dim() as u32, 256), rows.len() as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(ws.qkv_proj)
            .arg_ptr(self.weights.conv.weight)
            .arg_ptr(DevicePtr::NULL)
            .arg_ptr(ws.conv_out)
            .arg_u32(c.conv_dim() as u32)
            .arg_u32(c.conv_kernel as u32)
            .arg_u32(c.qk_channels() as u32)
            .arg_u32(c.head_dim as u32)
            .arg_f32(c.l2_eps);
        for r in 0..KDA_ROWS_MAX {
            launch = launch.arg_u64(rows.get(r).map_or(0, |s| s.state.conv.0));
        }
        for r in 0..KDA_ROWS_MAX {
            launch = launch.arg_u32(rows.get(r).map_or(0, |s| s.row as u32));
        }
        launch.launch(stream)
    }

    /// 2026-10-09: The recurrent step of `rows` (at most `KDA_ROWS_MAX`) in one launch:
    /// `kda_recurrent_decode_bf16_smem_rows` with the 1R+1W geometry, or
    /// `kda_recurrent_decode_bf16_rows_reg` under `METRALE_GLM_KDA_ROWS_REG=1` at head_dim
    /// `KDA_REG_D`.
    fn recurrent_rows(
        &self,
        gpu: &dyn GpuBackend,
        rows: &[RowStep],
        ws: &Glm5NextKdaWorkspace,
        stream: u64,
    ) -> Result<()> {
        check_launch(rows)?;
        let Some((vpb, smem)) = self.smem_geometry() else {
            bail!("KDA rows recurrence without the 1R+1W geometry");
        };
        let c = &self.cfg;
        let (qkv, cd, d) = (c.qkv_dim(), c.conv_dim(), c.head_dim);
        let io = RowIo::workspace(c, ws, 0);
        let reg = self.kernels.recurrent_rows_reg.0 != 0 && d == KDA_REG_D && kda_rows_reg();
        let mut launch = if reg {
            KernelLaunch::new(gpu, self.kernels.recurrent_rows_reg)
                .grid([c.heads as u32, 1, rows.len() as u32])
                .block([KDA_REG_D as u32, 1, 1])
                .shared_mem((3 * d * 4) as u32)
        } else {
            KernelLaunch::new(gpu, self.kernels.recurrent_smem_rows)
                .grid([c.heads as u32, (d / vpb) as u32, rows.len() as u32])
                .block([vpb as u32, 1, 1])
                .shared_mem(smem as u32)
        }
        .arg_ptr(io.conv_out)
        .arg_ptr(io.conv_out.offset(qkv * 2))
        .arg_ptr(io.conv_out.offset(qkv * 4))
        .arg_ptr(io.gate)
        .arg_ptr(io.beta)
        .arg_ptr(io.core)
        .arg_u32(c.heads as u32);
        launch = if reg {
            launch.arg_f32(1.0 / (d as f32).sqrt())
        } else {
            launch
                .arg_u32(d as u32)
                .arg_f32(1.0 / (d as f32).sqrt())
                .arg_u32(vpb as u32)
        };
        launch = launch
            .arg_u32(cd as u32)
            .arg_u32(qkv as u32)
            .arg_u32(c.heads as u32)
            .arg_u32(qkv as u32);
        for r in 0..KDA_ROWS_MAX {
            launch = launch.arg_u64(rows.get(r).map_or(0, |s| s.state.recurrent.0));
        }
        for r in 0..KDA_ROWS_MAX {
            launch = launch.arg_u32(rows.get(r).map_or(0, |s| s.row as u32));
        }
        launch.launch(stream)
    }
}

/// 2026-10-09: A rows launch holds 1..=`KDA_ROWS_MAX` rows, no two on one state.
fn check_launch(rows: &[RowStep]) -> Result<()> {
    if rows.is_empty() || rows.len() > KDA_ROWS_MAX {
        bail!(
            "KDA rows launch of {} rows (1..={KDA_ROWS_MAX})",
            rows.len()
        );
    }
    for (i, a) in rows.iter().enumerate() {
        if rows[..i]
            .iter()
            .any(|b| b.state.recurrent.0 == a.state.recurrent.0)
        {
            bail!("KDA rows launch: two rows step one state");
        }
    }
    Ok(())
}
