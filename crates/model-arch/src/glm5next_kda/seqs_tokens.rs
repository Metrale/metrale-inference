// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: `decode_seqs_tokens`, the stateful half of a batched speculative verify: each
//! sequence's run of consecutive workspace rows through the conv and the recurrence, up to
//! `KDA_SEQS_MAX` sequences per launch, in three launches (conv, conv window advance,
//! recurrence) instead of a conv and a recurrent launch per row index (`decode_seq_rows`) or
//! per row (`decode_rows_then`), plus a state copy per snapshotted row.
//!
//! Owner: model-arch (GLM-5.3-Flash KDA).
//! Invariants:
//! - Runs are laid out sequence-major from workspace row 0, in `runs` order.
//! - No two runs share a state; every state pointer is set. Checked before any launch.
//! - A run keeps at most one snapshot: the state after its first row.
//!
//! # Why every row's bits are the per-row walk's
//!
//! Block z = r of each `_seqs` kernel (`kernels/gb10/common/kda_seqs_tokens.cu`) is the
//! per-sequence token kernel's launch (`stateful_tokens`, `stateful_tokens_snap`) on run r's
//! rows and state, and those give each row the per-row walk's bits (`seq_tokens.rs`). Runs
//! share nothing, so neither the grouping nor the order across sequences reaches a row. The
//! conv snapshot is the window after one row, which `causal_conv1d_window_advance` at T = 1
//! leaves on a copy of the window (`stateful_tokens_snap`); the recurrent snapshot is the
//! column as it passes row 0, which the walk's copy after row 0 reads back.

use super::decode::RowIo;
use super::seq_tokens::SEQ_VB;
use super::*;

/// 2026-10-14: `#define KDA_SEQS_MAX` in `kda_seqs_tokens.cu`: the runs one launch takes.
pub const KDA_SEQS_MAX: usize = 16;

/// 2026-10-16: SNAPZ (`METRALE_GLM_KDA_SNAPZ=1`): where a run whose last verdict kept only its
/// first verify row finds that state. `flag` is the run's batch-position word on the device:
/// nonzero means the commit that would copy `(h, conv)` (the slot's first verify
/// intermediates) into the live state was deferred, so the kernels read the state from there;
/// zero reads the live state. Writes go where they always do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KdaDefer {
    pub flag: DevicePtr,
    pub h: DevicePtr,
    pub conv: DevicePtr,
}

/// 2026-10-14: One sequence's run: `rows` consecutive workspace rows stepping `state`, and
/// where to keep the state after the first of them (`(h_dst, conv_dst)`), if anywhere.
/// 2026-10-16: `defer`: SNAPZ, see [`KdaDefer`] (`None`: the live state, as before).
#[derive(Clone, Copy, Debug)]
pub struct KdaSeqRun {
    pub state: KdaSeqState,
    pub rows: usize,
    pub snapshot: Option<(DevicePtr, DevicePtr)>,
    pub defer: Option<KdaDefer>,
}

/// 2026-10-14: `METRALE_GLM_KDA_SEQS_TOKENS=1` steps a batched speculative verify's KDA runs
/// in [`Glm5NextKdaLayer::decode_seqs_tokens`]'s three launches (same bits per row). Off until
/// measured end to end. Read once.
pub fn kda_seqs_tokens() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("METRALE_GLM_KDA_SEQS_TOKENS").as_deref() == Ok("1"))
}

/// 2026-10-14: The total rows of `runs`, or why they cannot run as one batched verify: no run,
/// an empty run, an unset state, snapshot or defer pointer, or two runs on one state.
pub(super) fn check_runs(runs: &[KdaSeqRun]) -> Result<usize> {
    if runs.is_empty() {
        bail!("KDA seqs verify: no runs");
    }
    for (i, a) in runs.iter().enumerate() {
        let snap_unset = a.snapshot.is_some_and(|(h, c)| h.0 == 0 || c.0 == 0)
            || a.defer
                .is_some_and(|d| d.flag.0 == 0 || d.h.0 == 0 || d.conv.0 == 0);
        if a.rows == 0 || a.state.recurrent.0 == 0 || a.state.conv.0 == 0 || snap_unset {
            bail!("KDA seqs verify: run {i} has no rows or an unset pointer");
        }
        if runs[..i]
            .iter()
            .any(|b| b.state.recurrent.0 == a.state.recurrent.0 || b.state.conv.0 == a.state.conv.0)
        {
            bail!("KDA seqs verify: run {i} shares a state with an earlier run");
        }
    }
    Ok(runs.iter().map(|r| r.rows).sum())
}

impl Glm5NextKdaLayer {
    /// 2026-10-14: Whether this target and config can run [`Self::decode_seqs_tokens`].
    pub fn seqs_tokens_ready(&self) -> bool {
        let s = &self.kernels.seq;
        s.conv_seqs.0 != 0
            && s.window_seqs.0 != 0
            && s.recurrent_seqs.0 != 0
            && self.cfg.head_dim == KDA_REG_D
            && self.cfg.conv_kernel <= 4
    }

    /// 2026-10-14: Every run's rows through the KDA block: the projections once over all rows,
    /// then per `KDA_SEQS_MAX` runs one conv, one window-advance and one recurrent launch, then
    /// the output half. Refuses (before any launch) runs [`check_runs`] refuses or a target
    /// [`Self::seqs_tokens_ready`] turns down.
    pub fn decode_seqs_tokens(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        runs: &[KdaSeqRun],
        ws: &Glm5NextKdaWorkspace,
        stream: u64,
    ) -> Result<()> {
        let total = check_runs(runs)?;
        if !self.seqs_tokens_ready() {
            bail!("KDA seqs verify: the seqs kernels cannot run here");
        }
        // 2026-10-16: Once per process: the serve log's proof that the lever reached the seqs
        // kernels, not only that the variable was set.
        static DISPATCHED: std::sync::Once = std::sync::Once::new();
        DISPATCHED.call_once(|| {
            tracing::warn!(
                "METRALE_GLM_KDA_SEQS_TOKENS: multi-sequence KDA verify dispatched (rank-local, \
                 first call): {} runs, {total} rows",
                runs.len()
            );
        });
        self.rows_with(
            gpu,
            hidden,
            total,
            ws,
            Pack::InFrontEnd,
            GateAt::Kernel,
            stream,
            |_| Ok(()),
            || self.seqs_chunks(gpu, runs, ws, stream),
        )
    }

    /// 2026-10-14: The stateful half of [`Self::decode_seqs_tokens`]: per `KDA_SEQS_MAX` runs
    /// one [`Self::seqs_launches`], each run's first row its absolute workspace row.
    fn seqs_chunks(
        &self,
        gpu: &dyn GpuBackend,
        runs: &[KdaSeqRun],
        ws: &Glm5NextKdaWorkspace,
        stream: u64,
    ) -> Result<()> {
        let mut row0 = 0usize;
        for chunk in runs.chunks(KDA_SEQS_MAX) {
            let w: Vec<u32> = chunk
                .iter()
                .scan(row0, |r, run| {
                    let at = *r;
                    *r += run.rows;
                    Some(at as u32)
                })
                .collect();
            self.seqs_launches(gpu, chunk, &w, ws, stream)?;
            row0 += chunk.iter().map(|r| r.rows).sum::<usize>();
        }
        Ok(())
    }

    /// 2026-10-14: The three launches of one chunk of runs; `w[r]` is run r's first row.
    fn seqs_launches(
        &self,
        gpu: &dyn GpuBackend,
        runs: &[KdaSeqRun],
        w: &[u32],
        ws: &Glm5NextKdaWorkspace,
        stream: u64,
    ) -> Result<()> {
        let c = &self.cfg;
        let s = &self.kernels.seq;
        let (qkv, cd, d) = (c.qkv_dim(), c.conv_dim(), c.head_dim);
        let io = RowIo::workspace(c, ws, 0);
        let nseq = runs.len() as u32;
        let max_rows = runs.iter().map(|r| r.rows).max().unwrap_or(0) as u32;
        let ptrs = |l, f: fn(&KdaSeqRun) -> u64| seq_ptrs(l, runs, f);
        let rows = |l| seq_rows(l, runs, w);
        let conv = KernelLaunch::new(gpu, s.conv_seqs)
            .grid([div_ceil(cd as u32, 256), max_rows, nseq])
            .block([256, 1, 1])
            .arg_ptr(io.qkv_in)
            .arg_ptr(self.weights.conv.weight)
            .arg_ptr(DevicePtr::NULL)
            .arg_ptr(io.conv_out)
            .arg_u32(cd as u32)
            .arg_u32(c.conv_kernel as u32)
            .arg_u32(c.qk_channels() as u32)
            .arg_u32(d as u32)
            .arg_f32(c.l2_eps);
        let defer = |l, z: fn(&KdaDefer) -> DevicePtr| seq_defer(l, runs, z);
        defer(rows(ptrs(conv, |r| r.state.conv.0)), |d| d.conv).launch(stream)?;
        let window = KernelLaunch::new(gpu, s.window_seqs)
            .grid([div_ceil(cd as u32, 256), nseq, 1])
            .block([256, 1, 1])
            .arg_ptr(io.qkv_in)
            .arg_u32(cd as u32)
            .arg_u32(c.conv_kernel as u32);
        let window = ptrs(window, |r| r.state.conv.0);
        let window = rows(ptrs(window, |r| r.snapshot.map_or(0, |p| p.1.0)));
        defer(window, |d| d.conv).launch(stream)?;
        let recurrent = KernelLaunch::new(gpu, s.recurrent_seqs)
            .grid([c.heads as u32, (d / SEQ_VB) as u32, nseq])
            .block([SEQ_VB as u32, 1, 1])
            .shared_mem((3 * d * 4) as u32)
            .arg_ptr(io.conv_out)
            .arg_ptr(io.conv_out.offset(qkv * 2))
            .arg_ptr(io.conv_out.offset(qkv * 4))
            .arg_ptr(io.gate)
            .arg_ptr(io.beta)
            .arg_ptr(io.core)
            .arg_u32(c.heads as u32)
            .arg_f32(1.0 / (d as f32).sqrt())
            .arg_u32(cd as u32)
            .arg_u32(qkv as u32)
            .arg_u32(c.heads as u32)
            .arg_u32(qkv as u32);
        let recurrent = ptrs(recurrent, |r| r.state.recurrent.0);
        let recurrent = rows(ptrs(recurrent, |r| r.snapshot.map_or(0, |p| p.0.0)));
        defer(recurrent, |d| d.h).launch(stream)
    }

    /// 2026-10-16: SNAPZ on a group that keeps `decode_seq_rows` (no seqs kernels): the
    /// deferred commit of `state` on the device, `kda_snapz_resolve` once per blob: when the
    /// run's flag word is set, the live state <- `(defer.h, defer.conv)`; else nothing. Called
    /// before the group's first row of that sequence, so the rows read what the seqs kernels
    /// would have read.
    pub fn snapz_resolve(
        &self,
        gpu: &dyn GpuBackend,
        state: &KdaSeqState,
        defer: &KdaDefer,
        stream: u64,
    ) -> Result<()> {
        let k = self.kernels.seq.snapz_resolve;
        if k.0 == 0 {
            bail!("KDA SNAPZ: kda_snapz_resolve is not loaded");
        }
        let c = &self.cfg;
        let blobs = [
            (defer.h, state.recurrent, c.recurrent_state_elems() * 4),
            (defer.conv, state.conv, c.conv_state_elems() * 4),
        ];
        for (src, dst, bytes) in blobs {
            if bytes % 16 != 0 {
                bail!("KDA SNAPZ: a {bytes}-byte state is not whole 16-byte words");
            }
            let n16 = bytes / 16;
            KernelLaunch::new(gpu, k)
                .grid([div_ceil(n16 as u32, 256).min(SNAPZ_RESOLVE_BLOCKS), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(defer.flag)
                .arg_ptr(src)
                .arg_ptr(dst)
                .arg_u64(n16 as u64)
                .launch(stream)?;
        }
        Ok(())
    }
}

/// 2026-10-16: The grid cap of `kda_snapz_resolve` (grid-stride beyond it).
const SNAPZ_RESOLVE_BLOCKS: u32 = 1024;

/// 2026-10-16: SNAPZ's two per-run arguments: each run's flag word, then `z` of its defer (the
/// state it reads when the flag is set), zeros for a run without one and after the last.
fn seq_defer<'a>(
    l: KernelLaunch<'a>,
    runs: &[KdaSeqRun],
    z: fn(&KdaDefer) -> DevicePtr,
) -> KernelLaunch<'a> {
    let pick = |r: usize, f: &dyn Fn(&KdaDefer) -> DevicePtr| {
        runs.get(r).and_then(|x| x.defer).map_or(0, |d| f(&d).0)
    };
    let l = (0..KDA_SEQS_MAX).fold(l, |l, r| l.arg_u64(pick(r, &|d| d.flag)));
    (0..KDA_SEQS_MAX).fold(l, |l, r| l.arg_u64(pick(r, &z)))
}

/// 2026-10-14: `KDA_SEQS_MAX` pointer arguments: `f` of each run, then zeros.
fn seq_ptrs<'a>(
    l: KernelLaunch<'a>,
    runs: &[KdaSeqRun],
    f: fn(&KdaSeqRun) -> u64,
) -> KernelLaunch<'a> {
    (0..KDA_SEQS_MAX).fold(l, |l, r| l.arg_u64(runs.get(r).map_or(0, f)))
}

/// 2026-10-14: The first-row arguments `w`, then the row counts, each padded with zeros to
/// `KDA_SEQS_MAX`.
fn seq_rows<'a>(l: KernelLaunch<'a>, runs: &[KdaSeqRun], w: &[u32]) -> KernelLaunch<'a> {
    let l = (0..KDA_SEQS_MAX).fold(l, |l, r| l.arg_u32(w.get(r).copied().unwrap_or(0)));
    (0..KDA_SEQS_MAX).fold(l, |l, r| {
        l.arg_u32(runs.get(r).map_or(0, |x| x.rows as u32))
    })
}

#[cfg(test)]
#[path = "seqs_tokens_tests.rs"]
mod tests;
