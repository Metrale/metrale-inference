// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: Byte gate and microbench of the multi-sequence KDA verify
//! (`Glm5NextKdaLayer::decode_seqs_tokens`, `METRALE_GLM_KDA_SEQS_TOKENS=1`): C sequences of
//! `ks[s]` rows, packed sequence-major and run in groups of `DENSE_GEMV_BATCHM_MAX_M` rows as
//! `forward_spans` runs them, the runs planned by `kda_seqs_runs`, against the per-row walk
//! (`decode_rows_then` with each sequence's state repeated over its rows and a state copy after
//! each snapshotted row, as `kda_after_row` takes it).
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants:
//! - Exits with an error unless every row's output, every sequence's final recurrent and conv
//!   state and every snapshot are byte-identical to the walk's, and unless both mutation
//!   controls (one input element changed; one snapshot dropped) are caught.
//! - The timed figures are one KDA mixer call per arm (34 copies captured into one graph, the
//!   decode's KDA layers) at 22 heads (the TP=3 per-rank KDA).
//!
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!   cargo run -p metrale-model-arch --release --example glm5next_kda_seqs_gate \
//!       --features cuda,gpu-examples

#[path = "common/glm5next_rows_fixture.rs"]
mod fixture;
#[path = "common/graph_timing.rs"]
mod timing;

use anyhow::{Context, Result, bail};
use fixture::*;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::glm5next_kda::seqs_tokens::KdaSeqRun;
use metrale_model_arch::glm5next_kda::{Glm5NextKdaLayer, Glm5NextKdaWorkspace, KdaSeqState};
use metrale_model_arch::glm5next_layer::{GroupSpan, group_spans, kda_seqs_runs};
use metrale_model_layers::layers::ops::DENSE_GEMV_BATCHM_MAX_M;

/// 2026-10-14: (rows per sequence, snapshot rollback). k = 2 is the `--num-drafts 1` verify;
/// nine of them put a sequence across two groups; the ones mix in decode rows; the
/// snapshot-free case takes runs of 3..5 rows.
fn cases() -> Vec<(Vec<usize>, bool)> {
    vec![
        (vec![2, 2], true),
        (vec![2; 4], true),
        (vec![2; 8], true),
        (vec![2; 9], true),
        (vec![2; 16], true),
        (vec![2, 1, 2, 2, 1, 2, 1, 2], true),
        (vec![3, 5, 1, 4], false),
    ]
}

/// 2026-10-14: The decode's KDA layers, the copies per timed graph.
const COPIES: usize = 34;

fn groups(ks: &[usize]) -> Vec<(usize, usize, Vec<GroupSpan>)> {
    let total: usize = ks.iter().sum();
    let cap = DENSE_GEMV_BATCHM_MAX_M as usize;
    (0..total)
        .step_by(cap)
        .map(|base| {
            let m = cap.min(total - base);
            (base, m, group_spans(ks, base, m))
        })
        .collect()
}

/// 2026-10-14: One arm: each sequence's state and its k - 1 snapshot slots (none without
/// snapshot rollback), the slots filled with the same bytes in both arms.
struct Arm {
    st: Vec<KdaSeqState>,
    snaps: Vec<Vec<(DevicePtr, DevicePtr)>>,
}

struct Rig<'a> {
    gpu: &'a dyn GpuBackend,
    layer: Glm5NextKdaLayer,
    ws: Glm5NextKdaWorkspace,
    cb: usize,
    hb: usize,
}

impl Rig<'_> {
    fn arm(&self, init: &[KdaSeqState], ks: &[usize], snap: bool) -> Result<Arm> {
        let g = self.gpu;
        let fill = |n: usize| up(g, &vec![0xA5u8; n]);
        Ok(Arm {
            st: init
                .iter()
                .map(|s| {
                    Ok(KdaSeqState {
                        conv: copy(g, s.conv, self.cb)?,
                        recurrent: copy(g, s.recurrent, self.hb)?,
                    })
                })
                .collect::<Result<_>>()?,
            snaps: ks
                .iter()
                .map(|&k| {
                    let n = if snap { k.saturating_sub(1) } else { 0 };
                    (0..n)
                        .map(|_| Ok((fill(self.hb)?, fill(self.cb)?)))
                        .collect::<Result<_>>()
                })
                .collect::<Result<_>>()?,
        })
    }

    /// 2026-10-14: The per-row walk over every group; returns the output rows.
    fn walk(&self, hidden: DevicePtr, ks: &[usize], a: &Arm, s: u64) -> Result<Vec<u8>> {
        let (g, row) = (self.gpu, HIDDEN * 2);
        let mut out = vec![0u8; ks.iter().sum::<usize>() * row];
        for (base, m, spans) in groups(ks) {
            let row_states: Vec<KdaSeqState> = spans
                .iter()
                .flat_map(|sp| std::iter::repeat_n(a.st[sp.seq], sp.rows))
                .collect();
            let x = copy(g, hidden.offset(base * row), m * row)?;
            self.layer
                .decode_rows_then(g, x, &row_states, &self.ws, s, |r| {
                    let sp = spans
                        .iter()
                        .find(|sp| (sp.row0..sp.row0 + sp.rows).contains(&r))
                        .context("row outside every span")?;
                    match a.snaps[sp.seq].get(sp.t0 + r - sp.row0) {
                        Some(dst) => self.layer.snapshot_state(g, &row_states[r], *dst, s),
                        None => Ok(()),
                    }
                })?;
            out[base * row..(base + m) * row].copy_from_slice(&read(
                g,
                self.ws.final_out,
                m * row,
            )?);
            g.free(x)?;
        }
        Ok(out)
    }

    /// 2026-10-14: `kda_seqs_runs` per group, then `decode_seqs_tokens`; `drop_snap` clears
    /// the first planned snapshot (the mutation control).
    fn seqs(
        &self,
        hidden: DevicePtr,
        ks: &[usize],
        a: &Arm,
        s: u64,
        drop_snap: bool,
    ) -> Result<Vec<u8>> {
        let (g, row) = (self.gpu, HIDDEN * 2);
        let lists: Vec<Option<&[(DevicePtr, DevicePtr)]>> = a
            .snaps
            .iter()
            .map(|v: &Vec<(DevicePtr, DevicePtr)>| Some(v.as_slice()))
            .collect();
        let mut out = vec![0u8; ks.iter().sum::<usize>() * row];
        let mut dropped = !drop_snap;
        for (base, m, spans) in groups(ks) {
            let mut runs: Vec<KdaSeqRun> = kda_seqs_runs(&spans, &lists, &a.st, &[])
                .with_context(|| format!("the plan turned down group at row {base}"))?;
            if !dropped && let Some(r) = runs.iter_mut().find(|r| r.snapshot.is_some()) {
                r.snapshot = None;
                dropped = true;
            }
            let x = copy(g, hidden.offset(base * row), m * row)?;
            self.layer.decode_seqs_tokens(g, x, &runs, &self.ws, s)?;
            out[base * row..(base + m) * row].copy_from_slice(&read(
                g,
                self.ws.final_out,
                m * row,
            )?);
            g.free(x)?;
        }
        if !dropped {
            bail!("no snapshot to drop");
        }
        Ok(out)
    }

    /// 2026-10-14: Output rows, final states and snapshots of `b` against `a`.
    fn compare(&self, tag: &str, ks: &[usize], a: (&[u8], &Arm), b: (&[u8], &Arm)) -> Result<()> {
        let g = self.gpu;
        same(&format!("{tag} output rows"), a.0, b.0)?;
        for s in 0..ks.len() {
            let (x, y) = (a.1, b.1);
            let r = |p| read(g, p, self.hb);
            let c = |p| read(g, p, self.cb);
            same(
                &format!("{tag} seq {s} recurrent"),
                &r(x.st[s].recurrent)?,
                &r(y.st[s].recurrent)?,
            )?;
            same(
                &format!("{tag} seq {s} conv"),
                &c(x.st[s].conv)?,
                &c(y.st[s].conv)?,
            )?;
            for (t, (p, q)) in x.snaps[s].iter().zip(&y.snaps[s]).enumerate() {
                same(
                    &format!("{tag} seq {s} snapshot {t} recurrent"),
                    &r(p.0)?,
                    &r(q.0)?,
                )?;
                same(
                    &format!("{tag} seq {s} snapshot {t} conv"),
                    &c(p.1)?,
                    &c(q.1)?,
                )?;
            }
        }
        Ok(())
    }

    fn init(&self, rng: &mut Lcg, c: usize) -> Result<Vec<KdaSeqState>> {
        (0..c)
            .map(|_| {
                Ok(KdaSeqState {
                    conv: up_f32(self.gpu, &rng.vec(self.cb / 4, 0.5))?,
                    recurrent: up_f32(self.gpu, &rng.vec(self.hb / 4, 0.1))?,
                })
            })
            .collect()
    }
}

fn rig<'a>(gpu: &'a dyn GpuBackend, rng: &mut Lcg, heads: usize) -> Result<Rig<'a>> {
    let (cfg, layer, ws) = kda_fixture_heads(gpu, rng, heads)?;
    Ok(Rig {
        gpu,
        layer,
        ws,
        cb: cfg.conv_state_elems() * 4,
        hb: cfg.recurrent_state_elems() * 4,
    })
}

/// 2026-10-14: Every case byte for byte, then the two mutation controls.
fn gate(gpu: &dyn GpuBackend, rng: &mut Lcg) -> Result<()> {
    let r = rig(gpu, rng, 8)?;
    if !r.layer.seqs_tokens_ready() {
        bail!("the kda_seqs_tokens kernels are not in this build");
    }
    for (ks, snap) in cases() {
        let total: usize = ks.iter().sum();
        let init = r.init(rng, ks.len())?;
        let hidden = up_bf16(gpu, &rng.vec(total * HIDDEN, 1.0))?;
        let (a, b) = (r.arm(&init, &ks, snap)?, r.arm(&init, &ks, snap)?);
        let wa = r.walk(hidden, &ks, &a, STREAM)?;
        let wb = r.seqs(hidden, &ks, &b, STREAM, false)?;
        let tag = format!("ks={ks:?} snapshots={snap}");
        r.compare(&tag, &ks, (&wa, &a), (&wb, &b))?;
        println!("  {tag}: output, states and snapshots byte-identical");
    }
    let ks = vec![2usize; 4];
    let init = r.init(rng, 4)?;
    let x = rng.vec(8 * HIDDEN, 1.0);
    let hidden = up_bf16(gpu, &x)?;
    let mut y = x.clone();
    y[7 * HIDDEN + 5] += 0.5;
    let mutated = up_bf16(gpu, &y)?;
    let arms =
        || -> Result<(Arm, Arm)> { Ok((r.arm(&init, &ks, true)?, r.arm(&init, &ks, true)?)) };
    let (a, b) = arms()?;
    let (wa, wb) = (
        r.walk(hidden, &ks, &a, STREAM)?,
        r.seqs(mutated, &ks, &b, STREAM, false)?,
    );
    if r.compare("control input", &ks, (&wa, &a), (&wb, &b))
        .is_ok()
    {
        bail!("CONTROL MISSED: a changed input element passed the gate");
    }
    let (a, b) = arms()?;
    let (wa, wb) = (
        r.walk(hidden, &ks, &a, STREAM)?,
        r.seqs(hidden, &ks, &b, STREAM, true)?,
    );
    if r.compare("control snapshot", &ks, (&wa, &a), (&wb, &b))
        .is_ok()
    {
        bail!("CONTROL MISSED: a dropped snapshot passed the gate");
    }
    println!("  controls: a changed input and a dropped snapshot are both caught");
    Ok(())
}

/// 2026-10-14: One KDA mixer call per arm at 22 heads, C = 2, 4, 8 sequences of k = 2 rows
/// with snapshot rollback, captured (the walk's per-row launches and copies against three
/// launches).
fn bench(gpu: &dyn GpuBackend, rng: &mut Lcg) -> Result<()> {
    let r = rig(gpu, rng, 22)?;
    let s = gpu.default_stream();
    for c in [2usize, 4, 8] {
        let ks = vec![2usize; c];
        let init = r.init(rng, c)?;
        let hidden = up_bf16(gpu, &rng.vec(2 * c * HIDDEN, 1.0))?;
        let (a, b) = (r.arm(&init, &ks, true)?, r.arm(&init, &ks, true)?);
        let lists: Vec<Option<&[(DevicePtr, DevicePtr)]>> = b
            .snaps
            .iter()
            .map(|v: &Vec<(DevicePtr, DevicePtr)>| Some(v.as_slice()))
            .collect();
        let spans = group_spans(&ks, 0, 2 * c);
        let runs = kda_seqs_runs(&spans, &lists, &b.st, &[]).context("plan")?;
        let row_states: Vec<KdaSeqState> = a.st.iter().flat_map(|&st| [st, st]).collect();
        let old = timing::time_graph(gpu, s, COPIES, &mut |s| {
            r.layer
                .decode_rows_then(gpu, hidden, &row_states, &r.ws, s, |row| {
                    match a.snaps[row / 2].get(row % 2) {
                        Some(dst) => r.layer.snapshot_state(gpu, &row_states[row], *dst, s),
                        None => Ok(()),
                    }
                })
        })?;
        let new = timing::time_graph(gpu, s, COPIES, &mut |s| {
            r.layer.decode_seqs_tokens(gpu, hidden, &runs, &r.ws, s)
        })?;
        timing::report("KDA verify mixer (k=2)", c, old, new);
    }
    Ok(())
}

fn main() -> Result<()> {
    let sets = metrale_kernels::all_ptx_sets();
    let glm = sets
        .iter()
        .find(|s| s.target.model == "glm-5.3-flash")
        .context("glm-5.3-flash kernel target not built")?;
    let gpu = MetraleCudaBackend::new(0, &glm.modules)?;
    let mut rng = Lcg(0x6c6d_5e9d_0014);
    println!("GATE+BENCH: multi-sequence KDA verify (decode_seqs_tokens) vs the per-row walk");
    gate(&gpu, &mut rng)?;
    bench(&gpu, &mut rng)?;
    println!("PASS: the multi-sequence KDA verify is byte-identical to the per-row walk");
    Ok(())
}
