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
//!   state and every snapshot are byte-identical to the walk's, and unless every mutation
//!   control is caught: one input element changed, the last snapshot of an 8-row run dropped,
//!   two snapshot slots of a 4-row run swapped, one byte of a deep snapshot flipped.
//! - 2026-10-16: Spans of 3, 4, 5 and 8 rows keep every snapshot they owe (multi-snapshot
//!   kernels). `--negctl <input|drop-snap|swap-snap|flip-snap>` runs only that control as the
//!   gate: it must exit nonzero (a zero exit prints `NOT CAUGHT`).
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
use metrale_model_arch::glm5next_kda::seqs_tokens::{KdaSeqRun, KdaSnaps};
use metrale_model_arch::glm5next_kda::{Glm5NextKdaLayer, Glm5NextKdaWorkspace, KdaSeqState};
use metrale_model_arch::glm5next_layer::{GroupSpan, SeqsPlan, group_spans, kda_seqs_runs};
use metrale_model_layers::layers::ops::DENSE_GEMV_BATCHM_MAX_M;

/// 2026-10-14: (rows per sequence, snapshot rollback). k = 2 is the `--num-drafts 1` verify;
/// nine of them put a sequence across two groups; the ones mix in decode rows; the
/// snapshot-free case takes runs of 3..5 rows.
/// 2026-10-16: k = 3, 4, 5, 8 keep k - 1 snapshots per sequence (`[5, 8, 8]` puts an 8-row
/// sequence across two groups, mid-snapshots); 1-row sequences mix with 3-row ones.
fn cases() -> Vec<(Vec<usize>, bool)> {
    vec![
        (vec![2, 2], true),
        (vec![2; 4], true),
        (vec![2; 8], true),
        (vec![2; 9], true),
        (vec![2; 16], true),
        (vec![2, 1, 2, 2, 1, 2, 1, 2], true),
        (vec![3, 5, 1, 4], false),
        (vec![3, 3], true),
        (vec![3; 5], true),
        (vec![4; 4], true),
        (vec![8, 8], true),
        (vec![5, 8, 8], true),
        (vec![1, 3, 1, 3, 1, 3], true),
        (vec![8, 3, 4], true),
        (vec![3, 5, 1, 4], true),
    ]
}

/// 2026-10-16: A mutation of the seqs arm that the gate must catch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Perturb {
    None,
    /// One input element changed.
    Input,
    /// The last snapshot of an 8-row run (seven slots) not planned.
    DropSnap,
    /// Snapshot slots 1 and 2 of a 4-row run (three slots) swapped.
    SwapSnap,
    /// One byte of snapshot 6 of the 8-row sequence flipped after the arm.
    FlipSnap,
}

const CONTROLS: [(&str, Perturb); 4] = [
    ("input", Perturb::Input),
    ("drop-snap", Perturb::DropSnap),
    ("swap-snap", Perturb::SwapSnap),
    ("flip-snap", Perturb::FlipSnap),
];

/// 2026-10-16: The controls' batch: a 4-, an 8- and a 4-row sequence in one group.
const CONTROL_KS: [usize; 3] = [4, 8, 4];

/// 2026-10-16: The first run keeping `len` slots, its list changed by `f`; whether one was.
fn mutate_snaps(
    runs: &mut [KdaSeqRun],
    len: usize,
    f: impl Fn(&[(DevicePtr, DevicePtr)]) -> Vec<(DevicePtr, DevicePtr)>,
) -> Result<bool> {
    let Some(r) = runs
        .iter_mut()
        .find(|r| r.snapshots.as_slice().len() == len)
    else {
        return Ok(false);
    };
    r.snapshots = KdaSnaps::new(&f(r.snapshots.as_slice()))?;
    Ok(true)
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

/// 2026-10-16: Each sequence's slot list as `verify_multi` plans it: `None` for none kept.
fn snap_lists(a: &Arm) -> Vec<Option<&[(DevicePtr, DevicePtr)]>> {
    a.snaps
        .iter()
        .map(|v| (!v.is_empty()).then_some(v.as_slice()))
        .collect()
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

    /// 2026-10-14: `kda_seqs_runs` per group, then `decode_seqs_tokens`.
    /// 2026-10-16: `p` mutates the planned snapshots of the first run it fits (the controls);
    /// a group the plan walks fails the gate, since it would not test the kernels.
    fn seqs(
        &self,
        hidden: DevicePtr,
        ks: &[usize],
        a: &Arm,
        s: u64,
        p: Perturb,
    ) -> Result<Vec<u8>> {
        let (g, row) = (self.gpu, HIDDEN * 2);
        let lists = snap_lists(a);
        let mut out = vec![0u8; ks.iter().sum::<usize>() * row];
        let mut done = !matches!(p, Perturb::DropSnap | Perturb::SwapSnap);
        for (base, m, spans) in groups(ks) {
            let plan = kda_seqs_runs(&spans, ks, &lists, &a.st, &[])
                .with_context(|| format!("the plan refused group at row {base}"))?;
            let SeqsPlan::Runs(mut runs) = plan else {
                bail!("group at row {base} walks: {plan:?}");
            };
            if !done {
                done = match p {
                    Perturb::DropSnap => mutate_snaps(&mut runs, 7, |v| v[..6].to_vec())?,
                    _ => mutate_snaps(&mut runs, 3, |v| vec![v[0], v[2], v[1]])?,
                };
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
        if !done {
            bail!("no run fits the {p:?} control");
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

/// 2026-10-16: The gate's rig, refused when the seqs kernels are not in the build.
fn ready_rig<'a>(gpu: &'a dyn GpuBackend, rng: &mut Lcg) -> Result<Rig<'a>> {
    let r = rig(gpu, rng, 8)?;
    if !r.layer.seqs_tokens_ready() {
        bail!("the kda_seqs_tokens kernels are not in this build");
    }
    Ok(r)
}

/// 2026-10-16: One control on `CONTROL_KS`: the outer `Err` is a broken rig, the inner one the
/// gate's verdict (an `Err` there is the control caught).
fn control(r: &Rig, rng: &mut Lcg, p: Perturb) -> Result<Result<()>> {
    let ks = CONTROL_KS;
    let total: usize = ks.iter().sum();
    let init = r.init(rng, ks.len())?;
    let x = rng.vec(total * HIDDEN, 1.0);
    let hidden = up_bf16(r.gpu, &x)?;
    let mut y = x;
    if p == Perturb::Input {
        y[(total - 1) * HIDDEN + 5] += 0.5;
    }
    let fed = up_bf16(r.gpu, &y)?;
    let (a, b) = (r.arm(&init, &ks, true)?, r.arm(&init, &ks, true)?);
    let wa = r.walk(hidden, &ks, &a, STREAM)?;
    let wb = r.seqs(fed, &ks, &b, STREAM, p)?;
    if p == Perturb::FlipSnap {
        let at = b.snaps[1][6].0;
        let mut v = read(r.gpu, at, r.hb)?;
        v[r.hb / 2] ^= 0x01;
        r.gpu.copy_h2d(&v, at)?;
    }
    Ok(r.compare(&format!("control {p:?}"), &ks, (&wa, &a), (&wb, &b)))
}

/// 2026-10-14: Every case byte for byte, then the mutation controls.
fn gate(gpu: &dyn GpuBackend, rng: &mut Lcg) -> Result<()> {
    let r = ready_rig(gpu, rng)?;
    for (ks, snap) in cases() {
        let total: usize = ks.iter().sum();
        let init = r.init(rng, ks.len())?;
        let hidden = up_bf16(gpu, &rng.vec(total * HIDDEN, 1.0))?;
        let (a, b) = (r.arm(&init, &ks, snap)?, r.arm(&init, &ks, snap)?);
        let wa = r.walk(hidden, &ks, &a, STREAM)?;
        let wb = r.seqs(hidden, &ks, &b, STREAM, Perturb::None)?;
        let tag = format!("ks={ks:?} snapshots={snap}");
        r.compare(&tag, &ks, (&wa, &a), (&wb, &b))?;
        println!("  {tag}: output, states and snapshots byte-identical");
    }
    for (name, p) in CONTROLS {
        if control(&r, rng, p)?.is_ok() {
            bail!("CONTROL MISSED: the {name} control passed the gate");
        }
        println!("  control {name}: caught");
    }
    println!(
        "  controls: a changed input, a dropped, a swapped and a flipped snapshot are all caught"
    );
    Ok(())
}

/// 2026-10-14: One KDA mixer call per arm at 22 heads, C sequences of k rows with snapshot
/// rollback, captured (the walk's per-row launches and copies against three launches).
/// 2026-10-16: k = 3, 4 and 8 keep their k - 1 snapshots in the same three launches.
fn bench(gpu: &dyn GpuBackend, rng: &mut Lcg) -> Result<()> {
    let r = rig(gpu, rng, 22)?;
    let s = gpu.default_stream();
    for (k, c) in [
        (2usize, 2usize),
        (2, 4),
        (2, 8),
        (3, 4),
        (4, 2),
        (4, 4),
        (8, 2),
    ] {
        let ks = vec![k; c];
        let rows = k * c;
        let init = r.init(rng, c)?;
        let hidden = up_bf16(gpu, &rng.vec(rows * HIDDEN, 1.0))?;
        let (a, b) = (r.arm(&init, &ks, true)?, r.arm(&init, &ks, true)?);
        let spans = group_spans(&ks, 0, rows);
        let plan = kda_seqs_runs(&spans, &ks, &snap_lists(&b), &b.st, &[]).context("plan")?;
        let SeqsPlan::Runs(runs) = plan else {
            bail!("the k={k} C={c} bench plan walks: {plan:?}");
        };
        let row_states: Vec<KdaSeqState> =
            a.st.iter()
                .flat_map(|&st| std::iter::repeat_n(st, k))
                .collect();
        let old = timing::time_graph(gpu, s, COPIES, &mut |s| {
            r.layer
                .decode_rows_then(gpu, hidden, &row_states, &r.ws, s, |row| {
                    match a.snaps[row / k].get(row % k) {
                        Some(dst) => r.layer.snapshot_state(gpu, &row_states[row], *dst, s),
                        None => Ok(()),
                    }
                })
        })?;
        let new = timing::time_graph(gpu, s, COPIES, &mut |s| {
            r.layer.decode_seqs_tokens(gpu, hidden, &runs, &r.ws, s)
        })?;
        timing::report(&format!("KDA verify mixer (k={k})"), c, old, new);
    }
    Ok(())
}

/// 2026-10-16: `--negctl <name>` runs that control alone: exit 3 when caught, 0 (with
/// `NOT CAUGHT`) when missed, 1 when the rig fails.
fn negctl(gpu: &dyn GpuBackend, rng: &mut Lcg, name: &str) -> Result<()> {
    let p = CONTROLS
        .iter()
        .find(|c| c.0 == name)
        .map(|c| c.1)
        .with_context(|| format!("unknown control {name}"))?;
    let r = ready_rig(gpu, rng)?;
    match control(&r, rng, p)? {
        Err(e) => {
            println!("NEGCTL {name}: CAUGHT ({e:#})");
            std::process::exit(3);
        }
        Ok(()) => {
            println!("NEGCTL {name}: NOT CAUGHT");
            Ok(())
        }
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let sets = metrale_kernels::all_ptx_sets();
    let glm = sets
        .iter()
        .find(|s| s.target.model == "glm-5.3-flash")
        .context("glm-5.3-flash kernel target not built")?;
    let gpu = MetraleCudaBackend::new(0, &glm.modules)?;
    let mut rng = Lcg(0x6c6d_5e9d_0014);
    match args.as_slice() {
        [] => {}
        [flag, name] if flag == "--negctl" => return negctl(&gpu, &mut rng, name),
        _ => {
            bail!("usage: glm5next_kda_seqs_gate [--negctl <input|drop-snap|swap-snap|flip-snap>]")
        }
    }
    println!("GATE+BENCH: multi-sequence KDA verify (decode_seqs_tokens) vs the per-row walk");
    gate(&gpu, &mut rng)?;
    bench(&gpu, &mut rng)?;
    println!("PASS: the multi-sequence KDA verify is byte-identical to the per-row walk");
    Ok(())
}
