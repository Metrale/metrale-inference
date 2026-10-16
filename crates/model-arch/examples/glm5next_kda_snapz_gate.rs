// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-16: Byte gate of SNAPZ (`METRALE_GLM_KDA_SNAPZ=1`) on the KDA mixer: two
//! back-to-back k = 2 verifies of C sequences, where the first verdict keeps only the first row
//! of some sequences. The eager arm copies those sequences' first snapshot into the live state
//! (the commit) before the second verify; the deferred arm skips the copy, sets the sequences'
//! flag words and gives every run its `KdaDefer` in the second verify (`decode_seqs_tokens`);
//! the resolve arm skips the copy and runs `snapz_resolve` before a second verify without
//! defers (the `decode_seq_rows` groups' path).
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants:
//! - Exits with an error unless, for every case, the second verify's output rows, every
//!   sequence's final recurrent and conv state and its snapshot are byte-identical across the
//!   three arms; unless a deferred arm with every flag zero and nothing deferred equals the
//!   eager arm without commits (a zero flag reads the live state); and unless both controls
//!   (the flags cleared while the copies are skipped, in the deferred and in the resolve arm)
//!   are caught.
//!
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!   cargo run -p metrale-model-arch --release --example glm5next_kda_snapz_gate \
//!       --features cuda,gpu-examples

#[path = "common/glm5next_rows_fixture.rs"]
mod fixture;

use anyhow::{Context, Result, bail};
use fixture::*;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::glm5next_kda::seqs_tokens::KdaDefer;
use metrale_model_arch::glm5next_kda::{Glm5NextKdaLayer, Glm5NextKdaWorkspace, KdaSeqState};
use metrale_model_arch::glm5next_layer::{GroupSpan, SeqsPlan, group_spans, kda_seqs_runs};
use metrale_model_layers::layers::ops::DENSE_GEMV_BATCHM_MAX_M;

/// 2026-10-16: (rows per sequence, which sequences the first verdict keeps one row of). The
/// leading decode row puts the last of nine k = 2 sequences across two row groups; its second
/// group takes no defer.
fn cases() -> Vec<(Vec<usize>, Vec<bool>)> {
    let mut straddle = vec![2usize; 9];
    straddle.insert(0, 1);
    vec![
        (vec![2; 2], vec![true, false]),
        (vec![2; 4], vec![false, true, true, false]),
        (vec![2; 8], vec![true; 8]),
        (straddle, (0..10).map(|i| i > 0 && i % 3 != 1).collect()),
        (vec![2; 16], (0..16).map(|i| i % 2 == 0).collect()),
    ]
}

/// 2026-10-16: How the second verify finds the first verdict's one-row commits.
#[derive(Clone, Copy, PartialEq)]
enum Mode {
    /// 2026-10-16: The commit copies the snapshot into the live state.
    Eager,
    /// 2026-10-16: Flags set, every run deferred, the seqs kernels.
    Deferred,
    /// 2026-10-16: Flags set, `snapz_resolve`, then the seqs kernels without defers.
    Resolve,
}

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

/// 2026-10-16: One arm's states, each sequence's one snapshot slot, and the flag words.
struct Arm {
    st: Vec<KdaSeqState>,
    snaps: Vec<(DevicePtr, DevicePtr)>,
    flags: DevicePtr,
}

struct Rig<'a> {
    gpu: &'a dyn GpuBackend,
    layer: Glm5NextKdaLayer,
    ws: Glm5NextKdaWorkspace,
    cb: usize,
    hb: usize,
}

impl Rig<'_> {
    fn arm(&self, init: &[KdaSeqState]) -> Result<Arm> {
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
            snaps: init
                .iter()
                .map(|_| Ok((fill(self.hb)?, fill(self.cb)?)))
                .collect::<Result<_>>()?,
            flags: up(g, &vec![0u8; 4 * init.len()])?,
        })
    }

    /// 2026-10-16: One k = 2 verify of every sequence over `hidden`, in row groups; `defer`
    /// gives each run its sequence's defer (where `kda_seqs_runs` grants it).
    fn verify(&self, hidden: DevicePtr, ks: &[usize], a: &Arm, defer: bool) -> Result<Vec<u8>> {
        let (g, row, s) = (self.gpu, HIDDEN * 2, STREAM);
        // 2026-10-16: A decode row (k = 1) keeps no snapshot, as `verify_multi` plans it.
        let lists: Vec<Option<&[(DevicePtr, DevicePtr)]>> = a
            .snaps
            .iter()
            .zip(ks)
            .map(|(p, &k)| (k > 1).then(|| std::slice::from_ref(p)))
            .collect();
        let defers: Vec<Option<KdaDefer>> = if defer {
            a.snaps
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    Some(KdaDefer {
                        flag: a.flags.offset(4 * i),
                        h: p.0,
                        conv: p.1,
                    })
                })
                .collect()
        } else {
            Vec::new()
        };
        let mut out = vec![0u8; ks.iter().sum::<usize>() * row];
        for (base, m, spans) in groups(ks) {
            let plan = kda_seqs_runs(&spans, ks, &lists, &a.st, &defers)
                .with_context(|| format!("the plan refused group at row {base}"))?;
            let SeqsPlan::Runs(runs) = plan else {
                bail!("group at row {base} walks: {plan:?}");
            };
            let x = copy(g, hidden.offset(base * row), m * row)?;
            self.layer.decode_seqs_tokens(g, x, &runs, &self.ws, s)?;
            out[base * row..(base + m) * row].copy_from_slice(&read(
                g,
                self.ws.final_out,
                m * row,
            )?);
            g.free(x)?;
        }
        Ok(out)
    }

    /// 2026-10-16: The first verify, the first verdict's one-row commits as `mode` lands
    /// them (`flag_them`: whether the flag words say so), then the second verify.
    fn two(
        &self,
        ks: &[usize],
        init: &[KdaSeqState],
        h: (DevicePtr, DevicePtr),
        one_row: &[bool],
        mode: Mode,
        flag_them: bool,
    ) -> Result<(Vec<u8>, Arm)> {
        let g = self.gpu;
        let a = self.arm(init)?;
        self.verify(h.0, ks, &a, false)?;
        let words: Vec<u8> = one_row
            .iter()
            .flat_map(|&o| u32::from(o && flag_them).to_le_bytes())
            .collect();
        g.copy_h2d(&words, a.flags)?;
        for (i, &o) in one_row.iter().enumerate() {
            match mode {
                Mode::Eager if o => {
                    g.copy_d2d(a.snaps[i].0, a.st[i].recurrent, self.hb)?;
                    g.copy_d2d(a.snaps[i].1, a.st[i].conv, self.cb)?;
                }
                Mode::Resolve => {
                    let d = KdaDefer {
                        flag: a.flags.offset(4 * i),
                        h: a.snaps[i].0,
                        conv: a.snaps[i].1,
                    };
                    self.layer.snapz_resolve(g, &a.st[i], &d, STREAM)?;
                }
                _ => {}
            }
        }
        let out = self.verify(h.1, ks, &a, mode == Mode::Deferred)?;
        g.synchronize(STREAM)?;
        Ok((out, a))
    }

    /// 2026-10-16: Output rows, final states and snapshots of `b` against `a`.
    fn compare(&self, tag: &str, a: (&[u8], &Arm), b: (&[u8], &Arm)) -> Result<()> {
        let g = self.gpu;
        same(&format!("{tag} output rows"), a.0, b.0)?;
        let r = |p| read(g, p, self.hb);
        let c = |p| read(g, p, self.cb);
        for s in 0..a.1.st.len() {
            let (x, y) = (a.1, b.1);
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
            same(
                &format!("{tag} seq {s} snapshot recurrent"),
                &r(x.snaps[s].0)?,
                &r(y.snaps[s].0)?,
            )?;
            same(
                &format!("{tag} seq {s} snapshot conv"),
                &c(x.snaps[s].1)?,
                &c(y.snaps[s].1)?,
            )?;
        }
        Ok(())
    }
}

/// 2026-10-16: Every case across the three arms, the zero-flag case, then the controls.
fn gate(gpu: &dyn GpuBackend, rng: &mut Lcg) -> Result<()> {
    let (cfg, layer, ws) = kda_fixture_heads(gpu, rng, 8)?;
    let r = Rig {
        gpu,
        layer,
        ws,
        cb: cfg.conv_state_elems() * 4,
        hb: cfg.recurrent_state_elems() * 4,
    };
    if !r.layer.seqs_tokens_ready() {
        bail!("the kda_seqs_tokens kernels are not in this build");
    }
    let state = |rng: &mut Lcg, c: usize| -> Result<Vec<KdaSeqState>> {
        (0..c)
            .map(|_| {
                Ok(KdaSeqState {
                    conv: up_f32(gpu, &rng.vec(r.cb / 4, 0.5))?,
                    recurrent: up_f32(gpu, &rng.vec(r.hb / 4, 0.1))?,
                })
            })
            .collect()
    };
    for (ks, one_row) in cases() {
        let total: usize = ks.iter().sum();
        let init = state(rng, ks.len())?;
        let h = (
            up_bf16(gpu, &rng.vec(total * HIDDEN, 1.0))?,
            up_bf16(gpu, &rng.vec(total * HIDDEN, 1.0))?,
        );
        let eager = r.two(&ks, &init, h, &one_row, Mode::Eager, true)?;
        let tag = format!("ks={ks:?} one-row={one_row:?}");
        for (name, mode) in [("deferred", Mode::Deferred), ("resolve", Mode::Resolve)] {
            let got = r.two(&ks, &init, h, &one_row, mode, true)?;
            r.compare(
                &format!("{tag} {name}"),
                (&eager.0, &eager.1),
                (&got.0, &got.1),
            )?;
        }
        println!("  {tag}: deferred and resolved commits byte-identical to the eager commit");
        if one_row.iter().any(|&o| o) {
            for (name, mode) in [("deferred", Mode::Deferred), ("resolve", Mode::Resolve)] {
                let got = r.two(&ks, &init, h, &one_row, mode, false)?;
                if r.compare("control", (&eager.0, &eager.1), (&got.0, &got.1))
                    .is_ok()
                {
                    bail!("CONTROL MISSED: {tag} {name} with the flags cleared passed the gate");
                }
            }
        }
    }
    let c = 4;
    let ks = vec![2usize; c];
    let init = state(rng, c)?;
    let h = (
        up_bf16(gpu, &rng.vec(2 * c * HIDDEN, 1.0))?,
        up_bf16(gpu, &rng.vec(2 * c * HIDDEN, 1.0))?,
    );
    let none = vec![false; c];
    let eager = r.two(&ks, &init, h, &none, Mode::Eager, true)?;
    let got = r.two(&ks, &init, h, &none, Mode::Deferred, true)?;
    r.compare("zero flags", (&eager.0, &eager.1), (&got.0, &got.1))?;
    println!("  zero flags with defers read the live state; controls: cleared flags are caught");
    Ok(())
}

fn main() -> Result<()> {
    let sets = metrale_kernels::all_ptx_sets();
    let glm = sets
        .iter()
        .find(|s| s.target.model == "glm-5.3-flash")
        .context("glm-5.3-flash kernel target not built")?;
    let gpu = MetraleCudaBackend::new(0, &glm.modules)?;
    let mut rng = Lcg(0x736e_6170_7a16);
    println!("GATE: SNAPZ deferred KDA commit vs the eager commit");
    gate(&gpu, &mut rng)?;
    println!("PASS: SNAPZ is byte-identical to the eager commit");
    Ok(())
}
