// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: Byte gate and microbench of `METRALE_GLM_KDA_SCAN_ONORM`: the 64-token chunked
//! scan with the sigmoid-gated output RMSNorm in its epilogue (`kda_chunk64_scan_onorm`, a
//! (1, 2, 1) cluster per head exchanging row sums over DSMEM) against the path it replaces
//! (`kda_chunk64_scan` into the FP32 core rows, then `kda_o_norm_gated_bf16`), at GLM-5.3's TP=3
//! per-rank shape (22 heads of 128), for both input layouts, the conv in front of or inside the
//! prepare (`METRALE_GLM_KDA_CHUNK64_CONV`), the gate launched or inside the prepare
//! (`METRALE_GLM_KDA_GATE_IN_PREPARE`), and a recurrent state that is zero (a fresh prefill) or
//! not (a prefix-cache hit continuing a sequence).
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants:
//! - Exits with an error on the first byte of the fused arm that differs from scan-then-norm:
//!   the BF16 normed rows (with 64 poisoned rows past T that neither arm may write), the
//!   recurrent state, the conv state, or any slab's six factor buffers; and unless the fused arm
//!   leaves the poisoned core rows untouched. Rows 70, 300, 1100 and 8192 (or
//!   `GLM_PARITY_ROWS`) cover a partial chunk, a partial slab, many slabs with a partial tail
//!   chunk, and a prefill sub-chunk; each head's two column blocks are the two blocks of a
//!   cluster, so every row covers both halves of the exchange. The output gate is noise of
//!   scale 3 with every 1013th element +-40 (both ends of the sigmoid).
//! - Exits with an error unless the base output is finite and both negative controls fire:
//!   (a) the norm in another reduction order (`kda_o_norm_gated_bf16` at 64 threads per row:
//!   two squares per thread, two warp sums) differs from the base, so these rows tell partial
//!   orders apart; (b) the fused arm with one output-gate element (row T - 2, head 5, column 3)
//!   moved 4 across zero differs from the base output and leaves the recurrent state equal.
//! - Exits with an error on the first byte of scan-then-norm (the lever off) that differs from
//!   the same path with the base tree's `kda_chunk64_scan` (`GLM_PARITY_BASE_PTX`, loaded as a
//!   second module) in the core rows, the states or the factors: the scan gained a templated
//!   twin, and the lever-off scan must not move. `GLM_PARITY_FIDELITY_PTX` (this tree's
//!   `kda_chunk64.cu` compiled by the same command as the base) must equal the embedded module
//!   byte for byte, which proves the base was compiled as the build compiles.
//! - Times, as captured CUDA graphs (median [min..max] of 7, `common/graph_timing.rs`): the
//!   whole path then the norm against the fused path, the slab scans then the norm against the
//!   fused scans, and the norm alone.
//!
//!   GLM_PARITY_BASE_PTX=base.ptx GLM_PARITY_FIDELITY_PTX=new.ptx \
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!   cargo run -p metrale-model-arch --release --example glm5next_kda_scan_onorm_gate \
//!       --features cuda,gpu-examples

#[path = "common/glm5next_kda_chunk64_conv_arms.rs"]
mod arms;
#[path = "common/glm5next_kda_chunk64_case.rs"]
mod cases;
#[path = "common/glm5next_rows_fixture.rs"]
mod fixture;
#[path = "common/graph_timing.rs"]
mod timing;

use anyhow::{Context, Result, bail, ensure};
use arms::{
    Arm, Gate, GateRaw, H, In, Kernels, Layout, Norm, QKV, Run, SCAN_BLOCK, SCAN_GRID, fused,
    gate_inputs, gate_launch, scan, unfused,
};
use cases::{bumped_copy, case_rows, compare, ensure_finite, free_inputs, inputs, leak, run_arm};
use fixture::{Lcg, same, up_bf16};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_arch::glm5next_kda::chunk64::{
    CHUNK64 as C, CHUNK64_D as D, CHUNK64_SCAN_ONORM_SMEM, CHUNK64_SLAB as SLAB,
};

/// 2026-10-15: GLM-5.3-Flash's `rms_norm_eps`.
const EPS: f32 = 1e-5;
/// 2026-10-15: Poisoned rows after T in each output buffer: no arm may write them.
const PAD: usize = 64;
const POISON: u8 = 0xA5;

/// 2026-10-15: The kernels beyond the chunk64 arms': the gate, the two `_graw` prepares and the
/// norm.
struct More {
    gate: KernelHandle,
    prep_graw: KernelHandle,
    prep_conv_graw: KernelHandle,
    norm: KernelHandle,
}

/// 2026-10-15: Where the forget gate is computed (`METRALE_GLM_KDA_GATE_IN_PREPARE`).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Place {
    Kernel,
    Prepare,
}

/// 2026-10-15: One case: what every path launches on.
struct Ctx<'a> {
    g: &'a dyn GpuBackend,
    k: &'a Kernels,
    /// 2026-10-15: `k` with the base tree's `kda_chunk64_scan`.
    kb: &'a Kernels,
    m: &'a More,
    x: &'a In,
    r: &'a GateRaw,
}

impl Ctx<'_> {
    /// 2026-10-15: The forward of `a` (its scans fused with the norm when [`Arm::norm`] is set).
    fn path(
        &self,
        a: &Arm,
        layout: Layout,
        conv: bool,
        place: Place,
        s: u64,
        rec: Option<&mut Vec<[Vec<u8>; 6]>>,
    ) -> Result<()> {
        self.path_on(self.k, a, layout, conv, place, s, rec)
    }

    /// 2026-10-15: [`Ctx::path`] on the kernels `k`.
    #[allow(clippy::too_many_arguments)]
    fn path_on(
        &self,
        k: &Kernels,
        a: &Arm,
        layout: Layout,
        conv: bool,
        place: Place,
        s: u64,
        rec: Option<&mut Vec<[Vec<u8>; 6]>>,
    ) -> Result<()> {
        let (g, m, x, r) = (self.g, self.m, self.x, self.r);
        let (prep, gate) = match (conv, place) {
            (false, Place::Kernel) => (k.prep, Gate::Rows),
            (true, Place::Kernel) => (k.prep_conv, Gate::Rows),
            (false, Place::Prepare) => (m.prep_graw, Gate::Raw(*r)),
            (true, Place::Prepare) => (m.prep_conv_graw, Gate::Raw(*r)),
        };
        if place == Place::Kernel {
            gate_launch(g, m.gate, r, x, s)?;
        }
        if conv {
            fused(g, k, prep, gate, x, a, layout, x.src(layout), s, rec)
        } else {
            unfused(g, k, prep, gate, x, a, layout, s, rec)
        }
    }

    /// 2026-10-15: `kda_o_norm_gated_bf16` of `core` into `nm.out`, `threads` per row (the
    /// layer launches 128).
    fn norm(&self, core: DevicePtr, nm: &Norm, threads: u32, s: u64) -> Result<()> {
        KernelLaunch::new(self.g, self.m.norm)
            .grid([(self.x.t * H) as u32, 1, 1])
            .block([threads, 1, 1])
            .arg_ptr(core)
            .arg_ptr(nm.gate)
            .arg_ptr(nm.weight)
            .arg_ptr(nm.out)
            .arg_u32(D as u32)
            .arg_f32(nm.eps)
            .launch(s)
    }

    /// 2026-10-15: Every slab's scan of `a`, on the factors its last forward left (timing only).
    fn scans(&self, a: &Arm, s: u64) -> Result<()> {
        let nc = self.x.t.div_ceil(C);
        for c0 in (0..nc).step_by(SLAB) {
            scan(self.g, self.k, self.x, a, c0, SLAB.min(nc - c0), s)?;
        }
        Ok(())
    }
}

/// 2026-10-15: The output buffer's bytes, pad included.
fn out_bytes(t: usize) -> usize {
    (t + PAD) * QKV * 2
}

fn read(g: &dyn GpuBackend, s: u64, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    g.synchronize(s)?;
    let mut b = vec![0u8; n];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

/// 2026-10-15: An error unless the first `t` rows of `out` are finite BF16 and the pad is
/// still poison.
fn ensure_out(what: &str, out: &[u8], t: usize) -> Result<()> {
    let (rows, pad) = out.split_at(t * QKV * 2);
    let finite = rows
        .chunks_exact(2)
        .all(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32().is_finite());
    ensure!(
        finite,
        "{what}: the base output is not finite; the gate would prove nothing"
    );
    ensure!(
        pad.iter().all(|&b| b == POISON),
        "{what}: a row past T was written"
    );
    Ok(())
}

/// 2026-10-15: Head rows of `a` and `b` (`[t, H, D]` BF16) that differ anywhere.
fn rows_differ(a: &[u8], b: &[u8]) -> usize {
    a.chunks(D * 2)
        .zip(b.chunks(D * 2))
        .filter(|(x, y)| x != y)
        .count()
}

/// 2026-10-15: Scan-then-norm and the fused arm of one combination on `base`'s buffers; the
/// fused arm's core is poisoned first and must come back untouched. Returns the base run and
/// its output.
#[allow(clippy::too_many_arguments)]
fn pair(
    c: &Ctx,
    base: &mut Arm,
    nm: Norm,
    layout: Layout,
    conv: bool,
    place: Place,
    s: u64,
    what: &str,
) -> Result<(Run, Vec<u8>)> {
    let (g, x, n) = (c.g, c.x, out_bytes(c.x.t));
    base.norm = None;
    let a: &Arm = base;
    let tree = run_arm(g, s, x, a, &mut |rec| {
        c.path_on(c.kb, a, layout, conv, place, s, rec)
    })?;
    g.memset(nm.out, POISON, n)?;
    let want = run_arm(g, s, x, a, &mut |rec| {
        c.path(a, layout, conv, place, s, rec)
    })?;
    compare(&format!("{what} lever off vs base tree"), &tree, &want)?;
    c.norm(a.core, &nm, 128, s)?;
    let want_out = read(g, s, nm.out, n)?;
    ensure_finite(what, &want)?;
    ensure_out(what, &want_out, x.t)?;

    base.norm = Some(nm);
    g.memset(nm.out, POISON, n)?;
    let a: &Arm = base;
    let mut got = run_arm(g, s, x, a, &mut |rec| {
        g.memset(a.core, POISON, x.t * QKV * 4)?;
        c.path(a, layout, conv, place, s, rec)
    })?;
    base.norm = None;
    ensure!(
        got.core.iter().all(|&b| b == POISON),
        "{what}: the fused arm wrote the core rows"
    );
    got.core.clone_from(&want.core);
    compare(what, &want, &got)?;
    let got_out = read(g, s, nm.out, n)?;
    same(&format!("{what} normed rows"), &want_out, &got_out)?;
    Ok((want, want_out))
}

/// 2026-10-15: The two negative controls on the base of the last combination.
#[allow(clippy::too_many_arguments)]
fn controls(
    c: &Ctx,
    a: &mut Arm,
    nm: Norm,
    ctl_out: DevicePtr,
    want: &Run,
    want_out: &[u8],
    s: u64,
    what: &str,
) -> Result<()> {
    let (g, t) = (c.g, c.x.t);
    // 2026-10-15: (a): the norm at 64 threads per row (another partial order) on the base's
    // core rows must differ.
    let alt = Norm { out: ctl_out, ..nm };
    g.memset(ctl_out, POISON, out_bytes(t))?;
    c.norm(a.core, &alt, 64, s)?;
    let order = rows_differ(want_out, &read(g, s, ctl_out, out_bytes(t))?);
    ensure!(
        order > 0,
        "{what}: CONTROL (a) not caught: the 64-thread norm is byte-identical"
    );
    // 2026-10-15: (b): one output-gate element moved 4 across zero in the fused arm: its
    // output moves, its state does not.
    let at = ((t - 2) * QKV + 5 * D + 3) * 2;
    let bad = Norm {
        gate: bumped_copy(g, nm.gate, t * QKV * 2, at)?,
        out: ctl_out,
        ..nm
    };
    a.norm = Some(bad);
    g.memset(ctl_out, POISON, out_bytes(t))?;
    let b: &Arm = a;
    let ctl = run_arm(g, s, c.x, b, &mut |rec| {
        c.path(b, Layout::Packed, true, Place::Prepare, s, rec)
    })?;
    a.norm = None;
    let moved = rows_differ(want_out, &read(g, s, ctl_out, out_bytes(t))?);
    g.free(bad.gate)?;
    let state_equal = ctl.rec == want.rec;
    ensure!(
        moved > 0 && state_equal,
        "{what}: CONTROL (b) not caught: {moved} rows moved, state equal: {state_equal}"
    );
    println!(
        "{what}: NEG CONTROL (a) 64-thread norm: {order} of {} head rows differ - caught; (b) one \
         gate element: {moved} head row(s) differ, state equal - caught",
        t * H
    );
    Ok(())
}

/// 2026-10-15: The output gate rows `[t, QKV]` and the norm weight `[D]` around 1.
fn norm_inputs(g: &dyn GpuBackend, rng: &mut Lcg, t: usize) -> Result<Norm> {
    let mut og = rng.vec(t * QKV, 3.0);
    for (i, v) in og.iter_mut().enumerate() {
        if i % 1013 == 0 {
            *v = if i % 2 == 0 { 40.0 } else { -40.0 };
        }
    }
    let w: Vec<f32> = rng.vec(D, 0.5).iter().map(|u| u + 1.0).collect();
    Ok(Norm {
        gate: up_bf16(g, &og)?,
        weight: up_bf16(g, &w)?,
        out: g.alloc(out_bytes(t))?,
        eps: EPS,
    })
}

/// 2026-10-15: Every combination of one row count, its negative controls and its timings.
#[allow(clippy::too_many_arguments)]
fn case(
    g: &dyn GpuBackend,
    s: u64,
    k: &Kernels,
    kb: &Kernels,
    m: &More,
    rng: &mut Lcg,
    t: usize,
) -> Result<()> {
    let mut x = inputs(g, rng, t)?;
    let r = gate_inputs(g, rng, t)?;
    let nm = norm_inputs(g, rng, t)?;
    let ctl_out = g.alloc(out_bytes(t))?;
    let mut a = Arm::new(g, &x)?;
    let hit = x.rec0.clone();
    for (state, rec0) in [("prefix-hit", hit), ("fresh", vec![0.0; H * D * D])] {
        x.rec0 = rec0;
        let c = Ctx {
            g,
            k,
            kb,
            m,
            x: &x,
            r: &r,
        };
        let mut last = None;
        for layout in [Layout::Packed, Layout::Parts] {
            for conv in [false, true] {
                for place in [Place::Kernel, Place::Prepare] {
                    let what = format!("T={t} {state} {layout:?} conv={conv} gate={place:?}");
                    let (want, want_out) = pair(&c, &mut a, nm, layout, conv, place, s, &what)?;
                    println!(
                        "{what}: fused = scan then norm = base-tree scan then norm, byte for byte"
                    );
                    if layout == Layout::Packed && conv && place == Place::Prepare {
                        last = Some((want, want_out, what));
                    }
                }
            }
        }
        let (want, want_out, what) = last.context("the production combination did not run")?;
        // 2026-10-15: The base's core is the last combination's: rerun the production one.
        pair(
            &c,
            &mut a,
            nm,
            Layout::Packed,
            true,
            Place::Prepare,
            s,
            &what,
        )?;
        controls(&c, &mut a, nm, ctl_out, &want, &want_out, s, &what)?;
    }
    let c = Ctx {
        g,
        k,
        kb,
        m,
        x: &x,
        r: &r,
    };
    let copies = if t <= 300 { 8 } else { 2 };
    for (conv, place) in [(true, Place::Prepare), (false, Place::Kernel)] {
        let old = timing::time_graph(g, s, copies, &mut |s| {
            c.path(&a, Layout::Packed, conv, place, s, None)?;
            c.norm(a.core, &nm, 128, s)
        })?;
        a.norm = Some(nm);
        let new = timing::time_graph(g, s, copies, &mut |s| {
            c.path(&a, Layout::Packed, conv, place, s, None)
        })?;
        a.norm = None;
        timing::report(&format!("path conv={conv} {place:?}"), t, old, new);
    }
    let old = timing::time_graph(g, s, copies, &mut |s| {
        c.scans(&a, s)?;
        c.norm(a.core, &nm, 128, s)
    })?;
    a.norm = Some(nm);
    let new = timing::time_graph(g, s, copies, &mut |s| c.scans(&a, s))?;
    a.norm = None;
    timing::report("scans+norm vs fused scans", t, old, new);
    let plain = timing::time_graph(g, s, copies, &mut |s| c.scans(&a, s))?;
    let norm = timing::time_graph(g, s, copies, &mut |s| c.norm(a.core, &nm, 128, s))?;
    timing::report("scans vs norm alone", t, plain, norm);
    a.free(g)?;
    for p in [
        r.g_raw, r.dt_bias, r.a_log, nm.gate, nm.weight, nm.out, ctl_out,
    ] {
        g.free(p)?;
    }
    free_inputs(g, x)
}

fn main() -> Result<()> {
    let sets = metrale_kernels::all_ptx_sets();
    let glm = sets
        .iter()
        .find(|s| s.target.model == "glm-5.3-flash")
        .context("glm-5.3-flash kernel target not built")?;
    let mut modules = glm.modules.clone();
    let embedded = modules
        .iter()
        .find(|m| m.0 == "kda_chunk64")
        .context("no kda_chunk64 module")?
        .1;
    if leak("GLM_PARITY_FIDELITY_PTX")? != embedded {
        bail!(
            "GLM_PARITY_FIDELITY_PTX differs from the embedded kda_chunk64: the base PTX was not \
             compiled as the build compiles"
        );
    }
    modules.push(("kda_chunk64_base", leak("GLM_PARITY_BASE_PTX")?));
    let gpu = MetraleCudaBackend::new(0, &modules)?;
    let g: &dyn GpuBackend = &gpu;
    let s = g.create_stream()?;
    let k = Kernels::load(g)?;
    let kb = Kernels {
        scan: g.kernel("kda_chunk64_base", "kda_chunk64_scan")?,
        ..Kernels::load(g)?
    };
    let m = More {
        gate: g.kernel("kda_gate", "kda_gate_bf16")?,
        prep_graw: g.kernel("kda_chunk64_prepare", "kda_chunk64_prepare_graw")?,
        prep_conv_graw: g.kernel("kda_chunk64_prepare", "kda_chunk64_prepare_conv_graw")?,
        norm: g.kernel("kda_layer_ops", "kda_o_norm_gated_bf16")?,
    };
    let clusters =
        g.max_active_clusters(k.scan_onorm, SCAN_GRID, SCAN_BLOCK, CHUNK64_SCAN_ONORM_SMEM)?;
    ensure!(
        clusters > 0,
        "the device places no cluster of kda_chunk64_scan_onorm"
    );
    println!(
        "kda_chunk64_scan_onorm: {clusters} co-resident (1, 2, 1) clusters at grid {SCAN_GRID:?}, \
         {CHUNK64_SCAN_ONORM_SMEM} B shared"
    );
    let mut rng = Lcg(0x6f6e_6f72_6d5f_7363);
    for t in case_rows()? {
        case(g, s, &k, &kb, &m, &mut rng, t)?;
    }
    println!(
        "PASS: the norm in the chunk64 scan is byte-identical to the scan then the norm, and the \
         lever-off scan to the base tree's"
    );
    Ok(())
}
