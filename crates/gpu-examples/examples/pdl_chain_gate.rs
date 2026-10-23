// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-21: Race detector for programmatic dependent launch (`METRALE_GLM_PDL`) on the GLM-5.3
//! decode chain (`common/pdl_chain.rs`: `LAYERS` KDA + MoE layer pairs at M = 2). PDL changes only
//! timing, so a read hoisted above a kernel's `griddepcontrol.wait` is a race, not a numerics
//! change: this gate replays the captured chain many times on a backend with programmatic launches
//! and byte-compares every replay against a serialised reference.
//!
//! 1. Reference: the same chain captured on a backend without the mode, on `SETS` inputs (mHC
//!    streams and KDA state), each run twice (it must be deterministic).
//! 2. Control: `REPLAYS` serialised replays, cycling the inputs, against the reference.
//! 3. PDL: `REPLAYS` programmatic replays and `EAGER` eager runs, cycling the inputs. Successive
//!    runs take different inputs, so a kernel that read its predecessor's output early would read
//!    the previous run's bytes.
//!
//! Compared per run: the final mHC streams, every layer's KDA conv and recurrent state and their
//! snapshots, and the last layer's router logits, expert ids, routed and shared expert outputs.
//!
//! Sabotage (the gate's negative control): move a dependent read above the wait in one kernel,
//! rebuild, and the PDL arm must fail (see the report). Owner: gpu-examples.
//! Invariants: exits with an error on the first difference.
//!
//!   cargo run -p metrale-gpu-examples --release --example pdl_chain_gate --features cuda,gpu-examples

#[path = "common/pdl_chain_bufs.rs"]
mod bufs;
#[path = "common/pdl_chain.rs"]
mod chain;

use anyhow::{Context, Result, bail, ensure};
use bufs::*;
use chain::{Kern, Variant, kda_layer, moe_layer};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle};

const LAYERS: usize = 2;
const SETS: usize = 3;
const REPLAYS: usize = 300;
const EAGER: usize = 30;

struct Inputs {
    streams: Vec<Vec<u8>>,
    conv: Vec<DevicePtr>,
    rec: Vec<DevicePtr>,
}

fn inputs(g: &dyn GpuBackend, b: &Bufs) -> Result<Inputs> {
    let mut r = Rng(0x9e37_79b9_7f4a_7c15);
    let streams = (0..SETS)
        .map(|_| {
            f32s(
                &(0..STREAM_BYTES / 4)
                    .map(|_| r.unit() * 2.0)
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    let mut conv = Vec::new();
    let mut rec = Vec::new();
    for l in &b.layers {
        let c = g.alloc(CONV_BYTES)?;
        g.copy_d2d(l.conv, c, CONV_BYTES)?;
        conv.push(c);
        let h = g.alloc(REC_BYTES)?;
        g.copy_d2d(l.rec, h, REC_BYTES)?;
        rec.push(h);
    }
    Ok(Inputs { streams, conv, rec })
}

fn reset(g: &dyn GpuBackend, b: &Bufs, i: &Inputs, set: usize) -> Result<()> {
    g.copy_h2d(&i.streams[set], b.s.streams)?;
    // 2026-10-21: Poison what a run must write in full (the conv snapshots, and `expert_out`,
    // whose remote slots only the zeroing writes), so a byte left unwritten shows.
    g.memset(b.s.expert_out, 0x5A, (SLOTS * H * 2) as usize)?;
    for (n, l) in b.layers.iter().enumerate() {
        g.copy_d2d(i.conv[n], l.conv, CONV_BYTES)?;
        g.copy_d2d(i.rec[n], l.rec, REC_BYTES)?;
        g.memset(l.conv_int, 0x5A, CONV_BYTES)?;
    }
    Ok(())
}

/// 2026-10-21: Make every third expert's down projection another rank's (a null packed pointer,
/// as an EP rank's table holds): the down sweep writes nothing for its slots, so they read only
/// the zeroing of `expert_out`.
fn remote_down_experts(g: &dyn GpuBackend, b: &Bufs) -> Result<()> {
    let mut p = vec![0u8; EXPERTS as usize * 8];
    g.copy_d2h(b.down_t.packed, &mut p)?;
    for id in (0..EXPERTS as usize).step_by(3) {
        p[id * 8..id * 8 + 8].fill(0);
    }
    g.copy_h2d(&p, b.down_t.packed)?;
    Ok(())
}

/// 2026-10-21: (name, bytes) of everything a run is judged on.
fn outputs(g: &dyn GpuBackend, b: &Bufs, s: u64) -> Result<Vec<(String, Vec<u8>)>> {
    g.synchronize(s)?;
    let rd = |p: DevicePtr, n: usize| -> Result<Vec<u8>> {
        let mut v = vec![0u8; n];
        g.copy_d2h(p, &mut v)?;
        Ok(v)
    };
    let mut o = vec![("streams".to_string(), rd(b.s.streams, STREAM_BYTES)?)];
    for (n, l) in b.layers.iter().enumerate() {
        o.push((format!("L{n} conv state"), rd(l.conv, CONV_BYTES)?));
        o.push((format!("L{n} recurrent state"), rd(l.rec, REC_BYTES)?));
        o.push((format!("L{n} conv snapshot"), rd(l.conv_int, CONV_BYTES)?));
        o.push((format!("L{n} recurrent snapshot"), rd(l.h_int, REC_BYTES)?));
    }
    let x = &b.s;
    o.push((
        "router logits".into(),
        rd(x.logits, (T * EXPERTS * 4) as usize)?,
    ));
    o.push(("expert ids".into(), rd(x.ids, (SLOTS * 4) as usize)?));
    o.push((
        "routed out".into(),
        rd(x.expert_out, (SLOTS * H * 2) as usize)?,
    ));
    o.push(("shared out".into(), rd(x.shared_out, (T * H * 2) as usize)?));
    o.push((
        "KDA o_proj out".into(),
        rd(x.final_out, (T * H * 2) as usize)?,
    ));
    Ok(o)
}

fn enqueue(g: &dyn GpuBackend, k: &Kern, b: &Bufs, s: u64) -> Result<()> {
    for l in &b.layers {
        kda_layer(g, k, b, l, s)?;
        moe_layer(g, k, b, l, s)?;
    }
    Ok(())
}

fn capture(g: &dyn GpuBackend, k: &Kern, b: &Bufs, s: u64) -> Result<GraphHandle> {
    g.begin_capture(s)?;
    if let Err(e) = enqueue(g, k, b, s) {
        g.abort_capture_if_active(s);
        return Err(e);
    }
    g.end_capture(s)
}

/// 2026-10-21: The first output that differs from `want`, if any.
fn diff(got: &[(String, Vec<u8>)], want: &[(String, Vec<u8>)]) -> Option<String> {
    got.iter()
        .zip(want)
        .find(|(a, b)| a.1 != b.1)
        .map(|(a, b)| {
            let n = a.1.iter().zip(&b.1).filter(|(x, y)| x != y).count();
            format!("{} ({n} of {} bytes differ)", a.0, a.1.len())
        })
}

/// 2026-10-21: `runs` runs (graph replays when `graph`, else eager), inputs cycling; the number
/// that differ from the reference and the first difference.
fn arm(
    g: &dyn GpuBackend,
    k: &Kern,
    b: &Bufs,
    inp: &Inputs,
    want: &[Vec<(String, Vec<u8>)>],
    graph: Option<GraphHandle>,
    runs: usize,
    s: u64,
) -> Result<(usize, Option<String>)> {
    let (mut bad, mut first) = (0, None);
    for r in 0..runs {
        let set = r % SETS;
        reset(g, b, inp, set)?;
        match graph {
            Some(gr) => g.launch_graph(gr, s)?,
            None => enqueue(g, k, b, s)?,
        }
        if let Some(d) = diff(&outputs(g, b, s)?, &want[set]) {
            bad += 1;
            first.get_or_insert(format!("run {r} (input {set}): {d}"));
        }
    }
    Ok((bad, first))
}

fn main() -> Result<()> {
    let sets = metrale_kernels::all_ptx_sets();
    let glm = sets
        .iter()
        .find(|s| s.target.model == "glm-5.3-flash")
        .context("glm-5.3-flash kernel target not built")?;
    // 2026-10-21: The `_sl` sweep is launched stream-ordered only because its PTX body holds no
    // dependency wait (gpu-runtime ptx_pdl.rs); its plain twin must hold one.
    let ptx = glm
        .modules
        .iter()
        .find(|(n, _)| *n == "w4a4_gemv_mx_moe")
        .map(|(_, b)| String::from_utf8_lossy(b).into_owned())
        .context("w4a4_gemv_mx_moe PTX")?;
    for (entry, waits) in [
        ("w4a4_gemv_mx8_moe_union_sweep", true),
        ("w4a4_gemv_mx8_moe_union_sweep_sl", false),
    ] {
        let body = ptx
            .split(".entry ")
            .find(|b| b.starts_with(&format!("{entry}(")))
            .with_context(|| format!("{entry}: no .entry"))?;
        let body = body.split(".func ").next().unwrap_or(body);
        ensure!(
            body.contains("griddepcontrol.wait") == waits
                && body.contains("griddepcontrol.launch_dependents"),
            "RED: {entry}: griddepcontrol.wait {} (want {waits}), launch_dependents {}",
            body.contains("griddepcontrol.wait"),
            body.contains("griddepcontrol.launch_dependents")
        );
    }
    println!("  PTX: the plain sweep waits, the _sl sweep does not; both trigger");
    let off = MetraleCudaBackend::new(0, &glm.modules)?;
    let on = MetraleCudaBackend::new(0, &glm.modules)?;
    // 2026-10-21: The reference runs the old kernels; the new arms the levers `Variant` names.
    let v = Variant::from_env();
    let (ko, kv, kn) = (
        Kern::resolve(&off, Variant::OLD)?,
        Kern::resolve(&off, v)?,
        Kern::resolve(&on, v)?,
    );
    if !on.enable_programmatic_launches()? {
        bail!("programmatic launches unavailable");
    }
    let s = off.default_stream();
    println!(
        "GATE: PDL race detector, {LAYERS} KDA + MoE layer pairs at M = {T}, {SETS} inputs; new arms: {v:?}"
    );
    let b = Bufs::new(&off, 0x5eed_0001, LAYERS, 0.05)?;
    remote_down_experts(&off, &b)?;
    println!("  every third expert's down projection is remote (its slots read the zeroing)");
    let inp = inputs(&off, &b)?;

    let g_off = capture(&off, &ko, &b, s)?;
    let mut want = Vec::new();
    for set in 0..SETS {
        reset(&off, &b, &inp, set)?;
        off.launch_graph(g_off, s)?;
        let a = outputs(&off, &b, s)?;
        reset(&off, &b, &inp, set)?;
        off.launch_graph(g_off, s)?;
        ensure!(
            diff(&outputs(&off, &b, s)?, &a).is_none(),
            "reference not deterministic on input {set}"
        );
        want.push(a);
    }
    ensure!(
        diff(&want[1], &want[0]).is_some_and(|d| d.starts_with("streams")),
        "the inputs must change the outputs (else a stale read is invisible)"
    );
    println!("  reference: {SETS} inputs, each deterministic and different from the others");

    let (bad, first) = arm(&off, &ko, &b, &inp, &want, Some(g_off), REPLAYS, s)?;
    println!("  control  (serialised graph): {REPLAYS} replays, {bad} differ");
    ensure!(bad == 0, "control differs: {}", first.unwrap_or_default());

    let g_new = capture(&off, &kv, &b, s)?;
    let (bad, first) = arm(&off, &kv, &b, &inp, &want, Some(g_new), REPLAYS, s)?;
    println!("  new kernels (serialised graph): {REPLAYS} replays, {bad} differ");
    off.destroy_graph(g_new)?;
    ensure!(
        bad == 0,
        "RED: the new kernels changed the bytes: {}",
        first.unwrap_or_default()
    );

    let g_on = capture(&on, &kn, &b, s)?;
    let (bad_g, first_g) = arm(&on, &kn, &b, &inp, &want, Some(g_on), REPLAYS, s)?;
    println!(
        "  PDL graph:  {REPLAYS} replays, {bad_g} differ{}",
        first_g
            .as_deref()
            .map(|f| format!("; first: {f}"))
            .unwrap_or_default()
    );
    let (bad_e, first_e) = arm(&on, &kn, &b, &inp, &want, None, EAGER, s)?;
    println!(
        "  PDL eager:  {EAGER} runs, {bad_e} differ{}",
        first_e
            .as_deref()
            .map(|f| format!("; first: {f}"))
            .unwrap_or_default()
    );
    off.destroy_graph(g_off)?;
    on.destroy_graph(g_on)?;
    if bad_g + bad_e > 0 {
        bail!("RED: programmatic launches changed the bytes (a read ahead of griddepcontrol.wait)");
    }
    println!("PASS");
    Ok(())
}
