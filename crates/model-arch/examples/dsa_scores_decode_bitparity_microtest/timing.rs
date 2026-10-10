// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: The TIMING lines of `dsa_scores_decode_bitparity_microtest`: per live context, a
//! captured graph of one decode step's scores launches (`LAYERS` layers, each its own key
//! buffer, x `ROWS` verify rows), CUDA events around each replay, median of `REPLAYS` replays
//! after 3 warm-ups, for the plain and the decode kernel. Timing never decides the verdict.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.

use anyhow::{Result, bail};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::{Fixture, Kernels, LAYERS, MAX_POOLS, RAGGED, ROWS};

const TIMING_S: &[usize] = &[131_072, 262_144, RAGGED];
const REPLAYS: usize = 21;

// 2026-10-08: CUDA driver event API for kernel-only timing, declared as in `moe_microtest`.
unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

fn check(rc: i32, what: &str) -> Result<()> {
    if rc != 0 {
        bail!("{what} failed: status {rc}");
    }
    Ok(())
}

/// 2026-10-08: Median ms of one replay of a graph of the 33 launches of a decode step
/// (call i: layer i / 3, verify row i % 3), CUDA events around each replay.
fn time_step(
    g: &dyn GpuBackend,
    ks: &Kernels,
    f: &Fixture,
    decode: bool,
    st: u64,
    out: DevicePtr,
    vc: DevicePtr,
) -> Result<f64> {
    g.begin_capture(st)?;
    let body = (0..LAYERS * ROWS)
        .try_for_each(|i| f.launch(g, ks, decode, f.q, i / ROWS, i % ROWS, out, vc, st));
    if let Err(e) = body {
        g.abort_capture_if_active(st);
        return Err(e);
    }
    let graph = g.end_capture(st)?;
    for _ in 0..3 {
        g.launch_graph(graph, st)?;
    }
    g.synchronize(st)?;
    let mut ev = vec![(0u64, 0u64); REPLAYS];
    // SAFETY: plain CUDA driver event calls; the events are created, used and destroyed here,
    // and the backend made the context current when it was built.
    unsafe {
        for e in ev.iter_mut() {
            check(cuEventCreate(&mut e.0, 0), "cuEventCreate")?;
            check(cuEventCreate(&mut e.1, 0), "cuEventCreate")?;
        }
    }
    for e in &ev {
        // SAFETY: as above.
        unsafe { check(cuEventRecord(e.0, st), "cuEventRecord(start)")? };
        g.launch_graph(graph, st)?;
        // SAFETY: as above.
        unsafe { check(cuEventRecord(e.1, st), "cuEventRecord(end)")? };
    }
    g.synchronize(st)?;
    let mut ms = Vec::with_capacity(REPLAYS);
    for e in &ev {
        let mut t: f32 = 0.0;
        // SAFETY: as above; `t` outlives the call and both events have completed.
        unsafe {
            check(cuEventElapsedTime(&mut t, e.0, e.1), "cuEventElapsedTime")?;
            cuEventDestroy_v2(e.0);
            cuEventDestroy_v2(e.1);
        }
        ms.push(f64::from(t));
    }
    g.destroy_graph(graph)?;
    ms.sort_by(|x, y| x.total_cmp(y));
    Ok(ms[ms.len() / 2])
}

pub(super) fn timing(g: &dyn GpuBackend, ks: &Kernels, f: &Fixture) -> Result<()> {
    let st = g.create_stream()?;
    let (out, vc) = (g.alloc(MAX_POOLS * 4)?, g.alloc(MAX_POOLS)?);
    for &s in TIMING_S {
        f.set_context(g, ks, s)?;
        let plain = time_step(g, ks, f, false, st, out, vc)?;
        let dec = time_step(g, ks, f, true, st, out, vc)?;
        println!(
            "TIMING scores_plain S={s} ms_per_step={plain:.3} grid={} ({LAYERS} layers x {ROWS} \
             rows)",
            f.grids.0
        );
        println!(
            "TIMING scores_decode S={s} ms_per_step={dec:.3} grid={} speedup={:.2}x",
            f.grids.1,
            plain / dec
        );
    }
    g.free(out).ok();
    g.free(vc).ok();
    Ok(())
}
