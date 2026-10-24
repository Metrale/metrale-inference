// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: GPU time of a launch sequence measured the way the serve runs it: captured into
//! a CUDA graph and replayed. Shared by the GLM-5.3 row-batch microbenchmarks.
//!
//! Owner: model-arch examples.
//! Invariants:
//! - `f` is called `copies` times inside one capture; the figure is the replay time divided by
//!   `copies` and by the replays per sample. Long graphs keep the measurement about the kernels
//!   when another process shares the GPU (its time slices then stretch both arms alike).
#![allow(dead_code)]

use anyhow::Result;
use metrale_gpu_runtime::gpu::GpuBackend;

/// 2026-10-09: Replays per sample and samples per figure.
pub const ITERS: usize = 5;
pub const REPS: usize = 7;

/// 2026-10-09: Microseconds per call of `f` on stream `s`: (median, min, max) over `REPS`
/// samples.
pub fn time_graph(
    g: &dyn GpuBackend,
    s: u64,
    copies: usize,
    f: &mut dyn FnMut(u64) -> Result<()>,
) -> Result<(f64, f64, f64)> {
    g.begin_capture(s)?;
    for _ in 0..copies {
        f(s)?;
    }
    let graph = g.end_capture(s)?;
    for _ in 0..3 {
        g.launch_graph(graph, s)?;
    }
    g.synchronize(s)?;
    let mut t = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let t0 = std::time::Instant::now();
        for _ in 0..ITERS {
            g.launch_graph(graph, s)?;
        }
        g.synchronize(s)?;
        t.push(t0.elapsed().as_secs_f64() * 1e6 / (ITERS * copies) as f64);
    }
    g.destroy_graph(graph)?;
    t.sort_by(f64::total_cmp);
    Ok((t[REPS / 2], t[0], t[REPS - 1]))
}

/// 2026-10-09: One line: old and new (median [min..max]) and the ratio of the medians.
pub fn report(what: &str, rows: usize, old: (f64, f64, f64), new: (f64, f64, f64)) {
    println!(
        "{what:<28} R={rows:>2}: old {:7.1} us [{:.1}..{:.1}]  new {:7.1} us [{:.1}..{:.1}]  x{:.2}",
        old.0,
        old.1,
        old.2,
        new.0,
        new.1,
        new.2,
        old.0 / new.0
    );
}
