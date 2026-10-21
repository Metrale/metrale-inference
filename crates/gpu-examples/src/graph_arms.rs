// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: Interleaved, clock-warmed timing of captured CUDA graphs for the microbenches.
//! GB10's SM clock ramps for hundreds of milliseconds after an idle spell (input upload, a host
//! compare), so the first arm timed after one runs several times slower than the same graph a
//! second later (measured: one kernel 24.1 us first, 6.3 us later in the same process). Each arm
//! here is captured once; every round first replays the arms until `warm` has elapsed, then times
//! `trials` replays of each arm in turn; the result per arm is the median over rounds of each
//! round's median.
//!
//! Owner: gpu-examples.
//! Invariants: every arm is timed in every round, in the same order, after the same warm-up.

use anyhow::Result;
use metrale_gpu_runtime::gpu::GpuBackend;
use std::time::{Duration, Instant};

/// 2026-10-21: One arm's per-call microseconds: (median, min, max) over rounds.
pub type ArmUs = (f64, f64, f64);

/// 2026-10-21: How the arms are timed.
#[derive(Clone, Copy, Debug)]
pub struct Plan {
    /// 2026-10-21: Calls captured into one arm's graph (the per-call time divides by it).
    pub calls: usize,
    pub rounds: usize,
    pub trials: usize,
    pub warm: Duration,
}

/// 2026-10-21: Capture each arm (`calls` calls of `arm(stream, i)`), then time them per [`Plan`].
///
/// # Errors
/// A capture, launch or synchronize failure, or an arm's own error.
pub fn time_arms(
    g: &dyn GpuBackend,
    stream: u64,
    plan: Plan,
    arms: &[&dyn Fn(u64, usize) -> Result<()>],
) -> Result<Vec<ArmUs>> {
    let mut graphs = Vec::with_capacity(arms.len());
    for arm in arms {
        g.begin_capture(stream)?;
        for i in 0..plan.calls {
            if let Err(e) = arm(stream, i) {
                g.abort_capture_if_active(stream);
                return Err(e);
            }
        }
        graphs.push(g.end_capture(stream)?);
    }
    let mut per_round: Vec<Vec<f64>> = vec![Vec::with_capacity(plan.rounds); arms.len()];
    for _ in 0..plan.rounds {
        let t0 = Instant::now();
        while t0.elapsed() < plan.warm {
            for &gr in &graphs {
                g.launch_graph(gr, stream)?;
            }
            g.synchronize(stream)?;
        }
        for (a, &gr) in graphs.iter().enumerate() {
            let mut v = Vec::with_capacity(plan.trials);
            for _ in 0..plan.trials {
                let t = Instant::now();
                g.launch_graph(gr, stream)?;
                g.synchronize(stream)?;
                v.push(t.elapsed().as_secs_f64() * 1e6 / plan.calls as f64);
            }
            v.sort_by(f64::total_cmp);
            per_round[a].push(v[v.len() / 2]);
        }
    }
    for gr in graphs {
        g.destroy_graph(gr)?;
    }
    Ok(per_round
        .into_iter()
        .map(|mut v| {
            v.sort_by(f64::total_cmp);
            (v[v.len() / 2], v[0], v[v.len() - 1])
        })
        .collect())
}
