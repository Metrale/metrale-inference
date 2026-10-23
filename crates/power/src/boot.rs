// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The hierarchical bootstrap of a steady-state row (owner addendum 2026-10-22: never
//! an i.i.d. SE on autocorrelated power). One replicate:
//!
//! 1. per box, draw that box's repeats with replacement (the run level);
//! 2. per drawn repeat, resample the load window and both idle windows with the stationary
//!    bootstrap (Politis & Romano 1994) over their (energy, time) update pairs, so the
//!    autocorrelation inside a window survives the resample;
//! 3. recompute `W`, `J_total`, `J_dyn` per repeat and average them.
//!
//! The BCa interval (Efron 1987) takes its acceleration from the delete-one-repeat jackknife; a
//! single repeat falls back to the delete-one-block jackknife of its load window.
//!
//! Owner: metrale-power.
//! Invariants: the same inputs and seed give the same intervals on every target.

use alloc::vec::Vec;

use metrale_stats::bootstrap::{bca_interval, block_length, stationary_indices};
use metrale_stats::interval::Interval;
use metrale_stats::rng::SplitMix64;

use crate::Result;
use crate::integrate::WindowEnergy;

/// 2026-10-22: One repeat's resampling units.
#[derive(Clone, Debug)]
pub struct RepeatSeries<'a> {
    pub box_id: u32,
    pub load: &'a WindowEnergy,
    pub idle_a: &'a WindowEnergy,
    pub idle_b: &'a WindowEnergy,
    pub launches: f64,
}

/// 2026-10-22: The three quantities of a repeat: (W, J_total, J_dyn).
fn quantities(w_load: f64, w_static: f64, seconds: f64, launches: f64) -> [f64; 3] {
    [
        w_load,
        w_load * seconds / launches,
        (w_load - w_static) * seconds / launches,
    ]
}

fn ratio_mean(w: &WindowEnergy, idx: Option<&[usize]>) -> f64 {
    let (mut e, mut t) = (0.0, 0.0);
    let mut add = |i: usize| {
        e += w.series_w[i] * w.series_dt[i];
        t += w.series_dt[i];
    };
    match idx {
        Some(ix) => ix.iter().for_each(|&i| add(i)),
        None => (0..w.series_w.len()).for_each(&mut add),
    }
    if t > 0.0 { e / t } else { w.avg_w }
}

fn point(r: &RepeatSeries<'_>) -> [f64; 3] {
    let s = (ratio_mean(r.idle_a, None) + ratio_mean(r.idle_b, None)) / 2.0;
    quantities(ratio_mean(r.load, None), s, r.load.seconds, r.launches)
}

fn resampled(w: &WindowEnergy, rng: &mut SplitMix64) -> f64 {
    let n = w.series_w.len();
    if n == 0 {
        return w.avg_w;
    }
    let block = block_length(n) as f64;
    let ix = stationary_indices(n, block, rng);
    ratio_mean(w, Some(&ix))
}

fn mean3(v: &[[f64; 3]]) -> [f64; 3] {
    let n = v.len().max(1) as f64;
    let mut m = [0.0; 3];
    for x in v {
        for k in 0..3 {
            m[k] += x[k] / n;
        }
    }
    m
}

/// 2026-10-22: BCa intervals at `level` for (W, J_total, J_dyn) over `reps` replicates.
pub fn bca(
    repeats: &[RepeatSeries<'_>],
    reps: usize,
    level: f64,
    seed: u64,
) -> Result<[Interval; 3]> {
    let mut rng = SplitMix64::new(seed);
    let points: Vec<[f64; 3]> = repeats.iter().map(point).collect();
    let theta = mean3(&points);
    let mut boxes: Vec<u32> = repeats.iter().map(|r| r.box_id).collect();
    boxes.sort_unstable();
    boxes.dedup();
    let mut boot: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for _ in 0..reps {
        let mut draw = Vec::with_capacity(repeats.len());
        for b in &boxes {
            let members: Vec<&RepeatSeries<'_>> =
                repeats.iter().filter(|r| r.box_id == *b).collect();
            for _ in 0..members.len() {
                let r = members[rng.below(members.len())];
                let s = (resampled(r.idle_a, &mut rng) + resampled(r.idle_b, &mut rng)) / 2.0;
                draw.push(quantities(
                    resampled(r.load, &mut rng),
                    s,
                    r.load.seconds,
                    r.launches,
                ));
            }
        }
        let m = mean3(&draw);
        for k in 0..3 {
            boot[k].push(m[k]);
        }
    }
    let jack = jackknife(repeats, &points);
    let one = |k: usize| -> Result<Interval> {
        let j: Vec<f64> = jack.iter().map(|x| x[k]).collect();
        bca_interval(theta[k], &boot[k], &j, level)
    };
    Ok([one(0)?, one(1)?, one(2)?])
}

/// 2026-10-22: Delete-one-repeat means; with one repeat, delete-one-block of its load window.
fn jackknife(repeats: &[RepeatSeries<'_>], points: &[[f64; 3]]) -> Vec<[f64; 3]> {
    if repeats.len() >= 2 {
        return (0..points.len())
            .map(|i| {
                let rest: Vec<[f64; 3]> = points
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(_, p)| *p)
                    .collect();
                mean3(&rest)
            })
            .collect();
    }
    let Some(r) = repeats.first() else {
        return Vec::new();
    };
    let n = r.load.series_w.len();
    let block = block_length(n).max(1);
    let s = (ratio_mean(r.idle_a, None) + ratio_mean(r.idle_b, None)) / 2.0;
    (0..n.div_ceil(block))
        .map(|b| {
            let keep: Vec<usize> = (0..n).filter(|i| i / block != b).collect();
            quantities(
                ratio_mean(r.load, Some(&keep)),
                s,
                r.load.seconds,
                r.launches,
            )
        })
        .collect()
}
