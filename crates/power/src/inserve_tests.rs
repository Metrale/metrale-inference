// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The in-serve NNLS validator recovers known per-kernel power within 3 % with
//! intervals that cover it, refuses collinear kernels, and refuses too few windows.

use alloc::vec;
use alloc::vec::Vec;

use metrale_error::ErrorCode;
use metrale_stats::rng::SplitMix64;

use crate::inserve::{MIN_WINDOWS, ServeWindow, attribute};

const TRUTH: [f64; 3] = [30.0, 15.0, 50.0];
const STATIC_W: f64 = 12.5;

fn windows(n: usize, collinear: bool, seed: u64) -> Vec<ServeWindow> {
    let mut r = SplitMix64::new(seed);
    let mut noise = 0.0;
    (0..n)
        .map(|_| {
            let a = 0.6 * r.next_f64();
            let b = if collinear {
                0.5 * a
            } else {
                0.3 * r.next_f64()
            };
            let c = 0.1 * r.next_f64();
            noise = 0.5 * noise + 0.3 * r.normal();
            let occupancy = vec![a, b, c];
            let power_w =
                STATIC_W + occupancy.iter().zip(TRUTH).map(|(f, p)| f * p).sum::<f64>() + noise;
            ServeWindow { power_w, occupancy }
        })
        .collect()
}

#[test]
fn known_per_kernel_power_is_recovered_within_three_percent() {
    let a = attribute(&windows(600, false, 3), 3, 400, 0.95, 9).unwrap();
    assert!(
        (a.static_w - STATIC_W).abs() / STATIC_W < 0.03,
        "{}",
        a.static_w
    );
    for (k, p) in TRUTH.iter().enumerate() {
        assert!(
            (a.kernel_w[k] - p).abs() / p < 0.03,
            "kernel {k}: {} vs {p}",
            a.kernel_w[k]
        );
    }
}

#[test]
fn the_coefficient_intervals_cover_the_truth_at_their_level() {
    // 2026-10-22: One dataset's interval may miss; over 40 datasets each kernel's 95 % interval
    // must cover its true power at least 85 % of the time (binomial 5 % floor at n = 40: 34/40).
    let mut covered = [0u32; 3];
    for seed in 0..40 {
        let a = attribute(&windows(400, false, 100 + seed), 3, 200, 0.95, seed).unwrap();
        for (k, p) in TRUTH.iter().enumerate() {
            covered[k] += u32::from(a.intervals[k + 1].contains(*p));
        }
    }
    assert!(covered.iter().all(|c| *c >= 34), "{covered:?} of 40");
}

#[test]
fn collinear_kernels_are_refused_not_split() {
    let e = attribute(&windows(600, true, 3), 3, 100, 0.95, 9).unwrap_err();
    assert_eq!(e.code(), ErrorCode::EnergyIllConditioned, "{e}");
}

#[test]
fn too_few_windows_or_a_bad_occupancy_is_refused() {
    let e = attribute(&windows(MIN_WINDOWS - 1, false, 3), 3, 100, 0.95, 9).unwrap_err();
    assert_eq!(e.code(), ErrorCode::EnergyWindowRefused);
    let mut w = windows(MIN_WINDOWS, false, 3);
    w[0].occupancy[1] = 1.5;
    assert_eq!(
        attribute(&w, 3, 100, 0.95, 9).unwrap_err().code(),
        ErrorCode::EnergyInputInvalid
    );
}
