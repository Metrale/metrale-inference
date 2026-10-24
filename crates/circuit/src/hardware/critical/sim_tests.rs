// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The contention simulation: occupancy, the solo-work calibration identity, SM
//! residency and shared DRAM, on hand-built captures (times stated in [`super::fixture`]).

use super::capture::Capture;
use super::dag::build;
use super::fixture::{OpRow, capture, diamond};
use super::graph::Graph;
use super::maxplus::{fit_latency, longest, step_us, weights, work_in_context};
use super::sim::{Resources, SmLimits, calibrate_dram, per_sm, simulate, solo_work};

/// 2026-10-23: GB10 (DEVICES.toml: 48 SMs, 256 KB registers, 1536 threads, 100 KB shared
/// memory); 32 resident CTAs is the `--max-ctas-per-sm` the serve report passes.
fn gb10(dram: f64) -> Resources {
    Resources {
        sms: 48.0,
        sm: SmLimits {
            regs: 65_536,
            threads: 1_536,
            smem: 102_400,
            ctas: 32,
        },
        dram_bytes_per_us: dram,
        dram_peak_bytes_per_us: 273_000.0,
    }
}

fn named(c: &Capture, name: &str) -> Vec<usize> {
    (0..c.ops.len())
        .filter(|&i| c.ops[i].name() == name && c.ops[i].start() >= 0)
        .collect()
}

fn sim_step(g: &Graph, solo: &[f64], bytes: &[f64], r: &Resources) -> f64 {
    let w = weights(g, &work_in_context(g), Some(&fit_latency(g)));
    step_us(g, &simulate(g, &w, solo, bytes, &[], r))
}

/// 2026-10-23: Per step `step` [0,1] on s1, then `p` [2,12] on s1 and `q` [2,12] forked on s2
/// (4 CTAs each, no SM contention), the next step at 13 after both.
fn pair(steps: i64) -> Capture {
    let mut ops: Vec<OpRow> = Vec::new();
    for k in -1..=steps {
        let b = k * 13_000;
        let (c, s) = (
            u64::try_from(101 + k).unwrap(),
            10 * u64::try_from(k + 1).unwrap(),
        );
        ops.push(("k", "step", s + 1, b, b + 1_000, c, true, 1, 32, 16, 0));
        if k < steps {
            ops.push((
                "k",
                "p",
                s + 1,
                b + 2_000,
                b + 12_000,
                c,
                true,
                4,
                256,
                32,
                0,
            ));
            ops.push((
                "k",
                "q",
                s + 2,
                b + 2_000,
                b + 12_000,
                c,
                true,
                4,
                256,
                32,
                0,
            ));
        }
    }
    capture(ops, Vec::new(), Vec::new(), Vec::new())
}

#[test]
fn occupancy_counts_registers_threads_shared_memory_and_ctas() {
    let l = gb10(0.0).sm;
    assert_eq!(
        per_sm(256, 64, 0, &l),
        4,
        "8 warps x 2048 registers = 16384 of 65536"
    );
    assert_eq!(
        per_sm(256, 32, 0, &l),
        6,
        "registers allow 8, 1536 threads allow 6"
    );
    assert_eq!(
        per_sm(256, 65, 0, &l),
        3,
        "65 x 32 = 2080 registers per warp, allocated as 2304"
    );
    assert_eq!(per_sm(128, 255, 0, &l), 2, "4 warps x 8192 registers");
    assert_eq!(
        per_sm(32, 16, 50_000, &l),
        2,
        "shared memory plus the 1 KB reservation"
    );
    assert_eq!(per_sm(32, 16, 0, &l), 32, "the CTA limit");
    assert_eq!(per_sm(1024, 255, 0, &l), 1, "never below one");
}

#[test]
fn the_simulation_reproduces_its_own_calibration_schedule() {
    let c = diamond(4, 11_000);
    let g = build(&c, 1_000, 3_000).unwrap();
    let r = gb10(273_000.0);
    let bytes = vec![0.0; g.nodes.len()];
    let solo = solo_work(&g, &work_in_context(&g), &bytes, &r);
    for i in named(&c, "b1") {
        assert!(
            (solo[i] - 8.5).abs() < 1e-9,
            "b1 got no SM until a1 (all 48) drained: {}",
            solo[i]
        );
    }
    for i in named(&c, "a1") {
        assert!((solo[i] - 10.0).abs() < 1e-9);
    }
    assert!(
        (sim_step(&g, &solo, &bytes, &r) - 23.0).abs() < 1e-9,
        "{}",
        sim_step(&g, &solo, &bytes, &r)
    );
}

#[test]
fn an_off_path_kernel_that_holds_the_sms_delays_the_critical_one() {
    let c = diamond(4, 11_000);
    let g = build(&c, 1_000, 3_000).unwrap();
    let r = gb10(273_000.0);
    let bytes = vec![0.0; g.nodes.len()];
    let work = work_in_context(&g);
    let mut solo = solo_work(&g, &work, &bytes, &r);
    let mut cut = work.clone();
    for i in named(&c, "a1") {
        solo[i] = 2.0;
        cut[i] = 2.0;
    }
    let lat = fit_latency(&g);
    let maxplus = step_us(&g, &longest(&g, &weights(&g, &cut, Some(&lat))));
    assert!(
        (maxplus - 23.0).abs() < 1e-9,
        "max-plus: a1 is off the chain, subgradient 0"
    );
    let sim = sim_step(&g, &solo, &bytes, &r);
    assert!(
        (sim - 15.0).abs() < 1e-9,
        "b1 starts when a1 drains at 3.5, ends 12, a3 13..14, step 15: {sim}"
    );
}

#[test]
fn kernels_sharing_dram_slow_each_other() {
    let c = pair(4);
    let g = build(&c, 1_000, 3_000).unwrap();
    let mut bytes = vec![0.0; g.nodes.len()];
    for i in named(&c, "p").into_iter().chain(named(&c, "q")) {
        bytes[i] = 1_000.0;
    }
    let work = work_in_context(&g);
    let (d, same) = calibrate_dram(&g, &work, &bytes, 273_000.0, 0);
    assert_eq!(d.overlapped_intervals, 4);
    assert!(
        (d.bytes_per_us - 200.0).abs() < 1e-9,
        "p and q at 100 B/us each: {}",
        d.bytes_per_us
    );
    assert_eq!(
        (d.scaled, &same),
        (0, &bytes),
        "the schedule fits: nothing reduced"
    );
    let (capped, fit) = calibrate_dram(&g, &work, &bytes, 150.0, 0);
    assert!(
        (capped.bytes_per_us - 150.0).abs() < 1e-9,
        "never above the peak"
    );
    assert_eq!(
        capped.scaled, 8,
        "p and q of every step read at most 150 B/us together"
    );
    for i in named(&c, "p") {
        assert!((fit[i] - 750.0).abs() < 1e-9, "{}", fit[i]);
    }
    let bus = gb10(150.0);
    let solo = solo_work(&g, &work, &fit, &bus);
    assert!(
        (sim_step(&g, &solo, &fit, &bus) - 13.0).abs() < 1e-9,
        "the reduced bytes reproduce the capture"
    );
    assert!(
        (sim_step(&g, &solo, &fit, &gb10(100.0)) - 18.0).abs() < 1e-9,
        "150 B/us of demand on a 100 B/us bus: 15 us each"
    );
    let fast = gb10(d.bytes_per_us);
    let solo = solo_work(&g, &work, &bytes, &fast);
    assert!((sim_step(&g, &solo, &bytes, &fast) - 13.0).abs() < 1e-9);
    let half = gb10(100.0);
    let s = sim_step(&g, &solo, &bytes, &half);
    assert!(
        (s - 23.0).abs() < 1e-9,
        "half the shared bandwidth: both at rate 1/2, 20 us each: {s}"
    );
    let none = vec![0.0; g.nodes.len()];
    assert!(
        (sim_step(&g, &solo, &none, &half) - 13.0).abs() < 1e-9,
        "kernels that read nothing do not share it"
    );
}
