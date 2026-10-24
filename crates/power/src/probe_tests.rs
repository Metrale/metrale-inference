// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The 2026-10-22 probe's raw trace replays to DESIGN §1.2's numbers. The probe kept
//! neither its windows nor its launch counts, so the windows are reconstructed once from the
//! counter (the update where the counter's power crosses between the idle and load levels:
//! loads start at +4.00, 20.09, 36.10, 52.20, 68.19, 84.19 s from the first sample, 12 s each
//! or until the next baseline starts (the last cut at the trace's end); each idle baseline is the 4 s before its load) and §1.2's
//! W columns are checked: counter W, sensor W and static W per repeat inside the table's ranges
//! (widened 1.5 %: see `within`).
//! J/launch needs the launch counts the probe did not store; it is not checked here.
//! Recomputation is bitwise: the replayed trace's text, its hash and every per-repeat value are
//! pinned (the HPAR anchor for the probe).

use alloc::vec::Vec;

use metrale_abstract_io::MonoNs;
use metrale_abstract_io::power::{Declared, Field, Rail};

use crate::integrate::channels;
use crate::policy::{REFUSAL_FLOOR, WindowPolicy};
use crate::probe::import;
use crate::steady::{ChannelRepeat, measure, repeats};
use crate::trace::{Phase, PowerTrace, Window, sha256_hex};

const CSV: &str = include_str!("../../../test_data/energy/kenergy_trace_2026_10_22.csv");
const LOAD_STARTS_S: [f64; 6] = [4.00, 20.09, 36.10, 52.20, 68.19, 84.19];

fn windows(t: &mut PowerTrace) {
    let t0 = t.sources[0].samples[0].at.0;
    let end = t.sources[0].samples.last().unwrap().at.0;
    let ns = |s: f64| t0 + (s * 1e9) as u64;
    for (r, s) in LOAD_STARTS_S.iter().enumerate() {
        let next_idle = LOAD_STARTS_S.get(r + 1).map_or(end, |n| ns(n - 4.0));
        let r = r as u32;
        t.windows.push(Window {
            phase: Phase::Idle,
            box_id: 0,
            repeat: r,
            from: MonoNs(ns(s - 4.0)),
            to: MonoNs(ns(*s)),
            launches: None,
        });
        t.windows.push(Window {
            phase: Phase::Load,
            box_id: 0,
            repeat: r,
            from: MonoNs(ns(*s)),
            to: MonoNs(ns(s + 12.0).min(next_idle)),
            launches: None,
        });
    }
}

/// 2026-10-22: The probe's policy: 4 s baselines (below §3.1's 10 s, above the 3 s floor).
const PROBE: WindowPolicy = WindowPolicy {
    min_idle_s: 3.5,
    ..REFUSAL_FLOOR
};

fn replay() -> (PowerTrace, Vec<Vec<ChannelRepeat>>) {
    let mut t = import(CSV).unwrap();
    windows(&mut t);
    let reps = repeats(&t);
    let per_channel = channels(&t)
        .into_iter()
        .map(|c| {
            reps.iter()
                .map(|r| measure(&t, &PROBE, c, r).unwrap())
                .collect()
        })
        .collect();
    (t, per_channel)
}

/// 2026-10-22: §1.2's range widened by 1.5 %: the probe divided the counter's change between the
/// first and last in-window samples by the whole window, which reads low by up to one 100 ms
/// update per 12 s (≈ 0.8 %); this crate takes the power over the span between updates.
fn within(v: f64, lo: f64, hi: f64) -> bool {
    v >= lo * 0.985 && v <= hi * 1.015
}

#[test]
fn the_probe_trace_replays_to_the_design_table() {
    let (_, ch) = replay();
    let (counter, sensor) = (&ch[0], &ch[1]);
    for r in 0..3 {
        assert!(
            within(counter[r].load.avg_w, 17.6, 17.9),
            "stream counter W {}",
            counter[r].load.avg_w
        );
        assert!(
            within(sensor[r].load.avg_w, 18.2, 18.6),
            "stream sensor W {}",
            sensor[r].load.avg_w
        );
        let a = counter[r].idle_a.as_ref().unwrap().avg_w;
        assert!(within(a, 12.2, 12.6), "stream static W {a}");
    }
    for r in 3..6 {
        assert!(
            within(counter[r].load.avg_w, 43.9, 44.7),
            "fma counter W {}",
            counter[r].load.avg_w
        );
        assert!(
            within(sensor[r].load.avg_w, 57.4, 58.3),
            "fma sensor W {}",
            sensor[r].load.avg_w
        );
        let a = counter[r].idle_a.as_ref().unwrap().avg_w;
        assert!(within(a, 12.7, 13.7), "fma static W {a}");
    }
    // 2026-10-22: §1.3: the sensor reads +4 % on the memory-bound loop, +31 % on the FMA loop.
    let delta = |r: usize| {
        (sensor[r].load.joules - counter[r].load.joules) / counter[r].load.joules * 100.0
    };
    assert!((2.0..7.0).contains(&delta(0)), "{}", delta(0));
    assert!((28.0..34.0).contains(&delta(4)), "{}", delta(4));
}

#[test]
fn the_probe_declares_what_gb10_does_not_report() {
    let t = import(CSV).unwrap();
    let c = &t.sources[0].caps;
    assert!(
        matches!(c.declares(Field::MemClock), Declared::Unavailable(r) if r.contains("not supported"))
    );
    assert!(matches!(
        t.sources[1]
            .caps
            .declares(Field::AveragePower(Rail::GpuCore)),
        Declared::Unavailable(_)
    ));
}

#[test]
fn the_last_fma_repeat_has_no_idle_b_so_its_dynamic_energy_is_refused() {
    let (_, ch) = replay();
    assert!(ch[0][5].idle_b.is_none());
    assert_eq!(
        ch[0][5].static_w().unwrap_err().code(),
        metrale_error::ErrorCode::EnergyBaselineMissing
    );
}

#[test]
fn the_probe_replay_is_bitwise_stable() {
    let (t, ch) = replay();
    let text = t.encode();
    assert_eq!(
        crate::trace::PowerTrace::decode(&text).unwrap().encode(),
        text
    );
    let bits: Vec<u64> = ch
        .iter()
        .flat_map(|c| c.iter().map(|r| r.load.avg_w.to_bits()))
        .collect();
    let again: Vec<u64> = replay()
        .1
        .iter()
        .flat_map(|c| c.iter().map(|r| r.load.avg_w.to_bits()))
        .collect();
    assert_eq!(bits, again);
    assert_eq!(sha256_hex(text.as_bytes()), PINNED_TRACE_SHA256);
    assert_eq!(bits, PINNED_LOAD_W_BITS);
}

/// 2026-10-22: The replayed probe trace's hash and per-repeat load W (counter repeats 0..6, then
/// sensor), recorded when this test was written; any change to the import, the trace text or the
/// integration shows here.
const PINNED_TRACE_SHA256: &str =
    "7e8d77a81fb798b2fde6fea95652f4d2b1472d81beca6abe6a4afa878a08c175";
const PINNED_LOAD_W_BITS: [u64; 12] = [
    4625661995829110005,
    4625721396363794355,
    4625765156589055405,
    4631401912623393565,
    4631459190453104905,
    4631537508167623184,
    4625825497231459059,
    4625908946033361980,
    4625918647887057358,
    4633310038741507867,
    4633377619577428340,
    4633420505487669504,
];
