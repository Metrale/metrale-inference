// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: The cases of the SP pipeline's happens-before model (`glm5next_sp_pipeline_hb_tests`),
//! moved out of it unchanged to keep each file under 500 lines, and the cell pipeline's
//! after-mixer negative control.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use super::*;

fn variants() -> Vec<Variant> {
    let mut out = Vec::new();
    for n in 2..=4 {
        for m in 1..=3 {
            for dsa in [false, true] {
                for reduce in [false, true] {
                    for last in [false, true] {
                        out.push(Variant {
                            m,
                            dsa,
                            reduce,
                            last,
                            n,
                        });
                    }
                }
            }
        }
    }
    out
}

/// 2026-10-14: For 2-4 sub-chunks, 1-3 pieces, KDA and DSA, with and without the MLP reduce,
/// on the last layer and not: the program is race-free and every read sees its serial writer
/// (so the bytes are the serial body's), with every exchange waited before its consumer and
/// the comm stream joined before the later work.
#[test]
fn the_program_is_the_serial_body_on_two_streams() {
    for v in variants() {
        let lows = lower(&program(v.n, v.reduce, v.last), v);
        check(&lows, v, PIPELINE).unwrap_or_else(|e| panic!("{v:?}: {e}"));
    }
}

/// 2026-10-14: The negative controls: two normed slots (a head two sub-chunks ahead overwrites
/// the normed rows the MLP still reads), one receive workspace for both reduce-scatters, a
/// dropped consumer wait (each kind in turn), and no join each fail the check.
#[test]
fn the_check_catches_each_broken_variant() {
    let v = Variant {
        m: 2,
        dsa: false,
        reduce: true,
        last: true,
        n: 4,
    };
    let prims = program(v.n, v.reduce, v.last);
    let lows = lower(&prims, v);
    for lay in [
        Layout {
            normed_slots: 2,
            shared_recv: false,
        },
        Layout {
            normed_slots: NORMED_SLOTS,
            shared_recv: true,
        },
    ] {
        assert!(check(&lows, v, lay).is_err(), "{lay:?}");
    }
    for ev in [Event::Head, Event::MixerRs, Event::Mid, Event::MlpRs] {
        let i = prims
            .iter()
            .rposition(|&p| p == Prim::Wait(On::Main, ev))
            .unwrap();
        let mut dropped = prims.clone();
        dropped.remove(i);
        assert!(
            check(&lower(&dropped, v), v, PIPELINE).is_err(),
            "dropped the last wait on {ev:?}"
        );
    }
    let mut unjoined = prims.clone();
    assert_eq!(unjoined.pop(), Some(Prim::Wait(On::Main, Event::Join)));
    assert!(check(&lower(&unjoined, v), v, PIPELINE).is_err());
    let mut no_rs_in = prims.clone();
    let i = no_rs_in
        .iter()
        .position(|&p| p == Prim::Wait(On::Comm, Event::RsIn))
        .unwrap();
    no_rs_in.remove(i);
    assert!(check(&lower(&no_rs_in, v), v, PIPELINE).is_err());
}

/// 2026-10-14: The overlap is real in the steady state: the mixer's reduce-scatter of `s + 1`
/// and the head all-gather of `s + 2` are unordered with the MLP of `s`; the post-mix
/// all-gather of `s` with the mixer of `s + 1`; the MLP's reduce-scatter of `s` with the mixer
/// of `s + 2`.
#[test]
fn the_exchanges_overlap_the_mixer_and_the_mlp() {
    for m in 1..=3 {
        let v = Variant {
            m,
            dsa: false,
            reduce: true,
            last: false,
            n: 4,
        };
        let runs = clocks(&lower(&program(v.n, v.reduce, v.last), v));
        let at = |l: Label| runs.iter().position(|r| r.0 == l).unwrap();
        let free = |a: Label, b: Label| {
            let (a, b) = (at(a), at(b));
            let hb = |x: usize, y: usize| runs[x].2[runs[x].1] <= runs[y].2[runs[x].1];
            !hb(a, b) && !hb(b, a)
        };
        for s in 0..2 {
            assert!(free(Label::Rsm(s + 1), Label::Mlp(s)), "m {m} s {s}");
            assert!(free(Label::Agh(s + 2, m - 1), Label::Mlp(s)), "m {m} s {s}");
            assert!(
                free(Label::Agm(s, m - 1), Label::Mixer(s + 1)),
                "m {m} s {s}"
            );
            assert!(free(Label::Rst(s), Label::Mixer(s + 2)), "m {m} s {s}");
        }
    }
}

/// 2026-10-15: The cell pipeline's after-mixer work (capture, clamp) sits between `Mixer(s)` and
/// `Mixer(s + 1)` in every variant; the negative control moves `AfterMixer(s)` after
/// `Mixer(s + 1)` (the next cell would read the unclamped state) and must fail the check.
#[test]
fn the_after_mixer_work_precedes_the_next_mixer() {
    for v in variants() {
        let prims = program(v.n, v.reduce, v.last);
        let at = |p: Prim| prims.iter().position(|&q| q == p).unwrap();
        for s in 0..v.n {
            assert!(at(Prim::Mixer(s)) < at(Prim::AfterMixer(s)), "{v:?} s {s}");
            if s + 1 < v.n {
                assert!(
                    at(Prim::AfterMixer(s)) < at(Prim::Mixer(s + 1)),
                    "{v:?} s {s}"
                );
            }
        }
        for s in 0..v.n - 1 {
            let mut late = prims.clone();
            let a = late.remove(at(Prim::AfterMixer(s)));
            let m = late.iter().position(|&q| q == Prim::Mixer(s + 1)).unwrap();
            late.insert(m + 1, a);
            assert!(
                check(&lower(&late, v), v, PIPELINE).is_err(),
                "{v:?}: AfterMixer({s}) after Mixer({})",
                s + 1
            );
        }
    }
}
