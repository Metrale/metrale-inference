// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: The SP pipeline's [`program`] lowered (its sections through
//! `glm5next_sp_overlap::schedule_deferred`, as `run_section` enqueues them) onto a two-stream
//! happens-before model. The model checks that no two accesses of one buffer region race,
//! that every read sees the writer the serial body's order gives it, and that the exchanges
//! really overlap the mixer and the MLP; the negative controls (two normed slots, one shared
//! receive workspace, a dropped wait, no join) must each fail it.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use std::collections::HashMap;

use super::*;
use crate::glm5next_sp_overlap::{self as ov, Ev, Op};

/// 2026-10-14: A logical buffer of the layer body; `Streams`/`Hidden` are per sub-chunk rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Buf {
    Normed,
    FinalOut,
    RecvMid,
    RecvTail,
    FfnOut,
    Streams(usize),
    Hidden(usize),
    MixState,
}

/// 2026-10-14: A row region: this rank's rows of piece `j`, or the peers' rows of piece `j`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Rg {
    Own(usize),
    Peer(usize),
}

/// 2026-10-14: One device launch of the body: a stage's compute (per piece for the mHC
/// sections) or exchange.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Label {
    Start,
    End,
    HeadC(usize, usize),
    Agh(usize, usize),
    Mixer(usize),
    Rsm(usize),
    MidC(usize, usize),
    Agm(usize, usize),
    Mlp(usize),
    Rst(usize),
    TailC(usize, usize),
    Agt(usize, usize),
}

/// 2026-10-14: The layer variant: pieces per shard, DSA (its output over its normed input) or
/// KDA (`final_out`), the MLP reduce, the last layer, and the sub-chunk count.
#[derive(Clone, Copy, Debug)]
struct Variant {
    m: usize,
    dsa: bool,
    reduce: bool,
    last: bool,
    n: usize,
}

type Loc = (Buf, Rg);

fn own(b: Buf, j: usize) -> Loc {
    (b, Rg::Own(j))
}

fn peer_all(b: Buf, m: usize) -> Vec<Loc> {
    (0..m).map(|j| (b, Rg::Peer(j))).collect()
}

fn own_all(b: Buf, m: usize) -> Vec<Loc> {
    (0..m).map(|j| own(b, j)).collect()
}

fn all(b: Buf, m: usize) -> Vec<Loc> {
    let mut v = own_all(b, m);
    v.extend(peer_all(b, m));
    v
}

/// 2026-10-14: The `(reads, writes)` of a launch, on logical buffers. The pre-norms write the
/// mixed `hidden` rows, `hc_expand` reads them; DSA's output is its normed buffer.
fn access(l: Label, v: Variant) -> (Vec<Loc>, Vec<Loc>) {
    let m = v.m;
    let attn = if v.dsa { Buf::Normed } else { Buf::FinalOut };
    match l {
        Label::Start => (vec![], vec![]),
        Label::End => {
            let mut r = vec![(Buf::MixState, Rg::Own(0))];
            for s in 0..v.n {
                r.extend(all(Buf::Streams(s), m));
                r.extend(all(Buf::Hidden(s), m));
            }
            (r, vec![])
        }
        Label::HeadC(s, j) => (
            vec![own(Buf::Hidden(s), j), own(Buf::Streams(s), j)],
            vec![
                own(Buf::Hidden(s), j),
                own(Buf::Streams(s), j),
                own(Buf::Normed, j),
            ],
        ),
        Label::Agh(_, j) | Label::Agm(_, j) => {
            (vec![own(Buf::Normed, j)], vec![(Buf::Normed, Rg::Peer(j))])
        }
        Label::Mixer(_) => {
            let mut r = all(Buf::Normed, m);
            r.push((Buf::MixState, Rg::Own(0)));
            let mut w = all(attn, m);
            w.push((Buf::MixState, Rg::Own(0)));
            (r, w)
        }
        Label::Rsm(_) => (peer_all(attn, m), own_all(Buf::RecvMid, m)),
        Label::MidC(s, j) => (
            vec![own(attn, j), own(Buf::RecvMid, j), own(Buf::Streams(s), j)],
            vec![
                own(Buf::Streams(s), j),
                own(Buf::Hidden(s), j),
                own(Buf::Normed, j),
            ],
        ),
        Label::Mlp(_) => (all(Buf::Normed, m), all(Buf::FfnOut, m)),
        Label::Rst(_) => (peer_all(Buf::FfnOut, m), own_all(Buf::RecvTail, m)),
        Label::TailC(s, j) => {
            let mut r = vec![own(Buf::FfnOut, j), own(Buf::Streams(s), j)];
            if v.reduce {
                r.push(own(Buf::RecvTail, j));
            }
            let mut w = vec![own(Buf::Streams(s), j)];
            if v.last {
                w.push(own(Buf::Hidden(s), j));
            }
            (r, w)
        }
        Label::Agt(s, j) => (
            vec![own(Buf::Hidden(s), j)],
            vec![(Buf::Hidden(s), Rg::Peer(j))],
        ),
    }
}

/// 2026-10-14: Where a logical buffer of sub-chunk `s` lives: `normed_slots` normed slots
/// (the serial body has one), and the two receive workspaces or one shared.
#[derive(Clone, Copy, Debug)]
struct Layout {
    normed_slots: usize,
    shared_recv: bool,
}

const PIPELINE: Layout = Layout {
    normed_slots: NORMED_SLOTS,
    shared_recv: false,
};
const SERIAL: Layout = Layout {
    normed_slots: 1,
    shared_recv: true,
};

/// 2026-10-14: The physical buffer of a logical one of sub-chunk `s`: `(buffer, slot)`.
fn place(b: Buf, s: usize, lay: Layout) -> (Buf, usize) {
    match b {
        Buf::Normed if lay.normed_slots == NORMED_SLOTS => (b, normed_slot(s)),
        Buf::Normed => (b, s % lay.normed_slots),
        Buf::RecvTail if lay.shared_recv => (Buf::RecvMid, 0),
        Buf::RecvMid => (b, recv_slot(RsKind::Mixer)),
        Buf::RecvTail => (b, recv_slot(RsKind::Mlp)),
        _ => (b, 0),
    }
}

/// 2026-10-14: The sub-chunk whose buffers a launch uses.
fn sub_of(l: Label) -> usize {
    match l {
        Label::Start | Label::End => 0,
        Label::HeadC(s, _)
        | Label::Agh(s, _)
        | Label::Mixer(s)
        | Label::Rsm(s)
        | Label::MidC(s, _)
        | Label::Agm(s, _)
        | Label::Mlp(s)
        | Label::Rst(s)
        | Label::TailC(s, _)
        | Label::Agt(s, _) => s,
    }
}

/// 2026-10-14: An event of the model: the pipeline's or a section's.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Key {
    Pipe(Event),
    Sec(Ev),
}

/// 2026-10-14: One enqueue on the device model.
#[derive(Clone, Copy, Debug)]
enum Low {
    Rec(Key, On),
    Wait(On, Key),
    Run(On, Label),
}

/// 2026-10-14: `prims` as device enqueues, each section through `schedule_deferred` (what
/// `run_section` enqueues for a deferred section), between the earlier work (`Start`) and the
/// later work (`End`) on the main stream.
fn lower(prims: &[Prim], v: Variant) -> Vec<Low> {
    let mut out = vec![Low::Run(On::Main, Label::Start)];
    for &p in prims {
        match p {
            Prim::Record(e, s) => out.push(Low::Rec(Key::Pipe(e), s)),
            Prim::Wait(s, e) => out.push(Low::Wait(s, Key::Pipe(e))),
            Prim::Mixer(s) => out.push(Low::Run(On::Main, Label::Mixer(s))),
            Prim::Mlp(s) => out.push(Low::Run(On::Main, Label::Mlp(s))),
            Prim::Rs(RsKind::Mixer, s) => out.push(Low::Run(On::Comm, Label::Rsm(s))),
            Prim::Rs(RsKind::Mlp, s) => out.push(Low::Run(On::Comm, Label::Rst(s))),
            Prim::Section { stage, sub, done } => {
                let key = |e: Ev| match (e, done) {
                    (Ev::Out, Some(d)) => Key::Pipe(d),
                    _ => Key::Sec(e),
                };
                let (compute, gather): (fn(usize, usize) -> Label, fn(usize, usize) -> Label) =
                    match stage {
                        Stage::Head => (Label::HeadC, Label::Agh),
                        Stage::Mid => (Label::MidC, Label::Agm),
                        Stage::Tail => (Label::TailC, Label::Agt),
                        Stage::Mixer | Stage::Mlp => panic!("{stage:?} is not a section"),
                    };
                for op in ov::schedule_deferred(v.m, done.is_some()) {
                    out.push(match op {
                        Op::Record(e, s) => Low::Rec(key(e), s),
                        Op::Wait(s, e) => Low::Wait(s, key(e)),
                        Op::Compute(j) => Low::Run(On::Main, compute(sub, j)),
                        Op::AllGather(j, s) => Low::Run(s, gather(sub, j)),
                        Op::ReduceScatter(..) => panic!("a deferred section reduce-scatters"),
                    });
                }
            }
        }
    }
    out.push(Low::Run(On::Main, Label::End));
    out
}

/// 2026-10-14: The serial SP body's launches in order (one stream), `n` sub-chunks.
fn serial(v: Variant) -> Vec<Label> {
    let mut out = vec![Label::Start];
    for s in 0..v.n {
        out.extend((0..v.m).map(|j| Label::HeadC(s, j)));
        out.extend((0..v.m).map(|j| Label::Agh(s, j)));
        out.extend([Label::Mixer(s), Label::Rsm(s)]);
        out.extend((0..v.m).map(|j| Label::MidC(s, j)));
        out.extend((0..v.m).map(|j| Label::Agm(s, j)));
        out.push(Label::Mlp(s));
        if v.reduce {
            out.push(Label::Rst(s));
        }
        out.extend((0..v.m).map(|j| Label::TailC(s, j)));
        if v.last {
            out.extend((0..v.m).map(|j| Label::Agt(s, j)));
        }
    }
    out.push(Label::End);
    out
}

/// 2026-10-14: The launches of `lows` with their stream and vector clock: a launch advances its
/// stream's clock, a record snapshots it, a wait joins the snapshot of the event's last record
/// in host order (an unrecorded wait panics).
fn clocks(lows: &[Low]) -> Vec<(Label, usize, [usize; 2])> {
    let idx = |s: On| usize::from(s == On::Comm);
    let mut vc = [[0usize; 2]; 2];
    let mut ev: HashMap<Key, [usize; 2]> = HashMap::new();
    let mut out = Vec::new();
    for &l in lows {
        match l {
            Low::Rec(k, s) => {
                ev.insert(k, vc[idx(s)]);
            }
            Low::Wait(s, k) => {
                let e = ev
                    .get(&k)
                    .unwrap_or_else(|| panic!("wait on unrecorded {k:?}"));
                for c in 0..2 {
                    vc[idx(s)][c] = vc[idx(s)][c].max(e[c]);
                }
            }
            Low::Run(s, label) => {
                vc[idx(s)][idx(s)] += 1;
                out.push((label, idx(s), vc[idx(s)]));
            }
        }
    }
    out
}

/// 2026-10-14: Check `lows` against the serial body: no race between two launches touching a
/// physical region (one writing) that the model leaves unordered, and every read sees the
/// writer it has in the serial order. Returns the first violation.
fn check(lows: &[Low], v: Variant, lay: Layout) -> Result<(), String> {
    let runs = clocks(lows);
    let hb = |a: usize, b: usize| a != b && runs[a].2[runs[a].1] <= runs[b].2[runs[a].1];
    let phys = |label: Label, (b, r): Loc| (place(b, sub_of(label), lay), r);
    let acc: Vec<_> = runs.iter().map(|r| access(r.0, v)).collect();
    for a in 0..runs.len() {
        for b in a + 1..runs.len() {
            let (ra, wa) = &acc[a];
            let (rb, wb) = &acc[b];
            let touches = |w: &[Loc], other: &[Loc], la: Label, lb: Label| {
                w.iter()
                    .any(|&x| other.iter().any(|&y| phys(la, x) == phys(lb, y)))
            };
            let (la, lb) = (runs[a].0, runs[b].0);
            let conflict =
                touches(wa, wb, la, lb) || touches(wa, rb, la, lb) || touches(wb, ra, lb, la);
            if conflict && !hb(a, b) && !hb(b, a) {
                return Err(format!("race: {la:?} / {lb:?}"));
            }
        }
    }
    let mut want: HashMap<(Label, Loc), Option<Label>> = HashMap::new();
    let mut last: HashMap<((Buf, usize), Rg), Label> = HashMap::new();
    let serial_runs = serial(v);
    for &l in &serial_runs {
        let (r, w) = access(l, v);
        for x in r {
            want.insert(
                (l, x),
                last.get(&(place(x.0, sub_of(l), SERIAL), x.1)).copied(),
            );
        }
        for x in w {
            last.insert((place(x.0, sub_of(l), SERIAL), x.1), l);
        }
    }
    let mut seen: Vec<Label> = runs.iter().map(|r| r.0).collect();
    let mut expect = serial_runs.clone();
    seen.sort_by_key(|l| format!("{l:?}"));
    expect.sort_by_key(|l| format!("{l:?}"));
    if seen != expect {
        return Err("the launches differ from the serial body's".into());
    }
    for b in 0..runs.len() {
        for &x in &acc[b].0 {
            let p = phys(runs[b].0, x);
            let writers: Vec<usize> = (0..runs.len())
                .filter(|&a| hb(a, b) && acc[a].1.iter().any(|&y| phys(runs[a].0, y) == p))
                .collect();
            let got = writers
                .iter()
                .copied()
                .find(|&a| writers.iter().all(|&c| c == a || hb(c, a)))
                .map(|a| runs[a].0);
            let w = want[&(runs[b].0, x)];
            if got != w {
                return Err(format!(
                    "{:?} reads {x:?} from {got:?}, the serial body from {w:?}",
                    runs[b].0
                ));
            }
        }
    }
    Ok(())
}

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
