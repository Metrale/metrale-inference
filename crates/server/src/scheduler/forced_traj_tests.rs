// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The forced-trajectory mode: a record parses back to itself, a replay follows the
//! record whatever this serve's own picks are, and every way of leaving the record is refused.

use super::*;

const KEY: u64 = 0xabcd_0123_4567_89ef;

fn at(out: usize, kv: usize) -> At {
    At { key: KEY, out, kv }
}

/// 2026-10-23: A request's record: a bootstrap, an accepted verify, a rejected verify.
fn record() -> Vec<Rec> {
    vec![
        Rec {
            out: 1,
            kv: 100,
            last: 7,
            kind: RecKind::Boot { tok: 8 },
        },
        Rec {
            out: 2,
            kv: 101,
            last: 8,
            kind: RecKind::K2 {
                draft: 9,
                acc: true,
                e0: 9,
                e1: 10,
            },
        },
        Rec {
            out: 4,
            kv: 103,
            last: 10,
            kind: RecKind::K2 {
                draft: 11,
                acc: false,
                e0: 12,
                e1: 13,
            },
        },
    ]
}

fn log() -> String {
    let mut s = String::from("2026-10-23 INFO unrelated line\n");
    for r in record() {
        s.push_str(&format!(
            "2026-10-23T00:00:00Z INFO met: {}\n",
            rec_line(KEY, &r)
        ));
    }
    s
}

fn replay() -> ForcedTraj {
    ForcedTraj::new(TrajMode::Replay(Trajectory::parse(&log()).unwrap()))
}

#[test]
fn a_record_parses_back_to_the_steps_that_wrote_it() {
    let t = Trajectory::parse(&log()).unwrap();
    assert_eq!(t.reqs.len(), 1);
    assert_eq!(t.reqs[&KEY], record());
    assert!(
        Trajectory::parse("no record here\n").is_err(),
        "an empty record is refused"
    );
    assert!(
        Trajectory::parse("PAIRED_TRAJ rec k2 key=01 out=1\n").is_err(),
        "malformed"
    );
}

/// 2026-10-23: Whatever this serve's own picks, the replay goes on with the record's tokens,
/// so `draft == v0` reproduces the recorded verdict at every step.
#[test]
fn a_replay_follows_the_record_whatever_the_own_picks() {
    let f = replay();
    let mut last = 7;
    assert!(!f.feed(at(1, 100), false, &mut last).unwrap());
    assert_eq!(
        f.boot_pick(at(1, 100), last, 99).unwrap(),
        8,
        "the recorded pick, not 99"
    );

    // 2026-10-23: The first verify: this serve fed a different token (its own first pick).
    let mut last = 55;
    assert!(
        f.feed(at(2, 101), true, &mut last).unwrap(),
        "a differing fed token is replaced"
    );
    assert_eq!(last, 8);
    assert_eq!(
        f.k2_draft(at(2, 101), 77).unwrap(),
        9,
        "the recorded draft, not ours"
    );
    // 2026-10-23: Our own picks would have rejected (v0 = 3 != 9); the record accepted.
    let [d, v0, v1] = f.k2_verdict(at(2, 101), 8, 9, 77, [3, 4]).unwrap();
    assert_eq!((d, v0, v1), (9, 9, 10));
    assert!(d == v0, "accepted, as recorded");

    let mut last = 10;
    assert!(!f.feed(at(4, 103), true, &mut last).unwrap());
    // 2026-10-23: Our picks would have accepted (v0 = 11 = draft); the record rejected to 12.
    let [d, v0, _] = f.k2_verdict(at(4, 103), 10, 11, 11, [11, 5]).unwrap();
    assert_eq!((d, v0), (11, 12));
    assert!(d != v0, "rejected, as recorded");
    let mut last = 12;
    assert!(
        f.feed(at(5, 104), true, &mut last).is_err(),
        "past the record"
    );
}

#[test]
fn leaving_the_record_is_refused() {
    let f = replay();
    let mut last = 7;
    assert!(
        f.feed(at(1, 101), false, &mut last).is_err(),
        "another KV length"
    );
    assert!(
        f.feed(at(2, 100), false, &mut last).is_err(),
        "another output index"
    );
    assert!(
        f.feed(at(1, 100), true, &mut last).is_err(),
        "a verify where a bootstrap ran"
    );
    let other = At {
        key: 1,
        out: 1,
        kv: 100,
    };
    assert!(
        f.feed(other, false, &mut last).is_err(),
        "an unrecorded request"
    );
    // 2026-10-23: A verify that fed another draft than the record's.
    f.boot_pick(at(1, 100), 7, 8).unwrap();
    assert!(f.k2_verdict(at(2, 101), 8, 5, 5, [9, 10]).is_err());
}

/// 2026-10-23: Recording changes nothing a step does.
#[test]
fn recording_returns_the_steps_own_values() {
    let f = ForcedTraj::new(TrajMode::Record);
    let mut last = 3;
    assert!(!f.feed(at(1, 100), true, &mut last).unwrap());
    assert_eq!(last, 3);
    assert_eq!(f.k2_draft(at(1, 100), 4).unwrap(), 4);
    assert_eq!(
        f.k2_verdict(at(1, 100), 3, 4, 4, [5, 6]).unwrap(),
        [4, 5, 6]
    );
    assert_eq!(f.boot_pick(at(2, 101), 5, 6).unwrap(), 6);
}

#[test]
fn the_key_is_the_prompt_and_is_hashed_once_per_request() {
    assert_ne!(prompt_key(&[1, 2]), prompt_key(&[2, 1]));
    assert_ne!(prompt_key(&[1]), prompt_key(&[1, 0]));
    let f = ForcedTraj::new(TrajMode::Record);
    let t0 = std::time::Instant::now();
    let k = f.key(3, t0, &[1, 2, 3]);
    assert_eq!(k, prompt_key(&[1, 2, 3]));
    assert_eq!(f.key(3, t0, &[9]), k, "cached for the same request");
}
