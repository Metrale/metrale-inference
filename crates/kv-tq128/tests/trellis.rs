// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: Viterbi against exhaustive search, its tie rules on hand-built cases, and the
//! bitstream it packs into.

mod common;

use common::Rng;
use metrale_kv_tq128::trellis::{Trellis, is_walk, path_cost, viterbi};
use metrale_kv_tq128::{KvSide, Tq128Codec, Tq128Error, pack_trellis, trellis_state_at};

/// Every walk of `steps` states, depth first.
fn all_walks(t: Trellis, steps: usize, f: &mut dyn FnMut(&[u16])) {
    fn go(t: Trellis, steps: usize, acc: &mut Vec<u16>, f: &mut dyn FnMut(&[u16])) {
        if acc.len() == steps {
            f(acc);
            return;
        }
        let prev = usize::from(*acc.last().unwrap());
        for sym in 0..t.fan() {
            acc.push(t.next(prev, sym) as u16);
            go(t, steps, acc, f);
            acc.pop();
        }
    }
    for s0 in 0..t.states() {
        let mut acc = vec![s0 as u16];
        go(t, steps, &mut acc, f);
    }
}

fn check_optimal(t: Trellis, book: &[f32], x: &[f32]) {
    let path = viterbi(t, book, x).unwrap();
    assert!(is_walk(t, &path.states));
    assert_eq!(
        path_cost(book, x, &path.states).to_bits(),
        path.cost.to_bits()
    );
    let mut best = f32::INFINITY;
    all_walks(t, x.len(), &mut |w| best = best.min(path_cost(book, x, w)));
    assert_eq!(
        path.cost.to_bits(),
        best.to_bits(),
        "viterbi {} vs exhaustive {best}",
        path.cost
    );
}

#[test]
fn viterbi_finds_the_exhaustive_minimum_on_the_real_trellises() {
    let mut rng = Rng::new(20);
    let cases = [
        (Tq128Codec::Tcq3, 3, 6),
        (Tq128Codec::Tcq2, 3, 12),
        (Tq128Codec::Tcq1, 5, 12),
    ];
    for (codec, steps, trials) in cases {
        for side in [KvSide::Key, KvSide::Value] {
            for _ in 0..trials {
                let x = rng.normals(steps, 1.0 / 128f64.sqrt());
                check_optimal(codec.trellis().unwrap(), codec.codebook(side), &x);
            }
        }
    }
}

#[test]
fn viterbi_finds_the_exhaustive_minimum_on_toy_trellises() {
    let mut rng = Rng::new(21);
    for (k, l) in [(1, 2), (1, 3), (2, 3), (2, 4), (3, 5)] {
        let t = Trellis::new(k, l).unwrap();
        for _ in 0..40 {
            // Coarse values make exact cost ties common, exercising the comparisons.
            let book: Vec<f32> = (0..t.states())
                .map(|_| (rng.next_u64() % 5) as f32)
                .collect();
            let x: Vec<f32> = (0..6).map(|_| (rng.next_u64() % 5) as f32).collect();
            check_optimal(t, &book, &x);
        }
    }
}

#[test]
fn ties_resolve_to_the_lowest_predecessor_and_final_state() {
    let t = Trellis::new(1, 2).unwrap();
    // A flat book ties every path: the lowest final state (0) and predecessor 0 throughout.
    let p = viterbi(t, &[1.0; 4], &[0.0; 5]).unwrap();
    assert_eq!(p.states, vec![0; 5]);
    // Hand-derived: book[s] = s, x = [3, 0]. From s0 = 3 the successors are {1, 3}; [3, 1]
    // costs 0 + 1. Any other start costs at least 1 more.
    let p = viterbi(t, &[0.0, 1.0, 2.0, 3.0], &[3.0, 0.0]).unwrap();
    assert_eq!((p.states, p.cost), (vec![3, 1], 1.0));
    // x = [2.5]: states 2 and 3 tie at 0.25; the lower index wins.
    let p = viterbi(t, &[0.0, 1.0, 2.0, 3.0], &[2.5]).unwrap();
    assert_eq!(p.states, vec![2]);
}

#[test]
fn viterbi_is_deterministic() {
    let mut rng = Rng::new(22);
    let x = rng.normals(128, 0.09);
    for codec in [Tq128Codec::Tcq3, Tq128Codec::Tcq2, Tq128Codec::Tcq1] {
        let t = codec.trellis().unwrap();
        let a = viterbi(t, codec.codebook(KvSide::Key), &x).unwrap();
        let b = viterbi(t, codec.codebook(KvSide::Key), &x).unwrap();
        assert_eq!(a, b);
    }
}

#[test]
fn the_stream_reads_back_every_state() {
    // Hand-derived: k=1, L=2, states [3, 1] -> prefix bit 1, symbols 1, 0 -> 0b011.
    let t = Trellis::new(1, 2).unwrap();
    let mut out = [0xAA_u8; 1];
    pack_trellis(t, &[3, 1], &mut out).unwrap();
    assert_eq!(out, [0b011]);
    let mut rng = Rng::new(23);
    for codec in [Tq128Codec::Tcq3, Tq128Codec::Tcq2, Tq128Codec::Tcq1] {
        let t = codec.trellis().unwrap();
        for _ in 0..50 {
            let mut states = vec![(rng.next_u64() as usize % t.states()) as u16];
            for _ in 1..128 {
                let prev = usize::from(*states.last().unwrap());
                states.push(t.next(prev, rng.next_u64() as usize % t.fan()) as u16);
            }
            let mut out = vec![0xFF_u8; codec.payload_bytes()];
            pack_trellis(t, &states, &mut out).unwrap();
            for (i, &s) in states.iter().enumerate() {
                assert_eq!(trellis_state_at(t, &out, i), s, "{codec:?} t={i}");
            }
            let used = t.stream_bits(128);
            for bit in used..out.len() * 8 {
                assert_eq!(
                    out[bit / 8] >> (bit % 8) & 1,
                    0,
                    "pad bit {bit} of {codec:?}"
                );
            }
        }
    }
}

#[test]
fn bad_shapes_and_lengths_are_refused() {
    for (k, l) in [(0, 2), (3, 3), (4, 3), (3, 10)] {
        assert_eq!(Trellis::new(k, l), Err(Tq128Error::TrellisShape { k, l }));
    }
    let t = Trellis::TCQ2;
    assert_eq!(
        viterbi(t, &[0.0; 255], &[0.0]).unwrap_err(),
        Tq128Error::CodebookLength {
            expected: 256,
            found: 255
        }
    );
    assert_eq!(
        pack_trellis(t, &[0; 128], &mut [0; 32]).unwrap_err(),
        Tq128Error::BlockLength {
            expected: 33,
            found: 32
        }
    );
}
