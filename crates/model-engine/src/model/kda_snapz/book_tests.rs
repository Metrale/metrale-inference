// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-16: `Book` against the eager commit it replaces. A small device model (one live and
//! one first-snapshot state per slot, the flag table) runs random interleavings of batched
//! verifies (duplicate slots, positions past the table, failures; a position reads its
//! snapshot only where the flag is set AND the layer gives it a defer), commits and other
//! readers twice: once committing eagerly, once through the book. Every state a verify or a reader
//! sees must match. Controls: a driver that drops the flag table, and one that skips the
//! flushes `begin` asks for, must each diverge.

use super::*;

const SLOTS: usize = 6;
const CAP: usize = 4;

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn mix(s: u64, t: u64) -> u64 {
    (s ^ t).wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(17)
}

#[derive(Clone, Copy, PartialEq)]
enum Driver {
    Eager,
    Book,
    DropFlags,
    SkipBeginFlush,
}

struct Dev {
    live: Vec<u64>,
    snap: Vec<u64>,
    flags: Vec<u32>,
}

/// 2026-10-16: One run of `ops` random operations under `driver`; returns every state a
/// verify read or a reader saw, then each slot's final live state after a flush.
fn run(seed: u64, ops: usize, driver: Driver) -> Vec<u64> {
    let mut rng = Lcg(seed);
    let mut dev = Dev {
        live: (0..SLOTS as u64).map(|s| mix(s, 7)).collect(),
        snap: vec![0; SLOTS],
        flags: vec![0; CAP],
    };
    let mut book: Book<usize> = Book::new(CAP);
    let mut seen = Vec::new();
    let land = |dev: &mut Dev, plans: Vec<usize>| {
        for s in plans {
            dev.live[s] = dev.snap[s];
        }
    };
    for _ in 0..ops {
        match rng.below(4) {
            0..=1 => {
                let n = 1 + rng.below(SLOTS);
                let slots: Vec<usize> = (0..n).map(|_| rng.below(SLOTS)).collect();
                if driver != Driver::Eager {
                    let b = book.begin(&slots);
                    if driver != Driver::SkipBeginFlush {
                        land(&mut dev, b.flush);
                    }
                    if let Some(t) = b.upload
                        && driver != Driver::DropFlags
                    {
                        dev.flags = t;
                    }
                }
                let ok = rng.below(8) != 0;
                if !ok {
                    book.end(false, &slots);
                    continue;
                }
                for (i, &s) in slots.iter().enumerate() {
                    // 2026-10-16: The layer honours a flag only where it gives a defer: below
                    // the table, at a slot no other position shares (`defers_from`).
                    let defer = i < CAP && slots.iter().filter(|&&o| o == s).count() == 1;
                    let deferred = driver != Driver::Eager && defer && dev.flags[i] != 0;
                    let src = if deferred { dev.snap[s] } else { dev.live[s] };
                    seen.push(src);
                    let row0 = mix(src, rng.next());
                    dev.snap[s] = row0;
                    dev.live[s] = mix(row0, rng.next());
                }
                book.end(true, &slots);
                let mut done = Vec::new();
                for &s in &slots {
                    if done.contains(&s) {
                        continue;
                    }
                    done.push(s);
                    let armed = driver != Driver::Eager && book.take_armed(s);
                    if rng.below(2) == 0 {
                        if armed {
                            book.defer(s, s).unwrap();
                        } else {
                            dev.live[s] = dev.snap[s];
                        }
                    }
                }
            }
            2 => {
                let s = rng.below(SLOTS);
                land(&mut dev, book.flush_all());
                seen.push(dev.live[s]);
                dev.live[s] = mix(dev.live[s], rng.next());
            }
            _ => {
                land(&mut dev, book.flush_all());
                seen.extend(dev.live.iter().copied());
            }
        }
    }
    land(&mut dev, book.flush_all());
    seen.extend(dev.live.iter().copied());
    seen
}

/// 2026-10-16: Under every interleaving the book's deferred commits are invisible: each state
/// a verify or a reader sees is the eager one. The controls prove the comparison sees both a
/// flag table that never reaches the kernels and a deferred commit that never lands.
#[test]
fn deferred_commits_match_eager_commits_under_random_interleavings() {
    let mut drop_flags = 0;
    let mut skip_flush = 0;
    for seed in 0..400u64 {
        let eager = run(seed, 60, Driver::Eager);
        assert_eq!(run(seed, 60, Driver::Book), eager, "seed {seed}");
        drop_flags += usize::from(run(seed, 60, Driver::DropFlags) != eager);
        skip_flush += usize::from(run(seed, 60, Driver::SkipBeginFlush) != eager);
    }
    assert!(
        drop_flags > 100,
        "drop-flags control diverged on {drop_flags} seeds"
    );
    assert!(
        skip_flush > 0,
        "skip-flush control diverged on {skip_flush} seeds"
    );
}

/// 2026-10-16: Eligibility: below the table and the slot's only position.
#[test]
fn eligibility_needs_a_unique_slot_below_the_table() {
    let slots = [3, 1, 3, 2, 5];
    let got: Vec<bool> = (0..slots.len()).map(|i| eligible(&slots, i, 4)).collect();
    assert_eq!(got, [false, true, false, true, false]);
    assert!(eligible(&slots, 4, 5));
}

/// 2026-10-16: The table uploads on the first verify and on a change only; a failed verify
/// arms nothing, keeps the pending commit, and forces the next upload. A second deferral of
/// one slot is refused, and an arm is consumed once.
#[test]
fn uploads_arms_and_failures() {
    let mut b: Book<u8> = Book::new(4);
    let first = b.begin(&[0, 1]);
    assert_eq!(first.upload, Some(vec![0, 0, 0, 0]));
    assert_eq!(b.begin(&[0, 1]).upload, None);
    b.end(true, &[0, 1]);
    assert!(b.take_armed(1));
    assert!(!b.take_armed(1));
    b.defer(1, 9).unwrap();
    assert!(b.defer(1, 8).is_err());
    let next = b.begin(&[1, 0]);
    assert_eq!(
        next,
        Begin {
            flush: vec![],
            upload: Some(vec![1, 0, 0, 0])
        }
    );
    b.end(false, &[1, 0]);
    assert!(!b.take_armed(0));
    assert!(!b.is_idle());
    let again = b.begin(&[1, 0]);
    assert_eq!(again.upload, Some(vec![1, 0, 0, 0]));
    let dup = b.begin(&[1, 1]);
    assert_eq!(
        dup,
        Begin {
            flush: vec![9],
            upload: Some(vec![0, 0, 0, 0])
        }
    );
    assert!(b.flush_all().is_empty());
    assert!(b.is_idle());
}
