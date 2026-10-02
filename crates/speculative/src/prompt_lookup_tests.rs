// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: Tests for the prompt-lookup index and copy window, in three
//! parts: what it must propose, what it must refuse, and its boundaries. A
//! brute-force reference checks the index on random histories.

use super::*;

/// 2026-10-02: Reference: the continuation of the latest earlier occurrence
/// of the history's final n-gram that has a following token.
fn brute_force(history: &[u32], n: usize, max_len: usize) -> Option<Vec<u32>> {
    let len = history.len();
    if max_len == 0 || len <= n {
        return None;
    }
    let suffix = &history[len - n..];
    (n..len)
        .rev()
        .find(|&start| history[start - n..start] == *suffix)
        .map(|start| history[start..len.min(start + max_len)].to_vec())
}

fn indexed(history: &[u32], n: usize) -> PromptLookupIndex {
    let mut idx = PromptLookupIndex::new(n);
    idx.observe(history);
    idx
}

/// 2026-10-02: Deterministic xorshift, so a failure reproduces.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

// 2026-10-02: Part 1, what it must propose.

#[test]
fn copies_the_continuation_of_an_earlier_occurrence() {
    // 2026-10-02: [5 6 7] occurs at 1..4 and is followed by 8 9 10 11.
    let h = [1, 5, 6, 7, 8, 9, 10, 11, 2, 5, 6, 7];
    let idx = indexed(&h, 3);
    assert_eq!(idx.propose(&h, 3), Some(&[8, 9, 10][..]));
    assert_eq!(idx.propose(&h, 100), Some(&[8, 9, 10, 11, 2, 5, 6, 7][..]));
}

#[test]
fn prefers_the_latest_earlier_occurrence() {
    // 2026-10-02: [1 2] is followed by 3 at 0..2 and by 4 at 3..5.
    let h = [1, 2, 3, 1, 2, 4, 9, 1, 2];
    assert_eq!(indexed(&h, 2).propose(&h, 1), Some(&[4][..]));
}

#[test]
fn incremental_observe_equals_one_shot_observe() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let h: Vec<u32> = (0..400).map(|_| (rng.next() % 5) as u32).collect();
    let mut inc = PromptLookupIndex::new(3);
    for end in 0..=h.len() {
        inc.observe(&h[..end]);
        let one = indexed(&h[..end], 3);
        assert_eq!(
            inc.propose(&h[..end], 8),
            one.propose(&h[..end], 8),
            "at len {end}"
        );
    }
}

#[test]
fn matches_brute_force_on_random_histories() {
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    for case in 0..300 {
        let vocab = 2 + (rng.next() % 6) as u32;
        let len = (rng.next() % 120) as usize;
        let n = 1 + (rng.next() % 5) as usize;
        let max_len = (rng.next() % 20) as usize;
        let h: Vec<u32> = (0..len)
            .map(|_| (rng.next() % u64::from(vocab)) as u32)
            .collect();
        let got = indexed(&h, n).propose(&h, max_len).map(<[u32]>::to_vec);
        assert_eq!(
            got,
            brute_force(&h, n, max_len),
            "case {case}: n={n} max={max_len} h={h:?}"
        );
    }
}

#[test]
fn a_rewound_history_is_reindexed_with_its_new_tokens() {
    let mut idx = PromptLookupIndex::new(2);
    let mut h = vec![1, 2, 3, 4, 1, 2];
    idx.observe(&h);
    assert_eq!(idx.propose(&h, 1), Some(&[3][..]));
    // 2026-10-02: Rewind two tokens, then grow differently: [1 2] -> 7 is now
    // the latest occurrence.
    h.truncate(4);
    idx.observe(&h);
    h.extend([1, 2, 7, 5, 1, 2]);
    idx.observe(&h);
    assert_eq!(idx.propose(&h, 1), Some(&[7][..]));
    assert_eq!(
        idx.propose(&h, 1).map(<[u32]>::to_vec),
        brute_force(&h, 2, 1)
    );
}

// 2026-10-02: Part 2, what it must refuse.

#[test]
fn no_proposal_without_an_earlier_occurrence() {
    let h = [1, 2, 3, 4, 5, 6, 7, 8];
    assert_eq!(indexed(&h, 2).propose(&h, 4), None);
}

#[test]
fn never_proposes_from_the_suffix_itself() {
    // 2026-10-02: The only occurrence of [3 4] is the suffix.
    let h = [1, 2, 3, 4];
    assert_eq!(indexed(&h, 2).propose(&h, 4), None);
}

#[test]
fn a_stale_entry_after_a_rewind_is_never_used() {
    let mut idx = PromptLookupIndex::new(2);
    let h = [9, 9, 1, 2, 3, 1, 2];
    idx.observe(&h);
    // 2026-10-02: Rewind below the entry [1 2] -> 3 (position 4) and end on
    // [1 2] again with no earlier occurrence left in the kept history.
    let short = [9, 1, 2];
    idx.observe(&short);
    assert_eq!(idx.propose(&short, 4), None);
    assert_eq!(brute_force(&short, 2, 4), None);
}

#[test]
fn a_hash_hit_with_different_tokens_is_refused() {
    let h = [1, 2, 3, 4, 5, 6];
    let mut idx = indexed(&h, 2);
    // 2026-10-02: Plant a colliding entry: the key of the suffix [5 6] points
    // at a position whose preceding tokens differ.
    idx.latest.insert(ngram_key(&[5, 6]), 2);
    assert_eq!(idx.propose(&h, 4), None);
}

#[test]
fn a_full_window_that_misses_shrinks_and_a_partial_never_grows() {
    let mut w = CopyWindow::new(8, 2, 64);
    w.record(8, 7);
    assert_eq!(w.current(), 4);
    w.record(4, 0);
    assert_eq!(w.current(), 2);
}

// 2026-10-02: Part 3, boundaries.

#[test]
fn proposal_stops_at_max_len_and_at_the_history_end() {
    let h = [1, 2, 3, 1, 2];
    let idx = indexed(&h, 2);
    assert_eq!(idx.propose(&h, 0), None);
    assert_eq!(idx.propose(&h, 1), Some(&[3][..]));
    // 2026-10-02: Only 3 tokens follow the occurrence: 3 1 2.
    assert_eq!(idx.propose(&h, 50), Some(&[3, 1, 2][..]));
}

#[test]
fn histories_no_longer_than_n_propose_nothing() {
    for len in 0..=3 {
        let h: Vec<u32> = vec![7; len];
        assert_eq!(indexed(&h, 3).propose(&h, 4), None, "len {len}");
    }
    // 2026-10-02: One token longer is enough for a self-repeating run.
    let h = [7, 7, 7, 7];
    assert_eq!(indexed(&h, 3).propose(&h, 4), Some(&[7][..]));
}

#[test]
fn window_doubles_to_its_cap_and_halves_to_its_floor() {
    let mut w = CopyWindow::new(4, 2, 16);
    for want in [8, 16, 16] {
        w.record(w.current(), w.current());
        assert_eq!(w.current(), want);
    }
    for want in [8, 4, 2, 2] {
        w.record(w.current(), 0);
        assert_eq!(w.current(), want);
    }
    // 2026-10-02: An empty copy is not evidence either way.
    w.record(0, 0);
    assert_eq!(w.current(), 2);
}

#[test]
fn window_start_is_clamped_into_bounds() {
    assert_eq!(CopyWindow::new(1, 2, 8).current(), 2);
    assert_eq!(CopyWindow::new(99, 2, 8).current(), 8);
}

#[test]
#[should_panic(expected = "1 <= min <= max")]
fn window_rejects_inverted_bounds() {
    let _ = CopyWindow::new(4, 8, 2);
}

#[test]
#[should_panic(expected = "at least 1")]
fn index_rejects_zero_length_ngrams() {
    let _ = PromptLookupIndex::new(0);
}

// 2026-10-02: Per-sequence state.

const CFG: PromptLookupConfig = PromptLookupConfig {
    ngram: 2,
    max_drafts: 3,
    max_seqs: 8,
};

#[test]
fn seq_marks_a_copy_in_flight_and_settles_it() {
    let mut s = PromptLookupSeq::new(&CFG);
    let h = [1, 2, 3, 4, 5, 1, 2];
    assert_eq!(s.propose(&h, s.window()), Some(vec![3, 4, 5]));
    assert_eq!(s.in_flight(), 3);
    assert_eq!(s.settle(1), 3);
    assert_eq!(s.in_flight(), 0);
    // 2026-10-02: A partial accept halves the window: 3 -> 1.
    assert_eq!(s.window(), 1);
    assert_eq!(s.propose(&h, s.window()), Some(vec![3]));
    s.settle(1);
    assert_eq!(s.window(), 2);
}

#[test]
fn seq_without_a_match_has_nothing_in_flight() {
    let mut s = PromptLookupSeq::new(&CFG);
    let h = [1, 2, 3, 4, 5, 1, 2];
    assert!(s.propose(&h, 3).is_some());
    // 2026-10-02: A later miss must clear the earlier in-flight mark, or the
    // verdict would treat the drafter's own drafts as a copy.
    let miss = [1, 2, 3, 4, 5, 1, 2, 9, 9, 8];
    assert_eq!(s.propose(&miss, 3), None);
    assert_eq!(s.in_flight(), 0);
    assert_eq!(s.settle(0), 0);
    assert_eq!(s.window(), 3, "settling nothing must not move the window");
}

#[test]
fn seq_cap_bounds_the_copy_and_zero_cap_proposes_nothing() {
    let mut s = PromptLookupSeq::new(&CFG);
    let h = [1, 2, 3, 4, 5, 1, 2];
    assert_eq!(s.propose(&h, 1), Some(vec![3]));
    // 2026-10-02: The cap is the caller's; the window does not shorten it.
    assert_eq!(s.propose(&h, 5), Some(vec![3, 4, 5, 1, 2]));
    assert_eq!(s.propose(&h, 0), None);
    assert_eq!(s.in_flight(), 0);
}

#[test]
fn seq_abandon_clears_without_moving_the_window() {
    let mut s = PromptLookupSeq::new(&CFG);
    let h = [1, 2, 3, 4, 5, 1, 2];
    assert!(s.propose(&h, 3).is_some());
    s.abandon();
    assert_eq!(s.in_flight(), 0);
    assert_eq!(s.window(), 3);
}
