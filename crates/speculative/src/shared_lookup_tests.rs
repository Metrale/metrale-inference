// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: Tests for the cross-request token cache, in three parts: what
//! it must propose (path A), what it must refuse, isolation included (path B),
//! and its boundaries, eviction and memory (path C). A brute-force reference
//! checks it on random request histories.

use super::*;
use crate::prompt_lookup::{PromptLookupConfig, PromptLookupSeq};

const A: ScopeKey = ScopeKey {
    model: 1,
    tokenizer: 2,
    adapter: 0,
    tenant: 7,
};

fn cfg(budget_bytes: usize, max_scopes: usize, key_len: usize) -> SharedLookupConfig {
    SharedLookupConfig {
        budget_bytes,
        max_scopes,
        key_len,
    }
}

/// 2026-10-04: A cache large enough that nothing is evicted in these tests.
fn big(key_len: usize) -> SharedTokenCache {
    SharedTokenCache::new(cfg(1 << 22, 1, key_len)).unwrap()
}

/// 2026-10-04: Deterministic xorshift, so a failure reproduces.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn doc(&mut self, len: usize, vocab: u64) -> Vec<u32> {
        (0..len).map(|_| (self.next() % vocab) as u32).collect()
    }
}

// 2026-10-04: Path A, what it must propose.

#[test]
fn copies_a_continuation_from_an_earlier_request() {
    let mut c = big(3);
    c.insert(A, &[9, 1, 2, 3, 40, 41, 42, 43]);
    // 2026-10-04: A different request ends in [1 2 3].
    assert_eq!(c.propose(&A, &[5, 5, 1, 2, 3], 3), Some(vec![40, 41, 42]));
    assert_eq!(
        c.propose(&A, &[5, 5, 1, 2, 3], 100),
        Some(vec![40, 41, 42, 43])
    );
}

#[test]
fn prefers_the_latest_request() {
    let mut c = big(2);
    c.insert(A, &[1, 2, 3, 3, 3]);
    c.insert(A, &[8, 1, 2, 4, 4]);
    assert_eq!(c.propose(&A, &[1, 2], 2), Some(vec![4, 4]));
}

#[test]
fn matches_the_reference_on_random_request_histories() {
    let mut rng = Rng(0x5eed_1234_abcd_0001);
    for round in 0..40 {
        let k = 1 + (round % 4);
        let mut c = big(k);
        let docs: Vec<Vec<u32>> = (0..6)
            .map(|_| {
                let len = 30 + (rng.next() % 50) as usize;
                rng.doc(len, 6)
            })
            .collect();
        let mut latest: HashMap<Vec<u32>, u32> = HashMap::new();
        for d in &docs {
            c.insert(A, d);
            for e in k..d.len() {
                latest.insert(d[e - k..e].to_vec(), d[e]);
            }
        }
        for _ in 0..60 {
            let len = k + (rng.next() % 4) as usize;
            let h = rng.doc(len, 6);
            let key = h[h.len() - k..].to_vec();
            let got = c.propose(&A, &h, 6);
            // 2026-10-04: The first token is the latest request's next token.
            assert_eq!(
                got.as_ref().map(|g| g[0]),
                latest.get(&key).copied(),
                "k={k} key={key:?}"
            );
            // 2026-10-04: The whole copy is a slice of one request after an
            // occurrence of the key.
            if let Some(g) = got {
                let sound = docs.iter().any(|d| {
                    (k..d.len()).any(|e| d[e - k..e] == key[..] && d[e..].starts_with(&g))
                });
                assert!(sound, "copy {g:?} for key {key:?} is in no request");
            }
        }
    }
}

#[test]
fn every_copy_is_sound_when_slots_collide_and_the_ring_wraps() {
    // 2026-10-04: 1024 slots for up to 6^4 keys, and a 2048-token ring under
    // ~6000 stored tokens: hits must be re-checked, stale ones refused.
    let mut rng = Rng(0xc011_1de5);
    for k in 1..=4 {
        let config = cfg(16 * 1024, 1, k);
        let mut c = SharedTokenCache::new(config).unwrap();
        let docs: Vec<Vec<u32>> = (0..80).map(|_| rng.doc(80, 6)).collect();
        let mut hits = 0;
        for (i, d) in docs.iter().enumerate() {
            c.insert(A, d);
            let h = rng.doc(k, 6);
            if let Some(g) = c.propose(&A, &h, 6) {
                hits += 1;
                let sound = docs[..=i]
                    .iter()
                    .any(|d| (k..d.len()).any(|e| d[e - k..e] == h[..] && d[e..].starts_with(&g)));
                assert!(sound, "k={k}: copy {g:?} for key {h:?} is in no request");
            }
        }
        assert!(
            hits > 20,
            "k={k}: only {hits} hits, the check proves little"
        );
    }
}

#[test]
fn a_repeated_request_costs_no_ring_space() {
    let mut c = big(4);
    let doc: Vec<u32> = (0..500).collect();
    c.insert(A, &doc);
    let once = c.stored_tokens(&A);
    c.insert(A, &doc);
    assert_eq!(c.stored_tokens(&A), once);
    // 2026-10-04: A conversation re-sent with a new turn stores about the turn.
    let mut longer = doc.clone();
    longer.extend(1000..1100);
    c.insert(A, &longer);
    let added = c.stored_tokens(&A) - once;
    assert!(
        added <= 100 + 4 + 1,
        "stored {added} tokens for a 100-token turn"
    );
    assert_eq!(
        c.propose(&A, &[496, 497, 498, 499], 3),
        Some(vec![1000, 1001, 1002])
    );
}

#[test]
fn own_history_first_then_the_shared_cache() {
    let pl_cfg = PromptLookupConfig {
        ngram: 2,
        max_drafts: 8,
        max_seqs: 8,
        min_match: 2,
        miss_backoff: 0,
    };
    let mut seq = PromptLookupSeq::new(&pl_cfg);
    let mut c = big(2);
    c.insert(A, &[1, 2, 70, 71, 72]);
    // 2026-10-04: No own match: the shared copy, clipped to max_len, is in flight.
    let got = seq.propose_with(&[5, 1, 2], 2, |h, m| c.propose(&A, h, m));
    assert_eq!(got, Some(vec![70, 71]));
    assert!(seq.in_flight_shared());
    assert_eq!(seq.settle(2), 2);
    assert!(!seq.in_flight_shared());
    // 2026-10-04: An own match wins over the shared one (a fresh sequence:
    // the index assumes each history extends the last).
    let mut seq = PromptLookupSeq::new(&pl_cfg);
    let own = seq.propose_with(&[1, 2, 9, 9, 1, 2], 2, |h, m| c.propose(&A, h, m));
    assert_eq!(own, Some(vec![9, 9]));
    assert!(!seq.in_flight_shared());
}

// 2026-10-04: Path B, what it must refuse.

#[test]
fn never_crosses_any_scope_field() {
    let others = [
        ScopeKey { model: 9, ..A },
        ScopeKey { tokenizer: 9, ..A },
        ScopeKey { adapter: 9, ..A },
        ScopeKey { tenant: 9, ..A },
    ];
    let mut c = SharedTokenCache::new(cfg(1 << 22, 8, 3)).unwrap();
    c.insert(A, &[1, 2, 3, 4, 5, 6]);
    for o in others {
        assert_eq!(c.propose(&o, &[1, 2, 3], 3), None, "{o:?} read scope A");
        // 2026-10-04: Another scope's insert neither replaces nor evicts A's.
        c.insert(o, &[1, 2, 3, 66, 66, 66]);
        assert_eq!(c.propose(&A, &[1, 2, 3], 3), Some(vec![4, 5, 6]));
    }
}

#[test]
fn refuses_without_a_match() {
    let mut c = big(3);
    assert_eq!(c.propose(&A, &[1, 2, 3], 3), None, "empty cache");
    c.insert(A, &[1, 2, 3, 4]);
    assert_eq!(c.propose(&A, &[0, 2, 3], 3), None, "partial key");
    assert_eq!(c.propose(&A, &[1, 2, 3], 0), None, "max_len 0");
    assert_eq!(
        c.propose(&A, &[2, 3], 3),
        None,
        "history shorter than the key"
    );
}

#[test]
fn a_document_no_longer_than_the_key_is_not_stored() {
    let mut c = big(3);
    c.insert(A, &[1, 2, 3]);
    assert_eq!(c.scope_count(), 0);
    assert_eq!(c.stored_tokens(&A), 0);
}

#[test]
fn proposes_nothing_from_the_cooldown_or_a_shared_miss() {
    let pl_cfg = PromptLookupConfig {
        ngram: 2,
        max_drafts: 8,
        max_seqs: 8,
        min_match: 2,
        miss_backoff: 4,
    };
    let mut seq = PromptLookupSeq::new(&pl_cfg);
    let mut asked = 0;
    assert_eq!(
        seq.propose_with(&[1, 2], 3, |_, _| {
            asked += 1;
            Some(vec![])
        }),
        None
    );
    assert_eq!(seq.in_flight(), 0, "an empty shared copy is no copy");
    assert_eq!(
        seq.propose_with(&[1, 2], 3, |_, _| Some(vec![5])),
        Some(vec![5])
    );
    seq.settle(0);
    // 2026-10-04: The miss starts a cooldown; the cache is not asked.
    assert_eq!(
        seq.propose_with(&[1, 2], 3, |_, _| {
            asked += 1;
            Some(vec![5])
        }),
        None
    );
    assert_eq!(asked, 1);
}

// 2026-10-04: Path C, boundaries, eviction and memory.

#[test]
fn a_copy_stops_at_its_request_end() {
    let mut c = big(2);
    c.insert(A, &[1, 2, 3]);
    c.insert(A, &[4, 5, 6]);
    assert_eq!(c.propose(&A, &[1, 2], 10), Some(vec![3]));
}

#[test]
fn the_ring_evicts_the_oldest_requests() {
    let config = cfg(16 * 1024, 1, 3);
    let (_, ring) = config.scope_geometry().unwrap();
    let mut c = SharedTokenCache::new(config).unwrap();
    c.insert(A, &[1, 2, 3, 4, 5]);
    assert_eq!(c.propose(&A, &[1, 2, 3], 2), Some(vec![4, 5]));
    let mut rng = Rng(77);
    while c.stored_tokens(&A) < 2 * ring as u64 {
        c.insert(
            A,
            &rng.doc(200, 1 << 20)
                .iter()
                .map(|t| t + 100)
                .collect::<Vec<_>>(),
        );
    }
    assert_eq!(
        c.propose(&A, &[1, 2, 3], 2),
        None,
        "overwritten request still proposed"
    );
}

#[test]
fn the_least_recently_used_scope_is_dropped_whole() {
    let b = ScopeKey { tenant: 8, ..A };
    let d = ScopeKey { tenant: 9, ..A };
    let mut c = SharedTokenCache::new(cfg(1 << 20, 2, 2)).unwrap();
    c.insert(A, &[1, 2, 3]);
    c.insert(b, &[1, 2, 4]);
    assert_eq!(c.propose(&A, &[1, 2], 1), Some(vec![3]), "touches A");
    c.insert(d, &[1, 2, 5]);
    assert_eq!(c.scope_count(), 2);
    assert_eq!(c.propose(&b, &[1, 2], 1), None, "b was least recently used");
    assert_eq!(c.propose(&A, &[1, 2], 1), Some(vec![3]));
    assert_eq!(c.propose(&d, &[1, 2], 1), Some(vec![5]));
}

#[test]
fn memory_is_fixed_per_scope_and_within_the_budget() {
    for (budget, scopes) in [(1usize << 20, 1usize), (1 << 24, 3), (5 << 20, 16)] {
        let config = cfg(budget, scopes, 8);
        let per = config.scope_bytes().unwrap();
        assert!(per * scopes <= budget, "{per} x {scopes} > {budget}");
        assert!(
            per * scopes * 4 >= budget * 3,
            "{per} x {scopes} wastes over a quarter of {budget}"
        );
        let mut c = SharedTokenCache::new(config).unwrap();
        c.insert(A, &(0..100_000).collect::<Vec<u32>>());
        let s = &c.scopes[&A];
        assert_eq!(s.ring.len() * 4 + s.slots.len() * 8, per);
    }
    assert!(
        cfg(1 << 20, 512, 8).scope_geometry().is_err(),
        "2 KiB scopes"
    );
    assert!(cfg(1 << 20, 0, 8).scope_geometry().is_err());
}

#[test]
fn same_history_same_drafts() {
    let run = || {
        let mut c = SharedTokenCache::new(cfg(64 * 1024, 2, 3)).unwrap();
        let mut rng = Rng(4242);
        let mut out = Vec::new();
        for i in 0..200u64 {
            let scope = ScopeKey { tenant: i % 3, ..A };
            c.insert(scope, &rng.doc(60, 5));
            out.push(c.propose(&scope, &rng.doc(5, 5), 4));
        }
        out
    };
    let first = run();
    assert!(first.iter().any(Option::is_some));
    assert_eq!(first, run());
}
