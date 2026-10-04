// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: What `SharedLookup::store_finished` stores: the prompt and the
//! emitted tokens of finished, error-free sequences, and nothing else.

use super::*;
use crate::scheduler::test_support::active_seq;

fn cache() -> SharedLookup {
    SharedLookup::new(SharedLookupSetup {
        config: SharedLookupConfig {
            budget_bytes: 1 << 20,
            max_scopes: 4,
            key_len: 2,
        },
        model: 1,
        tokenizer: 2,
    })
    .unwrap()
}

fn seq(finished: bool, error: Option<&str>, out: Vec<u32>) -> ActiveSeq {
    let (mut a, _rx) = active_seq(0, 0);
    a.seq.tokens = vec![1, 2, 3, 4];
    a.seq.prompt_len = 4;
    a.output_tokens = out;
    a.finished = finished;
    a.error = error.map(str::to_owned);
    a.seq.lookup_tenant = Some(1);
    a
}

#[test]
fn stores_the_prompt_then_the_emitted_tokens() {
    let sl = cache();
    let a = seq(true, None, vec![10, 11, 12]);
    sl.store_finished(std::slice::from_ref(&a));
    let scope = sl.scope(&a.seq).unwrap();
    assert_eq!(sl.propose(&scope, &[3, 4], 3), Some(vec![10, 11, 12]));
}

#[test]
fn skips_failed_and_unfinished_sequences() {
    let sl = cache();
    let seqs = [
        seq(true, Some("device fault"), vec![20, 21, 22]),
        seq(false, None, vec![30, 31, 32]),
    ];
    sl.store_finished(&seqs);
    let scope = sl.scope(&seqs[0].seq).unwrap();
    assert_eq!(
        sl.propose(&scope, &[4, 20], 2),
        None,
        "a failed sequence was stored"
    );
    assert_eq!(
        sl.propose(&scope, &[4, 30], 2),
        None,
        "an unfinished sequence was stored"
    );
}

#[test]
fn a_sequence_without_a_tenant_has_no_scope() {
    let mut a = seq(true, None, vec![10, 11]);
    a.seq.lookup_tenant = None;
    assert_eq!(cache().scope(&a.seq), None);
}
