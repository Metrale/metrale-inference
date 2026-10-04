// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: The cross-request prompt-lookup cache through the real
//! scheduler loop, against the recording model. Path A: a request copies what
//! an earlier request of its tenant produced, with fewer verifies and the same
//! client bytes. Path B: across tenants, or with no tenant, nothing is copied
//! (the verify count equals the cache off). Boundaries and determinism are
//! the index's own tests (`metrale_speculative::shared_lookup`).
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use std::sync::Mutex;

use metrale_speculative::prompt_lookup::PromptLookupConfig;
use metrale_speculative::shared_lookup::SharedLookupConfig;

use super::model::ModelCfg;
use super::runner::{EOS, ReqSpec, RunOptions, Scenario, run_scenario};
use crate::auth::LookupTenant;
use crate::scheduler::shared_lookup_step::SharedLookupSetup;

static SERIAL: Mutex<()> = Mutex::new(());

const PL: PromptLookupConfig = PromptLookupConfig {
    ngram: 2,
    max_drafts: 3,
    max_seqs: 8,
    min_match: 2,
    miss_backoff: 0,
};

const SHARED: SharedLookupSetup = SharedLookupSetup {
    config: SharedLookupConfig {
        budget_bytes: 1 << 20,
        max_scopes: 4,
        key_len: 2,
    },
    model: 11,
    tokenizer: 12,
};

fn traced(sc: &Scenario) -> Vec<String> {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    run_scenario(sc)
}

/// 2026-10-04: Tokens no prompt contains (the harness prompt is the period
/// `2..=8` after its first token; the fake's vocabulary is 64), so only
/// another request can predict them.
fn novel() -> Vec<u32> {
    (40..56).collect()
}

/// 2026-10-04: Request 1 emits [`novel`] and finishes first; request 2 emits a
/// different preamble, then the same run. Both drafters miss every 2nd token.
fn pair(t1: Option<LookupTenant>, t2: Option<LookupTenant>) -> Vec<ReqSpec> {
    let mut first = novel();
    first.push(EOS);
    let mut second: Vec<u32> = (20..38).collect();
    second.extend(novel());
    second.push(EOS);
    [(1, first, t1), (2, second, t2)]
        .into_iter()
        .map(|(id, toks, tenant)| {
            let mut r = ReqSpec::new(id, 15, toks);
            r.draft_wrong_every = Some(2);
            r.lookup_tenant = tenant;
            r
        })
        .collect()
}

fn scenario(reqs: Vec<ReqSpec>, shared: Option<SharedLookupSetup>) -> Scenario {
    Scenario {
        name: "shared_lookup",
        cfg: ModelCfg {
            has_proposer: true,
            ..ModelCfg::default()
        },
        opts: RunOptions {
            use_speculative: true,
            num_drafts: 3,
            prompt_lookup: Some(PL),
            shared_lookup: shared,
            ..RunOptions::default()
        },
        reqs,
    }
}

/// 2026-10-04: What each client received, without the accepted-draft count.
fn outputs(lines: &[String]) -> Vec<String> {
    lines
        .iter()
        .filter(|l| l.starts_with("out s"))
        .map(|l| {
            l.split(", ")
                .filter(|f| !f.starts_with("acc="))
                .collect::<Vec<_>>()
                .join(", ")
        })
        .collect()
}

fn verifies(lines: &[String]) -> usize {
    lines
        .iter()
        .filter(|l| l.starts_with("decode_verify"))
        .count()
}

const T1: Option<LookupTenant> = Some(LookupTenant(1));
const T2: Option<LookupTenant> = Some(LookupTenant(2));

#[test]
fn copies_an_earlier_request_of_the_same_tenant() {
    let off = traced(&scenario(pair(T1, T1), None));
    let on = traced(&scenario(pair(T1, T1), Some(SHARED)));
    assert_eq!(outputs(&on), outputs(&off));
    assert!(
        verifies(&on) < verifies(&off),
        "shared cache verified {} times, own lookup alone {}",
        verifies(&on),
        verifies(&off)
    );
}

#[test]
fn never_copies_across_tenants_or_without_one() {
    let off = traced(&scenario(pair(T1, T2), None));
    for (t1, t2) in [(T1, T2), (None, None), (T1, None), (None, T1)] {
        let on = traced(&scenario(pair(t1, t2), Some(SHARED)));
        assert_eq!(outputs(&on), outputs(&off), "{t1:?} {t2:?}");
        assert_eq!(verifies(&on), verifies(&off), "{t1:?} -> {t2:?} copied");
    }
}
