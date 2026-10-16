// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-16: Tests for the cache-state rule (`CacheState`): an absent cached
//! count is not a 0, a warmed cell that restored nothing is labelled
//! cache-cold and counted, and none of it changes the sweep verdict. A child
//! module of `concurrency_tests`, so it shares that module's fixture builders.
//!
//! Owner: bench (concurrency).
//! Invariants: none beyond the types.

use super::super::verdict::{Exclusions, Floors, sweep_verdict};
use super::*;
use crate::result::Verdict;

/// 2026-10-16: A request of a `prompt`-token prompt with `cached` reported.
fn req(prompt: usize, cached: Option<u64>) -> RequestEvidence {
    let mut r = evidence(128);
    r.prompt_tokens = prompt;
    r.cached_prompt_tokens = cached;
    r
}

/// 2026-10-16: A cell of `n` such requests, all alike.
fn cell(n: usize, prompt: usize, cached: Option<u64>) -> Vec<RequestEvidence> {
    vec![req(prompt, cached); n]
}

#[test]
fn only_a_plan_that_warms_every_measured_prompt_counts_as_warmed() {
    assert!(prompt_plan(4, 1, Fixture::Essay).measured_prompts_warmed());
    assert!(prompt_plan(1, 2, Fixture::Natural).measured_prompts_warmed());
    // 2026-10-16: No warm-up round: no shared prefix, so the rule is off.
    assert!(!prompt_plan(4, 0, Fixture::Essay).measured_prompts_warmed());
    // 2026-10-16: A warm-up round that skipped a measured prompt did not warm
    // the cell.
    let mut partial = prompt_plan(4, 1, Fixture::Essay);
    partial.warmup_rounds[0].pop();
    assert!(!partial.measured_prompts_warmed());
}

#[test]
fn a_warm_control_is_warm_and_a_uniformly_cold_warmed_cell_is_cold() {
    // 2026-10-16: The control: a 192-token prefix restored in full.
    assert_eq!(
        cache_state(&cell(4, 192, Some(192)), true),
        CacheState::Warm
    );
    // 2026-10-16: The same warmed cell restoring nothing: cold, not controlled.
    assert_eq!(cache_state(&cell(4, 192, Some(0)), true), CacheState::Cold);
    // 2026-10-16: A uniform partial hit below `WARM_CACHE_FLOOR` (the chat
    // template only) is cold too.
    assert_eq!(cache_state(&cell(4, 192, Some(32)), true), CacheState::Cold);
    // 2026-10-16: A count the server did not send on some requests is not
    // warm, so the rest decide; it never turns a cold cell warm.
    assert_eq!(
        cache_state(&[req(192, Some(0)), req(192, None)], true),
        CacheState::Cold
    );
}

#[test]
fn the_rule_is_unchanged_without_a_warm_up_and_for_a_mixture() {
    // 2026-10-16: No warm-up round: whatever the counts, not judged.
    for cached in [Some(0), Some(192), None] {
        assert_eq!(
            cache_state(&cell(2, 192, cached), false),
            CacheState::NotWarmed
        );
    }
    // 2026-10-16: Warm and cold requests in one cell stay uncontrolled, as
    // before; so does an unreported request beside a warm one.
    assert_eq!(
        cache_state(&[req(192, Some(192)), req(192, Some(0))], true),
        CacheState::Mixed
    );
    assert_eq!(
        cache_state(&[req(192, Some(192)), req(192, None)], true),
        CacheState::Mixed
    );
    // 2026-10-16: A warmed cell with no request was uncontrolled before; it
    // still is.
    assert_eq!(cache_state(&[], true), CacheState::Mixed);
}

#[test]
fn a_cell_with_no_reported_count_is_not_flagged() {
    // 2026-10-16: A server that never sends `cached_tokens` gives nothing to
    // judge. Reading its silence as 0 would label every cell cache-cold.
    assert_eq!(
        cache_state(&cell(4, 192, None), true),
        CacheState::Unreported
    );
    let requests = [req(192, None), req(192, None)];
    assert_eq!(cache::min_cached_prompt(&requests), None);
    assert_eq!(cache::min_cached_prompt_pct(&requests), None);
    assert!(evidence_line(192, 2, &requests).contains("cached [unreported/192,unreported/192]"));
}

/// 2026-10-16: A sweep holding one cell of `requests`, flagged as `run_cell`
/// flags a warm-capable, warmed cell.
fn sweep_of(requests: Vec<RequestEvidence>) -> ConcurrencySweep {
    let mut b = configured(vec![4], vec![512]);
    let mut r = row(4, 100.0, Some(50.0), requests, 128);
    let state = cache_state(&r.requests, true);
    r.cache_uncontrolled = state == CacheState::Mixed;
    r.cache_cold = state == CacheState::Cold;
    b.rows.push(r);
    b
}

#[test]
fn cache_cold_cells_counts_cold_cells_and_absent_counts_stay_absent() {
    let cold = sweep_of(vec![req(512, Some(0)); 4]);
    let m = cold.metrics();
    assert_eq!(m.get("cache_cold_cells"), Some(&1.0));
    assert_eq!(m.get("cache_uncontrolled_cells"), Some(&0.0));
    assert_eq!(m.get("min_cached_prompt_tokens"), Some(&0.0));
    assert!(cold.rows[0].comparable());

    let warm = sweep_of(vec![req(512, Some(512)); 4]);
    assert_eq!(warm.metrics().get("cache_cold_cells"), Some(&0.0));

    // 2026-10-16: One unreported request leaves the sweep minimum unknown:
    // the key is absent, never 0.
    let mut quiet = vec![req(512, Some(512)); 3];
    quiet.push(req(512, None));
    let quiet = sweep_of(quiet);
    let m = quiet.metrics();
    assert!(!m.contains_key("min_cached_prompt_tokens"), "{m:?}");
    assert!(!m.contains_key("min_cached_prompt_pct"), "{m:?}");
    assert_eq!(quiet.table().rows[0][10].text, "unreported");
}

/// 2026-10-16: The sweep verdict as `next()` assembles it.
fn verdict_of(b: &ConcurrencySweep, metrics: &BTreeMap<String, f64>, floors: &Floors) -> Verdict {
    sweep_verdict(
        metrics,
        b.rows.len(),
        0,
        Exclusions {
            vacuous: b.rows.iter().filter(|r| r.vacuous).count(),
            cache_uncontrolled: b.rows.iter().filter(|r| r.cache_uncontrolled).count(),
            non_mtp_arm: b.rows.iter().filter(|r| r.arm_is_not_mtp()).count(),
        },
        VACUITY_FLOOR * 100.0,
        floors,
    )
}

#[test]
fn a_cold_by_design_cell_keeps_its_verdict_byte_for_byte() {
    // 2026-10-16: A prefix-cold cell like the required Qwen ladders' (every
    // request reports 0 cached after a warm-up). Before this rule it was
    // controlled and comparable. It still is, and the verdict, gating or
    // not, reads exactly as it did: the only change is the
    // `cache_cold_cells` key, which no verdict reads.
    let b = sweep_of(vec![req(128, Some(0)); 4]);
    assert!(!b.rows[0].cache_uncontrolled);
    let with_key = b.metrics();
    let mut without_key = with_key.clone();
    assert_eq!(without_key.remove("cache_cold_cells"), Some(1.0));
    for floors in [
        Floors::default(),
        Floors {
            per_c: vec![(4, 90.0)],
            peak: 0.0,
        },
        Floors {
            per_c: vec![(4, 110.0)],
            peak: 0.0,
        },
    ] {
        let new = verdict_of(&b, &with_key, &floors);
        let old = verdict_of(&b, &without_key, &floors);
        assert_eq!((new.kind, &new.reason), (old.kind, &old.reason));
    }
    assert_eq!(
        verdict_of(&b, &with_key, &Floors::default()).reason,
        "1 cells, no request errors, all above the vacuity floor"
    );
    assert_eq!(
        verdict_of(
            &b,
            &with_key,
            &Floors {
                per_c: vec![(4, 90.0)],
                peak: 0.0
            }
        )
        .reason,
        "1 cells, zero errors, zero vacuous — every populated floor met (C4 100.0/90.0)"
    );
}
