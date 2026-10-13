// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-13: The KV budget's two bounds and the driver credit, at the figures an H100 serve of
//! Qwen3.8-Flash-Next logged at KV sizing (1 slot, util 0.99): the reserve's driver terms are
//! charged once, not again for the driver use free memory already shows; the util budget still
//! binds; and nothing is credited without a baseline, a ledger, or on a class that credits none.

use super::{incurred_driver_credit, kv_budget_bound};

const MIB: usize = 1 << 20;

/// 2026-10-13: cuMemGetInfo total, 81,079 MiB.
const TOTAL: usize = 81_079 * MIB;
/// 2026-10-13: Free at context init: the total less the bare context's 527 MiB.
const BASELINE: usize = TOTAL - 527 * MIB;
/// 2026-10-13: The allocation ledger at KV sizing ("73.9 GB live").
const LEDGER: usize = 75_674 * MIB;
/// 2026-10-13: Small-allocation chunk slack ("0.05 GB").
const SLACK: usize = 51 * MIB;
/// 2026-10-13: Used beyond the ledger at KV sizing ("1.5 GB ... excluded"): context and driver.
const UNTRACKED: usize = 1_536 * MIB;
const ACTUAL_FREE: usize = TOTAL - LEDGER - UNTRACKED;
/// 2026-10-13: The reserve at 1 slot (2,940 MB) and its driver terms (1,024 + 1,284 MB).
const RESERVE: usize = 2_940 * MIB;
const DRIVER_TERMS: usize = 2_308 * MIB;

fn budget() -> usize {
    (TOTAL as f64 * 0.99) as usize
}

#[test]
fn driver_use_since_init_beyond_the_ledger_and_slack_is_credited_up_to_the_cap() {
    let credit = incurred_driver_credit(
        Some(BASELINE),
        ACTUAL_FREE,
        Some(LEDGER),
        SLACK,
        DRIVER_TERMS,
    );
    assert_eq!(credit, UNTRACKED - 527 * MIB - SLACK);
    let capped =
        incurred_driver_credit(Some(BASELINE), ACTUAL_FREE, Some(LEDGER), SLACK, 100 * MIB);
    assert_eq!(capped, 100 * MIB);
}

#[test]
fn no_baseline_no_ledger_or_a_zero_cap_credits_nothing() {
    assert_eq!(
        incurred_driver_credit(None, ACTUAL_FREE, Some(LEDGER), SLACK, DRIVER_TERMS),
        0
    );
    assert_eq!(
        incurred_driver_credit(Some(BASELINE), ACTUAL_FREE, None, SLACK, DRIVER_TERMS),
        0
    );
    assert_eq!(
        incurred_driver_credit(Some(BASELINE), ACTUAL_FREE, Some(LEDGER), SLACK, 0),
        0
    );
    // 2026-10-13: A ledger above what the device shows consumed credits nothing, never wraps.
    assert_eq!(
        incurred_driver_credit(Some(BASELINE), ACTUAL_FREE, Some(TOTAL), 0, DRIVER_TERMS),
        0
    );
}

#[test]
fn without_a_credit_free_memory_binds_and_charges_the_driver_twice() {
    let used = LEDGER + SLACK;
    let kv = kv_budget_bound(budget(), used, ACTUAL_FREE, RESERVE, 0, 0);
    assert_eq!(kv, ACTUAL_FREE - RESERVE);
    assert!(kv < budget() - used - RESERVE);
}

#[test]
fn with_the_credit_the_util_budget_binds_on_the_logged_serve() {
    let used = LEDGER + SLACK;
    let credit = incurred_driver_credit(
        Some(BASELINE),
        ACTUAL_FREE,
        Some(LEDGER),
        SLACK,
        DRIVER_TERMS,
    );
    let kv = kv_budget_bound(budget(), used, ACTUAL_FREE, RESERVE, credit, 0);
    assert_eq!(kv, budget() - used - RESERVE);
    // 2026-10-13: About 0.65 GiB more KV than the double charge left (0.9 -> 1.55 GiB).
    let before = kv_budget_bound(budget(), used, ACTUAL_FREE, RESERVE, 0, 0);
    assert!(kv - before > 600 * MIB, "{}", (kv - before) / MIB);
    // 2026-10-13: The pool and the reserve's undrawn part still fit in the free memory.
    assert!(kv + RESERVE - credit <= ACTUAL_FREE);
}

#[test]
fn a_co_tenant_after_init_reads_as_driver_use_and_the_pool_still_fits_free_memory() {
    let used = LEDGER + SLACK;
    let free = ACTUAL_FREE - 2_048 * MIB;
    let credit = incurred_driver_credit(Some(BASELINE), free, Some(LEDGER), SLACK, DRIVER_TERMS);
    assert_eq!(credit, DRIVER_TERMS);
    let kv = kv_budget_bound(budget(), used, free, RESERVE, credit, 0);
    assert_eq!(kv, free - (RESERVE - DRIVER_TERMS));
    assert!(kv + RESERVE - credit <= free);
    assert!(kv < budget() - used - RESERVE);
}

#[test]
fn the_dflash_reserve_comes_off_both_bounds() {
    let used = LEDGER + SLACK;
    let d = 64 * MIB;
    let credit = incurred_driver_credit(
        Some(BASELINE),
        ACTUAL_FREE,
        Some(LEDGER),
        SLACK,
        DRIVER_TERMS,
    );
    assert_eq!(
        kv_budget_bound(budget(), used, ACTUAL_FREE, RESERVE, credit, d),
        budget() - used - RESERVE - d
    );
    assert_eq!(
        kv_budget_bound(budget(), used, ACTUAL_FREE, RESERVE, 0, d),
        ACTUAL_FREE - RESERVE - d
    );
}
