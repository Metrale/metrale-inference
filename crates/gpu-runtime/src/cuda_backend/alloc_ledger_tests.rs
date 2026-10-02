// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: `chunk_slack` on hand-placed allocations: packed small ones, one that straddles a
//! chunk boundary, large ones that count nothing, and the empty ledger.
//!
//! Owner: gpu-runtime (CUDA backend).
//! Invariants: none beyond the types.

use super::*;

const C: u64 = SMALL_ALLOC_CHUNK;
const MIB: usize = 1 << 20;

#[test]
fn small_allocations_in_one_chunk_leave_the_rest_of_it() {
    let allocs = [(10 * C, 512 * 1024), (10 * C + 512 * 1024, 256 * 1024)];
    assert_eq!(chunk_slack(allocs.into_iter(), C), 2 * MIB - 768 * 1024);
}

#[test]
fn every_chunk_a_small_allocation_touches_counts_once() {
    // 2026-10-01: Two chunks with one allocation each, and one that straddles chunks 7 and 8.
    let allocs = [(3 * C, MIB), (5 * C + 4096, 4096), (8 * C - C / 4, MIB)];
    let small: usize = MIB + 4096 + MIB;
    assert_eq!(chunk_slack(allocs.into_iter(), C), 4 * 2 * MIB - small);
}

#[test]
fn allocations_of_a_chunk_or_more_count_nothing() {
    let allocs = [(0, 2 * MIB), (4 * C, 3 * MIB + 1), (20 * C, 64 * MIB)];
    assert_eq!(chunk_slack(allocs.into_iter(), C), 0);
    // 2026-10-01: A large allocation beside a small one in the same range adds nothing either.
    let mixed = [(30 * C, 3 * MIB), (32 * C, 4096)];
    assert_eq!(chunk_slack(mixed.into_iter(), C), 2 * MIB - 4096);
}

#[test]
fn no_allocations_no_slack() {
    assert_eq!(chunk_slack(std::iter::empty(), C), 0);
}
