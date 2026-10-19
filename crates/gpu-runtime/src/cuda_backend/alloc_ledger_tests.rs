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

/// 2026-10-19: A label names exactly one live base pointer and leaves the ledger with it: an
/// interior or unknown pointer takes none, and once freed the address answers `None` (a later
/// allocation at a reused address must not inherit a stale name).
#[test]
#[ignore = "requires a free CUDA device"]
fn a_label_lives_and_dies_with_its_allocation() {
    use crate::gpu::{DevicePtr, GpuBackend};
    let gpu = MetraleCudaBackend::new(0, &[]).expect("CUDA device 0");
    let p = GpuBackend::alloc(&gpu, 4 * MIB).unwrap();
    assert!(gpu.label_alloc(p, "first"));
    assert!(gpu.label_alloc(p, "second"));
    assert_eq!(gpu.alloc_label(p).as_deref(), Some("second"));
    assert!(!gpu.label_alloc(DevicePtr(p.0 + 256), "interior"));
    assert_eq!(gpu.alloc_label(DevicePtr(p.0 + 256)), None);
    GpuBackend::free(&gpu, p).unwrap();
    assert_eq!(gpu.alloc_label(p), None);
    assert!(!gpu.label_alloc(p, "freed"));
}
