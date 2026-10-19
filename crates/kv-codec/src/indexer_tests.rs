// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: The indexer plan's byte count against the formula the FP8 indexer cache used
//! before it read the registry, and its refusals.

use super::*;

fn fp8_row() -> KvCodecId {
    by_name("fp8_row").expect("registered")
}

/// 2026-10-18: `E * (D + 4) + 2 * S * (KP - 1) * D * 2` with `E = bs / KP`, the hand-written
/// layout of `dsa_indexer_fp8.cu` (E4M3 entries, f32 scales, BF16 key and gate ring), at every
/// block size, pool and ring depth on a grid.
#[test]
fn block_bytes_match_the_kernel_layout() {
    let d = 128u32;
    for kp in [2u32, 3, 4, 8, 16, 64] {
        for sets in [1u32, 2, 3, 7] {
            let plan = IndexerPlan::new(fp8_row(), d, kp, sets).expect("valid geometry");
            for pools in [1u32, 2, 4, 16, 64] {
                let bs = pools * kp;
                let (e, d, kp, sets) = (u64::from(bs / kp), 128u64, u64::from(kp), u64::from(sets));
                let want = e * (d + 4) + 2 * sets * (kp - 1) * d * 2;
                assert_eq!(
                    plan.block_bytes(bs).ok(),
                    Some(want),
                    "bs {bs} kp {kp} sets {sets}"
                );
            }
        }
    }
}

/// 2026-10-18: The value the FP8 indexer's own test pins: 16 tokens, pools of 4, one ring set.
#[test]
fn the_pinned_block_is_2064_bytes() {
    let plan = IndexerPlan::new(fp8_row(), 128, 4, 1).expect("valid geometry");
    assert_eq!(plan.block_bytes(16).ok(), Some(2064));
}

/// 2026-10-18: Every geometry rule refuses with the indexer code, never a silent size.
#[test]
fn bad_geometry_is_refused() {
    let code = |r: Result<_, Error>| r.map(|_: IndexerPlan| ()).unwrap_err().code();
    let want = ErrorCode::KvIndexerGeometry;
    assert_eq!(code(IndexerPlan::new(fp8_row(), 64, 4, 1)), want);
    assert_eq!(code(IndexerPlan::new(fp8_row(), 128, 1, 1)), want);
    assert_eq!(code(IndexerPlan::new(fp8_row(), 128, 4, 0)), want);
    for attention in KvCodecId::ATTENTION {
        assert_eq!(code(IndexerPlan::new(attention, 128, 4, 1)), want);
    }
    let plan = IndexerPlan::new(fp8_row(), 128, 4, 1).expect("valid geometry");
    for bs in [0u32, 3, 5, 18] {
        assert_eq!(plan.block_bytes(bs).unwrap_err().code(), want, "bs {bs}");
    }
}
