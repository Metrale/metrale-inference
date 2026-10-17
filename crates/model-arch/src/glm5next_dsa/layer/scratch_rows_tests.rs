// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: `select_scratch_rows` and the workspace it sizes: unchanged up to
//! `SELECT_SCRATCH_FULL_ROWS_CONTEXT`, capped beyond it (the 262,144-token claim context at
//! 2,048 prefill rows: 256 rows, 7.5 GiB -> 0.9 GiB over 11 layers), never below the batched
//! decode's rows nor above the workspace's.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants: none beyond the types.

use super::*;
use crate::glm5next_dsa::select::{DsaSelectGeometry, contiguous_pool_count};

const KP: usize = 4;

/// 2026-10-11: Bytes of the per-row regions (scores f32, candidacy u8, selected i32 at
/// select_k 512) for `rows` rows over `pools` pools.
fn per_row_bytes(rows: usize, pools: usize) -> usize {
    rows * pools * 5 + rows * 512 * 4
}

#[test]
fn every_row_up_to_the_full_rows_context() {
    for ceiling in [1usize, 3, 4, 100, 4096, 8192, 16_384, 32_767, 32_768] {
        for rows in [1usize, 16, 128, 2048] {
            assert_eq!(
                select_scratch_rows(rows, ceiling, KP, 16),
                rows,
                "ceiling {ceiling}"
            );
        }
    }
}

#[test]
fn the_claim_context_is_capped_at_the_full_rows_budget() {
    let r = select_scratch_rows(2048, 262_144, KP, 128);
    assert_eq!(r, 256);
    let pools = contiguous_pool_count(KP, 262_144);
    let full = contiguous_pool_count(KP, SELECT_SCRATCH_FULL_ROWS_CONTEXT);
    assert!(
        r * pools <= 2048 * full,
        "within the budget of all rows at the full-rows context"
    );
    let (before, after) = (per_row_bytes(2048, pools), per_row_bytes(r, pools));
    assert_eq!(
        before >> 20,
        644,
        "MiB per layer when every row was sized at the ceiling"
    );
    assert!(
        after * 11 < (1 << 30),
        "under 1 GiB over GLM-5.3's 11 DSA layers: {after}"
    );
}

#[test]
fn never_below_the_floor_nor_above_max_rows() {
    // 2026-10-11: 1,048,576 tokens: the budget gives 64 rows, the floor raises it to 128.
    assert_eq!(select_scratch_rows(2048, 1 << 20, KP, 128), 128);
    // 2026-10-11: A floor above the workspace's rows is capped at them.
    assert_eq!(select_scratch_rows(16, 262_144, KP, 128), 16);
    for ceiling in [40_960usize, 65_536, 131_072, 262_144, 524_288, 1 << 20] {
        for rows in [1usize, 16, 512, 2048] {
            let r = select_scratch_rows(rows, ceiling, KP, 16);
            assert!(
                r >= 16.min(rows) && r <= rows,
                "ceiling {ceiling} rows {rows}: {r}"
            );
        }
    }
}

#[test]
fn the_rows_shrink_as_the_ceiling_grows() {
    let mut last = usize::MAX;
    for ceiling in (32_768..=1 << 20).step_by(4096) {
        let r = select_scratch_rows(2048, ceiling, KP, 16);
        assert!(r <= last, "ceiling {ceiling}: {r} > {last}");
        last = r;
    }
}

/// 2026-10-11: The workspace at the claim context and 2,048 rows: its selection output holds
/// all 2,048 rows while one pass over the ceiling's pools holds 256 of them, so a 2,048-row
/// prefill sub-chunk at full context selects in 8 sub-passes.
#[test]
fn the_workspace_outputs_every_row_and_passes_the_capped_rows() {
    let gpu = metrale_gpu_runtime::gpu::mock::MockGpuBackend::new();
    let c = Glm5NextDsaConfig {
        hidden: 4096,
        index_heads: 32,
        index_head_dim: 128,
        index_kpool: KP,
        index_topk: 2048,
        always_select_tail: true,
        local_heads: 4,
        q_lora_rank: 1536,
        kv_lora_rank: 512,
        qk_nope_head_dim: 256,
        qk_rope_head_dim: 0,
        v_head_dim: 256,
        max_context: 262_144,
    };
    let w = Glm5NextDsaWorkspace::new(&gpu, &c, 2048).unwrap();
    let at_ceiling = DsaSelectGeometry::plan(&c, 262_144, 2048).unwrap();
    assert_eq!(w.select.rows_per_pass(&at_ceiling), 256);
    let at_32k = DsaSelectGeometry::plan(&c, 32_768, 2048).unwrap();
    assert_eq!(
        w.select.rows_per_pass(&at_32k),
        2048,
        "one pass up to the full-rows context"
    );
    assert!(
        w.select
            .row(2047, &c)
            .fits(
                &c,
                &DsaSelectGeometry {
                    q_rows: 1,
                    ..at_ceiling
                }
            )
            .is_ok()
    );
}
