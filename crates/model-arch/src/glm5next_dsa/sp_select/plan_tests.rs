// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The split plan at every chunk end of a 1,048,576-token prefill in 8,192-row
//! chunks, on the scratch the serve's workspace allocates (`select_scratch_geometry`): the even
//! slab plan splits every chunk on 2 and on 3 ranks (the positive control), and the stepped slab
//! plan the split ran before 2026-10-17 refused exactly the chunks ending at 196,608 * m tokens
//! on 3 ranks, where a 1,366- or 1,365-row slab ends in a 1-row sub-pass (the negative control).
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants: none beyond the types.

use metrale_gpu_runtime::gpu::mock::MockGpuBackend;

use super::super::Glm5NextDsaConfig;
use super::super::layer::select_scratch_geometry;
use super::super::select::{DsaSelectGeometry, DsaSelectScratch, passes_tests, row_passes};
use super::run::{SlabPlan, plan, plan_with};
use super::*;

/// 2026-10-17: The 1M serve's ceiling, its prefill chunk and the rows its workspace serves (the
/// widest batched pass).
const CEILING: usize = 1 << 20;
const CHUNK: usize = 8192;
const SERVE_ROWS: usize = 8208;

/// 2026-10-17: Every chunk passes the table, so only the sub-pass rule can refuse.
fn table() -> MinPoolsTable {
    lever::parse(Some("8192:1")).unwrap().unwrap()
}

/// 2026-10-17: `(seq, pools, rows per sub-pass)` of the chunk ending at each multiple of
/// `CHUNK` up to `CEILING`, on the serve's scratch at that ceiling.
fn chunk_ends() -> Vec<(usize, usize, usize)> {
    let c = Glm5NextDsaConfig {
        max_context: CEILING,
        ..passes_tests::cfg()
    };
    let gpu = MockGpuBackend::new();
    let geom = select_scratch_geometry(
        &c,
        SERVE_ROWS,
        crate::glm5next_layer::GlmLevers::declared().row_group,
    )
    .unwrap();
    let s = DsaSelectScratch::alloc_rows(&gpu, &c, &geom, SERVE_ROWS).unwrap();
    (1..=CEILING / CHUNK)
        .map(|n| {
            let g = DsaSelectGeometry::plan(&c, n * CHUNK, CHUNK).unwrap();
            (n * CHUNK, g.n_pools, s.rows_per_pass(&g))
        })
        .collect()
}

/// 2026-10-17: Every chunk end splits, on every rank of 2 and of 3, with the slabs `owned`
/// names. The sweep reaches sub-passes narrower than a slab (1,365 rows at 196,608 tokens).
#[test]
fn every_chunk_end_to_the_1m_ceiling_splits_on_2_and_3_ranks() {
    let (t, ends) = (table(), chunk_ends());
    assert_eq!(ends[196_608 / CHUNK - 1], (196_608, 49_152, 1365));
    for world in [2, 3] {
        for rank in 0..world {
            let own = owned(CHUNK, world, rank).unwrap();
            for &(seq, pools, per) in &ends {
                assert_eq!(
                    plan(Some(&t), world, rank, CHUNK, pools, true, per).unwrap(),
                    Some(own),
                    "world {world} rank {rank} seq {seq} per {per}"
                );
            }
        }
    }
}

/// 2026-10-17: The stepped slab plan refuses the split at exactly the chunks ending at 196,608,
/// 393,216, 589,824, 786,432 and 983,040 tokens on 3 ranks, and at none on 2 (2,048-row slabs).
#[test]
fn the_stepped_slab_plan_refused_the_chunks_at_196608_tokens_times_m() {
    let (t, ends) = (table(), chunk_ends());
    let stepped: SlabPlan = |slab, per| row_passes(&[slab], per);
    for world in [2, 3] {
        let refused: Vec<usize> = ends
            .iter()
            .filter(|&&(_, pools, per)| {
                plan_with(stepped, Some(&t), world, 0, CHUNK, pools, true, per)
                    .unwrap()
                    .is_none()
            })
            .map(|e| e.0)
            .collect();
        let want: Vec<usize> = if world == 3 {
            (1..=5).map(|m| 196_608 * m).collect()
        } else {
            Vec::new()
        };
        assert_eq!(refused, want, "world {world}");
    }
}
