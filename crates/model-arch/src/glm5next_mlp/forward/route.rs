// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Which routed-expert path a `forward_moe` of `rows` rows takes (`expert_route`),
//! read by `forward_moe_pieces` and by the multi-sequence prefill's grouping, so the two never
//! disagree.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - `forward_moe_pieces` takes exactly the path `expert_route` names for its row count.

use super::super::precision::MlpKernel;
use super::super::weights::Glm5NextMoeWeights;
use super::super::{Glm5NextMlpConfig, Glm5NextMlpKernels};
use super::{
    Glm5NextMlpWorkspace, MOE_ROW_BATCH_MAX_ROWS, MOE_ROW_UNION_MAX_IDS, forward_prefill_gemm,
    host_dispatch_forced, moe_row_groups, row_batch_disabled, row_batch_max,
};

/// 2026-10-09: The routed experts' path for one `forward_moe` call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpertRoute {
    /// 2026-10-09: W4A4 on the static activation scales (the precision plan's choice).
    W4a4,
    /// 2026-10-09: The grouped tensor-core GEMM over all rows (`forward_prefill_gemm`).
    Grouped,
    /// 2026-10-09: The row-union GEMV over sub-groups of `row_batch_max()` rows.
    RowBatched,
    /// 2026-10-09: One row at a time.
    PerRow,
}

/// 2026-10-09: The path a `forward_moe` of `rows` rows takes: the precision plan first, then the
/// grouped GEMM above `MOE_ROW_BATCH_MAX_ROWS` rows and from `prefill_gemm_min_rows()`, then
/// the row-union GEMV, else per row.
pub fn expert_route(
    k: &Glm5NextMlpKernels,
    cfg: &Glm5NextMlpConfig,
    w: &Glm5NextMoeWeights,
    ws: &Glm5NextMlpWorkspace,
    rows: usize,
) -> ExpertRoute {
    use crate::glm5next_layer::profile;
    if w.precision.kernel(rows) == MlpKernel::W4a4Static {
        return ExpertRoute::W4a4;
    }
    let grouped = rows > MOE_ROW_BATCH_MAX_ROWS
        && rows >= forward_prefill_gemm::prefill_gemm_min_rows()
        && forward_prefill_gemm::prefill_gemm_enabled()
        && !host_dispatch_forced()
        && !profile::trace_on()
        && k.moe_sort_by_expert.0 != 0
        && k.moe_grouped_gemm.0 != 0
        && k.combine_indexed.0 != 0
        && rows * cfg.top_k <= ws.max_total_expanded();
    if grouped {
        return ExpertRoute::Grouped;
    }
    let batched = rows >= 2
        && !host_dispatch_forced()
        && !row_batch_disabled()
        && !profile::trace_on()
        && k.moe_row_union.0 != 0
        && moe_row_groups(rows, row_batch_max()).iter().all(|&(_, w)| {
            w >= 2
                && w * cfg.top_k <= MOE_ROW_UNION_MAX_IDS
                && k.w4a16_gemv_sw_moe_batchm[w - 2].0 != 0
        });
    if batched {
        ExpertRoute::RowBatched
    } else {
        ExpertRoute::PerRow
    }
}
