// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: The NVFP4 MoE prefill on the grouped tensor-core expert kernels of the decode
//! (`forward_nvfp4_grouped_decode.rs`), behind `METRALE_MOE_PREFILL_TC`.
//!
//! The decode path computes each row independently of the others at any row count (the
//! kernels loop over an expert's rows), so a prefill pass of `m` rows can run it as is: the
//! declared W4A16 products (BF16 E2M1 x E4M3 weights, BF16 activations, FP32 sums), the per-row
//! router, row-invariant. It reads the checkpoint's own NVFP4 tables (the lean layout), not the
//! transposed W4A16 prefill copies. A prefill row then gets the bits a decode of the same input
//! gets; they differ from the transposed-copy prefill's (E4M3 x E4M3 MMAs on an unscaled cast).
//!
//! Owner: model-layers (MoE).
//! Invariants:
//! - Prefill differs from decode only in the shared expert's SiLU scratch: `expert_up_out`
//!   (unused by the grouped kernels) instead of `logits`, which a mixed step's decode rows hold
//!   while its prefill runs.

use super::forward_nvfp4_grouped_decode::Nvfp4GroupedLaunch;
use super::*;

/// 2026-10-05: The row policy of the grouped NVFP4 path: decode admits up to the selected
/// kernels' widest row count with the shared SiLU product in `logits`; prefill only the
/// tensor-core kernels (they loop over an expert's rows), any row count the arena holds, with
/// it in `expert_up_out`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum GroupedRows {
    Decode,
    /// 2026-10-05: Writes its rows into `moe_output` from row `out_row` on.
    Prefill {
        out_row: usize,
    },
}

impl GroupedRows {
    pub(super) fn max_rows(self, launch: &Nvfp4GroupedLaunch) -> usize {
        match self {
            Self::Decode => launch.max_rows,
            Self::Prefill { .. } if launch.max_rows == NVFP4_GROUPED_DECODE_TC_MAX_ROWS => {
                usize::MAX
            }
            Self::Prefill { .. } => 0,
        }
    }

    pub(super) fn shared_act(self, b: &metrale_gpu_runtime::buffers::BufferArena) -> DevicePtr {
        match self {
            Self::Decode => b.logits(),
            Self::Prefill { .. } => b.expert_up_out(),
        }
    }

    pub(super) fn shared_act_bytes(self, b: &metrale_gpu_runtime::buffers::BufferArena) -> usize {
        match self {
            Self::Decode => b.logits_bytes(),
            Self::Prefill { .. } => b.expert_gate_out_bytes(),
        }
    }

    /// 2026-10-05: The router: decode's per-row GEMV; for prefill the dense GEMM tiers, which
    /// share one summation order across row counts (`GroupedRouting::Prefill`).
    pub(super) fn routing(self) -> GroupedRouting {
        match self {
            Self::Decode => GroupedRouting::PerRow,
            Self::Prefill { .. } => GroupedRouting::Prefill,
        }
    }

    /// 2026-10-05: The first output row: `moe_output`, from `out_row` under prefill.
    pub(super) fn output(
        self,
        b: &metrale_gpu_runtime::buffers::BufferArena,
        h: usize,
    ) -> DevicePtr {
        match self {
            Self::Decode => b.moe_output(),
            Self::Prefill { out_row } => b.moe_output().offset(out_row * h * 2),
        }
    }
}

/// 2026-09-27: Shape admission without a GPU: `m` in `1..=max_rows` (2026-10-02: the selected
/// expert kernels' widest, `NVFP4_GROUPED_DECODE_MAX_ROWS` or `NVFP4_GROUPED_DECODE_TC_MAX_ROWS`),
/// `hidden % 32 == 0` (gate+up reads 32-element chunks) and `inter % 16 == 0` (down reads
/// 16-element blocks), and a shared expert as wide as a routed one (the shared SiLU product
/// shares the routed layout).
pub fn nvfp4_grouped_decode_shape_ok(
    m: usize,
    max_rows: usize,
    hidden: usize,
    inter: usize,
    shared_inter: usize,
) -> bool {
    (1..=max_rows).contains(&m)
        && hidden >= 32
        && hidden.is_multiple_of(32)
        && inter >= 16
        && inter.is_multiple_of(16)
        && shared_inter == inter
}

/// 2026-10-05: The resolved `--moe-prefill-tc` decision.
static MOE_PREFILL_TC: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// 2026-10-05: Publish the command line's `--moe-prefill-tc`; returns the value in force, which
/// differs from `enabled` when the cell was already resolved.
pub fn set_moe_prefill_tc_from_cli(enabled: bool) -> bool {
    let _ = MOE_PREFILL_TC.set(enabled);
    *MOE_PREFILL_TC.get().expect("just set")
}

/// 2026-10-05: Whether the NVFP4 MoE prefill runs the grouped tensor-core path
/// ([`MoeLayer::forward_nvfp4_grouped_prefill`]): `--moe-prefill-tc`, else
/// `METRALE_MOE_PREFILL_TC` (presence), resolved once per process.
pub fn moe_prefill_tc_enabled() -> bool {
    *MOE_PREFILL_TC.get_or_init(|| std::env::var_os("METRALE_MOE_PREFILL_TC").is_some())
}

impl MoeLayer {
    /// 2026-10-05: Whether the prefill of `m` rows takes the grouped tensor-core path: the lever,
    /// and every condition of the grouped decode except its row cap and its `logits` scratch, for
    /// the rows of one pass ([`Self::nvfp4_grouped_prefill_pass_rows`]).
    pub fn nvfp4_grouped_prefill_ok(&self, m: usize, ctx: &ForwardContext) -> bool {
        moe_prefill_tc_enabled() && self.nvfp4_grouped_prefill_pass_rows(m, ctx).is_some()
    }

    /// 2026-10-05: The rows one grouped pass of an `m`-row prefill takes: `m`, halved until the
    /// arena holds the pass (the SiLU product is FP32, twice the transposed path's BF16). The
    /// passes cut the rows, never a row, and each row's bits do not depend on its pass-mates, so
    /// the split changes no bit; it reads the active experts' weights once per pass.
    fn nvfp4_grouped_prefill_pass_rows(&self, m: usize, ctx: &ForwardContext) -> Option<usize> {
        let mut rows = m;
        while rows > 0 {
            if self.nvfp4_grouped_ok(rows, ctx, GroupedRows::Prefill { out_row: 0 }) {
                return Some(rows);
            }
            rows /= 2;
        }
        None
    }

    /// 2026-10-05: The NVFP4 MoE prefill of `m` rows on the grouped tensor-core path.
    pub(super) fn forward_nvfp4_grouped_prefill(
        &self,
        input: DevicePtr,
        m: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let pass = self
            .nvfp4_grouped_prefill_pass_rows(m, ctx)
            .filter(|_| moe_prefill_tc_enabled())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "forward_nvfp4_grouped_prefill: predicate false for m={m} (caller must gate on it)"
                )
            })?;
        let row_bytes = ctx.config.hidden_size * 2;
        let mut r0 = 0;
        while r0 < m {
            let rows = pass.min(m - r0);
            self.nvfp4_grouped_rows(
                input.offset(r0 * row_bytes),
                rows,
                GroupedRows::Prefill { out_row: r0 },
                ctx,
                stream,
            )?;
            r0 += rows;
        }
        Ok(())
    }
}
