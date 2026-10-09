// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: GLM-5.3-Flash MLP: the dense SwiGLU FFN and the routed NVFP4 MoE, their kernels and per-rank geometry.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - `Glm5NextMlpConfig::from_config` returns a config only when the dense and shared widths
//!   split over `tp_world_size` in [`BF16_GEMM_K_ALIGN`] units (`metrale_config::tp_split`),
//!   `num_experts` divides over `ep_world_size`, and `validate` passes. 2026-10-08: The width
//!   split may be uneven (2048 over three ranks: 688/680/680); an even one is unchanged.
//! - For such configs, the `local_expert_range`s of ranks `0..ep_world_size` partition
//!   `0..num_experts`, and `local_slot` is `None` for every id outside this rank's range.
//! - 2026-10-09: Under `ExpertShard::Sliced` (`--moe-expert-layout tp`) every rank's range is
//!   all of `0..num_experts` and `moe_intermediate` is the rank's slice of every expert
//!   (`expert_tp`); the routed sum is then a partial sum over the slices, reduced by the same
//!   all-reduce.
//!
//! # One all-reduce for both EP and TP
//!
//! With TP or EP above 1, a routed site's output is a partial sum on every rank: the shared
//! expert is TP-sharded (`local_shared_intermediate`) and each rank runs only the experts it
//! owns. `forward::forward_moe` adds the shared output and the routed slots in one combine
//! kernel, and the layer then reduces that output once. The combine has to come before the
//! reduce: adding a TP-sharded shared expert after the reduce would drop the other ranks'
//! part of it.
//!
//! # Numerics the kernels rely on
//!
//! * The SwiGLU clamp is asymmetric: `gate` is bounded above only, `up` on both sides. The
//!   limit is `ModelConfig::swiglu_limit`, which the `glm5_next` parser refuses to default.
//! * The router's correction bias steers selection only. The emitted weight is the chosen
//!   expert's unbiased sigmoid score.
//! * `routed_scaling_factor` multiplies the top-k weights; the shared expert is added unscaled.
//! * Every rank holds the whole router and ranks all `num_experts`, so for the same input every
//!   rank selects the same ids. Sharding the router would give each rank different partial
//!   logits and a different top-k.
//! * `num_experts` is the full routed-expert count; `local_experts` is this rank's share.
//!   `glm5next_router_topk` must be given the full count.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{GpuBackend, KernelHandle};

pub mod build;
pub mod build_w4a4;
mod config;
pub mod expert_tp;
pub mod forward;
pub mod forward_prefill_gemm;
pub mod precision;
pub mod weights;

pub use config::{Glm5NextMlpConfig, Glm5NextMlpKind};
pub use expert_tp::{EXPERT_TP_UNIT, ExpertShard, ExpertSlice};
pub use weights::{
    Glm5NextDenseMlpWeights, Glm5NextDenseSite, Glm5NextExpertWeights, Glm5NextMoeWeights,
};

/// 2026-09-25: Module of `kernels/gb10/common/glm5next_ffn.cu`. A `.cu` file not listed in
/// `common/KERNEL.toml`'s `[modules]` resolves under its file stem; the listed ones below do
/// not, so each module name here is checked against that table.
pub const FFN_MODULE: &str = "glm5next_ffn";
/// 2026-09-25: `[modules]`: `dense_gemm_bf16 = "gemm"`.
pub const GEMM_MODULE: &str = "gemm";
/// 2026-09-25: `[modules]`: `w4a16_gemm = "w4a16"`.
pub const W4A16_MODULE: &str = "w4a16";
/// 2026-09-25: Module of `w4a16_gemv.cu` (not in `[modules]`, so its file stem). Its GEMV
/// kernels run every routed-expert projection except the grouped prefill GEMM.
pub const W4A16_GEMV_MODULE: &str = "w4a16_gemv";
/// 2026-09-25: `[modules]`: `moe_permute = "moe"`, the token sort / permute / unpermute kernels.
pub const MOE_MODULE: &str = "moe";
/// 2026-09-25: `[modules]`: `moe_w4a16_grouped_gemm = "moe_w4a16"`, the tensor-core grouped W4A16 GEMM.
pub const MOE_GROUPED_MODULE: &str = "moe_w4a16";

/// 2026-10-08: Module of `w4a4_gemv_mx_moe.cu` (its file stem): the static-scale NVFP4
/// activation quantizer and the routed-expert W4A4 slot GEMV.
pub const W4A4_MOE_MODULE: &str = "w4a4_gemv_mx_moe";
/// 2026-10-08: Module of `w4a4_gemv_mx.cu` (its file stem): the W4A4 mx GEMVs.
pub const W4A4_MX_MODULE: &str = "w4a4_gemv_mx";

/// 2026-10-08: The unit the dense and shared-expert widths split over TP in. Their `down_proj`
/// runs with the rank's width as K, and `dense_gemv_bf16`, `dense_gemv_bf16_batchm` (both
/// `kernels/gb10/common/`) assume `K % 8 == 0` for their 16-byte row loads; the activation rows
/// those GEMMs read are also `width` apart, so the same rule keeps them 16-byte aligned.
pub const BF16_GEMM_K_ALIGN: usize = 8;

/// 2026-09-25: The most experts `glm5next_router_topk` can select per token: its per-token
/// selection lives in the shared arrays `sel_id[16]` and `sel_w[16]`.
pub const KERNEL_MAX_TOP_K: usize = 16;

/// 2026-09-25: The kernels a GLM MLP site launches.
///
/// `resolve` fails when a `gpu.kernel` entry point is missing (the GEMM/GEMV bases, `w4a16`,
/// `w4a16_gemv` and the three `glm5next_ffn.cu` kernels). The `try_kernel` ones are
/// `KernelHandle(0)` when absent, and the forward takes another path for each.
#[derive(Clone, Copy)]
pub struct Glm5NextMlpKernels {
    /// 2026-09-25: BF16 `C = A @ B^T` tile GEMM: dense FFN and shared expert.
    pub gemm: KernelHandle,
    /// 2026-09-25: The same GEMM with FP32 output, used for the router, because
    /// `glm5next_router_topk` reads FP32 logits.
    pub gemm_f32: KernelHandle,
    /// 2026-09-25: M=1 GEMVs for `gemm` / `gemm_f32`. `gemv_f32` is optional; when it is 0,
    /// the router's M=1 calls run the tile GEMM (`ops::dense_mm_bf16`).
    pub gemv: KernelHandle,
    pub gemv_f32: KernelHandle,
    /// 2026-09-25: `dense_gemv_bf16_batchm`: 2 to `DENSE_GEMV_BATCHM_MAX_M` (16) rows in one
    /// weight sweep, for the dense FFN and shared expert. When 0, those rows run the tile GEMM.
    pub gemv_batchm: KernelHandle,
    /// 2026-10-09: `dense_gemv_bf16_batchm_wide` (9..=16 rows, accumulators in registers);
    /// `0` when absent (`glm5next_layer::wide_gemv`).
    pub gemv_batchm_wide: KernelHandle,
    /// 2026-09-25: NVFP4 `w4a16_gemm` tile GEMM. Resolved, but not launched by this module.
    pub w4a16: KernelHandle,
    /// 2026-09-25: NVFP4 `C[1, N] = A[1, K] @ B[N, K]^T`, the per-expert decode GEMV.
    ///
    /// Its grid is tied to the kernel's `N_PER_BLOCK`; use `ops::w4a16_gemv_grid_x`.
    pub w4a16_gemv: KernelHandle,
    /// 2026-09-25: Single-warp-per-output variant of `w4a16_gemv`, checked bit-identical to it
    /// by `examples/w4a16_gemv_sw_microtest.rs`. When 0, the base kernel runs.
    ///
    /// Its grid is `ceil(N/8)` (`N_PER_BLOCK_SW`), not the base kernel's `ceil(N/4)`; launch it
    /// through `ops::w4a16_gemv_sw_raw` or `ops::w4a16_decode_gemv`, which pick the grid.
    pub w4a16_gemv_sw: KernelHandle,
    /// 2026-09-25: All `top_k` slots of one row in one launch, grid `(ceil(N/8), top_k, 1)`.
    /// Weights come from the global-id pointer tables indexed by the router's on-device ids.
    /// When 0, the forward reads the ids back to the host and launches per local expert.
    pub w4a16_gemv_sw_moe: KernelHandle,
    /// 2026-09-25: Row-batched `w4a16_gemv_sw_moe`, indexed `[rows - 2]` for rows 2..=16
    /// (`w4a16_gemv_sw_moe_batchm_m2` .. `_m16`; 2026-10-09: was 2..=8). Each expert in the union of the rows'
    /// selections is swept once for all the rows that picked it.
    ///
    /// grid.y is the union entry, not the slot, with extent `rows * top_k`; unfilled entries
    /// return on `u_eid < 0`. Needs [`Self::moe_row_union`].
    pub w4a16_gemv_sw_moe_batchm: [KernelHandle; forward::MOE_ROW_BATCH_MAX_ROWS - 1],
    /// 2026-09-25: Builds the union table the batched kernel indexes: one block of
    /// `rows * top_k` threads. `forward_moe` uses it only when `rows * top_k` is at most
    /// `MOE_ROW_UNION_MAX_IDS`.
    pub moe_row_union: KernelHandle,
    /// 2026-09-25: `glm5next_swiglu_clamp`, the asymmetric clamped SwiGLU. `moe_silu_mul`
    /// does not clamp.
    pub swiglu: KernelHandle,
    pub router: KernelHandle,
    pub combine: KernelHandle,
    /// 2026-09-25: `moe_sort_by_expert` (`moe_permute.cu`): counting sort of the `[rows, top_k]`
    /// ids into expert-contiguous order, writing `sorted_token_ids`, `expert_offsets` and
    /// `token_to_perm`. When 0, the grouped prefill path is off.
    pub moe_sort_by_expert: KernelHandle,
    /// 2026-09-25: The tensor-core grouped W4A16 GEMM (`mma.sync.aligned.m16n8k16`) for
    /// prefill, at the tile `forward_prefill_gemm::gemm_tile` picks. When 0, the grouped
    /// prefill path is off.
    pub moe_grouped_gemm: KernelHandle,
    /// 2026-09-25: [`Self::combine`] reading the routed rows in expert-sorted order through
    /// `token_to_perm`, with the same accumulation order and single rounding.
    pub combine_indexed: KernelHandle,
    /// 2026-10-08: `w4a4_quant_rows_static` (`w4a4_gemv_mx_moe.cu`): NVFP4 activations under a
    /// static per-tensor scale, the input of every W4A4 projection here.
    pub w4a4_quant_static: KernelHandle,
    /// 2026-10-08: `w4a4_gemv_mx8_moe_slots`: the routed experts' W4A4 GEMV, one block row per
    /// (token, slot), weights from the global-id pointer tables.
    pub w4a4_moe_slots: KernelHandle,
    /// 2026-10-09: `w4a4_gemv_mx{8,16}_moe_union`: the slot GEMV with each union expert swept
    /// once for every row that chose it, up to 8 and 16 rows.
    pub w4a4_moe_union: [KernelHandle; 2],
    /// 2026-10-08: The dense W4A4 GEMVs `w4a4_gemv_mx8`, `_mx16`, `_mx32` (`w4a4_gemv_mx.cu`),
    /// for up to 8, 16 and 32 rows.
    pub w4a4_mx: [KernelHandle; 3],
}

/// 2026-10-08: Widest launch on the W4A4 slot GEMV (each slot re-reads its expert); wider runs
/// the grouped W4A16 GEMM, logged as above declared. Unmeasured: the C16 decode width.
pub const MOE_W4A4_MAX_ROWS: usize = 16;

/// 2026-10-08: Rows one dense W4A4 launch covers (`w4a4_gemv_mx32`); wider launches run in
/// chunks of it, and each row's sums do not depend on the chunk.
pub const DENSE_W4A4_CHUNK_ROWS: usize = 32;

impl Glm5NextMlpKernels {
    /// 2026-10-09: The BF16-out batched GEMV pair `glm_mm` takes.
    pub(crate) fn batchm(&self) -> crate::glm5next_layer::wide_gemv::Batchm {
        crate::glm5next_layer::wide_gemv::Batchm {
            narrow: self.gemv_batchm,
            wide: self.gemv_batchm_wide,
        }
    }
}

impl Glm5NextMlpKernels {
    pub fn resolve(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            gemm: gpu.kernel(GEMM_MODULE, "dense_gemm_bf16")?,
            gemm_f32: gpu.kernel(GEMM_MODULE, "dense_gemm_bf16_f32out")?,
            gemv: gpu.kernel("gemv", "dense_gemv_bf16")?,
            gemv_batchm: metrale_model_layers::layers::try_kernel(
                gpu,
                "dense_gemv_bf16_batchm",
                "dense_gemv_bf16_batchm",
            ),
            gemv_batchm_wide: metrale_model_layers::layers::try_kernel(
                gpu,
                "dense_gemv_bf16_batchm",
                "dense_gemv_bf16_batchm_wide",
            ),
            gemv_f32: metrale_model_layers::layers::try_kernel(
                gpu,
                "gemv",
                "dense_gemv_bf16_fp32out",
            ),
            w4a16: gpu.kernel(W4A16_MODULE, "w4a16_gemm")?,
            w4a16_gemv: gpu.kernel(W4A16_GEMV_MODULE, "w4a16_gemv")?,
            w4a16_gemv_sw: metrale_model_layers::layers::try_kernel(
                gpu,
                W4A16_GEMV_MODULE,
                "w4a16_gemv_sw",
            ),
            w4a16_gemv_sw_moe: metrale_model_layers::layers::try_kernel(
                gpu,
                W4A16_GEMV_MODULE,
                "w4a16_gemv_sw_moe",
            ),
            // 2026-10-09: Tiers 2..=16, `[rows - 2]`.
            w4a16_gemv_sw_moe_batchm: std::array::from_fn(|i| {
                metrale_model_layers::layers::try_kernel(
                    gpu,
                    W4A16_GEMV_MODULE,
                    &format!("w4a16_gemv_sw_moe_batchm_m{}", i + 2),
                )
            }),
            moe_row_union: metrale_model_layers::layers::try_kernel(
                gpu,
                W4A16_GEMV_MODULE,
                "glm5next_moe_row_union",
            ),
            swiglu: gpu.kernel(FFN_MODULE, "glm5next_swiglu_clamp")?,
            router: gpu.kernel(FFN_MODULE, "glm5next_router_topk")?,
            combine: gpu.kernel(FFN_MODULE, "glm5next_moe_combine")?,
            moe_sort_by_expert: metrale_model_layers::layers::try_kernel(
                gpu,
                MOE_MODULE,
                "moe_sort_by_expert",
            ),
            // 2026-09-25: Each tile is its own entry point, so the tile
            // (`METRALE_GLM_MOE_GEMM_TILE`) is read before resolving. A missing tile
            // other than the base falls back to `GEMM_TILES[0]`, with a warning.
            moe_grouped_gemm: {
                let tile = forward_prefill_gemm::gemm_tile();
                let h =
                    metrale_model_layers::layers::try_kernel(gpu, MOE_GROUPED_MODULE, tile.name);
                if h.0 == 0 && tile.name != forward_prefill_gemm::GEMM_TILES[0].name {
                    tracing::warn!(
                        "GLM routed-MoE grouped GEMM tile `{}` is not in this target's PTX — \
                         falling back to `{}`",
                        tile.name,
                        forward_prefill_gemm::GEMM_TILES[0].name
                    );
                    metrale_model_layers::layers::try_kernel(
                        gpu,
                        MOE_GROUPED_MODULE,
                        forward_prefill_gemm::GEMM_TILES[0].name,
                    )
                } else {
                    h
                }
            },
            combine_indexed: metrale_model_layers::layers::try_kernel(
                gpu,
                FFN_MODULE,
                "glm5next_moe_combine_indexed",
            ),
            w4a4_quant_static: metrale_model_layers::layers::try_kernel(
                gpu,
                W4A4_MOE_MODULE,
                "w4a4_quant_rows_static",
            ),
            w4a4_moe_slots: metrale_model_layers::layers::try_kernel(
                gpu,
                W4A4_MOE_MODULE,
                "w4a4_gemv_mx8_moe_slots",
            ),
            w4a4_moe_union: ["w4a4_gemv_mx8_moe_union", "w4a4_gemv_mx16_moe_union"]
                .map(|e| metrale_model_layers::layers::try_kernel(gpu, W4A4_MOE_MODULE, e)),
            w4a4_mx: ["w4a4_gemv_mx8", "w4a4_gemv_mx16", "w4a4_gemv_mx32"]
                .map(|e| metrale_model_layers::layers::try_kernel(gpu, W4A4_MX_MODULE, e)),
        })
    }

    /// 2026-10-08: The routed experts' W4A4 row cap on this target: [`MOE_W4A4_MAX_ROWS`] when
    /// the static quantizer and the slot GEMV resolved, else 0.
    pub fn w4a4_expert_rows(&self) -> usize {
        if self.w4a4_quant_static.0 != 0 && self.w4a4_moe_slots.0 != 0 {
            MOE_W4A4_MAX_ROWS
        } else {
            0
        }
    }

    /// 2026-10-08: The dense MLP's W4A4 row cap: unbounded (chunked) when the static quantizer
    /// and the three mx GEMVs resolved, else 0.
    pub fn w4a4_dense_rows(&self) -> usize {
        if self.w4a4_quant_static.0 != 0 && self.w4a4_mx.iter().all(|k| k.0 != 0) {
            usize::MAX
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests;
