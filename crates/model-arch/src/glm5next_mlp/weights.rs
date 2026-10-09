// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Device weight handles for one GLM-5.3 MLP site, already sharded for this rank.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types; `build.rs` fills them.

use metrale_gpu_runtime::gpu::DevicePtr;

use super::precision::GroupPrecision;

/// 2026-09-25: One NVFP4 projection: packed `e2m1` pairs, per-16-element `e4m3` block scales,
/// and one global `f32` scale.
///
/// `scale_2` is held on the host and passed to the kernels by value (`arg_f32`), not as a
/// device pointer.
#[derive(Debug, Clone, Copy)]
pub struct Nvfp4Proj {
    /// 2026-09-25: `[out, in/2]` U8, two `e2m1` codes per byte.
    pub packed: DevicePtr,
    /// 2026-09-25: `[out, in/16]` F8_E4M3 block scales.
    pub scale: DevicePtr,
    /// 2026-09-25: The single global F32 scale.
    pub scale_2: f32,
    /// 2026-10-08: The checkpoint's static activation scale (`input_scale`, ModelOpt's
    /// `amax / (6 * 448)`), when it declares FP4 activations for this projection. The W4A4
    /// path quantizes the input under it and scales the output by `input_scale * scale_2`; the
    /// W4A16 path does not read it.
    pub input_scale: Option<f32>,
}

/// 2026-09-25: A BF16 SwiGLU MLP: the dense layers `0..first_k_dense_replace`, and the shared
/// expert of every routed layer.
///
/// TP: `gate_proj`/`up_proj` (stored `[inter, hidden]`) are sliced by row, `down_proj` (stored
/// `[hidden, inter]`) by column, so the output is a partial sum when `tp_world_size > 1`.
#[derive(Debug, Clone, Copy)]
pub struct Glm5NextDenseMlpWeights {
    /// 2026-09-25: `[local_inter, hidden]` BF16.
    pub gate_proj: DevicePtr,
    /// 2026-09-25: `[local_inter, hidden]` BF16.
    pub up_proj: DevicePtr,
    /// 2026-09-25: `[hidden, local_inter]` BF16.
    pub down_proj: DevicePtr,
}

/// 2026-10-08: The checkpoint's NVFP4 dense MLP of one rank, kept packed for the W4A4 path:
/// `gate_proj`/`up_proj` hold this rank's rows, `down_proj` its columns (as the BF16 form).
#[derive(Debug, Clone, Copy)]
pub struct Glm5NextDenseNvfp4Weights {
    pub gate_proj: Nvfp4Proj,
    pub up_proj: Nvfp4Proj,
    pub down_proj: Nvfp4Proj,
}

/// 2026-10-08: The static activation scales one W4A4 MLP group runs under: one for the input
/// of gate and up (they read the same activations), one for the input of down. A routed site
/// uses one pair for all its experts (`build_w4a4::uniform_act_scales`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct W4a4ActScales {
    pub gate_up: f32,
    pub down: f32,
}

/// 2026-10-08: A dense MLP site of one rank: the weight forms its precision plan reaches, BF16
/// (dequantized at load) and/or packed NVFP4 with its activation scales, and the plan.
#[derive(Debug, Clone)]
pub struct Glm5NextDenseSite {
    pub bf16: Option<Glm5NextDenseMlpWeights>,
    pub nvfp4: Option<(Glm5NextDenseNvfp4Weights, W4a4ActScales)>,
    pub precision: GroupPrecision,
}

/// 2026-09-25: One routed NVFP4 expert, owned whole by one EP rank.
#[derive(Debug, Clone, Copy)]
pub struct Glm5NextExpertWeights {
    pub gate_proj: Nvfp4Proj,
    pub up_proj: Nvfp4Proj,
    pub down_proj: Nvfp4Proj,
}

/// 2026-09-25: Device pointer tables for one projection over the full expert set.
///
/// Indexed by global expert id, so the grouped kernels go from the router's on-device `ids` to
/// weights without a host read. Experts another EP rank owns have null `packed`/`scale`
/// pointers; the kernels write nothing for those slots, and the caller zeroes the output first.
#[derive(Debug, Clone, Copy)]
pub struct Glm5NextExpertPtrTable {
    /// 2026-09-25: `[num_experts]` U64 device pointers to each expert's packed NVFP4 weight.
    pub packed_ptrs: DevicePtr,
    /// 2026-09-25: `[num_experts]` U64 device pointers to each expert's block scales.
    pub scale_ptrs: DevicePtr,
    /// 2026-09-25: `[num_experts]` F32 per-expert global scale.
    pub scale2_vals: DevicePtr,
}

/// 2026-09-25: The three projections' pointer tables for one routed site.
#[derive(Debug, Clone, Copy)]
pub struct Glm5NextMoePtrTables {
    pub gate: Glm5NextExpertPtrTable,
    pub up: Glm5NextExpertPtrTable,
    pub down: Glm5NextExpertPtrTable,
}

/// 2026-09-25: A routed MoE site's weights for this rank.
pub struct Glm5NextMoeWeights {
    /// 2026-09-25: `[num_experts, hidden]` BF16 router, the full matrix on every rank (see the
    /// module header). Run through the FP32-output GEMM.
    pub router: DevicePtr,
    /// 2026-09-25: `[num_experts]` F32 selection bias (`gate.e_score_correction_bias`). It
    /// steers selection only; the emitted weight is the chosen expert's unbiased score.
    pub router_bias: DevicePtr,
    /// 2026-09-25: The shared expert: dense BF16, TP-sharded, added to every token unscaled.
    pub shared: Glm5NextDenseMlpWeights,
    /// 2026-09-25: `local_experts` entries in ascending global id, indexed by
    /// `Glm5NextMlpConfig::local_slot(global_id)`, never by the global id.
    pub experts: Vec<Glm5NextExpertWeights>,
    /// 2026-09-25: Global-id-indexed pointer tables over the same experts, for the
    /// device-dispatched forward. Null entries mark remote ids.
    pub ptrs: Glm5NextMoePtrTables,
    /// 2026-10-08: Which kernel the routed experts run at each row count.
    pub precision: GroupPrecision,
    /// 2026-10-08: The experts' static activation scales, present when `precision` reaches
    /// the W4A4 kernels.
    pub act_scales: Option<W4a4ActScales>,
}
