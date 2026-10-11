// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: The grouped NVFP4 decode's kernel-selection types (`Nvfp4GroupedKernels`,
//! `Nvfp4GroupedLaunch`) and the `METRALE_NO_MOE_NVFP4_TC` kill switch. Split out of
//! `forward_nvfp4_grouped_decode.rs` to keep that file under the 500-line cap; no behavior
//! change. `nvfp4_lean.rs` adds a further `impl Nvfp4GroupedKernels` block of its own
//! (`lean_launch`, called from `select` below).
//!
//! Owner: model-layers (MoE).

use super::forward_nvfp4_grouped_decode::{
    NVFP4_GROUPED_DECODE_MAX_ROWS, NVFP4_GROUPED_DECODE_TC_MAX_ROWS,
};
use super::*;

/// 2026-09-27: The two expert kernels of this path, looked up with `try_kernel`; a zero handle
/// declines it. The sort, router and blend are the grouped FP8 decode's.
/// 2026-10-02: `gate_up_tc` / `down_tc` are the tensor-core twins (`moe_nvfp4_grouped_tc.cu`),
/// taken when both resolved unless `METRALE_NO_MOE_NVFP4_TC` is present. `declared_experts`: the
/// layer's routed and shared experts are the checkpoint's own NVFP4 (declared W4A16, not a
/// requantized copy), so decode takes this path at every width whatever the
/// `--expert-quantization` tier (which governs FP8 checkpoints only); set by the qwen35 loader
/// under `--weight-quantization declared` ([`MoeLayer::set_declared_nvfp4_experts`]).
pub(super) struct Nvfp4GroupedKernels {
    pub gate_up: KernelHandle,
    pub down: KernelHandle,
    pub gate_up_tc: KernelHandle,
    pub down_tc: KernelHandle,
    pub declared_experts: bool,
    /// 2026-10-04: The lean point's pair (`Nvfp4G16Lean`) and its load-time repack; `lean` is set
    /// once this layer's tables were repacked (`nvfp4_lean.rs`), after which only that pair reads them.
    pub gate_up_tc_lean: KernelHandle,
    pub down_tc_lean: KernelHandle,
    pub lean_repack: KernelHandle,
    pub lean: bool,
    /// 2026-10-02: The BF16 point's pair (`moe_bf16_grouped_tc.cu`, `forward_bf16_grouped_decode.rs`).
    pub bf16_gate_up_tc: KernelHandle,
    pub bf16_down_tc: KernelHandle,
}

impl Nvfp4GroupedKernels {
    /// 2026-09-27: One direct `try_kernel` call per kernel (`#[track_caller]` audit lines).
    pub(super) fn resolve(gpu: &dyn GpuBackend) -> Self {
        use super::super::try_kernel;
        const MODULE: &str = "moe_nvfp4_grouped";
        const TC: &str = "moe_nvfp4_grouped_tc";
        Self {
            gate_up: try_kernel(gpu, MODULE, "moe_expert_gate_up_act_nvfp4_grouped"),
            down: try_kernel(gpu, MODULE, "moe_expert_down_act_nvfp4_grouped"),
            gate_up_tc: try_kernel(gpu, TC, "moe_expert_gate_up_act_nvfp4_grouped_tc"),
            down_tc: try_kernel(gpu, TC, "moe_expert_down_act_nvfp4_grouped_tc"),
            declared_experts: false,
            gate_up_tc_lean: try_kernel(gpu, TC, "moe_expert_gate_up_act_nvfp4_grouped_tc_lean"),
            down_tc_lean: try_kernel(gpu, TC, "moe_expert_down_act_nvfp4_grouped_tc_lean"),
            lean_repack: try_kernel(gpu, TC, "nvfp4_tc_lean_repack"),
            lean: false,
            bf16_gate_up_tc: try_kernel(
                gpu,
                "moe_bf16_grouped_tc",
                "moe_expert_gate_up_act_bf16_grouped_tc",
            ),
            bf16_down_tc: try_kernel(
                gpu,
                "moe_bf16_grouped_tc",
                "moe_expert_down_act_bf16_grouped_tc",
            ),
        }
    }

    /// 2026-10-02: The gate+up and down launches for an `inter` x `hidden` expert: the
    /// tensor-core twins when on and the shape fits them, else the CUDA-core kernels.
    /// 2026-10-04: Always the lean pair once the tables are lean (the repack checked the shape).
    /// 2026-10-05: The CUDA-core pair when the routed down runs the grouped FP8 down kernel
    /// (`fp8_down`, the `nvfp4-gate-up` tier): that kernel reads the routed SiLU products as FP32
    /// rows, which only the CUDA-core gate+up writes; the tensor-core one writes BF16 hi + lo
    /// pairs, and an FP8 down over them decodes nonsense.
    pub(super) fn select(&self, hidden: u32, inter: u32, fp8_down: bool) -> Nvfp4GroupedLaunch {
        if let Some(lean) = self.lean_launch() {
            return lean;
        }
        if !fp8_down
            && nvfp4_grouped_tc_enabled()
            && self.gate_up_tc.0 != 0
            && self.down_tc.0 != 0
            && ops::nvfp4_grouped_tc_shape_ok(inter, hidden, ops::NVFP4_GROUPED_GATE_UP_TC)
            && ops::nvfp4_grouped_tc_shape_ok(hidden, inter, ops::NVFP4_GROUPED_DOWN_TC)
        {
            Nvfp4GroupedLaunch {
                gate_up: self.gate_up_tc,
                gate_up_geometry: ops::NVFP4_GROUPED_GATE_UP_TC,
                down: self.down_tc,
                down_geometry: ops::NVFP4_GROUPED_DOWN_TC,
                max_rows: NVFP4_GROUPED_DECODE_TC_MAX_ROWS,
            }
        } else {
            Nvfp4GroupedLaunch {
                gate_up: self.gate_up,
                gate_up_geometry: ops::NVFP4_GROUPED_GATE_UP_SCALAR,
                down: self.down,
                down_geometry: ops::NVFP4_GROUPED_DOWN_SCALAR,
                max_rows: NVFP4_GROUPED_DECODE_MAX_ROWS,
            }
        }
    }
}

/// 2026-10-02: The expert kernels one grouped NVFP4 decode launches, and the widest row count
/// they admit.
pub(super) struct Nvfp4GroupedLaunch {
    pub(super) gate_up: KernelHandle,
    pub(super) gate_up_geometry: ops::Fp8GroupedGeometry,
    pub(super) down: KernelHandle,
    pub(super) down_geometry: ops::Fp8GroupedGeometry,
    pub(super) max_rows: usize,
}

/// 2026-10-02: The grouped NVFP4 decode takes the tensor-core expert kernels unless
/// `METRALE_NO_MOE_NVFP4_TC` is present (a debugging kill switch: the CUDA-core kernels). Read
/// once per process. The two pairs differ in summation order, and in the SiLU product's
/// carrier (FP32 against BF16 hi + lo), so their bits differ; each is row-invariant.
pub(super) fn nvfp4_grouped_tc_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("METRALE_NO_MOE_NVFP4_TC").is_none())
}
