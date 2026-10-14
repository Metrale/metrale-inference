// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-27: `MoeLayer::forward_nvfp4_grouped_decode`: the NVFP4 MoE of `m` rows in one
//! routed+shared expert dispatch, under an NVFP4 `--expert-quantization` tier.
//!
//! The steps are those of the grouped FP8 decode (`forward_fp8_grouped_decode.rs`): the
//! per-row router (`GroupedRouting::PerRow`), `moe_fp8_grouped_sort` (the slot sort and the
//! active-expert list), gate+up and SiLU, down, and the grouped blend. The expert kernels
//! (`ops/nvfp4_moe_grouped.rs`) read each active expert's NVFP4 weights once per pass for all
//! the rows routed to it. When the layer keeps its FP8 experts too (the qwen35 loader does,
//! for a native-FP8 checkpoint), the shared expert runs in FP8 through the grouped FP8 kernels
//! (with no active experts), the NVFP4 kernels take only the routed experts, and prefill keeps
//! the FP8 experts; under `nvfp4-gate-up` every down projection, routed and shared, runs
//! through the grouped FP8 down kernel instead. A shared expert the checkpoint keeps at BF16
//! (2026-10-13) runs after the routed down through `run_bf16_shared_expert`, whose batched GEMV
//! rows are each the one-row GEMV's bits; the NVFP4 kernels then take only the routed experts.
//! Which shared expert runs is `nvfp4_grouped_admit.rs`'s `nvfp4_grouped_shared`. Every step computes a row independently of
//! the other rows, so a row's output bits do not depend on `m`: the path serves every width
//! from one row up, and a row-invariant tier policy (`row_tiers.rs`) holds with it on.
//! `model-arch/examples/nvfp4_moe_grouped_microtest.rs` checks this.
//!
//! Owner: model-layers (MoE).
//! Invariants: `forward_nvfp4_grouped_decode` launches nothing unless
//! `nvfp4_grouped_decode_ok(m, ctx)` holds.

use super::nvfp4_grouped_admit::Nvfp4GroupedShared;
use super::*;

/// 2026-09-27: Widest row count this path admits on the CUDA-core expert kernels, the grouped
/// FP8 decode's.
pub const NVFP4_GROUPED_DECODE_MAX_ROWS: usize =
    super::forward_fp8_grouped_decode::FP8_GROUPED_DECODE_MAX_ROWS;

/// 2026-10-02: Widest row count on the tensor-core expert kernels, the grouped FP8 tensor-core
/// decode's: one verify of 128 sequences at one draft.
pub const NVFP4_GROUPED_DECODE_TC_MAX_ROWS: usize =
    super::forward_fp8_grouped_decode::FP8_GROUPED_DECODE_TC_MAX_ROWS;

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
    /// 2026-10-14: The split pair (`_tc_split` / `_tc_mt1`: the unsplit pair's bytes on twice the
    /// warps), taken in its place when both resolved; zero with `METRALE_MOE_TC_SPLIT=0`.
    pub gate_up_tc_split: KernelHandle,
    pub down_tc_mt1: KernelHandle,
    /// 2026-10-14: The Hopper staged pair (`moe_nvfp4_decode_hopper`, the split pair's bytes),
    /// zero unless `METRALE_MOE_DECODE_HSM=1` and the module exists (`nvfp4_decode_hsm.rs`).
    pub gate_up_hsm: KernelHandle,
    pub down_hsm: KernelHandle,
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
        let (gate_up_hsm, down_hsm) = super::nvfp4_decode_hsm::resolve(gpu);
        Self {
            gate_up: try_kernel(gpu, MODULE, "moe_expert_gate_up_act_nvfp4_grouped"),
            down: try_kernel(gpu, MODULE, "moe_expert_down_act_nvfp4_grouped"),
            gate_up_tc: try_kernel(gpu, TC, "moe_expert_gate_up_act_nvfp4_grouped_tc"),
            down_tc: try_kernel(gpu, TC, "moe_expert_down_act_nvfp4_grouped_tc"),
            gate_up_tc_split: if moe_tc_split_disabled() {
                KernelHandle(0)
            } else {
                try_kernel(gpu, TC, "moe_expert_gate_up_act_nvfp4_grouped_tc_split")
            },
            down_tc_mt1: if moe_tc_split_disabled() {
                KernelHandle(0)
            } else {
                try_kernel(gpu, TC, "moe_expert_down_act_nvfp4_grouped_tc_mt1")
            },
            gate_up_hsm,
            down_hsm,
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
    pub(super) fn select(&self, hidden: u32, inter: u32) -> Nvfp4GroupedLaunch {
        if let Some(lean) = self.lean_launch() {
            return lean;
        }
        if let Some(hsm) = self.hsm_launch(hidden, inter) {
            return hsm;
        }
        if nvfp4_grouped_tc_enabled()
            && self.gate_up_tc_split.0 != 0
            && self.down_tc_mt1.0 != 0
            && ops::nvfp4_grouped_tc_pair_shape_ok(
                hidden,
                inter,
                ops::NVFP4_GROUPED_GATE_UP_TC_SPLIT,
                ops::NVFP4_GROUPED_DOWN_TC_MT1,
            )
        {
            return Nvfp4GroupedLaunch {
                gate_up: self.gate_up_tc_split,
                gate_up_geometry: ops::NVFP4_GROUPED_GATE_UP_TC_SPLIT,
                down: self.down_tc_mt1,
                down_geometry: ops::NVFP4_GROUPED_DOWN_TC_MT1,
                gate_up_smem: 0,
                down_smem: 0,
                max_rows: NVFP4_GROUPED_DECODE_TC_MAX_ROWS,
            };
        }
        if nvfp4_grouped_tc_enabled()
            && self.gate_up_tc.0 != 0
            && self.down_tc.0 != 0
            && ops::nvfp4_grouped_tc_expert_shape_ok(hidden, inter)
        {
            Nvfp4GroupedLaunch {
                gate_up: self.gate_up_tc,
                gate_up_geometry: ops::NVFP4_GROUPED_GATE_UP_TC,
                down: self.down_tc,
                down_geometry: ops::NVFP4_GROUPED_DOWN_TC,
                gate_up_smem: 0,
                down_smem: 0,
                max_rows: NVFP4_GROUPED_DECODE_TC_MAX_ROWS,
            }
        } else {
            Nvfp4GroupedLaunch {
                gate_up: self.gate_up,
                gate_up_geometry: ops::NVFP4_GROUPED_GATE_UP_SCALAR,
                down: self.down,
                down_geometry: ops::NVFP4_GROUPED_DOWN_SCALAR,
                gate_up_smem: 0,
                down_smem: 0,
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
    /// 2026-10-14: Dynamic shared bytes of each launch (0 but for the `_hsm` pair).
    pub(super) gate_up_smem: u32,
    pub(super) down_smem: u32,
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

/// 2026-10-14: `METRALE_MOE_TC_SPLIT` equal to `"0"`, read once per process: the unsplit
/// tensor-core pair runs instead of the split pair (same bytes).
fn moe_tc_split_disabled() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var("METRALE_MOE_TC_SPLIT").as_deref() == Ok("0"))
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

impl MoeLayer {
    /// 2026-09-27: The NVFP4 MoE of `m` rows: `input` is `[m, H]` BF16, the output lands in
    /// rows `0..m` of `moe_output()`. Returns an error, launching nothing, when
    /// `nvfp4_grouped_decode_ok(m, ctx)` is false.
    pub fn forward_nvfp4_grouped_decode(
        &self,
        input: DevicePtr,
        m: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        anyhow::ensure!(
            self.nvfp4_grouped_decode_ok(m, ctx),
            "forward_nvfp4_grouped_decode: predicate false for m={m} (caller must gate on it)"
        );
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.moe_intermediate_size as u32;
        let num_experts = ctx.config.num_experts as u32;
        let top_k = ctx.config.num_experts_per_tok as u32;
        let n = m as u32;
        let te = m * top_k as usize;
        if ctx.stats.once("log:moe_nvfp4_grouped_decode") {
            tracing::info!(
                "MoE NVFP4 grouped decode active (first use M={m}, top_k={top_k}, \
                 experts={num_experts}; one-time log)"
            );
        }

        let scratch = ctx.buffers.scratch();
        let indices_dev = scratch;
        let weights_dev = scratch.offset(te * 4);

        // 2026-09-27: The sort scratch reuses `gate_logits` (the layout of
        // `grouped_decode_buffer_need`). 2026-10-14: Routing and sort in `grouped_route_sorted`.
        let ne = num_experts as usize;
        let gate_logits = ctx.buffers.gate_logits();
        let sorted_token_ids = gate_logits;
        let sorted_expert_ids = gate_logits.offset(te * 4);
        let expert_offsets = gate_logits.offset(te * 4 * 2);
        let token_to_perm = gate_logits.offset(te * 4 * 2 + (ne + 1) * 4);
        let cap = ops::fp8_grouped_active_cap(n, top_k, num_experts);
        let active_experts = token_to_perm.offset(te * 4);
        let active_count = active_experts.offset(cap as usize * 4);
        self.grouped_route_sorted(
            input,
            m,
            indices_dev,
            weights_dev,
            ops::Fp8GroupedSortOut {
                sorted_token_ids,
                sorted_expert_ids,
                expert_offsets,
                token_to_perm,
                active_experts,
                active_count,
            },
            ctx,
            stream,
        )?;

        let act = ctx.buffers.expert_gate_out();
        let expert_down_out = ctx.buffers.expert_down_out();
        let shared_act = ctx.buffers.logits();
        let shared_out = ctx.buffers.attn_output();
        let tables = |t: &ExpertPtrTable| ops::Nvfp4ExpertTables {
            packed_ptrs: t.packed_ptrs,
            scale_ptrs: t.scale_ptrs,
            scale2_vals: t.scale2_vals,
        };
        let sh = &self.weights.shared_expert;
        // 2026-09-27: With the layer's FP8 experts kept, the grouped FP8 gate+up kernel runs the
        // FP8 shared expert alone (no active experts) and the NVFP4 one only the routed experts
        // (no shared rows). The down projections: under `nvfp4` the grouped FP8 down kernel runs
        // the shared expert alone and the NVFP4 one the routed experts; under `nvfp4-gate-up`
        // the grouped FP8 down kernel runs all of them from the FP8 experts.
        // 2026-10-13: A BF16 shared expert (`Nvfp4GroupedShared::Bf16`) runs after the routed
        // down through `run_bf16_shared_expert`; the NVFP4 kernels then launch no shared rows.
        let shared_form = self.nvfp4_grouped_shared();
        let fp8_shared = self
            .nvfp4_decode_fp8_shared()
            .filter(|_| shared_form == Some(Nvfp4GroupedShared::Fp8));
        let fp8_down = if crate::layers::expert_quantization().nvfp4_down() {
            None
        } else {
            self.fp8_down_weight_ptrs.as_ref().zip(fp8_shared)
        };
        if let Some(fsh) = fp8_shared {
            ops::moe_expert_gate_up_act_fp8_grouped(
                ctx.gpu,
                self.moe_expert_gate_up_act_fp8_grouped_k,
                ops::FP8_GROUPED_GATE_UP_SCALAR,
                input,
                DevicePtr::NULL,
                DevicePtr::NULL,
                DevicePtr::NULL,
                DevicePtr::NULL,
                act,
                expert_offsets,
                sorted_token_ids,
                active_experts,
                active_count,
                &fsh.gate_proj,
                &fsh.up_proj,
                shared_act,
                inter,
                h,
                0,
                n,
                stream,
            )?;
        }
        if let (Some(fsh), None) = (fp8_shared, fp8_down) {
            ops::moe_expert_down_act_fp8_grouped(
                ctx.gpu,
                self.moe_expert_down_act_fp8_grouped_k,
                ops::FP8_GROUPED_DOWN_SCALAR,
                act,
                DevicePtr::NULL,
                DevicePtr::NULL,
                expert_down_out,
                expert_offsets,
                active_experts,
                active_count,
                shared_act,
                &fsh.down_proj,
                shared_out,
                h,
                inter,
                0,
                n,
                stream,
            )?;
        }
        let nvfp4_shared_rows = if shared_form == Some(Nvfp4GroupedShared::Nvfp4) {
            n
        } else {
            0
        };
        let launch = self.nvfp4_grouped.select(h, inter);
        ops::moe_expert_gate_up_act_nvfp4_grouped(
            ctx.gpu,
            launch.gate_up,
            launch.gate_up_geometry,
            launch.gate_up_smem,
            input,
            tables(&self.gate_ptrs),
            tables(&self.up_ptrs),
            act,
            expert_offsets,
            sorted_token_ids,
            active_experts,
            active_count,
            &sh.gate_proj,
            &sh.up_proj,
            shared_act,
            inter,
            h,
            cap,
            nvfp4_shared_rows,
            stream,
        )?;
        if let Some((dp, fsh)) = fp8_down {
            ops::moe_expert_down_act_fp8_grouped(
                ctx.gpu,
                self.moe_expert_down_act_fp8_grouped_k,
                ops::FP8_GROUPED_DOWN_SCALAR,
                act,
                dp.weight_ptrs,
                dp.scale_ptrs,
                expert_down_out,
                expert_offsets,
                active_experts,
                active_count,
                shared_act,
                &fsh.down_proj,
                shared_out,
                h,
                inter,
                cap,
                n,
                stream,
            )?;
        } else {
            ops::moe_expert_down_act_nvfp4_grouped(
                ctx.gpu,
                launch.down,
                launch.down_geometry,
                launch.down_smem,
                act,
                tables(&self.down_ptrs),
                expert_down_out,
                expert_offsets,
                active_experts,
                active_count,
                shared_act,
                &sh.down_proj,
                shared_out,
                h,
                inter,
                cap,
                nvfp4_shared_rows,
                stream,
            )?;
        }
        if shared_form == Some(Nvfp4GroupedShared::Bf16) {
            // 2026-10-13: gate and up land in `shared_act` (`[m, inter]` each, the SiLU product
            // in place over gate), the BF16 `[m, H]` output in `shared_out`, which the blend
            // reads as it reads the fused kernels' shared term.
            let ran = self.run_bf16_shared_expert(
                input,
                n,
                h,
                ctx.config.shared_expert_intermediate_size as u32,
                shared_act,
                shared_act.offset(m * inter as usize * 2),
                shared_out,
                ctx,
                stream,
            )?;
            anyhow::ensure!(
                ran,
                "forward_nvfp4_grouped_decode: BF16 shared expert absent"
            );
        }
        ops::moe_weighted_sum_blend_fp8_grouped(
            ctx.gpu,
            self.moe_weighted_sum_blend_fp8_grouped_k,
            ctx.buffers.moe_output(),
            expert_down_out,
            weights_dev,
            token_to_perm,
            shared_out,
            input,
            self.weights.shared_expert_gate.weight,
            h,
            top_k,
            h,
            n,
            stream,
        )
    }
}

#[cfg(test)]
#[path = "forward_nvfp4_grouped_decode_tests.rs"]
mod tests;
