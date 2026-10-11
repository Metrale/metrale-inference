// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-07: MoE FFN whose routed experts are compressed-tensors `pack-quantized` INT4 or
//! INT8 (symmetric, group 128) and whose router and shared expert are BF16: Laguna-XS-2.1-INT4
//! on strix-hip. It never builds a [`super::MoeLayer`], whose NVFP4 lookups this target does
//! not ship.
//!
//! Owner: model-layers (packed-int MoE).
//! Invariants:
//! - Every forward entry point computes, for `n` rows at `input` (`[n, hidden]` BF16):
//!   BF16 router logits (`dense_gemm_bf16`), sigmoid top-k with the correction bias
//!   (`moe_topk_sigmoid_batched`: selection on `sigmoid + bias`, weights the unbiased
//!   sigmoids, normalized when `norm_topk_prob`, times `routed_scaling_factor`), the BF16
//!   shared expert, the routed experts in slot order (slot `s = t * top_k + j`) with the
//!   grouped packed-int GEMVs over per-expert pointer tables, and
//!   `out = bf16(bf16(sum_j w_j e_j) + shared)` (`moe_packed_int_combine`) into
//!   `ctx.buffers.moe_output()`.
//! - Scratch (all checked against the arena before the first launch): router logits in
//!   `gate_logits`; top-k ids then weights in `scratch` (the MoE prefill layout,
//!   `2 * n * top_k * 4` bytes); gate and up activations in `expert_gate_out` /
//!   `expert_up_out`; the down output in `expert_down_out`; the shared expert's output in
//!   `attn_output`.
//! - Weights are run as stored: `weight_packed` I32 and `weight_scale` BF16 device tensors,
//!   one u64 pointer table per projection with one entry per expert. Nothing is converted.
//!
//! Usage (the Laguna loader, `metrale_model_arch::weight_loader::laguna`):
//! ```ignore
//! let layer = PackedIntMoeLayer::new(weights, config, gpu)?;
//! let out = layer.forward_rows(input, n, &ctx, stream)?; // == ctx.buffers.moe_output()
//! ```

use anyhow::{Result, ensure};
use metrale_config::ModelConfig;
use metrale_config::precision_plan::packed_int::PackedIntScheme;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::quant_format::packed_int::{PACKED_INT_GEMV_MODULE, packed_int_gemv_kernels};
use crate::weight_map::DenseWeight;

/// 2026-10-07: Threads per block of every packed_int_gemv kernel (8 waves of 32).
const PI_THREADS: u32 = 256;
/// 2026-10-07: Output columns per block of the grouped GEMV (one per wave).
const PI_COLS_PER_BLOCK: u32 = 8;

/// 2026-10-07: One stored packed-int projection of one expert: device pointers to its
/// `weight_packed` words and `weight_scale` BF16 scales.
#[derive(Debug, Clone, Copy)]
pub struct PackedIntTensor {
    pub words: DevicePtr,
    pub scales: DevicePtr,
}

/// 2026-10-07: One routed expert's three projections.
#[derive(Debug, Clone, Copy)]
pub struct PackedIntExpert {
    pub gate_proj: PackedIntTensor,
    pub up_proj: PackedIntTensor,
    pub down_proj: PackedIntTensor,
}

/// 2026-10-07: Everything one packed-int MoE layer reads.
pub struct PackedIntMoeWeights {
    pub scheme: PackedIntScheme,
    /// 2026-10-07: Router `[num_experts, hidden]` BF16.
    pub gate: DenseWeight,
    /// 2026-10-07: `e_score_correction_bias` `[num_experts]` F32.
    pub correction_bias: DenseWeight,
    /// 2026-10-07: Shared expert, BF16: gate/up `[shared_inter, hidden]`, down
    /// `[hidden, shared_inter]`.
    pub shared_gate: DenseWeight,
    pub shared_up: DenseWeight,
    pub shared_down: DenseWeight,
    /// 2026-10-07: One entry per expert, in expert-id order.
    pub experts: Vec<PackedIntExpert>,
}

/// 2026-10-07: Device pointer tables of one projection: `[num_experts]` u64 word and scale
/// pointers, read by the grouped GEMV.
#[derive(Debug, Clone, Copy)]
struct PtrTable {
    words: DevicePtr,
    scales: DevicePtr,
}

pub struct PackedIntMoeLayer {
    w: PackedIntMoeWeights,
    num_experts: u32,
    top_k: u32,
    hidden: u32,
    inter: u32,
    shared_inter: u32,
    gate_table: PtrTable,
    up_table: PtrTable,
    down_table: PtrTable,
    pub(crate) grouped_k: KernelHandle,
    pub(crate) combine_k: KernelHandle,
    pub(crate) gemm_k: KernelHandle,
    pub(crate) topk_k: KernelHandle,
    pub(crate) silu_k: KernelHandle,
}

/// 2026-10-07: Upload one u64 per pointer, in order.
fn upload_ptr_table(gpu: &dyn GpuBackend, ptrs: &[DevicePtr]) -> Result<DevicePtr> {
    let bytes: Vec<u8> = ptrs.iter().flat_map(|p| p.0.to_le_bytes()).collect();
    let table = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(&bytes, table)?;
    Ok(table)
}

fn build_table(
    gpu: &dyn GpuBackend,
    experts: &[PackedIntExpert],
    pick: impl Fn(&PackedIntExpert) -> PackedIntTensor,
) -> Result<PtrTable> {
    let words: Vec<DevicePtr> = experts.iter().map(|e| pick(e).words).collect();
    let scales: Vec<DevicePtr> = experts.iter().map(|e| pick(e).scales).collect();
    Ok(PtrTable {
        words: upload_ptr_table(gpu, &words)?,
        scales: upload_ptr_table(gpu, &scales)?,
    })
}

impl PackedIntMoeLayer {
    /// 2026-10-07: Resolve the kernels (all required: a target without them fails here,
    /// naming the lookup) and upload the three pointer tables. Every expert must be present
    /// (no expert parallelism) and every pointer non-null.
    pub fn new(w: PackedIntMoeWeights, config: &ModelConfig, gpu: &dyn GpuBackend) -> Result<Self> {
        let num_experts = config.num_experts;
        ensure!(
            w.experts.len() == num_experts && num_experts > 0,
            "packed-int MoE: {} experts loaded, config declares {num_experts}",
            w.experts.len()
        );
        ensure!(
            config.num_experts_per_tok > 0 && config.num_experts_per_tok <= num_experts,
            "packed-int MoE: top_k {} out of range for {num_experts} experts",
            config.num_experts_per_tok
        );
        let g = w.scheme.group_size as usize;
        ensure!(
            config.hidden_size.is_multiple_of(g)
                && config.moe_intermediate_size.is_multiple_of(g)
                && config.shared_expert_intermediate_size > 0,
            "packed-int MoE: hidden {} and moe_intermediate {} must be multiples of {g}, \
             and a shared expert is required",
            config.hidden_size,
            config.moe_intermediate_size
        );
        for (e, x) in w.experts.iter().enumerate() {
            for (name, t) in [
                ("gate_proj", x.gate_proj),
                ("up_proj", x.up_proj),
                ("down_proj", x.down_proj),
            ] {
                ensure!(
                    !t.words.is_null() && !t.scales.is_null(),
                    "packed-int MoE: expert {e} {name} has a null device pointer"
                );
            }
        }
        for (name, p) in [
            ("router", w.gate.weight),
            ("correction bias", w.correction_bias.weight),
            ("shared gate_proj", w.shared_gate.weight),
            ("shared up_proj", w.shared_up.weight),
            ("shared down_proj", w.shared_down.weight),
        ] {
            ensure!(
                !p.is_null(),
                "packed-int MoE: {name} has a null device pointer"
            );
        }
        let (_, grouped_k) = packed_int_gemv_kernels(gpu, w.scheme)?;
        let layer = Self {
            num_experts: num_experts as u32,
            top_k: config.num_experts_per_tok as u32,
            hidden: config.hidden_size as u32,
            inter: config.moe_intermediate_size as u32,
            shared_inter: config.shared_expert_intermediate_size as u32,
            gate_table: build_table(gpu, &w.experts, |e| e.gate_proj)?,
            up_table: build_table(gpu, &w.experts, |e| e.up_proj)?,
            down_table: build_table(gpu, &w.experts, |e| e.down_proj)?,
            grouped_k,
            combine_k: gpu.kernel(PACKED_INT_GEMV_MODULE, "moe_packed_int_combine")?,
            gemm_k: gpu.kernel("gemm", "dense_gemm_bf16")?,
            topk_k: gpu.kernel("moe_topk_sig", "moe_topk_sigmoid_batched")?,
            silu_k: gpu.kernel("moe_silu_mul", "moe_silu_mul")?,
            w,
        };
        Ok(layer)
    }

    /// 2026-10-07: The scheme the grouped GEMV decodes.
    pub fn scheme(&self) -> PackedIntScheme {
        self.w.scheme
    }

    /// 2026-10-07: Fail before any launch when an arena buffer cannot hold `n` rows.
    fn check_capacity(&self, n: usize, ctx: &ForwardContext) -> Result<()> {
        let b = ctx.buffers;
        let (h, i, si, e, k) = (
            self.hidden as usize,
            self.inter as usize,
            self.shared_inter as usize,
            self.num_experts as usize,
            self.top_k as usize,
        );
        let slots = n * k;
        for (what, need, have) in [
            ("gate_logits", n * e * 2, b.gate_logits_bytes()),
            ("scratch", slots * 8, b.scratch_bytes()),
            (
                "expert_gate_out",
                (slots * i).max(n * si) * 2,
                b.expert_gate_out_bytes(),
            ),
            ("expert_down_out", slots * h * 2, b.expert_down_out_bytes()),
            ("attn_output", n * h * 2, b.attn_output_bytes()),
            ("moe_output", n * h * 2, b.moe_output_bytes()),
        ] {
            ensure!(
                need <= have,
                "packed-int MoE: {n} rows need {need} bytes of {what}, the arena has {have}"
            );
        }
        Ok(())
    }

    /// 2026-10-07: The layer's FFN for `n` rows; see the module invariants. Returns
    /// `ctx.buffers.moe_output()`.
    pub fn forward_rows(
        &self,
        input: DevicePtr,
        n: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        ensure!(n > 0, "packed-int MoE: zero rows");
        self.check_capacity(n, ctx)?;
        let gpu = ctx.gpu;
        let rows = n as u32;
        let (h, inter, si, top_k) = (self.hidden, self.inter, self.shared_inter, self.top_k);
        let slots = rows * top_k;

        let logits = ctx.buffers.gate_logits();
        ops::dense_gemm(
            gpu,
            self.gemm_k,
            input,
            &self.w.gate,
            logits,
            rows,
            self.num_experts,
            h,
            stream,
        )?;
        let ids = ctx.buffers.scratch();
        let weights = ids.offset(slots as usize * 4);
        ops::moe_topk_sigmoid_batched(
            gpu,
            self.topk_k,
            logits,
            self.w.correction_bias.weight,
            ids,
            weights,
            self.num_experts,
            top_k,
            ctx.config.norm_topk_prob,
            ctx.config.routed_scaling_factor as f32,
            rows,
            stream,
        )?;

        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();
        let shared_out = ctx.buffers.attn_output();
        ops::dense_gemm(
            gpu,
            self.gemm_k,
            input,
            &self.w.shared_gate,
            gate_out,
            rows,
            si,
            h,
            stream,
        )?;
        ops::dense_gemm(
            gpu,
            self.gemm_k,
            input,
            &self.w.shared_up,
            up_out,
            rows,
            si,
            h,
            stream,
        )?;
        ops::silu_mul(
            gpu,
            self.silu_k,
            gate_out,
            up_out,
            gate_out,
            rows * si,
            stream,
        )?;
        let shared_down = &self.w.shared_down;
        ops::dense_gemm(
            gpu,
            self.gemm_k,
            gate_out,
            shared_down,
            shared_out,
            rows,
            h,
            si,
            stream,
        )?;

        self.grouped(
            gpu,
            input,
            self.gate_table,
            ids,
            gate_out,
            top_k,
            inter,
            h,
            slots,
            stream,
        )?;
        self.grouped(
            gpu,
            input,
            self.up_table,
            ids,
            up_out,
            top_k,
            inter,
            h,
            slots,
            stream,
        )?;
        ops::silu_mul(
            gpu,
            self.silu_k,
            gate_out,
            up_out,
            gate_out,
            slots * inter,
            stream,
        )?;
        let down_out = ctx.buffers.expert_down_out();
        self.grouped(
            gpu,
            gate_out,
            self.down_table,
            ids,
            down_out,
            1,
            h,
            inter,
            slots,
            stream,
        )?;

        let output = ctx.buffers.moe_output();
        KernelLaunch::new(gpu, self.combine_k)
            .grid([rows, 1, 1])
            .block([PI_THREADS, 1, 1])
            .arg_ptr(down_out)
            .arg_ptr(weights)
            .arg_ptr(shared_out)
            .arg_ptr(output)
            .arg_i32(h as i32)
            .arg_i32(top_k as i32)
            .launch(stream)?;
        Ok(output)
    }

    /// 2026-10-07: `y[s] = W_{ids[s]} x[s / x_row_div]` over `slots` slots, `[n_out, k]`
    /// weights from `table`.
    #[allow(clippy::too_many_arguments)]
    fn grouped(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        table: PtrTable,
        ids: DevicePtr,
        y: DevicePtr,
        x_row_div: u32,
        n_out: u32,
        k: u32,
        slots: u32,
        stream: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, self.grouped_k)
            .grid([div_ceil(n_out, PI_COLS_PER_BLOCK), slots, 1])
            .block([PI_THREADS, 1, 1])
            .arg_ptr(x)
            .arg_ptr(table.words)
            .arg_ptr(table.scales)
            .arg_ptr(ids)
            .arg_ptr(y)
            .arg_i32(self.num_experts as i32)
            .arg_i32(x_row_div as i32)
            .arg_i32(n_out as i32)
            .arg_i32(k as i32)
            .launch(stream)
    }
}

#[cfg(test)]
#[path = "packed_int_moe_tests.rs"]
mod tests;
