// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: Launchers for the W8A8 tensor-core grouped FP8 MoE decode
//! (`kernels/gb10/common/moe_fp8_grouped_tc_w8a8.cu`): the per-(row, 128) E4M3 activation
//! quantization, gate+up with a re-quantized SiLU product, and down; plus where their
//! quantized activations live inside the grouped decode's two SiLU buffers.
//! 2026-09-29: And the `_hilo` gate+up, which keeps the SiLU product at FP32 precision for the
//! W8A16 down kernel, with its quantized input in the down output buffer
//! ([`Fp8GroupedW8a8HiloLayout`]).
//!
//! Owner: model-layers ops.
//! Invariants: [`Fp8GroupedW8a8Layout::new`] and [`Fp8GroupedW8a8HiloLayout::new`] refuse
//! buffers that cannot hold their layout.

use anyhow::{Result, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::{FP8_GROUPED_TC_ROWS_PER_PASS, Fp8GroupedGeometry};
use crate::weight_map::Fp8Weight;

/// 2026-09-28: `moe_expert_gate_up_act_fp8_grouped_tc_w8a8` (`TC8_GU_COLS`: one 128-column
/// quantization group of the down projection's K per CTA).
pub const FP8_GROUPED_GATE_UP_TC_W8A8: Fp8GroupedGeometry = Fp8GroupedGeometry {
    cols_per_cta: 128,
    rows_per_pass: FP8_GROUPED_TC_ROWS_PER_PASS,
    threads: 128,
};

/// 2026-09-28: `moe_expert_down_act_fp8_grouped_tc_w8a8` (`TC8_DOWN_COLS`).
pub const FP8_GROUPED_DOWN_TC_W8A8: Fp8GroupedGeometry = Fp8GroupedGeometry {
    cols_per_cta: 256,
    rows_per_pass: FP8_GROUPED_TC_ROWS_PER_PASS,
    threads: 128,
};

fn align16(x: usize) -> usize {
    x.div_ceil(16) * 16
}

/// 2026-09-28: Byte offsets of the W8A8 activations inside the grouped decode's buffers.
/// The routed SiLU buffer (`expert_gate_out`, sized for `[te, inter]` FP32) holds the routed
/// E4M3 products `[te, inter]`, their scales `[te, inter / 128]` FP32, the quantized layer
/// input `[m, hidden]` E4M3 and its scales `[m, hidden / 128]`; the shared SiLU buffer
/// (`logits`, `[m, inter]` FP32) holds the shared products and their scales.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fp8GroupedW8a8Layout {
    pub act_s: usize,
    pub xq: usize,
    pub xs: usize,
    pub sh_s: usize,
}

impl Fp8GroupedW8a8Layout {
    /// 2026-09-28: The layout for `m` rows, or an error when `hidden` or `inter` is not whole
    /// 128 groups or either buffer is too small.
    pub fn new(
        m: usize,
        top_k: usize,
        hidden: usize,
        inter: usize,
        routed_bytes: usize,
        shared_bytes: usize,
    ) -> Result<Self> {
        ensure!(
            hidden.is_multiple_of(128) && inter.is_multiple_of(128),
            "W8A8 grouped decode: hidden {hidden} and inter {inter} must be whole 128 groups"
        );
        let te = m * top_k;
        let act_s = align16(te * inter);
        let xq = align16(act_s + te * (inter / 128) * 4);
        let xs = align16(xq + m * hidden);
        let routed_end = xs + m * (hidden / 128) * 4;
        let sh_s = align16(m * inter);
        let shared_end = sh_s + m * (inter / 128) * 4;
        ensure!(
            routed_end <= routed_bytes && shared_end <= shared_bytes,
            "W8A8 grouped decode: needs {routed_end} + {shared_end} bytes, buffers hold {routed_bytes} + {shared_bytes}"
        );
        Ok(Self {
            act_s,
            xq,
            xs,
            sh_s,
        })
    }
}

/// 2026-09-29: Where the `_hilo` gate+up's quantized layer input lives: `[m, hidden]` E4M3 at
/// the start of the down output buffer (`expert_down_out`, `[te, hidden]` BF16), its scales
/// `[m, hidden / 128]` FP32 at `xs`. The down kernel writes that buffer only after gate+up has
/// read them. The SiLU buffers hold the BF16 hi|lo products `[te, 2 inter]` and `[m, 2 inter]`,
/// as under W8A16.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fp8GroupedW8a8HiloLayout {
    pub xs: usize,
}

impl Fp8GroupedW8a8HiloLayout {
    /// 2026-09-29: The layout for `m` rows, or an error when `hidden` or `inter` is not whole
    /// 128 groups or a buffer is too small.
    pub fn new(
        m: usize,
        top_k: usize,
        hidden: usize,
        inter: usize,
        routed_bytes: usize,
        shared_bytes: usize,
        down_out_bytes: usize,
    ) -> Result<Self> {
        ensure!(
            hidden.is_multiple_of(128) && inter.is_multiple_of(128),
            "W8A8 hi|lo grouped decode: hidden {hidden} and inter {inter} must be whole 128 groups"
        );
        let xs = align16(m * hidden);
        let down_end = xs + m * (hidden / 128) * 4;
        let (routed_end, shared_end) = (m * top_k * inter * 4, m * inter * 4);
        ensure!(
            down_end <= down_out_bytes && routed_end <= routed_bytes && shared_end <= shared_bytes,
            "W8A8 hi|lo grouped decode: needs {down_end} + {routed_end} + {shared_end} bytes, \
             buffers hold {down_out_bytes} + {routed_bytes} + {shared_bytes}"
        );
        Ok(Self { xs })
    }
}

/// 2026-09-28: E4M3 per (row, 128-K group) of `x` `[rows, k]` BF16 into `q` / `s`.
#[allow(clippy::too_many_arguments)]
pub fn moe_act_quant_e4m3(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    x: DevicePtr,
    q: DevicePtr,
    s: DevicePtr,
    rows: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([rows, 1, 1])
        .block([128, 1, 1])
        .arg_ptr(x)
        .arg_ptr(q)
        .arg_ptr(s)
        .arg_u32(k)
        .launch(stream)
}

/// 2026-09-28: The routed expert tables and the grouped decode's sort outputs the two W8A8
/// kernels read.
pub struct Fp8GroupedW8a8Rows {
    pub expert_offsets: DevicePtr,
    pub sorted_token_ids: DevicePtr,
    pub active_experts: DevicePtr,
    pub active_count: DevicePtr,
    pub cap: u32,
    pub num_tokens: u32,
}

/// 2026-09-28: W8A8 gate+up and SiLU: `(xq, xs)` the quantized layer input; writes the
/// quantized products `(act_q, act_s)` by sorted position and `(sh_q, sh_s)` by token.
/// 2026-09-29: With the `_hilo` kernel, `act.0` / `sh.0` receive the BF16 hi|lo products and
/// `act.1` / `sh.1` are unused (pass null).
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_gate_up_act_fp8_grouped_tc_w8a8(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    xq: DevicePtr,
    xs: DevicePtr,
    gate: (DevicePtr, DevicePtr),
    up: (DevicePtr, DevicePtr),
    act: (DevicePtr, DevicePtr),
    rows: &Fp8GroupedW8a8Rows,
    sh_gate: &Fp8Weight,
    sh_up: &Fp8Weight,
    sh: (DevicePtr, DevicePtr),
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    let g = FP8_GROUPED_GATE_UP_TC_W8A8;
    KernelLaunch::new(gpu, kernel)
        .grid([
            div_ceil(n, g.cols_per_cta),
            rows.cap + div_ceil(rows.num_tokens, g.rows_per_pass),
            1,
        ])
        .block([g.threads, 1, 1])
        .arg_ptr(xq)
        .arg_ptr(xs)
        .arg_ptr(gate.0)
        .arg_ptr(gate.1)
        .arg_ptr(up.0)
        .arg_ptr(up.1)
        .arg_ptr(act.0)
        .arg_ptr(act.1)
        .arg_ptr(rows.expert_offsets)
        .arg_ptr(rows.sorted_token_ids)
        .arg_ptr(rows.active_experts)
        .arg_ptr(rows.active_count)
        .arg_ptr(sh_gate.weight)
        .arg_ptr(sh_gate.row_scale)
        .arg_ptr(sh_up.weight)
        .arg_ptr(sh_up.row_scale)
        .arg_ptr(sh.0)
        .arg_ptr(sh.1)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(rows.cap)
        .arg_u32(rows.num_tokens)
        .launch(stream)
}

/// 2026-09-28: W8A8 down over the quantized SiLU products: routed rows into `output` by
/// sorted position, the shared rows into `sh_down_out` by token, both BF16.
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_down_act_fp8_grouped_tc_w8a8(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    act: (DevicePtr, DevicePtr),
    down: (DevicePtr, DevicePtr),
    output: DevicePtr,
    rows: &Fp8GroupedW8a8Rows,
    sh: (DevicePtr, DevicePtr),
    sh_down: &Fp8Weight,
    sh_down_out: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    let g = FP8_GROUPED_DOWN_TC_W8A8;
    KernelLaunch::new(gpu, kernel)
        .grid([
            div_ceil(n, g.cols_per_cta),
            rows.cap + div_ceil(rows.num_tokens, g.rows_per_pass),
            1,
        ])
        .block([g.threads, 1, 1])
        .arg_ptr(act.0)
        .arg_ptr(act.1)
        .arg_ptr(down.0)
        .arg_ptr(down.1)
        .arg_ptr(output)
        .arg_ptr(rows.expert_offsets)
        .arg_ptr(rows.active_experts)
        .arg_ptr(rows.active_count)
        .arg_ptr(sh.0)
        .arg_ptr(sh.1)
        .arg_ptr(sh_down.weight)
        .arg_ptr(sh_down.row_scale)
        .arg_ptr(sh_down_out)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(rows.cap)
        .arg_u32(rows.num_tokens)
        .launch(stream)
}
