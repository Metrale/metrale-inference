// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-10-19: The MoE on device-resident MXFP4 experts (the safetensors checkpoint), one
//! tensor-parallel rank's slice of every expert. A rank holds rows `[r0, r0 + inter)` of each
//! expert's w1 and w3 and the same columns of w2, where `inter` is the local width
//! (`MoeV41Cfg::inter`); the shared expert is sliced the same way.
//!
//! One call: the router and the selection (`route_launch`, `route_select`), one upload of the
//! expert-sorted plan, the input rounded to E4M3 blocks of 32, w1 and w3 on the grouped
//! pointer-table GEMM (bf16 out), the SwiGLU with the routing weight, the hidden rounded again, w2
//! on the f32-out GEMM into the reduction's partial rows, the shared expert's slice into the rows
//! after them, and [`TpReduce`] for the cross-rank sum and the reference's bf16 rounding sequence.
//!
//! Owner: model-arch, DeepSeek-V4.1 MoE.
//! Invariants:
//! - Every launch of a call is issued on its one stream, so the scratch is reused in order.
//! - The reduction's partial buffer holds `max_tokens * (topk + 1)` rows.

use anyhow::{Result, ensure};
use metrale_comm::CommBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::resident_plan::{BlobLayout, resident_plan};
use super::{MoeV41, MoeV41LayerWeights};
use crate::fp8_block_v41::act_quant_fp8;
use crate::tp_reduce_v41::{CombineRows, TpReduce};
use metrale_model_layers::layers::ops::{self, ResidentMat};
use metrale_model_layers::weight_map::DenseWeight;

/// 2026-10-19: The 64-row M tile of `moe_w4a16_grouped_gemm_ptrtable_e8m0` (`M_TILE`).
const PTRTABLE_M_TILE: usize = 64;
/// 2026-10-19: MXFP4 values per E8M0 scale.
const MX_BLOCK: usize = 32;

/// 2026-10-19: One layer's resident expert slices. Expert `e`'s w1 is `w1 + e * inter * dim / 2`
/// (packed, `[inter, dim / 2]`, the low nibble the even column) with scales at
/// `w1_s + e * inter * dim / 32`; w3 alike; w2 `[dim, inter / 2]` and `[dim, inter / 32]`.
#[derive(Clone, Copy, Debug)]
pub struct ResidentExperts {
    pub n_experts: usize,
    pub inter: usize,
    pub dim: usize,
    pub w1: DevicePtr,
    pub w1_s: DevicePtr,
    pub w3: DevicePtr,
    pub w3_s: DevicePtr,
    pub w2: DevicePtr,
    pub w2_s: DevicePtr,
}

impl ResidentExperts {
    /// 2026-10-19: Bytes of one expert's packed projection and of its scales (all three share
    /// the element count `inter * dim`).
    pub fn packed_bytes(&self) -> usize {
        self.inter * self.dim / 2
    }
    pub fn scale_bytes(&self) -> usize {
        self.inter * self.dim / MX_BLOCK
    }

    /// 2026-10-19: Pointer table `table` (w1, w1 scale, w3, w3 scale, w2, w2 scale) of expert `e`.
    pub fn addr(&self, table: usize, e: u32) -> u64 {
        let e = e as u64;
        let (base, stride) = match table {
            0 => (self.w1, self.packed_bytes()),
            1 => (self.w1_s, self.scale_bytes()),
            2 => (self.w3, self.packed_bytes()),
            3 => (self.w3_s, self.scale_bytes()),
            4 => (self.w2, self.packed_bytes()),
            _ => (self.w2_s, self.scale_bytes()),
        };
        base.0 + e * stride as u64
    }
}

/// 2026-10-19: The resident path's kernels, scratch and reduction, beside a [`MoeV41`] whose
/// config it was built for.
pub struct ResidentMoe {
    gemm: KernelHandle,
    gemm_f32: KernelHandle,
    act_quant: KernelHandle,
    layout: BlobLayout,
    max_tokens: usize,
    blob: DevicePtr,
    /// 2026-10-19: `[max_tokens, dim]` bf16: the input rounded to E4M3 blocks.
    xq: DevicePtr,
    /// 2026-10-19: `[max_pos, inter]` bf16 each.
    gate: DevicePtr,
    up: DevicePtr,
    h: DevicePtr,
    reduce: TpReduce,
}

impl ResidentMoe {
    /// 2026-10-19: Scratch for `moe`'s config over `world` ranks.
    pub fn new(gpu: &dyn GpuBackend, moe: &MoeV41, world: usize) -> Result<Self> {
        let c = &moe.cfg;
        ensure!(
            c.dim.is_multiple_of(MX_BLOCK) && c.inter.is_multiple_of(MX_BLOCK),
            "resident experts need dim and inter in whole {MX_BLOCK}-value blocks (got {} / {})",
            c.dim,
            c.inter
        );
        let max_pos = c.max_tokens * c.topk;
        let layout = BlobLayout {
            n_routed: c.n_routed,
            max_pos,
        };
        let alloc = |bytes: usize| gpu.alloc(bytes.max(16));
        Ok(Self {
            gemm: gpu.kernel(MODULE_GROUPED, "moe_w4a16_grouped_gemm_ptrtable_e8m0")?,
            gemm_f32: gpu.kernel(
                MODULE_GROUPED,
                "moe_w4a16_grouped_gemm_ptrtable_e8m0_f32out",
            )?,
            act_quant: gpu.kernel("attn_v41", "attn_v41_act_quant_fp8")?,
            layout,
            max_tokens: c.max_tokens,
            blob: alloc(layout.bytes())?,
            xq: alloc(c.max_tokens * c.dim * 2)?,
            gate: alloc(max_pos * c.inter * 2)?,
            up: alloc(max_pos * c.inter * 2)?,
            h: alloc(max_pos * c.inter * 2)?,
            reduce: TpReduce::new(gpu, world, max_pos + c.max_tokens, c.dim)?,
        })
    }

    pub fn reduce(&self) -> &TpReduce {
        &self.reduce
    }

    fn at(&self, off: usize) -> DevicePtr {
        DevicePtr(self.blob.0 + off as u64)
    }

    pub fn free(self, gpu: &dyn GpuBackend) -> Result<()> {
        for p in [self.blob, self.xq, self.gate, self.up, self.h] {
            gpu.free(p)?;
        }
        self.reduce.free(gpu)
    }
}

/// 2026-10-19: The module of the grouped W4A16 GEMMs.
const MODULE_GROUPED: &str = "moe_w4a16";

impl MoeV41 {
    /// 2026-10-19: One layer's MoE for `m` tokens on resident experts. Returns the bf16
    /// `[m, dim]` output (the `out` buffer, identical on every rank) and the routing.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_resident(
        &self,
        gpu: &dyn GpuBackend,
        w: &MoeV41LayerWeights,
        rx: &ResidentMoe,
        ex: &ResidentExperts,
        comm: Option<&dyn CommBackend>,
        x: DevicePtr,
        m: usize,
        stream: u64,
    ) -> Result<(DevicePtr, Vec<f32>, Vec<usize>)> {
        let c = &self.cfg;
        ensure!(
            (ex.n_experts, ex.inter, ex.dim) == (c.n_routed, c.inter, c.dim),
            "resident experts [{} x {} x {}] for a layer of [{} x {} x {}]",
            ex.n_experts,
            ex.inter,
            ex.dim,
            c.n_routed,
            c.inter,
            c.dim
        );
        ensure!(
            m <= rx.max_tokens,
            "{m} tokens exceed the resident scratch's {}",
            rx.max_tokens
        );
        self.route_launch(gpu, w, x, m, stream)?;
        let (weights, indices) = self.route_select(gpu, w, m, stream)?;
        let plan = resident_plan(&indices, &weights, m, c.topk, c.n_routed)?;
        let l = rx.layout;
        gpu.copy_h2d_async(&l.encode(&plan, |t, e| ex.addr(t, e))?, rx.blob, stream)?;
        gpu.copy_d2d_async(x, rx.xq, m * c.dim * 2, stream)?;
        act_quant_fp8(gpu, rx.act_quant, rx.xq, m * c.dim, stream)?;

        let (na, n_pos) = (plan.experts.len() as u32, plan.n_pos());
        let tiles = plan.max_rows.div_ceil(PTRTABLE_M_TILE) as u32;
        let grouped = |kernel, a, table: usize, out, n: usize, k: usize, tokens| {
            ops::moe_w4a16_grouped_gemm_ptrtable(
                gpu,
                kernel,
                a,
                rx.at(l.ptrs(table)),
                rx.at(l.ptrs(table + 1)),
                rx.at(l.scale2()),
                out,
                rx.at(l.offsets()),
                tokens,
                na,
                n as u32,
                k as u32,
                tiles,
                stream,
            )
        };
        let tokens = rx.at(l.tokens());
        grouped(rx.gemm, rx.xq, 0, rx.gate, c.inter, c.dim, tokens)?;
        grouped(rx.gemm, rx.xq, 2, rx.up, c.inter, c.dim, tokens)?;
        self.launch_n(gpu, self.k.swiglu, n_pos * c.inter, stream, |k| {
            k.arg_ptr(rx.gate)
                .arg_ptr(rx.up)
                .arg_ptr(rx.at(l.weights()))
                .arg_ptr(rx.h)
                .arg_u32(n_pos as u32)
                .arg_u32(c.inter as u32)
                .arg_f32(c.swiglu_limit)
        })?;
        act_quant_fp8(gpu, rx.act_quant, rx.h, n_pos * c.inter, stream)?;
        let partial = rx.reduce.partial();
        grouped(
            rx.gemm_f32,
            rx.h,
            4,
            partial,
            c.dim,
            c.inter,
            DevicePtr::NULL,
        )?;
        let shared_rows = DevicePtr(partial.0 + (n_pos * c.dim * 4) as u64);
        self.shared_partial(gpu, w, x, m, shared_rows, stream)?;
        rx.reduce.reduce(
            gpu,
            comm,
            CombineRows {
                rows: n_pos + m,
                pos: rx.at(l.pos()),
                topk: c.topk,
                shared_base: n_pos,
                m,
            },
            self.out,
            stream,
        )?;
        Ok((self.out, weights, indices))
    }

    /// 2026-10-19: This rank's slice of the shared expert: w1 and w3 (bf16 out), the SwiGLU, and
    /// w2 as f32 partial sums into `out` (`[m, dim]`). Each weight is bf16 or FP8 blocks.
    fn shared_partial(
        &self,
        gpu: &dyn GpuBackend,
        w: &MoeV41LayerWeights,
        x: DevicePtr,
        m: usize,
        out: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let c = &self.cfg;
        let proj = |a, wt: ResidentMat, out, n: usize, k: usize, f32_out: bool| -> Result<()> {
            let (a, wt) = self.fp8.lower(gpu, a, wt, m, n, k, stream)?;
            let ResidentMat::Bf16(p) = wt else {
                anyhow::bail!("a resident-path shared-expert weight must be bf16 or FP8 blocks")
            };
            let weight = DenseWeight { weight: p };
            let (mu, nu, ku) = (m as u32, n as u32, k as u32);
            match (f32_out, m) {
                (false, 1) => ops::dense_gemv(gpu, self.k.gemv, a, &weight, out, nu, ku, stream),
                (false, _) => {
                    ops::dense_gemm(gpu, self.k.gemm, a, &weight, out, mu, nu, ku, stream)
                }
                (true, _) => {
                    ops::dense_gemm(gpu, self.k.gemm_f32out, a, &weight, out, mu, nu, ku, stream)
                }
            }
        };
        proj(x, w.shared_w1, self.sg, c.inter, c.dim, false)?;
        proj(x, w.shared_w3, self.su, c.inter, c.dim, false)?;
        self.launch_n(gpu, self.k.swiglu, m * c.inter, stream, |k| {
            k.arg_ptr(self.sg)
                .arg_ptr(self.su)
                .arg_ptr(DevicePtr(0))
                .arg_ptr(self.sh)
                .arg_u32(m as u32)
                .arg_u32(c.inter as u32)
                .arg_f32(c.swiglu_limit)
        })?;
        proj(self.sh, w.shared_w2, out, c.dim, c.inter, true)
    }
}
