// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-27: The routed experts' W8A8 grouped GEMMs of `forward_prefill_fp8` on the native
//! e4m3 tensor-core MMA (`moe_w8a8_grouped_gemm_e4m3_gu` / `_dn`), in place of
//! `moe_w8a8_grouped_gemm_pm4`'s software E4M3 decode plus BF16 MMAs. Same E4M3 operands and
//! scales, same BF16 output; each MMA sums the same 16 products in the same K order and the
//! scale fold runs per 64 K as in PM4, so the output is bit-identical to PM4's (microbench:
//! 0 differing values on every routing and input distribution tested). 2.2-3.4x faster at
//! 1k-32k tokens.
//!
//! Owner: model-layers (MoE).
//! Invariants: runs only outside decode steps (`ctx.decode_step` keeps PM4, so decode is
//! untouched), and only for shapes the entry's `MoeE4m3Tile::fits` accepts.

use super::*;

/// 2026-09-27: Which routed projection a launch is: gate/up (N = inter, K = hidden, rows
/// gathered through `sorted_token_ids`) or down (N = hidden, K = inter, rows already sorted).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::layers::moe) enum E4m3Proj {
    GateUp,
    Down,
}

/// 2026-09-27: The two entry points and the SM count their grids are sized from. Handles
/// are 0 when the target lacks the module, or when `METRALE_NO_MOE_E4M3_GROUPED` is
/// present (PM4 then runs, for A/B measurement).
#[derive(Clone, Copy, Debug)]
pub(in crate::layers::moe) struct E4m3Kernels {
    gu: KernelHandle,
    dn: KernelHandle,
    sms: u32,
}

impl E4m3Kernels {
    pub(in crate::layers::moe) fn resolve(gpu: &dyn GpuBackend) -> Result<Self> {
        const MODULE: &str = "moe_w8a8_grouped_gemm_e4m3";
        let off = Self {
            gu: KernelHandle(0),
            dn: KernelHandle(0),
            sms: 0,
        };
        if std::env::var_os("METRALE_NO_MOE_E4M3_GROUPED").is_some() {
            return Ok(off);
        }
        let gu = crate::layers::try_target_kernel(gpu, MODULE, "moe_w8a8_grouped_gemm_e4m3_gu");
        let dn = crate::layers::try_target_kernel(gpu, MODULE, "moe_w8a8_grouped_gemm_e4m3_dn");
        if gu.0 == 0 || dn.0 == 0 {
            return Ok(off);
        }
        Ok(Self {
            gu,
            dn,
            sms: gpu.sm_count()?,
        })
    }

    fn pick(&self, proj: E4m3Proj) -> (KernelHandle, ops::MoeE4m3Tile) {
        match proj {
            E4m3Proj::GateUp => (self.gu, ops::MOE_E4M3_GU),
            E4m3Proj::Down => (self.dn, ops::MOE_E4M3_DN),
        }
    }
}

impl MoeLayer {
    /// 2026-09-27: Runs every `(weights, output)` pair of one routed projection (gate and up
    /// share a work-list: same offsets, NULL-ness and N) on the e4m3 kernel and returns true;
    /// returns false without launching when the handles, the work-list builder, the shape or
    /// the phase (a decode step) rule it out.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_e4m3_grouped(
        &self,
        proj: E4m3Proj,
        a_fp8: DevicePtr,
        a_scale: DevicePtr,
        projections: &[(&Fp8ExpertPtrTable, DevicePtr)],
        expert_offsets: DevicePtr,
        sorted_token_ids: DevicePtr,
        num_experts: u32,
        n: u32,
        k: u32,
        te: usize,
        fp8_scratch: &MoeFp8Scratch,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let (kernel, tile) = self.moe_e4m3.pick(proj);
        if kernel.0 == 0
            || self.moe_build_tile_worklist_k.0 == 0
            || ctx.decode_step
            || projections.is_empty()
            || !tile.fits(n, k)
        {
            return Ok(false);
        }
        let n_tiles = n / tile.n_tile;
        // 2026-09-27: At most ceil(te / m_tile) + num_experts M tiles, times n_tiles items of
        // two u32 words each.
        let cap_items =
            (te.div_ceil(tile.m_tile as usize) + num_experts as usize + 1) * n_tiles as usize;
        anyhow::ensure!(
            cap_items * 8 <= fp8_scratch.worklist_bytes,
            "e4m3 work-list of {cap_items} items exceeds the arena's {} bytes",
            fp8_scratch.worklist_bytes
        );
        if ctx.stats.once("log:moe_e4m3_grouped_prefill") {
            tracing::info!(
                "[metrale] MoE prefill: routed experts on the native e4m3 grouped GEMM \
                 (gate/up {:?}, down {:?}, {} SMs)",
                ops::MOE_E4M3_GU,
                ops::MOE_E4M3_DN,
                self.moe_e4m3.sms
            );
        }
        ops::moe_build_tile_worklist(
            ctx.gpu,
            self.moe_build_tile_worklist_k,
            expert_offsets,
            projections[0].0.weight_ptrs,
            fp8_scratch.worklist,
            fp8_scratch.total_tiles,
            num_experts,
            n_tiles,
            tile.m_tile,
            stream,
        )?;
        let grid = (self.moe_e4m3.sms * tile.ctas_per_sm).min(u32::try_from(cap_items)?);
        for &(weights, output) in projections {
            ops::moe_w8a8_grouped_gemm_e4m3(
                ctx.gpu,
                kernel,
                tile,
                a_fp8,
                a_scale,
                weights.weight_ptrs,
                weights.scale_ptrs,
                output,
                expert_offsets,
                sorted_token_ids,
                n,
                k,
                fp8_scratch.worklist,
                fp8_scratch.total_tiles,
                grid,
                stream,
            )?;
        }
        Ok(true)
    }
}
