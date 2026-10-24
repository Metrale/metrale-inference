// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-09-25: The per-stage launch wrappers every [`AttnV41`] path is built
//! from: the projection GEMM/GEMV (bf16 or `Q2_K`), RMSNorm, RoPE and the
//! fp4 query rounding and the cache-row packing.
//!
//! Owner: model-arch, DeepSeek-V4.1.
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use super::kv_codec::{RowPack, V41Row};
use super::{AttnMat, AttnTp, AttnV41, QUANT_BLOCKS_PER_LAUNCH};
use crate::deepseek_v41_ref::attn::FP8_BLOCK;
use metrale_model_layers::layers::ops;
use metrale_model_layers::layers::ops::{
    Q2K_MMQ_SMEM, kquant_mmq_gemm, kquant_mmvq_groups_w, kquant_mmvq_w, kquant_q8_1_rows,
};
use metrale_model_layers::weight_map::DenseWeight;

impl AttnV41 {
    /// 2026-10-19: `tp.partial = a . w^T` in f32 (`[m, n]`), for the cross-rank sum. Only bf16
    /// and FP8 block weights: the K-quant products have no f32-output form.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn gemm_f32_partial(
        &self,
        gpu: &dyn GpuBackend,
        tp: &AttnTp,
        a: DevicePtr,
        w: AttnMat,
        m: usize,
        n: usize,
        kk: usize,
        stream: u64,
    ) -> Result<()> {
        let (a, w) = self.fp8.lower(gpu, a, w, m, n, kk, stream)?;
        let AttnMat::Bf16(w) = w else {
            anyhow::bail!("a tensor-parallel output projection must be bf16 or FP8 blocks");
        };
        ops::dense_gemm(
            gpu,
            tp.gemm_f32out,
            a,
            &DenseWeight { weight: w },
            tp.reduce.partial(),
            m as u32,
            n as u32,
            kk as u32,
            stream,
        )
    }

    pub(super) fn gemm(
        &self,
        gpu: &dyn GpuBackend,
        a: DevicePtr,
        w: AttnMat,
        c: DevicePtr,
        m: usize,
        n: usize,
        kk: usize,
        stream: u64,
    ) -> Result<()> {
        let (a, w) = self.fp8.lower(gpu, a, w, m, n, kk, stream)?;
        let w = match w {
            AttnMat::Bf16(p) => p,
            AttnMat::Q3K(_) => anyhow::bail!("attention projections are never Q3_K"),
            AttnMat::Fp8Block { .. } => anyhow::bail!("an FP8 block projection was not lowered"),
            AttnMat::Q2K(blocks) => {
                let (m, n, kk) = (m as u32, n as u32, kk as u32);
                if m <= 8 {
                    kquant_q8_1_rows(gpu, self.k.q8_rows, a, self.a_q8, m, kk, stream)?;
                    return kquant_mmvq_w(
                        gpu,
                        self.k.mmvq_q2k_w,
                        blocks,
                        self.a_q8,
                        c,
                        n,
                        kk,
                        m,
                        stream,
                    );
                }
                ops::quantize_act_q8_1(gpu, self.k.quant_d2s6, a, self.a_q8, m, kk, stream)?;
                return kquant_mmq_gemm(
                    gpu,
                    self.k.mmq_q2k_nc,
                    self.k.mmq_q2k_wc,
                    self.a_q8,
                    blocks,
                    c,
                    m,
                    n,
                    kk,
                    Q2K_MMQ_SMEM,
                    stream,
                );
            }
        };
        if m == 1 {
            return ops::dense_gemv(
                gpu,
                self.k.gemv,
                a,
                &DenseWeight { weight: w },
                c,
                n as u32,
                kk as u32,
                stream,
            );
        }
        ops::dense_gemm(
            gpu,
            self.k.gemm,
            a,
            &DenseWeight { weight: w },
            c,
            m as u32,
            n as u32,
            kk as u32,
            stream,
        )
    }

    /// 2026-09-25: `og = wo_a(o_rot)` for all `groups` output groups at once
    /// (`m <= 8`, Q2_K `wo_a`): one q8_1 pass over the whole `[m, n_heads *
    /// head_dim]` row and one grouped launch writing `og`, where
    /// `wo_a_per_group` slices, projects and scatters each group.
    pub(super) fn wo_a_grouped(
        &self,
        gpu: &dyn GpuBackend,
        blocks: DevicePtr,
        o_rot: DevicePtr,
        m: usize,
        stream: u64,
    ) -> Result<()> {
        let c = &self.cfg;
        let width = (c.n_heads * c.head_dim) as u32;
        kquant_q8_1_rows(
            gpu,
            self.k.q8_rows,
            o_rot,
            self.a_q8,
            m as u32,
            width,
            stream,
        )?;
        kquant_mmvq_groups_w(
            gpu,
            self.k.mmvq_q2k_groups_w,
            blocks,
            self.a_q8,
            self.og,
            c.o_rank as u32,
            c.gw() as u32,
            m as u32,
            c.groups as u32,
            stream,
        )
    }

    pub(super) fn rmsnorm(
        &self,
        gpu: &dyn GpuBackend,
        f32_in: bool,
        x: DevicePtr,
        w: DevicePtr,
        out: DevicePtr,
        rows: usize,
        dim: usize,
        stream: u64,
    ) -> Result<()> {
        if rows == 0 {
            return Ok(());
        }
        KernelLaunch::new(
            gpu,
            if f32_in {
                self.k.rmsnorm_f32
            } else {
                self.k.rmsnorm_bf16
            },
        )
        .grid([rows as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(x)
        .arg_ptr(w)
        .arg_ptr(out)
        .arg_u32(dim as u32)
        .arg_f32(self.cfg.eps)
        .launch(stream)
    }

    pub(super) fn rope(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        pos: DevicePtr,
        rows: usize,
        row_len: usize,
        yarn: bool,
        inverse: bool,
        stream: u64,
    ) -> Result<()> {
        if rows == 0 {
            return Ok(());
        }
        KernelLaunch::new(gpu, self.k.rope)
            .grid([rows as u32, 1, 1])
            .block([(self.cfg.rope_dim / 2).max(32) as u32, 1, 1])
            .arg_ptr(x)
            .arg_ptr(pos)
            .arg_ptr(if yarn { self.fc_yarn } else { self.fc_plain })
            .arg_u32(row_len as u32)
            .arg_u32(self.cfg.rope_dim as u32)
            .arg_u32(inverse as u32)
            .launch(stream)
    }

    /// 2026-10-19: `rows` bf16 rows of `values` at `x`, quantised as the reference does, into
    /// `row`'s packed rows at `out`, with the packing kernel resolved for its codec at init.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn pack_rows(
        &self,
        gpu: &dyn GpuBackend,
        row: &V41Row,
        x: DevicePtr,
        rows: usize,
        values: usize,
        out: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let group = match row.pack {
            RowPack::Fp8E8m0 => FP8_BLOCK,
            RowPack::Fp4 { group, .. } => usize::try_from(group)?,
        };
        let n_blocks = rows * values / group;
        if n_blocks == 0 {
            return Ok(());
        }
        let grid = [(n_blocks as u32).div_ceil(QUANT_BLOCKS_PER_LAUNCH), 1, 1];
        let launch = match row.pack {
            RowPack::Fp8E8m0 => KernelLaunch::new(gpu, self.k.kv_pack_fp8)
                .grid(grid)
                .block([256, 1, 1])
                .arg_ptr(x)
                .arg_u32(n_blocks as u32),
            RowPack::Fp4 { e4m3_scale, .. } => KernelLaunch::new(gpu, self.k.kv_pack_fp4)
                .grid(grid)
                .block([256, 1, 1])
                .arg_ptr(x)
                .arg_u32(n_blocks as u32)
                .arg_u32(group as u32)
                .arg_u32(u32::from(e4m3_scale)),
        };
        launch.arg_ptr(out).arg_u32(values as u32).launch(stream)
    }

    pub(super) fn fp4_quant(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        n_values: usize,
        block: usize,
        e4m3_scale: bool,
        stream: u64,
    ) -> Result<()> {
        let n_blocks = n_values / block;
        if n_blocks == 0 {
            return Ok(());
        }
        KernelLaunch::new(gpu, self.k.fp4_quant)
            .grid([(n_blocks as u32).div_ceil(QUANT_BLOCKS_PER_LAUNCH), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(x)
            .arg_u32(n_blocks as u32)
            .arg_u32(block as u32)
            .arg_u32(e4m3_scale as u32)
            .launch(stream)
    }
}
