// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-09-25: [`EngramV41::apply`]: the `wkv` projection of the rows in the
//! workspace (single token: off the raw Q2_K blocks when the layer carries
//! them, else the bf16 GEMV; several tokens: the bf16 GEMM) and the gate onto
//! the highway.
//! 2026-10-19: An E4M3 `wkv` is projected by the GEMV/GEMM that decode it in the kernel, after
//! the rows are rounded to E4M3 blocks when the weight says so.
//!
//! Owner: model-arch (DeepSeek-V4.1 engram).
//! Invariants:
//! - `apply` refuses more than `max_tokens` tokens, a layer without weights,
//!   a `wkv` in a K-quant form, and a Q2_K `in_features` that is not a
//!   multiple of 256, before any launch.

use anyhow::{Context, Result, bail, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use super::EngramV41;
use crate::fp8_block_v41::act_quant_fp8;
use metrale_model_layers::layers::ops::{self, ResidentMat};
use metrale_model_layers::weight_map::DenseWeight;

impl EngramV41 {
    /// 2026-09-25: Project the rows already in the workspace and gate `tokens`
    /// positions of `streams` (`[T, hc, dim]` f32) in place, for model layer
    /// `layer`. Enqueued on `stream`.
    pub fn apply(
        &self,
        gpu: &dyn GpuBackend,
        layer: usize,
        streams: DevicePtr,
        tokens: usize,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            tokens <= self.max_tokens,
            "engram: {tokens} tokens exceeds the {} workspace",
            self.max_tokens
        );
        let w = self
            .layer(layer)
            .with_context(|| format!("engram: layer {layer} has no weights"))?;
        let (rows, n, k) = (self.rows_of(w), self.out_features(), self.in_features());
        match w.wkv {
            ResidentMat::Fp8Block {
                w: codes,
                scale,
                qdq,
            } => {
                if qdq {
                    act_quant_fp8(gpu, self.act_quant_k, rows, tokens * k, stream)?;
                }
                self.fp8
                    .project(gpu, rows, codes, scale, self.kv, tokens, n, k, stream)?;
            }
            ResidentMat::Bf16(p) => self.project_bf16(gpu, w.wkv_q2k, p, rows, tokens, stream)?,
            other => bail!("engram: layer {layer} wkv {other:?} has no projection"),
        }
        KernelLaunch::new(gpu, self.gate_k)
            .grid([tokens as u32, self.hc as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(streams)
            .arg_ptr(self.kv)
            .arg_ptr(w.qk)
            .arg_u32(self.dim as u32)
            .arg_u32(self.hc as u32)
            .arg_f32(self.eps)
            .launch(stream)
    }

    /// 2026-10-19: The projection off a bf16 `wkv` (`wkv_q2k`: its raw Q2_K blocks, or null).
    fn project_bf16(
        &self,
        gpu: &dyn GpuBackend,
        wkv_q2k: DevicePtr,
        wkv: DevicePtr,
        rows: DevicePtr,
        tokens: usize,
        stream: u64,
    ) -> Result<()> {
        let wkv = DenseWeight { weight: wkv };
        let (n, k) = (self.out_features() as u32, self.in_features() as u32);
        if tokens == 1 && !wkv_q2k.is_null() {
            ensure!(
                k.is_multiple_of(256),
                "engram: in_features {k} is not Q2_K blocks"
            );
            KernelLaunch::new(gpu, self.gemv_q2k_k)
                .grid([n.div_ceil(4), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(rows)
                .arg_ptr(wkv_q2k)
                .arg_ptr(self.kv)
                .arg_u32(n)
                .arg_u32(k)
                .launch(stream)
        } else if tokens == 1 {
            ops::dense_gemv(gpu, self.gemv_k, rows, &wkv, self.kv, n, k, stream)
        } else {
            ops::dense_gemm(
                gpu,
                self.gemm_k,
                rows,
                &wkv,
                self.kv,
                tokens as u32,
                n,
                k,
                stream,
            )
        }
    }
}
