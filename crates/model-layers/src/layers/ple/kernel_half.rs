// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-13: The kernel half of a PLE step, shared by `forward` (one
//! sequence, `T` tokens) and `forward_rows` (`n` sequences, one token each):
//! the two projections, the gate, the dilated conv and the highway add. The
//! host half (ids, gather) is in `layer.rs` and `rows.rs`.
//!
//! Owner: model-layers (PLE).
//! Invariants: none beyond the types.

use anyhow::{Context, Result};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::PleLayer;
use crate::layers::ops;

impl PleLayer {
    /// 2026-10-13: `key = emb @ key_proj`, `value = emb @ value_proj` over
    /// `num_tokens` rows of `emb`. `gemm_k` is the pipelined kernel, so it must
    /// go through `ops::dense_gemm_bf16_pipelined` (grid [ceil(n,128),
    /// ceil(m,128)], block 256), not `ops::dense_gemm` (grid [ceil(n,16),
    /// ceil(m,16)], block 16x16): a kernel launched with the other wrapper's
    /// grid reads out of bounds.
    pub(super) fn project(
        &self,
        num_tokens: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let c = self.hc_mult * self.hidden;
        ops::dense_gemm_bf16_pipelined(
            gpu,
            self.gemm_k,
            self.emb,
            &self.key_proj,
            self.key,
            num_tokens as u32,
            c as u32,
            self.hidden as u32,
            stream,
        )
        .context("PLE key_proj")?;
        ops::dense_gemm_bf16_pipelined(
            gpu,
            self.gemm_k,
            self.emb,
            &self.value_proj,
            self.value,
            num_tokens as u32,
            self.hidden as u32,
            self.hidden as u32,
            stream,
        )
        .context("PLE value_proj")
    }

    /// 2026-10-13: The gate over `num_tokens` rows of `highway`, into `gated`
    /// and `gated_normed` (one block per token).
    pub(super) fn gate(
        &self,
        highway: DevicePtr,
        num_tokens: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        ops::ple_gate(
            gpu,
            self.gate_k,
            highway,
            self.key,
            self.value,
            self.norm_query.weight,
            self.norm_key.weight,
            self.norm_conv.weight,
            self.gated,
            self.gated_normed,
            num_tokens as u32,
            self.hidden as u32,
            self.hc_mult as u32,
            self.eps,
            stream,
        )
    }

    /// 2026-10-13: The dilated depthwise conv of `num_tokens` consecutive rows of
    /// one sequence against its `conv_state`, `out = gated + silu(conv)`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn conv(
        &self,
        gated_normed: DevicePtr,
        gated: DevicePtr,
        conv_state: DevicePtr,
        out: DevicePtr,
        num_tokens: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let c = self.hc_mult * self.hidden;
        ops::ple_conv(
            gpu,
            self.conv_k,
            gated_normed,
            gated,
            self.conv1d.weight,
            conv_state,
            out,
            num_tokens as u32,
            c as u32,
            self.k_size as u32,
            self.dilation as u32,
            stream,
        )
    }

    /// 2026-10-13: `highway += out` over `num_tokens` rows, in FP32.
    pub(super) fn add_highway(
        &self,
        highway: DevicePtr,
        num_tokens: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let c = self.hc_mult * self.hidden;
        ops::ple_add_highway(
            gpu,
            self.add_k,
            self.out,
            highway,
            (num_tokens * c) as u32,
            stream,
        )
    }
}
