// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-13: The FFN of `n` decode rows, one arm per batch shape, with the
//! residual step injected: `residual_add` on the plain multi-sequence paths,
//! `hc_post_site` under a hyper-connection highway. The GDN and attention layers'
//! plain and highway multi-sequence paths all run [`ffn_rows`]; which arm runs is
//! decided by `ffn_rows_select` from the layer's capabilities.
//!
//! Owner: model-layers.
//! Invariants:
//! - Every arm writes its rows into `moe_output` (whole, per chunk or per row)
//!   and hands each chunk to `sink(rows, row0, m)` before the next chunk
//!   overwrites it. `rows` holds `m` contiguous BF16 rows of `hidden_size`, the
//!   rows `row0..row0 + m` of the batch.
//! - The sink is elementwise per row (a residual add or the highway injection),
//!   so the chunking is a launch count, never a numeric difference.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::FfnComponent;
use super::moe::GroupedRouting;
use crate::layer::ForwardContext;

/// 2026-10-13: The FFN arm of one multi-sequence decode step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FfnArm {
    /// 2026-10-13: A dense FFN under a fixed `--activation-quantization` (`forward_dense_fixed`).
    DenseFixed,
    /// 2026-10-13: An NVFP4 `--expert-quantization` tier's grouped decode (`forward_nvfp4_grouped`).
    Nvfp4Grouped,
    /// 2026-10-13: The FP8 MoE's grouped kernels with the per-row router: the bits of the
    /// per-row `forward` at every row count (`forward_fp8_grouped_decode_routed`).
    Fp8RowInvariant,
    /// 2026-10-13: The fused two-row kernel (`forward_k2`).
    K2,
    /// 2026-10-13: The fused three-row kernel (`forward_k3`).
    K3,
    /// 2026-10-13: All rows in one `forward_prefill` (each weight read once).
    Prefill,
    /// 2026-10-13: A dense FFN's batched GEMV over chunks of at most `chunk` rows (`try_forward_km`).
    Km { chunk: usize },
    /// 2026-10-13: The FP8 MoE grouped across all rows (`forward_fp8_grouped_decode`).
    Fp8Grouped,
    /// 2026-10-13: The C=4 MoE with FP32 atomic down-projection adds (`forward_atomic_c4_decode`).
    AtomicC4,
    /// 2026-10-13: Token-major MoE decode, `forward_batched` when it declines (`forward_token_major_decode`).
    TokenMajor,
    /// 2026-10-13: Even `n`: the batch two rows at a time through `forward_k2`.
    PairwiseK2,
    /// 2026-10-13: All rows in one `forward_batched`.
    Batched,
    /// 2026-10-13: One `forward` per row.
    PerRow,
}

/// 2026-10-13: The residual step of one chunk of FFN output rows:
/// `sink(rows, row0, m)`.
pub(crate) type RowSink<'s> = dyn FnMut(DevicePtr, usize, usize) -> Result<()> + 's;

/// 2026-10-13: Run `arm` over the `n` BF16 rows at `normed` and hand each chunk of
/// output rows to `sink` (see the module header).
pub(crate) fn ffn_rows(
    ffn: &FfnComponent,
    arm: FfnArm,
    normed: DevicePtr,
    n: usize,
    ctx: &ForwardContext,
    stream: u64,
    sink: &mut RowSink<'_>,
) -> Result<()> {
    let h = ctx.config.hidden_size;
    let bf16 = 2usize;
    let out = ctx.buffers.moe_output();
    match arm {
        FfnArm::DenseFixed => {
            ffn.forward_dense_fixed(normed, n, ctx, stream)?;
            sink(out, 0, n)
        }
        FfnArm::Nvfp4Grouped => {
            ffn.forward_nvfp4_grouped(normed, n, ctx, stream)?;
            sink(out, 0, n)
        }
        FfnArm::Fp8RowInvariant => {
            ffn.forward_fp8_grouped_decode_routed(normed, n, GroupedRouting::PerRow, ctx, stream)?;
            sink(out, 0, n)
        }
        FfnArm::K2 => {
            ffn.forward_k2(normed, ctx, stream)?;
            sink(out, 0, 2)
        }
        FfnArm::K3 => {
            ffn.forward_k3(normed, ctx, stream)?;
            sink(out, 0, 3)
        }
        FfnArm::Prefill => {
            ffn.forward_prefill(normed, n, ctx, stream)?;
            sink(out, 0, n)
        }
        FfnArm::Km { chunk } => {
            // 2026-10-13: Each chunk reuses `moe_output[0..m]`, so its sink runs
            // before the next chunk overwrites it.
            let mut done = 0usize;
            while done < n {
                let m = (n - done).min(chunk);
                let used =
                    ffn.try_forward_km(normed.offset(done * h * bf16), m as u32, ctx, stream)?;
                anyhow::ensure!(
                    used,
                    "ffn_rows: Km arm selected but can_forward_km({m}) is false"
                );
                sink(out, done, m)?;
                done += m;
            }
            Ok(())
        }
        FfnArm::Fp8Grouped => {
            ffn.forward_fp8_grouped_decode(normed, n, ctx, stream)?;
            sink(out, 0, n)
        }
        FfnArm::AtomicC4 => {
            ffn.forward_atomic_c4_decode(normed, n, ctx, stream)?;
            sink(out, 0, n)
        }
        FfnArm::TokenMajor => {
            // 2026-09-25: With a resident MoE adapter or unsupported expert
            // weights, `forward_token_major_decode` runs `forward_batched`
            // itself; any error it returns also falls back to `forward_batched`.
            if ffn
                .forward_token_major_decode(normed, n, ctx, stream)
                .is_err()
            {
                ffn.forward_batched(normed, n, ctx, stream)?;
            }
            sink(out, 0, n)
        }
        FfnArm::PairwiseK2 => {
            for pair in 0..n / 2 {
                let row0 = pair * 2;
                ffn.forward_k2(normed.offset(row0 * h * bf16), ctx, stream)?;
                sink(out, row0, 2)?;
            }
            Ok(())
        }
        FfnArm::Batched => {
            ffn.forward_batched(normed, n, ctx, stream)?;
            sink(out, 0, n)
        }
        FfnArm::PerRow => {
            for i in 0..n {
                let row = ffn.forward(normed.offset(i * h * bf16), ctx, stream)?;
                sink(row, i, 1)?;
            }
            Ok(())
        }
    }
}
