// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Launch adapters of the paged decode attention contracts: the canonical case
//! (`metrale_accuracy::refs::attention`: `q`, the NHD `k_cache`/`v_cache` pages, `block_table`,
//! `seq_lens`, `sm_scale`) reaches the kernel through the engine's own launcher,
//! `ops::paged_decode_attn_bf16`, with the arguments the BF16 decode route passes
//! (`run_paged_decode/bf16_fp8.rs`): one CTA per (q head, sequence), q rows at stride
//! `q_heads * head_dim`, no sliding window.
//!
//! Owner: server CLI.
//! Invariants:
//! - The adapter is chosen by `case.launcher`; the symbol launched is `case.kernel`.
//! - The kernel reads `HDIM` (its build's head dim) elements per head whatever `head_dim` says;
//!   the target the runner loads is the one the point's instances run, so the two agree. A head
//!   dim no paged decode build uses is refused before the launch.

use anyhow::Result;
use metrale_accuracy::case::Case;
use metrale_accuracy::refs::attention::Geom;
use metrale_gpu_runtime::gpu::KernelHandle;
use metrale_model_layers::layers::ops;

use super::accuracy_adapters::{Adapter, not_runnable};
use super::accuracy_gpu::Dev;

/// 2026-10-09: The adapter of each launcher.
pub(crate) const ADAPTERS: &[(&str, Adapter)] = &[("paged_decode::paged_decode_attn", paged_bf16)];

/// 2026-10-09: Head dims the gb10 `paged_decode_attn.cu` builds instantiate (the family's
/// `head_dim` points: `HDIM` 256 by default, 128 by `-DHDIM=128`).
const HEAD_DIMS: [usize; 2] = [128, 256];

fn paged_bf16(dev: &mut Dev<'_>, case: &Case, kernel: KernelHandle) -> Result<Vec<u8>> {
    let g = Geom::of(case).map_err(not_runnable)?;
    if !HEAD_DIMS.contains(&g.head_dim) {
        return Err(not_runnable(format!(
            "head_dim {} is not a paged decode build ({HEAD_DIMS:?})",
            g.head_dim
        )));
    }
    let scale = case.scalar("sm_scale").map_err(not_runnable)?;
    let tensor = |n: &str| case.tensor(n).map_err(not_runnable);
    let q = dev.upload(tensor("q")?)?;
    let k = dev.upload(tensor("k_cache")?)?;
    let v = dev.upload(tensor("v_cache")?)?;
    let bt = dev.upload(tensor("block_table")?)?;
    let lens = dev.upload(tensor("seq_lens")?)?;
    let bytes = g.rows * g.width() * 2;
    let y = dev.output(bytes)?;
    ops::paged_decode_attn_bf16(
        dev.gpu,
        kernel,
        q,
        k,
        v,
        y,
        bt,
        lens,
        g.blocks as u32,
        g.rows as u32,
        g.q_heads as u32,
        g.kv_heads as u32,
        g.head_dim as u32,
        g.page as u32,
        scale as f32,
        g.width() as u32,
        0,
        dev.stream,
    )?;
    dev.read(y, bytes)
}
