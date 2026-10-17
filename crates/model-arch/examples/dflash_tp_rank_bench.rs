// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The compute of one rank of the tensor-parallel GLM-5.3 DFlash2 drafter's
//! propose: rank 0 of 3 at the real per-rank shapes (5 layers, 12 query / 3 KV heads of 128,
//! a 4096-wide MLP slice, 51648 LM-head rows, the DFlash2 conv and selector), alone on one
//! GPU with [`AloneComm`] (no collective moves data or waits; the all-gather copies on the
//! stream). The wall of a real propose adds the collectives' latency: the exchange, the `fc`
//! rows' broadcast and, per layer, two all-reduces, plus one all-gather.
//!
//! Owner: model-arch examples.
//! Invariants:
//! - Each propose adds 2 context rows (a C1 step that accepted one draft), after a first one
//!   over a 512-row prompt; the first 10 proposes (warm-up and graph capture) are not timed.
//! - Times are host walls of `propose` (it ends on the drafts' device-to-host copy), median
//!   and range over 50 proposes per draft count.
//!
//! The first argument is the `--draft-weight-quantization` tier (`declared`, `fp8`, `fp8-aux`
//! or `nvfp4`; required); `METRALE_DFLASH_TP_GRAPHS=1` in the environment runs the propose on
//! CUDA graphs. The rank's NVFP4 copies take their scales from its own tensors (the bench holds
//! no whole ones), which moves no byte count.
//!
//! Run (GB10, < 3 GB of device memory):
//!   cargo run --release -p metrale-model-arch --features cuda,gpu-examples \
//!     --example dflash_tp_rank_bench -- nvfp4

#[path = "common/glm5next_rows_fixture.rs"]
mod fixture;
#[path = "common/dflash_tp_fixture.rs"]
#[allow(dead_code)]
mod tpf;

use std::sync::Arc;

use anyhow::Result;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::dflash_head::DraftWeightQuantization;
use metrale_model_arch::dflash_head::tp::DflashTp;
use metrale_model_layers::layer::ForwardContext;
use metrale_model_layers::speculative::DraftProposer;
use tpf::{AloneComm, Dims, HIDDEN, Rng, TAPS, Tensors, append_ctx};

const MAX_SEQ: usize = 2048;
/// 2026-10-10: Rank 0 of the GLM-5.3 drafter over 3 ranks (`tp::plan`).
const RANK0: Dims = Dims {
    layers: 5,
    q_heads: 12,
    kv_heads: 3,
    inter: 4096,
    vocab: 51648,
};

fn main() -> Result<()> {
    // 2026-10-10: SAFETY: single-threaded, before any lever read: a 256-row ctx_window keeps
    // the scratch small (the 512-row prompt takes two chunks).
    unsafe {
        std::env::set_var("METRALE_DFLASH_CTX_WINDOW", "256");
    }
    let quant = std::env::args()
        .nth(1)
        .ok_or_else(|| anyhow::anyhow!("usage: dflash_tp_rank_bench <declared|fp8|fp8-aux|nvfp4>"))
        .and_then(|a| DraftWeightQuantization::parse(&a).map_err(anyhow::Error::msg))?;
    let gpu = Arc::new(MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?);
    let mut rng = Rng(0x7270_6265_6e63_6831);
    let t = Tensors::new(gpu.as_ref(), &mut rng, RANK0, 154_880)?;
    let tp = DflashTp {
        rank: 0,
        world: 3,
        vocab_base: 0,
    };
    let head = t.head_local(gpu.as_ref(), tp, MAX_SEQ, quant)?;
    let comm = AloneComm {
        world: 3,
        gpu: gpu.clone(),
    };
    let f = fixture::Fwd::new(gpu.as_ref())?;
    let ctx = ForwardContext {
        comm: Some(&comm),
        ..f.ctx(gpu.as_ref(), false, None)
    };
    let stream = gpu.default_stream();
    println!(
        "--draft-weight-quantization {}, METRALE_DFLASH_TP_GRAPHS={:?}",
        quant.name(),
        std::env::var("METRALE_DFLASH_TP_GRAPHS").unwrap_or_default(),
    );
    for drafts in [3usize, 7] {
        let mut state = head.alloc_state(gpu.as_ref())?;
        let mut pos = 0;
        let mut t_us = Vec::new();
        for step in 0..60 {
            let n = if step == 0 { 512 } else { 2 };
            append_ctx(
                gpu.as_ref(),
                state.as_mut(),
                &rng.bf16_bytes(n * TAPS * HIDDEN, 1.0),
                pos,
            )?;
            pos += n;
            gpu.synchronize(stream)?;
            let t0 = std::time::Instant::now();
            head.propose(
                (step * 37 % 16000) as u32,
                DevicePtr::NULL,
                pos,
                drafts,
                state.as_mut(),
                &ctx,
                stream,
                None,
                None,
                None,
            )?;
            if step >= 10 {
                t_us.push(t0.elapsed().as_secs_f64() * 1e6);
            }
        }
        t_us.sort_by(f64::total_cmp);
        println!(
            "rank 0 of 3, {drafts} drafts (block {}): propose median {:.0} us (min {:.0}, \
             max {:.0}) over {} proposes",
            drafts + 1,
            t_us[t_us.len() / 2],
            t_us[0],
            t_us[t_us.len() - 1],
            t_us.len()
        );
        head.free_state(gpu.as_ref(), state.as_mut())?;
    }
    Ok(())
}
