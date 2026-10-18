// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The tensor-parallel DFlash2 drafter (`dflash_head::tp`, `--draft-tp-size`)
//! against the one-rank drafter, on one GPU: every rank of a TP=1 and a TP=3 drafter runs in
//! its own thread with an in-process communicator ([`LocalComm`]: the collectives as host
//! copies, the all-reduce summing the ranks' BF16 values in rank order in f32 and rounding
//! once), over the same weights and the same context taps.
//!
//! Owner: model-arch examples.
//! Invariants:
//! - Returns an error unless the TP=1 drafter's drafts equal the one-rank drafter's on every
//!   propose (the split plumbing, the header, the context shipping and the gathered top-16
//!   are then exact), and the TP=3 drafter completes every propose with every rank in step.
//! - On a tier with weight copies, returns an error unless every TP=3 rank's copies (the seven
//!   layer GEMMs, the LM head, and on `fp8-aux` / `nvfp4` the auxiliary GEMMs) equal, codes
//!   and scales, their rows or columns of the one-rank drafter's copies. Negative control: rank
//!   1's q_proj copy must differ from rank 0's rows, so the comparison can fail.
//! - Prints the TP=3 draft agreement with the one-rank drafter (position by position): the
//!   ranks sum BF16 partials of o_proj and down, so a draft can differ where two candidates
//!   are within that rounding; nothing else differs.
//!
//! The first argument is the `--draft-weight-quantization` tier every drafter builds with
//! (`declared`, `fp8`, `fp8-aux` or `nvfp4`); it is required.
//!
//! Shapes: the GLM-5.3 DFlash2 drafter's (hidden 4096, 32 query / 8 KV heads of 128, block 8,
//! DFlash2 conv and selector), with 2 layers, a 6144-wide MLP and a 16384-row vocab so the
//! five drafters fit in < 3 GB. Weights are uniform in +-sqrt(3 / K) (unit-variance
//! activations), norms 1. Each scenario proposes after context lengths that exercise the
//! inline header, its overflow and two `ctx_window` chunks.
//!
//! Run (GB10, < 3 GB of device memory):
//!   cargo run --release -p metrale-gpu-examples --features cuda,gpu-examples \
//!     --example dflash_tp_parity -- nvfp4

#[path = "common/dflash_copy_check.rs"]
mod copy_check;
#[path = "common/glm5next_rows_fixture.rs"]
mod fixture;
#[path = "common/dflash_tp_fixture.rs"]
#[allow(dead_code)]
mod tpf;

use std::sync::Arc;

use anyhow::{Result, ensure};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::dflash_head::tp;
use metrale_model_arch::dflash_head::{BlockDiffusionDraftHead, DraftWeightQuantization};
use metrale_model_layers::layer::ForwardContext;
use metrale_model_layers::speculative::{DraftProposer, ProposerState};
use tpf::{Dims, GAMMA, HIDDEN, LocalComm, Rng, TAPS, Tensors, append_ctx};

const MAX_SEQ: usize = 1024;
const DIMS: Dims = Dims {
    layers: 2,
    q_heads: 32,
    kv_heads: 8,
    inter: 6144,
    vocab: 16384,
};

/// 2026-10-10: One rank's drafter, its state and its communicator.
struct Rank {
    head: BlockDiffusionDraftHead,
    state: Box<dyn ProposerState>,
    comm: LocalComm,
}

/// 2026-10-10: Run one propose on every rank of `ranks` (rank 0 proposes, the others follow)
/// and return rank 0's drafts.
fn tp_propose(
    gpu: &Arc<MetraleCudaBackend>,
    ranks: &mut [Rank],
    token: u32,
    pos: usize,
) -> Result<Vec<u32>> {
    std::thread::scope(|s| {
        let handles: Vec<_> = ranks
            .iter_mut()
            .map(|r| {
                let gpu = gpu.clone();
                s.spawn(move || -> Result<Vec<u32>> {
                    gpu.bind_to_thread()?;
                    let f = fixture::Fwd::new(gpu.as_ref())?;
                    let ctx = ForwardContext {
                        comm: Some(&r.comm),
                        ..f.ctx(gpu.as_ref(), false, None)
                    };
                    let stream = gpu.default_stream();
                    if r.comm.rank == 0 {
                        r.head.propose(
                            token,
                            DevicePtr::NULL,
                            pos,
                            GAMMA - 1,
                            r.state.as_mut(),
                            &ctx,
                            stream,
                            None,
                            None,
                            None,
                        )
                    } else {
                        r.head
                            .propose_follower(Some(r.state.as_mut()), &ctx, stream)?;
                        Ok(Vec::new())
                    }
                })
            })
            .collect();
        let mut out = Vec::new();
        for (i, h) in handles.into_iter().enumerate() {
            let d = h
                .join()
                .map_err(|_| anyhow::anyhow!("rank {i} panicked"))??;
            if i == 0 {
                out = d;
            }
        }
        Ok(out)
    })
}

fn build_ranks(
    gpu: &Arc<MetraleCudaBackend>,
    t: &Tensors,
    world: usize,
    quant: DraftWeightQuantization,
) -> Result<Vec<Rank>> {
    LocalComm::group(gpu, world)
        .into_iter()
        .map(|comm| {
            let head = t.head(gpu.as_ref(), Some((comm.rank, world)), MAX_SEQ, quant)?;
            let state = head.alloc_state(gpu.as_ref())?;
            Ok(Rank { head, state, comm })
        })
        .collect()
}

/// 2026-10-10: Every TP=3 rank's weight copies against the one-rank drafter's
/// (`copy_check::check_rank_copies`), then the negative control.
fn check_copies(
    gpu: &dyn GpuBackend,
    whole: &BlockDiffusionDraftHead,
    ranks: &[Rank],
) -> Result<()> {
    let d = DIMS;
    let mut blocks = 0;
    for r in ranks {
        let p = tp::plan(
            d.q_heads,
            d.kv_heads,
            d.inter,
            d.vocab,
            ranks.len(),
            r.comm.rank,
        )?;
        blocks += copy_check::check_rank_copies(gpu, whole, &r.head, &p, whole.head_dim)?;
    }
    let (Some(rank0_rows), Some(rank1)) = (
        whole.layers[0].q_proj_q.as_ref(),
        ranks[1].head.layers[0].q_proj_q.as_ref(),
    ) else {
        anyhow::bail!("no q_proj copy for the negative control");
    };
    let lq = ranks[1].head.num_q_heads * whole.head_dim;
    let h = whole.hidden_size;
    let copy = |w, rows| copy_check::WeightCopy { w, rows, cols: h };
    ensure!(
        !copy_check::same_block(
            gpu,
            copy(rank0_rows, d.q_heads * whole.head_dim),
            (0, 0),
            copy(rank1, lq),
            (0, 0),
            (lq, h)
        )?,
        "negative control: rank 1's q_proj copy equals rank 0's rows; the comparison cannot fail"
    );
    println!(
        "weight copies: {blocks} blocks over {} ranks equal to the one-rank drafter's (codes and \
         scales); negative control differs",
        ranks.len()
    );
    Ok(())
}

fn main() -> Result<()> {
    // 2026-10-10: SAFETY: single-threaded here, before any thread or lever read. Every drafter
    // runs without CUDA graphs (the in-process collectives are host copies); a 256-row
    // ctx_window makes the longest context take two chunks.
    unsafe {
        std::env::set_var("METRALE_DFLASH_PROPOSE_NO_GRAPH", "1");
        std::env::set_var("METRALE_DFLASH_CTX_WINDOW", "256");
    }
    let quant = std::env::args()
        .nth(1)
        .ok_or_else(|| anyhow::anyhow!("usage: dflash_tp_parity <declared|fp8|fp8-aux|nvfp4>"))
        .and_then(|a| DraftWeightQuantization::parse(&a).map_err(anyhow::Error::msg))?;
    println!("--draft-weight-quantization {}", quant.name());
    let gpu = Arc::new(MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?);
    let mut rng = Rng(0x6466_6c61_7368_7470);
    let t = Tensors::new(gpu.as_ref(), &mut rng, DIMS, DIMS.vocab)?;
    let reference = t.head(gpu.as_ref(), None, MAX_SEQ, quant)?;
    let mut ref_state = reference.alloc_state(gpu.as_ref())?;
    let mut tp1 = build_ranks(&gpu, &t, 1, quant)?;
    let mut tp3 = build_ranks(&gpu, &t, 3, quant)?;
    if quant.quantizes_layers() {
        check_copies(gpu.as_ref(), &reference, &tp3)?;
    }

    // 2026-10-10: Context growth per propose: 40 rows (inline header), +3, +100 (overflow),
    // +300 (two ctx_window chunks), then +1 for the rest.
    let mut grow: Vec<usize> = vec![40, 3, 100, 300];
    grow.extend(std::iter::repeat_n(1, 16));
    let (mut pos, mut agree, mut total, mut exact_tp1) = (0usize, 0usize, 0usize, 0usize);
    let row = TAPS * HIDDEN * 2;
    for (step, &n) in grow.iter().enumerate() {
        let taps = rng.bf16_bytes(n * TAPS * HIDDEN, 1.0);
        ensure!(taps.len() == n * row);
        append_ctx(gpu.as_ref(), ref_state.as_mut(), &taps, pos)?;
        append_ctx(gpu.as_ref(), tp1[0].state.as_mut(), &taps, pos)?;
        append_ctx(gpu.as_ref(), tp3[0].state.as_mut(), &taps, pos)?;
        pos += n;
        let token = (rng.unit().abs() * 16000.0) as u32;
        let stream = gpu.default_stream();
        let f = fixture::Fwd::new(gpu.as_ref())?;
        let want = reference.propose(
            token,
            DevicePtr::NULL,
            pos,
            GAMMA - 1,
            ref_state.as_mut(),
            &f.ctx(gpu.as_ref(), false, None),
            stream,
            None,
            None,
            None,
        )?;
        let got1 = tp_propose(&gpu, &mut tp1, token, pos)?;
        ensure!(
            got1 == want,
            "step {step}: the TP=1 drafter drafted {got1:?}, the one-rank drafter {want:?}"
        );
        exact_tp1 += 1;
        let got3 = tp_propose(&gpu, &mut tp3, token, pos)?;
        ensure!(
            got3.len() == want.len(),
            "step {step}: TP=3 drafted {} tokens",
            got3.len()
        );
        let same = want.iter().zip(&got3).take_while(|(a, b)| a == b).count();
        agree += want.iter().zip(&got3).filter(|(a, b)| a == b).count();
        total += want.len();
        println!(
            "step {step:2}: ctx {pos:4}  one-rank {want:?}\n         TP=3 {got3:?} (prefix agreement {same}/{})",
            want.len()
        );
    }
    println!(
        "TP=1 equal to the one-rank drafter on {exact_tp1}/{} proposes; TP=3 draft agreement \
         {agree}/{total} ({:.1}%)",
        grow.len(),
        100.0 * agree as f64 / total as f64
    );
    Ok(())
}
