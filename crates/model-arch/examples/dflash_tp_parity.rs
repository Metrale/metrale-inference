// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The tensor-parallel DFlash2 drafter (`dflash_head::tp`, `METRALE_DFLASH_TP=1`)
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
//! - Prints the TP=3 draft agreement with the one-rank drafter (position by position): the
//!   ranks sum BF16 partials of o_proj and down, so a draft can differ where two candidates
//!   are within that rounding; nothing else differs.
//!
//! Shapes: the GLM-5.3 DFlash2 drafter's (hidden 4096, 32 query / 8 KV heads of 128, block 8,
//! DFlash2 conv and selector), with 2 layers, a 6144-wide MLP and a 16384-row vocab so the
//! five drafters fit in < 3 GB. Weights are uniform in +-sqrt(3 / K) (unit-variance
//! activations), norms 1. Each scenario proposes after context lengths that exercise the
//! inline header, its overflow and two `ctx_window` chunks.
//!
//! Run (GB10, < 3 GB of device memory):
//!   cargo run --release -p metrale-model-arch --features cuda,gpu-examples \
//!     --example dflash_tp_parity

#[path = "common/glm5next_rows_fixture.rs"]
#[allow(dead_code)]
mod fixture;

use std::sync::{Arc, Barrier, Mutex};

use anyhow::{Result, bail, ensure};
use half::bf16;
use metrale_comm::CommBackend;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::dflash_head::BlockDiffusionDraftHead;
use metrale_model_arch::dflash_head::tp::{self, DflashTp};
use metrale_model_arch::weight_loader::dflash_loader::{
    DflashLayerWeights, DflashWeights, parse_dflash_config,
};
use metrale_model_layers::layer::ForwardContext;
use metrale_model_layers::speculative::{DraftProposer, ProposerState};
use metrale_model_layers::weight_map::DenseWeight;

const HIDDEN: usize = 4096;
const LAYERS: usize = 2;
const INTER: usize = 6144;
const VOCAB: usize = 16384;
const TAPS: usize = 5;
const MAX_SEQ: usize = 1024;
const GAMMA: usize = 8;
const CONFIG: &str = r#"{
  "hidden_size": 4096, "num_hidden_layers": 2, "intermediate_size": 6144,
  "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
  "vocab_size": 16384, "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "sliding_window": 2048,
  "dflash_config": {"block_size": 8, "mask_token_id": 16300,
    "target_layer_ids": [5, 14, 24, 33, 42], "selector_rank": 256, "selector_top_k": 16,
    "conv_kernel_size": 2, "conv_group_size": 16}
}"#;

struct Rng(u64);
impl Rng {
    fn unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn upload(gpu: &dyn GpuBackend, rng: &mut Rng, n: usize, amp: f32) -> Result<DevicePtr> {
    let bytes: Vec<u8> = (0..n)
        .flat_map(|_| bf16::from_f32(rng.unit() * amp).to_bits().to_le_bytes())
        .collect();
    let p = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(&bytes, p)?;
    Ok(p)
}

fn ones(gpu: &dyn GpuBackend, n: usize) -> Result<DevicePtr> {
    let bytes: Vec<u8> = (0..n)
        .flat_map(|_| bf16::ONE.to_bits().to_le_bytes())
        .collect();
    let p = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(&bytes, p)?;
    Ok(p)
}

/// 2026-10-10: The drafter's device tensors, shared by every drafter built from them.
#[derive(Clone, Copy)]
struct Tensors {
    fc: DevicePtr,
    norm: DevicePtr,
    head_norm: DevicePtr,
    layers: [[DevicePtr; 13]; LAYERS],
    pred: DevicePtr,
    succ: DevicePtr,
    hproj: DevicePtr,
    embed: DevicePtr,
    lm_head: DevicePtr,
}

impl Tensors {
    fn new(gpu: &dyn GpuBackend, rng: &mut Rng) -> Result<Self> {
        let amp = |k: usize| (3.0 / k as f32).sqrt();
        let (q, kv) = (32 * 128, 8 * 128);
        let mut layers = [[DevicePtr(0); 13]; LAYERS];
        for l in layers.iter_mut() {
            *l = [
                ones(gpu, HIDDEN)?,                                      // input_layernorm
                ones(gpu, HIDDEN)?,                                      // post_attention_layernorm
                upload(gpu, rng, q * HIDDEN, amp(HIDDEN))?,              // q_proj
                upload(gpu, rng, kv * HIDDEN, amp(HIDDEN))?,             // k_proj
                upload(gpu, rng, kv * HIDDEN, amp(HIDDEN))?,             // v_proj
                upload(gpu, rng, HIDDEN * q, amp(q))?,                   // o_proj
                ones(gpu, 128)?,                                         // q_norm
                ones(gpu, 128)?,                                         // k_norm
                upload(gpu, rng, INTER * HIDDEN, amp(HIDDEN))?,          // gate_proj
                upload(gpu, rng, INTER * HIDDEN, amp(HIDDEN))?,          // up_proj
                upload(gpu, rng, HIDDEN * INTER, amp(INTER))?,           // down_proj
                upload(gpu, rng, 2 * 2 * HIDDEN, 0.5)?,                  // conv base kernels (both)
                upload(gpu, rng, 2 * 1024 * HIDDEN, amp(HIDDEN) * 0.1)?, // conv projections
            ];
        }
        Ok(Self {
            fc: upload(gpu, rng, HIDDEN * TAPS * HIDDEN, amp(TAPS * HIDDEN))?,
            norm: ones(gpu, HIDDEN)?,
            head_norm: ones(gpu, HIDDEN)?,
            layers,
            pred: upload(gpu, rng, VOCAB * 256, 1.0)?,
            succ: upload(gpu, rng, VOCAB * 256, 1.0)?,
            hproj: upload(gpu, rng, 256 * HIDDEN, amp(HIDDEN))?,
            embed: upload(gpu, rng, VOCAB * HIDDEN, 1.0)?,
            lm_head: upload(gpu, rng, VOCAB * HIDDEN, amp(HIDDEN) * 4.0)?,
        })
    }

    fn weights(&self) -> Result<DflashWeights> {
        let w = |p: DevicePtr| DenseWeight { weight: p };
        let layers = self
            .layers
            .iter()
            .map(|l| DflashLayerWeights {
                input_layernorm: w(l[0]),
                post_attention_layernorm: w(l[1]),
                q_proj: w(l[2]),
                k_proj: w(l[3]),
                v_proj: w(l[4]),
                o_proj: w(l[5]),
                q_norm: w(l[6]),
                k_norm: w(l[7]),
                gate_proj: w(l[8]),
                up_proj: w(l[9]),
                down_proj: w(l[10]),
                attention_conv_base: Some(w(l[11])),
                attention_conv_proj: Some(w(l[12])),
                mlp_conv_base: Some(w(l[11])),
                mlp_conv_proj: Some(w(l[12].offset(1024 * HIDDEN * 2))),
                o_proj_cols: None,
                down_proj_cols: None,
            })
            .collect();
        Ok(DflashWeights {
            config: parse_dflash_config(CONFIG)?,
            fc: w(self.fc),
            hidden_norm: w(self.head_norm),
            norm: w(self.norm),
            layers,
            draft_id_to_target_id: None,
            markov_w1: None,
            markov_w2: None,
            confidence_proj: None,
            confidence_bias: None,
            selector_pred: Some(w(self.pred)),
            selector_succ: Some(w(self.succ)),
            selector_hidden_proj: Some(w(self.hproj)),
        })
    }

    /// 2026-10-10: A drafter for `tp` (`None`: the one-rank drafter).
    fn head(
        &self,
        gpu: &dyn GpuBackend,
        tp: Option<(usize, usize)>,
    ) -> Result<BlockDiffusionDraftHead> {
        let weights = self.weights()?;
        let (weights, lm_head, tp) = match tp {
            None => (weights, self.lm_head, None),
            Some((rank, world)) => {
                let p = tp::plan(32, 8, INTER, VOCAB, world, rank)?;
                (
                    tp::shard_weights(weights, &p, gpu)?,
                    self.lm_head.offset(p.vocab.0 * HIDDEN * 2),
                    Some(DflashTp {
                        rank,
                        world,
                        vocab_base: p.vocab.0,
                    }),
                )
            }
        };
        BlockDiffusionDraftHead::from_weights(
            weights,
            self.embed,
            lm_head,
            None,
            None,
            HIDDEN,
            Some(GAMMA),
            Some(2048),
            gpu,
            MAX_SEQ,
            1,
            tp,
        )
    }
}

/// 2026-10-10: Collectives between in-process ranks on one GPU: each operation copies the
/// ranks' buffers through the host behind a barrier.
struct LocalComm {
    rank: usize,
    world: usize,
    gpu: Arc<MetraleCudaBackend>,
    shared: Arc<(Barrier, Mutex<Vec<Vec<u8>>>)>,
}

impl LocalComm {
    fn read(&self, ptr: u64, bytes: usize) -> Result<Vec<u8>> {
        self.gpu.synchronize(self.gpu.default_stream())?;
        let mut b = vec![0u8; bytes];
        self.gpu.copy_d2h(DevicePtr(ptr), &mut b)?;
        Ok(b)
    }

    /// 2026-10-10: Deposit `mine`, wait for every rank, read every deposit, wait again.
    fn exchange(&self, mine: Vec<u8>) -> Vec<Vec<u8>> {
        self.shared.1.lock().unwrap()[self.rank] = mine;
        self.shared.0.wait();
        let all = self.shared.1.lock().unwrap().clone();
        self.shared.0.wait();
        all
    }
}

impl CommBackend for LocalComm {
    fn all_reduce(&self, ptr: u64, bytes: usize) -> Result<()> {
        let all = self.exchange(self.read(ptr, bytes)?);
        let out: Vec<u8> = (0..bytes / 2)
            .flat_map(|i| {
                let s: f32 = all
                    .iter()
                    .map(|b| bf16::from_le_bytes([b[2 * i], b[2 * i + 1]]).to_f32())
                    .sum();
                bf16::from_f32(s).to_le_bytes()
            })
            .collect();
        self.gpu.copy_h2d(&out, DevicePtr(ptr))
    }
    fn all_gather(&self, send: u64, recv: u64, bytes: usize) -> Result<()> {
        let all = self.exchange(self.read(send, bytes)?);
        self.gpu.copy_h2d(&all.concat(), DevicePtr(recv))
    }
    fn broadcast(&self, ptr: u64, bytes: usize, root: usize) -> Result<()> {
        let mine = if self.rank == root {
            self.read(ptr, bytes)?
        } else {
            Vec::new()
        };
        let all = self.exchange(mine);
        if self.rank != root {
            self.gpu.copy_h2d(&all[root], DevicePtr(ptr))?;
        }
        Ok(())
    }
    fn reduce_scatter(&self, _: u64, _: u64, _: usize) -> Result<()> {
        bail!("LocalComm: reduce_scatter is not used by the drafter")
    }
    fn barrier(&self) -> Result<()> {
        self.shared.0.wait();
        Ok(())
    }
    fn send_to(&self, _: u64, _: usize, _: usize, _: u64) -> Result<()> {
        bail!("LocalComm: send_to is not used by the drafter")
    }
    fn recv_from(&self, _: u64, _: usize, _: usize, _: u64) -> Result<()> {
        bail!("LocalComm: recv_from is not used by the drafter")
    }
    fn rank(&self) -> usize {
        self.rank
    }
    fn world_size(&self) -> usize {
        self.world
    }
}

/// 2026-10-10: Append `taps` rows (host BF16 bytes) at positions `pos0..` to a drafter state
/// that holds a context accumulator.
fn append_ctx(
    gpu: &dyn GpuBackend,
    st: &mut dyn ProposerState,
    taps: &[u8],
    pos0: usize,
) -> Result<()> {
    let d = st
        .as_any_mut()
        .downcast_mut::<metrale_model_arch::dflash_head::DflashProposerState>()
        .ok_or_else(|| anyhow::anyhow!("not a DFlash state"))?;
    let row = TAPS * HIDDEN * 2;
    let n = taps.len() / row;
    gpu.copy_h2d(taps, d.ctx_hidden_acc.offset(d.ctx_len * row))?;
    for i in 0..n {
        d.ctx_positions.push((pos0 + i) as i32);
    }
    d.ctx_len += n;
    Ok(())
}

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
                        r.head.propose_follower(r.state.as_mut(), &ctx, stream)?;
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

fn build_ranks(gpu: &Arc<MetraleCudaBackend>, t: &Tensors, world: usize) -> Result<Vec<Rank>> {
    let shared = Arc::new((Barrier::new(world), Mutex::new(vec![Vec::new(); world])));
    (0..world)
        .map(|rank| {
            let head = t.head(gpu.as_ref(), Some((rank, world)))?;
            let state = head.alloc_state(gpu.as_ref())?;
            Ok(Rank {
                head,
                state,
                comm: LocalComm {
                    rank,
                    world,
                    gpu: gpu.clone(),
                    shared: shared.clone(),
                },
            })
        })
        .collect()
}

fn main() -> Result<()> {
    // 2026-10-10: SAFETY: single-threaded here, before any thread or lever read. Every drafter
    // runs without CUDA graphs (the in-process collectives are host copies); a 256-row
    // ctx_window makes the longest context take two chunks. `METRALE_DFLASH_DRAFTER_FP8=0`
    // from the caller's environment selects the BF16 weights for every drafter alike.
    unsafe {
        std::env::set_var("METRALE_DFLASH_PROPOSE_NO_GRAPH", "1");
        std::env::set_var("METRALE_DFLASH_CTX_WINDOW", "256");
    }
    let gpu = Arc::new(MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?);
    let mut rng = Rng(0x6466_6c61_7368_7470);
    let t = Tensors::new(gpu.as_ref(), &mut rng)?;
    let reference = t.head(gpu.as_ref(), None)?;
    let mut ref_state = reference.alloc_state(gpu.as_ref())?;
    let mut tp1 = build_ranks(&gpu, &t, 1)?;
    let mut tp3 = build_ranks(&gpu, &t, 3)?;

    // 2026-10-10: Context growth per propose: 40 rows (inline header), +3, +100 (overflow),
    // +300 (two ctx_window chunks), then +1 for the rest.
    let mut grow: Vec<usize> = vec![40, 3, 100, 300];
    grow.extend(std::iter::repeat_n(1, 16));
    let (mut pos, mut agree, mut total, mut exact_tp1) = (0usize, 0usize, 0usize, 0usize);
    let row = TAPS * HIDDEN * 2;
    for (step, &n) in grow.iter().enumerate() {
        let taps: Vec<u8> = (0..n * TAPS * HIDDEN)
            .flat_map(|_| bf16::from_f32(rng.unit()).to_bits().to_le_bytes())
            .collect();
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
