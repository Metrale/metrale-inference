// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The fixture the tensor-parallel DFlash2 drafter examples share
//! (`dflash_tp_parity`, `dflash_tp_rank_bench`): a drafter of given dims on random weights,
//! drafters built from it whole, split or already local, and two in-process communicators.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.

use std::sync::{Arc, Barrier, Mutex};

use anyhow::{Result, bail};
use half::bf16;
use metrale_comm::CommBackend;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::dflash_head::tp::{self, DflashTp};
use metrale_model_arch::dflash_head::{BlockDiffusionDraftHead, DraftWeightQuantization};
use metrale_model_arch::weight_loader::dflash_loader::{
    DflashLayerWeights, DflashWeights, parse_dflash_config,
};
use metrale_model_layers::speculative::ProposerState;
use metrale_model_layers::weight_map::DenseWeight;

pub const HIDDEN: usize = 4096;
pub const TAPS: usize = 5;
pub const GAMMA: usize = 8;
/// 2026-10-10: The embedding rows uploaded: every token id an example uses is below it (the
/// mask token is 16300).
pub const EMBED_ROWS: usize = 16384;

/// 2026-10-10: A drafter's shape: its layer count, query / KV heads (of 128), MLP width and
/// LM-head rows.
#[derive(Clone, Copy, Debug)]
pub struct Dims {
    pub layers: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub inter: usize,
    pub vocab: usize,
}

impl Dims {
    fn config(&self) -> String {
        format!(
            r#"{{"hidden_size": {HIDDEN}, "num_hidden_layers": {}, "intermediate_size": {},
            "num_attention_heads": {}, "num_key_value_heads": {}, "head_dim": 128,
            "vocab_size": {}, "rms_norm_eps": 1e-5, "rope_theta": 10000.0,
            "sliding_window": 2048,
            "dflash_config": {{"block_size": 8, "mask_token_id": 16300,
              "target_layer_ids": [5, 14, 24, 33, 42], "selector_rank": 256,
              "selector_top_k": 16, "conv_kernel_size": 2, "conv_group_size": 16}}}}"#,
            self.layers, self.inter, self.q_heads, self.kv_heads, self.vocab
        )
    }
}

pub struct Rng(pub u64);
impl Rng {
    pub fn unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    /// 2026-10-10: `n` BF16 values uniform in +-`amp`, as bytes.
    pub fn bf16_bytes(&mut self, n: usize, amp: f32) -> Vec<u8> {
        (0..n)
            .flat_map(|_| bf16::from_f32(self.unit() * amp).to_bits().to_le_bytes())
            .collect()
    }
}

fn upload(gpu: &dyn GpuBackend, rng: &mut Rng, n: usize, amp: f32) -> Result<DevicePtr> {
    let bytes = rng.bf16_bytes(n, amp);
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

/// 2026-10-10: The drafter's device tensors (uniform in +-sqrt(3 / K), so activations stay
/// near unit variance; norms 1), shared by every drafter built from them.
pub struct Tensors {
    pub dims: Dims,
    fc: DevicePtr,
    norm: DevicePtr,
    head_norm: DevicePtr,
    layers: Vec<[DevicePtr; 13]>,
    pred: DevicePtr,
    succ: DevicePtr,
    hproj: DevicePtr,
    embed: DevicePtr,
    lm_head: DevicePtr,
}

impl Tensors {
    /// 2026-10-10: `pred`/`succ` hold `codebook_rows` rows (the global vocab the selector
    /// indexes, which a local drafter's `dims.vocab` does not cover).
    pub fn new(
        gpu: &dyn GpuBackend,
        rng: &mut Rng,
        dims: Dims,
        codebook_rows: usize,
    ) -> Result<Self> {
        let amp = |k: usize| (3.0 / k as f32).sqrt();
        let (q, kv) = (dims.q_heads * 128, dims.kv_heads * 128);
        let mut layers = Vec::with_capacity(dims.layers);
        for _ in 0..dims.layers {
            layers.push([
                ones(gpu, HIDDEN)?,
                ones(gpu, HIDDEN)?,
                upload(gpu, rng, q * HIDDEN, amp(HIDDEN))?,
                upload(gpu, rng, kv * HIDDEN, amp(HIDDEN))?,
                upload(gpu, rng, kv * HIDDEN, amp(HIDDEN))?,
                upload(gpu, rng, HIDDEN * q, amp(q))?,
                ones(gpu, 128)?,
                ones(gpu, 128)?,
                upload(gpu, rng, dims.inter * HIDDEN, amp(HIDDEN))?,
                upload(gpu, rng, dims.inter * HIDDEN, amp(HIDDEN))?,
                upload(gpu, rng, HIDDEN * dims.inter, amp(dims.inter))?,
                upload(gpu, rng, 2 * 2 * HIDDEN, 0.5)?,
                upload(gpu, rng, 2 * 1024 * HIDDEN, amp(HIDDEN) * 0.1)?,
            ]);
        }
        Ok(Self {
            dims,
            fc: upload(gpu, rng, HIDDEN * TAPS * HIDDEN, amp(TAPS * HIDDEN))?,
            norm: ones(gpu, HIDDEN)?,
            head_norm: ones(gpu, HIDDEN)?,
            layers,
            pred: upload(gpu, rng, codebook_rows * 256, 1.0)?,
            succ: upload(gpu, rng, codebook_rows * 256, 1.0)?,
            hproj: upload(gpu, rng, 256 * HIDDEN, amp(HIDDEN))?,
            embed: upload(gpu, rng, EMBED_ROWS * HIDDEN, 1.0)?,
            lm_head: upload(gpu, rng, dims.vocab * HIDDEN, amp(HIDDEN) * 4.0)?,
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
                tp_src: None,
            })
            .collect();
        Ok(DflashWeights {
            config: parse_dflash_config(&self.dims.config())?,
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
            tp_lm_head: None,
        })
    }

    /// 2026-10-10: A drafter with `quant`'s weight copies: whole (`split` `None`), or rank `r`
    /// of `world` cut from these whole tensors (`tp::plan` + `tp::shard_weights`).
    pub fn head(
        &self,
        gpu: &dyn GpuBackend,
        split: Option<(usize, usize)>,
        max_seq: usize,
        quant: DraftWeightQuantization,
    ) -> Result<BlockDiffusionDraftHead> {
        let weights = self.weights()?;
        let d = self.dims;
        let (weights, lm_head, tp) = match split {
            None => (weights, self.lm_head, None),
            Some((rank, world)) => {
                let p = tp::plan(d.q_heads, d.kv_heads, d.inter, d.vocab, world, rank)?;
                let (weights, lm_head) = tp::shard_weights(weights, &p, self.lm_head, gpu)?;
                (
                    weights,
                    lm_head,
                    Some(DflashTp {
                        rank,
                        world,
                        vocab_base: p.vocab.0,
                    }),
                )
            }
        };
        self.build(gpu, weights, lm_head, tp, max_seq, quant)
    }

    /// 2026-10-10: A drafter whose tensors already are one rank's part (`tp`): for timing a
    /// rank of a drafter too large to hold whole. Its NVFP4 copies take their scales from the
    /// rank's own tensors (it has no whole ones), which moves no byte count.
    pub fn head_local(
        &self,
        gpu: &dyn GpuBackend,
        tp: DflashTp,
        max_seq: usize,
        quant: DraftWeightQuantization,
    ) -> Result<BlockDiffusionDraftHead> {
        self.build(gpu, self.weights()?, self.lm_head, Some(tp), max_seq, quant)
    }

    fn build(
        &self,
        gpu: &dyn GpuBackend,
        weights: DflashWeights,
        lm_head: DevicePtr,
        tp: Option<DflashTp>,
        max_seq: usize,
        quant: DraftWeightQuantization,
    ) -> Result<BlockDiffusionDraftHead> {
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
            max_seq,
            1,
            tp,
            quant,
            &metrale_settings::groups::device::declared()?,
        )
    }
}

/// 2026-10-10: Append `taps` rows (host BF16 bytes) at positions `pos0..` to a drafter state
/// that holds a context accumulator.
pub fn append_ctx(
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
    gpu.copy_h2d(taps, d.ctx_hidden_acc.offset(d.ctx_len * row))?;
    for i in 0..taps.len() / row {
        d.ctx_positions.push((pos0 + i) as i32);
    }
    d.ctx_len += taps.len() / row;
    Ok(())
}

/// 2026-10-10: Collectives between in-process ranks on one GPU: each operation copies the
/// ranks' buffers through the host behind a barrier; the all-reduce sums the ranks' BF16
/// values in rank order in f32 and rounds once.
pub struct LocalComm {
    pub rank: usize,
    pub world: usize,
    pub gpu: Arc<MetraleCudaBackend>,
    pub shared: Arc<(Barrier, Mutex<Vec<Vec<u8>>>)>,
}

impl LocalComm {
    /// 2026-10-10: The communicators of `world` in-process ranks.
    pub fn group(gpu: &Arc<MetraleCudaBackend>, world: usize) -> Vec<LocalComm> {
        let shared = Arc::new((Barrier::new(world), Mutex::new(vec![Vec::new(); world])));
        (0..world)
            .map(|rank| LocalComm {
                rank,
                world,
                gpu: gpu.clone(),
                shared: shared.clone(),
            })
            .collect()
    }

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

/// 2026-10-10: One rank of a `world`-rank group running alone, for timing its compute: the
/// all-reduce and the broadcasts do nothing, the all-gather fills every rank's slot with this
/// rank's data on the stream (no host copy, so it captures in a graph).
pub struct AloneComm {
    pub world: usize,
    pub gpu: Arc<MetraleCudaBackend>,
}

impl CommBackend for AloneComm {
    fn all_reduce(&self, _: u64, _: usize) -> Result<()> {
        Ok(())
    }
    fn all_gather(&self, send: u64, recv: u64, bytes: usize) -> Result<()> {
        for r in 0..self.world {
            self.gpu.copy_d2d_async(
                DevicePtr(send),
                DevicePtr(recv + (r * bytes) as u64),
                bytes,
                self.gpu.default_stream(),
            )?;
        }
        Ok(())
    }
    fn broadcast(&self, _: u64, _: usize, _: usize) -> Result<()> {
        Ok(())
    }
    fn reduce_scatter(&self, _: u64, _: u64, _: usize) -> Result<()> {
        bail!("AloneComm: reduce_scatter is not used by the drafter")
    }
    fn barrier(&self) -> Result<()> {
        Ok(())
    }
    fn send_to(&self, _: u64, _: usize, _: usize, _: u64) -> Result<()> {
        bail!("AloneComm: send_to is not used by the drafter")
    }
    fn recv_from(&self, _: u64, _: usize, _: usize, _: u64) -> Result<()> {
        bail!("AloneComm: recv_from is not used by the drafter")
    }
    fn rank(&self) -> usize {
        0
    }
    fn world_size(&self) -> usize {
        self.world
    }
}
