// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: The screened LM head of the fused multi-rank K=2 verify (`METRALE_SCREEN_HEAD`,
//! default on; kernels and bound in `model-layers ops/screen_head.rs`).
//!
//! The full head of that verify zeroes the `[2, vocab]` logits, projects this rank's BF16 vocab
//! shard into them (`dense_gemv_bf16_batchm`), all-reduces 620 KB and runs `argmax_bf16` per
//! row on every rank. The screened head reads an INT8 copy of the shard instead, recomputes only
//! the rows a rigorous bound cannot rule out with the full head's own arithmetic, swaps 16 bytes
//! of argmax keys over the one-shot exchange and resolves the same two tokens into the same
//! scratch words on every rank. The logits are then not written; host reads of them recompute
//! what they read exactly (`screen_head_lazy.rs`).
//!
//! Which verifies take it: rank 0 asks for it per request (the scheduler's `screen_picks_ok`)
//! in the fused announcement (`FLAG_K2_SCREEN_HEAD`), only when this model built the screen; a
//! worker follows the flag. Every other head call is the full head.
//!
//! Owner: model-engine (LM head).
//! Invariants:
//! - The screen copy is built before the KV pool is sized, so the pool's budget charges it
//!   (`kv_sizing.rs` counts every placed byte); `METRALE_SCREEN_HEAD=0` never allocates it.
//! - `armed` is consumed by the next K=2 verify, armed or not; rank 0 arms it in the announce
//!   and a worker from the announcement's flag, so every rank's verify takes the same head and
//!   issues the same collectives.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use anyhow::{Context, Result, bail};
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_model_layers::layers::ops::{
    self, SCREEN_MAX_M, SCREEN_STAMP_BYTES, SCREEN_STATE_BYTES, ScreenKernels, ScreenShard,
};
use metrale_model_layers::layers::try_target_kernel;
use metrale_model_layers::weight_map::DenseWeight;

use super::lm_head_vocab_tp::{HeadRanks, head_vocab_shard};
use super::types::TransformerModel;

/// 2026-10-21: Rows a screened verify projects (the K=2 verify).
pub(crate) const SCREEN_ROWS: usize = 2;

/// 2026-10-21: The screen copy of this rank's head shard and its scratch.
pub(crate) struct ScreenHead {
    pub(super) kernels: ScreenKernels,
    pub(super) exact_id: KernelHandle,
    pub(super) shard: ScreenShard,
    /// 2026-10-21: `[SCREEN_ROWS, hidden]` BF16: the screened verify's normed rows, kept for the
    /// lazy logits.
    pub(super) stash: DevicePtr,
    /// 2026-10-21: Lazy single-logit reads land here.
    pub(super) probe: DevicePtr,
    pub(super) kappa: f32,
    armed: AtomicBool,
    /// 2026-10-21: The last K=2 verify was screened and no host head call came after it.
    pub(super) lazy: AtomicBool,
    pub(super) screened: AtomicU64,
    pub(super) materialized: AtomicU64,
}

impl ScreenHead {
    /// 2026-10-21: Take the arming for the verify about to run.
    pub(crate) fn take_armed(&self) -> bool {
        self.armed.swap(false, Ordering::Relaxed)
    }
}

/// 2026-10-21: Build the screen copy when the lever is on and this serve can use it: a BF16 head
/// (no NVFP4/FP8 head), a vocab-sharded head over two or more ranks, the fused K=2 wire asked
/// for, a one-shot exchange of at least the key bytes, a hidden size the kernels take, and the
/// kernels in this build. `None` (logged) otherwise. A weight the bound does not cover fails.
pub(crate) fn build_screen_head(
    gpu: &dyn GpuBackend,
    settings: &metrale_settings::InferenceServerSettings,
    config: &ModelConfig,
    head: &DenseWeight,
    bf16_head: bool,
    comm: Option<&dyn metrale_comm::CommBackend>,
) -> Result<Option<ScreenHead>> {
    if !settings.speculative.screen_head {
        tracing::info!("screened LM head: off (METRALE_SCREEN_HEAD=0)");
        return Ok(None);
    }
    let (v, k) = (config.vocab_size, config.hidden_size as u32);
    let world = comm.map(|c| (c.world_size(), c.rank()));
    let shard = head_vocab_shard(
        v,
        HeadRanks::Every,
        settings.kernels.no_lmhead_vocab_tp,
        world,
    );
    let kernels = ScreenKernels {
        scan: try_target_kernel(gpu, "screen_head", "screen_head_scan"),
        rescore: try_target_kernel(gpu, "screen_head", "screen_head_rescore"),
        resolve: try_target_kernel(gpu, "screen_head", "screen_head_resolve"),
    };
    let quant = try_target_kernel(gpu, "screen_head", "screen_head_quant_int8");
    let exact_id = try_target_kernel(gpu, "screen_head", "screen_head_exact_id");
    let why = if !bf16_head {
        Some("the head is not BF16")
    } else if shard.is_none() {
        Some("the head is not vocab-sharded")
    } else if !metrale_model_layers::speculative::k2_fused_requested() {
        Some("the fused K=2 wire is not asked for")
    } else if comm.is_none_or(|c| c.oneshot_max_bytes() < SCREEN_ROWS * 8) {
        Some("no one-shot exchange")
    } else if !ops::screen_shape_ok(SCREEN_ROWS as u32, k) {
        Some("the hidden size")
    } else if !kernels.loaded() || quant.0 == 0 || exact_id.0 == 0 {
        Some("the screen kernels are not in this build")
    } else {
        None
    };
    let (Some((begin, len)), None) = (shard, why) else {
        tracing::info!("screened LM head: not built ({})", why.unwrap_or("-"));
        return Ok(None);
    };
    let rows = len as u32;
    let stream = gpu.default_stream();
    let w = head.weight.offset(begin * k as usize * 2);
    let sh = ScreenShard {
        w,
        q: gpu.alloc(len * k as usize)?,
        s: gpu.alloc(len * k as usize / 32)?,
        upper: gpu.alloc(SCREEN_MAX_M as usize * len * 4)?,
        state: gpu.alloc(SCREEN_STATE_BYTES)?,
        rows,
        k,
        base: u32::try_from(begin)?,
    };
    let bad = gpu.alloc(4)?;
    gpu.memset(bad, 0, 4)?;
    ops::screen_head_quant_int8(gpu, quant, w, sh.q, sh.s, bad, rows, k, stream)?;
    gpu.synchronize(stream)?;
    let mut b = [0u8; 4];
    gpu.copy_d2h(bad, &mut b)?;
    gpu.free(bad)?;
    let flags = u32::from_le_bytes(b);
    if flags != 0 {
        bail!(
            "screened LM head: the screen copy of rows {begin}..{} is outside its bound (flags \
             {flags:#x}: 1 = a non-finite weight, 2 = a residual past s/2); METRALE_SCREEN_HEAD=0 \
             serves the full head",
            begin + len
        );
    }
    let mib = (len * k as usize + len * k as usize / 32 + SCREEN_MAX_M as usize * len * 4) as f64
        / (1024.0 * 1024.0);
    tracing::info!(
        "screened LM head: INT8 copy of head rows {begin}..{} ({mib:.1} MiB, placed before the KV \
         pool is sized; METRALE_SCREEN_HEAD=0 frees it)",
        begin + len
    );
    Ok(Some(ScreenHead {
        kernels,
        exact_id,
        shard: sh,
        stash: gpu.alloc(SCREEN_ROWS * k as usize * 2)?,
        probe: gpu.alloc(SCREEN_STAMP_BYTES)?,
        kappa: ops::screen_kappa(k),
        armed: AtomicBool::new(false),
        lazy: AtomicBool::new(false),
        screened: AtomicU64::new(0),
        materialized: AtomicU64::new(0),
    }))
}

impl TransformerModel {
    /// 2026-10-21: The screen, when built.
    pub(crate) fn screen_head(&self) -> Option<&ScreenHead> {
        self.device_argmax.screen.get()
    }

    /// 2026-10-21: Install the screen the factory built before the KV pool.
    pub(crate) fn install_screen_head(&self, screen: Option<ScreenHead>) -> Result<()> {
        if let Some(s) = screen {
            self.device_argmax
                .screen
                .set(s)
                .map_err(|_| anyhow::anyhow!("screened LM head installed twice"))?;
        }
        Ok(())
    }

    /// 2026-10-21: Whether a K=2 verify here may end in the screen: built, and no head feature
    /// the screen does not reproduce (token overlays, a softcap, another head dtype).
    fn screen_usable(&self) -> bool {
        self.screen_head().is_some()
            && self.overlays.is_none()
            && self.logit_softcap_kernel.0 == 0
            && self.lm_head_q6k.is_none()
            && self.lm_head_fp8.is_none()
            && self.lm_head_nvfp4.is_none()
            && self.dense_gemv_batchm_kernel.0 != 0
    }

    /// 2026-10-21: Rank 0, at the fused announcement: arm the next verify when `wanted` and the
    /// screen is usable; returns whether it armed (the announcement's flag).
    pub(crate) fn screen_arm(&self, wanted: bool) -> bool {
        let on = wanted && self.screen_usable();
        if let Some(s) = self.screen_head() {
            s.armed.store(on, Ordering::Relaxed);
        }
        on
    }

    /// 2026-10-21: A worker, from the announcement's flag: rank 0 armed, so this rank must too.
    pub(crate) fn screen_arm_from_block(&self, flag: bool) -> Result<()> {
        if flag && !self.screen_arm(true) {
            bail!(
                "rank 0 announced a screened K=2 verify, but this rank cannot run the screened \
                 head (METRALE_SCREEN_HEAD, head or kernels differ)"
            );
        }
        if !flag {
            self.screen_arm(false);
        }
        Ok(())
    }

    /// 2026-10-21: The head of the K=2 verify into the argmax words at the head of scratch:
    /// screened, or the full head and `argmax_bf16` per row. `graph`: under capture the exchange
    /// runs on the communicator's own stream, as the layers' exchanges do.
    pub(crate) fn verify_head_k2(
        &self,
        normed: DevicePtr,
        k: usize,
        screened: bool,
        graph: bool,
        stream: u64,
    ) -> Result<()> {
        let vocab = self.config.vocab_size;
        let argmax_out = self.buffers.scratch();
        if !screened {
            self.lm_head_batched(
                normed,
                k as u32,
                self.buffers.logits(),
                HeadRanks::Every,
                stream,
            )?;
            for t in 0..k {
                ops::argmax_bf16(
                    self.gpu.as_ref(),
                    self.argmax_kernel,
                    self.buffers.logits().offset(t * vocab * 2),
                    argmax_out.offset(t * 4),
                    vocab as u32,
                    stream,
                )?;
            }
            return Ok(());
        }
        let sh = self
            .screen_head()
            .context("screened verify without a screen")?;
        let comm = self
            .comm_ref()
            .context("screened verify without a communicator")?;
        let gpu = self.gpu.as_ref();
        let h = self.config.hidden_size;
        gpu.copy_d2d_async(normed, sh.stash, k * h * 2, stream)?;
        ops::screen_head_keys(
            gpu,
            &sh.kernels,
            &sh.shard,
            normed,
            k as u32,
            sh.kappa,
            stream,
        )?;
        let order = (!graph).then_some(stream);
        let peers = comm
            .exchange_partials(sh.shard.state.0, k * 8, order)?
            .context(
                "the screened head needs the one-shot exchange (--all-reduce-oneshot-max-kb)",
            )?;
        ops::screen_head_resolve(
            gpu,
            sh.kernels.resolve,
            sh.shard.state,
            DevicePtr(peers.recv),
            (peers.slot_elems * 2) as u64,
            (peers.world_size - 1) as u32,
            k as u32,
            argmax_out,
            stream,
        )?;
        // 2026-10-21: The stamp: the logits now hold no head output (`screen_head_lazy.rs`).
        gpu.memset_async(self.buffers.logits(), 0xFF, SCREEN_STAMP_BYTES, stream)
    }

    /// 2026-10-21: After every K=2 verify (host side, replayed or not): its logits are lazy
    /// exactly when it was screened.
    pub(crate) fn screen_after_verify(&self, screened: bool) {
        if let Some(s) = self.screen_head() {
            s.lazy.store(screened, Ordering::Relaxed);
            if screened && s.screened.fetch_add(1, Ordering::Relaxed) == 0 {
                tracing::info!(
                    "screened LM head ACTIVE: K=2 verify argmax from the INT8 screen and an \
                     exact rescore, same tokens (kill switch METRALE_SCREEN_HEAD=0)"
                );
            }
        }
    }
}
