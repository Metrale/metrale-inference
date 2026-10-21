// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The sharded device argmax at the end of a one-row decode step
//! (`METRALE_EP_DEVICE_ARGMAX=1`, multi-rank, v3 wire, vocab-sharded BF16 head).
//!
//! Without it, the vocab-parallel head zeroes the logits row, projects this rank's slice into
//! it, all-reduces the 310 KB row, and rank 0 runs a one-block argmax. With it, each rank keys
//! its slice (`argmax_key_shard_bf16`), the ranks swap their 16-byte keys in a one-shot
//! exchange, and every rank resolves the same token into its feed cell (`argmax_key_resolve`).
//! The token is bit for bit the one `argmax_bf16_batch_feed` gives over the whole row under the
//! step's row mask (`metrale_sampling::feed_argmax_key`). The logits row is then not complete,
//! so a step is armed only when its readback is the device argmax.
//!
//! Rank 0's router arms the next step (`arm`), uploading the row mask; a worker arms from the
//! command block's flag and mask (`arm_from_block`). The step takes the arming when it starts
//! (`begin_step`); its graph key carries the choice.
//!
//! Owner: model-engine decode.
//! Invariants:
//! - A step ends in the device argmax on every rank or on none: rank 0 sends the flag in the
//!   step's command block, and a worker arms only from it.
//! - `armed` is consumed by the next decode step, armed or not.

use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, bail};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_model_layers::ep_wire::{self, EpWire};
use metrale_model_layers::layers::ops;
use metrale_model_layers::layers::try_target_kernel;
use parking_lot::Mutex;

use super::lm_head_vocab_tp::HeadRanks;
use super::types::TransformerModel;
use crate::traits::RowMask;

/// 2026-10-10: Bytes of the key pair `[p, q]`.
const KEY_BYTES: usize = 16;

pub(crate) struct DeviceArgmax {
    shard_kernel: KernelHandle,
    resolve_kernel: KernelHandle,
    /// 2026-10-10: `[p, q]` u64 keys of this rank's slice; NULL without the kernels.
    keys: DevicePtr,
    /// 2026-10-10: The next decode step's row mask when it ends in the device argmax.
    armed: Mutex<Option<RowMask>>,
    /// 2026-10-10: The decode step in flight ends in the device argmax.
    current: AtomicBool,
    /// 2026-10-21: The screened head of the K=2 verify (`screen_head.rs`), installed once by the
    /// factory.
    pub(crate) screen: std::sync::OnceLock<super::screen_head::ScreenHead>,
}

impl DeviceArgmax {
    /// 2026-10-10: Resolve the kernels; allocate the keys only when both are loaded.
    pub(crate) fn new(gpu: &dyn GpuBackend) -> Result<Self> {
        let shard_kernel = try_target_kernel(gpu, "argmax_key", "argmax_key_shard_bf16");
        let resolve_kernel = try_target_kernel(gpu, "argmax_key", "argmax_key_resolve");
        let keys = if shard_kernel.0 != 0 && resolve_kernel.0 != 0 {
            gpu.alloc(KEY_BYTES)?
        } else {
            DevicePtr::NULL
        };
        Ok(Self {
            shard_kernel,
            resolve_kernel,
            keys,
            armed: Mutex::new(None),
            current: AtomicBool::new(false),
            screen: std::sync::OnceLock::new(),
        })
    }

    /// 2026-10-10: Drop any arming not yet taken by a step.
    pub(crate) fn disarm(&self) {
        *self.armed.lock() = None;
    }

    /// 2026-10-10: The mask of the armed step, if any (rank 0, for its command block).
    pub(crate) fn armed_mask(&self) -> Option<RowMask> {
        *self.armed.lock()
    }

    /// 2026-10-10: Start a decode step: take the arming. `true` when this step ends in the
    /// device argmax.
    pub(crate) fn begin_step(&self) -> bool {
        let keys = self.armed.lock().take().is_some();
        self.current.store(keys, Ordering::Relaxed);
        keys
    }

    /// 2026-10-10: Whether the decode step in flight ends in the device argmax.
    pub(crate) fn current(&self) -> bool {
        self.current.load(Ordering::Relaxed)
    }
}

impl TransformerModel {
    /// 2026-10-10: `(begin, len)` of this rank's head slice when a one-row decode step can end
    /// in the device argmax; `None` otherwise. The conditions are those under which `lm_head`
    /// takes its vocab-sharded BF16 branch and leaves the slice's logits untouched afterwards.
    fn device_argmax_shard(&self) -> Option<(usize, usize)> {
        let da = &self.device_argmax;
        let eligible = self.settings.runtime.ep_device_argmax
            && EpWire::get().one_block()
            && self.multi_rank_protocol_active()
            && da.keys != DevicePtr::NULL
            && self.feed_cells != DevicePtr::NULL
            && self.lm_head_q6k.is_none()
            && self.lm_head_fp8.is_none()
            && self.lm_head_nvfp4.is_none()
            && !self.use_fp32_logits
            && self.overlays.is_none()
            && self.logit_softcap_kernel.0 == 0
            && self.logit_softcap_fp32_kernel.0 == 0
            && self.circuit.read().is_none()
            && !self.profile
            && self.layers_capture_with_comm();
        if !eligible {
            return None;
        }
        self.lmhead_vocab_shard(self.config.vocab_size as u32, HeadRanks::Every)
    }

    /// 2026-10-10: Rank 0, before a decode step whose readback is the device argmax: arm the
    /// step under `masks` (one row) and upload the mask. `false` (nothing armed) when the step
    /// cannot end in the device argmax; the caller then reads the logits as before. Any earlier
    /// arming is dropped first, so `masks = &[]` disarms.
    pub(crate) fn arm_device_argmax_dispatch(&self, masks: &[RowMask]) -> Result<bool> {
        self.device_argmax.disarm();
        let [mask] = masks else {
            return Ok(false);
        };
        if self.device_argmax_shard().is_none() {
            return Ok(false);
        }
        let bytes: Vec<u8> = mask.iter().flat_map(|w| w.to_le_bytes()).collect();
        self.gpu
            .copy_h2d_async(&bytes, self.feed_masks, self.gpu.default_stream())?;
        *self.device_argmax.armed.lock() = Some(*mask);
        Ok(true)
    }

    /// 2026-10-10: A worker, on a decode command whose block sets
    /// [`ep_wire::FLAG_DEVICE_ARGMAX`]: copy the block's mask (still in the command buffer)
    /// into the feed mask and arm the step. Errors when this rank cannot end the step in the
    /// device argmax: rank 0 can, and the ranks would issue different collectives.
    pub(crate) fn arm_device_argmax_from_block(&self, mask: RowMask) -> Result<()> {
        if self.device_argmax_shard().is_none() {
            bail!(
                "rank 0 armed the device argmax, but this rank cannot end the step in it \
                 (lever, wire, head or kernels differ)"
            );
        }
        self.gpu.copy_d2d_async(
            self.ep_cmd_buf.offset(ep_wire::CMD_BLOCK_MASK),
            self.feed_masks,
            8,
            self.gpu.default_stream(),
        )?;
        *self.device_argmax.armed.lock() = Some(mask);
        Ok(())
    }

    /// 2026-10-10: The head of a decode step that ends in the device argmax: this rank's slice
    /// of the BF16 head into its range of the logits row (as `lm_head`'s sharded branch, without
    /// the zeroing and the all-reduce), its keys, the 16-byte exchange, and the resolve into
    /// feed cell 0, on every rank. `graph` as for the layers' exchanges: under capture the
    /// exchange runs on the communicator's own stream.
    pub(crate) fn lm_head_device_argmax(
        &self,
        hidden: DevicePtr,
        graph: bool,
        stream: u64,
    ) -> Result<()> {
        let (begin, len) = self
            .device_argmax_shard()
            .context("device argmax step on a head that cannot take it")?;
        let comm = self
            .comm_ref()
            .context("device argmax step without a communicator")?;
        let h = self.config.hidden_size as u32;
        let logits = self.buffers.logits();
        let da = &self.device_argmax;
        self.screen_clear_lazy();
        ops::dense_gemv(
            self.gpu.as_ref(),
            self.dense_gemv_kernel,
            hidden,
            &metrale_model_layers::weight_map::DenseWeight {
                weight: self.lm_head_weight.weight.offset(begin * h as usize * 2),
            },
            logits.offset(begin * 2),
            len as u32,
            h,
            stream,
        )?;
        self.gpu.memset_async(da.keys, 0, KEY_BYTES, stream)?;
        ops::argmax_key_shard_bf16(
            self.gpu.as_ref(),
            da.shard_kernel,
            logits.offset(begin * 2),
            self.feed_masks,
            da.keys,
            begin as u32,
            len as u32,
            stream,
        )?;
        let order = (!graph).then_some(stream);
        let peers = comm
            .exchange_partials(da.keys.0, KEY_BYTES, order)?
            .context(
                "the device argmax needs the one-shot exchange (--all-reduce-oneshot-max-kb)",
            )?;
        ops::argmax_key_resolve(
            self.gpu.as_ref(),
            da.resolve_kernel,
            da.keys,
            DevicePtr(peers.recv),
            (peers.slot_elems * 2) as u64,
            (peers.world_size - 1) as u32,
            self.feed_masks,
            self.feed_cells,
            stream,
        )
    }

    /// 2026-10-10: Copy the `rows` feed cells a device-argmax step resolved to `dst`: with an
    /// event, an async copy then the event (the router's readback ring); without, a blocking
    /// copy.
    pub(crate) fn device_argmax_to_host_dispatch(
        &self,
        dst: &mut [u8],
        event: Option<u64>,
    ) -> Result<()> {
        let stream = self.gpu.default_stream();
        match event {
            Some(e) => {
                self.gpu.copy_d2h_async(self.feed_cells, dst, stream)?;
                self.gpu.record_event(e, stream)
            }
            None => self.gpu.copy_d2h(self.feed_cells, dst),
        }
    }
}
