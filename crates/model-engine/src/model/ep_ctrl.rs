// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: `METRALE_GLM_EP_CTRL_COMM=1`: the control channel of the multi-rank protocol.
//!
//! Without it every command block and K=2 verdict from rank 0 is an NCCL broadcast on the main
//! communicator, on the compute stream, so on every rank it completes only after all the GPU
//! work queued before it (the drafter and its collectives included): a worker cannot learn its
//! next command until its own propose and rank 0's are done. With it those messages travel on a
//! second communicator on its own greatest-priority stream ([`CtrlChannel`]); payload
//! broadcasts and data collectives stay on the main communicator. Each communicator keeps its
//! own order, which is the same on every rank: rank 0 sends a command's block on the control
//! channel, then its later words on the main one, and a worker reads them in that order.
//!
//! Wire: a frame `[step, kind, payload]` (u32 little-endian), the payload a v3 command block
//! ([`Kind::Block`]) or the K=2 verdict words ([`Kind::Verdict`]). `step` counts control
//! messages from 0 on every rank, so a skipped, repeated or reordered message is an error on the
//! worker that reads it, never a command run in the wrong place.
//!
//! Owner: model-engine (EP worker protocol).
//! Invariants:
//! - A control op never runs while either stream captures a graph, and never waits for the
//!   compute stream unless the payload is produced there (rank 0's fed decode: [`DeviceArg`]).
//! - A worker reads a frame with a copy ordered on the control stream only
//!   ([`CtrlDevice::read`]), never with `copy_d2h`, which waits for the compute stream.
//! - Any failure of a control op, or an unhealthy communicator seen by one, poisons both
//!   communicators: every later operation on either fails, and every rank stops.
//! - A worker reads every device argument of a block from its host copy
//!   ([`needs_device_mirror`]): the compute stream never reads the control buffer, so the next
//!   receive cannot overwrite what a queued kernel has yet to read.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

use anyhow::{Context, Result, bail, ensure};
use metrale_comm::CommBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::ep_wire::{self, CmdBlock, EpWire};
use metrale_model_layers::speculative::K2_VERDICT_WORDS;

/// 2026-10-21: The lever's env name, for messages and the rank agreement.
pub(crate) const ENV: &str = "METRALE_GLM_EP_CTRL_COMM";

/// 2026-10-21: Bytes of the frame header `[step, kind]`.
pub(crate) const HEADER_BYTES: usize = 8;

/// 2026-10-21: Bytes of the largest frame, and so of the control buffer.
pub(crate) const FRAME_MAX: usize = HEADER_BYTES + ep_wire::CMD_BLOCK_BYTES;

/// 2026-10-21: Page-locked staging slots of rank 0's sends. A slot is rewritten only after the
/// copy out of it completed ([`CtrlChannel::send`] waits for its event), so with this many a
/// send waits only when this many are still queued behind receives the workers have not posted.
const RING: usize = 8;

/// 2026-10-21: What a control frame carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub(crate) enum Kind {
    /// 2026-10-21: A v3 command block (`ep_wire`), the first message of every command.
    Block = 1,
    /// 2026-10-21: The K=2 verdict words (`speculative::K2Verdict::to_words`).
    Verdict = 2,
}

impl Kind {
    /// 2026-10-21: Payload bytes of a frame of this kind.
    pub(crate) fn payload_bytes(self) -> usize {
        match self {
            Self::Block => ep_wire::CMD_BLOCK_BYTES,
            Self::Verdict => K2_VERDICT_WORDS * 4,
        }
    }

    fn frame_bytes(self) -> usize {
        HEADER_BYTES + self.payload_bytes()
    }
}

/// 2026-10-21: The channel a control message takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Route {
    /// 2026-10-21: The main communicator on the compute stream (the lever off: today's wire).
    Main,
    /// 2026-10-21: The control communicator on its own stream.
    Ctrl,
}

/// 2026-10-21: The route for the lever and whether a control channel is attached. The lever on
/// without a channel, or a channel with the lever off, is an error: falling back would put this
/// rank on another wire than its peers.
pub(crate) fn route(lever: bool, attached: bool) -> Result<Route> {
    match (lever, attached) {
        (false, false) => Ok(Route::Main),
        (true, true) => Ok(Route::Ctrl),
        (true, false) => bail!(
            "{ENV}=1 but no control communicator is attached; the serve must bootstrap it \
             before the first command"
        ),
        (false, true) => bail!("a control communicator is attached with {ENV} off"),
    }
}

/// 2026-10-21: Refuse the lever off the v3 wire: the channel carries whole command blocks.
pub(crate) fn check_config(lever: bool, wire: EpWire) -> Result<()> {
    ensure!(
        !lever || wire.one_block(),
        "{ENV}=1 needs METRALE_EP_PROTOCOL=v3 (the control channel carries v3 command blocks)"
    );
    Ok(())
}

/// 2026-10-21: Whether a worker's MTP propose may leave its lone draft on the device. With the
/// control channel the propose's readback is the only host wait between a worker's verdict and
/// its next command, and nothing needs it: a worker verifies its own draft only when rank 0
/// announces [`DEVICE_DRAFT`](metrale_model_layers::speculative::DEVICE_DRAFT), and otherwise
/// rank 0's host token. `finish_mtp_propose` still requires the device-step lever, one draft and
/// no grammar mask, and the proposer runs the same launches and collectives either way
/// (`propose_device`, or `propose` when the draft cannot stay on the device).
pub(crate) fn worker_proposes_on_device(ctrl: bool) -> bool {
    ctrl
}

/// 2026-10-21: Whether a received block has an argument a kernel reads from the command buffer
/// on the compute stream: the fed decode's token (`arg`) or the device argmax's row mask. Such a
/// block is copied from the host into that buffer on the compute stream, so the control buffer is
/// never read there.
pub(crate) fn needs_device_mirror(block: &CmdBlock) -> bool {
    block.cmd == crate::model::trait_impl::decode_fed_ep::EP_CMD_DECODE_FED
        || block.flags & ep_wire::FLAG_DEVICE_ARGMAX != 0
}

/// 2026-10-21: The frame `[step, kind, payload]`, padded to [`FRAME_MAX`]; returns it and its
/// length.
pub(crate) fn encode(step: u32, kind: Kind, payload: &[u8]) -> Result<([u8; FRAME_MAX], usize)> {
    ensure!(
        payload.len() == kind.payload_bytes(),
        "control {kind:?} payload of {} bytes, expected {}",
        payload.len(),
        kind.payload_bytes()
    );
    let mut f = [0u8; FRAME_MAX];
    f[..4].copy_from_slice(&step.to_le_bytes());
    f[4..8].copy_from_slice(&(kind as u32).to_le_bytes());
    f[HEADER_BYTES..HEADER_BYTES + payload.len()].copy_from_slice(payload);
    Ok((f, HEADER_BYTES + payload.len()))
}

/// 2026-10-21: The payload of `frame` when it is message `step` and of `kind`.
pub(crate) fn decode(frame: &[u8], step: u32, kind: Kind) -> Result<&[u8]> {
    ensure!(
        frame.len() == kind.frame_bytes(),
        "control frame of {} bytes, expected {} for {kind:?}",
        frame.len(),
        kind.frame_bytes()
    );
    let word = |i: usize| u32::from_le_bytes([frame[i], frame[i + 1], frame[i + 2], frame[i + 3]]);
    let (got_step, got_kind) = (word(0), word(4));
    ensure!(
        got_step == step,
        "control message {got_step} where {step} was due: a message was skipped, repeated or \
         reordered between rank 0 and this rank"
    );
    ensure!(
        got_kind == kind as u32,
        "control message {step} is of kind {got_kind}, expected {kind:?} ({})",
        kind as u32
    );
    Ok(&frame[HEADER_BYTES..])
}

/// 2026-10-21: A send's payload bytes produced on the device: `bytes` at `src`, written by work
/// queued on `producer`, copied over payload offset `at`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DeviceArg {
    pub src: DevicePtr,
    pub at: usize,
    pub bytes: usize,
    pub producer: u64,
}

/// 2026-10-21: The device operations of the control channel (SBIO: the channel's ordering is
/// tested against a recording double; production is any [`GpuBackend`]).
pub(crate) trait CtrlDevice {
    /// 2026-10-21: Host to device on `stream` from a source kept unchanged until the copy is done.
    fn stage(&self, src: &[u8], dst: DevicePtr, stream: u64) -> Result<()>;
    fn device_copy(&self, src: DevicePtr, dst: DevicePtr, bytes: usize, stream: u64) -> Result<()>;
    fn record(&self, event: u64, stream: u64) -> Result<()>;
    /// 2026-10-21: `stream` waits for `event` on the device.
    fn wait(&self, stream: u64, event: u64) -> Result<()>;
    /// 2026-10-21: The host waits for `event`.
    fn host_wait(&self, event: u64) -> Result<()>;
    /// 2026-10-21: Device to host ordered after the work on `stream` only.
    fn read(&self, src: DevicePtr, dst: &mut [u8], stream: u64) -> Result<()>;
    fn capturing(&self, stream: u64) -> bool;
    /// 2026-10-21: The compute stream (the backend's default stream).
    fn compute_stream(&self) -> u64;
}

impl<G: GpuBackend + ?Sized> CtrlDevice for G {
    fn stage(&self, src: &[u8], dst: DevicePtr, stream: u64) -> Result<()> {
        self.copy_h2d_async_retained(src, dst, stream)
    }
    fn device_copy(&self, src: DevicePtr, dst: DevicePtr, bytes: usize, stream: u64) -> Result<()> {
        self.copy_d2d_async(src, dst, bytes, stream)
    }
    fn record(&self, event: u64, stream: u64) -> Result<()> {
        self.record_event(event, stream)
    }
    fn wait(&self, stream: u64, event: u64) -> Result<()> {
        self.stream_wait_event(stream, event)
    }
    fn host_wait(&self, event: u64) -> Result<()> {
        self.event_synchronize(event)
    }
    fn read(&self, src: DevicePtr, dst: &mut [u8], stream: u64) -> Result<()> {
        self.copy_d2h_on_stream(src, dst, stream)
    }
    fn capturing(&self, stream: u64) -> bool {
        self.stream_is_capturing(stream)
    }
    fn compute_stream(&self) -> u64 {
        self.default_stream()
    }
}

/// 2026-10-21: The control communicator, its stream and buffers, and the message counter.
pub(crate) struct CtrlChannel {
    comm: Arc<dyn CommBackend>,
    main: Arc<dyn CommBackend>,
    stream: u64,
    /// 2026-10-21: The device frame every control broadcast goes through ([`FRAME_MAX`] bytes).
    buf: DevicePtr,
    /// 2026-10-21: Rank 0's page-locked staging, [`RING`] slots of [`FRAME_MAX`] bytes (address).
    pinned: usize,
    slot_events: [u64; RING],
    next_slot: AtomicUsize,
    /// 2026-10-21: Recorded on a [`DeviceArg`]'s producer stream before the control stream waits.
    produced: u64,
    /// 2026-10-21: The next message's step, the same sequence on every rank.
    step: AtomicU32,
}

impl CtrlChannel {
    /// 2026-10-21: A channel over `comm` (its broadcasts run on `stream`) beside `main`.
    ///
    /// # Errors
    /// The two communicators name another rank or world, or an allocation fails.
    pub(crate) fn new(
        gpu: &dyn GpuBackend,
        comm: Arc<dyn CommBackend>,
        main: Arc<dyn CommBackend>,
        stream: u64,
    ) -> Result<Self> {
        ensure!(
            comm.rank() == main.rank() && comm.world_size() == main.world_size(),
            "control communicator rank {}/{} differs from the main one's {}/{}",
            comm.rank(),
            comm.world_size(),
            main.rank(),
            main.world_size()
        );
        let buf = gpu.alloc(FRAME_MAX)?;
        let pinned = gpu.alloc_host_pinned(RING * FRAME_MAX)? as usize;
        let mut slot_events = [0u64; RING];
        for e in &mut slot_events {
            *e = gpu.create_event()?;
        }
        Ok(Self {
            comm,
            main,
            stream,
            buf,
            pinned,
            slot_events,
            next_slot: AtomicUsize::new(0),
            produced: gpu.create_event()?,
            step: AtomicU32::new(0),
        })
    }

    /// 2026-10-21: The control stream.
    pub(crate) fn stream(&self) -> u64 {
        self.stream
    }

    /// 2026-10-21: Both communicators usable.
    pub(crate) fn is_healthy(&self) -> bool {
        self.main.is_healthy() && self.comm.is_healthy()
    }

    /// 2026-10-21: Mark both communicators unhealthy.
    pub(crate) fn poison_both(&self) {
        self.main.poison();
        self.comm.poison();
    }

    /// 2026-10-21: Poison both communicators and tag `e`.
    fn fail(&self, e: anyhow::Error) -> anyhow::Error {
        self.poison_both();
        e.context(format!(
            "{ENV}: control channel failed on rank {}; both communicators poisoned; stop all \
             ranks before retrying",
            self.comm.rank()
        ))
    }

    fn ensure_live<D: CtrlDevice + ?Sized>(&self, dev: &D) -> Result<()> {
        ensure!(
            !dev.capturing(self.stream) && !dev.capturing(dev.compute_stream()),
            "a control op inside a graph capture (control ops are never captured)"
        );
        ensure!(self.is_healthy(), "a communicator is unhealthy");
        Ok(())
    }

    /// 2026-10-21: Rank 0: send `payload` as message `kind`, without waiting for it: staged in a
    /// page-locked slot, copied and broadcast on the control stream. With `arg`, its bytes are
    /// copied over the payload on the control stream once the producer's work queued so far is
    /// done, the only wait on the compute stream a send takes.
    pub(crate) fn send<D: CtrlDevice + ?Sized>(
        &self,
        dev: &D,
        kind: Kind,
        payload: &[u8],
        arg: Option<DeviceArg>,
    ) -> Result<()> {
        self.send_inner(dev, kind, payload, arg)
            .map_err(|e| self.fail(e))
    }

    fn send_inner<D: CtrlDevice + ?Sized>(
        &self,
        dev: &D,
        kind: Kind,
        payload: &[u8],
        arg: Option<DeviceArg>,
    ) -> Result<()> {
        ensure!(self.comm.rank() == 0, "control messages are sent by rank 0");
        self.ensure_live(dev)?;
        if let Some(a) = arg {
            ensure!(
                a.at + a.bytes <= payload.len(),
                "device argument {}..{} outside the {}-byte payload",
                a.at,
                a.at + a.bytes,
                payload.len()
            );
        }
        let step = self.step.fetch_add(1, Ordering::Relaxed);
        let (frame, len) = encode(step, kind, payload)?;
        let slot = self.next_slot.fetch_add(1, Ordering::Relaxed) % RING;
        dev.host_wait(self.slot_events[slot])?;
        // 2026-10-21: SAFETY: `pinned..pinned + RING * FRAME_MAX` is the page-locked region
        // `new` allocated and never frees while the channel lives; slot `slot` lies inside it,
        // `len <= FRAME_MAX`, and the copy out of this slot completed (its event, above). Only the
        // scheduler thread sends.
        let staged = unsafe {
            let p = (self.pinned as *mut u8).add(slot * FRAME_MAX);
            std::ptr::copy_nonoverlapping(frame.as_ptr(), p, len);
            std::slice::from_raw_parts(p as *const u8, len)
        };
        dev.stage(staged, self.buf, self.stream)?;
        dev.record(self.slot_events[slot], self.stream)?;
        if let Some(a) = arg {
            dev.record(self.produced, a.producer)?;
            dev.wait(self.stream, self.produced)?;
            dev.device_copy(
                a.src,
                self.buf.offset(HEADER_BYTES + a.at),
                a.bytes,
                self.stream,
            )?;
        }
        self.comm.broadcast_enqueue(self.buf.0, len, 0)
    }

    /// 2026-10-21: A worker: the payload of the next message, which must be of `kind`. A block
    /// is waited for without a deadline (it may arrive after server idle time); a verdict with
    /// the communicator's deadline.
    pub(crate) fn recv<D: CtrlDevice + ?Sized>(&self, dev: &D, kind: Kind) -> Result<Vec<u8>> {
        self.recv_inner(dev, kind).map_err(|e| self.fail(e))
    }

    fn recv_inner<D: CtrlDevice + ?Sized>(&self, dev: &D, kind: Kind) -> Result<Vec<u8>> {
        ensure!(self.comm.rank() != 0, "rank 0 sends control messages");
        self.ensure_live(dev)?;
        let len = kind.frame_bytes();
        match kind {
            Kind::Block => self.comm.recv_command_block(self.buf.0, len, 0),
            Kind::Verdict => self.comm.broadcast(self.buf.0, len, 0),
        }
        .context("control receive")?;
        let mut frame = vec![0u8; len];
        dev.read(self.buf, &mut frame, self.stream)?;
        let step = self.step.fetch_add(1, Ordering::Relaxed);
        decode(&frame, step, kind).map(<[u8]>::to_vec)
    }
}

#[cfg(test)]
#[path = "ep_ctrl_tests.rs"]
mod tests;
