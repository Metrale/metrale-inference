// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: The control channel's wire (frames, step counter, routing, the device mirror) and
//! its ordering, over a two-rank pair of host links (`test_pair_comm`) and a device double that
//! records the stream of every operation (`Rec`). Mocks: the links stand in for NCCL (no GPU on
//! the test host) and `Rec` for the CUDA streams, whose order is what these tests check.
//!
//! Owner: model-engine.
//! Invariants: every expectation is restated here, never read back from the code under test.

use std::sync::{Arc, Mutex};

use metrale_comm::CommBackend;
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::ep_wire::{self, EpWire, NO_MASK};

use super::*;
use crate::test_pair_comm::run_pair_two_links;

/// 2026-10-21: The control stream the tests give the channel; the compute stream is the mock's 0.
const CTRL: u64 = 0x77;
const COMPUTE: u64 = 0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Ev {
    Stage(u64),
    DeviceCopy(u64),
    Record(u64),
    Wait(u64),
    HostWait,
    Read(u64),
}

/// 2026-10-21: The device double: memory through the mock, every call logged with its stream.
struct Rec<'a> {
    gpu: &'a MockGpuBackend,
    log: Mutex<Vec<Ev>>,
    capturing: Option<u64>,
}

impl<'a> Rec<'a> {
    fn new(gpu: &'a MockGpuBackend) -> Self {
        Self {
            gpu,
            log: Mutex::new(Vec::new()),
            capturing: None,
        }
    }
    fn push(&self, e: Ev) {
        self.log.lock().unwrap().push(e);
    }
    fn log(&self) -> Vec<Ev> {
        self.log.lock().unwrap().clone()
    }
}

impl CtrlDevice for Rec<'_> {
    fn stage(&self, src: &[u8], dst: DevicePtr, stream: u64) -> anyhow::Result<()> {
        self.push(Ev::Stage(stream));
        self.gpu.copy_h2d(src, dst)
    }
    fn device_copy(
        &self,
        src: DevicePtr,
        dst: DevicePtr,
        bytes: usize,
        stream: u64,
    ) -> anyhow::Result<()> {
        self.push(Ev::DeviceCopy(stream));
        self.gpu.copy_d2d(src, dst, bytes)
    }
    fn record(&self, _event: u64, stream: u64) -> anyhow::Result<()> {
        self.push(Ev::Record(stream));
        Ok(())
    }
    fn wait(&self, stream: u64, _event: u64) -> anyhow::Result<()> {
        self.push(Ev::Wait(stream));
        Ok(())
    }
    fn host_wait(&self, _event: u64) -> anyhow::Result<()> {
        self.push(Ev::HostWait);
        Ok(())
    }
    fn read(&self, src: DevicePtr, dst: &mut [u8], stream: u64) -> anyhow::Result<()> {
        self.push(Ev::Read(stream));
        self.gpu.copy_d2h(src, dst)
    }
    fn capturing(&self, stream: u64) -> bool {
        self.capturing == Some(stream)
    }
    fn compute_stream(&self) -> u64 {
        COMPUTE
    }
}

fn channel(
    gpu: &Arc<MockGpuBackend>,
    main: Arc<dyn CommBackend>,
    ctrl: Arc<dyn CommBackend>,
) -> CtrlChannel {
    CtrlChannel::new(gpu.as_ref(), ctrl, main, CTRL).unwrap()
}

fn verdict_bytes(w: [u32; 4]) -> Vec<u8> {
    w.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// 2026-10-21: A frame decodes only as the message due: the next step and its kind. A skipped
/// (step ahead), repeated (step behind) or foreign (other kind) message, or a frame of the wrong
/// size, is refused naming why; a payload of the wrong size is never encoded.
#[test]
fn a_frame_decodes_only_as_the_message_due() {
    let block = ep_wire::host_block(3, 0xFFFF_FFF2);
    let (f, len) = encode(7, Kind::Block, &block).unwrap();
    assert_eq!(len, HEADER_BYTES + ep_wire::CMD_BLOCK_BYTES);
    assert_eq!(decode(&f[..len], 7, Kind::Block).unwrap(), &block[..]);
    let e = decode(&f[..len], 8, Kind::Block).unwrap_err().to_string();
    assert!(e.contains("skipped"), "{e}");
    let e = decode(&f[..len], 6, Kind::Block).unwrap_err().to_string();
    assert!(e.contains("skipped, repeated"), "{e}");
    let v = verdict_bytes([1, 2, 3, 4]);
    let (fv, lv) = encode(7, Kind::Verdict, &v).unwrap();
    let e = decode(&fv[..lv], 7, Kind::Block).unwrap_err().to_string();
    assert!(
        e.contains("frame of"),
        "a verdict frame is shorter than a block's: {e}"
    );
    let mut wrong_kind = f;
    wrong_kind[4] = Kind::Verdict as u8;
    let e = decode(&wrong_kind[..len], 7, Kind::Block)
        .unwrap_err()
        .to_string();
    assert!(e.contains("kind 2"), "{e}");
    assert!(
        encode(0, Kind::Verdict, &block).is_err(),
        "a block is no verdict payload"
    );
}

/// 2026-10-21: The control channel is taken exactly when the lever is on and a channel is
/// attached; either alone is an error, never a silent fall back to the main wire.
#[test]
fn the_route_is_the_control_channel_only_with_lever_and_channel() {
    assert_eq!(route(false, false).unwrap(), Route::Main);
    assert_eq!(route(true, true).unwrap(), Route::Ctrl);
    assert!(route(true, false).is_err(), "lever on, nothing attached");
    assert!(route(false, true).is_err(), "attached with the lever off");
    assert!(check_config(true, EpWire::V3).is_ok());
    assert!(
        check_config(true, EpWire::V2).is_err(),
        "a v2 wire has no blocks"
    );
    assert!(
        check_config(false, EpWire::V1).is_ok(),
        "lever off: any wire"
    );
}

/// 2026-10-21: A worker copies a block into the command buffer exactly when a kernel reads it
/// there: the fed decode's token and the device argmax's mask. The fused K=2 block's own flag
/// (`FLAG_K2_DEVICE_VERDICT`) and every host command need no copy.
#[test]
fn only_fed_and_device_argmax_blocks_are_mirrored_to_the_device() {
    use crate::model::trait_impl::decode_fed_ep::EP_CMD_DECODE_FED;
    let parse = |b: [u8; ep_wire::CMD_BLOCK_BYTES]| ep_wire::parse_block(&b);
    assert!(needs_device_mirror(&parse(ep_wire::host_block(
        1,
        EP_CMD_DECODE_FED
    ))));
    assert!(needs_device_mirror(&parse(ep_wire::decode_block(
        1,
        151_329,
        ep_wire::FLAG_DEVICE_ARGMAX,
        [3, NO_MASK]
    ))));
    for block in [
        ep_wire::host_block(1, 151_329),
        ep_wire::host_block(1, ep_wire::EP_CMD_ALLOC_SLOT),
        metrale_model_layers::speculative::k2_announce_block(1, [5, 6], true),
        metrale_model_layers::speculative::k2_announce_block(1, [5, 6], false),
    ] {
        assert!(!needs_device_mirror(&parse(block)), "{:?}", parse(block));
    }
}

/// 2026-10-21: Blocks and verdicts cross on the control link only, in order, with their bytes;
/// the main link carries nothing. Rank 0 stages and broadcasts on the control stream and never
/// waits on the compute stream for a host payload; the worker reads every frame with a copy on
/// the control stream.
#[test]
fn control_messages_cross_on_the_control_link_and_stream_only() {
    let blocks = [
        ep_wire::host_block(2, 0xFFFF_FFF2),
        ep_wire::host_block(4, 151_000),
    ];
    let verdict = verdict_bytes([1, 151_329, 1, 900]);
    let v2 = verdict.clone();
    let [(r0, main0, ctrl0), (r1, main1, ctrl1)] =
        run_pair_two_links(move |rank, gpu, main, ctrl| {
            let ch = channel(gpu, main, ctrl);
            let dev = Rec::new(gpu);
            if rank == 0 {
                ch.send(&dev, Kind::Block, &blocks[0], None).unwrap();
                ch.send(&dev, Kind::Verdict, &v2, None).unwrap();
                ch.send(&dev, Kind::Block, &blocks[1], None).unwrap();
                return (dev.log(), Vec::new());
            }
            let got = vec![
                ch.recv(&dev, Kind::Block).unwrap(),
                ch.recv(&dev, Kind::Verdict).unwrap(),
                ch.recv(&dev, Kind::Block).unwrap(),
            ];
            (dev.log(), got)
        });
    assert_eq!(r1.1, vec![blocks[0].to_vec(), verdict, blocks[1].to_vec()]);
    assert!(
        main0.is_empty() && main1.is_empty(),
        "the main link carried a control message"
    );
    assert_eq!(ctrl0.len(), 3);
    assert_eq!(ctrl0, ctrl1);
    assert!(ctrl0.iter().all(|&(rv, root)| !rv && root == 0));
    let sends = r0.0;
    assert!(
        sends
            .iter()
            .all(|e| matches!(e, Ev::Stage(CTRL) | Ev::Record(CTRL) | Ev::HostWait)),
        "a host-payload send touched another stream or waited on one: {sends:?}"
    );
    assert_eq!(sends.iter().filter(|e| **e == Ev::Stage(CTRL)).count(), 3);
    assert_eq!(
        r1.0,
        vec![Ev::Read(CTRL); 3],
        "worker reads off the control stream"
    );
}

/// 2026-10-21: A payload produced on the device (the fed decode's token) is copied over the
/// payload on the control stream only after that stream waits for the producer's work: record
/// on the producer, wait on the control stream, then the copy. The worker gets the host words
/// with the device bytes over `arg`.
#[test]
fn a_device_argument_waits_for_its_producer_before_it_is_copied() {
    const TOKEN: u32 = 151_329;
    let block = ep_wire::host_block(5, 0xFFFF_FFE1);
    let [(r0, ..), (r1, ..)] = run_pair_two_links(move |rank, gpu, main, ctrl| {
        let ch = channel(gpu, main, ctrl);
        let dev = Rec::new(gpu);
        if rank == 0 {
            let src = gpu.alloc(4).unwrap();
            gpu.copy_h2d(&TOKEN.to_le_bytes(), src).unwrap();
            let arg = DeviceArg {
                src,
                at: ep_wire::CMD_BLOCK_ARG,
                bytes: 4,
                producer: COMPUTE,
            };
            ch.send(&dev, Kind::Block, &block, Some(arg)).unwrap();
            return (dev.log(), Vec::new());
        }
        (dev.log(), ch.recv(&dev, Kind::Block).unwrap())
    });
    let log = r0.0;
    let at = |e: Ev| {
        log.iter()
            .position(|x| *x == e)
            .unwrap_or_else(|| panic!("{e:?} in {log:?}"))
    };
    assert!(at(Ev::Stage(CTRL)) < at(Ev::Record(COMPUTE)));
    assert!(at(Ev::Record(COMPUTE)) < at(Ev::Wait(CTRL)), "{log:?}");
    assert!(at(Ev::Wait(CTRL)) < at(Ev::DeviceCopy(CTRL)), "{log:?}");
    let got = ep_wire::parse_block(&r1.1.as_slice().try_into().unwrap());
    assert_eq!((got.seq_id, got.cmd, got.arg), (5, 0xFFFF_FFE1, TOKEN));
}

/// 2026-10-21: A message out of step on a worker (here: one ahead of the counter, as after a
/// lost message) fails the receive and poisons both of that rank's communicators.
#[test]
fn a_step_skew_fails_the_worker_and_poisons_both_communicators() {
    let block = ep_wire::host_block(1, 2);
    let [_, (r1, ..)] = run_pair_two_links(move |rank, gpu, main, ctrl| {
        let ch = channel(gpu, main.clone(), ctrl.clone());
        let dev = Rec::new(gpu);
        if rank == 0 {
            ch.send(&dev, Kind::Block, &block, None).unwrap();
            return (String::new(), true, true);
        }
        ch.step.store(1, Ordering::Relaxed);
        let e = format!("{:#}", ch.recv(&dev, Kind::Block).unwrap_err());
        (e, main.is_healthy(), ctrl.is_healthy())
    });
    assert!(
        r1.0.contains("control message 0 where 1 was due"),
        "{}",
        r1.0
    );
    assert!(r1.0.contains("both communicators poisoned"), "{}", r1.0);
    assert_eq!((r1.1, r1.2), (false, false));
}

/// 2026-10-21: What rank 0 sees sending one block after `arm` set up its device double or its
/// main communicator: the error text, both communicators' health after it, and the broadcasts
/// that reached the control link.
fn refused_send(arm: fn(&mut Rec<'_>, &dyn CommBackend)) -> (String, bool, bool, usize) {
    let [((e, main_ok, ctrl_ok), _, ctrl_ops), _] =
        run_pair_two_links(move |rank, gpu, main, ctrl| {
            let ch = channel(gpu, main.clone(), ctrl.clone());
            if rank != 0 {
                return (String::new(), true, true);
            }
            let mut dev = Rec::new(gpu);
            arm(&mut dev, main.as_ref());
            let e = ch.send(&dev, Kind::Block, &ep_wire::host_block(1, 2), None);
            let e = format!("{:#}", e.expect_err("the send must be refused"));
            (e, main.is_healthy(), ctrl.is_healthy())
        });
    (e, main_ok, ctrl_ok, ctrl_ops.len())
}

/// 2026-10-21: A control op refuses to run inside a graph capture on either stream, and next to
/// an unhealthy main communicator. Each refusal reaches no link and leaves both communicators
/// poisoned, so every later op on either fails too.
#[test]
fn a_capture_or_an_unhealthy_communicator_refuses_the_op_and_poisons_both() {
    let cases: [(&str, fn(&mut Rec<'_>, &dyn CommBackend), &str); 3] = [
        (
            "capture on compute",
            |d, _| d.capturing = Some(COMPUTE),
            "graph capture",
        ),
        (
            "capture on control",
            |d, _| d.capturing = Some(CTRL),
            "graph capture",
        ),
        ("main unhealthy", |_, m| m.poison(), "unhealthy"),
    ];
    for (name, arm, why) in cases {
        let (e, main_ok, ctrl_ok, ops) = refused_send(arm);
        assert!(e.contains(why), "{name}: {e}");
        assert!(e.contains("both communicators poisoned"), "{name}: {e}");
        assert_eq!((main_ok, ctrl_ok), (false, false), "{name}");
        assert_eq!(ops, 0, "{name}: a refused op reached the link");
    }
}
