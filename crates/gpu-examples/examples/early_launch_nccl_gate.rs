// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: GPU gate for the early graph launch on graphs shaped as NCCL (2.28-2.30
//! `enqueue.cc` + `strongstream.cc`) captures a network collective: the proxy-upload host task
//! sits on NCCL's own capture stream, rooted with only an external wait on the comm's
//! `serialEvent` (no edge from the kernels that produce the collective's input), followed by an
//! external record of `serialEvent`; the collective kernel waits for the host task.
//!
//! 1. Gate: a graph with a programmatic (PDL) edge ahead of its "collective" runs its host task
//!    while the compute stream's earlier work is still running when launched early, and only
//!    after that work when launched plainly (the control: the gate the early launch removes).
//!    The PDL pair still copies its data. No `METRALE_NCCL_HOST_HOIST` is involved.
//! 2. Order: a graph launched plainly behind 2 ms of compute work, then a second graph launched
//!    early, must run their host tasks in launch order (NCCL posts proxy ops in that order).
//!    Without the hoist it holds; with the hoist on (the CUDA backend rewires host nodes before
//!    instantiation) the second graph's host task loses its `serialEvent` wait, which this arm
//!    reports, so the early launch is refused with the hoist.
//!
//! Owner: gpu-examples.
//! Invariants: exits with an error unless arm 1 holds in every iteration (early before, plain
//! after), and arm 2 holds in every iteration without the hoist.
//!
//!   cargo run -p metrale-gpu-examples --release --example early_launch_nccl_gate \
//!       --features cuda,gpu-examples

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use anyhow::{Result, ensure};
use cudarc::driver::sys;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::early_launch::EarlyLaunch;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_settings::groups::device::DeviceSettings;

#[path = "common/pdl_probe_ptx.rs"]
mod probe_ptx;

const THREADS: u32 = 128;
const ITERS: usize = 40;
/// 2026-10-21: The compute stream's earlier work: one block spinning 2 ms.
const BUSY_NS: u32 = 2_000_000;

static BASE: OnceLock<Instant> = OnceLock::new();
static SEQ: AtomicU64 = AtomicU64::new(0);
/// 2026-10-21: Per graph id: (sequence number + 1, host nanoseconds) of its last host task.
static TAGS: [(AtomicU64, AtomicU64); 2] = [
    (AtomicU64::new(0), AtomicU64::new(0)),
    (AtomicU64::new(0), AtomicU64::new(0)),
];

fn now_ns() -> u64 {
    BASE.get_or_init(Instant::now).elapsed().as_nanos() as u64
}

unsafe extern "C" fn tag(p: *mut std::ffi::c_void) {
    let id = p as usize;
    TAGS[id]
        .0
        .store(SEQ.fetch_add(1, Ordering::SeqCst) + 1, Ordering::SeqCst);
    TAGS[id].1.store(now_ns(), Ordering::SeqCst);
}

fn ok(rc: sys::CUresult, what: &str) -> Result<()> {
    ensure!(rc == sys::CUresult::CUDA_SUCCESS, "{what}: {rc:?}");
    Ok(())
}

struct Rig<'a> {
    gpu: &'a MetraleCudaBackend,
    producer: KernelHandle,
    consumer: KernelHandle,
    plain: KernelHandle,
    data: DevicePtr,
    out: DevicePtr,
    out2: DevicePtr,
    t: DevicePtr,
    /// 2026-10-21: NCCL's host capture stream and its comm's `serialEvent`.
    host: u64,
    serial: u64,
}

impl<'a> Rig<'a> {
    fn new(gpu: &'a MetraleCudaBackend) -> Result<Self> {
        let n = THREADS as usize * 4;
        let r = Self {
            gpu,
            producer: gpu.kernel("pdl_probe", "pdl_probe_producer")?,
            consumer: gpu.kernel("pdl_probe", "pdl_probe_consumer")?,
            plain: gpu.kernel("pdl_probe", "pdl_probe_plain")?,
            data: gpu.alloc(n)?,
            out: gpu.alloc(n)?,
            out2: gpu.alloc(n)?,
            t: gpu.alloc(64)?,
            host: gpu.create_stream()?,
            serial: gpu.create_event()?,
        };
        // 2026-10-21: NCCL's first capture records `serialEvent` on its live stream.
        gpu.record_event(r.serial, r.host)?;
        gpu.synchronize(r.host)?;
        Ok(r)
    }

    fn producer(&self, s: u64, value: u32, spin_ns: u32) -> Result<()> {
        KernelLaunch::new(self.gpu as &dyn GpuBackend, self.producer)
            .block([THREADS, 1, 1])
            .arg_ptr(self.data)
            .arg_ptr(self.t)
            .arg_u32(value)
            .arg_u32(spin_ns)
            .launch(s)
    }

    fn copy(&self, k: KernelHandle, from: DevicePtr, to: DevicePtr, s: u64) -> Result<()> {
        KernelLaunch::new(self.gpu as &dyn GpuBackend, k)
            .block([THREADS, 1, 1])
            .arg_ptr(from)
            .arg_ptr(to)
            .arg_ptr(self.t.offset(8))
            .launch(s)
    }

    /// 2026-10-21: Capture graph `id` on `compute`: [PDL producer -> consumer], then NCCL's host
    /// task (rooted, serialEvent wait, record), then the collective joined behind it.
    fn capture(
        &self,
        early: Option<&EarlyLaunch>,
        compute: u64,
        id: usize,
        pdl: bool,
    ) -> Result<GraphHandle> {
        let g: &dyn GpuBackend = self.gpu;
        EarlyLaunch::begin_capture_with(early, g, compute)?;
        if pdl {
            self.producer(compute, 1000, 20_000)?;
            self.copy(self.consumer, self.data, self.out, compute)?;
        }
        let (fork, join) = (g.create_event()?, g.create_event()?);
        g.record_event(fork, compute)?;
        g.stream_wait_event(self.host, fork)?;
        let h = self.host as sys::CUstream;
        unsafe {
            ok(
                sys::cuStreamUpdateCaptureDependencies_v2(
                    h,
                    std::ptr::null_mut(),
                    std::ptr::null(),
                    0,
                    1,
                ),
                "rooting the host capture stream",
            )?;
        }
        g.stream_wait_event_external(self.host, self.serial)?;
        unsafe {
            ok(
                sys::cuLaunchHostFunc(h, Some(tag), id as *mut _),
                "cuLaunchHostFunc",
            )?;
            ok(
                sys::cuEventRecordWithFlags(self.serial as sys::CUevent, h, 1),
                "external serialEvent record",
            )?;
        }
        g.record_event(join, self.host)?;
        g.stream_wait_event(compute, join)?;
        self.copy(
            self.plain,
            if pdl { self.out } else { self.data },
            self.out2,
            compute,
        )?;
        g.end_capture(compute)
    }
}

/// 2026-10-21: Arm 1: (iterations whose host task ran before the busy work ended) of `ITERS`.
fn gate_arm(r: &Rig, early: Option<&EarlyLaunch>) -> Result<usize> {
    let g: &dyn GpuBackend = r.gpu;
    let compute = g.default_stream();
    let graph = r.capture(early, compute, 0, true)?;
    let busy_done = g.create_event()?;
    let mut before = 0;
    for _ in 0..ITERS {
        g.memset(r.out2, 0, THREADS as usize * 4)?;
        r.producer(compute, 7, BUSY_NS)?;
        g.record_event(busy_done, compute)?;
        EarlyLaunch::launch_with(early, g, graph, compute)?;
        let rc = loop {
            let rc = unsafe { sys::cuEventQuery(busy_done as sys::CUevent) };
            if rc != sys::CUresult::CUDA_ERROR_NOT_READY {
                break rc;
            }
        };
        ok(rc, "cuEventQuery")?;
        let t_done = now_ns();
        g.synchronize(compute)?;
        let t_cb = TAGS[0].1.load(Ordering::SeqCst);
        before += usize::from(t_cb < t_done);
        let mut b = vec![0u8; THREADS as usize * 4];
        g.copy_d2h(r.out2, &mut b)?;
        let want: Vec<u8> = (0..THREADS)
            .flat_map(|i| (1000 + i).to_le_bytes())
            .collect();
        ensure!(b == want, "the PDL pair or the collective read stale data");
    }
    g.destroy_graph(graph)?;
    Ok(before)
}

/// 2026-10-21: Arm 2: iterations whose host tasks ran out of launch order, of `ITERS`.
fn order_arm(r: &Rig, early: &EarlyLaunch) -> Result<usize> {
    let g: &dyn GpuBackend = r.gpu;
    let compute = g.default_stream();
    let first = r.capture(None, compute, 0, false)?;
    let second = r.capture(Some(early), compute, 1, false)?;
    let mut bad = 0;
    for _ in 0..ITERS {
        r.producer(compute, 7, BUSY_NS)?;
        g.launch_graph(first, compute)?;
        early.launch(g, second, compute)?;
        g.synchronize(compute)?;
        let (a, b) = (
            TAGS[0].0.load(Ordering::SeqCst),
            TAGS[1].0.load(Ordering::SeqCst),
        );
        bad += usize::from(a > b);
    }
    g.destroy_graph(first)?;
    g.destroy_graph(second)?;
    Ok(bad)
}

fn backend(hoist: bool) -> Result<MetraleCudaBackend> {
    let modules = [("pdl_probe", probe_ptx::PTX.as_bytes())];
    let mut d = DeviceSettings::declared()?;
    d.nccl_host_hoist = hoist;
    let gpu = MetraleCudaBackend::with_settings(0, &modules, &d)?;
    ensure!(
        gpu.enable_programmatic_launches()?,
        "programmatic launches unavailable"
    );
    Ok(gpu)
}

fn main() -> Result<()> {
    now_ns();
    println!("GATE: early launch on NCCL-shaped graphs with PDL edges");
    let gpu = backend(false)?;
    let r = Rig::new(&gpu)?;
    let early = EarlyLaunch::new(&gpu, gpu.default_stream())?;
    let (e, p) = (gate_arm(&r, Some(&early))?, gate_arm(&r, None)?);
    println!("  host task before the busy work ended: early {e}/{ITERS}, plain {p}/{ITERS}");
    ensure!(
        e == ITERS && p == 0,
        "the early launch did not remove the gate"
    );
    let bad = order_arm(&r, &early)?;
    println!("  host tasks out of launch order, no hoist: {bad}/{ITERS}");
    ensure!(bad == 0, "early launch broke NCCL's host-task order");
    let hoisted = backend(true)?;
    let r = Rig::new(&hoisted)?;
    let early = EarlyLaunch::new(&hoisted, hoisted.default_stream())?;
    let bad = order_arm(&r, &early)?;
    println!(
        "  host tasks out of launch order, METRALE_NCCL_HOST_HOIST on: {bad}/{ITERS} (why the pair is refused)"
    );
    println!("PASS");
    Ok(())
}
