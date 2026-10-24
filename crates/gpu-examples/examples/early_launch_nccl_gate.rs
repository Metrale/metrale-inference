// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: GPU gate for the early graph launch on graphs shaped as NCCL (2.28-2.30
//! `enqueue.cc` + `strongstream.cc`) captures a network collective: the proxy-upload host task
//! sits on NCCL's own capture stream, rooted with only an external wait on the comm's
//! `serialEvent` (no edge from the kernels that produce the collective's input), followed by an
//! external record of `serialEvent`; the collective kernel waits for the host task.
//!
//! 1. Gate: a graph with a programmatic (PDL) edge ahead of its "collective": the collective's
//!    device wait beyond its own producer (only the host task can extend it) is, in every
//!    iteration, shorter launched early than any plain launch's
//!    (the control: there the host task is dispatched only after that work, the gate the early
//!    launch removes). The PDL pair still copies its data. No `METRALE_NCCL_HOST_HOIST`.
//! 2. Order: a graph launched plainly behind 2 ms of compute work, then a second graph launched
//!    early, must run their host tasks in launch order (NCCL posts proxy ops in that order).
//!    Without the hoist it holds; with the hoist on (the CUDA backend rewires host nodes before
//!    instantiation) the second graph's host task loses its `serialEvent` wait, which this arm
//!    reports, so the early launch is refused with the hoist.
//!
//! Owner: gpu-examples.
//! Invariants: exits with an error unless every early wait of arm 1 is below every plain wait
//! on a host under half busy (it refuses to judge otherwise),
//! and arm 2 holds in every iteration without the hoist.
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
/// 2026-10-21: Timestamp slots (u64 globaltimer ns) in `Rig::t`.
const T_BUSY_END: usize = 0;
const T_COLLECTIVE: usize = 1;
/// 2026-10-21: The end of the graph's own (PDL) producer, the collective's last kernel input.
const T_GRAPH_PRODUCER_END: usize = 2;

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

    /// 2026-10-21: `producer` writing its end time (globaltimer ns) to timestamp `slot`.
    fn producer(&self, s: u64, value: u32, spin_ns: u32, slot: usize) -> Result<()> {
        KernelLaunch::new(self.gpu as &dyn GpuBackend, self.producer)
            .block([THREADS, 1, 1])
            .arg_ptr(self.data)
            .arg_ptr(self.t.offset(slot * 8))
            .arg_u32(value)
            .arg_u32(spin_ns)
            .launch(s)
    }

    /// 2026-10-21: A copy kernel writing its start time to timestamp `slot`.
    fn copy(
        &self,
        k: KernelHandle,
        from: DevicePtr,
        to: DevicePtr,
        s: u64,
        slot: usize,
    ) -> Result<()> {
        KernelLaunch::new(self.gpu as &dyn GpuBackend, k)
            .block([THREADS, 1, 1])
            .arg_ptr(from)
            .arg_ptr(to)
            .arg_ptr(self.t.offset(slot * 8))
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
            self.producer(compute, 1000, 20_000, T_GRAPH_PRODUCER_END)?;
            self.copy(
                self.consumer,
                self.data,
                self.out,
                compute,
                T_GRAPH_PRODUCER_END + 1,
            )?;
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
            T_COLLECTIVE,
        )?;
        g.end_capture(compute)
    }
}

/// 2026-10-21: Arm 1: per iteration after one uncounted warmup launch (the first launch of a
/// graph uploads it and wakes the driver's host-callback thread, once per process), the device
/// wait (globaltimer, us) of the graph's collective beyond its own kernel input: its start minus
/// the end of the graph's producer. Only the host task can hold it longer. Device clocks only: a
/// host timestamp is taken by a thread the OS may deschedule (under 40 busy CPU loops a plain
/// launch read as "host task before the busy end" in 1 of 40), which the first version measured.
/// Also prints the gap from the busy work's end to the collective's start.
fn gate_arm(r: &Rig, early: Option<&EarlyLaunch>) -> Result<Vec<i64>> {
    let g: &dyn GpuBackend = r.gpu;
    let compute = g.default_stream();
    let graph = r.capture(early, compute, 0, true)?;
    let (mut gaps, mut waits) = (Vec::new(), Vec::new());
    for it in 0..=ITERS {
        g.memset(r.out2, 0, THREADS as usize * 4)?;
        g.memset(r.t, 0, 64)?;
        r.producer(compute, 7, BUSY_NS, T_BUSY_END)?;
        EarlyLaunch::launch_with(early, g, graph, compute)?;
        g.synchronize(compute)?;
        let mut t = [0u8; 64];
        g.copy_d2h(r.t, &mut t)?;
        let at = |i: usize| i64::from_le_bytes(t[i * 8..i * 8 + 8].try_into().expect("8 bytes"));
        ensure!(
            at(T_BUSY_END) != 0 && at(T_COLLECTIVE) != 0,
            "iteration {it}: no timestamps"
        );
        let gap = (at(T_COLLECTIVE) - at(T_BUSY_END)) / 1000;
        let wait = (at(T_COLLECTIVE) - at(T_GRAPH_PRODUCER_END)) / 1000;
        if it == 0 {
            println!("    warmup (uncounted): gap {gap} us, wait {wait} us");
        } else {
            gaps.push(gap);
            waits.push(wait);
        }
        let mut b = vec![0u8; THREADS as usize * 4];
        g.copy_d2h(r.out2, &mut b)?;
        let want: Vec<u8> = (0..THREADS)
            .flat_map(|i| (1000 + i).to_le_bytes())
            .collect();
        ensure!(b == want, "the PDL pair or the collective read stale data");
    }
    println!(
        "    {} gaps busy end -> collective start (us): {gaps:?}",
        if early.is_some() { "early" } else { "plain" }
    );
    println!("      collective wait beyond its producer (us): {waits:?}");
    g.destroy_graph(graph)?;
    Ok(waits)
}

/// 2026-10-21: Arm 2: iterations whose host tasks ran out of launch order, of `ITERS`.
fn order_arm(r: &Rig, early: &EarlyLaunch) -> Result<usize> {
    let g: &dyn GpuBackend = r.gpu;
    let compute = g.default_stream();
    let first = r.capture(None, compute, 0, false)?;
    let second = r.capture(Some(early), compute, 1, false)?;
    let mut bad = 0;
    for it in 0..ITERS {
        TAGS[0].0.store(0, Ordering::SeqCst);
        TAGS[1].0.store(0, Ordering::SeqCst);
        r.producer(compute, 7, BUSY_NS, T_BUSY_END)?;
        g.launch_graph(first, compute)?;
        early.launch(g, second, compute)?;
        g.synchronize(compute)?;
        let (a, b) = (
            TAGS[0].0.load(Ordering::SeqCst),
            TAGS[1].0.load(Ordering::SeqCst),
        );
        ensure!(a != 0 && b != 0, "iteration {it}: a host task did not run");
        bad += usize::from(a > b);
    }
    g.destroy_graph(first)?;
    g.destroy_graph(second)?;
    Ok(bad)
}

/// 2026-10-21: (busy, total) jiffies over all CPUs (`/proc/stat` line `cpu`).
fn cpu_ticks() -> Result<(u64, u64)> {
    let stat = std::fs::read_to_string("/proc/stat")?;
    let v: Vec<u64> = stat
        .lines()
        .next()
        .unwrap_or_default()
        .split_whitespace()
        .skip(1)
        .filter_map(|x| x.parse().ok())
        .collect();
    ensure!(v.len() >= 5, "unexpected /proc/stat");
    let total: u64 = v.iter().sum();
    Ok((total - v[3] - v[4], total))
}

fn busy_fraction(a: (u64, u64), b: (u64, u64)) -> f64 {
    (b.0 - a.0) as f64 / (b.1 - a.1).max(1) as f64
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
    let (cpu0, e) = (cpu_ticks()?, gate_arm(&r, Some(&early))?);
    let busy = busy_fraction(cpu0, cpu_ticks()?);
    let p = gate_arm(&r, None)?;
    let (e_max, p_min) = (e.iter().max().copied(), p.iter().min().copied());
    println!(
        "  collective wait beyond its producer: early max {e_max:?} us, plain min {p_min:?} us \
         (host CPU {:.0}% busy during the early arm)",
        busy * 100.0
    );
    // 2026-10-21: The early launch hides the host task's dispatch inside the compute stream's
    // earlier work only when the host-callback thread gets a core. On an oversubscribed host
    // (40 busy loops on 20 cores) it can be starved past that work (a 992 us wait in 1 of 40),
    // as a serve's would be; a timing gate does not judge on a busy host (benchmark rule).
    ensure!(
        busy < 0.5,
        "host CPU {:.0}% busy during the early arm: rerun on an idle host",
        busy * 100.0
    );
    ensure!(
        e_max < p_min,
        "the early launch did not remove the gate: an early wait reaches a plain one"
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
