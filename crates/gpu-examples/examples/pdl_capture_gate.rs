// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-21: GPU gate for programmatic dependent launch through the backend
//! (`GpuBackend::enable_programmatic_launches`, gpu-runtime `registry_pdl`): on two backends of
//! the probe module (`common/pdl_probe.cu`), one with the mode on and one without, it captures
//! producer -> consumer -> plain on one stream and checks
//!
//! 1. the captured graph's edges: with the mode on, producer -> consumer is a programmatic edge
//!    (`CU_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC`) and consumer -> plain (an entry without
//!    `griddepcontrol.wait`) stays a full one; with the mode off every edge is full;
//! 2. the replayed graph: with the mode on the consumer starts (globaltimer, before its wait)
//!    before the producer has finished, on every replay, and still copies the producer's data
//!    (the wait held its read back); with the mode off it never starts before the producer ends;
//! 3. the same two facts on eager launches.
//!
//! Owner: gpu-examples.
//! Invariants: exits with an error on the first failed check.
//!
//!   cargo run -p metrale-gpu-examples --release --example pdl_capture_gate \
//!       --features cuda,gpu-examples

use anyhow::{Result, bail, ensure};
use cudarc::driver::sys;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

#[path = "common/pdl_probe_ptx.rs"]
mod probe_ptx;
const THREADS: u32 = 128;
const SPIN_NS: u32 = 40_000;
const REPLAYS: usize = 50;

struct Probe<'a> {
    gpu: &'a MetraleCudaBackend,
    producer: KernelHandle,
    consumer: KernelHandle,
    plain: KernelHandle,
    data: DevicePtr,
    out: DevicePtr,
    out2: DevicePtr,
    t_end: DevicePtr,
    t_start: DevicePtr,
    t_plain: DevicePtr,
}

impl<'a> Probe<'a> {
    fn new(gpu: &'a MetraleCudaBackend) -> Result<Self> {
        let n = THREADS as usize * 4;
        Ok(Self {
            gpu,
            producer: gpu.kernel("pdl_probe", "pdl_probe_producer")?,
            consumer: gpu.kernel("pdl_probe", "pdl_probe_consumer")?,
            plain: gpu.kernel("pdl_probe", "pdl_probe_plain")?,
            data: gpu.alloc(n)?,
            out: gpu.alloc(n)?,
            out2: gpu.alloc(n)?,
            t_end: gpu.alloc(8)?,
            t_start: gpu.alloc(8)?,
            t_plain: gpu.alloc(8)?,
        })
    }

    /// 2026-10-21: producer (writing `value`), consumer, plain, on `s`.
    fn enqueue(&self, s: u64, value: u32) -> Result<()> {
        let g: &dyn GpuBackend = self.gpu;
        KernelLaunch::new(g, self.producer)
            .block([THREADS, 1, 1])
            .arg_ptr(self.data)
            .arg_ptr(self.t_end)
            .arg_u32(value)
            .arg_u32(SPIN_NS)
            .launch(s)?;
        KernelLaunch::new(g, self.consumer)
            .block([THREADS, 1, 1])
            .arg_ptr(self.data)
            .arg_ptr(self.out)
            .arg_ptr(self.t_start)
            .launch(s)?;
        KernelLaunch::new(g, self.plain)
            .block([THREADS, 1, 1])
            .arg_ptr(self.out)
            .arg_ptr(self.out2)
            .arg_ptr(self.t_plain)
            .launch(s)
    }

    /// 2026-10-21: (consumer started before producer ended, consumer and plain copied `value`).
    fn observe(&self, value: u32) -> Result<(bool, bool)> {
        let g: &dyn GpuBackend = self.gpu;
        let u64_at = |p: DevicePtr| -> Result<u64> {
            let mut b = [0u8; 8];
            g.copy_d2h(p, &mut b)?;
            Ok(u64::from_le_bytes(b))
        };
        let words = |p: DevicePtr| -> Result<Vec<u32>> {
            let mut b = vec![0u8; THREADS as usize * 4];
            g.copy_d2h(p, &mut b)?;
            Ok(b.chunks(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        };
        let want: Vec<u32> = (0..THREADS).map(|i| value + i).collect();
        let early = u64_at(self.t_start)? < u64_at(self.t_end)?;
        Ok((early, words(self.out)? == want && words(self.out2)? == want))
    }
}

/// 2026-10-21: The (type, from_port) of every edge of `graph`, in the driver's order.
fn edges(graph: sys::CUgraph) -> Result<Vec<(u8, u8)>> {
    let mut n: usize = 0;
    let rc = unsafe {
        sys::cuGraphGetEdges_v2(
            graph,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            &mut n,
        )
    };
    ensure!(
        rc == sys::CUresult::CUDA_SUCCESS,
        "cuGraphGetEdges_v2 count: {rc:?}"
    );
    let mut from = vec![std::ptr::null_mut(); n];
    let mut to = vec![std::ptr::null_mut(); n];
    // 2026-10-21: SAFETY: CUgraphEdgeData is plain bytes; zero is its default value.
    let mut data: Vec<sys::CUgraphEdgeData> = vec![unsafe { std::mem::zeroed() }; n];
    let rc = unsafe {
        sys::cuGraphGetEdges_v2(
            graph,
            from.as_mut_ptr(),
            to.as_mut_ptr(),
            data.as_mut_ptr(),
            &mut n,
        )
    };
    ensure!(
        rc == sys::CUresult::CUDA_SUCCESS,
        "cuGraphGetEdges_v2: {rc:?}"
    );
    Ok(data.iter().map(|e| (e.type_, e.from_port)).collect())
}

/// 2026-10-21: Capture `p`'s three launches, check the edges, then replay and observe.
fn graph_arm(p: &Probe, on: bool) -> Result<()> {
    let s = p.gpu.default_stream();
    let cs = s as sys::CUstream;
    let rc = unsafe {
        sys::cuStreamBeginCapture_v2(cs, sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
    };
    ensure!(rc == sys::CUresult::CUDA_SUCCESS, "begin capture: {rc:?}");
    p.enqueue(s, 1000)?;
    let mut graph: sys::CUgraph = std::ptr::null_mut();
    let rc = unsafe { sys::cuStreamEndCapture(cs, &mut graph) };
    ensure!(rc == sys::CUresult::CUDA_SUCCESS, "end capture: {rc:?}");
    let e = edges(graph)?;
    println!(
        "  mode {}: captured edges (type, from_port) = {e:?}",
        if on { "on " } else { "off" }
    );
    ensure!(e.len() == 2, "expected 2 edges, got {e:?}");
    let programmatic = sys::CUgraphDependencyType::CU_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC as u8;
    let n_prog = e.iter().filter(|(t, _)| *t == programmatic).count();
    ensure!(
        n_prog == usize::from(on),
        "mode {on}: {n_prog} programmatic edges, want {} (producer -> consumer only)",
        usize::from(on)
    );
    let mut exec: sys::CUgraphExec = std::ptr::null_mut();
    let rc = unsafe { sys::cuGraphInstantiateWithFlags(&mut exec, graph, 0) };
    ensure!(rc == sys::CUresult::CUDA_SUCCESS, "instantiate: {rc:?}");
    let mut early = 0;
    for _ in 0..REPLAYS {
        p.gpu.memset(p.out, 0, THREADS as usize * 4)?;
        p.gpu.memset(p.out2, 0, THREADS as usize * 4)?;
        let rc = unsafe { sys::cuGraphLaunch(exec, cs) };
        ensure!(rc == sys::CUresult::CUDA_SUCCESS, "graph launch: {rc:?}");
        p.gpu.synchronize(s)?;
        let (e, ok) = p.observe(1000)?;
        ensure!(
            ok,
            "mode {on}: consumer or plain read the producer's data early"
        );
        early += usize::from(e);
    }
    unsafe {
        sys::cuGraphExecDestroy(exec);
        sys::cuGraphDestroy(graph);
    }
    println!(
        "  mode {}: graph replays {REPLAYS}, consumer began before producer ended in {early}, data correct in all",
        if on { "on " } else { "off" }
    );
    ensure!(
        if on { early == REPLAYS } else { early == 0 },
        "mode {on}: overlap {early}/{REPLAYS}"
    );
    Ok(())
}

/// 2026-10-21: The same on eager launches.
fn eager_arm(p: &Probe, on: bool) -> Result<()> {
    let s = p.gpu.default_stream();
    let mut early = 0;
    for r in 0..REPLAYS as u32 {
        p.enqueue(s, 7 * r)?;
        p.gpu.synchronize(s)?;
        let (e, ok) = p.observe(7 * r)?;
        ensure!(ok, "eager mode {on}: wrong data at launch {r}");
        early += usize::from(e);
    }
    println!(
        "  mode {}: eager launches {REPLAYS}, overlapped {early}",
        if on { "on " } else { "off" }
    );
    ensure!(
        if on { early == REPLAYS } else { early == 0 },
        "eager mode {on}: overlap {early}/{REPLAYS}"
    );
    Ok(())
}

fn main() -> Result<()> {
    let modules = [("pdl_probe", probe_ptx::PTX.as_bytes())];
    println!("GATE: programmatic dependent launch through the CUDA backend, eager and captured");
    let off = MetraleCudaBackend::new(0, &modules)?;
    let on = MetraleCudaBackend::new(0, &modules)?;
    if !on.enable_programmatic_launches()? {
        bail!("this backend reports programmatic launches unavailable");
    }
    let (po, pn) = (Probe::new(&off)?, Probe::new(&on)?);
    graph_arm(&po, false)?;
    graph_arm(&pn, true)?;
    eager_arm(&po, false)?;
    eager_arm(&pn, true)?;
    println!("PASS");
    Ok(())
}
