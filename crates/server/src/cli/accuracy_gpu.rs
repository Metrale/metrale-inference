// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The GPU side of `met accuracy`: a `KernelRunner` over this binary's compiled
//! kernel targets. It loads the PTX set of the target a point's instances run, uploads the
//! case's canonical operands, launches through the adapter of the case's launcher
//! (`accuracy_adapters`), and returns the output bytes after checking the guard bands around
//! the output buffer.
//!
//! Owner: server CLI.
//! Invariants:
//! - Every output buffer is filled with `runner::SENTINEL` before the launch, with a guard band
//!   on each side; a write outside the output is an error, a column the kernel never writes
//!   stays a far-out-of-bound value.
//! - A target this binary did not compile is an error naming the targets it did compile.

use std::collections::BTreeSet;

use anyhow::{Context, Result, bail, ensure};
use metrale_accuracy::case::{Case, Tensor};
use metrale_accuracy::runner::{KernelRunner, RunError, SENTINEL};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_kernels::TargetPtxSet;

/// 2026-10-09: Bytes of each guard band around an output buffer.
const GUARD: usize = 256;

/// 2026-10-09: The error of a write outside the output buffer.
pub(crate) const OUT_OF_BOUNDS: &str = "the kernel wrote outside its output buffer";

/// 2026-10-09: The runner.
pub(crate) struct GpuRunner {
    hardware: String,
    sets: Vec<TargetPtxSet>,
    current: Option<(String, MetraleCudaBackend)>,
    used: BTreeSet<String>,
    closures: serde_json::Value,
    /// 2026-10-10: The envelope sweep runs candidates the engine's routers would send elsewhere:
    /// the adapters skip their route checks (never set by `met accuracy`).
    sweep: bool,
}

fn key(hw: &str, s: &TargetPtxSet) -> String {
    format!("{hw}/{}/{}", s.target.model, s.target.quant)
}

impl GpuRunner {
    /// 2026-10-09: A runner over the binary's targets of `hardware`.
    pub(crate) fn new(hardware: &str) -> Result<Self> {
        let closures: serde_json::Value = serde_json::from_str(metrale_kernels::TARGET_CLOSURES)
            .context("the binary's TARGET_CLOSURES")?;
        let sets = metrale_kernels::all_ptx_sets();
        ensure!(!sets.is_empty(), "this binary compiled no kernel target");
        let compiled: Vec<String> = sets.iter().map(|s| key(hardware, s)).collect();
        ensure!(
            compiled.iter().any(|k| closures.get(k).is_some()),
            "this binary's targets {compiled:?} are not `{hardware}` targets (closures: {closures})"
        );
        Ok(GpuRunner {
            hardware: hardware.to_string(),
            sets,
            current: None,
            used: BTreeSet::new(),
            closures,
            sweep: false,
        })
    }

    /// 2026-10-10: A runner for the envelope sweep: candidates launch where their own launcher
    /// can, whatever today's router picks there.
    pub(crate) fn for_sweep(hardware: &str) -> Result<Self> {
        let mut r = Self::new(hardware)?;
        r.sweep = true;
        Ok(r)
    }

    /// 2026-10-10: Load the first compiled target of this hardware.
    pub(crate) fn select_any(&mut self) -> Result<()> {
        let all: BTreeSet<String> = self.sets.iter().map(|s| key(&self.hardware, s)).collect();
        self.select(&all)
    }

    /// 2026-10-10: Time `case`: `warmup` launches, then `reps` repetitions of `iters` launches
    /// back to back with one synchronize each; per repetition the mean microseconds of one
    /// launch sequence. Operands are uploaded once into enough rotating copies to exceed the
    /// L2 (`timing_copies`), so each launch reads its weight from DRAM as a decode step does.
    pub(crate) fn time(
        &mut self,
        case: &Case,
        warmup: usize,
        iters: usize,
        reps: usize,
    ) -> std::result::Result<Vec<f64>, RunError> {
        let (_, gpu) = self
            .current
            .as_ref()
            .ok_or_else(|| RunError::Unavailable("no target selected".into()))?;
        let bytes: usize = case.tensors.values().map(|t| t.bytes.len()).sum();
        let mut dev = Dev {
            gpu,
            stream: gpu.default_stream(),
            allocs: Vec::new(),
            sweep: self.sweep,
            timing: Some(Timing::new(timing_copies(bytes))),
        };
        let fault = |e: anyhow::Error| RunError::Fault(format!("{e:#}"));
        let once = |dev: &mut Dev<'_>| -> std::result::Result<(), RunError> {
            super::accuracy_adapters::launch(dev, case)?;
            if let Some(t) = dev.timing.as_mut() {
                t.iter += 1;
            }
            Ok(())
        };
        let mut out = Vec::with_capacity(reps);
        let r = (|| {
            for _ in 0..warmup {
                once(&mut dev)?;
            }
            gpu.synchronize(dev.stream).map_err(fault)?;
            for _ in 0..reps {
                let t0 = std::time::Instant::now();
                for _ in 0..iters {
                    once(&mut dev)?;
                }
                gpu.synchronize(dev.stream).map_err(fault)?;
                out.push(t0.elapsed().as_secs_f64() * 1e6 / iters.max(1) as f64);
            }
            Ok(())
        })();
        let freed = dev
            .free_all()
            .map_err(|e| RunError::Fault(format!("free: {e:#}")));
        r.and(freed).map(|()| out)
    }

    /// 2026-10-09: Load the first of `targets` (`hw/model/quant`) this binary compiled.
    pub(crate) fn select(&mut self, targets: &BTreeSet<String>) -> Result<()> {
        if let Some((cur, _)) = &self.current
            && targets.contains(cur)
        {
            return Ok(());
        }
        let Some(set) = self
            .sets
            .iter()
            .find(|s| targets.contains(&key(&self.hardware, s)))
        else {
            bail!(
                "none of {targets:?} is compiled in this binary (it has {:?}); build with METRALE_TARGET_MODEL",
                self.sets
                    .iter()
                    .map(|s| key(&self.hardware, s))
                    .collect::<Vec<_>>()
            );
        };
        let k = key(&self.hardware, set);
        self.current = None;
        let backend = MetraleCudaBackend::new(0, &set.modules)?;
        self.used.insert(k.clone());
        self.current = Some((k, backend));
        Ok(())
    }

    /// 2026-10-09: `target=closure` for every target this run loaded.
    pub(crate) fn closures_used(&self) -> Vec<String> {
        self.used
            .iter()
            .map(|k| {
                let h = self
                    .closures
                    .get(k)
                    .and_then(|c| c.get("hash"))
                    .and_then(|h| h.as_str())
                    .unwrap_or("unattested");
                format!("{k}={h}")
            })
            .collect()
    }
}

impl KernelRunner for GpuRunner {
    fn run(&mut self, case: &Case) -> std::result::Result<Vec<u8>, RunError> {
        let (_, gpu) = self
            .current
            .as_ref()
            .ok_or_else(|| RunError::Unavailable("no target selected".into()))?;
        let mut dev = Dev {
            gpu,
            stream: gpu.default_stream(),
            allocs: Vec::new(),
            sweep: self.sweep,
            timing: None,
        };
        let r = super::accuracy_adapters::launch(&mut dev, case);
        let freed = dev
            .free_all()
            .map_err(|e| RunError::Fault(format!("free: {e:#}")));
        match (r, freed) {
            (Ok(out), Ok(())) => Ok(out),
            (Err(e), _) => Err(e),
            (Ok(_), Err(e)) => Err(e),
        }
    }

    fn closure(&self) -> String {
        self.closures_used().join(",")
    }

    fn device(&self) -> String {
        format!("{} GPU 0", self.hardware)
    }
}

/// 2026-10-10: Bytes the rotating operand copies of a timed case should exceed: twice the
/// GB10's 24 MB L2, so no launch finds its weight cached.
const TIMING_COLD_BYTES: usize = 48 << 20;
/// 2026-10-10: Most copies of a small case's operands.
const TIMING_MAX_COPIES: usize = 64;

fn timing_copies(case_bytes: usize) -> usize {
    TIMING_COLD_BYTES
        .div_ceil(case_bytes.max(1))
        .clamp(1, TIMING_MAX_COPIES)
}

/// 2026-10-10: A timed case's buffers: every operand uploaded once into `copies` copies, the
/// launch `iter` reading copy `iter % copies`; one output buffer, never read back.
pub(crate) struct Timing {
    copies: usize,
    pub(crate) iter: usize,
    inputs: std::collections::HashMap<usize, Vec<DevicePtr>>,
    out: Option<DevicePtr>,
}

impl Timing {
    fn new(copies: usize) -> Self {
        Timing {
            copies,
            iter: 0,
            inputs: std::collections::HashMap::new(),
            out: None,
        }
    }
}

/// 2026-10-09: Device buffers of one launch, freed together.
pub(crate) struct Dev<'a> {
    pub(crate) gpu: &'a dyn GpuBackend,
    pub(crate) stream: u64,
    allocs: Vec<DevicePtr>,
    /// 2026-10-10: Skip the adapters' route checks (the envelope sweep).
    pub(crate) sweep: bool,
    /// 2026-10-10: Timing mode: cached operands, no output read-back.
    timing: Option<Timing>,
}

impl Dev<'_> {
    /// 2026-10-09: Upload a tensor's bytes.
    pub(crate) fn upload(&mut self, t: &Tensor) -> Result<DevicePtr> {
        if let Some(tm) = self.timing.as_mut() {
            let key = std::sync::Arc::as_ptr(&t.bytes) as usize;
            if let Some(c) = tm.inputs.get(&key) {
                return Ok(c[tm.iter % c.len()]);
            }
            let mut copies = Vec::with_capacity(tm.copies);
            for _ in 0..tm.copies {
                let p = self.gpu.alloc(t.bytes.len().max(1))?;
                self.allocs.push(p);
                self.gpu.copy_h2d(&t.bytes, p)?;
                copies.push(p);
            }
            let p = copies[tm.iter % copies.len()];
            tm.inputs.insert(key, copies);
            return Ok(p);
        }
        let p = self.gpu.alloc(t.bytes.len().max(1))?;
        self.allocs.push(p);
        self.gpu.copy_h2d(&t.bytes, p)?;
        Ok(p)
    }

    /// 2026-10-09: An output buffer of `bytes`, sentinel-filled, inside two guard bands.
    pub(crate) fn output(&mut self, bytes: usize) -> Result<DevicePtr> {
        if let Some(out) = self.timing.as_ref().and_then(|t| t.out) {
            return Ok(out);
        }
        let base = self.gpu.alloc(bytes + 2 * GUARD)?;
        self.allocs.push(base);
        self.gpu.memset(base, SENTINEL, bytes + 2 * GUARD)?;
        let out = DevicePtr(base.0 + GUARD as u64);
        if let Some(t) = self.timing.as_mut() {
            t.out = Some(out);
        }
        Ok(out)
    }

    /// 2026-10-09: Wait for the launches, read an output back, and check its guard bands.
    pub(crate) fn read(&mut self, out: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
        if self.timing.is_some() {
            return Ok(Vec::new());
        }
        self.gpu.synchronize(self.stream)?;
        let mut all = vec![0u8; bytes + 2 * GUARD];
        self.gpu
            .copy_d2h(DevicePtr(out.0 - GUARD as u64), &mut all)?;
        let (head, rest) = all.split_at(GUARD);
        let (body, tail) = rest.split_at(bytes);
        ensure!(
            head.iter().chain(tail).all(|&b| b == SENTINEL),
            "{OUT_OF_BOUNDS}"
        );
        Ok(body.to_vec())
    }

    fn free_all(&mut self) -> Result<()> {
        let mut first = None;
        for p in self.allocs.drain(..) {
            if let Err(e) = self.gpu.free(p) {
                first.get_or_insert(e);
            }
        }
        first.map_or(Ok(()), Err)
    }
}
