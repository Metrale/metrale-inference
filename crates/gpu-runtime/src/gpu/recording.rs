// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: `RecordingGpu`, a launch-recording device port for equivalence gates: two code
//! paths driven over the same synthetic inputs must put the same work on the stream. It hands
//! out one distinct handle per `(module, function)`, so a launch is recorded by the kernel it
//! names, not by a shared placeholder handle as [`super::mock::MockGpuBackend`] does, and it
//! records every stream operation (launch, memset, copy, event) in one ordered log.
//!
//! Owner: gpu-runtime.
//! Invariants:
//! - Nothing is executed: a memset or copy is recorded, never applied, so synthetic pointers
//!   need no allocation behind them. Allocation goes to an inner mock, whose pointers are
//!   recorded like any other.
//! - Two lookups of one `(module, function)` return the same handle; two different pairs never
//!   do. A launch of a handle this device never returned is recorded as `?<handle>`.
//! - `deny` makes later lookups of a pair fail, so a path that probes for an absent kernel can be
//!   driven down its other branch.

use super::mock::{MockArg, MockGpuBackend};
use super::*;
use parking_lot::Mutex;
use std::collections::{BTreeSet, HashMap};

/// 2026-10-23: First handle value handed out (distinct from the mock's `0xDEAD`).
const HANDLE_BASE: u64 = 0x5EC0_0000;

/// 2026-10-23: One recorded stream operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Op {
    /// 2026-10-23: A kernel launch, by `module::func`.
    Launch {
        kernel: String,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem: u32,
        stream: u64,
        args: Vec<MockArg>,
    },
    /// 2026-10-23: `memset_async` (or `memset`, stream `None`).
    Memset {
        ptr: DevicePtr,
        value: u8,
        bytes: usize,
        stream: Option<u64>,
    },
    /// 2026-10-23: `copy_d2d_async` (or `copy_d2d`, stream `None`).
    CopyD2d {
        src: DevicePtr,
        dst: DevicePtr,
        bytes: usize,
        stream: Option<u64>,
    },
    /// 2026-10-23: `memcpy_2d_async` / `copy_d2d_2d_async`.
    Copy2d { copy: Memcpy2d, stream: u64 },
    /// 2026-10-23: `record_event(event, stream)`.
    RecordEvent { event: u64, stream: u64 },
    /// 2026-10-23: `stream_wait_event(stream, event)`.
    WaitEvent { stream: u64, event: u64 },
    /// 2026-10-23: A collective or exchange a recording communicator logged
    /// ([`RecordingGpu::record_comm`]): its call with arguments, and the stream it was ordered
    /// on (`None`: host-ordered).
    Comm { call: String, stream: Option<u64> },
}

/// 2026-10-23: The stream and event graph of a log: each operation reduced to the stream it is
/// ordered on and the events it records or waits, in order (a launch's kernel and arguments
/// dropped). Two paths with equal graphs fork, join and order collectives the same way.
pub fn stream_graph(ops: &[Op]) -> Vec<String> {
    ops.iter()
        .map(|o| match o {
            Op::Launch { stream, .. } => format!("launch@{stream}"),
            Op::Memset { stream, .. } | Op::CopyD2d { stream, .. } => format!("copy@{stream:?}"),
            Op::Copy2d { stream, .. } => format!("copy@{stream}"),
            Op::RecordEvent { event, stream } => format!("record e{event}@{stream}"),
            Op::WaitEvent { stream, event } => format!("wait {stream}<-e{event}"),
            Op::Comm { stream, .. } => format!("comm@{stream:?}"),
        })
        .collect()
}

impl Op {
    /// 2026-10-23: The kernel of a launch; `None` for any other operation.
    pub fn kernel(&self) -> Option<&str> {
        match self {
            Op::Launch { kernel, .. } => Some(kernel),
            _ => None,
        }
    }
}

/// 2026-10-23: The recording device port.
pub struct RecordingGpu {
    inner: MockGpuBackend,
    names: Mutex<HashMap<(String, String), KernelHandle>>,
    by_handle: Mutex<HashMap<u64, String>>,
    denied: Mutex<BTreeSet<(String, String)>>,
    ops: Mutex<Vec<Op>>,
    next_event: Mutex<u64>,
    /// 2026-10-23: Side streams are distinct and non-zero (the mock's are the default stream 0),
    /// so a log shows which launches a fork moved off the main stream.
    next_stream: Mutex<u64>,
    sms: u32,
}

impl RecordingGpu {
    /// 2026-10-23: A device reporting `sms` SMs (the count some launch shapes are derived from).
    pub fn new(sms: u32) -> Self {
        Self {
            inner: MockGpuBackend::new(),
            names: Mutex::new(HashMap::new()),
            by_handle: Mutex::new(HashMap::new()),
            denied: Mutex::new(BTreeSet::new()),
            ops: Mutex::new(Vec::new()),
            next_event: Mutex::new(1),
            next_stream: Mutex::new(0x5_0000),
            sms,
        }
    }

    /// 2026-10-23: Every later lookup of `module::func` fails.
    pub fn deny(&self, module: &str, func: &str) {
        self.denied.lock().insert((module.into(), func.into()));
    }

    /// 2026-10-23: The operations recorded so far, in issue order, and empty the log.
    pub fn take(&self) -> Vec<Op> {
        std::mem::take(&mut *self.ops.lock())
    }

    /// 2026-10-23: The `module::func` a handle names; `None` for one this device never returned.
    pub fn name_of(&self, h: KernelHandle) -> Option<String> {
        self.by_handle.lock().get(&h.0).cloned()
    }

    fn push(&self, op: Op) {
        self.ops.lock().push(op);
    }

    /// 2026-10-23: Log a communicator call in order with the device operations (a recording
    /// `CommBackend` built over this device calls it).
    pub fn record_comm(&self, call: impl Into<String>, stream: Option<u64>) {
        self.push(Op::Comm {
            call: call.into(),
            stream,
        });
    }

    fn label(&self, h: KernelHandle) -> String {
        self.name_of(h).unwrap_or_else(|| format!("?{:#x}", h.0))
    }
}

impl GpuBackend for RecordingGpu {
    fn op_cache(&self) -> &crate::op_cache::OpCache {
        self.inner.op_cache()
    }

    fn alloc(&self, bytes: usize) -> Result<DevicePtr> {
        self.inner.alloc(bytes)
    }

    fn alloc_managed(&self, bytes: usize) -> Result<DevicePtr> {
        self.inner.alloc(bytes)
    }

    fn free(&self, ptr: DevicePtr) -> Result<()> {
        self.inner.free(ptr)
    }

    fn copy_h2d(&self, src: &[u8], dst: DevicePtr) -> Result<()> {
        self.inner.copy_h2d(src, dst)
    }

    fn copy_d2h(&self, src: DevicePtr, dst: &mut [u8]) -> Result<()> {
        self.inner.copy_d2h(src, dst)
    }

    fn copy_d2d(&self, src: DevicePtr, dst: DevicePtr, bytes: usize) -> Result<()> {
        self.push(Op::CopyD2d {
            src,
            dst,
            bytes,
            stream: None,
        });
        Ok(())
    }

    fn copy_d2d_async(
        &self,
        src: DevicePtr,
        dst: DevicePtr,
        bytes: usize,
        stream: u64,
    ) -> Result<()> {
        self.push(Op::CopyD2d {
            src,
            dst,
            bytes,
            stream: Some(stream),
        });
        Ok(())
    }

    fn memcpy_2d_async(&self, copy: Memcpy2d, stream: u64) -> metrale_error::Result<()> {
        self.push(Op::Copy2d { copy, stream });
        Ok(())
    }

    fn launch(
        &self,
        func: KernelHandle,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem: u32,
        stream: u64,
        _params: &mut [*mut std::ffi::c_void],
    ) -> Result<()> {
        self.push(Op::Launch {
            kernel: self.label(func),
            grid,
            block,
            shared_mem,
            stream,
            args: Vec::new(),
        });
        Ok(())
    }

    fn launch_typed(
        &self,
        func: KernelHandle,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem: u32,
        stream: u64,
        args: &[KernelArg<'_>],
    ) -> Result<()> {
        let args = args
            .iter()
            .map(|a| match a {
                KernelArg::Buffer(p) => MockArg::Buffer(*p),
                KernelArg::Bytes(b) => MockArg::Bytes(b.to_vec()),
            })
            .collect();
        self.push(Op::Launch {
            kernel: self.label(func),
            grid,
            block,
            shared_mem,
            stream,
            args,
        });
        Ok(())
    }

    fn synchronize(&self, stream: u64) -> Result<()> {
        self.inner.synchronize(stream)
    }

    fn default_stream(&self) -> u64 {
        0
    }

    fn has_module(&self, _module: &str) -> bool {
        true
    }

    fn kernel(&self, module: &str, func_name: &str) -> Result<KernelHandle> {
        let key = (module.to_string(), func_name.to_string());
        anyhow::ensure!(
            !self.denied.lock().contains(&key),
            "Kernel lookup {module}::{func_name}: missing"
        );
        let mut names = self.names.lock();
        if let Some(h) = names.get(&key) {
            return Ok(*h);
        }
        let h = KernelHandle(HANDLE_BASE + names.len() as u64);
        names.insert(key, h);
        self.by_handle
            .lock()
            .insert(h.0, format!("{module}::{func_name}"));
        Ok(h)
    }

    fn memset(&self, ptr: DevicePtr, value: u8, bytes: usize) -> Result<()> {
        self.push(Op::Memset {
            ptr,
            value,
            bytes,
            stream: None,
        });
        Ok(())
    }

    fn memset_async(&self, ptr: DevicePtr, value: u8, bytes: usize, stream: u64) -> Result<()> {
        self.push(Op::Memset {
            ptr,
            value,
            bytes,
            stream: Some(stream),
        });
        Ok(())
    }

    fn create_stream(&self) -> Result<u64> {
        let mut s = self.next_stream.lock();
        *s += 1;
        Ok(*s)
    }

    fn create_event(&self) -> Result<u64> {
        let mut e = self.next_event.lock();
        *e += 1;
        Ok(*e)
    }

    fn record_event(&self, event: u64, stream: u64) -> Result<()> {
        self.push(Op::RecordEvent { event, stream });
        Ok(())
    }

    fn stream_wait_event(&self, stream: u64, event: u64) -> Result<()> {
        self.push(Op::WaitEvent { stream, event });
        Ok(())
    }

    fn total_memory(&self) -> Result<usize> {
        self.inner.total_memory()
    }

    fn free_memory(&self) -> Result<usize> {
        self.inner.free_memory()
    }

    fn sm_count(&self) -> Result<u32> {
        Ok(self.sms)
    }
}

#[cfg(test)]
#[path = "recording_tests.rs"]
mod tests;
