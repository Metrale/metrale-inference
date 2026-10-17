// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Opaque device handles and the typed kernel argument every backend
//! accepts.
//!
//! Owner: gpu-runtime.
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-25: A device address: a `CUdeviceptr` on CUDA, an `MTLBuffer`
/// `gpuAddress` on Metal. 0 is null.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DevicePtr(pub u64);

impl DevicePtr {
    pub const NULL: Self = Self(0);

    pub fn is_null(self) -> bool {
        self.0 == 0
    }

    /// 2026-09-25: The address `bytes` past this one.
    pub fn offset(self, bytes: usize) -> Self {
        Self(self.0 + bytes as u64)
    }
}

/// 2026-09-25: A kernel function: a `CUfunction` on CUDA, a pipeline slab index
/// on Metal.
#[derive(Debug, Clone, Copy)]
pub struct KernelHandle(pub u64);

/// 2026-09-25: An instantiated CUDA graph (`CUgraphExec`).
#[derive(Debug, Clone, Copy)]
pub struct GraphHandle(pub u64);

/// 2026-09-25: A typed kernel argument for `launch_typed`. The Metal backend
/// binds buffers with `setBuffer:offset:atIndex:` and bytes with
/// `setBytes:length:atIndex:`; CUDA packs both into parameter slots.
#[derive(Debug, Clone, Copy)]
pub enum KernelArg<'a> {
    /// 2026-09-25: A device address. The Metal backend resolves it to its
    /// owning `MTLBuffer` and offset; CUDA passes the `u64` as is.
    Buffer(DevicePtr),
    /// 2026-09-25: Scalar or struct bytes, passed by value. CUDA packs them into
    /// 8-byte slots, zero-padding the last (`pack_kernel_args`).
    Bytes(&'a [u8]),
}

impl fmt::Display for DevicePtr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DevicePtr(0x{:x})", self.0)
    }
}

/// 2026-10-17: One pitched device-to-device copy: `height` rows of `width_bytes`, source rows
/// `src_pitch` bytes apart from `src` and destination rows `dst_pitch` bytes apart from `dst`.
/// The argument of `GpuBackend::memcpy_2d_async`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Memcpy2d {
    pub src: DevicePtr,
    pub src_pitch: usize,
    pub dst: DevicePtr,
    pub dst_pitch: usize,
    pub width_bytes: usize,
    pub height: usize,
}
