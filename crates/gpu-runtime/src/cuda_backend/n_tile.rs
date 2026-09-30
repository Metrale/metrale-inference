// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-29: Read the N tile width a kernel publishes, at handle resolution.
//!
//! Owner: gpu-runtime (CUDA backend).
//! Invariants: a handle's recorded tile is the value of its module's `<entry>_n_tile`
//! device symbol, read once when the handle is resolved.

use anyhow::Result;

use crate::gpu::KernelHandle;
use crate::op_cache::OpCache;
use crate::registry::MetraleRegistry;

/// 2026-09-29: When `module` defines a 4-byte `<func_name>_n_tile` symbol, copy it to the
/// host and record it for `handle` (`GpuBackend::kernel_n_tile`). Most kernels publish
/// none; an absent symbol records nothing and is not an error. A symbol that exists but
/// cannot be read is.
pub(super) fn record_published(
    registry: &MetraleRegistry,
    op_cache: &OpCache,
    module: &str,
    func_name: &str,
    handle: KernelHandle,
) -> Result<()> {
    let Ok((dptr, 4)) = registry.device_symbol(module, &format!("{func_name}_n_tile")) else {
        return Ok(());
    };
    let mut v = 0u32;
    let stream = registry.raw_stream();
    // 2026-09-29: SAFETY: `v` is 4 bytes and outlives the synchronize below.
    unsafe { registry.copy_d2h_async((&mut v as *mut u32).cast(), dptr, 4, stream) }
        .and_then(|_| registry.stream_synchronize(stream))
        .map_err(|e| anyhow::anyhow!("{module}::{func_name}_n_tile: {e}"))?;
    op_cache.record_n_tile(handle, v);
    Ok(())
}
