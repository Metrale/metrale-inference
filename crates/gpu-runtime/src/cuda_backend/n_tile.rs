// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-29: Read the launch facts a kernel publishes, at handle resolution: its N tile
//! width and whether it casts its A operand to E4M3 with no scale.
//!
//! Owner: gpu-runtime (CUDA backend).
//! Invariants: a handle's recorded facts are the values of its module's `<entry>_n_tile`
//! and `<entry>_a_e4m3` device symbols, read once when the handle is resolved.

use anyhow::Result;

use crate::gpu::KernelHandle;
use crate::op_cache::OpCache;
use crate::registry::MetraleRegistry;

/// 2026-09-29: Record what `module` publishes for `func_name` on `handle`: the 4-byte
/// `<func_name>_n_tile` (`GpuBackend::kernel_n_tile`) and a non-zero 4-byte
/// `<func_name>_a_e4m3` (`GpuBackend::kernel_casts_a_to_e4m3`). Most kernels publish
/// neither; an absent symbol records nothing and is not an error. A symbol that exists
/// but cannot be read is.
pub(super) fn record_published(
    registry: &MetraleRegistry,
    op_cache: &OpCache,
    module: &str,
    func_name: &str,
    handle: KernelHandle,
) -> Result<()> {
    if let Some(v) = read_u32(registry, module, &format!("{func_name}_n_tile"))? {
        op_cache.record_n_tile(handle, v);
    }
    if read_u32(registry, module, &format!("{func_name}_a_e4m3"))?.is_some_and(|v| v != 0) {
        op_cache.record_a_e4m3(handle);
    }
    Ok(())
}

/// 2026-09-29: The value of `module`'s 4-byte device symbol `name`, `None` when the module
/// defines no such symbol.
fn read_u32(registry: &MetraleRegistry, module: &str, name: &str) -> Result<Option<u32>> {
    let Ok((dptr, 4)) = registry.device_symbol(module, name) else {
        return Ok(None);
    };
    let mut v = 0u32;
    let stream = registry.raw_stream();
    // 2026-09-29: SAFETY: `v` is 4 bytes and outlives the synchronize below.
    unsafe { registry.copy_d2h_async((&mut v as *mut u32).cast(), dptr, 4, stream) }
        .and_then(|_| registry.stream_synchronize(stream))
        .map_err(|e| anyhow::anyhow!("{module}::{name}: {e}"))?;
    Ok(Some(v))
}
