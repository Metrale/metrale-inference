// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: Thread-block cluster occupancy (`cuOccupancyMaxActiveClusters`): how many
//! clusters of a kernel the device holds at once at a launch shape, so a launcher whose
//! blocks wait on each other can refuse a shape that cannot be placed before it launches.
//!
//! Owner: gpu-runtime (CUDA backend).
//! Invariants:
//! - The kernel carries its cluster dims at compile time (`__cluster_dims__`); the query
//!   passes no launch attributes, so it reads those dims.
//! - The kernel is opted in to `shared_mem` bytes of dynamic shared memory first, as the
//!   launch does, so the answer is the launch's.

use std::ffi::c_void;
use std::sync::OnceLock;

use anyhow::{Result, anyhow, bail};

use super::MetraleCudaBackend;
use super::tensormap::libcuda_symbol;
use crate::gpu::KernelHandle;
use crate::registry::{RawCudaFunc, cuda_error_text, opt_in_dynamic_smem};

/// 2026-10-15: `CUlaunchConfig` (cuda.h), with no attributes.
#[repr(C)]
struct CuLaunchConfig {
    grid: [u32; 3],
    block: [u32; 3],
    shared_mem: u32,
    stream: *mut c_void,
    attrs: *mut c_void,
    num_attrs: u32,
}

/// 2026-10-15: The prototype of `cuOccupancyMaxActiveClusters` in `cuda.h`, resolved in the
/// loaded driver once (`libcuda_symbol`): the CUDA 13.0 link stub does not export it.
type MaxActiveClustersFn = unsafe extern "C" fn(
    num_clusters: *mut i32,
    func: *mut c_void,
    config: *const CuLaunchConfig,
) -> i32;

fn max_active_clusters_fn() -> Result<MaxActiveClustersFn> {
    static CACHED: OnceLock<Option<usize>> = OnceLock::new();
    let addr = CACHED
        .get_or_init(|| libcuda_symbol(c"cuOccupancyMaxActiveClusters"))
        .ok_or_else(|| anyhow!("the loaded libcuda has no cuOccupancyMaxActiveClusters"))?;
    // 2026-10-15: SAFETY: `addr` is libcuda's `cuOccupancyMaxActiveClusters`, whose prototype
    // in `cuda.h` is `MaxActiveClustersFn`.
    Ok(unsafe { std::mem::transmute::<usize, MaxActiveClustersFn>(addr) })
}

impl MetraleCudaBackend {
    /// 2026-10-15: `GpuBackend::max_active_clusters`. A failed query is an error, never 0.
    pub(super) fn max_active_clusters_cu(
        &self,
        kernel: KernelHandle,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem: u32,
    ) -> Result<u32> {
        let raw = RawCudaFunc(kernel.0 as *mut c_void);
        opt_in_dynamic_smem(raw, shared_mem)?;
        let config = CuLaunchConfig {
            grid,
            block,
            shared_mem,
            stream: std::ptr::null_mut(),
            attrs: std::ptr::null_mut(),
            num_attrs: 0,
        };
        let mut n: i32 = 0;
        let query = max_active_clusters_fn()?;
        // 2026-10-15: SAFETY: `raw` is a resolved CUfunction, `config` outlives the call and
        // names no attributes, and `n` is a valid out-pointer.
        let status = unsafe { query(&mut n, raw.0, &config) };
        if status != 0 {
            bail!(
                "cuOccupancyMaxActiveClusters(grid {grid:?}, block {block:?}, smem {shared_mem}) \
                 failed: {}",
                cuda_error_text(status)
            );
        }
        if n < 0 {
            bail!("cuOccupancyMaxActiveClusters reported {n} clusters");
        }
        Ok(n as u32)
    }
}
