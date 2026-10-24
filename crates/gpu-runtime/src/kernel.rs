// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `KernelModule`, a PTX module loaded through cudarc.
//!
//! Owner: gpu-runtime.
//! Invariants: none beyond the types.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, CudaModule};
use cudarc::nvrtc::Ptx;

use metrale_core::error::{MetraleError, Result};

/// 2026-09-25: A PTX module loaded into a CUDA context through cudarc.
pub struct KernelModule {
    module: Arc<CudaModule>,
}

impl KernelModule {
    /// 2026-09-25: Load PTX source text into `ctx`.
    pub fn from_ptx_src(ctx: &Arc<CudaContext>, ptx_src: &str) -> Result<Self> {
        let ptx = Ptx::from_src(ptx_src);
        let module = ctx
            .load_module(ptx)
            .map_err(|e| MetraleError::ModuleLoad(format!("PTX load failed: {e}")))?;
        Ok(Self { module })
    }

    /// 2026-09-25: Look up a kernel function by name.
    pub fn get_function(&self, name: &str) -> Result<CudaFunction> {
        self.module
            .load_function(name)
            .map_err(|e| MetraleError::ModuleLoad(format!("Function '{name}' not found: {e}")))
    }
}
