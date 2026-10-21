// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: Which launches of a [`super::MetraleCudaBackend`] carry programmatic stream
//! serialization (`registry_pdl`): every kernel handle resolved from an entry that waits on its
//! grid dependencies (`MetraleRegistry::entry_waits`), once the model has switched the mode on
//! (`GpuBackend::enable_programmatic_launches`). The switch is one-way: set once at load, never
//! cleared, so every launch after it, captured or eager, takes the same path.
//!
//! Owner: gpu-runtime.
//! Invariants:
//! - A handle is listed only when its entry's own PTX body holds `griddepcontrol.wait`.
//! - Before the switch, and for every unlisted handle, launches go through `cuLaunchKernel`
//!   unchanged.

use std::collections::HashSet;
use std::sync::OnceLock;

use cudarc::driver::LaunchConfig;
use parking_lot::RwLock;

use super::MetraleCudaBackend;
use crate::registry::RawCudaFunc;

/// 2026-10-21: The mode and the waiting handles of one backend.
#[derive(Default)]
pub(crate) struct ProgrammaticLaunches {
    on: OnceLock<()>,
    waiting: RwLock<HashSet<u64>>,
}

impl ProgrammaticLaunches {
    /// 2026-10-21: Note that `handle` was resolved from a waiting entry.
    pub(crate) fn note_waiting(&self, handle: u64) {
        self.waiting.write().insert(handle);
    }

    /// 2026-10-21: Switch the mode on (idempotent).
    pub(crate) fn switch_on(&self) {
        let _ = self.on.set(());
    }

    /// 2026-10-21: Whether a launch of `handle` carries the attribute.
    pub(crate) fn applies(&self, handle: u64) -> bool {
        self.on.get().is_some() && self.waiting.read().contains(&handle)
    }

    /// 2026-10-21: How many resolved handles wait on their grid dependencies.
    pub(crate) fn waiting_count(&self) -> usize {
        self.waiting.read().len()
    }
}

impl MetraleCudaBackend {
    /// 2026-10-21: `GpuBackend::enable_programmatic_launches`: switch the mode on and report
    /// it. Under `metrale_scale` (no `cuLaunchKernelEx`) the mode stays off and this says so.
    pub(crate) fn enable_programmatic(&self) -> bool {
        #[cfg(metrale_scale)]
        {
            tracing::info!(
                "programmatic dependent launch: unavailable on this driver, launches stay serialised"
            );
            false
        }
        #[cfg(not(metrale_scale))]
        {
            self.pdl.switch_on();
            tracing::info!(
                "programmatic dependent launch: on for {} resolved kernels that wait on their grid dependencies",
                self.pdl.waiting_count()
            );
            true
        }
    }

    /// 2026-10-21: Launch `raw_func` on `stream`, programmatic when [`ProgrammaticLaunches::applies`].
    ///
    /// # Safety
    /// As `MetraleRegistry::launch_on_stream`.
    pub(super) unsafe fn launch_raw(
        &self,
        raw_func: RawCudaFunc,
        cfg: LaunchConfig,
        stream: u64,
        params: &mut [*mut std::ffi::c_void],
    ) -> metrale_core::error::Result<()> {
        #[cfg(not(metrale_scale))]
        if self.pdl.applies(raw_func.0 as u64) {
            // 2026-10-21: SAFETY: `applies` holds only for a handle whose entry waits.
            return unsafe {
                self.registry()
                    .launch_on_stream_programmatic(raw_func, cfg, stream, params)
            };
        }
        unsafe {
            self.registry()
                .launch_on_stream(raw_func, cfg, stream, params)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ProgrammaticLaunches;

    #[test]
    fn only_a_waiting_handle_after_the_switch_is_programmatic() {
        let p = ProgrammaticLaunches::default();
        p.note_waiting(7);
        assert!(!p.applies(7), "off until switched on");
        p.switch_on();
        assert!(p.applies(7));
        assert!(!p.applies(8), "a handle that does not wait never is");
        p.switch_on();
        assert!(p.applies(7), "switching on twice keeps it on");
        assert_eq!(p.waiting_count(), 1);
    }
}
