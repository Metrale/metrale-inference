// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The process-scoped CUDA context and stream (`CudaHost`), and
//! `release`, which unloads one model's [`crate::registry`] modules.
//!
//! Owner: gpu-runtime.
//! Invariants:
//! - The process has at most one `CudaHost`, on the ordinal and stream priority
//!   of the first `host` call; a later call with another ordinal or priority is
//!   an error, and a failed first creation is returned as the same error to
//!   every later call.
//! - `release` unloads nothing while another `Arc` to the registry is live.

use std::sync::{Arc, OnceLock};

use cudarc::driver::{CudaContext, CudaStream};

use crate::registry::MetraleRegistry;
use metrale_core::error::{MetraleError, Result};
use metrale_settings::groups::device::StreamPriority;

/// 2026-09-25: The CUDA context and stream for the process. Each model's
/// `MetraleRegistry` loads its modules into this one context.
pub struct CudaHost {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    ordinal: usize,
    priority: StreamPriority,
}

impl CudaHost {
    pub fn ordinal(&self) -> usize {
        self.ordinal
    }
}

// 2026-09-25: SAFETY: the same argument as for `MetraleRegistry` in
// `registry.rs`: these are handles into a context that is never destroyed.
unsafe impl Send for CudaHost {}
unsafe impl Sync for CudaHost {}

/// 2026-09-25: The process's CUDA host. Static because it belongs to the
/// (process, device) pair, not to a model: it is created once and kept across
/// model loads.
static HOST: OnceLock<std::result::Result<Arc<CudaHost>, String>> = OnceLock::new();

/// 2026-09-25: Get, or on the first call create, the process CUDA host on
/// `ordinal`, its stream at `priority`. The first call fixes both; asking for
/// another ordinal or priority is an error.
pub fn host(ordinal: usize, priority: StreamPriority) -> Result<Arc<CudaHost>> {
    let result = HOST.get_or_init(|| {
        let ctx = CudaContext::new(ordinal).map_err(|e| format!("{e}"))?;
        let under = main_stream_under_greatest(priority)?;
        let stream = if priority.high {
            let (least, greatest) =
                cudarc::driver::result::ctx::stream_priority_range().map_err(|e| format!("{e}"))?;
            let prio = main_stream_priority(least, greatest, under)?;
            tracing::info!(
                "METRALE_MAIN_STREAM_HIGH_PRIORITY: process stream at priority {prio} \
                 (range {least}..={greatest}; side streams stay at {least}; \
                 METRALE_MAIN_STREAM_UNDER_GREATEST={})",
                u8::from(under)
            );
            ctx.new_stream_with_priority(prio)
                .map_err(|e| format!("{e}"))?
        } else {
            ctx.new_stream().map_err(|e| format!("{e}"))?
        };
        Ok(Arc::new(CudaHost {
            ctx,
            stream,
            ordinal,
            priority,
        }))
    });
    match result {
        Ok(h) if h.ordinal == ordinal && h.priority == priority => Ok(h.clone()),
        Ok(h) if h.ordinal == ordinal => Err(MetraleError::ModuleLoad(format!(
            "CUDA host stream already created at {:?} — cannot recreate it at {priority:?}",
            h.priority
        ))),
        Ok(h) => Err(MetraleError::ModuleLoad(format!(
            "CUDA host already bound to GPU {} — cannot rebind to {ordinal}",
            h.ordinal
        ))),
        Err(msg) => Err(MetraleError::ModuleLoad(msg.clone())),
    }
}

/// 2026-10-11: `METRALE_MAIN_STREAM_HIGH_PRIORITY=1` creates the process stream at the
/// context's greatest priority; every other stream (`create_stream`) keeps the default, the
/// least. Graphs launched on it run at its priority. On GB10 a large device-to-device copy on a
/// side stream (the SSM rollback, 49 MB) holds every kernel launched after it until it ends; at
/// a higher priority the process stream's kernels are dispatched ahead of the copy's. That is
/// `priority.high`.
///
/// 2026-10-14: `METRALE_MAIN_STREAM_UNDER_GREATEST=1` (`priority.under_greatest`), which needs
/// `high`: the process stream sits one level under the greatest priority, so a stream made at the
/// greatest (`create_stream_greatest_priority`: the GLM SP comm stream) is dispatched strictly
/// ahead of it instead of tying with it.
fn main_stream_under_greatest(priority: StreamPriority) -> std::result::Result<bool, String> {
    let on = priority.under_greatest;
    if on && !priority.high {
        return Err(
            "METRALE_MAIN_STREAM_UNDER_GREATEST=1 needs METRALE_MAIN_STREAM_HIGH_PRIORITY=1".into(),
        );
    }
    Ok(on)
}

/// 2026-10-14: The high-priority process stream's priority in the context's range (`least` the
/// numerically greatest value, `greatest` the smallest): `greatest`, or one level under it when
/// `under`, which must still be strictly above `least` so side streams stay below it.
fn main_stream_priority(
    least: i32,
    greatest: i32,
    under: bool,
) -> std::result::Result<i32, String> {
    if !under {
        return Ok(greatest);
    }
    let p = greatest + 1;
    if p >= least {
        return Err(format!(
            "METRALE_MAIN_STREAM_UNDER_GREATEST=1: the priority range {least}..={greatest} has no \
             level between the greatest and the side streams' least"
        ));
    }
    Ok(p)
}

/// 2026-09-25: Unload a model's kernel modules. Errors, unloading nothing, while
/// another strong `Arc` to the registry is live; otherwise drops every module
/// the registry holds.
pub fn release(registry: Arc<MetraleRegistry>) -> Result<()> {
    let outstanding = Arc::strong_count(&registry) - 1;
    if outstanding > 0 {
        return Err(MetraleError::ModuleLoad(format!(
            "cannot release the kernel modules: {outstanding} handle(s) are still live. \
             Something is holding the previous model's registry — find it before swapping, \
             or the next model will run against unloaded modules."
        )));
    }
    let mut owned = Arc::try_unwrap(registry).map_err(|_| {
        MetraleError::ModuleLoad("registry handle count changed during release".to_string())
    })?;
    let failures = owned.unload_raw();
    if !failures.is_empty() {
        return Err(MetraleError::ModuleLoad(format!(
            "{} module(s) failed to unload: {}",
            failures.len(),
            failures.join("; ")
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn under_greatest_needs_high_priority() {
        let p = |high, under_greatest| StreamPriority {
            high,
            under_greatest,
        };
        assert_eq!(main_stream_under_greatest(p(false, false)), Ok(false));
        assert_eq!(main_stream_under_greatest(p(true, false)), Ok(false));
        assert_eq!(main_stream_under_greatest(p(true, true)), Ok(true));
        assert_eq!(
            main_stream_under_greatest(p(false, true)),
            Err(
                "METRALE_MAIN_STREAM_UNDER_GREATEST=1 needs METRALE_MAIN_STREAM_HIGH_PRIORITY=1"
                    .into()
            )
        );
    }

    #[test]
    fn under_greatest_is_one_level_below_and_above_the_least() {
        assert_eq!(main_stream_priority(0, -5, false), Ok(-5));
        assert_eq!(main_stream_priority(0, -5, true), Ok(-4));
        assert!(main_stream_priority(0, -1, true).is_err());
        assert!(main_stream_priority(0, 0, true).is_err());
    }
}
