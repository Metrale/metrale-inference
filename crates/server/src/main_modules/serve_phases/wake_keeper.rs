// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: `METRALE_HOST_WAKE_KEEPER` at serve start: resolve the mode and the core, find the
//! CUDA driver's host-function thread with a probe, and start the keeper
//! (`metrale_gpu_runtime::host_wake`), which pins that thread, the other CUDA driver helpers and
//! NCCL's proxy threads to the core and holds the core out of deep idle.
//!
//! Owner: server startup (`met serve`).
//! Invariants:
//! - Started after the communicator exists (its proxy threads are pinned at start; later ones by
//!   the keeper's rescan) and owned by the thread that runs the model (the scheduler on rank 0,
//!   the command loop on a worker), so it stops when the model's user ends.
//! - On a multi-rank serve the keeper refuses to start without a named NCCL proxy thread.

use anyhow::{Context, Result};
use metrale_gpu_runtime::host_wake::{CoreChoice, KeeperMode};

/// 2026-10-21: The keeper a serve asked for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct WakePlan {
    pub mode: KeeperMode,
    pub core: CoreChoice,
}

/// 2026-10-21: The plan from the two levers' text: `None` when the keeper is off.
///
/// # Errors
/// An unknown mode or core, or the keeper on without a core.
pub(crate) fn wake_plan(keeper: &str, core: Option<&str>) -> Result<Option<WakePlan>> {
    let Some(mode) = KeeperMode::parse(keeper).context("METRALE_HOST_WAKE_KEEPER")? else {
        return Ok(None);
    };
    let core = core.with_context(|| {
        format!(
            "METRALE_HOST_WAKE_KEEPER={} needs METRALE_HOST_WAKE_CORE (gpu-irq or a CPU number)",
            mode.label()
        )
    })?;
    Ok(Some(WakePlan {
        mode,
        core: CoreChoice::parse(core).context("METRALE_HOST_WAKE_CORE")?,
    }))
}

/// 2026-10-21: A running keeper (dropping it stops it).
#[cfg(all(feature = "cuda", target_os = "linux"))]
pub(crate) type KeeperGuard = metrale_gpu_runtime::host_wake::WakeKeeper;
/// 2026-10-21: No keeper on this build; [`start`] refuses one.
#[cfg(not(all(feature = "cuda", target_os = "linux")))]
#[derive(Debug)]
pub(crate) struct KeeperGuard;

/// 2026-10-21: The plan `settings` asks for (resolved once, before the load: ROTI).
///
/// # Errors
/// As [`wake_plan`].
pub(crate) fn plan_from_settings(
    settings: &metrale_settings::InferenceServerSettings,
) -> Result<Option<WakePlan>> {
    let rt = &settings.runtime;
    wake_plan(&rt.host_wake_keeper, rt.host_wake_core.as_deref())
}

/// 2026-10-21: The keeper `plan` asks for, started; `None` when it is off.
///
/// # Errors
/// The core cannot be resolved; the probe or the keeper fails; a multi-rank serve has no named
/// NCCL proxy thread; or this build has no keeper.
pub(crate) fn start(
    plan: Option<WakePlan>,
    gpu: &dyn metrale_gpu_runtime::gpu::GpuBackend,
    world_size: usize,
) -> Result<Option<KeeperGuard>> {
    plan.map(|p| start_plan(p, gpu, world_size)).transpose()
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
fn start_plan(
    plan: WakePlan,
    gpu: &dyn metrale_gpu_runtime::gpu::GpuBackend,
    world_size: usize,
) -> Result<KeeperGuard> {
    use metrale_gpu_runtime::host_wake::{KeeperConfig, Role, WakeKeeper, os};
    let cpu = match plan.core {
        CoreChoice::Cpu(c) => c,
        CoreChoice::GpuIrq => os::gpu_irq_cpu().context("METRALE_HOST_WAKE_CORE=gpu-irq")?,
    };
    let host_fn = metrale_gpu_runtime::cuda_backend::host_task_probe::host_task_tid(
        gpu.default_stream(),
    )?;
    let keeper = WakeKeeper::start(KeeperConfig {
        mode: plan.mode,
        cpu,
        roles: vec![Role::CudaDriver, Role::NcclProxy],
        tids: vec![host_fn],
        rescan: std::time::Duration::from_secs(2),
    })?;
    let pinned = keeper.pinned();
    anyhow::ensure!(
        world_size == 1 || pinned.iter().any(|p| p.role == Some(Role::NcclProxy)),
        "METRALE_HOST_WAKE_KEEPER: no NCCL proxy thread found by name on a {world_size}-rank \
         serve; NCCL names its threads only with NCCL_SET_THREAD_NAME=1"
    );
    tracing::info!(
        mode = %plan.mode.label(),
        cpu,
        host_fn,
        pinned = ?pinned.iter().map(|p| (p.tid, p.comm.as_str())).collect::<Vec<_>>(),
        "host wake keeper started"
    );
    Ok(keeper)
}

#[cfg(not(all(feature = "cuda", target_os = "linux")))]
fn start_plan(
    plan: WakePlan,
    _gpu: &dyn metrale_gpu_runtime::gpu::GpuBackend,
    _world_size: usize,
) -> Result<KeeperGuard> {
    anyhow::bail!(
        "METRALE_HOST_WAKE_KEEPER={} needs a CUDA build on Linux",
        plan.mode.label()
    )
}

#[cfg(test)]
#[path = "wake_keeper_tests.rs"]
mod tests;
