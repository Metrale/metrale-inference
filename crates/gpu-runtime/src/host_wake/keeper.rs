// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: The wake keeper thread: holds one core out of deep idle ([`KeeperMode`]) and pins
//! the threads a collective waits on to it, rescanning for threads created later (NCCL starts its
//! progress thread at the first connection).
//!
//! Owner: gpu-runtime.
//! Invariants:
//! - The keeper never pins itself by role (its name, [`KEEPER_NAME`], matches no role).
//! - [`WakeKeeper::stop`] (or drop) ends and joins the thread; pinned threads stay pinned.

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};

use super::os;
use super::plan::{KeeperMode, Role, classify};

/// 2026-10-21: The keeper thread's name.
pub const KEEPER_NAME: &str = "metrale-wake";

/// 2026-10-21: What a keeper holds and pins.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KeeperConfig {
    pub mode: KeeperMode,
    pub cpu: u32,
    /// 2026-10-21: The roles whose threads are pinned to `cpu` (empty: none, the keeper alone).
    pub roles: Vec<Role>,
    /// 2026-10-21: Threads pinned whatever their name (the host-function worker found by a probe).
    pub tids: Vec<i32>,
    /// 2026-10-21: How often the task list is rescanned.
    pub rescan: Duration,
}

/// 2026-10-21: A thread the keeper pinned.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pinned {
    pub tid: i32,
    pub comm: String,
    pub role: Option<Role>,
}

/// 2026-10-21: The keeper's cost when it stops.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KeeperCost {
    pub cpu_ns: u64,
    pub wall_ns: u64,
}

impl KeeperCost {
    /// 2026-10-21: CPU time as a share of one core.
    #[must_use]
    pub fn core_share(&self) -> f64 {
        self.cpu_ns as f64 / self.wall_ns.max(1) as f64
    }
}

/// 2026-10-21: The threads of `tasks` to pin that are not in `known` yet: those whose role is in
/// `roles`, and those listed in `tids`. Adds them to `known`. Never the keeper itself.
#[must_use]
pub fn select(
    tasks: &[(i32, String)],
    roles: &[Role],
    tids: &[i32],
    known: &mut BTreeSet<i32>,
) -> Vec<Pinned> {
    let mut out = Vec::new();
    for (tid, comm) in tasks {
        let role = classify(comm);
        let wanted = role.is_some_and(|r| roles.contains(&r)) || tids.contains(tid);
        if wanted && comm != KEEPER_NAME && known.insert(*tid) {
            out.push(Pinned {
                tid: *tid,
                comm: comm.clone(),
                role,
            });
        }
    }
    out
}

fn sweep(cfg: &KeeperConfig, known: &mut BTreeSet<i32>, pinned: &Mutex<Vec<Pinned>>) -> Result<()> {
    for p in select(&os::tasks()?, &cfg.roles, &cfg.tids, known) {
        // 2026-10-21: A thread that exited between the listing and the pin is not an error.
        if let Err(e) = os::pin(p.tid, cfg.cpu) {
            if std::path::Path::new(&format!("/proc/self/task/{}", p.tid)).exists() {
                return Err(e);
            }
            continue;
        }
        tracing::info!(tid = p.tid, comm = %p.comm, cpu = cfg.cpu, "wake keeper: pinned");
        pinned.lock().map_err(|_| anyhow::anyhow!("pinned list poisoned"))?.push(p);
    }
    Ok(())
}

/// 2026-10-21: A running keeper.
#[derive(Debug)]
pub struct WakeKeeper {
    tid: i32,
    stop: Arc<AtomicBool>,
    pinned: Arc<Mutex<Vec<Pinned>>>,
    handle: Option<JoinHandle<Result<KeeperCost>>>,
}

impl WakeKeeper {
    /// 2026-10-21: Pin the threads `cfg` names that exist now, then start the keeper on
    /// `cfg.cpu`.
    ///
    /// # Errors
    /// `cfg.cpu` is outside the CPUs this process may use, a pin fails, or the thread cannot
    /// start or set its policy.
    pub fn start(cfg: KeeperConfig) -> Result<Self> {
        let allowed = os::affinity(0)?;
        ensure!(
            allowed.contains(&cfg.cpu),
            "wake keeper cpu {} is not one this process may use ({allowed:?})",
            cfg.cpu
        );
        let pinned = Arc::new(Mutex::new(Vec::new()));
        let mut known = BTreeSet::new();
        sweep(&cfg, &mut known, &pinned)?;
        let stop = Arc::new(AtomicBool::new(false));
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let (s, p) = (stop.clone(), pinned.clone());
        let handle = std::thread::Builder::new()
            .name(KEEPER_NAME.into())
            .spawn(move || {
                let setup = os::pin(0, cfg.cpu).and_then(|()| match cfg.mode {
                    KeeperMode::SpinIdle => os::sched_idle_self(),
                    KeeperMode::Timer { .. } => os::timer_slack_self(1),
                    KeeperMode::SpinYield | KeeperMode::SpinPause => Ok(()),
                });
                let _ = ready_tx.send(match &setup {
                    Ok(()) => Ok(os::gettid()),
                    Err(e) => Err(format!("{e:#}")),
                });
                setup?;
                run(&cfg, &s, &mut known, &p)
            })
            .context("spawning the wake keeper")?;
        let tid = match ready_rx.recv() {
            Ok(Ok(tid)) => tid,
            Ok(Err(e)) => bail!("wake keeper setup: {e}"),
            Err(_) => bail!("wake keeper thread ended before its setup"),
        };
        Ok(Self {
            tid,
            stop,
            pinned,
            handle: Some(handle),
        })
    }

    /// 2026-10-21: The keeper thread's kernel id.
    #[must_use]
    pub fn tid(&self) -> i32 {
        self.tid
    }

    /// 2026-10-21: The threads pinned so far.
    #[must_use]
    pub fn pinned(&self) -> Vec<Pinned> {
        self.pinned.lock().map(|v| v.clone()).unwrap_or_default()
    }

    /// 2026-10-21: End and join the keeper.
    ///
    /// # Errors
    /// The keeper failed (a rescan pin) or panicked.
    pub fn stop(mut self) -> Result<KeeperCost> {
        self.join()
    }

    fn join(&mut self) -> Result<KeeperCost> {
        self.stop.store(true, Ordering::Release);
        let h = self.handle.take().context("wake keeper already stopped")?;
        h.join()
            .map_err(|_| anyhow::anyhow!("wake keeper panicked"))?
    }
}

impl Drop for WakeKeeper {
    fn drop(&mut self) {
        if self.handle.is_some()
            && let Err(e) = self.join()
        {
            tracing::error!("wake keeper: {e:#}");
        }
    }
}

/// 2026-10-21: The keeper loop: hold the core per `cfg.mode`, rescan every `cfg.rescan`.
fn run(
    cfg: &KeeperConfig,
    stop: &AtomicBool,
    known: &mut BTreeSet<i32>,
    pinned: &Mutex<Vec<Pinned>>,
) -> Result<KeeperCost> {
    let (t0, c0) = (Instant::now(), os::thread_cpu_ns());
    let mut next = t0 + cfg.rescan;
    let mut n: u32 = 0;
    while !stop.load(Ordering::Acquire) {
        match cfg.mode {
            KeeperMode::SpinIdle | KeeperMode::SpinPause => os::pause(),
            KeeperMode::SpinYield => std::thread::yield_now(),
            KeeperMode::Timer { period_us } => {
                std::thread::sleep(Duration::from_micros(u64::from(period_us)));
            }
        }
        n = n.wrapping_add(1);
        let check = matches!(cfg.mode, KeeperMode::Timer { .. }) || n.is_multiple_of(4096);
        if check && Instant::now() >= next {
            sweep(cfg, known, pinned)?;
            next = Instant::now() + cfg.rescan;
        }
    }
    Ok(KeeperCost {
        cpu_ns: os::thread_cpu_ns() - c0,
        wall_ns: t0.elapsed().as_nanos() as u64,
    })
}

#[cfg(test)]
#[path = "keeper_tests.rs"]
mod tests;
