// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: The wake keeper's decisions as pure functions: the mode and core parsed from the
//! lever text, the core the GPU interrupt lands on read off `/proc/interrupts`, and which threads
//! of the process are the ones a collective waits on.
//!
//! Owner: gpu-runtime.
//! Invariants: no I/O here (`os.rs` reads the files and hands their text in).

use anyhow::{Context, Result, bail, ensure};

/// 2026-10-21: How the keeper holds its core out of deep idle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeeperMode {
    /// 2026-10-21: A `SCHED_IDLE` thread spinning on the `yield` hint: the core never idles, and
    /// any other runnable thread of the core preempts the spinner at once.
    SpinIdle,
    /// 2026-10-21: A normal-policy thread calling `sched_yield` in a loop.
    SpinYield,
    /// 2026-10-21: A normal-policy thread spinning on the `yield` hint (no syscall).
    SpinPause,
    /// 2026-10-21: A thread sleeping `period_us` at a time (timer slack 1 ns), so the idle
    /// governor always sees a timer at most `period_us` away and never picks a state whose
    /// target residency is longer.
    Timer { period_us: u32 },
}

impl KeeperMode {
    /// 2026-10-21: `off` (`None`), `spin-idle`, `spin-yield`, `spin-pause` or `timer:<us>`
    /// (1..=100000).
    ///
    /// # Errors
    /// Any other text.
    pub fn parse(text: &str) -> Result<Option<Self>> {
        Ok(Some(match text.trim() {
            "off" => return Ok(None),
            "spin-idle" => Self::SpinIdle,
            "spin-yield" => Self::SpinYield,
            "spin-pause" => Self::SpinPause,
            t => {
                let us = t
                    .strip_prefix("timer:")
                    .with_context(|| format!("wake keeper mode {t:?}: expected off, spin-idle, spin-yield, spin-pause or timer:<us>"))?;
                let period_us: u32 = us
                    .parse()
                    .with_context(|| format!("wake keeper timer period {us:?}"))?;
                ensure!(
                    (1..=100_000).contains(&period_us),
                    "wake keeper timer period {period_us} us: expected 1..=100000"
                );
                Self::Timer { period_us }
            }
        }))
    }

    /// 2026-10-21: `mode` as one number the ranks compare at startup: 0 off, 1 spin-idle,
    /// 2 spin-yield, 3 spin-pause, 1_000_000 + the period for a timer.
    #[must_use]
    pub fn agreement_value(mode: Option<Self>) -> u64 {
        match mode {
            None => 0,
            Some(Self::SpinIdle) => 1,
            Some(Self::SpinYield) => 2,
            Some(Self::SpinPause) => 3,
            Some(Self::Timer { period_us }) => 1_000_000 + u64::from(period_us),
        }
    }

    /// 2026-10-21: The mode as `parse` reads it.
    #[must_use]
    pub fn label(self) -> String {
        match self {
            Self::SpinIdle => "spin-idle".into(),
            Self::SpinYield => "spin-yield".into(),
            Self::SpinPause => "spin-pause".into(),
            Self::Timer { period_us } => format!("timer:{period_us}"),
        }
    }
}

/// 2026-10-21: Which core the keeper holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoreChoice {
    /// 2026-10-21: This logical CPU.
    Cpu(u32),
    /// 2026-10-21: The CPU the GPU's first interrupt vector is delivered to ([`gpu_irq_core`]).
    GpuIrq,
}

impl CoreChoice {
    /// 2026-10-21: `gpu-irq` or a CPU number.
    ///
    /// # Errors
    /// Any other text.
    pub fn parse(text: &str) -> Result<Self> {
        match text.trim() {
            "gpu-irq" => Ok(Self::GpuIrq),
            t => t
                .parse()
                .map(Self::Cpu)
                .with_context(|| format!("wake keeper core {t:?}: expected gpu-irq or a CPU number")),
        }
    }
}

/// 2026-10-21: The interrupt number of the first `nvidia` line of `/proc/interrupts` (the GPU's
/// vector 0, which carries its completion interrupts: 1.6M of them on GB10 against tens of
/// thousands on the other vectors).
///
/// # Errors
/// No `nvidia` line, or its number does not parse.
pub fn gpu_irq(interrupts: &str) -> Result<u32> {
    let line = interrupts
        .lines()
        .find(|l| l.split_whitespace().last() == Some("nvidia"))
        .context("no nvidia interrupt in /proc/interrupts")?;
    let n = line.split(':').next().unwrap_or_default().trim();
    n.parse()
        .with_context(|| format!("nvidia interrupt number {n:?}"))
}

/// 2026-10-21: The single CPU of an `effective_affinity_list` (`"10"`).
///
/// # Errors
/// A list of more than one CPU, a range, or no CPU: the interrupt has no single home core.
pub fn single_cpu(affinity_list: &str) -> Result<u32> {
    let t = affinity_list.trim();
    if t.contains([',', '-']) || t.is_empty() {
        bail!("interrupt effective affinity {t:?} is not one CPU");
    }
    t.parse()
        .with_context(|| format!("interrupt effective affinity {t:?}"))
}

/// 2026-10-21: A thread a collective waits on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    /// 2026-10-21: A CUDA driver helper (`cuda-EvtHandlr`, `cuda<hex>`): the host-function
    /// worker that runs NCCL's proxy-upload callbacks and the event handler that wakes it.
    DriverHelper,
    /// 2026-10-21: An NCCL proxy thread (`NCCL Progress`, `NCCL Service`). NCCL names its threads
    /// only with `NCCL_SET_THREAD_NAME=1`.
    CollectiveProxy,
}

/// 2026-10-21: The role of a thread from its `comm` (at most 15 bytes).
#[must_use]
pub fn classify(comm: &str) -> Option<Role> {
    let c = comm.trim_end();
    if c.starts_with("NCCL Progress") || c.starts_with("NCCL Service") {
        Some(Role::CollectiveProxy)
    } else if c == "cuda-EvtHandlr"
        || c.strip_prefix("cuda")
            .is_some_and(|h| !h.is_empty() && h.bytes().all(|b| b.is_ascii_hexdigit()))
    {
        Some(Role::DriverHelper)
    } else {
        None
    }
}

#[cfg(test)]
#[path = "plan_tests.rs"]
mod tests;
