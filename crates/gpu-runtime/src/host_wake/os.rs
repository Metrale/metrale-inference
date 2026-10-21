// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: The Linux calls of the wake keeper: thread ids, CPU affinity, the `SCHED_IDLE`
//! policy, timer slack, the process's task list and the GPU interrupt's home core. Nothing here
//! changes host power management (no cpuidle, sysfs or PM-QoS write): only this process's own
//! threads.
//!
//! Owner: gpu-runtime.
//! Invariants: every call acts on a thread of this process.

use anyhow::{Context, Result, ensure};

use super::plan;

/// 2026-10-21: The calling thread's kernel id.
#[must_use]
pub fn gettid() -> i32 {
    // SAFETY: gettid takes no arguments and cannot fail.
    unsafe { libc::gettid() }
}

/// 2026-10-21: Restrict thread `tid` (0 = the caller) to CPU `cpu`.
///
/// # Errors
/// The kernel refuses (no such thread, or `cpu` outside the process's cpuset).
pub fn pin(tid: i32, cpu: u32) -> Result<()> {
    set_affinity(tid, &[cpu])
}

/// 2026-10-21: Restrict thread `tid` (0 = the caller) to `cpus`.
///
/// # Errors
/// A CPU beyond `CPU_SETSIZE`, or the kernel refuses.
pub fn set_affinity(tid: i32, cpus: &[u32]) -> Result<()> {
    // SAFETY: a zeroed cpu_set_t is the empty set; CPU_SET writes inside it (checked bound).
    let rc = unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        for &cpu in cpus {
            ensure!(
                (cpu as usize) < libc::CPU_SETSIZE as usize,
                "cpu {cpu} beyond CPU_SETSIZE"
            );
            libc::CPU_SET(cpu as usize, &mut set);
        }
        libc::sched_setaffinity(tid, std::mem::size_of::<libc::cpu_set_t>(), &set)
    };
    ensure!(
        rc == 0,
        "sched_setaffinity(tid {tid}, cpus {cpus:?}): {}",
        std::io::Error::last_os_error()
    );
    Ok(())
}

/// 2026-10-21: The CPUs thread `tid` (0 = the caller) may run on.
///
/// # Errors
/// The kernel refuses (no such thread).
pub fn affinity(tid: i32) -> Result<Vec<u32>> {
    // SAFETY: sched_getaffinity fills the zeroed set; CPU_ISSET reads inside it.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        let rc = libc::sched_getaffinity(tid, std::mem::size_of::<libc::cpu_set_t>(), &mut set);
        ensure!(
            rc == 0,
            "sched_getaffinity(tid {tid}): {}",
            std::io::Error::last_os_error()
        );
        Ok((0..libc::CPU_SETSIZE as u32)
            .filter(|&c| libc::CPU_ISSET(c as usize, &set))
            .collect())
    }
}

/// 2026-10-21: Put the caller under `SCHED_IDLE` (lowering a thread's own policy needs no
/// privilege).
///
/// # Errors
/// The kernel refuses.
pub fn sched_idle_self() -> Result<()> {
    let param = libc::sched_param { sched_priority: 0 };
    // SAFETY: param outlives the call; pid 0 is the calling thread.
    let rc = unsafe { libc::sched_setscheduler(0, libc::SCHED_IDLE, &param) };
    ensure!(
        rc == 0,
        "sched_setscheduler(SCHED_IDLE): {}",
        std::io::Error::last_os_error()
    );
    Ok(())
}

/// 2026-10-21: Thread `tid`'s scheduling policy (`libc::SCHED_*`; 0 = the caller).
#[must_use]
pub fn sched_policy(tid: i32) -> i32 {
    // SAFETY: plain query.
    unsafe { libc::sched_getscheduler(tid) }
}

/// 2026-10-21: The caller's timer slack, so a short sleep wakes on time instead of up to 50 us
/// late (the default slack).
///
/// # Errors
/// The kernel refuses.
pub fn timer_slack_self(ns: u64) -> Result<()> {
    // SAFETY: PR_SET_TIMERSLACK takes the slack as its only argument.
    let rc = unsafe { libc::prctl(libc::PR_SET_TIMERSLACK, ns as libc::c_ulong, 0, 0, 0) };
    ensure!(
        rc == 0,
        "prctl(PR_SET_TIMERSLACK): {}",
        std::io::Error::last_os_error()
    );
    Ok(())
}

/// 2026-10-21: CPU time the calling thread has used, ns.
#[must_use]
pub fn thread_cpu_ns() -> u64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: ts outlives the call.
    unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut ts) };
    ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
}

/// 2026-10-21: One spin-wait hint (`yield` on aarch64), never a wait-for-event.
#[inline]
pub fn pause() {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: `yield` is a hint with no architectural effect.
    unsafe {
        std::arch::asm!("yield", options(nomem, nostack, preserves_flags));
    }
    #[cfg(not(target_arch = "aarch64"))]
    std::hint::spin_loop();
}

/// 2026-10-21: This process's threads as (tid, comm).
///
/// # Errors
/// `/proc/self/task` cannot be listed. A thread that exits while it is read is skipped.
pub fn tasks() -> Result<Vec<(i32, String)>> {
    let mut out = Vec::new();
    for e in std::fs::read_dir("/proc/self/task").context("listing /proc/self/task")? {
        let e = e?;
        let Some(tid) = e.file_name().to_str().and_then(|s| s.parse().ok()) else {
            continue;
        };
        if let Ok(comm) = std::fs::read_to_string(e.path().join("comm")) {
            out.push((tid, comm.trim_end_matches('\n').to_owned()));
        }
    }
    Ok(out)
}

/// 2026-10-21: The CPU the GPU's completion interrupt is delivered to.
///
/// # Errors
/// No nvidia interrupt, or its effective affinity is not one CPU.
pub fn gpu_irq_cpu() -> Result<u32> {
    let irq = plan::gpu_irq(&std::fs::read_to_string("/proc/interrupts")?)?;
    let path = format!("/proc/irq/{irq}/effective_affinity_list");
    plan::single_cpu(&std::fs::read_to_string(&path).with_context(|| path.clone())?)
        .with_context(|| format!("GPU interrupt {irq}"))
}
