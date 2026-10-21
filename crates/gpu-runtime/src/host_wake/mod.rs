// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: Host wake keeper (`METRALE_HOST_WAKE_KEEPER`): keep the threads a collective waits
//! on (the CUDA driver's host-function worker that runs NCCL's proxy-upload callbacks, and NCCL's
//! proxy threads) on one core that this process holds out of deep idle, so the first collective
//! of a graph does not wait for a CPU to leave a 433 us idle state. Changes no host power
//! setting: the keeper is an ordinary thread of this process.
//!
//! Measured (`host_wake_bench`, spark-256a, 2026-10-21, load < 2): after 2.5 / 5 ms of GPU work a
//! host task plus a condvar-woken proxy thread waited p50 1115 / 1344 us; with `timer:1000` on the
//! GPU interrupt's core and the threads pinned there 68 / 68 us at ~0 CPU (spinners: the same at
//! one core; every core busy: 40). Pinning alone halves it; a keeper on another core does not
//! remove it: the interrupt's core itself must stay out of deep idle.
//!
//! Owner: gpu-runtime.
//! Invariants: see `keeper.rs`; `plan.rs` is pure and builds on every OS, the threads and
//! syscalls only on Linux.

#[cfg(target_os = "linux")]
mod keeper;
#[cfg(target_os = "linux")]
pub mod os;
mod plan;

#[cfg(target_os = "linux")]
pub use keeper::{KEEPER_NAME, KeeperConfig, KeeperCost, Pinned, WakeKeeper, select};
pub use plan::{CoreChoice, KeeperMode, Role, classify, gpu_irq, single_cpu};
