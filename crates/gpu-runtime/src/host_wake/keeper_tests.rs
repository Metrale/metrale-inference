// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: The keeper's selection, and its lifecycle on the real OS: it pins the threads it
//! should (present at start and created later), leaves the others alone, and stops.
//!
//! Owner: gpu-runtime.
//! Invariants: the lifecycle tests use only this process's own threads (no privilege, no host
//! setting).

use std::collections::BTreeSet;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use super::*;
use crate::host_wake::os;

fn tasks(v: &[(i32, &str)]) -> Vec<(i32, String)> {
    v.iter().map(|(t, c)| (*t, (*c).to_owned())).collect()
}

#[test]
fn select_takes_roles_and_listed_tids_once() {
    let t = tasks(&[
        (1, "met"),
        (2, "cuda-EvtHandlr"),
        (3, "NCCL Progress 0"),
        (4, "tokio-rt-worker"),
        (5, KEEPER_NAME),
    ]);
    let mut known = BTreeSet::new();
    let got: Vec<i32> = select(&t, &[Role::CudaDriver, Role::NcclProxy], &[4, 5], &mut known)
        .iter()
        .map(|p| p.tid)
        .collect();
    assert_eq!(got, vec![2, 3, 4]);
    assert!(select(&t, &[Role::CudaDriver, Role::NcclProxy], &[4], &mut known).is_empty());
    let mut fresh = BTreeSet::new();
    let only_cuda: Vec<i32> = select(&t, &[Role::CudaDriver], &[], &mut fresh)
        .iter()
        .map(|p| p.tid)
        .collect();
    assert_eq!(only_cuda, vec![2]);
    assert!(select(&t, &[], &[], &mut BTreeSet::new()).is_empty());
}

/// 2026-10-21: A thread named `name` that reports its tid and lives until `stop` drops.
fn named(name: &str) -> (i32, mpsc::Sender<()>, std::thread::JoinHandle<()>) {
    let (tid_tx, tid_rx) = mpsc::channel();
    let (stop_tx, stop_rx) = mpsc::channel::<()>();
    let h = std::thread::Builder::new()
        .name(name.into())
        .spawn(move || {
            tid_tx.send(os::gettid()).unwrap();
            let _ = stop_rx.recv();
        })
        .unwrap();
    (tid_rx.recv().unwrap(), stop_tx, h)
}

fn alive(tid: i32) -> bool {
    std::path::Path::new(&format!("/proc/self/task/{tid}")).exists()
}

fn wait_for(what: &str, f: impl Fn() -> bool) {
    let t0 = Instant::now();
    while !f() {
        assert!(t0.elapsed() < Duration::from_secs(5), "timed out: {what}");
        std::thread::sleep(Duration::from_millis(5));
    }
}

#[test]
fn keeper_pins_present_and_later_threads_and_stops() {
    let allowed = os::affinity(0).unwrap();
    let cpu = *allowed.last().unwrap();
    let (early, s1, h1) = named("NCCL Progress 7");
    let (bystander, s2, h2) = named("bystander");
    let k = WakeKeeper::start(KeeperConfig {
        mode: KeeperMode::Timer { period_us: 1000 },
        cpu,
        roles: vec![Role::NcclProxy],
        tids: vec![],
        rescan: Duration::from_millis(10),
    })
    .unwrap();
    assert_eq!(os::affinity(early).unwrap(), vec![cpu], "present thread not pinned");
    let (late, s3, h3) = named("NCCL Service  7");
    wait_for("the rescan pins a later thread", || {
        os::affinity(late).unwrap() == vec![cpu]
    });
    assert_eq!(os::affinity(bystander).unwrap(), allowed, "bystander pinned");
    let keeper = k.tid();
    assert_eq!(os::affinity(keeper).unwrap(), vec![cpu], "keeper not on its core");
    let pinned: Vec<i32> = k.pinned().iter().map(|p| p.tid).collect();
    assert_eq!(pinned, vec![early, late]);
    let cost = k.stop().unwrap();
    assert!(cost.wall_ns > 0 && cost.core_share() < 0.5, "{cost:?}");
    assert!(!alive(keeper), "keeper thread still alive after stop");
    for (s, h) in [(s1, h1), (s2, h2), (s3, h3)] {
        drop(s);
        h.join().unwrap();
    }
}

#[test]
fn spin_idle_keeper_runs_under_sched_idle_and_drop_joins() {
    let cpu = *os::affinity(0).unwrap().last().unwrap();
    let k = WakeKeeper::start(KeeperConfig {
        mode: KeeperMode::SpinIdle,
        cpu,
        roles: vec![],
        tids: vec![],
        rescan: Duration::from_secs(1),
    })
    .unwrap();
    let keeper = k.tid();
    assert_eq!(os::affinity(keeper).unwrap(), vec![cpu]);
    assert_eq!(os::sched_policy(keeper), libc::SCHED_IDLE);
    drop(k);
    assert!(!alive(keeper), "drop did not join the keeper");
}

#[test]
fn keeper_refuses_a_cpu_the_process_cannot_use() {
    let allowed = os::affinity(0).unwrap();
    let outside = (0..1024).find(|c| !allowed.contains(c)).unwrap();
    let e = WakeKeeper::start(KeeperConfig {
        mode: KeeperMode::Timer { period_us: 1000 },
        cpu: outside,
        roles: vec![],
        tids: vec![],
        rescan: Duration::from_secs(1),
    })
    .unwrap_err();
    assert!(format!("{e:#}").contains("may use"), "{e:#}");
}
