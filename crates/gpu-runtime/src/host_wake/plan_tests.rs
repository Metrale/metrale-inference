// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: The wake keeper's parsing and classification on the texts the GB10 hosts produce.
//!
//! Owner: gpu-runtime.
//! Invariants: fixtures are verbatim shapes from spark-256a (`/proc/interrupts`, thread comms).

use super::*;

/// 2026-10-21: Three lines of spark-256a's `/proc/interrupts` (columns trimmed to 4 CPUs): an
/// mlx5 vector, then the GPU's first two vectors.
const INTERRUPTS: &str = "           CPU0       CPU1       CPU2       CPU3
395:     263526          0          0          0 ITS-PCI-MSIX-0000:01:00.0   1 Edge      mlx5_comp0@pci:0000:01:00.0
482:          0          0          0          0 ITS-PCI-MSIX-000f:01:00.0   0 Edge      nvidia
483:          0          0          0          0 ITS-PCI-MSIX-000f:01:00.0   1 Edge      nvidia
";

#[test]
fn gpu_irq_is_the_first_nvidia_vector() {
    assert_eq!(gpu_irq(INTERRUPTS).unwrap(), 482);
}

#[test]
fn gpu_irq_refuses_a_host_without_one() {
    let no_gpu: String = INTERRUPTS.lines().take(2).collect::<Vec<_>>().join("\n");
    assert!(gpu_irq(&no_gpu).is_err());
}

#[test]
fn single_cpu_takes_one_cpu_only() {
    assert_eq!(single_cpu("10\n").unwrap(), 10);
    for bad in ["0-19", "0,10", "", "x"] {
        assert!(single_cpu(bad).is_err(), "{bad:?} accepted");
    }
}

#[test]
fn modes_round_trip_and_reject_unknown() {
    for t in ["spin-idle", "spin-yield", "spin-pause", "timer:250"] {
        assert_eq!(KeeperMode::parse(t).unwrap().unwrap().label(), t);
    }
    assert_eq!(KeeperMode::parse("off").unwrap(), None);
    for bad in ["", "on", "1", "timer:", "timer:0", "timer:100001", "timer:-5", "spin"] {
        assert!(KeeperMode::parse(bad).is_err(), "{bad:?} accepted");
    }
}

#[test]
fn agreement_values_are_distinct() {
    let modes = ["off", "spin-idle", "spin-yield", "spin-pause", "timer:1", "timer:250", "timer:100000"];
    let mut v: Vec<u64> = modes
        .iter()
        .map(|m| KeeperMode::agreement_value(KeeperMode::parse(m).unwrap()))
        .collect();
    v.sort_unstable();
    v.dedup();
    assert_eq!(v.len(), modes.len(), "{v:?}");
}

#[test]
fn core_choice_parses() {
    assert_eq!(CoreChoice::parse("gpu-irq").unwrap(), CoreChoice::GpuIrq);
    assert_eq!(CoreChoice::parse("10").unwrap(), CoreChoice::Cpu(10));
    for bad in ["", "irq", "-1", "unset"] {
        assert!(CoreChoice::parse(bad).is_err(), "{bad:?} accepted");
    }
}

/// 2026-10-21: The comms of a GLM-5.3 rank-0 serve (`/proc/<pid>/task/*/comm`, 2026-10-21) plus
/// NCCL's names under `NCCL_SET_THREAD_NAME=1`.
#[test]
fn classify_finds_driver_and_proxy_threads_only() {
    let driver = ["cuda-EvtHandlr", "cuda00002c0000c"];
    let proxy = ["NCCL Progress 0", "NCCL Service  0"];
    let other = [
        "tokio-rt-worker",
        "met",
        "NCCL RAS",
        "NCCL IbAsync  0",
        "cuda",
        "cudaXYZ",
        "metrale-wake",
    ];
    for c in driver {
        assert_eq!(classify(c), Some(Role::DriverHelper), "{c}");
    }
    for c in proxy {
        assert_eq!(classify(c), Some(Role::CollectiveProxy), "{c}");
    }
    for c in other {
        assert_eq!(classify(c), None, "{c}");
    }
}
