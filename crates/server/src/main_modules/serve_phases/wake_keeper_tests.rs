// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: The wake keeper's lever resolution: off stays off whatever the core says, on
//! needs an explicit core (no default core), and bad text is refused.
//!
//! Owner: server startup.
//! Invariants: pure (no thread is started here; the keeper's lifecycle is tested in gpu-runtime).

use super::*;

#[test]
fn off_is_none_with_or_without_a_core() {
    assert_eq!(wake_plan("off", None).unwrap(), None);
    assert_eq!(wake_plan("off", Some("10")).unwrap(), None);
}

#[test]
fn on_needs_an_explicit_core() {
    let e = wake_plan("spin-idle", None).unwrap_err();
    assert!(format!("{e:#}").contains("METRALE_HOST_WAKE_CORE"), "{e:#}");
    assert_eq!(
        wake_plan("timer:250", Some("gpu-irq")).unwrap(),
        Some(WakePlan {
            mode: KeeperMode::Timer { period_us: 250 },
            core: CoreChoice::GpuIrq,
        })
    );
    assert_eq!(
        wake_plan("spin-idle", Some("10")).unwrap(),
        Some(WakePlan {
            mode: KeeperMode::SpinIdle,
            core: CoreChoice::Cpu(10),
        })
    );
}

#[test]
fn bad_text_is_refused() {
    assert!(wake_plan("on", Some("10")).is_err());
    assert!(wake_plan("spin-idle", Some("ten")).is_err());
}
