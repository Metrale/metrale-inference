// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Completion polling for NCCL operations, and the unhealthy flag
//! that stops later submissions. The deadline bounds only the polling, not an
//! NCCL or driver call that hangs.
//!
//! Owner: metrale-comm.
//! Invariants:
//! - `poison_on_error` never clears the flag; only a caller storing `false`
//!   does (`NcclBackend`'s reconnect).
use anyhow::{Result, bail};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

/// 2026-09-26: Call `ready` until it returns `true`, pausing between calls.
///
/// # Errors
/// The first error from `ready`, or a deadline error once `elapsed()` reaches
/// `timeout` while not ready.
pub fn poll_completion(
    timeout: Duration,
    mut elapsed: impl FnMut() -> Duration,
    mut ready: impl FnMut() -> Result<bool>,
    mut pause: impl FnMut(),
) -> Result<()> {
    loop {
        if ready()? {
            return Ok(());
        }
        if elapsed() >= timeout {
            bail!(
                "collective completion deadline exceeded after {} ms",
                timeout.as_millis()
            );
        }
        pause();
    }
}

/// 2026-09-26: Call `ready` until it returns `true` or an error, with no
/// deadline: for the first word of a worker command, which may wait through
/// server idle time. `ready` must check transport errors, or a peer that
/// disappears without one is waited for forever.
pub fn poll_idle_command(
    mut ready: impl FnMut() -> Result<bool>,
    mut pause: impl FnMut(),
) -> Result<()> {
    loop {
        if ready()? {
            return Ok(());
        }
        pause();
    }
}

/// 2026-10-09: The environment variable that sets [`PollPause`]: `<spin_us>:<sleep_us>`.
pub const POLL_ENV: &str = "METRALE_COMM_POLL";

/// 2026-10-09: The value [`POLL_ENV`] takes when unset: yield for 200 us, then sleep 50 us
/// per poll. A rank-0 command broadcast completes within the yield window; a worker waiting
/// for its next command through its own decode graph falls through to the short sleeps.
pub const POLL_DEFAULT: &str = "200:50";

/// 2026-10-09: What a completion wait does before its next `cuStreamQuery`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PauseAction {
    Yield,
    Sleep(Duration),
}

/// 2026-10-09: The pause between completion polls. While the wait is younger than `spin`
/// the thread yields, so a broadcast that completes in microseconds is seen in
/// microseconds; after that it sleeps `sleep` per poll. A fixed 1 ms sleep (`0:1000`, the
/// policy before this one) cost about 1.05 ms per command word on every rank: two words
/// per single-sequence decode step on rank 0, measured as 1.86 ms of a 36.6 ms step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PollPause {
    pub spin: Duration,
    pub sleep: Duration,
}

impl PollPause {
    /// 2026-10-09: Parse a [`POLL_ENV`] value, [`POLL_DEFAULT`] when `value` is `None`.
    ///
    /// # Errors
    /// A value that is not two whole numbers of microseconds joined by `:`, or a zero
    /// `sleep` (a busy loop past the yield window must be asked for with a long `spin`).
    pub fn parse(value: Option<&str>) -> Result<Self> {
        let text = value.unwrap_or(POLL_DEFAULT);
        let parsed = text.split_once(':').and_then(|(s, z)| {
            Some((s.trim().parse::<u64>().ok()?, z.trim().parse::<u64>().ok()?))
        });
        let Some((spin_us, sleep_us)) = parsed else {
            bail!("{POLL_ENV} must be <spin_us>:<sleep_us>, got {text:?}");
        };
        anyhow::ensure!(
            sleep_us > 0,
            "{POLL_ENV}: sleep_us must be > 0, got {text:?}"
        );
        Ok(Self {
            spin: Duration::from_micros(spin_us),
            sleep: Duration::from_micros(sleep_us),
        })
    }

    /// 2026-10-09: The pause after a not-ready poll, `waited` into the wait.
    pub fn action(&self, waited: Duration) -> PauseAction {
        if waited < self.spin {
            PauseAction::Yield
        } else {
            PauseAction::Sleep(self.sleep)
        }
    }
}

/// 2026-09-26: Set `unhealthy` when `result` is an error; return `result`.
pub fn poison_on_error(result: Result<()>, unhealthy: &AtomicBool) -> Result<()> {
    if result.is_err() {
        unhealthy.store(true, Ordering::Release);
    }
    result
}

/// 2026-09-26: Refuse a submission while `unhealthy` is set.
pub fn ensure_healthy(unhealthy: &AtomicBool, rank: usize, world: usize, op: &str) -> Result<()> {
    anyhow::ensure!(
        !unhealthy.load(Ordering::Acquire),
        "NCCL rank={rank} world_size={world} op={op}: communicator unhealthy; stop all ranks before retrying"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn a_peer_that_never_completes_returns_at_the_deadline() {
        let ticks = Cell::new(0);
        let err = poll_completion(
            Duration::from_millis(3),
            || Duration::from_millis(ticks.get()),
            || Ok(false),
            || ticks.set(ticks.get() + 1),
        )
        .unwrap_err();
        assert_eq!(ticks.get(), 3);
        assert!(err.to_string().contains("deadline exceeded"));
    }

    #[test]
    fn idle_command_can_arrive_after_long_idle_but_payload_still_times_out() {
        let ticks = Cell::new(0);
        poll_idle_command(|| Ok(ticks.get() == 90), || ticks.set(ticks.get() + 1)).unwrap();
        assert_eq!(ticks.get(), 90);
        ticks.set(0);
        assert!(
            poll_completion(
                Duration::from_secs(30),
                || Duration::from_secs(ticks.get()),
                || Ok(ticks.get() == 90),
                || ticks.set(ticks.get() + 1),
            )
            .is_err()
        );
        assert_eq!(ticks.get(), 30);
    }

    #[test]
    fn idle_command_still_checks_errors_and_poisons_the_communicator() {
        let ticks = Cell::new(0);
        let unhealthy = AtomicBool::new(false);
        let result = poll_idle_command(
            || {
                if ticks.get() == 90 {
                    anyhow::bail!("peer lost during idle");
                }
                Ok(false)
            },
            || ticks.set(ticks.get() + 1),
        );
        assert!(poison_on_error(result, &unhealthy).is_err());
        assert!(ensure_healthy(&unhealthy, 1, 2, "broadcast").is_err());
        assert_eq!(ticks.get(), 90);
    }

    #[test]
    fn completion_failure_poison_blocks_later_submissions() {
        let unhealthy = AtomicBool::new(false);
        assert!(poison_on_error(Ok(()), &unhealthy).is_ok());
        assert!(ensure_healthy(&unhealthy, 3, 8, "all_reduce").is_ok());
        let failure = poll_completion(
            Duration::ZERO,
            || Duration::ZERO,
            || Ok(false),
            || panic!("deadline already reached"),
        );
        assert!(poison_on_error(failure, &unhealthy).is_err());
        let msg = ensure_healthy(&unhealthy, 3, 8, "all_reduce")
            .unwrap_err()
            .to_string();
        assert!(
            msg.contains("rank=3") && msg.contains("op=all_reduce") && msg.contains("unhealthy")
        );
        // 2026-09-26: A later success leaves the flag set.
        assert!(poison_on_error(Ok(()), &unhealthy).is_ok());
        assert!(unhealthy.load(Ordering::Acquire));
    }

    #[test]
    fn poll_pause_yields_inside_the_window_then_sleeps() {
        let p = PollPause::parse(None).unwrap();
        assert_eq!(p, PollPause::parse(Some(POLL_DEFAULT)).unwrap());
        assert_eq!(p.action(Duration::ZERO), PauseAction::Yield);
        assert_eq!(p.action(Duration::from_micros(199)), PauseAction::Yield);
        assert_eq!(
            p.action(Duration::from_micros(200)),
            PauseAction::Sleep(Duration::from_micros(50))
        );
        // 2026-10-09: The pre-2026-10-09 policy: never yield, sleep 1 ms.
        let legacy = PollPause::parse(Some("0:1000")).unwrap();
        assert_eq!(
            legacy.action(Duration::ZERO),
            PauseAction::Sleep(Duration::from_millis(1))
        );
    }

    #[test]
    fn poll_pause_refuses_malformed_values() {
        for bad in [
            "", "200", "200:", ":50", "a:b", "200:0", "-1:50", "200:50:1",
        ] {
            assert!(PollPause::parse(Some(bad)).is_err(), "{bad:?} accepted");
        }
    }

    #[test]
    fn completion_and_driver_errors_do_not_keep_polling() {
        assert!(
            poll_completion(
                Duration::ZERO,
                || Duration::ZERO,
                || Ok(true),
                || panic!("already complete")
            )
            .is_ok()
        );
        let err = poll_completion(
            Duration::from_secs(30),
            || Duration::ZERO,
            || anyhow::bail!("peer lost"),
            || panic!("already failed"),
        )
        .unwrap_err();
        assert!(err.to_string().contains("peer lost"));
    }
}
