// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: The pure core of the `glm5next_comm_corun_probe` example: per-arm order
//! statistics and the co-run verdict. No I/O; `main.rs` measures and prints.
//!
//! A trial enqueues a chain of identical load launches on the main stream and, once the first
//! launch has finished, one grouped NCCL send/recv on the comm stream. `comm_us` is the comm
//! stream's span from that submission to the transfer's end, `tail_us` the span from the same
//! submission to the end of the load chain. A transfer that cannot get a CTA onto an SM until
//! the load drains ends at or after the chain (`comm_us / tail_us >= LOCKOUT_RATIO`); one that
//! starts promptly ends within the first part of the tail (`<= PASS_RATIO`).
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants: every statistic is over a non-empty sample; the verdict is `Inconclusive` when
//! the load tail is too short to tell a prompt transfer from a locked-out one.

use anyhow::{Result, ensure};

/// 2026-10-15: A transfer ending at or past 90% of the remaining load span waited for the
/// load: the last wave of a multi-wave chain is under 10% of it at the chain lengths probed.
pub const LOCKOUT_RATIO: f64 = 0.9;
/// 2026-10-15: A transfer ending within the first half of the remaining load span started
/// while the load still held the SMs, which is what an overlapped SP pipeline needs.
pub const PASS_RATIO: f64 = 0.5;
/// 2026-10-15: The remaining load span must be at least this many solo transfers long, or a
/// prompt transfer and a locked-out one end at nearly the same time.
pub const MIN_TAIL_OVER_COMM: f64 = 3.0;

/// 2026-10-15: Median, 10th and 90th percentile and count of one sample.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stats {
    pub median: f64,
    pub p10: f64,
    pub p90: f64,
    pub n: usize,
}

/// 2026-10-15: Order statistics of `v` (nearest rank on the sorted sample).
pub fn stats(v: &[f64]) -> Result<Stats> {
    ensure!(!v.is_empty(), "stats of an empty sample");
    ensure!(v.iter().all(|x| x.is_finite()), "non-finite sample value");
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let at = |q: f64| s[((s.len() - 1) as f64 * q).round() as usize];
    Ok(Stats {
        median: at(0.5),
        p10: at(0.1),
        p90: at(0.9),
        n: s.len(),
    })
}

/// 2026-10-15: One co-run trial (microseconds).
#[derive(Clone, Copy, Debug)]
pub struct Corun {
    /// 2026-10-15: Comm submission to the end of the transfer.
    pub comm_us: f64,
    /// 2026-10-15: Comm submission to the end of the load chain.
    pub tail_us: f64,
    /// 2026-10-15: The whole load chain.
    pub load_us: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// 2026-10-15: The transfer ends early in the load tail: it starts under the load.
    Pass,
    /// 2026-10-15: Between the two bounds: it starts late but before the chain drains.
    Delayed,
    /// 2026-10-15: The transfer ends with or after the load chain: no SM until it drains.
    LockOut,
    /// 2026-10-15: The tail is shorter than `MIN_TAIL_OVER_COMM` solo transfers.
    Inconclusive,
}

/// 2026-10-15: One arm's summary.
#[derive(Clone, Copy, Debug)]
pub struct Report {
    pub comm_solo: Stats,
    pub load_solo: Stats,
    pub comm_corun: Stats,
    pub tail: Stats,
    pub load_corun: Stats,
    /// 2026-10-15: Per-trial `comm_us / tail_us`.
    pub ratio: Stats,
    pub verdict: Verdict,
}

/// 2026-10-15: The arm's verdict from its solo and co-run samples.
pub fn judge(comm_solo: &[f64], load_solo: &[f64], corun: &[Corun]) -> Result<Report> {
    let col = |f: fn(&Corun) -> f64| corun.iter().map(f).collect::<Vec<_>>();
    let ratios: Vec<f64> = corun.iter().map(|c| c.comm_us / c.tail_us).collect();
    let r = Report {
        comm_solo: stats(comm_solo)?,
        load_solo: stats(load_solo)?,
        comm_corun: stats(&col(|c| c.comm_us))?,
        tail: stats(&col(|c| c.tail_us))?,
        load_corun: stats(&col(|c| c.load_us))?,
        ratio: stats(&ratios)?,
        verdict: Verdict::Inconclusive,
    };
    let verdict = if r.tail.median < MIN_TAIL_OVER_COMM * r.comm_solo.median {
        Verdict::Inconclusive
    } else if r.ratio.median >= LOCKOUT_RATIO {
        Verdict::LockOut
    } else if r.ratio.median <= PASS_RATIO {
        Verdict::Pass
    } else {
        Verdict::Delayed
    };
    Ok(Report { verdict, ..r })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trials(comm: f64, tail: f64) -> Vec<Corun> {
        (0..9)
            .map(|i| Corun {
                comm_us: comm + i as f64,
                tail_us: tail + i as f64,
                load_us: tail + 1000.0,
            })
            .collect()
    }

    #[test]
    fn stats_are_order_statistics() {
        let s = stats(&[5.0, 1.0, 3.0, 2.0, 4.0]).unwrap();
        assert_eq!((s.median, s.p10, s.p90, s.n), (3.0, 1.0, 5.0, 5));
        assert!(stats(&[]).is_err());
        assert!(stats(&[1.0, f64::NAN]).is_err());
    }

    #[test]
    fn prompt_transfer_passes() {
        let r = judge(&[600.0; 5], &[20_000.0; 5], &trials(700.0, 15_000.0)).unwrap();
        assert_eq!(r.verdict, Verdict::Pass);
    }

    #[test]
    fn transfer_ending_with_the_load_is_locked_out() {
        let r = judge(&[600.0; 5], &[20_000.0; 5], &trials(15_100.0, 15_000.0)).unwrap();
        assert_eq!(r.verdict, Verdict::LockOut);
    }

    #[test]
    fn late_start_is_delayed() {
        let r = judge(&[600.0; 5], &[20_000.0; 5], &trials(10_000.0, 15_000.0)).unwrap();
        assert_eq!(r.verdict, Verdict::Delayed);
    }

    #[test]
    fn short_tail_is_inconclusive_even_when_ratio_is_high() {
        let r = judge(&[600.0; 5], &[2_000.0; 5], &trials(1_500.0, 1_600.0)).unwrap();
        assert_eq!(r.verdict, Verdict::Inconclusive);
    }

    #[test]
    fn empty_corun_sample_errors() {
        assert!(judge(&[600.0], &[20_000.0], &[]).is_err());
    }
}
