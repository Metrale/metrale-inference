// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Steady-state attribution of one repeat (DESIGN §3.1): a load window between two
//! interleaved idle baselines A and B, per channel:
//!
//! - `static_w = (W_A + W_B) / 2` — both sides are required, so a linear drift over the repeat
//!   cancels (the one-sided baseline of the 2026-10-22 probe does not);
//! - `J_total = E_load / launches`, `J_dyn = (E_load − static_w · T_load) / launches`,
//!   `W = E_load / T_load`.
//!
//! Refusals, each a coded error naming the rule: a window below the policy's length, update or
//! launch floor (`EnergyWindowRefused`); a missing baseline (`EnergyBaselineMissing`); baselines
//! drifting apart by more than the policy allows (`EnergyBaselineDrift`); a mean SM clock below
//! the floor (`EnergyClockFloor`); too many throttled samples (`EnergyThrottled`).
//!
//! Owner: metrale-power.
//! Invariants: baselines pair by adjacency on the same box: A is the last idle window ending at
//! or before the load starts with no load between, B the first idle window starting at or after
//! it ends with no load between (so one idle window serves as B of one repeat and A of the next).

use alloc::format;
use alloc::string::ToString;
use alloc::vec::Vec;

use metrale_abstract_io::power::Field;
use metrale_error::{ErrorCode, error};

use crate::Result;
use crate::integrate::{
    Channel, Integral, WindowEnergy, integrate_window, max_of, mean_of, throttle_frac,
};
use crate::policy::WindowPolicy;
use crate::trace::{Phase, PowerTrace, Window};

/// 2026-10-22: Fewest distinct values a power reading (not a counter) must show inside a load
/// window: the policy's `min_updates` is a counter rule (DESIGN §1.3, 30 updates of a 100 ms
/// counter); a 250 ms sensor shows a quarter as many over the same window.
pub const MIN_READING_UPDATES: u64 = 10;

/// 2026-10-22: One load window and its two baselines, as indices into `PowerTrace::windows`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Repeat {
    pub load: usize,
    pub idle_a: Option<usize>,
    pub idle_b: Option<usize>,
}

/// 2026-10-22: The repeats of `t`, in window order.
#[must_use]
pub fn repeats(t: &PowerTrace) -> Vec<Repeat> {
    let w = &t.windows;
    let mut order: Vec<usize> = (0..w.len()).collect();
    order.sort_by_key(|&i| (w[i].box_id, w[i].from.0, w[i].to.0));
    let mut out = Vec::new();
    for (pos, &i) in order.iter().enumerate() {
        if w[i].phase != Phase::Load {
            continue;
        }
        let same_box = |&&j: &&usize| w[j].box_id == w[i].box_id;
        let idle_a = order[..pos]
            .iter()
            .rev()
            .filter(same_box)
            .take_while(|&&j| w[j].phase == Phase::Idle)
            .find(|&&j| w[j].to <= w[i].from)
            .copied();
        let idle_b = order[pos + 1..]
            .iter()
            .filter(same_box)
            .take_while(|&&j| w[j].phase == Phase::Idle)
            .find(|&&j| w[j].from >= w[i].to)
            .copied();
        out.push(Repeat {
            load: i,
            idle_a,
            idle_b,
        });
    }
    out
}

/// 2026-10-22: One channel over one repeat.
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelRepeat {
    pub channel: Channel,
    pub box_id: u32,
    pub repeat: u32,
    pub idle_a: Option<WindowEnergy>,
    pub idle_b: Option<WindowEnergy>,
    pub load: WindowEnergy,
    pub launches: Option<u64>,
}

impl ChannelRepeat {
    /// 2026-10-22: The interleaved static power; both baselines are required.
    pub fn static_w(&self) -> Result<f64> {
        let side = |w: &Option<WindowEnergy>, which: &'static str| {
            w.as_ref()
                .map(|w| w.avg_w)
                .ok_or_else(|| error!(ErrorCode::EnergyBaselineMissing, self.repeat, which))
        };
        Ok((side(&self.idle_a, "A")? + side(&self.idle_b, "B")?) / 2.0)
    }

    /// 2026-10-22: |A − B| / mean(A, B), percent.
    pub fn drift_pct(&self) -> Result<f64> {
        let s = self.static_w()?;
        let (a, b) = (
            self.idle_a.as_ref().map_or(0.0, |w| w.avg_w),
            self.idle_b.as_ref().map_or(0.0, |w| w.avg_w),
        );
        Ok(if s > 0.0 {
            (a - b).abs() / s * 100.0
        } else {
            0.0
        })
    }

    fn launches(&self) -> Result<f64> {
        self.launches
            .filter(|n| *n > 0)
            .map(|n| n as f64)
            .ok_or_else(|| {
                error!(
                    ErrorCode::EnergyWindowRefused,
                    format!("load {}", self.repeat),
                    "its launch count was not recorded".to_string()
                )
            })
    }

    /// 2026-10-22: J per launch, total.
    pub fn j_total(&self) -> Result<f64> {
        Ok(self.load.joules / self.launches()?)
    }

    /// 2026-10-22: J per launch above the interleaved static power.
    pub fn j_dyn(&self) -> Result<f64> {
        Ok((self.load.joules - self.static_w()? * self.load.seconds) / self.launches()?)
    }
}

fn refuse(w: &Window, why: alloc::string::String) -> metrale_error::Error {
    error!(
        ErrorCode::EnergyWindowRefused,
        format!("{} box {} repeat {}", w.phase.as_str(), w.box_id, w.repeat),
        why
    )
}

/// 2026-10-22: `c` over repeat `r` of `t`, with the window floors of `p` applied (the baseline
/// pairing is reported, not refused, here: [`ChannelRepeat::static_w`] refuses a missing side).
pub fn measure(t: &PowerTrace, p: &WindowPolicy, c: Channel, r: &Repeat) -> Result<ChannelRepeat> {
    let lw = &t.windows[r.load];
    if lw.seconds() < p.min_load_s {
        return Err(refuse(
            lw,
            format!("{:.3} s is shorter than {} s", lw.seconds(), p.min_load_s),
        ));
    }
    if let Some(n) = lw.launches.filter(|n| *n < p.min_launches) {
        return Err(refuse(
            lw,
            format!("{n} launches is fewer than {}", p.min_launches),
        ));
    }
    let load = integrate_window(t, c, lw)
        .ok_or_else(|| refuse(lw, "the channel has no energy in it".to_string()))?;
    let need = match c.integral {
        Integral::Counter => p.min_updates,
        Integral::Instant | Integral::Average => MIN_READING_UPDATES,
    };
    if load.updates < need {
        return Err(refuse(
            lw,
            format!("{} updates is fewer than {need}", load.updates),
        ));
    }
    let idle = |i: Option<usize>| -> Result<Option<WindowEnergy>> {
        let Some(i) = i else { return Ok(None) };
        let w = &t.windows[i];
        if w.seconds() < p.min_idle_s {
            return Err(refuse(
                w,
                format!("{:.3} s is shorter than {} s", w.seconds(), p.min_idle_s),
            ));
        }
        Ok(integrate_window(t, c, w))
    };
    Ok(ChannelRepeat {
        channel: c,
        box_id: lw.box_id,
        repeat: lw.repeat,
        idle_a: idle(r.idle_a)?,
        idle_b: idle(r.idle_b)?,
        load,
        launches: lw.launches,
    })
}

/// 2026-10-22: The run's fingerprint over one load window: mean SM clock, max temperature, the
/// throttled share (each from the first source that reports it; `None` when none does).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LoadFacts {
    pub sm_mhz: Option<f64>,
    pub temp_c: Option<f64>,
    pub throttle_pct: Option<f64>,
}

/// 2026-10-22: [`LoadFacts`] of load window `w`, refused by `p`'s clock and throttle rules.
pub fn load_facts(t: &PowerTrace, p: &WindowPolicy, w: &Window) -> Result<LoadFacts> {
    let first = |f: Field| {
        t.sources
            .iter()
            .position(|s| s.caps.available().contains(&f))
    };
    let sm =
        first(Field::SmClock).and_then(|i| mean_of(t, i, w, |x| x.clocks.sm_mhz.map(f64::from)));
    let temp = first(Field::GpuTemp).and_then(|i| max_of(t, i, w, |x| x.temps.gpu_c));
    let thr = first(Field::Throttle)
        .and_then(|i| throttle_frac(t, i, w))
        .map(|f| f * 100.0);
    if let Some(mhz) = sm.filter(|m| *m < p.min_sm_mhz) {
        return Err(error!(
            ErrorCode::EnergyClockFloor,
            libm::round(mhz),
            p.min_sm_mhz
        ));
    }
    if let Some(pct) = thr.filter(|x| *x > p.max_throttle_pct) {
        return Err(error!(ErrorCode::EnergyThrottled, pct, p.max_throttle_pct));
    }
    Ok(LoadFacts {
        sm_mhz: sm,
        temp_c: temp,
        throttle_pct: thr,
    })
}
