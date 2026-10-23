// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Energy of one channel (a source's rail) over one window.
//!
//! - A cumulative counter is differenced through [`CounterTracker`] (wrap and reset handled) and
//!   its mean power taken between the first and the last sample that saw the value change, so
//!   the estimate is not biased by the up-to-one-period wait before the first update and after
//!   the last (each edge error ≤ ½ period). The window's energy is that power × the window length.
//! - A power reading (instant, or the device's average when that is all it has) is integrated
//!   left-Riemann: each sample holds until the next one, the last until the window ends.
//!
//! The per-update power series is kept for the autocorrelation-aware statistics (`derive`).
//!
//! Owner: metrale-power.
//! Invariants: a window with fewer than two counter updates, or no power sample, has no energy
//! (`None`), never 0.

use alloc::vec::Vec;

use metrale_abstract_io::MonoNs;
use metrale_abstract_io::power::{Field, PowerSample, Rail, SourceKind};

use crate::counter::{Advance, CounterTracker};
use crate::trace::{PowerTrace, SourceTrace, Window, secs};

/// 2026-10-22: How a channel's energy is formed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Integral {
    /// 2026-10-22: A cumulative energy counter, differenced.
    Counter,
    /// 2026-10-22: An instant power reading, integrated.
    Instant,
    /// 2026-10-22: The device's own averaged power, integrated.
    Average,
}

/// 2026-10-22: One measurable (source, rail) pair of a trace.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Channel {
    /// 2026-10-22: Index into [`PowerTrace::sources`].
    pub source: usize,
    pub rail: Rail,
    pub integral: Integral,
}

impl Channel {
    /// 2026-10-22: The source kind of this channel in `t`.
    #[must_use]
    pub fn kind(&self, t: &PowerTrace) -> SourceKind {
        t.sources[self.source].caps.kind()
    }
}

/// 2026-10-22: Every channel of `t`: per source and rail, the counter when there is one, else the
/// instant reading, else the device average (one channel per rail; the order is the source's).
#[must_use]
pub fn channels(t: &PowerTrace) -> Vec<Channel> {
    let mut out = Vec::new();
    for (i, s) in t.sources.iter().enumerate() {
        let avail = s.caps.available();
        for rail in Rail::ALL {
            let integral = if avail.contains(&Field::Energy(rail)) {
                Integral::Counter
            } else if avail.contains(&Field::InstantPower(rail)) {
                Integral::Instant
            } else if avail.contains(&Field::AveragePower(rail)) {
                Integral::Average
            } else {
                continue;
            };
            out.push(Channel {
                source: i,
                rail,
                integral,
            });
        }
    }
    out
}

/// 2026-10-22: One channel over one window.
#[derive(Clone, Debug, PartialEq)]
pub struct WindowEnergy {
    /// 2026-10-22: Energy over the window, J.
    pub joules: f64,
    /// 2026-10-22: Window length, s.
    pub seconds: f64,
    /// 2026-10-22: Mean power, W.
    pub avg_w: f64,
    /// 2026-10-22: Samples inside the window.
    pub samples: u64,
    /// 2026-10-22: Distinct value updates inside the window.
    pub updates: u64,
    /// 2026-10-22: Counter wraps and resets seen inside the window.
    pub wraps: u64,
    pub resets: u64,
    /// 2026-10-22: Power per update interval (counter) or per sample (reading), W, in time order.
    pub series_w: Vec<f64>,
    /// 2026-10-22: Each `series_w` entry's interval length, s (so `Σ w·dt / Σ dt` is `avg_w`
    /// for a counter, and a resample keeps energy and time together).
    pub series_dt: Vec<f64>,
    /// 2026-10-22: Each `series_w` entry's end instant.
    pub series_at: Vec<MonoNs>,
}

fn inside(s: &SourceTrace, from: MonoNs, to: MonoNs) -> impl Iterator<Item = &PowerSample> {
    s.samples.iter().filter(move |x| x.at >= from && x.at <= to)
}

/// 2026-10-22: `c` over `[from, to]` of trace `t`.
#[must_use]
pub fn integrate(t: &PowerTrace, c: Channel, from: MonoNs, to: MonoNs) -> Option<WindowEnergy> {
    let s = &t.sources[c.source];
    match c.integral {
        Integral::Counter => counter(s, c.rail, from, to),
        Integral::Instant | Integral::Average => reading(s, c, from, to),
    }
}

/// 2026-10-22: [`integrate`] over a [`Window`].
#[must_use]
pub fn integrate_window(t: &PowerTrace, c: Channel, w: &Window) -> Option<WindowEnergy> {
    integrate(t, c, w.from, w.to)
}

fn counter(s: &SourceTrace, rail: Rail, from: MonoNs, to: MonoNs) -> Option<WindowEnergy> {
    let unit = s.caps.counter(rail)?.unit_j;
    let mut tracker = CounterTracker::default();
    let (mut samples, mut updates, mut wraps, mut resets) = (0u64, 0u64, 0u64, 0u64);
    // 2026-10-22: The span runs from the first observed update (the value changed between two
    // samples) to the last; the energy before the first update belongs to an earlier interval.
    let mut anchor: Option<MonoNs> = None;
    let mut last: Option<MonoNs> = None;
    let mut counts = 0u64;
    let (mut series, mut series_dt, mut series_at) = (Vec::new(), Vec::new(), Vec::new());
    for x in inside(s, from, to) {
        samples += 1;
        let Some(raw) = x.energy.get(&rail) else {
            continue;
        };
        let adv = tracker.advance(raw.0);
        match adv {
            Advance::First | Advance::Delta(0) => continue,
            Advance::Wrapped(_) => wraps += 1,
            Advance::Reset(_) => resets += 1,
            Advance::Delta(_) => {}
        }
        let Some(prev) = last else {
            anchor = Some(x.at);
            last = Some(x.at);
            continue;
        };
        let d = adv.millijoules();
        let dt = secs(prev, x.at);
        if dt > 0.0 {
            series.push(d as f64 * unit / dt);
            series_dt.push(dt);
            series_at.push(x.at);
        }
        updates += 1;
        counts += d;
        last = Some(x.at);
    }
    let span = secs(anchor?, last?);
    if updates < 2 || span <= 0.0 {
        return None;
    }
    let seconds = secs(from, to);
    let avg_w = counts as f64 * unit / span;
    Some(WindowEnergy {
        joules: avg_w * seconds,
        seconds,
        avg_w,
        samples,
        updates,
        wraps,
        resets,
        series_w: series,
        series_dt,
        series_at,
    })
}

fn power_of(x: &PowerSample, c: Channel) -> Option<f64> {
    let r = x.rails.get(&c.rail)?;
    match c.integral {
        Integral::Instant => r.instant_w,
        _ => r.average_w,
    }
}

fn reading(s: &SourceTrace, c: Channel, from: MonoNs, to: MonoNs) -> Option<WindowEnergy> {
    let pts: Vec<(MonoNs, f64)> = inside(s, from, to)
        .filter_map(|x| power_of(x, c).map(|p| (x.at, p)))
        .collect();
    if pts.is_empty() {
        return None;
    }
    let mut joules = 0.0;
    let mut updates = 0u64;
    for (i, (at, p)) in pts.iter().enumerate() {
        let next = pts.get(i + 1).map_or(to, |n| n.0);
        joules += p * secs(*at, next);
        if i == 0 || pts[i - 1].1.to_bits() != p.to_bits() {
            updates += 1;
        }
    }
    let seconds = secs(from, to);
    Some(WindowEnergy {
        joules,
        seconds,
        avg_w: if seconds > 0.0 { joules / seconds } else { 0.0 },
        samples: pts.len() as u64,
        updates,
        wraps: 0,
        resets: 0,
        series_w: pts.iter().map(|p| p.1).collect(),
        series_dt: pts
            .iter()
            .enumerate()
            .map(|(i, p)| secs(p.0, pts.get(i + 1).map_or(to, |n| n.0)))
            .collect(),
        series_at: pts.iter().map(|p| p.0).collect(),
    })
}

/// 2026-10-22: Mean of `f` over the samples of source `src` inside `w` that carry it.
#[must_use]
pub fn mean_of(
    t: &PowerTrace,
    src: usize,
    w: &Window,
    f: impl Fn(&PowerSample) -> Option<f64>,
) -> Option<f64> {
    let v: Vec<f64> = inside(&t.sources[src], w.from, w.to)
        .filter_map(f)
        .collect();
    (!v.is_empty()).then(|| v.iter().sum::<f64>() / v.len() as f64)
}

/// 2026-10-22: Max of `f` over the samples of source `src` inside `w` that carry it.
#[must_use]
pub fn max_of(
    t: &PowerTrace,
    src: usize,
    w: &Window,
    f: impl Fn(&PowerSample) -> Option<f64>,
) -> Option<f64> {
    inside(&t.sources[src], w.from, w.to)
        .filter_map(f)
        .fold(None, |m, v| Some(m.map_or(v, |m: f64| m.max(v))))
}

/// 2026-10-22: Fraction of the samples of source `src` inside `w` that carry a throttle reason;
/// `None` when the source reports no throttle field.
#[must_use]
pub fn throttle_frac(t: &PowerTrace, src: usize, w: &Window) -> Option<f64> {
    let v: Vec<bool> = inside(&t.sources[src], w.from, w.to)
        .filter_map(|x| x.throttle.map(|m| m != 0))
        .collect();
    (!v.is_empty()).then(|| v.iter().filter(|b| **b).count() as f64 / v.len() as f64)
}
