// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Cost per million tokens, derived and never stored (owner decision 5: electricity
//! AND amortized hardware; every input explicit, PCND: no default price, lifetime or utilization):
//!
//! - electricity `$/Mtok = J/tok × $/kWh ÷ 3.6` (1 Mtok × 1 J/tok = 1 MJ = 1/3.6 kWh);
//! - hardware `$/Mtok = price ÷ (lifetime_h × 3600 × utilization) ÷ tok/s × 10⁶`.
//!
//! Each carries two intervals (owner addendum item 5): the first-order delta method and a
//! parametric bootstrap from the inputs' sampling distributions; a disagreement between them is
//! flagged on the result.
//!
//! Owner: metrale-power.
//! Invariants: no input has a default; a missing input is `CostInputMissing`.

use alloc::string::ToString;

use metrale_error::{ErrorCode, error};
use metrale_stats::dist::t_ppf;
use metrale_stats::interval::{CiMethod, Interval, Summary};
use metrale_stats::rng::SplitMix64;

use crate::Result;

/// 2026-10-22: The explicit inputs; every one is required.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CostInputs {
    pub electricity_usd_per_kwh: Option<f64>,
    pub hardware_usd: Option<f64>,
    pub lifetime_hours: Option<f64>,
    pub utilization: Option<f64>,
}

/// 2026-10-22: One cost line.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CostLine {
    pub usd_per_mtok: f64,
    pub delta: Interval,
    pub bootstrap: Interval,
    /// 2026-10-22: `false` when the two intervals disagree beyond the tolerance.
    pub methods_agree: bool,
}

/// 2026-10-22: Electricity, hardware and their sum.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cost {
    pub electricity: CostLine,
    pub hardware: CostLine,
    pub total_usd_per_mtok: f64,
}

/// 2026-10-22: Relative tolerance between the delta-method and bootstrap interval endpoints.
pub const AGREE_TOL: f64 = 0.25;

fn need(v: Option<f64>, name: &'static str) -> Result<f64> {
    let v = v.ok_or_else(|| error!(ErrorCode::CostInputMissing, name))?;
    if !(v.is_finite() && v > 0.0) {
        return Err(error!(
            ErrorCode::CostInputInvalid,
            name,
            "must be positive".to_string()
        ));
    }
    Ok(v)
}

/// 2026-10-22: J per token from J per launch at `rows` tokens per launch.
#[must_use]
pub fn per_token(j_per_launch: Summary, rows: u32) -> Summary {
    let r = f64::from(rows.max(1));
    Summary {
        mean: j_per_launch.mean / r,
        se: j_per_launch.se / r,
        df: j_per_launch.df,
    }
}

fn draw(s: &Summary, rng: &mut SplitMix64) -> f64 {
    s.mean + s.se * rng.normal()
}

fn sym(est: f64, se: f64, df: f64, level: f64, method: CiMethod) -> Interval {
    let t = t_ppf(0.5 + level / 2.0, df.max(1.0));
    Interval {
        lo: est - t * se,
        hi: est + t * se,
        level,
        method,
    }
}

/// 2026-10-22: Cost of serving at `j_per_tok` (J/token) and `tok_per_s`, with intervals at
/// `level` (bootstrap: `reps` draws from `seed`).
pub fn cost(
    j_per_tok: Summary,
    tok_per_s: Summary,
    inputs: &CostInputs,
    level: f64,
    reps: usize,
    seed: u64,
) -> Result<Cost> {
    let price = need(inputs.electricity_usd_per_kwh, "--electricity-usd-per-kwh")?;
    let hw = need(inputs.hardware_usd, "--hardware-usd")?;
    let life = need(inputs.lifetime_hours, "--lifetime-hours")?;
    let util = need(inputs.utilization, "--utilization")?;
    if util > 1.0 {
        return Err(error!(
            ErrorCode::CostInputInvalid,
            "--utilization",
            "must be at most 1".to_string()
        ));
    }
    if !(j_per_tok.mean > 0.0 && tok_per_s.mean > 0.0) {
        return Err(error!(
            ErrorCode::CostInputInvalid,
            "J/token and tok/s",
            "must be positive".to_string()
        ));
    }
    let k_e = price / 3.6;
    let usd_per_s = hw / (life * 3600.0 * util);
    let e = j_per_tok.mean * k_e;
    let h = usd_per_s / tok_per_s.mean * 1e6;
    let e_delta = sym(e, j_per_tok.se * k_e, j_per_tok.df, level, CiMethod::Delta);
    // 2026-10-22: d(c/x)/dx = -c/x²: the first-order delta method for a reciprocal.
    let h_se = usd_per_s * 1e6 * tok_per_s.se / (tok_per_s.mean * tok_per_s.mean);
    let h_delta = sym(h, h_se, tok_per_s.df, level, CiMethod::Delta);
    let mut rng = SplitMix64::new(seed);
    let mut be = alloc::vec::Vec::with_capacity(reps);
    let mut bh = alloc::vec::Vec::with_capacity(reps);
    for _ in 0..reps {
        be.push(draw(&j_per_tok, &mut rng) * k_e);
        let x = draw(&tok_per_s, &mut rng);
        if x > 0.0 {
            bh.push(usd_per_s / x * 1e6);
        }
    }
    let e_boot = metrale_stats::bootstrap::percentile_interval(&be, level)?;
    let h_boot = metrale_stats::bootstrap::percentile_interval(&bh, level)?;
    let line = |v: f64, d: Interval, b: Interval| CostLine {
        usd_per_mtok: v,
        delta: d,
        bootstrap: b,
        methods_agree: metrale_stats::ratio::agree(&d, &b, AGREE_TOL),
    };
    Ok(Cost {
        electricity: line(e, e_delta, e_boot),
        hardware: line(h, h_delta, h_boot),
        total_usd_per_mtok: e + h,
    })
}
