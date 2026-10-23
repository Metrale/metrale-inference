// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Cross-instrument agreement (owner addendum 2026-10-22, item 6): the sources
//! disagree (GB10: counter vs sensor by up to 31 %), so every pair is reported, never hidden:
//!
//! - `sensor_delta_pct["a~b"]`: the signed energy delta of `b` against `a` over the pooled load
//!   windows, flagged above the policy's limit;
//! - `energy_agreement["a~b"]`: an errors-in-variables (Deming, λ = 1: orthogonal) fit of `b`'s
//!   mean power on `a`'s and the Bland-Altman bias and 95 % limits of agreement, over every window
//!   (idle and load) both measured. OLS would attenuate the slope, since both axes are noisy.
//!
//! Owner: metrale-power.
//! Invariants: pairs are ordered by channel order (headline first), keyed `"<a>~<b>"`.

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use crate::Result;
use crate::derive::key;
use crate::integrate::Channel;
use crate::policy::WindowPolicy;
use crate::row::{Agreement, Ci};
use crate::steady::ChannelRepeat;
use crate::trace::PowerTrace;

type Measured = [(Channel, Vec<ChannelRepeat>)];

fn pairs(m: &Measured) -> impl Iterator<Item = (usize, usize)> + '_ {
    (0..m.len()).flat_map(move |i| (i + 1..m.len()).map(move |j| (i, j)))
}

/// 2026-10-22: Signed load-energy deltas per pair, and the flags of those beyond the limit.
pub(crate) fn sensor_deltas(
    t: &PowerTrace,
    m: &Measured,
    p: &WindowPolicy,
) -> (BTreeMap<String, f64>, Vec<String>) {
    let mut out = BTreeMap::new();
    let mut flags = Vec::new();
    for (i, j) in pairs(m) {
        let e = |k: usize| m[k].1.iter().map(|r| r.load.joules).sum::<f64>();
        let (a, b) = (e(i), e(j));
        if a <= 0.0 {
            continue;
        }
        let name = format!("{}~{}", key(t, &m[i].0), key(t, &m[j].0));
        let pct = (b - a) / a * 100.0;
        if pct.abs() > p.sensor_flag_pct {
            flags.push(format!("sensor_delta>{}%:{name}", p.sensor_flag_pct));
        }
        out.insert(name, pct);
    }
    (out, flags)
}

/// 2026-10-22: The windows' mean power of one channel, in (repeat, A, load, B) order.
fn powers(reps: &[ChannelRepeat]) -> Vec<Option<f64>> {
    reps.iter()
        .flat_map(|r| {
            [
                r.idle_a.as_ref().map(|w| w.avg_w),
                Some(r.load.avg_w),
                r.idle_b.as_ref().map(|w| w.avg_w),
            ]
        })
        .collect()
}

/// 2026-10-22: Deming and Bland-Altman per pair with at least three shared windows.
pub(crate) fn agreements(t: &PowerTrace, m: &Measured) -> Result<BTreeMap<String, Agreement>> {
    let mut out = BTreeMap::new();
    for (i, j) in pairs(m) {
        let (x, y): (Vec<f64>, Vec<f64>) = powers(&m[i].1)
            .into_iter()
            .zip(powers(&m[j].1))
            .filter_map(|(a, b)| Some((a?, b?)))
            .unzip();
        if x.len() < 3 {
            continue;
        }
        let d = metrale_stats::agreement::deming(&x, &y, 1.0)?;
        let ba = metrale_stats::agreement::bland_altman(&x, &y, 0.95)?;
        out.insert(
            format!("{}~{}", key(t, &m[i].0), key(t, &m[j].0)),
            Agreement {
                deming_slope: d.slope,
                deming_intercept_w: d.intercept,
                ba_bias_w: ba.bias,
                ba_loa_w: Ci {
                    lo: ba.loa_lo,
                    hi: ba.loa_hi,
                },
                n: x.len() as u64,
            },
        );
    }
    Ok(out)
}
