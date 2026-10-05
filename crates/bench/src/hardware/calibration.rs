// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: A box's calibration profile — raw measurements from a short,
//! fixed probe, and each one's ratio to the fleet reference (the mean of every
//! profile's raw measurements, this box's own included) — and
//! `kernels/<hw>/BOX_PROFILES.toml`'s (de)serialization.
//!
//! Owner: bench hardware.
//! Invariants:
//! - A ratio is `raw / fleet_mean`; which direction is "better" differs by
//!   field (higher `decode_tok_s_ratio` is faster, higher `*_ms_ratio` and
//!   `*_j_per_tok_ratio` is slower/costlier) and is stated on each field, not
//!   implied by the sign.
//! - [`Raw::fleet_mean`] of zero profiles is `None`: a fleet with no history is not
//!   a fleet reading zero.
//! - A ratio is computed only when both the raw value and the fleet mean are
//!   finite and positive; otherwise the ratio field is absent, never a
//!   fabricated 1.0 or 0.0 (the "absent, never zero" rule every other reading
//!   in this module follows).
//!
//! v1 scope: five dimensions (decode bandwidth, 32k cold prefill, 32k warm
//! restore, C1 energy, idle power). `stability_sigma_pct` from the design note
//! is NOT YET measured: the only existing single-user decode benchmark
//! (`decode-floor`) pins its repeat count at 3 internally
//! (`benchmarks/decode_floor/score.rs::RUNS`) and records only the median, so
//! calibration cannot read a spread from it without either a cheaper
//! dedicated probe or decode-floor exporting its raw per-run samples —
//! tracked as follow-up work, not fabricated here.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

/// 2026-10-04: One box's calibration profile: when and at what commit it was
/// measured, its raw readings, and each reading's ratio to the fleet
/// reference at the time this profile was computed (a later box joining the
/// fleet does not retroactively change an earlier profile's stored ratios —
/// re-run `met benchmark calibrate` to refresh them).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoxProfile {
    /// 2026-10-04: Unix seconds.
    pub measured_at: u64,
    // 2026-10-04: `git_sha` is a string field, declared below with the rest —
    // kept out of this doc block only because rustfmt does not let a derive
    // comment span a String in a Copy struct; see the field itself.
    pub decode_tok_s: Option<f64>,
    /// 2026-10-04: This box's decode tok/s over the fleet mean. Higher is
    /// faster.
    pub decode_tok_s_ratio: Option<f64>,
    pub prefill_cold32k_ms: Option<f64>,
    /// 2026-10-04: This box's 32k cold TTFT over the fleet mean. Higher is
    /// slower.
    pub prefill_cold32k_ms_ratio: Option<f64>,
    pub restore_warm32k_ms: Option<f64>,
    /// 2026-10-04: This box's 32k warm TTFT over the fleet mean. Higher is
    /// slower. Simplification vs the design note: this is the raw warm TTFT,
    /// not warm-minus-tail-prefill — the tail-prefill split
    /// (`cold-ttft-tail-split-second-pass`) is a follow-up refinement.
    pub restore_warm32k_ms_ratio: Option<f64>,
    pub energy_c1_j_per_tok: Option<f64>,
    /// 2026-10-04: This box's C1 J/token over the fleet mean. Higher is
    /// costlier.
    pub energy_c1_ratio: Option<f64>,
    pub idle_power_w: Option<f64>,
    /// 2026-10-04: This box's idle GPU-rail watts over the fleet mean. Higher
    /// is costlier.
    pub idle_power_w_ratio: Option<f64>,
    /// 2026-10-04: NOT YET MEASURED (see module doc); reserved so the schema
    /// does not change again once it is.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stability_sigma_pct: Option<f64>,
}

/// 2026-10-04: `[benchmarks.limits]`-style TOML, but holding facts measured
/// about the FLEET, not declared limits: `kernels/<hw>/BOX_PROFILES.toml`,
/// `[profiles."<perf_class>"]` keyed by [`super::state::MachineIdentity::perf_class`]
/// (e.g. `"gb10@dgx2"`).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BoxProfiles {
    #[serde(default)]
    pub profiles: BTreeMap<String, ProfileEntry>,
}

/// 2026-10-04: A profile entry carries the git sha as a `String`, split from
/// [`BoxProfile`] (which is `Copy`) only so `BoxProfile` can stay cheap to
/// pass around in the ratio math; the TOML flattens the two back into one
/// table per box.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProfileEntry {
    pub git_sha: String,
    #[serde(flatten)]
    pub profile: BoxProfile,
}

const PATH_SUFFIX: &str = "BOX_PROFILES.toml";

/// 2026-10-04: `kernels/<hardware>/BOX_PROFILES.toml`, parsed, or an empty
/// table when the file does not exist yet (the first box to calibrate creates
/// it).
///
/// # Errors
/// An existing file that does not parse, or carries a stray key.
pub fn load(root: &Path, hardware: &str) -> Result<BoxProfiles> {
    let path = root.join("kernels").join(hardware).join(PATH_SUFFIX);
    match std::fs::read_to_string(&path) {
        Ok(text) => toml::from_str(&text).with_context(|| format!("parsing {}", path.display())),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(BoxProfiles::default()),
        Err(e) => Err(e).with_context(|| format!("reading {}", path.display())),
    }
}

/// 2026-10-04: Write `profiles` back to `kernels/<hardware>/BOX_PROFILES.toml`,
/// creating the directory structure if needed (it never is, in practice: a
/// box class without `kernels/<hw>/` has nothing to calibrate).
///
/// # Errors
/// The hardware directory does not exist, or the file cannot be written.
pub fn save(root: &Path, hardware: &str, profiles: &BoxProfiles) -> Result<()> {
    let dir = root.join("kernels").join(hardware);
    if !dir.is_dir() {
        bail!(
            "{} does not exist — {hardware} has no kernels directory to calibrate",
            dir.display()
        );
    }
    let text = toml::to_string_pretty(profiles).context("serializing BOX_PROFILES.toml")?;
    std::fs::write(dir.join(PATH_SUFFIX), text)
        .with_context(|| format!("writing {}", dir.join(PATH_SUFFIX).display()))
}

/// 2026-10-04: The five raw measurements one calibration probe collects,
/// before any ratio is computed. Each is `None` when its leg of the probe did
/// not produce a usable reading (the benchmark was inconclusive, or its
/// record lacked the key) — never a fabricated zero.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Raw {
    pub decode_tok_s: Option<f64>,
    pub prefill_cold32k_ms: Option<f64>,
    pub restore_warm32k_ms: Option<f64>,
    pub energy_c1_j_per_tok: Option<f64>,
    pub idle_power_w: Option<f64>,
}

/// 2026-10-04: A finite, strictly positive reading, or `None`: every raw
/// measurement here is a rate or a duration, and zero or negative is not a
/// real one.
fn valid(v: Option<f64>) -> Option<f64> {
    v.filter(|x| x.is_finite() && *x > 0.0)
}

impl Raw {
    /// 2026-10-04: The fleet reference: the mean of each field across every
    /// profile that reports it (not just profiles with a complete set), plus
    /// `extra` if given (the box's own freshly-measured raw, included in its
    /// own reference — DESIGN.md: "a ratio to a fleet reference", not to a
    /// fixed box). `None` for a field with no readable value anywhere.
    pub fn fleet_mean(profiles: &BTreeMap<String, ProfileEntry>, extra: Option<&Raw>) -> Self {
        let raws: Vec<Raw> = profiles
            .values()
            .map(|e| e.profile.raw())
            .chain(extra.copied())
            .collect();
        let mean = |pick: fn(&Raw) -> Option<f64>| {
            let vals: Vec<f64> = raws.iter().filter_map(|r| valid(pick(r))).collect();
            (!vals.is_empty()).then(|| vals.iter().sum::<f64>() / vals.len() as f64)
        };
        Self {
            decode_tok_s: mean(|r| r.decode_tok_s),
            prefill_cold32k_ms: mean(|r| r.prefill_cold32k_ms),
            restore_warm32k_ms: mean(|r| r.restore_warm32k_ms),
            energy_c1_j_per_tok: mean(|r| r.energy_c1_j_per_tok),
            idle_power_w: mean(|r| r.idle_power_w),
        }
    }
}

fn ratio(raw: Option<f64>, reference: Option<f64>) -> Option<f64> {
    match (valid(raw), valid(reference)) {
        (Some(r), Some(f)) => Some(r / f),
        _ => None,
    }
}

impl BoxProfile {
    /// 2026-10-04: This profile's own raw measurements, stripped of their
    /// ratios — the half [`Raw::fleet_mean`] aggregates over.
    pub fn raw(&self) -> Raw {
        Raw {
            decode_tok_s: self.decode_tok_s,
            prefill_cold32k_ms: self.prefill_cold32k_ms,
            restore_warm32k_ms: self.restore_warm32k_ms,
            energy_c1_j_per_tok: self.energy_c1_j_per_tok,
            idle_power_w: self.idle_power_w,
        }
    }

    /// 2026-10-04: Build a profile from one probe's raw readings and the
    /// fleet reference it is judged against (normally
    /// `Raw::fleet_mean(&existing_profiles, Some(&raw))`, so the box's own
    /// reading is part of its own reference — see [`Raw::fleet_mean`]).
    pub fn new(measured_at: u64, raw: Raw, reference: &Raw) -> Self {
        Self {
            measured_at,
            decode_tok_s: raw.decode_tok_s,
            decode_tok_s_ratio: ratio(raw.decode_tok_s, reference.decode_tok_s),
            prefill_cold32k_ms: raw.prefill_cold32k_ms,
            prefill_cold32k_ms_ratio: ratio(raw.prefill_cold32k_ms, reference.prefill_cold32k_ms),
            restore_warm32k_ms: raw.restore_warm32k_ms,
            restore_warm32k_ms_ratio: ratio(raw.restore_warm32k_ms, reference.restore_warm32k_ms),
            energy_c1_j_per_tok: raw.energy_c1_j_per_tok,
            energy_c1_ratio: ratio(raw.energy_c1_j_per_tok, reference.energy_c1_j_per_tok),
            idle_power_w: raw.idle_power_w,
            idle_power_w_ratio: ratio(raw.idle_power_w, reference.idle_power_w),
            stability_sigma_pct: None,
        }
    }
}

/// 2026-10-04: Pull the five raw readings this probe needs out of one leg's
/// gate-record metrics map, by the exact keys those benchmarks already write
/// (SSOT: no new instrumentation). `None` fields are legs that did not run or
/// did not produce the key, not zeros.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LegMetrics<'a> {
    /// 2026-10-04: `decode-floor`'s record: `server_decode_tok_s`,
    /// `gpu_rail_joules_per_token`, `gpu_rail_idle_power_w`.
    pub decode_floor: Option<&'a BTreeMap<String, f64>>,
    /// 2026-10-04: `high-isl-ttft-cold`'s record: `median_ms`.
    pub high_isl_cold: Option<&'a BTreeMap<String, f64>>,
    /// 2026-10-04: `high-isl-ttft-warm`'s record: `median_ms`.
    pub high_isl_warm: Option<&'a BTreeMap<String, f64>>,
}

/// 2026-10-04: The per-dimension bands a box's ratio must sit inside, stated
/// explicitly (PCND) because there is no fleet history yet to ratchet them
/// from. Coordinator-approved 2026-10-04: narrower than a blanket figure for
/// the dimensions the known incident (dgx2 vs dgx3, 6.6-13% apart on energy)
/// would otherwise sail through at a wider band.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bands {
    /// 2026-10-04: `|ratio - 1|` must be at most this for decode bandwidth.
    pub decode: f64,
    /// 2026-10-04: Same, for 32k cold prefill.
    pub prefill: f64,
    /// 2026-10-04: Same, for 32k warm restore. Wider than the others: the
    /// restore-path ratio is the least precisely measured dimension (see
    /// `BoxProfile::restore_warm32k_ms`'s doc — raw TTFT, not yet
    /// warm-minus-tail-prefill).
    pub restore: f64,
    /// 2026-10-04: Same, for C1 energy (J/token).
    pub energy: f64,
    /// 2026-10-04: Ceiling (not a `|ratio - 1|` band: this is a coefficient
    /// of variation, not a ratio-to-fleet) on `stability_sigma_pct`, once it
    /// is measured (module doc: not yet). Reserved so a caller does not need
    /// a second constant when it lands.
    pub stability_cv: f64,
}

/// 2026-10-04: Coordinator-approved 2026-10-04: decode/prefill ±3%, restore
/// ±10%, energy ±5%, stability CV ≤2%.
pub const V1_BANDS: Bands = Bands {
    decode: 0.03,
    prefill: 0.03,
    restore: 0.10,
    energy: 0.05,
    stability_cv: 0.02,
};

/// 2026-10-04: A profile older than this is stale and refused, not warned:
/// a silently-stale calibration would pass a box that changed underneath it
/// (new driver, VBIOS, thermal paste) with the OLD ratios. Stated explicitly
/// (PCND); 30 days is a starting point pending fleet history, not a
/// measured constant.
pub const MAX_PROFILE_AGE_S: u64 = 30 * 24 * 3600;

/// 2026-10-04: Why a box is not healthy enough on its calibration profile:
/// one finding per failing dimension, plus the all-or-nothing missing/stale
/// cases. The caller decides severity (PCND: missing/stale is the only
/// REFUSE case — see the module using this, `bench_certify::preflight` —
/// a dimension outside its band is a WARN, recorded and printed, never
/// silently passed).
#[derive(Clone, Debug, PartialEq)]
pub enum HealthConcern {
    /// 2026-10-04: No profile for this box at all: never calibrated.
    Missing,
    /// 2026-10-04: The profile exists but is older than `MAX_PROFILE_AGE_S`.
    Stale { age_s: u64, max_age_s: u64 },
    /// 2026-10-04: One dimension's ratio is outside its band.
    OutOfBand {
        dimension: &'static str,
        ratio: f64,
        band: f64,
    },
}

impl std::fmt::Display for HealthConcern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Missing => write!(
                f,
                "no calibration profile for this box — run `met benchmark calibrate`"
            ),
            Self::Stale { age_s, max_age_s } => write!(
                f,
                "calibration profile is {:.0} day(s) old, past the {:.0}-day limit — \
                 re-run `met benchmark calibrate`",
                *age_s as f64 / 86_400.0,
                *max_age_s as f64 / 86_400.0
            ),
            Self::OutOfBand {
                dimension,
                ratio,
                band,
            } => write!(
                f,
                "{dimension} reads {:.1}% {} the fleet mean, outside the ±{:.0}% band",
                (ratio - 1.0).abs() * 100.0,
                if *ratio >= 1.0 { "above" } else { "below" },
                band * 100.0
            ),
        }
    }
}

/// 2026-10-04: Is `ratio` within `band` of 1.0? `None` (an unmeasured
/// dimension — the ratio itself absent) is never a concern on its own: a
/// dimension this gate does not depend on, or one no leg has measured yet,
/// says nothing about health. Absence becomes a concern only at the
/// whole-profile level ([`HealthConcern::Missing`]).
fn in_band(ratio: Option<f64>, band: f64) -> bool {
    ratio.is_none_or(|r| (r - 1.0).abs() <= band)
}

/// 2026-10-04: Every reason this box is not calibration-healthy, checking
/// every dimension the profile has a ratio for (not just the dimensions one
/// gate depends on — see the module doc's note on per-gate mapping being
/// deferred). `profile: None` yields exactly [`HealthConcern::Missing`] and
/// nothing else: there is nothing further to check.
pub fn health(profile: Option<&BoxProfile>, bands: &Bands, now_s: u64) -> Vec<HealthConcern> {
    let Some(p) = profile else {
        return vec![HealthConcern::Missing];
    };
    let mut out = Vec::new();
    let age_s = now_s.saturating_sub(p.measured_at);
    if age_s > MAX_PROFILE_AGE_S {
        out.push(HealthConcern::Stale {
            age_s,
            max_age_s: MAX_PROFILE_AGE_S,
        });
    }
    let dims: [(&'static str, Option<f64>, f64); 4] = [
        ("decode bandwidth", p.decode_tok_s_ratio, bands.decode),
        (
            "32k cold prefill",
            p.prefill_cold32k_ms_ratio,
            bands.prefill,
        ),
        (
            "32k warm restore",
            p.restore_warm32k_ms_ratio,
            bands.restore,
        ),
        ("C1 energy", p.energy_c1_ratio, bands.energy),
    ];
    for (dimension, ratio, band) in dims {
        if !in_band(ratio, band) {
            out.push(HealthConcern::OutOfBand {
                dimension,
                ratio: ratio.expect("in_band only fails on Some"),
                band,
            });
        }
    }
    out
}

pub fn extract(legs: LegMetrics<'_>) -> Raw {
    let get = |m: Option<&BTreeMap<String, f64>>, k: &str| m.and_then(|m| m.get(k)).copied();
    Raw {
        decode_tok_s: get(legs.decode_floor, "server_decode_tok_s"),
        prefill_cold32k_ms: get(legs.high_isl_cold, "median_ms"),
        restore_warm32k_ms: get(legs.high_isl_warm, "median_ms"),
        energy_c1_j_per_tok: get(legs.decode_floor, "gpu_rail_joules_per_token"),
        idle_power_w: get(legs.decode_floor, "gpu_rail_idle_power_w"),
    }
}

#[cfg(test)]
#[path = "calibration_tests.rs"]
mod tests;
