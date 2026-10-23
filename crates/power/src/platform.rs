// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Each platform's declared headline order (owner decision 2026-10-22, PCND): which
//! present source heads a row, per scope. Every present source is still sampled and stored on
//! the row (`energy_by_source`); the order chooses only the headline, and an absent order is an
//! error (`EnergyNoPreferredSource`), never a guessed fallback.
//!
//! Owner: metrale-power.
//! Invariants: a platform id resolves to exactly one declaration.

use alloc::format;
use alloc::vec;
use alloc::vec::Vec;

use metrale_abstract_io::power::{HeadlinePreference, Rail, Scope, SourceKind};
use metrale_error::{ErrorCode, error};

use crate::Result;

/// 2026-10-22: One platform's headline orders and the rails each scope covers.
#[derive(Clone, Debug, PartialEq)]
pub struct PlatformPower {
    pub platform: &'static str,
    pub gpu_rail: HeadlinePreference,
    pub gpu_rails: Vec<Rail>,
    pub whole_system: HeadlinePreference,
    pub system_rails: Vec<Rail>,
}

fn pref(platform: &'static str, scope: Scope, order: Vec<SourceKind>) -> HeadlinePreference {
    HeadlinePreference {
        platform,
        scope,
        order,
    }
}

/// 2026-10-22: The declared platforms. GB10 and the NVIDIA datacentre parts: the NVML counter
/// first (an integral with no sampling lag, §1.3), then the sensor, then the `nvidia-smi` child;
/// whole-system only from an external meter. gfx1151: the APU socket (CPU + GPU). Apple: the
/// IOReport GPU channel.
#[must_use]
pub fn declared() -> Vec<PlatformPower> {
    use SourceKind::*;
    let nvidia = |p: &'static str, rails: Vec<Rail>| PlatformPower {
        platform: p,
        gpu_rail: pref(
            p,
            Scope::GpuRail,
            vec![GpuDriverCounter, GpuDriverSensor, NvidiaSmi, Hwmon],
        ),
        gpu_rails: rails,
        whole_system: pref(p, Scope::WholeSystem, vec![External]),
        system_rails: vec![Rail::Module, Rail::Wall],
    };
    vec![
        nvidia("gb10", vec![Rail::GpuCore]),
        nvidia("h100", vec![Rail::GpuCore, Rail::Board]),
        nvidia("h200", vec![Rail::GpuCore, Rail::Board]),
        nvidia("b200", vec![Rail::GpuCore, Rail::Board]),
        nvidia("gb300", vec![Rail::GpuCore, Rail::Board]),
        PlatformPower {
            platform: "gfx1151",
            gpu_rail: pref("gfx1151", Scope::GpuRail, vec![AmdSmi, Hwmon]),
            gpu_rails: vec![Rail::Soc],
            whole_system: pref("gfx1151", Scope::WholeSystem, vec![External]),
            system_rails: vec![Rail::Module, Rail::Wall],
        },
        PlatformPower {
            platform: "apple",
            gpu_rail: pref("apple", Scope::GpuRail, vec![AppleIoReport]),
            gpu_rails: vec![Rail::GpuCore],
            whole_system: pref("apple", Scope::WholeSystem, vec![External]),
            system_rails: vec![Rail::Module, Rail::Wall],
        },
    ]
}

/// 2026-10-22: The declaration for `platform`.
pub fn platform(platform: &str) -> Result<PlatformPower> {
    declared()
        .into_iter()
        .find(|p| p.platform == platform)
        .ok_or_else(|| {
            error!(
                ErrorCode::EnergyInputInvalid,
                format!("platform `{platform}` declares no power headline order")
            )
        })
}

impl PlatformPower {
    /// 2026-10-22: The headline order and rails of `scope`.
    #[must_use]
    pub fn scope(&self, scope: Scope) -> (&HeadlinePreference, &[Rail]) {
        match scope {
            Scope::GpuRail => (&self.gpu_rail, &self.gpu_rails),
            Scope::WholeSystem => (&self.whole_system, &self.system_rails),
        }
    }
}

/// 2026-10-22: Parse a stored scope.
pub fn parse_scope(s: &str) -> Result<Scope> {
    [Scope::GpuRail, Scope::WholeSystem]
        .into_iter()
        .find(|x| x.as_str() == s)
        .ok_or_else(|| {
            error!(
                ErrorCode::EnergyInputInvalid,
                format!("unknown scope `{s}`")
            )
        })
}
