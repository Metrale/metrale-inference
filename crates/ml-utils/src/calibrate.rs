// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: Router calibration from a mock's recorded expert loads. A histogram-routed mock
//! sizes each router's bias column for unit noise; under the mock's own activations the noise a
//! token's other channels give differs per layer. For each mock MoE layer, the scale `lambda`
//! at which the fitted bias, under unit noise, best reproduces the recorded loads measures the
//! realised bias-to-noise ratio, and the layer's next gain is its current gain over `lambda`.
//!
//! Owner: metrale-ml-utils.
//! Invariants:
//! - Recorded rows are in first-touch order, which is ascending mock layer order of the MoE
//!   layers; the routers are matched to them in that order, and a count mismatch is refused.
//! - A recording whose expert count or top-k differs from the profile is refused.

use std::collections::BTreeMap;

use crate::error::{MlError, Result};
use crate::plan::MockPlan;
use crate::rng::Stream;
use crate::routing::{RoutingProfile, realized_ratio, total_variation};

/// 2026-10-04: One mock MoE layer's calibration.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerCalibration {
    /// 2026-10-04: The mock layer.
    pub mock_layer: usize,
    /// 2026-10-04: The source layer its router reproduces.
    pub source_layer: usize,
    /// 2026-10-04: Total-variation distance of the recorded loads from the profile row.
    pub tv_recorded: f64,
    /// 2026-10-04: The realised bias scale.
    pub lambda: f32,
    /// 2026-10-04: Distance of the unit-noise model at `lambda` from the recorded loads.
    pub tv_model: f64,
    /// 2026-10-04: The gain the recording was made with.
    pub gain_before: f32,
    /// 2026-10-04: The calibrated gain.
    pub gain: f32,
    /// 2026-10-04: Experts never picked in the recording, and in the profile row.
    pub never_picked: (usize, usize),
}

/// 2026-10-04: The mock layer of a mock tensor name (`...layers.<n>....`).
fn mock_layer(name: &str) -> Option<usize> {
    let mut parts = name.split('.');
    while let Some(p) = parts.next() {
        if p == "layers" {
            return parts.next()?.parse().ok();
        }
    }
    None
}

/// 2026-10-04: Calibrate every router of `plan` (built from `profile`) from `recorded`.
pub fn calibrate(
    plan: &MockPlan,
    profile: &RoutingProfile,
    recorded: &RoutingProfile,
) -> Result<Vec<LayerCalibration>> {
    if (recorded.experts, recorded.top_k) != (profile.experts, profile.top_k) {
        return Err(MlError::Routing(format!(
            "the recording has {} experts top-{}, the profile {} top-{}",
            recorded.experts, recorded.top_k, profile.experts, profile.top_k
        )));
    }
    let mut routers = Vec::with_capacity(plan.routers.len());
    for f in &plan.routers {
        let l = mock_layer(&f.tensor)
            .ok_or_else(|| MlError::Routing(format!("no layer number in {}", f.tensor)))?;
        routers.push((l, f));
    }
    routers.sort_by_key(|(l, _)| *l);
    if routers.len() != recorded.layers.len() {
        return Err(MlError::Routing(format!(
            "the recording has {} MoE layers, the mock {} routers",
            recorded.layers.len(),
            routers.len()
        )));
    }
    let zeros = |r: &[u64]| r.iter().filter(|&&c| c == 0).count();
    let mut out = Vec::with_capacity(routers.len());
    for ((mock, f), row) in routers.into_iter().zip(&recorded.layers) {
        let target = &profile.layers[f.profile_row];
        let stream = Stream::for_tensor(plan.seed, &f.tensor, "calibration", &[]).derive(2);
        let (lambda, tv_model) = realized_ratio(&f.bias, profile.top_k, row, stream);
        out.push(LayerCalibration {
            mock_layer: mock,
            source_layer: f.source_layer,
            tv_recorded: total_variation(target, row),
            lambda,
            tv_model,
            gain_before: f.gain,
            gain: f.gain / lambda,
            never_picked: (zeros(row), zeros(target)),
        });
    }
    Ok(out)
}

/// 2026-10-04: The gains of `layers`, by source layer.
pub fn gains(layers: &[LayerCalibration]) -> BTreeMap<usize, f32> {
    layers.iter().map(|l| (l.source_layer, l.gain)).collect()
}

#[cfg(test)]
#[path = "calibrate_tests.rs"]
mod calibrate_tests;
