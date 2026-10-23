// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The in-serve validator (DESIGN §3.2): per window `w` (100 ms counter windows of a
//! serve under load), `P_w = P_static + Σ_k f_{k,w} · P_k`, with `f_{k,w}` the share of the window
//! kernel `k` occupied (from kernel timestamps). Solved by non-negative least squares
//! (Lawson-Hanson), with per-kernel intervals from a moving-block bootstrap over windows (the
//! windows are autocorrelated). Collinear kernels cannot be told apart: when the column-scaled
//! design's condition number or any variance inflation factor exceeds its declared limit
//! (Belsley 1980), the attribution is refused (`EnergyIllConditioned`), never split by guess.
//!
//! Owner: metrale-power.
//! Invariants: at least [`MIN_WINDOWS`] windows; every occupancy in `[0, 1]`.

use alloc::format;
use alloc::string::ToString;
use alloc::vec::Vec;

use metrale_error::{ErrorCode, error};
use metrale_stats::interval::Interval;
use metrale_stats::nnls::{
    MAX_CONDITION, MAX_VIF, Matrix, bootstrap_coefficients, condition_number, nnls, vif,
};

use crate::Result;

/// 2026-10-22: Fewest windows an attribution accepts (DESIGN §3.2).
pub const MIN_WINDOWS: usize = 300;

/// 2026-10-22: One window: its mean power and each kernel's occupancy share.
#[derive(Clone, Debug, PartialEq)]
pub struct ServeWindow {
    pub power_w: f64,
    pub occupancy: Vec<f64>,
}

/// 2026-10-22: Static power and per-kernel power while running, with intervals.
#[derive(Clone, Debug, PartialEq)]
pub struct ServeAttribution {
    pub static_w: f64,
    pub kernel_w: Vec<f64>,
    /// 2026-10-22: Static first, then each kernel.
    pub intervals: Vec<Interval>,
    pub condition: f64,
    pub vif: Vec<f64>,
}

/// 2026-10-22: Attribute `windows` over `kernels` kernels.
pub fn attribute(
    windows: &[ServeWindow],
    kernels: usize,
    reps: usize,
    level: f64,
    seed: u64,
) -> Result<ServeAttribution> {
    if windows.len() < MIN_WINDOWS {
        return Err(error!(
            ErrorCode::EnergyWindowRefused,
            "in-serve regression".to_string(),
            format!("{} windows is fewer than {MIN_WINDOWS}", windows.len())
        ));
    }
    let cols = kernels + 1;
    let mut data = Vec::with_capacity(windows.len() * cols);
    for w in windows {
        if w.occupancy.len() != kernels || w.occupancy.iter().any(|f| !(0.0..=1.0).contains(f)) {
            return Err(error!(
                ErrorCode::EnergyInputInvalid,
                format!("a window needs {kernels} occupancies in [0, 1]")
            ));
        }
        data.push(1.0);
        data.extend_from_slice(&w.occupancy);
    }
    let a = Matrix::new(windows.len(), cols, data)?;
    let b: Vec<f64> = windows.iter().map(|w| w.power_w).collect();
    let condition = condition_number(&a)?;
    let factors = vif(&occupancy_only(windows, kernels)?)?;
    let worst = factors.iter().copied().fold(0.0, f64::max);
    if condition > MAX_CONDITION || worst > MAX_VIF {
        return Err(error!(
            ErrorCode::EnergyIllConditioned,
            condition, MAX_CONDITION, worst
        ));
    }
    let x = nnls(&a, &b)?;
    let intervals = bootstrap_coefficients(&a, &b, reps, level, seed)?;
    Ok(ServeAttribution {
        static_w: x[0],
        kernel_w: x[1..].to_vec(),
        intervals,
        condition,
        vif: factors,
    })
}

fn occupancy_only(windows: &[ServeWindow], kernels: usize) -> Result<Matrix> {
    let data: Vec<f64> = windows
        .iter()
        .flat_map(|w| w.occupancy.iter().copied())
        .collect();
    Matrix::new(windows.len(), kernels, data)
}
