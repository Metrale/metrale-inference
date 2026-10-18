// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The NLLB beam-search levers, resolved once from the startup settings when
//! `NllbGpuModel::new` builds the model.
//!
//! Owner: model-engine NLLB.
//! Invariants: nothing on the beam path reads the environment.

/// 2026-10-17: The resolved NLLB beam levers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NllbLevers {
    /// 2026-10-17: The `telemetry.nllb_beam_profile` switch (`METRALE_NLLB_BEAM_PROFILE`): time
    /// the encoder, the LM head and each beam step.
    pub beam_profile: bool,
    /// 2026-10-17: The `sampling.nllb_host_topk` switch (`METRALE_NLLB_HOST_TOPK`): the host
    /// always scans the full vocab.
    pub host_topk: bool,
    /// 2026-10-17: The fewest rows the on-device top-k runs at: the non-negative
    /// `sampling.nllb_device_topk_min_rows` (`METRALE_NLLB_DEVICE_TOPK_MIN_ROWS`), else 64.
    pub device_topk_min_rows: usize,
}

impl NllbLevers {
    pub fn from_settings(s: &metrale_settings::InferenceServerSettings) -> Self {
        Self {
            beam_profile: s.telemetry.nllb_beam_profile,
            host_topk: s.sampling.nllb_host_topk,
            device_topk_min_rows: s
                .sampling
                .nllb_device_topk_min_rows
                .and_then(|v| usize::try_from(v).ok())
                .unwrap_or(64),
        }
    }
}

#[cfg(test)]
#[path = "levers_tests.rs"]
mod tests;
