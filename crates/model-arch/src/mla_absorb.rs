// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `METRALE_MLA_ABSORB_THREADS`, the worker count of the host-side MLA absorption
//! GEMMs (`glm5next_dsa::build::absorb_q` and the Mistral/LongCat `phase_qk_absorbed`), resolved
//! once from the startup settings. One lever tunes both; neither reads the process environment.
//!
//! Owner: model-arch (MLA weight loading).
//! Invariants: the count never changes the result (each worker owns disjoint output rows), and
//! [`AbsorbThreads::for_rows`] is always in `1..=rows.max(1)`.

use metrale_settings::InferenceServerSettings;

/// 2026-10-17: The requested worker count, `None` for the available parallelism.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AbsorbThreads(Option<usize>);

impl AbsorbThreads {
    /// 2026-10-17: Resolve from the startup settings: a positive count is the request; zero, a
    /// negative value or unset asks for the available parallelism.
    pub fn from_settings(s: &InferenceServerSettings) -> Self {
        Self(
            s.kernels
                .mla_absorb_threads
                .and_then(|n| usize::try_from(n).ok())
                .filter(|&n| n > 0),
        )
    }

    /// 2026-09-25: Workers for `rows` output rows: the request, else the available parallelism,
    /// clamped to `1..=rows`. `1` runs the absorption sequentially.
    pub fn for_rows(self, rows: usize) -> usize {
        self.0
            .unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1)
            })
            .clamp(1, rows.max(1))
    }
}

#[cfg(test)]
#[path = "mla_absorb_tests.rs"]
mod tests;
