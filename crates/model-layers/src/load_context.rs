// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: [`LoadContext`], what a weight loader reads besides the checkpoint: the settings
//! resolved at startup and the model levers derived from them once. It is the load-time
//! counterpart of the two fields of the same names on [`crate::layer::ForwardContext`].
//!
//! Owner: model-layers.
//! Invariants: a loader never reads the process environment; a lever it needs is a field of
//! one of these two values.

use crate::layers::ops::ModelLevers;
use metrale_settings::InferenceServerSettings;

/// 2026-10-17: The resolved settings and model levers a loader reads, borrowed for one load.
#[derive(Clone, Copy, Debug)]
pub struct LoadContext<'a> {
    pub settings: &'a InferenceServerSettings,
    pub levers: &'a ModelLevers,
}
