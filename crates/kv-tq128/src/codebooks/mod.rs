// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: The codebooks of the 128-group codecs, verbatim from spiritbuun/buun-llama-cpp
//! (MIT) at 34ef0ef7c6ea. These consts are the only copy: the kernels receive a codebook as a
//! pointer argument, uploaded from here, so there is no second table to drift. Credit
//! spiritbuun.
//!
//! Owner: kv-tq128.
//! Invariants: a TCQ book has one entry per trellis state; the turbo4 book has 16 levels.

// 2026-10-18: Upstream's 9-significant-digit literals are kept as written so the table can be
// diffed against the pinned source; each still parses to the f32 the upstream compiler builds.
#![allow(clippy::excessive_precision)]

mod tcq1;
mod tcq2;
mod tcq3;
mod turbo4;

pub use tcq1::{TCQ1_K, TCQ1_V};
pub use tcq2::{TCQ2_K, TCQ2_V};
pub use tcq3::{TCQ3_K, TCQ3_V};
pub use turbo4::{TURBO4_LEVELS, TURBO4_MID};
