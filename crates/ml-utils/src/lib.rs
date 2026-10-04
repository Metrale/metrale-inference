// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: Model utilities for the Metrale Engine, as pure plans: `met ml-utils` and the
//! serve flag `--mock` are their callers.
//!
//! A *mock* (rehearsal) checkpoint keeps a model's architecture exactly (every dim, format,
//! scale layout, layer kind and per-layer precision) and keeps only some of its layers, with
//! synthetic weights, so kernels can be iterated and timed on new hardware without the full
//! weights (ml-utils DESIGN.md).
//!
//! | module | role |
//! |---|---|
//! | [`spec`] | the mock spec (no defaults) |
//! | [`index`], [`st_format`] | tensor index from safetensors headers; the container format |
//! | [`schedule`], [`rename`], [`quant_meta`] | which layers are kept; renaming; metadata parity |
//! | [`scheme`], [`values`], [`synth`], [`rng`] | storage schemes, value classes, encoding, streams |
//! | [`routing`] | expert-load profiles reproduced in router weights |
//! | [`plan`], [`mod@write`] | the whole plan; writing it through a sink |
//! | [`inspect`] | what `met ml-utils inspect` reports |
//! | [`extrapolate`] | full-model estimates from mock measurements |
//! | [`io`] | the I/O traits (SBIO) |
//!
//! Owner: metrale-ml-utils.
//! Invariants:
//! - No file, network, GPU, environment or clock access in this crate: every byte crosses
//!   [`io::CheckpointSource`] or [`io::CheckpointSink`], or is a function argument.
//! - Synthesis is deterministic: the same spec, seed and checkpoint give the same bytes on every
//!   platform and thread count.

pub mod error;
pub mod extrapolate;
pub mod index;
pub mod inspect;
pub mod io;
pub mod plan;
pub mod quant_meta;
pub mod rename;
pub mod rng;
pub mod routing;
pub mod schedule;
pub mod scheme;
pub mod spec;
pub mod st_format;
pub mod synth;
pub mod values;
pub mod write;

pub use error::{MlError, Result};
pub use index::{Dtype, TensorEntry, TensorIndex};
pub use plan::{MockInputs, MockPlan, OutTensor, Unit, plan_mock, resolved_digest, synthesize};
pub use routing::RoutingProfile;
pub use spec::MockSpec;
pub use write::{
    MOCK_METADATA_KEY, RESOLVED_FILE, SourceTexts, read_source, write_mock, write_skeleton,
};

/// 2026-10-03: Toy checkpoints for this crate's and its callers' tests.
#[cfg(any(test, feature = "test-utils"))]
pub mod testkit;
