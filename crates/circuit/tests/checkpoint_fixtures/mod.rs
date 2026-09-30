// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: The fixture side of the checkpoint tests (`checkpoints.rs`, Path A, and
//! `checkpoint_refusals.rs`, Paths B and C): reading tests/fixtures/checkpoints/<org>--<name>/ and
//! resolving it under the declared tier. Split out of checkpoints.rs when it passed 500 lines.
//!
//! Owner: metrale-circuit tests.
//! Invariants: none beyond the types.

#![allow(dead_code)]

use std::collections::BTreeMap;

use metrale_circuit::ir::Circuit;
use metrale_circuit::{
    CheckpointError, Format, QuantMetadata, ResolvedCheckpoint, Scale, ServePrecision,
    resolve_checkpoint,
};

use crate::common;

pub fn fixture(name: &str) -> (String, Option<String>) {
    let dir = common::root()
        .join("crates/circuit/tests/fixtures/checkpoints")
        .join(name);
    let config = std::fs::read_to_string(dir.join("config.json"))
        .unwrap_or_else(|e| panic!("{name}/config.json: {e}"));
    (
        config,
        std::fs::read_to_string(dir.join("hf_quant_config.json")).ok(),
    )
}

pub fn resolve(name: &str) -> Result<ResolvedCheckpoint, CheckpointError> {
    let (config, hq) = fixture(name);
    resolve_checkpoint(
        &config,
        QuantMetadata {
            hf_quant_config: hq.as_deref(),
        },
        &ServePrecision::Declared,
    )
}

pub fn ok(name: &str) -> ResolvedCheckpoint {
    resolve(name).unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// 2026-09-30: Layer kinds as counts per kind.
pub fn kinds(r: &ResolvedCheckpoint) -> BTreeMap<&'static str, usize> {
    let mut m = BTreeMap::new();
    for k in &r.shape.layer_kinds {
        *m.entry(k.name()).or_default() += 1;
    }
    m
}

pub fn dim(r: &ResolvedCheckpoint, d: &str) -> u64 {
    *r.shape.dims.get(d).unwrap_or_else(|| panic!("no dim {d}"))
}

/// 2026-09-30: The node `id` of `c`.
pub fn node<'c>(c: &'c Circuit, id: &str) -> &'c metrale_circuit::ir::Node {
    &c.nodes[c.node(id).unwrap_or_else(|| panic!("no node {id}"))]
}

/// 2026-09-30: The format of `id`'s first input edge.
pub fn input_format(c: &Circuit, id: &str) -> Format {
    c.edges[node(c, id).inputs[0]].format
}

pub const FP8_TOKEN: Format = Format::Fp8E4m3 {
    scale: Scale::PerToken,
};
pub const FP8_TENSOR: Format = Format::Fp8E4m3 {
    scale: Scale::PerTensor,
};
pub const NVFP4: Format = Format::Nvfp4 { group: 16 };
