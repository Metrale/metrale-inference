// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-07: Pure GPT-OSS architecture construction before executable lowering.
//!
//! Owner: metrale-circuit. The package describes the pinned 20B checkpoint's math,
//! state and declared storage. It deliberately does not register a serving target.
//! A node marked `unlowered` cannot match any fusion rule, even a generic op match.

use crate::checkpoint::CheckpointError;
use crate::config_map::{ConfigMap, map_config};
use crate::{Circuit, PrecisionTable, instantiate};

const CONFIG_MAP: &str = include_str!("../../../kernels/circuits/gpt_oss.config.toml");
const CIRCUIT: &str = include_str!("../../../kernels/circuits/gpt_oss.toml");
const PRECISION: &str = include_str!("../../../kernels/circuits/precision/gpt-oss-20b.toml");

/// 2026-10-07: Describe GPT-OSS-20B from supplied checkpoint metadata, refusing
/// missing, changed or unclassified math. This is an architecture graph, not a
/// runnable kernel plan; no GPU or file I/O occurs here.
pub fn architecture(config: &serde_json::Value) -> Result<Circuit, CheckpointError> {
    let map = ConfigMap::parse(CONFIG_MAP)?;
    let mapped = map_config(&map, config)?;
    let precision = PrecisionTable::parse(PRECISION)?;
    Ok(instantiate(CIRCUIT, &[], &mapped.shape, &precision)?)
}
