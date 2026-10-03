// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The caches over the recurrent state (the prefix-cache snapshots, the
//! decode-rollback ring, the carried-state verify's stash and tables) sized from the circuit's
//! declarations (`metrale_circuit::cache_states`, `memory::caches::cache_terms`), so the server's
//! pre-load reserve and the snapshot pool's allocation read one source (LIFECYCLE-DESIGN.md 15.3;
//! MEMORY-DESIGN.md section 3, steps 1 and 2).
//!
//! Owner: model-layers (SSM reserve).
//! Invariants:
//! - A model whose `model_type` has a circuit is sized from its declarations only; a model
//!   without one keeps the transitional arithmetic (`ModelConfig::ssm_*_state_bytes`, no
//!   last-hidden row), as `PoolPlan` does, until M3.
//! - A per-layer cache counts once per recurrent layer; a model-level one once.

use anyhow::{Context, Result, ensure};
use metrale_circuit::memory::caches::{CacheInputs, cache_terms};
use metrale_circuit::state::{StateDecl, StateDtype, StateKind, VerifySteps};
use metrale_config::ModelConfig;
use std::collections::BTreeMap;

use super::{UnitSource, recurrent_units, state_dims, state_formats};

/// 2026-10-03: The caches of one model, ready to size.
#[derive(Debug, Clone)]
pub struct CachePlan {
    /// 2026-10-03: Where the sizes come from.
    pub source: UnitSource,
    layers: usize,
    layer: Vec<StateDecl>,
    model: Vec<StateDecl>,
    formats: BTreeMap<String, StateDtype>,
    /// 2026-10-03: Transitional only: one layer's FP32 h and conv bytes.
    transitional: (usize, usize),
}

/// 2026-10-03: One prefix-cache slot's per-layer units and its last-hidden row, as the snapshot
/// pool allocates them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefixUnits {
    pub h: usize,
    pub conv: usize,
    pub hidden: usize,
}

impl CachePlan {
    /// 2026-10-03: The caches of `config`'s model under `--ssm-h-dtype` (`h_f16_pool`).
    pub fn new(config: &ModelConfig, h_f16_pool: bool) -> Result<Self> {
        let layers = config.num_ssm_layers();
        let mut dims = state_dims(config);
        dims.insert("hidden".into(), config.hidden_size as u64);
        let transitional = (config.ssm_h_state_bytes(), config.ssm_conv_state_bytes());
        let found =
            metrale_circuit::cache_states(&config.model_type, &dims).with_context(|| {
                format!(
                    "the circuit of `{}` declares no usable cache",
                    config.model_type
                )
            })?;
        let (source, layer, model) = match found {
            Some(c) => (UnitSource::Circuit, c.layer, c.model),
            None => (UnitSource::Transitional, Vec::new(), Vec::new()),
        };
        Ok(Self {
            source,
            layers,
            layer,
            model,
            formats: state_formats(h_f16_pool),
            transitional,
        })
    }

    /// 2026-10-03: Bytes of the caches of the kinds `keep` selects under `inputs`.
    fn bytes(&self, inputs: &CacheInputs, keep: impl Fn(StateKind) -> bool) -> Result<usize> {
        let sum = |decls: &[StateDecl]| -> Result<u64> {
            Ok(cache_terms(decls, &self.formats, inputs)?
                .iter()
                .filter(|t| keep(t.kind) && !t.host)
                .map(|t| t.bytes)
                .sum())
        };
        let (per_layer, model) = (sum(&self.layer)?, sum(&self.model)?);
        let total = per_layer
            .checked_mul(self.layers as u64)
            .and_then(|l| l.checked_add(model))
            .context("cache bytes overflow")?;
        Ok(usize::try_from(total)?)
    }

    /// 2026-10-03: The prefix cache (Marconi) at `slots` slots: every layer's h and conv
    /// snapshots and the last-hidden row (the row the reserve left out before 2026-10-03).
    pub fn marconi_bytes(&self, slots: usize) -> Result<usize> {
        if self.source == UnitSource::Transitional {
            return Ok(slots * self.layers * (self.transitional.0 + self.transitional.1));
        }
        let inputs = CacheInputs {
            prefix_snapshot_slots: slots as u64,
            ..Default::default()
        };
        self.bytes(&inputs, |k| k == StateKind::PrefixSnapshot)
    }

    /// 2026-10-03: One sequence's decode-rollback ring slot over every layer.
    pub fn ring_seq_bytes(&self) -> Result<usize> {
        if self.source == UnitSource::Transitional {
            return Ok(self.layers * (self.transitional.0 + self.transitional.1));
        }
        let inputs = CacheInputs {
            ring: (1, 1),
            ..Default::default()
        };
        self.bytes(&inputs, |k| k == StateKind::RingSnapshot)
    }

    /// 2026-10-03: The carried-state verify's stash for `slots` verify slots (the dummy
    /// included) and its tables for `table_rows` rows; 0 for a model whose circuit declares no
    /// carry (the transitional source binds none).
    pub fn carry_bytes(&self, slots: usize, table_rows: usize) -> Result<usize> {
        let inputs = CacheInputs {
            carry: (slots as u64, table_rows as u64),
            ..Default::default()
        };
        self.bytes(&inputs, |k| {
            matches!(k, StateKind::CarryStash | StateKind::CarryTable)
        })
    }

    /// 2026-10-03: One prefix slot's units, as the snapshot pool allocates them.
    pub fn prefix_units(&self, config: &ModelConfig) -> Result<PrefixUnits> {
        if self.source == UnitSource::Transitional {
            return Ok(PrefixUnits {
                h: self.transitional.0,
                conv: self.transitional.1,
                hidden: config.hidden_size * 2,
            });
        }
        let (recurrent, _) = recurrent_units(config)?;
        let unit_of = |v: VerifySteps| -> Result<usize> {
            let of = recurrent
                .iter()
                .find(|d| d.verify == Some(v))
                .with_context(|| format!("no recurrent state keeps {v:?} intermediates"))?;
            let snap = self
                .layer
                .iter()
                .find(|d| {
                    d.kind == StateKind::PrefixSnapshot
                        && d.copies.as_deref() == Some(of.id.as_str())
                })
                .with_context(|| {
                    format!("the circuit declares no prefix snapshot of `{}`", of.id)
                })?;
            Ok(usize::try_from(snap.unit_bytes(&self.formats)?)?)
        };
        let hidden: Vec<_> = self
            .model
            .iter()
            .filter(|d| d.kind == StateKind::PrefixSnapshot && d.copies.is_none())
            .collect();
        ensure!(
            hidden.len() == 1,
            "the circuit declares {} last-hidden prefix rows; the snapshot pool keeps one",
            hidden.len()
        );
        Ok(PrefixUnits {
            h: unit_of(VerifySteps::H)?,
            conv: unit_of(VerifySteps::Conv)?,
            hidden: usize::try_from(hidden[0].unit_bytes(&self.formats)?)?,
        })
    }
}

#[cfg(test)]
#[path = "cache_plan_tests.rs"]
mod cache_plan_tests;
