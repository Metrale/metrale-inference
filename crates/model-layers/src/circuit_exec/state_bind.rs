// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The circuit's state programs (`metrale_circuit::state_ops`) bound to the SSM
//! pool's layer order: each node becomes (SSM layer, h or conv, bytes, conversion), checked
//! against the pool's own unit sizes, so the copies the engine issues between steps are sized by
//! the circuit and a disagreement refuses the build instead of copying a wrong width.
//!
//! Owner: model-layers circuit executor.
//! Invariants:
//! - A node's part is its state's verify kind (`h_steps` is h, `conv_steps` is conv), the rule
//!   `ssm_reserve::PoolPlan` sizes the pool by.
//! - A program is complete (one node per recurrent target state) or empty; an empty program
//!   is refused when it is asked for on a model with recurrent layers.
//! - A copy or zero between pool places writes exactly the pool's unit for its part.

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail, ensure};
use metrale_circuit::Circuit;
use metrale_circuit::state::{StateDtype, StateKind, VerifySteps};
use metrale_circuit::state_ops::{StateOp, StatePlace, StateProgramId, state_programs};

/// 2026-10-03: The pool state a node moves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatePart {
    H,
    Conv,
}

/// 2026-10-03: An element conversion on the way (the prefix cache keeps FP32).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Conversion {
    /// 2026-10-03: f16 to f32 (`ssm_h_state_f16_to_f32`).
    Widen,
    /// 2026-10-03: f32 to f16 (`ssm_h_state_f32_to_f16`).
    Narrow,
}

/// 2026-10-03: One bound node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoundStateNode {
    /// 2026-10-03: Index among the model's recurrent layers (the pool's layer index).
    pub ssm_layer: usize,
    pub part: StatePart,
    /// 2026-10-03: Bytes written.
    pub bytes: usize,
    /// 2026-10-03: The node's op.
    pub op: StateOp,
    /// 2026-10-03: `None` for a bit copy.
    pub conversion: Option<Conversion>,
}

/// 2026-10-03: The pool's unit sizes, as allocated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolUnits {
    pub h_stored: usize,
    pub conv: usize,
}

/// 2026-10-03: The SSM pool a build binds the state programs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StatePool {
    /// 2026-10-03: Its unit sizes, as allocated.
    pub units: PoolUnits,
    /// 2026-10-03: `--ssm-h-dtype f16-pool`.
    pub h_f16: bool,
}

impl StatePool {
    /// 2026-10-03: Bind `circuit`'s programs to this pool, whose recurrent layers are the
    /// linear-attention layers of `config` in layer order (the pool's order).
    pub fn bind(
        &self,
        circuit: &Circuit,
        config: &metrale_config::ModelConfig,
    ) -> Result<StatePrograms> {
        let recurrent: Vec<usize> = (0..config.num_hidden_layers)
            .filter(|&i| config.layer_type(i) == metrale_config::LayerType::LinearAttention)
            .collect();
        StatePrograms::bind(
            circuit,
            &crate::ssm_reserve::state_formats(self.h_f16),
            &recurrent,
            self.units,
        )
    }
}

/// 2026-10-03: Every state program, bound.
#[derive(Debug, Clone, Default)]
pub struct StatePrograms {
    by_id: BTreeMap<StateProgramId, Vec<BoundStateNode>>,
    recurrent_layers: usize,
    /// 2026-10-03: Bytes of the prefix cache's last-hidden row (`head.prefix_hidden`); `None`
    /// when the circuit declares none.
    prefix_hidden_bytes: Option<usize>,
}

impl StatePrograms {
    /// 2026-10-03: Bind `circuit`'s programs under `formats` to a pool whose recurrent layers
    /// are the model layers `recurrent_layers` (in pool order) and whose units are `units`.
    pub fn bind(
        circuit: &Circuit,
        formats: &BTreeMap<String, StateDtype>,
        recurrent_layers: &[usize],
        units: PoolUnits,
    ) -> Result<Self> {
        let expected = circuit
            .states
            .iter()
            .filter(|s| {
                s.kind == StateKind::Recurrent && s.section == metrale_circuit::Section::Main
            })
            .count();
        ensure!(
            expected == 2 * recurrent_layers.len(),
            "the circuit declares {expected} recurrent target states; the pool has {} \
             recurrent layers of h and conv",
            recurrent_layers.len()
        );
        let mut by_id = BTreeMap::new();
        for p in state_programs(circuit) {
            ensure!(
                p.nodes.is_empty() || p.nodes.len() == expected,
                "state program `{}` covers {} of the {expected} recurrent states; its \
                 snapshot is declared for some layers only",
                p.id.name(),
                p.nodes.len()
            );
            let mut bound = Vec::with_capacity(p.nodes.len());
            for n in &p.nodes {
                let decl = &circuit.states[n.state];
                let layer = decl
                    .layer
                    .with_context(|| format!("state `{}` belongs to no layer", decl.id))?;
                let ssm_layer = recurrent_layers
                    .iter()
                    .position(|&l| l == layer)
                    .with_context(|| {
                        format!(
                            "state `{}` is on layer {layer}, not a recurrent one",
                            decl.id
                        )
                    })?;
                let part = match decl.verify {
                    Some(VerifySteps::H) => StatePart::H,
                    Some(VerifySteps::Conv) => StatePart::Conv,
                    None => bail!(
                        "recurrent state `{}` names no verify intermediates",
                        decl.id
                    ),
                };
                let bytes = usize::try_from(n.bytes(circuit, formats)?)?;
                let (src, dst) = n.dtypes(circuit, formats)?;
                let conversion = match (n.op, src, dst) {
                    (StateOp::Convert { .. }, StateDtype::F16, StateDtype::F32) => {
                        Some(Conversion::Widen)
                    }
                    (StateOp::Convert { .. }, StateDtype::F32, StateDtype::F16) => {
                        Some(Conversion::Narrow)
                    }
                    (_, a, b) if a == b => None,
                    (StateOp::Copy { .. }, ..) => None,
                    (op, a, b) => bail!(
                        "node `{}`: no {op:?} from {} to {}",
                        n.id,
                        a.name(),
                        b.name()
                    ),
                };
                let unit = match part {
                    StatePart::H => units.h_stored,
                    StatePart::Conv => units.conv,
                };
                let pool_side = !matches!(n.op.writes(), StatePlace::Ring | StatePlace::Prefix);
                let raw_into_cache = matches!(n.op, StateOp::Copy { .. });
                ensure!(
                    !(pool_side || raw_into_cache) || bytes == unit,
                    "node `{}` writes {bytes} bytes; the pool's {part:?} unit is {unit}",
                    n.id
                );
                bound.push(BoundStateNode {
                    ssm_layer,
                    part,
                    bytes,
                    op: n.op,
                    conversion,
                });
            }
            by_id.insert(p.id, bound);
        }
        let prefix_hidden_bytes = metrale_circuit::state_ops::prefix_hidden(circuit)
            .map(|i| circuit.states[i].unit_bytes(formats))
            .transpose()?
            .map(usize::try_from)
            .transpose()?;
        Ok(Self {
            by_id,
            recurrent_layers: recurrent_layers.len(),
            prefix_hidden_bytes,
        })
    }

    /// 2026-10-03: Bytes of the prefix cache's last-hidden row, checked against `held`, the
    /// bytes the snapshot pool's row holds: the copy moves the plan's bytes or refuses.
    pub fn prefix_hidden_bytes(&self, held: usize) -> Result<usize> {
        let bytes = self
            .prefix_hidden_bytes
            .context("the circuit declares no prefix-cache hidden row")?;
        ensure!(
            bytes == held,
            "the circuit's prefix hidden row is {bytes} bytes; the snapshot pool holds {held}"
        );
        Ok(bytes)
    }

    /// 2026-10-03: The bound nodes of `id`; an error for a program the circuit leaves empty on
    /// a model with recurrent layers.
    pub fn nodes(&self, id: StateProgramId) -> Result<&[BoundStateNode]> {
        let nodes = self.by_id.get(&id).map(Vec::as_slice).unwrap_or_default();
        ensure!(
            !nodes.is_empty() || self.recurrent_layers == 0,
            "the circuit declares no state for `{}` on this model",
            id.name()
        );
        Ok(nodes)
    }
}

#[cfg(test)]
#[path = "state_bind_tests.rs"]
mod state_bind_tests;
