// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: A group of the plan being compiled ([`GroupRef`]), split from `compile.rs` by
//! exact copy: its members, their edges and the op check an emitter makes first.
//!
//! Owner: model-layers circuit executor.
//! Invariants: see `compile.rs`.

use anyhow::{Context, Result, ensure};
use metrale_circuit::{Circuit, Group};

/// 2026-09-28: A group of the plan being compiled.
pub(crate) struct GroupRef<'a> {
    pub circuit: &'a Circuit,
    pub group: &'a Group,
    pub index: usize,
}

impl<'a> GroupRef<'a> {
    /// 2026-09-28: Member `i`, in pattern order.
    pub fn node(&self, i: usize) -> &'a metrale_circuit::ir::Node {
        &self.circuit.nodes[self.group.nodes[i]]
    }

    /// 2026-09-28: Input `j` of member `i`.
    pub fn input(&self, i: usize, j: usize) -> Result<usize> {
        let n = self.node(i);
        n.inputs
            .get(j)
            .copied()
            .with_context(|| format!("`{}` has no input {j}", n.id))
    }

    /// 2026-09-28: Output `j` of member `i`.
    pub fn output(&self, i: usize, j: usize) -> Result<usize> {
        let n = self.node(i);
        n.outputs
            .get(j)
            .copied()
            .with_context(|| format!("`{}` has no output {j}", n.id))
    }

    /// 2026-09-28: Refuse a group whose members are not exactly `ops`, in order.
    pub fn expect_ops(&self, emitter: &str, ops: &[&str]) -> Result<()> {
        let got: Vec<String> = self
            .group
            .nodes
            .iter()
            .map(|&n| self.circuit.nodes[n].op.name())
            .collect();
        ensure!(
            got.len() == ops.len() && got.iter().zip(ops).all(|(g, w)| g.starts_with(w)),
            "emitter `{emitter}` cannot launch group {} ({:?}); it takes {ops:?}",
            self.index,
            got
        );
        Ok(())
    }
}
