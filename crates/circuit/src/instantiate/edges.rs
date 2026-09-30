// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: The edge half of the instantiation builder: declaring an edge, the inserted
//! activation quantizers, an absent (`when`) node's identity, and the format checks.
//!
//! Owner: metrale-circuit.
//! Invariants: see [`crate::circuit_toml`].

use std::collections::BTreeMap;

use super::{Builder, dup};
use crate::circuit_toml::{CircuitError, NodeFile, split_shape};
use crate::dims::DimExpr;
use crate::format::{Format, Scale};
use crate::ir::{Edge, Node, OpKind};

impl Builder<'_> {
    /// 2026-09-30: An absent node: its one output names its one input, which must have the
    /// output's format and width.
    pub(super) fn elide(
        &mut self,
        template: &str,
        nf: &NodeFile,
        local: &mut BTreeMap<String, usize>,
    ) -> Result<(), CircuitError> {
        let bad = |detail: String| {
            CircuitError::Layout(format!(
                "block `{template}` node `{}` (absent under `{}`): {detail}",
                nf.id,
                nf.when.as_deref().unwrap_or_default()
            ))
        };
        let ([input], [out]) = (nf.inputs.as_slice(), nf.out.as_slice()) else {
            return Err(bad(
                "an optional node must have one input and one output".into()
            ));
        };
        let e = *local
            .get(input)
            .ok_or_else(|| CircuitError::DanglingInput {
                block: template.to_string(),
                node: nf.id.clone(),
                edge: input.clone(),
            })?;
        if local.contains_key(&out.edge) {
            return Err(dup(template, &out.edge));
        }
        let format = Format::parse(&out.format).map_err(|source| CircuitError::Format {
            block: template.to_string(),
            edge: out.edge.clone(),
            source,
        })?;
        let (_, dim) = split_shape(&out.shape)
            .ok_or_else(|| bad(format!("shape `{}` is not `<rows> x <dim>`", out.shape)))?;
        let width = DimExpr::parse(dim)
            .and_then(|d| d.eval(&self.shape.dims))
            .map_err(|source| CircuitError::Dim {
                block: template.to_string(),
                source,
            })?;
        let src = &self.circuit.edges[e];
        if src.format != format || src.dim_value != width {
            return Err(bad(format!(
                "its output `{}` ({} x {width}) is not its input `{}` ({} x {})",
                out.edge,
                format.name(),
                src.id,
                src.format.name(),
                src.dim_value
            )));
        }
        local.insert(out.edge.clone(), e);
        Ok(())
    }

    pub(super) fn edge(
        &mut self,
        template: &str,
        prefix: &str,
        producer: usize,
        ef: &crate::circuit_toml::EdgeFile,
    ) -> Result<usize, CircuitError> {
        let block = template.to_string();
        let format = Format::parse(&ef.format).map_err(|source| CircuitError::Format {
            block: block.clone(),
            edge: ef.edge.clone(),
            source,
        })?;
        if !format.is_edge_format() {
            return Err(CircuitError::WeightFormatOnEdge {
                block,
                edge: ef.edge.clone(),
                format: format.name(),
            });
        }
        let shape_err = |detail: String| CircuitError::Shape {
            block: template.to_string(),
            edge: ef.edge.clone(),
            detail,
        };
        let (rows, dim) = split_shape(&ef.shape)
            .ok_or_else(|| shape_err(format!("shape `{}` is not `<rows> x <dim>`", ef.shape)))?;
        let rows = DimExpr::parse(rows).map_err(|e| shape_err(e.to_string()))?;
        let dim = DimExpr::parse(dim).map_err(|e| shape_err(e.to_string()))?;
        if !rows.names().any(|n| n == "n") {
            return Err(shape_err(format!(
                "rows `{}` must scale with `n`",
                rows.text()
            )));
        }
        let mut with_n = self.shape.dims.clone();
        with_n.insert("n".into(), 1);
        for expr in [&rows, &dim] {
            if let Some(bad) = expr
                .names()
                .find(|n| *n != "n" && !self.file.dims.iter().any(|d| d == n))
            {
                return Err(shape_err(format!(
                    "`{bad}` is not in the circuit's `dims` list"
                )));
            }
            expr.eval(&with_n).map_err(|source| CircuitError::Dim {
                block: template.to_string(),
                source,
            })?;
        }
        let dim_value = dim
            .eval(&self.shape.dims)
            .map_err(|source| CircuitError::Dim {
                block: template.to_string(),
                source,
            })?;
        self.circuit.edges.push(Edge {
            id: format!("{prefix}.{}", ef.edge),
            format,
            rows,
            dim,
            dim_value,
            producer: Some(producer),
            consumers: Vec::new(),
            is_output: false,
        });
        Ok(self.circuit.edges.len() - 1)
    }

    /// 2026-09-29: The formats a weight-reading node's binding resolves to; `None` for any
    /// other node.
    /// 2026-09-29: `x` quantized to `format`: the output of the `act_quant` node that quantizes
    /// it, inserted before the first projection that reads `x` in that format and shared by
    /// every later one. 2026-09-30: A static per-tensor activation scale belongs to one
    /// projection (its `<module>.input_scale`), so that quantizer is bound to it and not shared.
    pub(super) fn quantized(
        &mut self,
        x: usize,
        format: Format,
        template: &str,
        layer: Option<usize>,
        consumer: (&str, Option<&String>),
    ) -> usize {
        let static_scale = matches!(
            format,
            Format::Fp8E4m3 {
                scale: Scale::PerTensor
            }
        );
        let owner = static_scale.then(|| consumer.0.to_string());
        let key = (x, format, owner.clone());
        if let Some(&q) = self.quantized.get(&key) {
            return q;
        }
        let src = self.circuit.edges[x].clone();
        let (prefix, edge_local) = src.id.rsplit_once('.').unwrap_or(("", src.id.as_str()));
        let local = match &owner {
            Some(c) => format!("{c}_quant"),
            None => format!("{edge_local}_quant"),
        };
        let binding = match (static_scale, consumer.1) {
            (true, Some(module)) => vec![format!("{module}.input_scale")],
            _ => Vec::new(),
        };
        let node = self.circuit.nodes.len();
        self.circuit.edges[x].consumers.push(node);
        self.circuit.edges.push(Edge {
            id: format!("{prefix}.{local}"),
            format,
            rows: src.rows,
            dim: src.dim,
            dim_value: src.dim_value,
            producer: Some(node),
            consumers: Vec::new(),
            is_output: false,
        });
        let q = self.circuit.edges.len() - 1;
        self.circuit.nodes.push(Node {
            id: format!("{prefix}.{local}"),
            local,
            op: OpKind::ActQuant(format),
            inputs: vec![x],
            outputs: vec![q],
            weight: None,
            binding,
            params: Default::default(),
            layer,
            block: template.to_string(),
            state: Vec::new(),
        });
        self.quantized.insert(key, q);
        q
    }

    /// 2026-09-30: Whether edge `e` may be widened to F32: an elementwise product (SiLU·mul,
    /// ReLU²) no node reads yet.
    pub(super) fn widenable(&self, e: usize) -> bool {
        let edge = &self.circuit.edges[e];
        edge.consumers.is_empty()
            && edge.producer.is_some_and(|p| {
                matches!(self.circuit.nodes[p].op, OpKind::SiluMul | OpKind::Relu2)
            })
    }

    pub(super) fn expect(
        &self,
        id: &str,
        op: &OpKind,
        e: usize,
        want: Format,
    ) -> Result<(), CircuitError> {
        if self.circuit.edges[e].format == want {
            Ok(())
        } else {
            Err(self.mismatch(id, op, e, Some(want)))
        }
    }

    pub(super) fn mismatch(
        &self,
        id: &str,
        op: &OpKind,
        e: usize,
        want: Option<Format>,
    ) -> CircuitError {
        let edge = &self.circuit.edges[e];
        CircuitError::FormatMismatch {
            node: id.to_string(),
            op: op.name(),
            edge: edge.id.clone(),
            format: edge.format.name(),
            expected: want.map(|f| f.name()),
        }
    }
}
