// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: Instantiate a circuit TOML against an arch shape and a precision resolver.
//!
//! The prologue blocks run first, then each layer's blocks in the order its kind maps to,
//! then the epilogue, then the draft blocks. A block's `stream_in` binds to the edge the
//! previous stream-writing block left, so layer `i`'s output edge IS layer `i + 1`'s input
//! edge and a rule can fuse across the boundary.
//!
//! Owner: metrale-circuit.
//! Invariants: see [`crate::circuit_toml`]; every check there is enforced here, in file
//! order, and the first failure is returned.

use std::collections::{BTreeMap, BTreeSet};

use crate::circuit_toml::{
    BlockFile, CircuitError, CircuitFile, LayoutRule, NodeFile, layout_rule, parse_file,
    split_shape,
};
use crate::dims::DimExpr;
use crate::format::Format;
use crate::ir::{ArchShape, BlockInstance, Circuit, Edge, LayerKind, Node, OpKind, Section};
use crate::precision::{EdgePrecision, LinearFormats};

/// 2026-09-28: Parse `text` and instantiate it for `shape`, asking `precision` for every
/// weight-reading node's formats.
pub fn instantiate(
    text: &str,
    shape: &ArchShape,
    precision: &dyn EdgePrecision,
) -> Result<Circuit, CircuitError> {
    let file = parse_file(text)?;
    let rule = layout_rule(&file.layout)?;
    check_dims(&file, shape)?;
    let sequence = block_sequence(&file, &rule, &shape.layer_kinds)?;
    let mut b = Builder {
        file: &file,
        shape,
        precision,
        circuit: Circuit {
            arch: file.arch.clone(),
            description: file.description.clone(),
            nodes: Vec::new(),
            edges: Vec::new(),
            blocks: Vec::new(),
            layer_kinds: shape.layer_kinds.clone(),
            dims: shape.dims.clone(),
        },
        stream: None,
    };
    for (template, layer, section) in sequence {
        b.block(&template, layer, section)?;
    }
    b.finish()
}

fn check_dims(file: &CircuitFile, shape: &ArchShape) -> Result<(), CircuitError> {
    for d in &file.dims {
        if !shape.dims.contains_key(d) {
            return Err(CircuitError::ShapeMismatch(format!(
                "circuit `{}` needs dim `{d}`, which the arch shape does not give",
                file.arch
            )));
        }
    }
    Ok(())
}

type Planned = (String, Option<usize>, Section);

fn block_sequence(
    file: &CircuitFile,
    rule: &LayoutRule,
    kinds: &[LayerKind],
) -> Result<Vec<Planned>, CircuitError> {
    if kinds.is_empty() {
        return Err(CircuitError::Layout("the arch shape has no layers".into()));
    }
    let mut used = BTreeSet::new();
    let mut out = Vec::new();
    let known = |name: &String, used: &mut BTreeSet<String>| {
        if !file.block.contains_key(name) {
            return Err(CircuitError::Layout(format!("no block template `{name}`")));
        }
        used.insert(name.clone());
        Ok(())
    };
    for name in &file.prologue {
        known(name, &mut used)?;
        out.push((name.clone(), None, Section::Main));
    }
    for (i, kind) in kinds.iter().enumerate() {
        if let LayoutRule::Interval { period } = rule {
            let want = if (i + 1) % period == 0 {
                LayerKind::FullAttention
            } else {
                LayerKind::LinearAttention
            };
            if *kind != want {
                return Err(CircuitError::Layout(format!(
                    "layer {i} is {} but the interval-{period} layout puts {} there",
                    kind.name(),
                    want.name()
                )));
            }
        }
        let blocks = file.layout.blocks.get(kind.name()).ok_or_else(|| {
            CircuitError::Layout(format!(
                "layer {i} is {}, which the layout maps to no blocks",
                kind.name()
            ))
        })?;
        for name in blocks {
            known(name, &mut used)?;
            out.push((name.clone(), Some(i), Section::Main));
        }
    }
    for name in &file.epilogue {
        known(name, &mut used)?;
        out.push((name.clone(), None, Section::Main));
    }
    for name in &file.draft {
        known(name, &mut used)?;
        out.push((name.clone(), None, Section::Draft));
    }
    if !file.draft.is_empty() && file.draft_module.is_none() {
        return Err(CircuitError::Layout(
            "`draft` blocks need a `draft_module` for their bindings".into(),
        ));
    }
    if let Some(unused) = file.block.keys().find(|k| !used.contains(*k)) {
        return Err(CircuitError::Layout(format!(
            "block template `{unused}` is never used"
        )));
    }
    Ok(out)
}

struct Builder<'a> {
    file: &'a CircuitFile,
    shape: &'a ArchShape,
    precision: &'a dyn EdgePrecision,
    circuit: Circuit,
    stream: Option<usize>,
}

impl Builder<'_> {
    fn block(
        &mut self,
        template: &str,
        layer: Option<usize>,
        section: Section,
    ) -> Result<(), CircuitError> {
        let file = self.file;
        let tpl: &BlockFile = &file.block[template];
        let (prefix, module) = match (layer, section) {
            (Some(i), _) => (
                format!("l{i}.{template}"),
                Some(file.layer_module.replace("{i}", &i.to_string())),
            ),
            (None, Section::Draft) => (format!("draft.{template}"), file.draft_module.clone()),
            (None, Section::Main) => (template.to_string(), None),
        };
        let first = self.circuit.nodes.len();
        let mut local: BTreeMap<String, usize> = BTreeMap::new();
        let stream_in = match &tpl.stream_in {
            Some(name) => {
                let edge = self.stream.ok_or_else(|| {
                    CircuitError::Layout(format!(
                        "block `{template}` reads the stream before any block writes it"
                    ))
                })?;
                local.insert(name.clone(), edge);
                Some(edge)
            }
            None => None,
        };
        let mut node_ids = BTreeSet::new();
        for nf in &tpl.node {
            if !node_ids.insert(nf.id.as_str()) {
                return Err(dup(template, &nf.id));
            }
            self.node(template, &prefix, layer, module.as_deref(), nf, &mut local)?;
        }
        for out in &tpl.outputs {
            let e = *local.get(out).ok_or_else(|| {
                CircuitError::Layout(format!(
                    "block `{template}` declares output `{out}`, which it does not produce"
                ))
            })?;
            self.circuit.edges[e].is_output = true;
        }
        let stream_out = match &tpl.stream_out {
            Some(name) => {
                let e = *local.get(name).ok_or_else(|| {
                    CircuitError::Layout(format!(
                        "block `{template}` writes stream `{name}`, which it does not produce"
                    ))
                })?;
                self.stream = Some(e);
                Some(e)
            }
            None => None,
        };
        self.circuit.blocks.push(BlockInstance {
            template: template.to_string(),
            layer,
            section,
            first,
            end: self.circuit.nodes.len(),
            stream_in,
            stream_out,
        });
        Ok(())
    }

    fn node(
        &mut self,
        template: &str,
        prefix: &str,
        layer: Option<usize>,
        module: Option<&str>,
        nf: &NodeFile,
        local: &mut BTreeMap<String, usize>,
    ) -> Result<(), CircuitError> {
        let fmt = nf
            .format
            .as_deref()
            .map(Format::parse)
            .transpose()
            .map_err(|source| CircuitError::Format {
                block: template.to_string(),
                edge: nf.id.clone(),
                source,
            })?;
        let op =
            OpKind::parse(&nf.op, nf.role.as_deref(), fmt).map_err(|source| CircuitError::Op {
                block: template.to_string(),
                node: nf.id.clone(),
                source,
            })?;
        let id = format!("{prefix}.{}", nf.id);
        let idx = self.circuit.nodes.len();
        let mut inputs = Vec::with_capacity(nf.inputs.len());
        for name in &nf.inputs {
            let e = *local.get(name).ok_or_else(|| CircuitError::DanglingInput {
                block: template.to_string(),
                node: nf.id.clone(),
                edge: name.clone(),
            })?;
            self.circuit.edges[e].consumers.push(idx);
            inputs.push(e);
        }
        let mut outputs = Vec::with_capacity(nf.out.len());
        for ef in &nf.out {
            if local.contains_key(&ef.edge) {
                return Err(dup(template, &ef.edge));
            }
            let e = self.edge(template, prefix, idx, ef)?;
            local.insert(ef.edge.clone(), e);
            outputs.push(e);
        }
        if let OpKind::ActQuant(f) = op {
            for &o in &outputs {
                self.expect(&id, &op, o, f)?;
            }
        }
        let mut binding = Vec::with_capacity(nf.binding.len());
        for b in &nf.binding {
            binding.push(match (b.contains("{L}"), module) {
                (true, Some(m)) => b.replace("{L}", m),
                (true, None) => {
                    return Err(CircuitError::Binding {
                        node: id.clone(),
                        detail: format!("`{b}` names {{L}} outside a layer or draft block"),
                    });
                }
                (false, _) => b.clone(),
            });
        }
        let weight = self.resolve(&id, &op, &binding, &inputs)?;
        for &e in &inputs {
            let f = self.circuit.edges[e].format;
            if weight.is_none() && !f.is_plain() && !matches!(op, OpKind::Copy) {
                return Err(self.mismatch(&id, &op, e, None));
            }
        }
        self.circuit.nodes.push(Node {
            id,
            local: nf.id.clone(),
            op,
            inputs,
            outputs,
            weight,
            binding,
            params: nf.params.clone(),
            layer,
            block: template.to_string(),
        });
        Ok(())
    }

    fn edge(
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

    fn resolve(
        &self,
        id: &str,
        op: &OpKind,
        binding: &[String],
        inputs: &[usize],
    ) -> Result<Option<Format>, CircuitError> {
        if !op.reads_linear_weight() {
            return Ok(None);
        }
        let Some(first) = binding.first() else {
            return Err(CircuitError::Binding {
                node: id.to_string(),
                detail: format!("`{}` reads a weight but binds no module", op.name()),
            });
        };
        // 2026-09-28: Experts share one declared precision; `*` is asked as expert 0.
        let ask = |m: &str| self.precision.linear(&m.replace('*', "0"));
        let formats: LinearFormats = ask(first);
        for m in &binding[1..] {
            let other = ask(m);
            if other != formats {
                return Err(CircuitError::MixedPrecision {
                    node: id.to_string(),
                    detail: format!(
                        "{first} = {}/{}, {m} = {}/{}",
                        formats.weight, formats.activation, other.weight, other.activation
                    ),
                });
            }
        }
        let Some(&x) = inputs.first() else {
            return Err(CircuitError::Binding {
                node: id.to_string(),
                detail: "a weight-reading node needs an activation input".into(),
            });
        };
        self.expect(id, op, x, formats.activation)?;
        Ok(Some(formats.weight))
    }

    fn expect(&self, id: &str, op: &OpKind, e: usize, want: Format) -> Result<(), CircuitError> {
        if self.circuit.edges[e].format == want {
            Ok(())
        } else {
            Err(self.mismatch(id, op, e, Some(want)))
        }
    }

    fn mismatch(&self, id: &str, op: &OpKind, e: usize, want: Option<Format>) -> CircuitError {
        let edge = &self.circuit.edges[e];
        CircuitError::FormatMismatch {
            node: id.to_string(),
            op: op.name(),
            edge: edge.id.clone(),
            format: edge.format.name(),
            expected: want.map(|f| f.name()),
        }
    }

    fn finish(self) -> Result<Circuit, CircuitError> {
        for e in &self.circuit.edges {
            if e.consumers.is_empty() && !e.is_output {
                return Err(CircuitError::DanglingOutput(e.id.clone()));
            }
        }
        Ok(self.circuit)
    }
}

fn dup(block: &str, name: &str) -> CircuitError {
    CircuitError::Duplicate {
        block: block.to_string(),
        name: name.to_string(),
    }
}

#[cfg(test)]
#[path = "instantiate_tests.rs"]
mod instantiate_tests;
