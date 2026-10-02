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
    BlockFile, CircuitError, CircuitFile, LayoutRule, NodeFile, layout_rule, parse_file, when_holds,
};
use crate::format::Format;
use crate::ir::{ArchShape, BlockInstance, Circuit, LayerKind, Node, OpKind, Section};
use crate::precision::{EdgePrecision, LinearFormats};

/// 2026-09-28: Parse `text`, merge the block libraries it includes (`includes`: name to
/// text), and instantiate it for `shape`, asking `precision` for every weight-reading node's
/// formats.
pub fn instantiate(
    text: &str,
    includes: &[(&str, &str)],
    shape: &ArchShape,
    precision: &dyn EdgePrecision,
) -> Result<Circuit, CircuitError> {
    let file = parse_file(text, includes)?;
    let rule = layout_rule(&file.layout)?;
    check_dims(&file, shape)?;
    let sequence = block_sequence(&file, &rule, &shape.layer_kinds, &shape.dims)?;
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
            states: Vec::new(),
        },
        stream: None,
        quantized: BTreeMap::new(),
    };
    for (template, layer, section, module) in sequence {
        b.block(&template, layer, section, module)?;
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

/// 2026-09-30: One block to instantiate: template, layer, section, and a module that replaces
/// the draft module for its `{L}`.
type Planned = (String, Option<usize>, Section, Option<String>);

fn block_sequence(
    file: &CircuitFile,
    rule: &LayoutRule,
    kinds: &[LayerKind],
    dims: &BTreeMap<String, u64>,
) -> Result<Vec<Planned>, CircuitError> {
    if kinds.is_empty() {
        return Err(CircuitError::Layout("the arch shape has no layers".into()));
    }
    let mut used = BTreeSet::new();
    let mut out = Vec::new();
    let known = |name: &str, used: &mut BTreeSet<String>| {
        if !file.block.contains_key(name) {
            return Err(CircuitError::Layout(format!("no block template `{name}`")));
        }
        used.insert(name.to_string());
        Ok(())
    };
    for name in &file.prologue {
        known(name, &mut used)?;
        out.push((name.clone(), None, Section::Main, None));
    }
    // 2026-09-30: The layout blocks per kind, with the overrides of every switch that holds.
    let mut layout = file.layout.blocks.clone();
    for (w, over) in &file.layout.when {
        for names in over.values() {
            for n in names {
                known(n, &mut used)?;
            }
        }
        if when_holds(w, &file.dims, dims)? {
            layout.extend(over.clone());
        }
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
        let blocks = layout.get(kind.name()).ok_or_else(|| {
            CircuitError::Layout(format!(
                "layer {i} is {}, which the layout maps to no blocks",
                kind.name()
            ))
        })?;
        for name in blocks {
            known(name, &mut used)?;
            out.push((name.clone(), Some(i), Section::Main, None));
        }
    }
    for name in &file.epilogue {
        known(name, &mut used)?;
        out.push((name.clone(), None, Section::Main, None));
    }
    let on = match &file.draft_when {
        Some(w) => when_holds(w, &file.dims, dims)?,
        None => true,
    };
    let mut draft = &file.draft;
    for (w, list) in &file.draft_variant {
        for entry in list {
            known(entry.split('@').next().unwrap_or_default(), &mut used)?;
        }
        if when_holds(w, &file.dims, dims)? {
            draft = list;
        }
    }
    for entry in &file.draft {
        known(entry.split('@').next().unwrap_or_default(), &mut used)?;
    }
    if on {
        for entry in draft {
            let (name, module) = match entry.split_once('@') {
                Some((n, m)) => (n.to_string(), Some(m.to_string())),
                None => (entry.clone(), None),
            };
            out.push((name, None, Section::Draft, module));
        }
    }
    if !file.draft.is_empty() && file.draft_module.is_none() {
        return Err(CircuitError::Layout(
            "`draft` blocks need a `draft_module` for their bindings".into(),
        ));
    }
    if let Some(unused) = file.local_blocks.iter().find(|k| !used.contains(*k)) {
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
    /// 2026-09-29: `(edge, format, owning projection of a static scale)` to the edge holding
    /// it quantized to that format.
    quantized: BTreeMap<(usize, Format, Option<String>), usize>,
}

impl Builder<'_> {
    fn block(
        &mut self,
        template: &str,
        layer: Option<usize>,
        section: Section,
        module_override: Option<String>,
    ) -> Result<(), CircuitError> {
        let file = self.file;
        let tpl: &BlockFile = &file.block[template];
        let (prefix, module) = match (layer, section) {
            (Some(i), _) => (
                format!("l{i}.{template}"),
                Some(file.layer_module.replace("{i}", &i.to_string())),
            ),
            (None, Section::Draft) => (
                format!("draft.{template}"),
                module_override.or_else(|| file.draft_module.clone()),
            ),
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
        let first_state = self.circuit.states.len();
        for sf in &tpl.state {
            let decl = state_decl(
                template,
                &prefix,
                layer,
                section,
                sf,
                &tpl.state,
                &self.shape.dims,
            )?;
            if self.circuit.states.iter().any(|s| s.id == decl.id) {
                return Err(dup(template, &sf.id));
            }
            self.circuit.states.push(decl);
        }
        let mut node_ids = BTreeSet::new();
        for nf in &tpl.node {
            if !node_ids.insert(nf.id.as_str()) {
                return Err(dup(template, &nf.id));
            }
            self.node(template, &prefix, layer, module.as_deref(), nf, &mut local)?;
        }
        for out in &tpl.outputs {
            let name = out.edge();
            let e = *local.get(name).ok_or_else(|| {
                CircuitError::Layout(format!(
                    "block `{template}` declares output `{name}`, which it does not produce"
                ))
            })?;
            self.circuit.edges[e].is_output = true;
            if let crate::circuit_toml::OutputFile::Bound { buffer, .. } = out {
                let b = crate::model_buffer::ModelBuffer::parse(buffer).ok_or_else(|| {
                    CircuitError::Layout(format!(
                        "block `{template}` output `{name}`: `{buffer}` is no model buffer \
                         (logits, tokens, draft_embed)"
                    ))
                })?;
                self.circuit.edges[e].binds = Some(b);
            }
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
        check_state_access(template, &self.circuit, first_state, first)?;
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
        if let Some(w) = &nf.when
            && !when_holds(w, &self.file.dims, &self.shape.dims)?
        {
            return self.elide(template, nf, local);
        }
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
        let mut inputs = Vec::with_capacity(nf.inputs.len());
        for name in &nf.inputs {
            let e = *local.get(name).ok_or_else(|| CircuitError::DanglingInput {
                block: template.to_string(),
                node: nf.id.clone(),
                edge: name.clone(),
            })?;
            inputs.push(e);
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
        let formats = match nf.unquantized {
            true => edges::unquantized(&id, &op, inputs.first().map(|&x| self.circuit.edges[x].format))?,
            false => self.resolve(&id, &op, &binding, &inputs)?,
        };
        if let Some(f) = formats {
            let x = inputs[0];
            let have = self.circuit.edges[x].format;
            // 2026-09-30: A projection whose precision reads F32 where the circuit has a 16-bit
            // elementwise product (the grouped FP8 expert kernels read the FP32 SiLU product)
            // makes its producer write F32, while it is the edge's only reader. A 16-bit edge
            // into a projection declared with quantized activations gets the quantizer node.
            if have == Format::Bf16 && f.activation == Format::F32 && self.widenable(x) {
                self.circuit.edges[x].format = Format::F32;
            } else if have != f.activation {
                let plain_source = matches!(have, Format::Bf16 | Format::F32);
                if !plain_source || f.activation.is_plain() || !f.activation.is_edge_format() {
                    return Err(self.mismatch(&id, &op, x, Some(f.activation)));
                }
                inputs[0] =
                    self.quantized(x, f.activation, template, layer, (&nf.id, binding.first()));
            }
        }
        let idx = self.circuit.nodes.len();
        for &e in &inputs {
            self.circuit.edges[e].consumers.push(idx);
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
        let weight = formats.map(|f| f.weight);
        for &e in &inputs {
            let f = self.circuit.edges[e].format;
            if weight.is_none() && !f.is_plain() && !matches!(op, OpKind::Copy) {
                return Err(self.mismatch(&id, &op, e, None));
            }
        }
        let state = self.state_refs(template, prefix, nf)?;
        self.circuit.nodes.push(Node {
            id,
            local: nf.id.clone(),
            op,
            inputs,
            outputs,
            weight,
            binding,
            params: self.params(nf)?,
            layer,
            block: template.to_string(),
            state,
        });
        Ok(())
    }

    fn resolve(
        &self,
        id: &str,
        op: &OpKind,
        binding: &[String],
        inputs: &[usize],
    ) -> Result<Option<LinearFormats>, CircuitError> {
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
        if inputs.is_empty() {
            return Err(CircuitError::Binding {
                node: id.to_string(),
                detail: "a weight-reading node needs an activation input".into(),
            });
        }
        Ok(Some(formats))
    }

    /// 2026-09-30: A node's params: its own, plus those of each `params_when` switch that
    /// holds, with every `{dim}` replaced by the dim's value.
    fn params(&self, nf: &NodeFile) -> Result<BTreeMap<String, String>, CircuitError> {
        let mut out = nf.params.clone();
        for (w, extra) in &nf.params_when {
            if when_holds(w, &self.file.dims, &self.shape.dims)? {
                out.extend(extra.clone());
            }
        }
        for v in out.values_mut() {
            if let Some(name) = v.strip_prefix('{').and_then(|r| r.strip_suffix('}')) {
                let d = self.shape.dims.get(name).ok_or_else(|| {
                    CircuitError::ShapeMismatch(format!(
                        "node `{}` param `{v}` names dim `{name}`, which the arch shape lacks",
                        nf.id
                    ))
                })?;
                *v = d.to_string();
            }
        }
        Ok(out)
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

#[path = "instantiate/edges.rs"]
mod edges;
mod states;
use states::check_state_access;
pub(crate) use states::state_decl;

#[cfg(test)]
#[path = "instantiate_tests.rs"]
mod instantiate_tests;
