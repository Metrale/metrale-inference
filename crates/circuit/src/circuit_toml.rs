// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: The `kernels/circuits/<arch>.toml` schema: block templates, the layout rule
//! that maps a model's layer kinds to templates, and the typed load errors.
//!
//! Owner: metrale-circuit.
//! Invariants:
//! - Every edge states its format and its shape (`"n x hidden"`); nothing defaults.
//! - Loading is fail-fast: the first unknown op, dangling or duplicate edge, format no
//!   consumer accepts, or template/layout mismatch is returned as a [`CircuitError`].

use std::collections::BTreeMap;

use serde::Deserialize;

use crate::dims::DimError;
use crate::format::FormatError;
use crate::ir::OpParseError;

/// 2026-09-28: Why a circuit did not load or instantiate.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CircuitError {
    /// 2026-09-28: Not valid TOML, or not the schema.
    #[error("circuit TOML: {0}")]
    Parse(String),
    /// 2026-09-28: An op the vocabulary does not have, or a bad role or format qualifier.
    #[error("block `{block}` node `{node}`: {source}")]
    Op {
        /// 2026-09-28: Template.
        block: String,
        /// 2026-09-28: Node id.
        node: String,
        /// 2026-09-28: The op error.
        source: OpParseError,
    },
    /// 2026-09-28: A format spelling no format matches.
    #[error("block `{block}` edge `{edge}`: {source}")]
    Format {
        /// 2026-09-28: Template.
        block: String,
        /// 2026-09-28: Edge.
        edge: String,
        /// 2026-09-28: The format error.
        source: FormatError,
    },
    /// 2026-09-28: A weight-only scale layout (`channel`, `block`) on an edge.
    #[error("block `{block}` edge `{edge}`: `{format}` is a weight layout, not an edge format")]
    WeightFormatOnEdge {
        /// 2026-09-28: Template.
        block: String,
        /// 2026-09-28: Edge.
        edge: String,
        /// 2026-09-28: The format.
        format: String,
    },
    /// 2026-09-28: A shape that is not `<rows> x <dim>`, or a bad expression in it.
    #[error("block `{block}` edge `{edge}`: {detail}")]
    Shape {
        /// 2026-09-28: Template.
        block: String,
        /// 2026-09-28: Edge.
        edge: String,
        /// 2026-09-28: What was wrong.
        detail: String,
    },
    /// 2026-09-30: A state declaration that does not parse or evaluate.
    #[error("block `{block}` state `{state}`: {detail}")]
    State {
        /// 2026-09-30: Template.
        block: String,
        /// 2026-09-30: State id.
        state: String,
        /// 2026-09-30: What was wrong.
        detail: String,
    },
    /// 2026-09-28: A dimension the arch shape does not define, or an overflow.
    #[error("block `{block}`: {source}")]
    Dim {
        /// 2026-09-28: Template.
        block: String,
        /// 2026-09-28: The dimension error.
        source: DimError,
    },
    /// 2026-09-28: A node reads an edge nothing produced.
    #[error("block `{block}` node `{node}` reads `{edge}`, which no earlier node produces")]
    DanglingInput {
        /// 2026-09-28: Template.
        block: String,
        /// 2026-09-28: Node id.
        node: String,
        /// 2026-09-28: Edge.
        edge: String,
    },
    /// 2026-09-28: An edge nothing reads that is not a declared output.
    #[error("edge `{0}` is produced but never read, and is not a declared output")]
    DanglingOutput(String),
    /// 2026-09-28: Two producers of one edge, or two nodes with one id.
    #[error("block `{block}`: `{name}` is defined twice")]
    Duplicate {
        /// 2026-09-28: Template.
        block: String,
        /// 2026-09-28: Edge or node id.
        name: String,
    },
    /// 2026-09-28: An edge whose format a reader does not accept.
    #[error(
        "node `{node}` ({op}) does not accept `{edge}` as {format}{}",
        expected.as_ref().map(|e| format!(" (expects {e})")).unwrap_or_default()
    )]
    FormatMismatch {
        /// 2026-09-28: The reading node.
        node: String,
        /// 2026-09-28: Its op.
        op: String,
        /// 2026-09-28: The edge.
        edge: String,
        /// 2026-09-28: The edge's format.
        format: String,
        /// 2026-09-28: What the reader expects, when it is one format.
        expected: Option<String>,
    },
    /// 2026-09-28: The modules of one node resolve to different formats.
    #[error("node `{node}`: bound modules resolve to different formats ({detail})")]
    MixedPrecision {
        /// 2026-09-28: Node id.
        node: String,
        /// 2026-09-28: The two resolutions.
        detail: String,
    },
    /// 2026-09-28: A weight-reading node without a binding, or a binding on a node that
    /// reads no weight but is not a norm.
    #[error("node `{node}`: {detail}")]
    Binding {
        /// 2026-09-28: Node id.
        node: String,
        /// 2026-09-28: What was wrong.
        detail: String,
    },
    /// 2026-09-28: A layout that names a missing template, or a model whose layers the
    /// layout does not describe.
    #[error("layout: {0}")]
    Layout(String),
    /// 2026-09-28: An included block file that was not supplied, does not parse, or defines a
    /// block another file also defines.
    #[error("include `{name}`: {detail}")]
    Include {
        /// 2026-09-28: The include's name.
        name: String,
        /// 2026-09-28: What was wrong.
        detail: String,
    },
    /// 2026-09-28: A dim the circuit requires that the arch shape does not give, or one it
    /// gives that the circuit does not declare.
    #[error("arch shape: {0}")]
    ShapeMismatch(String),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CircuitFile {
    pub schema: u32,
    pub arch: String,
    pub description: String,
    pub layer_module: String,
    pub include: Vec<String>,
    pub dims: Vec<String>,
    pub layout: LayoutFile,
    pub prologue: Vec<String>,
    pub epilogue: Vec<String>,
    pub draft: Vec<String>,
    pub draft_module: Option<String>,
    /// 2026-09-30: The draft blocks are instantiated only when this switch holds
    /// ([`when_holds`]); absent, always.
    pub draft_when: Option<String>,
    /// 2026-09-30: Per switch, a draft block list replacing `draft` while it holds. A draft
    /// entry `template@module` instantiates `template` with `{L}` = `module` (one template
    /// serves a layer and a draft module).
    #[serde(default)]
    pub draft_variant: BTreeMap<String, Vec<String>>,
    pub block: BTreeMap<String, BlockFile>,
    /// 2026-09-30: The blocks the circuit file itself defines (not its libraries'); only these
    /// must all be used.
    #[serde(skip)]
    pub local_blocks: std::collections::BTreeSet<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct LayoutFile {
    pub kind: String,
    pub period: Option<usize>,
    pub blocks: BTreeMap<String, Vec<String>>,
    /// 2026-09-30: Per switch ([`when_holds`]), layer kinds whose blocks are replaced while it
    /// holds (`moe_latent = { moe = ["moe_latent"] }`).
    #[serde(default)]
    pub when: BTreeMap<String, BTreeMap<String, Vec<String>>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct BlockFile {
    pub stream_in: Option<String>,
    pub stream_out: Option<String>,
    #[serde(default)]
    pub outputs: Vec<OutputFile>,
    pub node: Vec<NodeFile>,
    /// 2026-09-30: The state the block keeps between steps (`crate::state`).
    #[serde(default)]
    pub state: Vec<StateFile>,
}

/// 2026-09-30: A declared output: the edge, and (in a served circuit) the model buffer it binds
/// to (`crate::model_buffer::ModelBuffer`). A bare edge name declares an output with no buffer.
#[derive(Deserialize)]
#[serde(untagged)]
pub(crate) enum OutputFile {
    Bare(String),
    Bound { edge: String, buffer: String },
}

impl OutputFile {
    /// 2026-09-30: The edge it names.
    pub fn edge(&self) -> &str {
        match self {
            Self::Bare(e) | Self::Bound { edge: e, .. } => e,
        }
    }
}

/// 2026-09-30: One `[[block.<name>.state]]`: `kind` (`recurrent` | `paged_kv`, 2026-10-02: or a
/// cache kind, `crate::state::StateKind`), `format` (a dtype or a `{key}` the plan inputs give),
/// `shape` (axes of dim expressions joined by ` x `: one slot of a recurrent state, one token of
/// a KV side, one unit of a cache), and for a recurrent state the verify intermediates it keeps
/// (`h_steps` | `conv_steps`). 2026-10-02: A snapshot kind names the state of its block it copies
/// (`of`) instead of a shape, so the copy cannot be sized apart from the original.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct StateFile {
    pub id: String,
    pub kind: String,
    /// 2026-09-30: `sequence` (recurrent) or `model` (a KV pool); see `crate::state::Lifetime`.
    pub lifetime: String,
    pub format: String,
    pub shape: Option<String>,
    pub verify: Option<String>,
    pub of: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct NodeFile {
    pub id: String,
    pub op: String,
    /// 2026-09-30: The block states the node touches, by state id (`crate::state::StateAccess`).
    #[serde(default)]
    pub state: BTreeMap<String, String>,
    pub role: Option<String>,
    pub format: Option<String>,
    #[serde(default, rename = "in")]
    pub inputs: Vec<String>,
    pub out: Vec<EdgeFile>,
    #[serde(default)]
    pub binding: Vec<String>,
    #[serde(default)]
    pub params: BTreeMap<String, String>,
    /// 2026-09-30: The node is present only when this switch holds ([`when_holds`]). An absent
    /// node has one input and one output of the same format and width, and its output IS its
    /// input (the identity it degenerates to: a norm, bias or projection the checkpoint lacks).
    pub when: Option<String>,
    /// 2026-09-30: Params merged in when their switch holds (`attn_bias = { bias = "true" }`).
    #[serde(default)]
    pub params_when: BTreeMap<String, BTreeMap<String, String>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct EdgeFile {
    pub edge: String,
    pub format: String,
    pub shape: String,
}

/// 2026-09-28: The layer-kind rule a circuit declares.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayoutRule {
    /// 2026-09-28: `full_attention` on every layer `i` with `(i + 1) % period == 0`,
    /// `linear_attention` elsewhere; the HF `full_attention_interval` convention.
    Interval {
        /// 2026-09-28: The interval.
        period: usize,
    },
    /// 2026-09-28: Any sequence of the kinds `blocks` maps.
    List,
}

/// 2026-09-28: A block library: `kernels/circuits/blocks/<name>.toml`, holding only block
/// templates that several circuits share.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BlocksFile {
    schema: u32,
    #[allow(dead_code)]
    description: String,
    block: BTreeMap<String, BlockFile>,
}

/// 2026-09-28: The block libraries a circuit TOML includes, in its order, so the caller knows
/// which files to read.
pub fn includes_of(text: &str) -> Result<Vec<String>, CircuitError> {
    Ok(parse_one(text)?.include)
}

fn parse_one(text: &str) -> Result<CircuitFile, CircuitError> {
    let file: CircuitFile = toml::from_str(text).map_err(|e| CircuitError::Parse(e.to_string()))?;
    if file.schema != 1 {
        return Err(CircuitError::Parse(format!(
            "schema {} (this build reads 1)",
            file.schema
        )));
    }
    Ok(file)
}

/// 2026-09-28: Parse a circuit and merge in its includes. `includes` maps an include name to
/// its text; one not supplied, a library that does not parse, and a block defined twice are
/// errors, never an override.
pub(crate) fn parse_file(
    text: &str,
    includes: &[(&str, &str)],
) -> Result<CircuitFile, CircuitError> {
    let mut file = parse_one(text)?;
    file.local_blocks = file.block.keys().cloned().collect();
    for name in file.include.clone() {
        let err = |detail: String| CircuitError::Include {
            name: name.clone(),
            detail,
        };
        let lib_text = includes
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, t)| *t)
            .ok_or_else(|| err("not supplied".into()))?;
        let lib: BlocksFile = toml::from_str(lib_text).map_err(|e| err(e.to_string()))?;
        if lib.schema != 1 {
            return Err(err(format!("schema {} (this build reads 1)", lib.schema)));
        }
        for (block, tpl) in lib.block {
            if file.block.contains_key(&block) {
                return Err(err(format!("block `{block}` is defined twice")));
            }
            file.block.insert(block, tpl);
        }
    }
    Ok(file)
}

pub(crate) fn layout_rule(l: &LayoutFile) -> Result<LayoutRule, CircuitError> {
    match (l.kind.as_str(), l.period) {
        ("interval", Some(p)) if p > 0 => Ok(LayoutRule::Interval { period: p }),
        ("interval", _) => Err(CircuitError::Layout(
            "`interval` needs a `period` of at least 1".into(),
        )),
        ("list", None) => Ok(LayoutRule::List),
        ("list", Some(_)) => Err(CircuitError::Layout("`list` takes no `period`".into())),
        (other, _) => Err(CircuitError::Layout(format!(
            "unknown layout kind `{other}` (interval | list)"
        ))),
    }
}

/// 2026-09-30: Whether switch `expr` holds: `name` when the dim `name` is non-zero, `!name`
/// when it is zero. The dim must be in the circuit's `dims` list.
pub(crate) fn when_holds(
    expr: &str,
    declared: &[String],
    dims: &BTreeMap<String, u64>,
) -> Result<bool, CircuitError> {
    let (negate, name) = match expr.trim().strip_prefix('!') {
        Some(n) => (true, n.trim()),
        None => (false, expr.trim()),
    };
    if !declared.iter().any(|d| d == name) {
        return Err(CircuitError::ShapeMismatch(format!(
            "switch `{expr}` names `{name}`, which is not in the circuit's `dims` list"
        )));
    }
    let v = dims.get(name).copied().ok_or_else(|| {
        CircuitError::ShapeMismatch(format!("switch `{expr}`: the arch shape lacks `{name}`"))
    })?;
    Ok((v != 0) != negate)
}

/// 2026-09-28: Split `"<rows> x <dim>"`.
pub(crate) fn split_shape(s: &str) -> Option<(&str, &str)> {
    let (rows, dim) = s.split_once(" x ")?;
    Some((rows.trim(), dim.trim()))
}
