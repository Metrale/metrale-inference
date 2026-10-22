// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: `kbp-graph/v1`: what the store's 3D view renders, for one plan arm
//! (DESIGN.md §10).
//!
//! Owner: blueprint.
//! Invariants:
//! - The circuit records joined with one `PlanSpec` arm: only the nodes that arm runs, in
//!   execution order (`order`), and the node-to-node edges between them, one per
//!   (tensor, reader). Indices are the circuit's own. No coordinates: layout is the site's.
//! - `collapsed` gives one template per block template and the per-layer layout.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use metrale_error::{Error, ErrorCode, error};

use super::ExportContext;
use super::schema::json_struct;
use crate::frame::Opened;
use crate::types::{EdgeStateSpec, Mode};

/// 2026-10-22: The graph schema id.
pub const GRAPH_SCHEMA: &str = "kbp-graph/v1";

json_struct! {
    /// 2026-10-22: The arm.
    pub struct GraphArm {
        /// 2026-10-22: Mode.
        pub mode: String,
        /// 2026-10-22: Rows.
        pub rows: u64,
        /// 2026-10-22: Plan digest (hex).
        pub digest: String,
    }
}

json_struct! {
    /// 2026-10-22: One layer.
    pub struct GraphLayer {
        /// 2026-10-22: Layer index.
        pub index: u32,
        /// 2026-10-22: Layer kind.
        pub kind: String,
        /// 2026-10-22: Its block templates joined by `+`.
        pub template: String,
    }
}

json_struct! {
    /// 2026-10-22: A node's formats under the arm.
    pub struct GraphPrecision {
        /// 2026-10-22: Weight format.
        pub weight: Option<String>,
        /// 2026-10-22: Stored format of the first input.
        pub activation: Option<String>,
        /// 2026-10-22: Stored format of the first output.
        pub output: Option<String>,
    }
}

json_struct! {
    /// 2026-10-22: A state a node touches.
    pub struct GraphState {
        /// 2026-10-22: State id.
        pub state: String,
        /// 2026-10-22: `read`, `write`, `update`, `snapshot`.
        pub access: String,
    }
}

json_struct! {
    /// 2026-10-22: One node the arm runs.
    pub struct GraphNode {
        /// 2026-10-22: Circuit index.
        pub index: u32,
        /// 2026-10-22: Node id.
        pub id: String,
        /// 2026-10-22: Execution order within the arm.
        pub order: u64,
        /// 2026-10-22: Layer.
        pub layer: Option<u32>,
        /// 2026-10-22: Block template.
        pub block: String,
        /// 2026-10-22: Its layer's kind.
        pub kind: Option<String>,
        /// 2026-10-22: Op spelling.
        pub op: String,
        /// 2026-10-22: Formats.
        pub precision: GraphPrecision,
        /// 2026-10-22: Group index.
        pub group: u64,
        /// 2026-10-22: The group's first kernel, `module::func`.
        pub kernel: Option<String>,
        /// 2026-10-22: Its family.
        pub family: Option<String>,
        /// 2026-10-22: Its compute unit.
        pub compute: Option<String>,
        /// 2026-10-22: Group numerics class.
        pub numerics: String,
        /// 2026-10-22: States touched.
        pub state: Vec<GraphState>,
    }
}

json_struct! {
    /// 2026-10-22: One (tensor, reader) edge.
    pub struct GraphEdge {
        /// 2026-10-22: Tensor id.
        pub id: String,
        /// 2026-10-22: Producer node index.
        pub from: u32,
        /// 2026-10-22: Reader node index.
        pub to: u32,
        /// 2026-10-22: Stored format under the arm.
        pub format: String,
        /// 2026-10-22: Row expression.
        pub rows: String,
        /// 2026-10-22: Dim expression.
        pub dim: String,
        /// 2026-10-22: Kept inside a group.
        pub fused: bool,
    }
}

json_struct! {
    /// 2026-10-22: One launch group.
    pub struct GraphGroup {
        /// 2026-10-22: Group index.
        pub id: u64,
        /// 2026-10-22: Rule.
        pub rule: String,
        /// 2026-10-22: Kernels.
        pub kernels: Vec<String>,
        /// 2026-10-22: Repetition.
        pub repeat: String,
        /// 2026-10-22: Numerics class.
        pub numerics: String,
        /// 2026-10-22: Node indices.
        pub nodes: Vec<u32>,
    }
}

json_struct! {
    /// 2026-10-22: One block template, from its first instance.
    pub struct GraphTemplate {
        /// 2026-10-22: Template-local node ids.
        pub nodes: Vec<String>,
        /// 2026-10-22: Their ops.
        pub ops: Vec<String>,
    }
}

json_struct! {
    /// 2026-10-22: The collapsed view.
    pub struct GraphCollapsed {
        /// 2026-10-22: Per layer: its templates joined by `+`.
        pub layout: Vec<String>,
        /// 2026-10-22: By template name.
        pub templates: BTreeMap<String, GraphTemplate>,
    }
}

json_struct! {
    /// 2026-10-22: `kbp-graph/v1`.
    pub struct Graph {
        /// 2026-10-22: `kbp-graph/v1`.
        pub schema: String,
        /// 2026-10-22: `sha256:<hex>`.
        pub blueprint_id: String,
        /// 2026-10-22: Lineage (hex).
        pub lineage: String,
        /// 2026-10-22: Revision.
        pub revision: u64,
        /// 2026-10-22: Exporting engine version.
        pub exported_by: String,
        /// 2026-10-22: The arm.
        pub arm: GraphArm,
        /// 2026-10-22: Layers.
        pub layers: Vec<GraphLayer>,
        /// 2026-10-22: Nodes in execution order.
        pub nodes: Vec<GraphNode>,
        /// 2026-10-22: Edges.
        pub edges: Vec<GraphEdge>,
        /// 2026-10-22: Groups.
        pub groups: Vec<GraphGroup>,
        /// 2026-10-22: Collapsed view.
        pub collapsed: GraphCollapsed,
    }
}

fn idx(i: usize) -> u32 {
    u32::try_from(i).unwrap_or(u32::MAX)
}

/// 2026-10-22: The graph of `arm` in a verified file.
pub fn graph(o: &Opened, arm: (Mode, u64), ctx: &ExportContext) -> Result<Graph, Error> {
    let bp = &o.blueprint;
    let c = &bp.circuit;
    let plan = bp
        .plans
        .iter()
        .find(|p| (p.mode, p.rows) == arm)
        .ok_or_else(|| {
            error!(
                ErrorCode::InvalidArgument,
                "--arm", "the blueprint has no such plan arm"
            )
        })?;
    let mut per_layer: BTreeMap<u32, Vec<&str>> = BTreeMap::new();
    let mut templates = BTreeMap::new();
    for b in &c.blocks {
        if let Some(l) = b.layer {
            per_layer.entry(l).or_default().push(&b.template);
        }
        templates.entry(b.template.clone()).or_insert_with(|| {
            let ns = c.nodes.get(b.first as usize..b.end as usize).unwrap_or(&[]);
            GraphTemplate {
                nodes: ns.iter().map(|n| n.local.clone()).collect(),
                ops: ns.iter().map(|n| n.op.clone()).collect(),
            }
        });
    }
    let layout: Vec<String> = per_layer.values().map(|t| t.join("+")).collect();
    let layers = c
        .layer_kinds
        .iter()
        .enumerate()
        .map(|(i, k)| GraphLayer {
            index: idx(i),
            kind: String::from(k.name()),
            template: layout.get(i).cloned().unwrap_or_default(),
        })
        .collect();
    let fmt = |e: u32| plan.edge_formats.get(e as usize).cloned();
    let mut nodes = Vec::new();
    let mut in_arm = BTreeMap::new();
    for (gi, g) in plan.groups.iter().enumerate() {
        let first = g.kernels.first();
        let spec = first.and_then(|k| bp.kernels.spec(k));
        for &ni in &g.nodes {
            let n = c.nodes.get(ni as usize).ok_or_else(|| {
                error!(
                    ErrorCode::BlueprintDecode,
                    "plan", "node index out of range"
                )
            })?;
            in_arm.insert(ni, ());
            nodes.push(GraphNode {
                index: ni,
                id: n.id.clone(),
                order: u64::try_from(nodes.len()).unwrap_or(u64::MAX),
                layer: n.layer,
                block: n.block.clone(),
                kind: n
                    .layer
                    .and_then(|l| c.layer_kinds.get(l as usize))
                    .map(|k| String::from(k.name())),
                op: n.op.clone(),
                precision: GraphPrecision {
                    weight: n.weight.clone(),
                    activation: n.inputs.first().and_then(|&e| fmt(e)),
                    output: n.outputs.first().and_then(|&e| fmt(e)),
                },
                group: u64::try_from(gi).unwrap_or(u64::MAX),
                kernel: first.map(|k| alloc::format!("{k}")),
                family: spec.and_then(|s| s.family.clone()),
                compute: spec.and_then(|s| s.compute.clone()),
                numerics: String::from(g.numerics.class()),
                state: n
                    .state
                    .iter()
                    .filter_map(|(si, a)| {
                        c.states.get(*si as usize).map(|s| GraphState {
                            state: s.id.clone(),
                            access: String::from(a.name()),
                        })
                    })
                    .collect(),
            });
        }
    }
    let mut edges = Vec::new();
    for (ei, e) in c.edges.iter().enumerate() {
        let Some(from) = e.producer.filter(|p| in_arm.contains_key(p)) else {
            continue;
        };
        for &to in e.consumers.iter().filter(|t| in_arm.contains_key(t)) {
            edges.push(GraphEdge {
                id: e.id.clone(),
                from,
                to,
                format: fmt(idx(ei)).unwrap_or_else(|| e.format.clone()),
                rows: e.rows.clone(),
                dim: e.dim.clone(),
                fused: matches!(plan.edge_states.get(ei), Some(EdgeStateSpec::Fused(_))),
            });
        }
    }
    let groups = plan
        .groups
        .iter()
        .enumerate()
        .map(|(i, g)| GraphGroup {
            id: u64::try_from(i).unwrap_or(u64::MAX),
            rule: g.rule.clone(),
            kernels: g.kernels.iter().map(|k| alloc::format!("{k}")).collect(),
            repeat: g.repeat.name(),
            numerics: String::from(g.numerics.class()),
            nodes: g.nodes.clone(),
        })
        .collect();
    Ok(Graph {
        schema: String::from(GRAPH_SCHEMA),
        blueprint_id: alloc::format!("{}", o.id),
        lineage: bp.lineage.0.hex(),
        revision: bp.revision,
        exported_by: ctx.exported_by.clone(),
        arm: GraphArm {
            mode: String::from(plan.mode.name()),
            rows: plan.rows,
            digest: plan.digest.hex(),
        },
        layers,
        nodes,
        edges,
        groups,
        collapsed: GraphCollapsed { layout, templates },
    })
}
