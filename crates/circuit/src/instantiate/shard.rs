// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The rank half of the instantiation builder: each edge's sharding on this rank,
//! the collectives derived where a consumer needs another sharding than its producer made, and
//! the idle nodes of a part placed on a group this rank is not in ([`crate::parallel`]).
//!
//! Owner: metrale-circuit.
//! Invariants:
//! - Only a rank circuit has a [`Par`]; the whole-model circuit never reaches this file.
//! - A collective is a node of the circuit with its group (`params.group`, ascending ranks) and,
//!   for a broadcast, its root (`params.root`); its output edge has the tensor's whole width for
//!   a gather and the input's otherwise.

use crate::prelude::*;

use alloc::collections::BTreeMap;

use super::Builder;
use crate::circuit_toml::{CircuitError, EdgeFile, NodeFile, split_shape};
use crate::dims::DimExpr;
use crate::format::Format;
use crate::ir::{Edge, Node, OpKind};
use crate::parallel::{Layout, ParallelSpec, Scope, Shard, passes_partial};

/// 2026-10-23: The rank a circuit is instantiated for.
pub(crate) struct Par<'a> {
    pub spec: &'a ParallelSpec,
    pub layout: &'a Layout,
    pub rank: u64,
    /// 2026-10-23: The whole-model dims (a gathered edge's width).
    pub whole: BTreeMap<String, u64>,
    /// 2026-10-23: Edge to its sharding; an edge not listed is replicated.
    pub shards: BTreeMap<usize, Shard>,
    /// 2026-10-23: Each split `marked` part's slice length on this rank (its marked edges'
    /// width).
    pub marked: BTreeMap<String, u64>,
    /// 2026-10-23: Edges only idle nodes read on this rank (the input of a part this rank is
    /// outside of): computed, and left unread here.
    pub idle_reads: alloc::collections::BTreeSet<usize>,
    /// 2026-10-23: The current block's state accesses its idle nodes would make (a replicated
    /// cache written here and read only by a part this rank is outside of).
    pub idle_access: Vec<(usize, crate::state::StateAccess)>,
}

/// 2026-10-23: An edge this rank does not hold (its producer idles here), as the part's group
/// holds it.
#[derive(Clone)]
pub(crate) struct Absent {
    shard: Shard,
    id: String,
    format: Format,
    rows: DimExpr,
    dim: DimExpr,
}

/// 2026-10-23: What a node does on this rank.
pub(super) enum Run {
    /// 2026-10-23: It runs; its outputs are sharded so.
    Go(Vec<Shard>),
    /// 2026-10-23: The rank idles for it.
    Idle,
}

fn group_text(g: &[u64]) -> String {
    g.iter().map(u64::to_string).collect::<Vec<_>>().join(",")
}

impl Builder<'_> {
    fn par_err(&self, template: &str, nf: &NodeFile, detail: String) -> CircuitError {
        CircuitError::Parallel(format!("block `{template}` node `{}`: {detail}", nf.id))
    }

    /// 2026-10-23: The sharding an output edge's shape (and mark) gives it on this rank.
    fn out_shard(&self, template: &str, nf: &NodeFile, ef: &EdgeFile) -> Result<Shard, CircuitError> {
        let p = self.par.as_ref().expect("a rank circuit");
        let (_, dim) = split_shape(&ef.shape).ok_or_else(|| {
            self.par_err(template, nf, format!("shape `{}` is not `<rows> x <dim>`", ef.shape))
        })?;
        let dim = DimExpr::parse(dim).map_err(|e| self.par_err(template, nf, e.to_string()))?;
        let names: Vec<&str> = dim.names().collect();
        let mut found: Vec<&str> = Vec::new();
        for name in &names {
            if let Some(s) = p.spec.get(*name)
                && s.scope == Scope::Every
            {
                let part = crate::parallel::part_of(p.spec, name);
                if p.layout.parts.get(part).is_some_and(|pl| pl.splits()) {
                    found.push(part);
                }
            }
        }
        if let Some(m) = &ef.split {
            let ok = p.spec.get(m).is_some_and(|s| s.scope == Scope::Marked && s.tie.is_none())
                && names.contains(&m.as_str());
            if !ok {
                return Err(self.par_err(
                    template,
                    nf,
                    format!("edge `{}` is marked `split = \"{m}\"`, which is no marked part its shape names", ef.edge),
                ));
            }
            if p.layout.parts[m].splits() {
                found.push(m.as_str());
            }
        }
        if let Some(other) = found.iter().find(|f| **f != found[0]) {
            return Err(self.par_err(
                template,
                nf,
                format!("edge `{}` would split along parts `{}` and `{other}`", ef.edge, found[0]),
            ));
        }
        let split = found.first().copied();
        Ok(match split {
            None => Shard::Replicated,
            Some(d) => Shard::Split {
                dim: d.to_string(),
                group: p.layout.parts[d].group.clone(),
            },
        })
    }

    fn input_shards(
        &self,
        template: &str,
        nf: &NodeFile,
        local: &BTreeMap<String, usize>,
    ) -> Result<Vec<Shard>, CircuitError> {
        nf.inputs
            .iter()
            .map(|name| match (local.get(name), self.absent.get(name)) {
                (Some(&e), _) => Ok(self.shard_of(e)),
                (None, Some(a)) => Ok(a.shard.clone()),
                (None, None) => Err(CircuitError::DanglingInput {
                    block: template.to_string(),
                    node: nf.id.clone(),
                    edge: name.clone(),
                }),
            })
            .collect()
    }

    /// 2026-10-23: Record `outputs`' sharding; a marked edge's width is this rank's slice.
    pub(super) fn record_shards(
        &mut self,
        template: &str,
        outputs: &[usize],
        shards: Vec<Shard>,
    ) -> Result<(), CircuitError> {
        for (&e, s) in outputs.iter().zip(shards) {
            let p = self.par.as_mut().expect("a rank circuit");
            if let Shard::Split { dim, .. } = &s
                && let Some(&len) = p.marked.get(dim)
            {
                let mut dims = self.shape.dims.clone();
                dims.insert(dim.clone(), len);
                self.circuit.edges[e].dim_value =
                    self.circuit.edges[e]
                        .dim
                        .eval(&dims)
                        .map_err(|source| CircuitError::Dim {
                            block: template.to_string(),
                            source,
                        })?;
            }
            if s != Shard::Replicated {
                p.shards.insert(e, s);
            }
        }
        Ok(())
    }

    /// 2026-10-23: The block's stream and outputs leave it whole on every rank, or sliced over the
    /// whole world: a partial is summed and a part on a sub-group is gathered and broadcast here,
    /// at the end of the block that made it, on every rank alike.
    pub(super) fn seal_boundary(
        &mut self,
        template: &str,
        prefix: &str,
        layer: Option<usize>,
        tpl: &crate::circuit_toml::BlockFile,
        local: &mut BTreeMap<String, usize>,
    ) -> Result<(), CircuitError> {
        let world = self.par.as_ref().expect("a rank circuit").layout.world();
        let names: Vec<String> = tpl
            .stream_out
            .iter()
            .cloned()
            .chain(tpl.outputs.iter().map(|o| o.edge().to_string()))
            .collect();
        for name in names {
            let shard = match (local.get(&name), self.absent.get(&name)) {
                (Some(&e), _) => self.shard_of(e),
                (None, Some(a)) => a.shard.clone(),
                (None, None) => continue,
            };
            match shard {
                Shard::Partial { group } => {
                    self.boundary(OpKind::AllReduce, template, prefix, layer, &name, &group, local)?
                }
                Shard::Split { group, .. } if group != world => {
                    self.boundary(OpKind::AllGather, template, prefix, layer, &name, &group, local)?
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// 2026-10-23: Drop the block's states this rank holds none of (sized 0 by a part it idles
    /// for) that no node it runs touches, renumbering the block's state references; returns the
    /// idle nodes' accesses to the states kept.
    pub(super) fn drop_idle_states(
        &mut self,
        first_state: usize,
        first_node: usize,
    ) -> Vec<(usize, crate::state::StateAccess)> {
        let touched: alloc::collections::BTreeSet<usize> = self.circuit.nodes[first_node..]
            .iter()
            .flat_map(|n| n.state.iter().map(|(s, _)| *s))
            .collect();
        let mut remap = BTreeMap::new();
        let mut kept = Vec::new();
        for (i, st) in self.circuit.states.drain(first_state..).enumerate() {
            let idx = first_state + i;
            if st.elements == 0 && !touched.contains(&idx) {
                continue;
            }
            remap.insert(idx, first_state + kept.len());
            kept.push(st);
        }
        self.circuit.states.extend(kept);
        for n in &mut self.circuit.nodes[first_node..] {
            for (s, _) in &mut n.state {
                if let Some(&to) = remap.get(s) {
                    *s = to;
                }
            }
        }
        let p = self.par.as_mut().expect("a rank circuit");
        core::mem::take(&mut p.idle_access)
            .into_iter()
            .filter_map(|(s, a)| remap.get(&s).map(|&to| (to, a)))
            .collect()
    }

    pub(super) fn shard_of(&self, e: usize) -> Shard {
        self.par
            .as_ref()
            .and_then(|p| p.shards.get(&e).cloned())
            .unwrap_or(Shard::Replicated)
    }

    /// 2026-10-23: Decide `nf` on this rank: convert each input the node cannot read as sharded
    /// (inserting the collectives, rewiring `local`), and either run it with its outputs'
    /// sharding or idle it (recording its outputs as absent).
    pub(super) fn shard_node(
        &mut self,
        template: &str,
        prefix: &str,
        layer: Option<usize>,
        nf: &NodeFile,
        op: &OpKind,
        local: &mut BTreeMap<String, usize>,
    ) -> Result<Run, CircuitError> {
        let mut outs = Vec::with_capacity(nf.out.len());
        for ef in &nf.out {
            outs.push(self.out_shard(template, nf, ef)?);
        }
        let ins = self.input_shards(template, nf, local)?;
        let partial: Vec<bool> = ins.iter().map(|s| matches!(s, Shard::Partial { .. })).collect();
        let groups: Vec<&[u64]> = ins
            .iter()
            .filter_map(|s| match s {
                Shard::Partial { group } => Some(group.as_slice()),
                _ => None,
            })
            .collect();
        let keep = passes_partial(op, &partial) && groups.windows(2).all(|w| w[0] == w[1]);
        let contraction = op.reads_linear_weight();
        for (name, s) in nf.inputs.iter().zip(&ins) {
            match s {
                Shard::Partial { group } if !keep => {
                    let g = group.clone();
                    self.boundary(OpKind::AllReduce, template, prefix, layer, name, &g, local)?;
                }
                Shard::Split { group, .. } if !contraction => {
                    // 2026-10-23: A node with no output edge writes this rank's state slice (a
                    // snapshot, a cache write): rank-local.
                    if !outs.is_empty() && !outs.iter().any(|o| o == s) {
                        let g = group.clone();
                        self.boundary(OpKind::AllGather, template, prefix, layer, name, &g, local)?;
                    }
                }
                Shard::Split { .. } if outs.iter().any(|o| *o != Shard::Replicated && o != s) => {
                    return Err(self.par_err(
                        template,
                        nf,
                        "a contraction reads one part's slice into another part's slice (a \
                         resharding between two parts, not modelled)"
                            .into(),
                    ));
                }
                _ => {}
            }
        }
        let ins = self.input_shards(template, nf, local)?;
        let sliced_out = outs.iter().any(|o| *o != Shard::Replicated);
        if let Some(g) = ins.iter().find_map(|s| match s {
            Shard::Split { group, .. } if contraction && !sliced_out => Some(group.clone()),
            Shard::Partial { group } if keep => Some(group.clone()),
            _ => None,
        }) {
            for o in &mut outs {
                *o = Shard::Partial { group: g.clone() };
            }
        }
        let p = self.par.as_ref().expect("a rank circuit");
        let (rank, world) = (p.rank, p.layout.world());
        let present = nf.inputs.iter().all(|n| local.contains_key(n));
        // 2026-10-23: A node with no output runs where its inputs are held.
        let here = if outs.is_empty() {
            ins.iter().all(|i| i.group(&world).contains(&rank))
        } else {
            outs.iter().all(|o| o.group(&world).contains(&rank))
        };
        if present && here {
            return Ok(Run::Go(outs));
        }
        if here {
            return Err(self.par_err(template, nf, "an input this rank does not hold reaches a node it runs".into()));
        }
        let refs = self.state_refs(template, prefix, nf)?;
        let p = self.par.as_mut().expect("a rank circuit");
        p.idle_reads.extend(nf.inputs.iter().filter_map(|n| local.get(n).copied()));
        p.idle_access.extend(refs);
        for (ef, shard) in nf.out.iter().zip(outs) {
            let (rows, dim) = split_shape(&ef.shape).expect("checked by out_shard");
            let parse = |t: &str| DimExpr::parse(t).map_err(|e| self.par_err(template, nf, e.to_string()));
            let a = Absent {
                shard,
                id: format!("{prefix}.{}", ef.edge),
                format: Format::parse(&ef.format).map_err(|source| CircuitError::Format {
                    block: template.to_string(),
                    edge: ef.edge.clone(),
                    source,
                })?,
                rows: parse(rows)?,
                dim: parse(dim)?,
            };
            self.absent.insert(ef.edge.clone(), a);
        }
        Ok(Run::Idle)
    }

    /// 2026-10-23: Make `name` whole on every rank: the reduce or gather over `group` (on its
    /// ranks), then, for a group smaller than the world, a broadcast from its first rank.
    fn boundary(
        &mut self,
        op: OpKind,
        template: &str,
        prefix: &str,
        layer: Option<usize>,
        name: &str,
        group: &[u64],
        local: &mut BTreeMap<String, usize>,
    ) -> Result<(), CircuitError> {
        let p = self.par.as_ref().expect("a rank circuit");
        let world = p.layout.world();
        let member = group.contains(&p.rank);
        let mut e = local.get(name).copied();
        if member {
            let x = e.expect("a member holds its group's tensor");
            e = Some(self.collective(op, template, prefix, layer, name, Some(x), None, group, None));
        }
        if group != world.as_slice() {
            let src = e.map(|x| self.circuit.edges[x].clone());
            let a = self.absent.get(name).cloned();
            let root = group[0];
            e = Some(self.collective(OpKind::Broadcast, template, prefix, layer, name, e, src.or_else(|| {
                a.map(|a| Edge {
                    id: a.id,
                    format: a.format,
                    rows: a.rows,
                    dim: a.dim,
                    dim_value: 0,
                    producer: None,
                    consumers: Vec::new(),
                    is_output: false,
                    binds: None,
                })
            }), &world, Some(root)));
        }
        let e = e.ok_or_else(|| CircuitError::Parallel(format!("block `{template}`: `{name}` reaches no rank")))?;
        self.absent.remove(name);
        local.insert(name.to_string(), e);
        Ok(())
    }

    /// 2026-10-23: One collective node over `group` reading `input` (none on a broadcast's
    /// receiving rank), its output edge described by `like` when there is no input.
    #[allow(clippy::too_many_arguments)]
    fn collective(
        &mut self,
        op: OpKind,
        template: &str,
        prefix: &str,
        layer: Option<usize>,
        name: &str,
        input: Option<usize>,
        like: Option<Edge>,
        group: &[u64],
        root: Option<u64>,
    ) -> usize {
        let src = input
            .map(|x| self.circuit.edges[x].clone())
            .or(like)
            .expect("a collective has an input or a described output");
        let p = self.par.as_ref().expect("a rank circuit");
        let whole = |d: &DimExpr| d.eval(&p.whole).unwrap_or(src.dim_value);
        let dim_value = match (op, input) {
            (OpKind::AllGather, _) | (OpKind::Broadcast, None) => whole(&src.dim),
            _ => src.dim_value,
        };
        let suffix = op.name();
        let node = self.circuit.nodes.len();
        if let Some(x) = input {
            self.circuit.edges[x].consumers.push(node);
        }
        let base = src.id.split('@').next().unwrap_or(&src.id).to_string();
        self.circuit.edges.push(Edge {
            id: format!("{base}@{suffix}"),
            format: src.format,
            rows: src.rows.clone(),
            dim: src.dim.clone(),
            dim_value,
            producer: Some(node),
            consumers: Vec::new(),
            is_output: false,
            binds: None,
        });
        let out = self.circuit.edges.len() - 1;
        let mut params = BTreeMap::new();
        params.insert("group".to_string(), group_text(group));
        if let Some(r) = root {
            params.insert("root".to_string(), r.to_string());
        }
        self.circuit.nodes.push(Node {
            id: format!("{prefix}.{name}@{suffix}"),
            local: format!("{name}@{suffix}"),
            op,
            inputs: input.into_iter().collect(),
            outputs: vec![out],
            weight: None,
            binding: Vec::new(),
            params,
            layer,
            block: template.to_string(),
            state: Vec::new(),
        });
        out
    }
}
