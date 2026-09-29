// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: Plan to program. [`layout`] states the storage the emitters' kernels need (the
//! residual stream and the logits bound to the model's buffers, in-place outputs, packed
//! outputs); the caller places the rest in one workspace allocated at boot; [`compile`] asks
//! each group's emitter for its launches.
//!
//! Owner: model-layers circuit executor.
//! Invariants:
//! - Every stream edge (a block's `stream_in` or `stream_out`) is the model's `hidden` buffer,
//!   updated in place, as the legacy layers update it; every declared output is `logits`.
//! - The program has exactly `plan.launches()` launches; any other count is a compile error.
//! - A group's weights are checked against the format its nodes were resolved to.

use std::collections::BTreeSet;

use anyhow::{Context, Result, anyhow, bail, ensure};
use metrale_circuit::planner::{BufferPlan, Layout};
use metrale_circuit::{Circuit, FusionPlan, Group, Mode};
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::bindings::{BoundWeight, CircuitLayer, HeadBinding, WeightSlot};
use super::emitters::emitter;
use super::kernels::KernelTable;
use super::program::{Launch, Program, RunFn};
use crate::layer::AttnMetadataDev;

/// 2026-09-28: Device addresses that never move after boot.
#[derive(Clone)]
pub struct Fixed {
    /// 2026-09-28: The residual stream (`BufferArena::hidden_states`).
    pub hidden: DevicePtr,
    /// 2026-09-28: The copy of the stream the norms write (`BufferArena::residual`).
    pub residual: DevicePtr,
    /// 2026-09-28: The logits buffer the step returns.
    pub logits: DevicePtr,
    /// 2026-09-28: The single-sequence step's attention metadata, at its fixed upload
    /// address. `max_blocks_per_seq` is ignored: a step supplies it.
    pub meta: AttnMetadataDev,
    /// 2026-09-28: The multi-sequence step's attention metadata (one row per sequence), at its
    /// fixed upload address; `max_blocks_per_seq` is ignored.
    pub batch_meta: AttnMetadataDev,
    /// 2026-09-28: The quantized-activation scratch the NVFP4 MMQ GEMMs read
    /// (`BufferArena::ffn_act_q8`), sized for the widest batch.
    pub ffn_act_q8: DevicePtr,
    /// 2026-09-28: K pool per attention layer (`PagedKvCache::k_pool_ptr`).
    pub k_pools: Vec<DevicePtr>,
    /// 2026-09-28: V pool per attention layer.
    pub v_pools: Vec<DevicePtr>,
    /// 2026-09-28: Tokens per KV block.
    pub block_size: u32,
    /// 2026-09-28: `PagedKvCache::cache_stride`.
    pub cache_stride: u64,
}

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

/// 2026-09-28: What an emitter reads while compiling one group.
pub(crate) struct Cx<'a> {
    pub g: GroupRef<'a>,
    pub gpu: &'a dyn GpuBackend,
    pub mode: Mode,
    pub rows: u64,
    pub config: &'a ModelConfig,
    pub fixed: &'a Fixed,
    pub layers: &'a [CircuitLayer],
    pub head: &'a HeadBinding,
    ptrs: &'a [Option<DevicePtr>],
    strides: &'a [Option<u64>],
    formats: &'a [metrale_circuit::Format],
    handles: Vec<KernelHandle>,
    launches: Vec<Launch>,
}

impl<'a> Cx<'a> {
    fn buffer(&self, edge: usize) -> Result<DevicePtr> {
        self.ptrs[edge].with_context(|| {
            format!(
                "edge `{}` has no buffer in this plan (fused away or outside the section)",
                self.g.circuit.edges[edge].id
            )
        })
    }

    fn row_bytes(&self, edge: usize) -> Result<u64> {
        let e = &self.g.circuit.edges[edge];
        self.formats[edge]
            .bytes(1, e.dim_value)
            .with_context(|| format!("edge `{}` has no row size", e.id))
    }

    /// 2026-09-28: Bytes from one row of an edge to the next.
    fn stride_bytes(&self, edge: usize) -> Result<u64> {
        match self.strides[edge] {
            Some(s) => Ok(s),
            None => self.row_bytes(edge),
        }
    }

    /// 2026-09-28: The buffer of a materialised edge whose rows are contiguous; an edge laid out
    /// in a row pack is refused, since a kernel reading it here would step rows by its width.
    pub fn ptr(&self, edge: usize) -> Result<DevicePtr> {
        let p = self.buffer(edge)?;
        ensure!(
            self.rows <= 1 || self.stride_bytes(edge)? == self.row_bytes(edge)?,
            "edge `{}` is laid out in a row pack; it must be read with its row stride",
            self.g.circuit.edges[edge].id
        );
        Ok(p)
    }

    /// 2026-09-28: The buffer of a materialised edge and its row stride in elements, for a
    /// kernel that takes the stride.
    pub fn strided(&self, edge: usize) -> Result<(DevicePtr, u32)> {
        let e = &self.g.circuit.edges[edge];
        let (stride, row) = (self.stride_bytes(edge)?, self.row_bytes(edge)?);
        ensure!(
            e.dim_value > 0 && row % e.dim_value == 0 && stride % (row / e.dim_value) == 0,
            "edge `{}`: a {stride}-byte row stride is not whole elements",
            e.id
        );
        Ok((
            self.buffer(edge)?,
            u32::try_from(stride / (row / e.dim_value))?,
        ))
    }

    /// 2026-09-28: Row `row` of a materialised edge: its buffer plus `row` row strides.
    pub fn row_ptr(&self, edge: usize, row: usize) -> Result<DevicePtr> {
        Ok(self
            .buffer(edge)?
            .offset(row * self.stride_bytes(edge)? as usize))
    }

    /// 2026-09-28: The attention metadata this plan's steps upload.
    pub fn meta(&self) -> AttnMetadataDev {
        if self.mode == Mode::Decode {
            self.fixed.meta
        } else {
            self.fixed.batch_meta
        }
    }

    /// 2026-09-28: Launches of each kernel this group makes: its repeat at the plan's rows.
    pub fn reps(&self) -> usize {
        self.g.group.repeat.count(self.rows) as usize
    }

    /// 2026-09-28: The group's `i`-th kernel.
    pub fn handle(&self, i: usize) -> Result<KernelHandle> {
        self.handles
            .get(i)
            .copied()
            .with_context(|| format!("group {} has no kernel {i}", self.g.index))
    }

    /// 2026-09-28: The binding of the layer member `i` belongs to.
    pub fn layer(&self, i: usize) -> Result<&'a CircuitLayer> {
        let n = self.g.node(i);
        let l = n
            .layer
            .with_context(|| format!("`{}` is outside the layers", n.id))?;
        self.layers
            .get(l)
            .with_context(|| format!("no binding for layer {l}"))
    }

    /// 2026-09-28: Member `i`'s weight in `slot`, checked against the node's resolved format when
    /// the node reads a linear weight.
    pub fn weight(&self, i: usize, slot: WeightSlot) -> Result<BoundWeight> {
        let n = self.g.node(i);
        let w = *self
            .layer(i)?
            .weights
            .get(&slot)
            .with_context(|| format!("`{}`: its layer binds no {slot:?}", n.id))?;
        if let Some(f) = n.weight {
            ensure!(
                f.name().starts_with(w.family()),
                "`{}`: the plan resolved {} but the layer holds {}",
                n.id,
                f.name(),
                w.family()
            );
        }
        Ok(w)
    }

    /// 2026-09-28: Queue one launch of kernel `k` of this group.
    pub fn push(&mut self, k: usize, run: RunFn) -> Result<()> {
        let kid = self
            .g
            .group
            .kernels
            .get(k)
            .with_context(|| format!("group {} has no kernel {k}", self.g.index))?;
        self.launches.push(Launch {
            group: self.g.index,
            kernel: format!("{}::{}", kid.module, kid.func),
            run,
        });
        Ok(())
    }
}

/// 2026-09-28: How an emitter id turns a group into launches.
pub(crate) trait OpEmitter: Sync {
    /// 2026-09-28: The emitter id FUSIONS.toml names.
    fn id(&self) -> &'static str;

    /// 2026-09-28: Storage its kernels require of the group's edges (in-place, packed).
    fn constrain(&self, _g: &GroupRef<'_>, _layout: &mut Layout) -> Result<()> {
        Ok(())
    }

    /// 2026-09-28: Queue the group's launches.
    fn emit(&self, cx: &mut Cx<'_>) -> Result<()>;
}

/// 2026-09-28: The storage constraints of `plan`.
pub fn layout(circuit: &Circuit, plan: &FusionPlan) -> Result<Layout> {
    let mut layout = Layout::default();
    let materialized =
        |e: usize| plan.edge_states[e] == Some(metrale_circuit::EdgeState::Materialized);
    for b in &circuit.blocks {
        for e in b.stream_in.into_iter().chain(b.stream_out) {
            if materialized(e) {
                layout.external.insert(e);
            }
        }
    }
    for (e, edge) in circuit.edges.iter().enumerate() {
        if edge.is_output && materialized(e) {
            layout.external.insert(e);
        }
    }
    for (index, group) in plan.groups.iter().enumerate() {
        let g = GroupRef {
            circuit,
            group,
            index,
        };
        emitter(&group.emitter)?.constrain(&g, &mut layout)?;
    }
    Ok(layout)
}

/// 2026-09-28: Everything [`compile`] reads besides the plan.
pub struct Inputs<'a> {
    pub gpu: &'a dyn GpuBackend,
    pub config: &'a ModelConfig,
    pub kernels: &'a KernelTable,
    pub fixed: &'a Fixed,
    pub layers: &'a [CircuitLayer],
    pub head: &'a HeadBinding,
}

/// 2026-09-28: Compile `plan` with its buffers placed by `buffers` at `workspace`.
pub fn compile(
    circuit: &Circuit,
    plan: &FusionPlan,
    layout: &Layout,
    buffers: &BufferPlan,
    workspace: DevicePtr,
    inp: &Inputs<'_>,
) -> Result<Program> {
    let mut ptrs: Vec<Option<DevicePtr>> = vec![None; circuit.edges.len()];
    let mut strides: Vec<Option<u64>> = vec![None; circuit.edges.len()];
    for s in &buffers.slots {
        ptrs[s.edge] = Some(workspace.offset(s.offset as usize));
        strides[s.edge] = Some(s.row_stride);
    }
    for &e in &layout.external {
        ptrs[e] = Some(if circuit.edges[e].is_output {
            inp.fixed.logits
        } else {
            inp.fixed.hidden
        });
    }
    let mut launches = Vec::with_capacity(plan.launches() as usize);
    for (index, group) in plan.groups.iter().enumerate() {
        let g = GroupRef {
            circuit,
            group,
            index,
        };
        let handles = group
            .kernels
            .iter()
            .map(|k| inp.kernels.handle(k))
            .collect::<Result<Vec<_>>>()?;
        let mut cx = Cx {
            g,
            gpu: inp.gpu,
            mode: plan.mode,
            rows: plan.rows,
            config: inp.config,
            fixed: inp.fixed,
            layers: inp.layers,
            head: inp.head,
            ptrs: &ptrs,
            strides: &strides,
            formats: &plan.edge_formats,
            handles,
            launches: Vec::new(),
        };
        emitter(&group.emitter)?
            .emit(&mut cx)
            .with_context(|| format!("group {index} (rule `{}`)", group.rule))?;
        let want = group.kernels.len() as u64 * group.repeat.count(plan.rows);
        if cx.launches.len() as u64 != want {
            bail!(
                "emitter `{}` queued {} launches for group {index}; the plan counts {want}",
                group.emitter,
                cx.launches.len()
            );
        }
        launches.append(&mut cx.launches);
    }
    ensure!(launches.len() as u64 == plan.launches());
    Ok(Program {
        mode: plan.mode,
        rows: plan.rows,
        plan_digest: plan.digest.clone(),
        launches,
    })
}

/// 2026-09-28: Refuse a model the circuit misdescribes: an unbound layer, a layer or head
/// feature no rule models, or a layer whose mixer is not the circuit's.
pub fn check_bindings(
    circuit: &Circuit,
    layers: &[Option<CircuitLayer>],
    head: &HeadBinding,
) -> Result<Vec<CircuitLayer>> {
    let mut problems = BTreeSet::new();
    let mut out = Vec::with_capacity(layers.len());
    for (i, l) in layers.iter().enumerate() {
        let Some(l) = l else {
            problems.insert(format!("layer {i} has no circuit binding"));
            continue;
        };
        for u in &l.unmodelled {
            problems.insert(format!("layer {i}: {u}"));
        }
        let want_gdn =
            circuit.layer_kinds.get(i) == Some(&metrale_circuit::LayerKind::LinearAttention);
        let is_gdn = matches!(l.mixer, super::bindings::MixerFacts::Gdn(_));
        if want_gdn != is_gdn {
            problems.insert(format!(
                "layer {i}: the circuit and the model disagree on its kind"
            ));
        }
        out.push(l.clone());
    }
    for u in &head.unmodelled {
        problems.insert(format!("head: {u}"));
    }
    if !problems.is_empty() {
        return Err(anyhow!(
            "the circuit does not model this model:\n  {}",
            problems.into_iter().collect::<Vec<_>>().join("\n  ")
        ));
    }
    ensure!(
        out.len() == circuit.layer_kinds.len(),
        "the model has {} layers, the circuit {}",
        out.len(),
        circuit.layer_kinds.len()
    );
    Ok(out)
}
