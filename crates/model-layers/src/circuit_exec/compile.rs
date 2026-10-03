// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: Plan to program. [`layout`] states the storage the emitters' kernels need (the
//! residual stream and the logits bound to the model's buffers, in-place outputs, packed
//! outputs); the caller places the rest in one workspace allocated at boot; [`compile`] asks
//! each group's emitter for its launches.
//!
//! Owner: model-layers circuit executor.
//! Invariants:
//! - Every stream edge (a block's `stream_in` or `stream_out`) is the model's `hidden` buffer,
//!   updated in place, as the legacy layers update it; every declared output is the model
//!   buffer its producer writes (`logits`, `tokens`).
//! - The program has exactly `plan.launches()` launches; any other count is a compile error.
//! - A group's weights are checked against the format its nodes were resolved to.

use anyhow::{Context, Result, bail, ensure};
use metrale_circuit::planner::{BufferPlan, Layout};
use metrale_circuit::{Circuit, FusionPlan, Group, Mode};
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::bindings::{BoundWeight, CircuitLayer, HeadBinding, WeightSlot};
use super::emitters::emitter;
pub use super::fixed::{DraftFixed, Fixed};
use super::kernels::KernelTable;
use super::program::{Launch, LaunchKind, Program, RunFn};

#[path = "compile_place.rs"]
mod place;
pub(super) use place::external_buffer;
pub use place::{Placement, layout};
use place::{segment_of, segments};
use crate::layer::AttnMetadataDev;

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
    pub draft: Option<&'a CircuitLayer>,
    /// 2026-10-03: The legacy forward's buffer arena, which a prefill plan's edges live in
    /// (`prefill::placement`); `None` for a build without one (the mock fixtures).
    pub arena: Option<&'a metrale_gpu_runtime::buffers::BufferArena>,
    table: Option<&'a metrale_circuit::RowTable>,
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
    pub fn meta(&self) -> Result<AttnMetadataDev> {
        Ok(match self.mode {
            Mode::Draft => self.draft_fixed()?.meta_rows(self.rows)?,
            Mode::Decode => self.fixed.meta,
            Mode::MultiSeq => self.fixed.batch_meta,
            Mode::Verify => self.fixed.verify_meta,
            Mode::VerifyBatch => self.fixed.verify_batch_meta,
            // 2026-10-03: A prefill pass uploads its metadata per call; its emitters read it
            // from the step (`StepEnv::prefill`).
            Mode::Prefill | Mode::PrefillChunk => anyhow::bail!(
                "a prefill plan reads its attention metadata from the step, not the build"
            ),
        })
    }

    /// 2026-09-30: The plan's row table (a batched verify's); `None` in other modes.
    pub fn table(&self) -> Option<&'a metrale_circuit::RowTable> {
        self.table
    }

    /// 2026-09-30: The handle of `k`, one of the group's kernels (a per-run launch).
    pub fn run_handle(&self, k: &metrale_circuit::KernelId) -> Result<KernelHandle> {
        let i = self
            .g
            .group
            .kernels
            .iter()
            .position(|x| x == k)
            .with_context(|| format!("group {} does not list {k}", self.g.index))?;
        self.handle(i)
    }

    /// 2026-09-30: Queue one launch of `k`, one of the group's kernels.
    pub fn push_kernel(&mut self, k: &metrale_circuit::KernelId, run: RunFn) -> Result<()> {
        let i = self
            .g
            .group
            .kernels
            .iter()
            .position(|x| x == k)
            .with_context(|| format!("group {} does not list {k}", self.g.index))?;
        self.push(i, run)
    }

    /// 2026-10-03: The legacy buffer arena a prefill emitter writes its scratch to, as the
    /// legacy layer it mirrors does; an error in a build without one.
    pub fn arena(&self) -> Result<&'a metrale_gpu_runtime::buffers::BufferArena> {
        self.arena
            .context("a prefill plan compiled without the model's buffer arena")
    }

    /// 2026-09-29: The draft head's buffers; an error without a bound draft head.
    pub fn draft_fixed(&self) -> Result<&'a DraftFixed> {
        self.fixed
            .draft
            .as_ref()
            .context("a draft plan without the draft head's buffers")
    }

    /// 2026-09-29: The GDN state row `row` reads: its own sequence's in a multi-sequence
    /// step, the one sequence's in a verify or a decode.
    pub fn state_row(&self, row: usize) -> usize {
        if self.mode == Mode::MultiSeq { row } else { 0 }
    }

    /// 2026-09-28: Launches of each kernel this group makes: its repeat at the plan's rows.
    pub fn reps(&self) -> usize {
        self.g.group.repeat.count(self.rows).unwrap_or(0) as usize
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
        if self.mode == Mode::Draft {
            return self
                .draft
                .with_context(|| format!("`{}`: no draft head is bound", n.id));
        }
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
            kind: LaunchKind::Kernel,
            run,
        });
        Ok(())
    }

    /// 2026-09-29: Queue one copy-engine transfer of this group (its rule's `copies`).
    pub fn push_copy(&mut self, run: RunFn) -> Result<()> {
        ensure!(
            self.g.group.copies.is_some() || self.g.group.runs.iter().any(|r| r.copies.is_some()),
            "group {} declares no copies",
            self.g.index
        );
        self.launches.push(Launch {
            group: self.g.index,
            kernel: "copy".to_string(),
            kind: LaunchKind::Copy,
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

/// 2026-09-28: Everything [`compile`] reads besides the plan.
pub struct Inputs<'a> {
    pub gpu: &'a dyn GpuBackend,
    pub config: &'a ModelConfig,
    pub kernels: &'a KernelTable,
    pub fixed: &'a Fixed,
    pub layers: &'a [CircuitLayer],
    pub head: &'a HeadBinding,
    pub draft: Option<&'a CircuitLayer>,
    /// 2026-10-03: The legacy buffer arena (`Cx::arena`); `None` in the mock fixtures.
    pub arena: Option<&'a metrale_gpu_runtime::buffers::BufferArena>,
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
        ptrs[e] = Some(external_buffer(circuit, e, inp.fixed, plan.mode)?);
    }
    compile_placed(circuit, plan, &Placement { ptrs, strides }, inp)
}

/// 2026-10-03: Compile `plan` with every edge at `placement` (a prefill plan's edges live in
/// the legacy buffer arena, `prefill::placement`).
pub fn compile_placed(
    circuit: &Circuit,
    plan: &FusionPlan,
    placement: &Placement,
    inp: &Inputs<'_>,
) -> Result<Program> {
    let (ptrs, strides) = (&placement.ptrs, &placement.strides);
    let mut launches = Vec::with_capacity(plan.launches() as usize);
    let mut owners = Vec::with_capacity(plan.launches() as usize);
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
            draft: inp.draft,
            arena: inp.arena,
            table: plan.table.as_ref(),
            ptrs,
            strides,
            formats: &plan.edge_formats,
            handles,
            launches: Vec::new(),
        };
        emitter(&group.emitter)?
            .emit(&mut cx)
            .with_context(|| format!("group {index} (rule `{}`)", group.rule))?;
        let want = group.launch_count(plan.rows);
        let want_copies = group.copy_count(plan.rows);
        let copies = cx
            .launches
            .iter()
            .filter(|l| l.kind == LaunchKind::Copy)
            .count() as u64;
        let kernels = cx.launches.len() as u64 - copies;
        if (kernels, copies) != (want, want_copies) {
            bail!(
                "emitter `{}` queued {kernels} launches and {copies} copies for group {index}; \
                 the plan counts {want} and {want_copies}",
                group.emitter,
            );
        }
        owners.extend(std::iter::repeat_n(segment_of(circuit, group), cx.launches.len()));
        launches.append(&mut cx.launches);
    }
    ensure!(launches.len() as u64 == plan.launches() + plan.copies());
    Ok(Program {
        mode: plan.mode,
        rows: plan.rows,
        plan_digest: plan.digest.clone(),
        launches,
        segments: segments(&owners),
    })
}
