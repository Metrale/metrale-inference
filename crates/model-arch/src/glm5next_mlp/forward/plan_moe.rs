// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: A routed MoE site run from its plan groups instead of `forward_moe`'s levers: the
//! steps a plan names ([`MoeSteps`], over plan kernel ids, built by `glm5next_plan`), bound to
//! this device's handles ([`MoeSteps::bind`]) and run through the launch helpers `forward_moe`
//! itself uses, so a plan-executed site makes the launches its hand-written path makes when the
//! plan and the levers agree, and the plan's when they do not.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - [`MoePlan::run`] reads no lever: every kernel, grid and stored width is a field of the plan,
//!   decided when it was compiled; the workspace and weights are the site's own buffers.
//! - A plan is bound only to a site whose dims are the plan's (the rank's expert slice, top-k,
//!   expert count) and whose expert tables match its scale format (`_cs` entries read base
//!   tables); anything else is refused before a launch.
//! - The shared expert is not modelled here yet: the caller runs it on the hand-written path,
//!   starting it before the router (`shared_start`: a fork onto the side stream) and finishing
//!   it before the combine (`shared_finish`: the join, or the whole expert in line), as
//!   `forward_moe` does.

use anyhow::{Context, Result, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_model_layers::layers::ops;

use super::super::Glm5NextMlpConfig;
use super::super::weights::{Glm5NextMoeWeights, W4a4ActScales};
use super::Glm5NextMlpWorkspace;
pub use super::topk_combine::{combine_blocks_per_row, topk_block};
use super::topk_combine::{combine_raw, topk_raw};
pub use super::w4a4::SweepEntries;
use super::w4a4::{SweepShape, quant_static, row_union_raw, sweep_raw};
use crate::glm5next_mlp::build_w4a4::{check_w4a4_k, check_w4a4_k64};

/// 2026-10-23: The router logits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RouterStep<K> {
    /// 2026-10-23: One M = 1 FP32-out GEMV per row.
    PerRow(K),
    /// 2026-10-23: One fixed-count batched FP32-out launch over all rows.
    Batched(K),
}

/// 2026-10-23: The top-k: its kernel and block width.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TopkStep<K> {
    pub kernel: K,
    pub block: u32,
}

/// 2026-10-23: How a routed projection stores its experts' block scales: the `_cs` entries read
/// the compact codes through a per-expert base table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScaleForm {
    Plain,
    Compact,
}

/// 2026-10-23: One routed W4A4 projection: the activation quantizer and the row width it stores,
/// the union builder (union entries only), the sweep and which entry list it reads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProjStep<K> {
    pub quant: K,
    pub k_store: usize,
    pub union: Option<K>,
    pub sweep: K,
    pub entries: SweepEntries,
    pub scales: ScaleForm,
}

/// 2026-10-23: The routed SwiGLU, or the one that also zeroes `expert_out`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ActStep<K> {
    Plain(K),
    Zero(K),
}

/// 2026-10-23: The combine and its blocks per row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlendStep<K> {
    pub kernel: K,
    pub blocks_per_row: u32,
}

/// 2026-10-23: What the plan launches for one routed MoE site at `rows` rows, with the rank dims
/// it was planned at.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MoeSteps<K> {
    pub rows: usize,
    pub hidden: usize,
    pub moe_inter: usize,
    pub top_k: usize,
    pub experts: usize,
    pub router: RouterStep<K>,
    pub topk: TopkStep<K>,
    pub gate_up: ProjStep<K>,
    pub act: ActStep<K>,
    pub down: ProjStep<K>,
    pub blend: BlendStep<K>,
}

/// 2026-10-23: A site's steps bound to this device, with the sweep's persistent grid.
pub type MoePlan = (MoeSteps<KernelHandle>, u32);

impl<K> ProjStep<K> {
    fn map<L>(self, f: &mut impl FnMut(K) -> Result<L>) -> Result<ProjStep<L>> {
        Ok(ProjStep {
            quant: f(self.quant)?,
            k_store: self.k_store,
            union: self.union.map(&mut *f).transpose()?,
            sweep: f(self.sweep)?,
            entries: self.entries,
            scales: self.scales,
        })
    }
}

impl<K> MoeSteps<K> {
    /// 2026-10-23: Every kernel through `f`, in launch order.
    pub fn map<L>(self, mut f: impl FnMut(K) -> Result<L>) -> Result<MoeSteps<L>> {
        Ok(MoeSteps {
            rows: self.rows,
            hidden: self.hidden,
            moe_inter: self.moe_inter,
            top_k: self.top_k,
            experts: self.experts,
            router: match self.router {
                RouterStep::PerRow(k) => RouterStep::PerRow(f(k)?),
                RouterStep::Batched(k) => RouterStep::Batched(f(k)?),
            },
            topk: TopkStep {
                kernel: f(self.topk.kernel)?,
                block: self.topk.block,
            },
            gate_up: self.gate_up.map(&mut f)?,
            act: match self.act {
                ActStep::Plain(k) => ActStep::Plain(f(k)?),
                ActStep::Zero(k) => ActStep::Zero(f(k)?),
            },
            down: self.down.map(&mut f)?,
            blend: BlendStep {
                kernel: f(self.blend.kernel)?,
                blocks_per_row: self.blend.blocks_per_row,
            },
        })
    }

    /// 2026-10-23: Whether the plan zeroes `expert_out` with a memset before the experts: not when
    /// the one-row own-slots down sweep writes every row itself, nor when the SwiGLU zeroes it.
    pub fn memsets_expert_out(&self) -> bool {
        !(self.down.entries == SweepEntries::OwnSlots || matches!(self.act, ActStep::Zero(_)))
    }
}

/// 2026-10-23: The row width a W4A4 activation quantizer stores for K `k`: `k` for the plain
/// entry (K a multiple of 128), K padded to 128 for the `_k64` twin.
pub fn k_store_for(k64: bool, k: usize, what: &str) -> Result<usize> {
    if k64 {
        check_w4a4_k64(k, what)?;
        ensure!(
            !k.is_multiple_of(128),
            "GLM W4A4 {what}: K = {k} is a multiple of 128; the plain quantizer serves it"
        );
        Ok(k.next_multiple_of(128))
    } else {
        check_w4a4_k(k, what)?;
        Ok(k)
    }
}

impl MoeSteps<KernelHandle> {
    /// 2026-10-23: Refuse a site this plan was not compiled for: other dims, too many rows for
    /// the workspace, or expert tables in the other scale form.
    pub fn check_site(
        &self,
        cfg: &Glm5NextMlpConfig,
        w: &Glm5NextMoeWeights,
        ws: &Glm5NextMlpWorkspace,
    ) -> Result<()> {
        let plan = (self.hidden, self.moe_inter, self.top_k, self.experts);
        let site = (cfg.hidden, cfg.moe_intermediate, cfg.top_k, cfg.num_experts);
        ensure!(
            plan == site,
            "GLM MoE plan: planned (hidden, expert slice, top_k, experts) {plan:?}, the site holds \
             {site:?}"
        );
        ensure!(
            cfg.local_experts == cfg.num_experts,
            "GLM MoE plan: the planned layout holds every expert on the rank; this rank holds {} \
             of {}",
            cfg.local_experts,
            cfg.num_experts
        );
        ensure!(
            self.rows >= 1 && self.rows <= ws.max_rows && self.rows * self.top_k <= ws.w4a4_rows,
            "GLM MoE plan: {} rows do not fit a workspace of {} rows ({} W4A4 slots)",
            self.rows,
            ws.max_rows,
            ws.w4a4_rows
        );
        ensure!(
            self.hidden <= ws.w4a4_k && self.down.k_store <= ws.w4a4_k,
            "GLM MoE plan: stored widths {} / {} exceed the W4A4 scratch's {}",
            self.hidden,
            self.down.k_store,
            ws.w4a4_k
        );
        let form = |t: &super::super::weights::Glm5NextExpertPtrTable| {
            if t.scale_bases.is_null() {
                ScaleForm::Plain
            } else {
                ScaleForm::Compact
            }
        };
        for (step, tables, what) in [
            (&self.gate_up, [&w.ptrs.gate, &w.ptrs.up], "gate/up"),
            (&self.down, [&w.ptrs.down, &w.ptrs.down], "down"),
        ] {
            for t in tables {
                ensure!(
                    form(t) == step.scales,
                    "GLM MoE plan: the {what} sweep reads {:?} scales, the bound table holds {:?}",
                    step.scales,
                    form(t)
                );
            }
        }
        ensure!(
            w.act_scales.is_some(),
            "GLM MoE plan: the plan runs W4A4 experts but no static activation scales are bound"
        );
        Ok(())
    }

    /// 2026-10-23: The site at `rows` rows of `x` into `out`, in the plan's order: `shared_start`,
    /// router, top-k, the routed experts (zeroing first where the plan does not zero otherwise),
    /// `shared_finish`, the combine.
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        gpu: &dyn GpuBackend,
        ctas: u32,
        cfg: &Glm5NextMlpConfig,
        w: &Glm5NextMoeWeights,
        ws: &Glm5NextMlpWorkspace,
        x: DevicePtr,
        out: DevicePtr,
        rows: usize,
        shared_start: impl FnOnce() -> Result<()>,
        shared_finish: impl FnOnce() -> Result<()>,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            rows == self.rows,
            "GLM MoE plan: compiled for {} rows, run at {rows}",
            self.rows
        );
        let scales: W4a4ActScales = w.act_scales.context("GLM MoE plan: no activation scales")?;
        let (n_e, h) = (self.experts, self.hidden);
        shared_start()?;
        match &self.router {
            RouterStep::PerRow(gemv) => {
                let k = ops::DenseMmKernels {
                    gemm: KernelHandle(0),
                    gemv: *gemv,
                    batchm: KernelHandle(0),
                };
                for r in 0..rows {
                    let (a, c) = (x.offset(r * h * 2), ws.logits.offset(r * n_e * 4));
                    ops::dense_mm_bf16(gpu, &k, a, w.router, c, 1, n_e, h, stream)?;
                }
            }
            RouterStep::Batched(entry) => {
                crate::glm5next_layer::wide_gemv::launch_fixed(
                    gpu, *entry, x, w.router, ws.logits, rows, n_e, h, stream,
                )?;
            }
        }
        topk_raw(
            gpu,
            self.topk.kernel,
            self.topk.block,
            cfg,
            w,
            ws,
            rows,
            stream,
        )?;
        if self.memsets_expert_out() {
            let bytes = super::w4a4::expert_out_bytes(cfg, rows);
            gpu.memset_async(ws.expert_out, 0, bytes, stream)?;
        }
        let shape = |n, k, act_div| SweepShape {
            ctas,
            n,
            k,
            rows,
            top_k: self.top_k,
            act_div,
            num_experts: n_e,
        };
        let bases = |s: &ProjStep<KernelHandle>, t: DevicePtr| match s.scales {
            ScaleForm::Plain => None,
            ScaleForm::Compact => Some(t),
        };
        let mi = self.moe_inter;
        let g = &self.gate_up;
        quant_static(
            gpu,
            g.quant,
            ws,
            x,
            rows,
            h,
            g.k_store,
            scales.gate_up,
            stream,
        )?;
        if let Some(u) = g.union.or(self.down.union) {
            row_union_raw(gpu, u, ws, rows, self.top_k, stream)?;
        }
        let gate_up = [(&w.ptrs.gate, ws.a_gate), (&w.ptrs.up, ws.a_up)];
        let gb = bases(g, w.ptrs.gate.scale_bases);
        sweep_raw(
            gpu,
            ws,
            (g.sweep, g.entries),
            gb,
            &gate_up,
            shape(mi, h, self.top_k),
            stream,
        )?;
        let slots = rows * self.top_k;
        match self.act {
            ActStep::Plain(k) => super::launch::swiglu(
                gpu,
                k,
                ws.a_gate,
                ws.a_up,
                ws.a_act,
                slots * mi,
                cfg.swiglu_limit,
                stream,
            )?,
            ActStep::Zero(k) => {
                let bytes = super::w4a4::expert_out_bytes(cfg, rows);
                ensure!(
                    bytes.is_multiple_of(16),
                    "GLM MoE plan: the zeroing SwiGLU writes 16-byte words; expert_out is {bytes}"
                );
                super::launch::swiglu_zero(
                    gpu,
                    k,
                    ws,
                    slots * mi,
                    cfg.swiglu_limit,
                    bytes / 16,
                    stream,
                )?
            }
        }
        let d = &self.down;
        quant_static(
            gpu,
            d.quant,
            ws,
            ws.a_act,
            slots,
            mi,
            d.k_store,
            scales.down,
            stream,
        )?;
        let down = [(&w.ptrs.down, ws.expert_out)];
        let db = bases(d, w.ptrs.down.scale_bases);
        sweep_raw(
            gpu,
            ws,
            (d.sweep, d.entries),
            db,
            &down,
            shape(h, mi, 1),
            stream,
        )?;
        shared_finish()?;
        combine_raw(
            gpu,
            self.blend.kernel,
            self.blend.blocks_per_row,
            cfg,
            ws,
            out,
            rows,
            false,
            stream,
        )
    }
}

/// 2026-10-23: Bind `steps` to `gpu` (each plan kernel by its `(module, func)`) for a site of
/// `cfg`, `w` and `ws`, which [`MoeSteps::check_site`] must accept; with the sweep grid of this
/// device.
pub fn bind(
    gpu: &dyn GpuBackend,
    steps: MoeSteps<metrale_circuit::KernelId>,
    cfg: &Glm5NextMlpConfig,
    w: &Glm5NextMoeWeights,
    ws: &Glm5NextMlpWorkspace,
) -> Result<MoePlan> {
    let bound = steps.map(|k| metrale_model_layers::circuit_exec::family::plan_kernel(gpu, &k))?;
    bound.check_site(cfg, w, ws)?;
    Ok((bound, crate::glm5next_mlp::w4a4_sweep_ctas(gpu)?))
}

/// 2026-10-23: The site from `plan` with its shared expert on the hand-written path, as
/// `forward_moe` runs it over the site's rows as one piece: forked onto the workspace's side
/// stream when its fork takes `rows` (`METRALE_GLM_SHARED_FORK`), else in line before the
/// combine; `k` is read for the shared expert only.
#[allow(clippy::too_many_arguments)]
pub fn run_with_legacy_shared(
    plan: &MoePlan,
    gpu: &dyn GpuBackend,
    k: &crate::glm5next_mlp::Glm5NextMlpKernels,
    cfg: &Glm5NextMlpConfig,
    w: &Glm5NextMoeWeights,
    ws: &Glm5NextMlpWorkspace,
    x: DevicePtr,
    out: DevicePtr,
    rows: usize,
    stream: u64,
) -> Result<()> {
    let (steps, ctas) = plan;
    let pieces = [rows];
    let expert =
        |scratch, s| super::pieces::shared_expert(gpu, k, cfg, w, x, ws, scratch, &pieces, s);
    let fork = ws.shared_fork.as_ref().filter(|f| f.takes(rows));
    let start = || match fork {
        Some(f) => f.fork(gpu, stream, |side, scratch| expert(scratch, side)),
        None => Ok(()),
    };
    let finish = || match fork {
        Some(f) => f.join(gpu, stream),
        None => expert(ws.dense_scratch(), stream),
    };
    steps.run(gpu, *ctas, cfg, w, ws, x, out, rows, start, finish, stream)
}

#[cfg(test)]
#[path = "plan_moe_tests.rs"]
mod tests;
