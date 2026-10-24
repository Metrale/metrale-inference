// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The emitters of a routed MoE site (`moe` block): `moe_router` (router logits and
//! top-k), `glm_moe_w4a4` (a W4A4 projection's quantizer, union and sweep), `glm_swiglu` (the
//! routed SwiGLU) and `moe_blend` / `moe_blend_split` (the combine). Each turns its group's
//! typed fields into one step of [`MoeSteps`]: the kernels the plan names, the widths from the
//! rank circuit's dims, launch shapes by the rules the hand-written path sizes them with.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - An emitter recognizes the entries it was written for by the plan's `(module, func)` and
//!   refuses any other; it never guesses a launch shape for an unknown entry.
//! - The shared expert's group is a declared legacy site of the block: compiled sites list it,
//!   the plan does not launch it.

use anyhow::{Context, Result, bail, ensure};
use metrale_circuit::{KernelId, Mode};
use metrale_model_layers::circuit_exec::family::{GroupEmitter, PlanGroup};

use super::GlmBlock;
use crate::glm5next_mlp::forward::plan_moe::{
    ActStep, BlendStep, MoeSteps, ProjStep, RouterStep, ScaleForm, SweepEntries, TopkStep,
    combine_blocks_per_row, k_store_for, topk_block,
};

/// 2026-10-23: The typed steps of one MoE site, as its groups fill them.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct MoeSlots {
    pub router: Option<RouterStep<KernelId>>,
    pub topk: Option<TopkStep<KernelId>>,
    pub gate_up: Option<ProjStep<KernelId>>,
    pub act: Option<ActStep<KernelId>>,
    pub down: Option<ProjStep<KernelId>>,
    pub blend: Option<BlendStep<KernelId>>,
    /// 2026-10-24: What the plan names for the shared expert (its group: the segment or not, its
    /// rule's `sm_budget`); `None` when the site has no shared-expert group.
    pub shared: Option<crate::glm5next_dense_seg::DenseSite>,
}

/// 2026-10-23: `(func, entry list, rows it serves, _k64 twin)` of every routed sweep entry a plan
/// may name, without the `_cs` suffix (`w4a4_gemv_mx_moe.cu`).
const SWEEPS: [(&str, SweepEntries, (u64, u64), bool); 8] = [
    (
        "w4a4_gemv_mx8_moe_slots_sweep",
        SweepEntries::OwnSlots,
        (1, 1),
        false,
    ),
    (
        "w4a4_gemv_mx8_moe_slots_sweep_k64",
        SweepEntries::OwnSlots,
        (1, 1),
        true,
    ),
    (
        "w4a4_gemv_mx8_moe_union_sweep",
        SweepEntries::Union,
        (2, 8),
        false,
    ),
    (
        "w4a4_gemv_mx8_moe_union_sweep_sl",
        SweepEntries::Union,
        (2, 8),
        false,
    ),
    (
        "w4a4_gemv_mx16_moe_union_sweep",
        SweepEntries::Union,
        (9, 16),
        false,
    ),
    (
        "w4a4_gemv_mx16_moe_union_sweep_sl",
        SweepEntries::Union,
        (9, 16),
        false,
    ),
    (
        "w4a4_gemv_mx8_moe_union_sweep_k64",
        SweepEntries::Union,
        (2, 8),
        true,
    ),
    (
        "w4a4_gemv_mx16_moe_union_sweep_k64",
        SweepEntries::Union,
        (9, 16),
        true,
    ),
];

/// 2026-10-23: The module of the routed W4A4 entries.
const W4A4_MOE: &str = crate::glm5next_mlp::W4A4_MOE_MODULE;

fn usize_dim(g: &PlanGroup<'_>, name: &str) -> Result<usize> {
    usize::try_from(g.dim(name)?).with_context(|| format!("dim `{name}` overflows usize"))
}

fn expect_func(k: &KernelId, module: &str, funcs: &[&str]) -> Result<()> {
    ensure!(
        k.module == module && funcs.contains(&k.func.as_str()),
        "the plan names `{k}`; this emitter launches {module}::{{{}}}",
        funcs.join(",")
    );
    Ok(())
}

fn expect_moe(g: &PlanGroup<'_>, emitter: &str) -> Result<()> {
    ensure!(
        g.block() == "moe" && matches!(g.mode, Mode::Decode | Mode::Verify | Mode::MultiSeq),
        "emitter `{emitter}` serves the main layers' `moe` block at decode, verify and \
         multi-sequence decode; group {} is {} at {:?}",
        g.index,
        g.site(),
        g.mode
    );
    Ok(())
}

fn set<T>(slot: &mut Option<T>, v: T, what: &str) -> Result<()> {
    ensure!(slot.is_none(), "the MoE site has two {what} groups");
    *slot = Some(v);
    Ok(())
}

/// 2026-10-23: `moe_router`: the router logits (`gate`), the top-k (`top_k`), or both in one
/// group (the batched router rules).
pub struct MoeRouter;

const GEMV_F32: &str = "dense_gemv_bf16_fp32out";
const FEW_F32: (&str, &str) = (
    "dense_gemv_bf16_batchm_few",
    "dense_gemv_bf16_batchm_few_fp32out",
);
const WIDE_F32: (&str, &str) = (
    "dense_gemv_bf16_batchm",
    "dense_gemv_bf16_batchm_wide_fp32out",
);

fn topk_step(g: &PlanGroup<'_>, k: &KernelId) -> Result<TopkStep<KernelId>> {
    expect_func(
        k,
        crate::glm5next_mlp::FFN_MODULE,
        &["glm5next_router_topk", "glm5next_router_topk_rank"],
    )?;
    let rank = k.func == "glm5next_router_topk_rank";
    Ok(TopkStep {
        kernel: k.clone(),
        block: topk_block(usize_dim(g, "experts")?, rank)?,
    })
}

impl GroupEmitter<GlmBlock> for MoeRouter {
    fn id(&self) -> &'static str {
        "moe_router"
    }

    fn emit(&self, g: &PlanGroup<'_>, site: &mut GlmBlock) -> Result<()> {
        expect_moe(g, self.id())?;
        let s = &mut site.moe;
        match g.locals().as_slice() {
            ["gate"] => {
                g.expect_kernels(self.id(), 1)?;
                ensure!(
                    g.group.repeat == metrale_circuit::rules::Repeat::PerRow,
                    "the M = 1 router GEMV runs once per row; the group repeats {}",
                    g.group.repeat.name()
                );
                expect_func(g.kernel(0)?, "gemv", &[GEMV_F32])?;
                set(
                    &mut s.router,
                    RouterStep::PerRow(g.kernel(0)?.clone()),
                    "router",
                )
            }
            ["top_k"] => {
                g.expect_kernels(self.id(), 1)?;
                set(&mut s.topk, topk_step(g, g.kernel(0)?)?, "top-k")
            }
            ["gate", "top_k"] => {
                g.expect_kernels(self.id(), 2)?;
                let e = g.kernel(0)?;
                let (rows, fits) = (g.rows, |lo: u64, hi: u64| (lo..=hi).contains(&g.rows));
                let ok = (e.module == FEW_F32.0 && e.func == FEW_F32.1 && fits(2, 8))
                    || (e.module == WIDE_F32.0 && e.func == WIDE_F32.1 && fits(9, 16));
                ensure!(ok, "the plan names `{e}` for {rows} router rows");
                set(&mut s.router, RouterStep::Batched(e.clone()), "router")?;
                set(&mut s.topk, topk_step(g, g.kernel(1)?)?, "top-k")
            }
            other => bail!("`moe_router` takes [gate], [top_k] or [gate,top_k]; got {other:?}"),
        }
    }
}

/// 2026-10-23: `glm_moe_w4a4`: one routed projection, gate and up together or down.
pub struct GlmMoeW4a4;

fn sweep_step(
    g: &PlanGroup<'_>,
    quant: &KernelId,
    union: Option<&KernelId>,
    sweep: &KernelId,
    k: usize,
    what: &str,
) -> Result<ProjStep<KernelId>> {
    expect_func(
        quant,
        W4A4_MOE,
        &["w4a4_quant_rows_static", "w4a4_quant_rows_static_k64"],
    )?;
    let k64 = quant.func.ends_with("_k64");
    ensure!(sweep.module == W4A4_MOE, "the plan names sweep `{sweep}`");
    let (base, scales) = match sweep.func.strip_suffix("_cs") {
        Some(b) => (b, ScaleForm::Compact),
        None => (sweep.func.as_str(), ScaleForm::Plain),
    };
    let &(_, entries, (lo, hi), sweep_k64) = SWEEPS
        .iter()
        .find(|s| s.0 == base)
        .with_context(|| format!("`{sweep}` is not a routed sweep entry this emitter knows"))?;
    ensure!(
        (lo..=hi).contains(&g.rows),
        "`{sweep}` serves {lo}..={hi} rows; the plan has {}",
        g.rows
    );
    ensure!(
        sweep_k64 == k64,
        "the {what} quantizer `{quant}` and sweep `{sweep}` disagree on the _k64 layout"
    );
    match (entries, union) {
        (SweepEntries::OwnSlots, None) => {}
        (SweepEntries::Union, Some(u)) => expect_func(
            u,
            crate::glm5next_mlp::W4A16_GEMV_MODULE,
            &["glm5next_moe_row_union"],
        )?,
        // 2026-10-23: The down sweep reads the union the gate/up group built (`finish` checks).
        (SweepEntries::Union, None) if what == "down" => {}
        (e, u) => bail!("the {what} sweep reads {e:?} entries with union builder {u:?}"),
    }
    Ok(ProjStep {
        quant: quant.clone(),
        k_store: k_store_for(k64, k, what)?,
        union: union.cloned(),
        sweep: sweep.clone(),
        entries,
        scales,
    })
}

impl GroupEmitter<GlmBlock> for GlmMoeW4a4 {
    fn id(&self) -> &'static str {
        "glm_moe_w4a4"
    }

    fn emit(&self, g: &PlanGroup<'_>, site: &mut GlmBlock) -> Result<()> {
        expect_moe(g, self.id())?;
        let n = g.group.kernels.len();
        match g.locals().as_slice() {
            ["xn_quant", "experts_gate_up"] => {
                ensure!(matches!(n, 2 | 3), "gate/up names {n} kernels");
                let union = (n == 3).then(|| g.kernel(1)).transpose()?;
                let step = sweep_step(
                    g,
                    g.kernel(0)?,
                    union,
                    g.kernel(n - 1)?,
                    usize_dim(g, "hidden")?,
                    "gate/up",
                )?;
                set(&mut site.moe.gate_up, step, "gate/up")
            }
            ["eact_quant", "experts_down"] => {
                g.expect_kernels(self.id(), 2)?;
                let k = usize_dim(g, "moe_inter")?;
                let step = sweep_step(g, g.kernel(0)?, None, g.kernel(1)?, k, "down")?;
                set(&mut site.moe.down, step, "down")
            }
            other => bail!("`glm_moe_w4a4` takes a quantized gate/up or down; got {other:?}"),
        }
    }
}

/// 2026-10-23: `glm_swiglu` over the routed experts.
pub struct GlmSwiglu;

impl GroupEmitter<GlmBlock> for GlmSwiglu {
    fn id(&self) -> &'static str {
        "glm_swiglu"
    }

    fn emit(&self, g: &PlanGroup<'_>, site: &mut GlmBlock) -> Result<()> {
        expect_moe(g, self.id())?;
        g.expect_locals(self.id(), &["experts_clamp", "experts_act"])?;
        g.expect_kernels(self.id(), 1)?;
        let k = g.kernel(0)?;
        let ffn = crate::glm5next_mlp::FFN_MODULE;
        expect_func(
            k,
            ffn,
            &["glm5next_swiglu_clamp", "glm5next_swiglu_clamp_zero"],
        )?;
        let step = if k.func.ends_with("_zero") {
            ActStep::Zero(k.clone())
        } else {
            ActStep::Plain(k.clone())
        };
        set(&mut site.moe.act, step, "SwiGLU")
    }
}

/// 2026-10-23: `moe_blend` (one block per row) and `moe_blend_split` (one block per `ACT_BLOCK`
/// elements of `hidden`): the combine.
pub struct MoeBlend {
    pub split: bool,
}

impl GroupEmitter<GlmBlock> for MoeBlend {
    fn id(&self) -> &'static str {
        if self.split {
            "moe_blend_split"
        } else {
            "moe_blend"
        }
    }

    fn emit(&self, g: &PlanGroup<'_>, site: &mut GlmBlock) -> Result<()> {
        expect_moe(g, self.id())?;
        g.expect_locals(self.id(), &["blend"])?;
        g.expect_kernels(self.id(), 1)?;
        let k = g.kernel(0)?;
        expect_func(
            k,
            crate::glm5next_mlp::FFN_MODULE,
            &["glm5next_moe_combine"],
        )?;
        let step = BlendStep {
            kernel: k.clone(),
            blocks_per_row: combine_blocks_per_row(self.split, usize_dim(g, "hidden")?),
        };
        set(&mut site.moe.blend, step, "combine")
    }
}

impl MoeSlots {
    /// 2026-10-23: The site's steps; an error naming the first step no group filled.
    pub fn finish(self, g: &PlanGroup<'_>) -> Result<MoeSteps<KernelId>> {
        let need = |what: &str| format!("the MoE site's plan has no {what} group");
        let (gate_up, down) = (
            self.gate_up.with_context(|| need("gate/up"))?,
            self.down.with_context(|| need("down"))?,
        );
        ensure!(
            down.entries == SweepEntries::OwnSlots || gate_up.union.is_some(),
            "the down sweep reads the rows' union, which no gate/up group builds"
        );
        ensure!(
            gate_up.entries == down.entries,
            "gate/up reads {:?} entries and down {:?}",
            gate_up.entries,
            down.entries
        );
        Ok(MoeSteps {
            rows: usize::try_from(g.rows)?,
            hidden: usize_dim(g, "hidden")?,
            moe_inter: usize_dim(g, "moe_inter")?,
            top_k: usize_dim(g, "top_k")?,
            experts: usize_dim(g, "experts")?,
            router: self.router.with_context(|| need("router"))?,
            topk: self.topk.with_context(|| need("top-k"))?,
            gate_up,
            act: self.act.with_context(|| need("SwiGLU"))?,
            down,
            blend: self.blend.with_context(|| need("combine"))?,
            shared: self.shared.unwrap_or(crate::glm5next_dense_seg::DenseSite::Planned {
                segment: false,
                sms: None,
            }),
        })
    }
}
