// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The emitter of a KDA mixer's own groups (`kda`): the pack (`[qkv]`), the conv
//! (`[short_conv,conv_ckpt,l2]`: one row's update, or a snapshotting verify's window advances,
//! token conv and live-window advance), the W8A8 gate chain (`[fa_quant,f_b,gates,ga_quant,g_b]`),
//! the recurrence (`[recur]`) and the output norm when it is its own group (`[out_norm]`); and
//! (2026-10-23, [`GlmW4a16`]) the block's `glm_w4a16` groups: the six projections
//! (`[q,k,v,f_a,b,g_a]`), `[o]` at one row and `[out_norm,o]` at two.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Recognizes the entries `glm5next_kda::plan` launches (`glm5next_kda::entry`) by the plan's
//!   `(module, func)`; refuses any other, never guessing a launch form.
//! - A verify's conv names one window advance per snapshot (rows - 1), all one entry.

use anyhow::{Context, Result, bail, ensure};
use metrale_circuit::{KernelId, Mode};
use metrale_model_layers::circuit_exec::family::{GroupEmitter, PlanGroup};

use super::GlmBlock;
use crate::glm5next_kda::entry;
use crate::glm5next_kda::plan::{ConvStep, KdaSteps, RecurStep};
use crate::glm5next_w4a16_plan::{W4a16Step, entry as w4a16};

/// 2026-10-23: The KDA block's own nodes this emitter compiles, by group.
const PACK: [&str; 1] = ["qkv"];
const CONV: [&str; 3] = ["short_conv", "conv_ckpt", "l2"];
const GATES: [&str; 5] = ["fa_quant", "f_b", "gates", "ga_quant", "g_b"];
const RECUR: [&str; 1] = ["recur"];
const OUT_NORM: [&str; 1] = ["out_norm"];
const SIX: [&str; 6] = ["q", "k", "v", "f_a", "b", "g_a"];
const O: [&str; 1] = ["o"];
const NORM_O: [&str; 2] = ["out_norm", "o"];

/// 2026-10-23: The projections and their norms of the block: the `glm_w4a16` sites the layer runs.
const PROJECTIONS: [&str; 8] = ["q", "k", "v", "f_a", "b", "g_a", "o", "out_norm"];

/// 2026-10-23: A KDA mixer's steps as its groups fill them.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct KdaSlots {
    pub pack: Option<KernelId>,
    pub gates: Option<KernelId>,
    pub conv: Option<ConvStep<KernelId>>,
    pub recur: Option<RecurStep<KernelId>>,
    pub out_norm: Option<KernelId>,
    pub six: Option<W4a16Step<KernelId>>,
    pub o: Option<W4a16Step<KernelId>>,
}

/// 2026-10-23: Whether a group is one of the KDA block's own groups this emitter compiles.
pub fn is_site_group(g: &PlanGroup<'_>) -> bool {
    g.block() == "kda" && ["kda", "glm_w4a16"].contains(&g.group.emitter.as_str())
}

/// 2026-10-23: Whether a group is one of the block's projection groups the layer runs.
pub fn is_legacy_group(g: &PlanGroup<'_>) -> bool {
    g.block() == "kda" && g.locals().iter().all(|l| PROJECTIONS.contains(l))
}

fn is(k: &KernelId, (module, func): (&str, &str)) -> bool {
    k.module == module && k.func == func
}

fn want(k: &KernelId, e: (&str, &str)) -> Result<KernelId> {
    ensure!(
        is(k, e),
        "the plan names `{k}`; this emitter launches {}::{}",
        e.0,
        e.1
    );
    Ok(k.clone())
}

fn set<T>(slot: &mut Option<T>, v: T, what: &str) -> Result<()> {
    ensure!(slot.is_none(), "the KDA site has two {what} groups");
    *slot = Some(v);
    Ok(())
}

/// 2026-10-23: `kda` over a KDA mixer's own groups.
pub struct Kda;

impl GroupEmitter<GlmBlock> for Kda {
    fn id(&self) -> &'static str {
        "kda"
    }

    fn emit(&self, g: &PlanGroup<'_>, site: &mut GlmBlock) -> Result<()> {
        ensure!(
            matches!(g.mode, Mode::Decode | Mode::Verify),
            "`kda` runs one sequence's decode or verify here; the plan is {:?}",
            g.mode
        );
        ensure!(
            g.block() == "kda",
            "`kda` serves the `kda` block; group {} is {}",
            g.index,
            g.site()
        );
        let s = &mut site.kda;
        let locals = g.locals();
        let one = |e| -> Result<KernelId> {
            g.expect_kernels(self.id(), 1)?;
            want(g.kernel(0)?, e)
        };
        if locals == PACK {
            set(&mut s.pack, one(entry::PACK)?, "pack")
        } else if locals == GATES {
            set(&mut s.gates, one(entry::GATES_ROWS)?, "gate-chain")
        } else if locals == OUT_NORM {
            set(&mut s.out_norm, one(entry::O_NORM)?, "output-norm")
        } else if locals == RECUR {
            g.expect_kernels(self.id(), 1)?;
            let k = g.kernel(0)?;
            let step = if is(k, entry::RECUR_REG) {
                RecurStep::Row(k.clone())
            } else {
                RecurStep::Snap(want(k, entry::RECUR_SNAP)?)
            };
            set(&mut s.recur, step, "recurrence")
        } else if locals == CONV {
            set(&mut s.conv, conv_step(g)?, "conv")
        } else {
            bail!(
                "`kda` takes a KDA pack, conv, gate, recurrence or output-norm group; got {locals:?}"
            )
        }
    }
}

/// 2026-10-23: `glm_w4a16` over a KDA mixer's projection groups: the six projections in one
/// pipelined launch, `o` on the one-row GEMV, or the output norm then `o` pipelined over two rows.
pub struct GlmW4a16;

impl GroupEmitter<GlmBlock> for GlmW4a16 {
    fn id(&self) -> &'static str {
        "glm_w4a16"
    }

    fn emit(&self, g: &PlanGroup<'_>, site: &mut GlmBlock) -> Result<()> {
        ensure!(
            matches!(g.mode, Mode::Decode | Mode::Verify),
            "`glm_w4a16` runs one sequence's decode or verify here; the plan is {:?}",
            g.mode
        );
        ensure!(
            g.block() == "kda",
            "`glm_w4a16` serves the `kda` block's projections here; group {} is {}",
            g.index,
            g.site()
        );
        let s = &mut site.kda;
        let locals = g.locals();
        let ks = &g.group.kernels;
        let pipe = |k: &KernelId| -> Result<W4a16Step<KernelId>> {
            let e = w4a16::pipe(g.rows)
                .with_context(|| format!("the pipelined GEMV has no {}-row entry", g.rows))?;
            Ok(W4a16Step::Pipe(want(k, e)?))
        };
        if locals == SIX {
            g.expect_kernels(self.id(), 1)?;
            set(&mut s.six, pipe(&ks[0])?, "six-projection")
        } else if locals == O {
            g.expect_kernels(self.id(), 1)?;
            ensure!(g.rows == 1, "a {}-row `o` on the one-row GEMV", g.rows);
            set(
                &mut s.o,
                W4a16Step::Gemv(want(&ks[0], w4a16::GEMV)?),
                "o-projection",
            )
        } else if locals == NORM_O {
            g.expect_kernels(self.id(), 2)?;
            set(&mut s.out_norm, want(&ks[0], entry::O_NORM)?, "output-norm")?;
            set(&mut s.o, pipe(&ks[1])?, "o-projection")
        } else {
            bail!(
                "`glm_w4a16` takes the KDA six projections, `o` or the output norm with `o`; got \
                 {locals:?}"
            )
        }
    }
}

/// 2026-10-23: The conv group's step: one kernel for one row; for a verify of `rows` rows,
/// `rows - 1` window advances, the token conv, the live window's advance.
fn conv_step(g: &PlanGroup<'_>) -> Result<ConvStep<KernelId>> {
    let ks = &g.group.kernels;
    if ks.len() == 1 {
        return Ok(ConvStep::Row(want(&ks[0], entry::CONV_ROW)?));
    }
    let snaps = usize::try_from(g.rows)?
        .checked_sub(1)
        .filter(|&s| s >= 1)
        .with_context(|| format!("a {}-row conv names {} kernels", g.rows, ks.len()))?;
    ensure!(
        ks.len() == snaps + 2,
        "a {}-row verify conv names {} kernels; it launches {snaps} window advances, the token \
         conv and the window advance",
        g.rows,
        ks.len()
    );
    let advance_to = want(&ks[0], entry::CONV_WINDOW_TO)?;
    for k in &ks[1..snaps] {
        ensure!(
            *k == advance_to,
            "the verify conv's snapshots name `{k}` and `{advance_to}`"
        );
    }
    Ok(ConvStep::TokensSnap {
        advance_to,
        tokens: want(&ks[snaps], entry::CONV_TOKENS)?,
        window: want(&ks[snaps + 1], entry::CONV_WINDOW)?,
    })
}

impl KdaSlots {
    /// 2026-10-23: The mixer's steps; an error naming a missing group, or a conv and recurrence
    /// of different forms.
    pub fn finish(self, g: &PlanGroup<'_>) -> Result<KdaSteps<KernelId>> {
        let need = |what: &str| format!("the layer's KDA plan has no {what} group");
        let conv = self.conv.with_context(|| need("conv"))?;
        let recur = self.recur.with_context(|| need("recurrence"))?;
        ensure!(
            matches!(
                (&conv, &recur),
                (ConvStep::Row(_), RecurStep::Row(_))
                    | (ConvStep::TokensSnap { .. }, RecurStep::Snap(_))
            ),
            "the KDA conv {conv:?} and recurrence {recur:?} are of different forms"
        );
        Ok(KdaSteps {
            six: self.six.with_context(|| need("six-projection"))?,
            out_norm: self.out_norm.with_context(|| need("output-norm"))?,
            o: self.o.with_context(|| need("o-projection"))?,
            rows: usize::try_from(g.rows)?,
            heads: usize::try_from(g.dim("lin_v_heads")?)?,
            head_dim: usize::try_from(g.dim("lin_v_dim")?)?,
            pack: self.pack.with_context(|| need("pack"))?,
            gates: self.gates.with_context(|| need("gate-chain"))?,
            conv,
            recur,
        })
    }
}

/// 2026-10-23: The KDA levers the rules do not model, at a value other than the one the rules
/// name: installation refuses them. `glm_kda_row_reg` and `glm_kda_verify_tokens` are policy
/// settings (`install::plan_settings`); the conv pack does not reach a gated shape (one row, or
/// a verify that snapshots: `Glm5NextKdaLayer::pack_for`).
pub fn unmodelled(levers: &crate::glm5next_layer::GlmLevers) -> Vec<&'static str> {
    let (k, d) = (&levers.kda, &levers.dense);
    let ahead = |floor: Option<usize>| floor.is_some_and(|f| f <= 2);
    [
        (
            k.gates_fused,
            "METRALE_GLM_KDA_GATES_FUSED (the one-row entry; the rules name the rows entry)",
        ),
        (
            !k.gates_rows,
            "METRALE_GLM_KDA_GATES_ROWS=0 (the rules name kda_gates_w8a8_k128_rows)",
        ),
        (
            !k.conv_snap_to,
            "METRALE_GLM_KDA_CONV_SNAP_TO=0 (the rules name causal_conv1d_window_advance_to)",
        ),
        (
            !d.gemv_pipe,
            "METRALE_GLM_W4A16_GEMV_PIPE=0 (the rules name w4a16_gemv_pipe)",
        ),
        (
            ahead(d.tc_min_rows) || ahead(d.dequant_min_rows),
            "METRALE_GLM_W4A16_TC_GEMM_MIN_ROWS or _DEQUANT_GEMM_MIN_ROWS at 1 or 2 (ahead of the \
             GEMV the rules name)",
        ),
        (
            d.batch_invariant,
            "METRALE_GLM_BATCH_INVARIANT (the one-row row-tile order)",
        ),
    ]
    .into_iter()
    .filter_map(|(set, what)| set.then_some(what))
    .collect()
}

/// 2026-10-23: Compile and bind layer `layer`'s KDA mixer `kda` from each gated plan of `plans`
/// into `into` (`None` clears); returns the executed sites, each `layer L n=R: <site>`.
pub fn install(
    gpu: &dyn metrale_gpu_runtime::gpu::GpuBackend,
    layer: usize,
    plans: Option<metrale_model_layers::circuit_exec::family::RankPlans<'_>>,
    kda: &crate::glm5next_kda::Glm5NextKdaLayer,
    into: &crate::glm5next_kda::plan::InstalledKda,
) -> Result<Vec<String>> {
    let Some(rp) = plans else {
        into.set(Vec::new());
        return Ok(Vec::new());
    };
    let refused = unmodelled(&kda.levers);
    ensure!(
        refused.is_empty(),
        "GLM layer {layer}: the plan's KDA rules do not model {}",
        refused.join("; ")
    );
    ensure!(
        crate::glm5next_w4a16_dense::enabled(),
        "GLM layer {layer}: the plan's KDA projections are NVFP4 (--dense-quantization w4a16); \
         this serve runs another dense tier"
    );
    // 2026-10-23: The W4A16 pipe's block cap reads the SM count (`w4a16_gemv_pipe_cap`).
    let sms = gpu.sm_count()?;
    let mut next = Vec::new();
    for plan in rp.plans {
        if !super::install::GATED.contains(&(plan.mode, plan.rows)) {
            continue;
        }
        let site = super::compile_kda(rp.circuit, plan, layer)?;
        let s = site.steps;
        ensure!(
            (s.heads, s.head_dim) == (kda.cfg.heads, kda.cfg.head_dim),
            "GLM layer {layer}: the plan's KDA heads {} x {} are not the layer's {} x {}",
            s.heads,
            s.head_dim,
            kda.cfg.heads,
            kda.cfg.head_dim
        );
        let rows = s.rows;
        let bound = crate::glm5next_kda::plan::KdaPlan {
            steps: s.map(|k| metrale_model_layers::circuit_exec::family::plan_kernel(gpu, &k))?,
            sms,
        };
        let sites = site
            .executed
            .iter()
            .map(|e| format!("layer {layer} n={rows}: {e}"))
            .collect();
        next.push((bound, sites));
    }
    let sites = next
        .iter()
        .flat_map(|(_, s): &(_, Vec<String>)| s.clone())
        .collect();
    into.set(next);
    Ok(sites)
}

#[cfg(test)]
#[path = "kda_tests.rs"]
mod tests;
