// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The GLM-5.3 levers the paired in-serve A/B instrument can switch
//! (`METRALE_PAIRED_AB`, `metrale_model_layers::layers::ops::paired_ab`). Each is a kernel
//! choice that keeps the output bits; this module maps its name to the change of the layer's
//! resolved kernels, so the engine never names a GLM lever (FACE).
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Outside the instrument, or on its A arm, a layer runs its resolved kernels unchanged.
//! - A name this module does not register fails the step that reads it, on any arm, so a typo
//!   never measures A against A.

use anyhow::{Result, bail};
use metrale_model_layers::layer::{AbLevers, ForwardContext};

use crate::glm5next_layer::GlmLevers;
use crate::glm5next_mlp::Glm5NextMlpKernels;

/// 2026-10-23: The registered levers.
pub(super) const NAMES: [&str; 3] = [
    // 2026-10-23: The MoE router's few-row GEMV at one output per block
    // (`glm5next_router_gemv_npb1`): the instrument's known-small validation lever.
    "router_npb1",
    // 2026-10-23: `METRALE_GLM_SWIGLU_DOWN_FOLD` (`glm5next_swiglu_down`).
    "swiglu_down",
    // 2026-10-23: `METRALE_GLM_KDA_CONV_ONE` (`glm5next_kda/seq_tokens.rs`).
    "kda_conv_one",
];

/// 2026-10-23: `base` under this pass's instrument levers.
pub(super) fn mlp_kernels(
    base: &Glm5NextMlpKernels,
    ctx: &ForwardContext,
) -> Result<Glm5NextMlpKernels> {
    let Some(run) = ctx
        .stats
        .paired_ab
        .get(ctx.settings.telemetry.paired_ab.as_deref())?
    else {
        return Ok(*base);
    };
    apply(base, &run.spec().names, ctx.ab_levers)
}

/// 2026-10-23: `base` with every lever of `names` that `on` holds applied.
pub(super) fn apply(
    base: &Glm5NextMlpKernels,
    names: &[String],
    on: AbLevers,
) -> Result<Glm5NextMlpKernels> {
    let mut k = *base;
    for (i, name) in names.iter().enumerate() {
        match name.as_str() {
            "router_npb1" => {
                if on.has(i) {
                    k.gemv_batchm_few = k.gemv_batchm_few.with_router_npb1()?;
                }
            }
            "swiglu_down" => {
                // 2026-10-23: Without the pipe the fold never launches: B would run A.
                if !k.levers.dense.gemv_seg || !k.levers.dense.gemv_pipe {
                    bail!("METRALE_PAIRED_AB: `swiglu_down` needs the W4A16 GEMV pipe");
                }
                k.levers.dense.swiglu_down |= on.has(i);
            }
            // 2026-10-23: A KDA lever: [`kda_levers`] applies it.
            "kda_conv_one" => {}
            other => bail!(
                "METRALE_PAIRED_AB: GLM-5.3 registers no lever `{other}` (registered: {})",
                NAMES.join(", ")
            ),
        }
    }
    Ok(k)
}

/// 2026-10-23: The KDA layer's levers `base` under this pass's instrument levers; `None` when
/// they are unchanged (off, or the A arm), so the layer runs as resolved.
pub(super) fn kda_levers(base: GlmLevers, ctx: &ForwardContext) -> Result<Option<GlmLevers>> {
    let Some(run) = ctx
        .stats
        .paired_ab
        .get(ctx.settings.telemetry.paired_ab.as_deref())?
    else {
        return Ok(None);
    };
    kda_apply(base, &run.spec().names, ctx.ab_levers)
}

/// 2026-10-23: `base` with every KDA lever of `names` that `on` holds applied.
pub(super) fn kda_apply(
    base: GlmLevers,
    names: &[String],
    on: AbLevers,
) -> Result<Option<GlmLevers>> {
    let mut l = base;
    for (i, name) in names.iter().enumerate() {
        if name == "kda_conv_one" {
            // 2026-10-23: Without the verify token kernels the one launch never runs: B is A.
            if !l.kda.verify_tokens {
                bail!("METRALE_PAIRED_AB: `kda_conv_one` needs METRALE_GLM_KDA_VERIFY_TOKENS");
            }
            l.kda.conv_one |= on.has(i);
        }
    }
    Ok((l != base).then_some(l))
}

#[cfg(test)]
#[path = "ab_levers_tests.rs"]
mod tests;
