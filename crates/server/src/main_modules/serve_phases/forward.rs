// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: `--forward`: the forward selection `serve_load` applies to the built model, with
//! the circuit instance resolved from the checkpoint and the kernel target.
//!
//! Owner: server startup (`met serve`).
//! Invariants:
//! - `circuit` names exactly the instance for this checkpoint on this target, or boot fails.
//! - 2026-10-23: resolved once at boot ([`resolve_forward`]): an explicit `--forward` is kept;
//!   unset, the mode is `circuit` where an instance serves the checkpoint on this target and
//!   `legacy` otherwise, and boot logs which and why (a declared mode, never a silent fallback).

use anyhow::{Context, Result, bail};
use metrale_circuit::Instance;
use metrale_model_engine::traits::ForwardSelect;
use metrale_model_layers::circuit_exec::{Fusions, TargetModules, sources};

use crate::cli::{self, flag_values::ForwardArg};

/// 2026-09-28: `hw/model/quant` of a compiled kernel target, as INSTANCES.toml spells it.
pub(crate) fn circuit_target(t: &metrale_core::target::KernelTarget) -> Result<String> {
    let sm = metrale_core::arch::parse_sm_arch(t.arch)
        .with_context(|| format!("kernel arch `{}` is not an sm_ arch", t.arch))?;
    let hw = metrale_core::arch::target_hint((sm.major, sm.minor))
        .with_context(|| format!("no kernels/<hw>/ ships for {}", t.arch))?;
    Ok(format!("{hw}/{}/{}", t.model, t.quant))
}

/// 2026-10-23: The forward boot runs, why, and (for a circuit mode) its instance.
#[derive(Debug)]
pub(crate) struct ResolvedForward {
    pub mode: ForwardArg,
    pub reason: String,
    pub instance: Option<Instance>,
}

/// 2026-10-23: `--forward` resolved from the explicit flag (`None`: unset), the positional
/// checkpoint id, the circuit target of this build (`Err`: the build has none, with why) and the
/// serve's tensor-parallel ranks. An explicit circuit mode without an instance is refused.
pub(crate) fn resolve_forward(
    explicit: Option<ForwardArg>,
    checkpoint: Option<&str>,
    target: Result<&str, &str>,
    ranks: u64,
) -> Result<ResolvedForward> {
    let legacy = |reason: String| ResolvedForward {
        mode: ForwardArg::Legacy,
        reason,
        instance: None,
    };
    match explicit {
        Some(ForwardArg::Legacy) => Ok(legacy("--forward legacy (explicit)".into())),
        Some(mode) => {
            let checkpoint =
                checkpoint.context("--forward circuit needs the checkpoint id as the model argument")?;
            let target = match target {
                Ok(t) => t,
                Err(why) => bail!("--forward {}: {why}", mode.name()),
            };
            Ok(ResolvedForward {
                mode,
                reason: format!("--forward {} (explicit)", mode.name()),
                instance: Some(sources::instance_for(checkpoint, target, ranks)?),
            })
        }
        None => {
            let Some(checkpoint) = checkpoint else {
                return Ok(legacy(
                    "--forward unset and no checkpoint id (the positional MODEL argument) to look \
                     up a circuit instance"
                        .into(),
                ));
            };
            let target = match target {
                Ok(t) => t,
                Err(why) => return Ok(legacy(format!("--forward unset; no circuit target: {why}"))),
            };
            Ok(match sources::find_instance(checkpoint, target, ranks)? {
                Some(i) => ResolvedForward {
                    mode: ForwardArg::Circuit,
                    reason: format!(
                        "--forward unset; circuit instance `{}` serves `{checkpoint}` on `{target}` \
                         at {ranks} rank(s)",
                        i.recipe
                    ),
                    instance: Some(i),
                },
                None => legacy(format!(
                    "--forward unset; no circuit instance serves `{checkpoint}` on `{target}` at \
                     {ranks} rank(s) (kernels/circuits/INSTANCES.toml)"
                )),
            })
        }
    }
}

/// 2026-09-28: The selection `--forward` asks for; `config_json` is the served checkpoint's.
/// 2026-10-23: `verify_rows`, the verify widths the serve's speculation runs.
pub(crate) fn forward_select(
    args: &cli::ServeArgs,
    ptx_set: &metrale_kernels::TargetPtxSet,
    config_json: &str,
    verify_rows: Vec<u64>,
) -> Result<ForwardSelect> {
    let target = circuit_target(&ptx_set.target).map_err(|e| format!("{e:#}"));
    let r = resolve_forward(
        args.forward,
        args.model.as_deref(),
        target.as_deref().map_err(String::as_str),
        u64::try_from(args.tp_size)?,
    )?;
    tracing::info!("forward: {} ({})", r.mode.name(), r.reason);
    let fusions = match r.mode {
        ForwardArg::Legacy => return Ok(ForwardSelect::Legacy),
        ForwardArg::Circuit => Fusions::All,
        ForwardArg::CircuitReference => Fusions::ReferenceOnly,
    };
    let instance = r.instance.context("a circuit forward resolves with its instance")?;
    Ok(ForwardSelect::Circuit {
        instance: Box::new(instance),
        fusions,
        modules: TargetModules(ptx_set.modules.clone()),
        config_json: config_json.to_string(),
        verify_rows,
    })
}

#[cfg(test)]
#[path = "forward_tests.rs"]
mod tests;
