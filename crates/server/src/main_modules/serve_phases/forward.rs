// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: `--forward`: the forward selection `serve_load` applies to the built model, with
//! the circuit instance resolved from the checkpoint and the kernel target.
//!
//! Owner: server startup (`met serve`).
//! Invariants:
//! - `circuit` names exactly the instance for this checkpoint on this target, or boot fails.
//! - 2026-10-24: the checkpoint is the positional id, else the model's identity (the loaded
//!   config.json's circuit arch and shape against INSTANCES.toml); an unset `--forward` that
//!   resolves to legacy logs its reason at WARN.
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

/// 2026-10-23: Whether `e` is the circuit's refusal of this rank count (a part that cannot split
/// at N): an unset `--forward` then runs legacy and says why; an explicit one is refused.
fn refused_at(e: &anyhow::Error) -> bool {
    e.chain().any(|c| {
        c.downcast_ref::<metrale_error::Error>()
            .is_some_and(|x| x.code() == metrale_error::ErrorCode::CircuitRanksRefused)
    })
}

/// 2026-10-23: The forward boot runs, why, and (for a circuit mode) its instance.
#[derive(Debug)]
pub(crate) struct ResolvedForward {
    pub mode: ForwardArg,
    pub reason: String,
    pub instance: Option<Instance>,
}

/// 2026-10-24: The checkpoint id an instance is looked up by: the positional checkpoint id, else
/// the model's identity, the instance whose circuit arch and shape the loaded `config.json` maps
/// to on `target` ([`sources::checkpoints_matching`]). `Err` (the outer `Ok`): no unique
/// instance, with why.
fn identity(
    checkpoint: Option<&str>,
    config_json: Option<&str>,
    target: &str,
) -> Result<Result<String, String>> {
    if let Some(c) = checkpoint {
        return Ok(Ok(c.to_string()));
    }
    let Some(config_json) = config_json else {
        return Ok(Err(
            "no checkpoint id (the positional MODEL argument) and no config.json to look up a \
             circuit instance"
                .into(),
        ));
    };
    let (arch, ids) = match sources::checkpoints_matching(config_json, target) {
        Ok(found) => found,
        Err(e) => {
            return Ok(Err(format!(
                "the loaded config.json maps to no circuit: {e:#}"
            )));
        }
    };
    Ok(match ids.as_slice() {
        [one] => Ok(one.clone()),
        [] => Err(format!(
            "no circuit instance on `{target}` matches the loaded checkpoint (circuit arch \
             `{arch}`, its config.json shape; kernels/circuits/INSTANCES.toml)"
        )),
        many => Err(format!(
            "the loaded checkpoint (circuit arch `{arch}`) matches the instances of {many:?} on \
             `{target}`: name one as the model argument"
        )),
    })
}

/// 2026-10-23: `--forward` resolved from the explicit flag (`None`: unset), the model's identity
/// (the positional checkpoint id, else 2026-10-24 the loaded `config_json`: [`identity`]), the
/// circuit target of this build (`Err`: the build has none, with why) and the serve's
/// tensor-parallel ranks. An explicit circuit mode without an instance is refused.
pub(crate) fn resolve_forward(
    explicit: Option<ForwardArg>,
    checkpoint: Option<&str>,
    config_json: Option<&str>,
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
            let target = match target {
                Ok(t) => t,
                Err(why) => bail!("--forward {}: {why}", mode.name()),
            };
            let checkpoint = match identity(checkpoint, config_json, target)? {
                Ok(c) => c,
                Err(why) => bail!("--forward {}: {why}", mode.name()),
            };
            Ok(ResolvedForward {
                mode,
                reason: format!("--forward {} (explicit)", mode.name()),
                instance: Some(sources::instance_for(&checkpoint, target, ranks)?),
            })
        }
        None => {
            let target = match target {
                Ok(t) => t,
                Err(why) => {
                    return Ok(legacy(format!("--forward unset; no circuit target: {why}")));
                }
            };
            let checkpoint = match identity(checkpoint, config_json, target)? {
                Ok(c) => c,
                Err(why) => return Ok(legacy(format!("--forward unset; {why}"))),
            };
            let found = match sources::find_instance(&checkpoint, target, ranks) {
                Err(e) if refused_at(&e) => {
                    return Ok(legacy(format!(
                        "--forward unset; the circuit cannot run `{checkpoint}` at {ranks} rank(s): \
                         {e:#}"
                    )));
                }
                other => other?,
            };
            Ok(match found {
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
    ptx_set: &metrale_kernel_set::TargetPtxSet,
    config_json: &str,
    verify_rows: Vec<u64>,
) -> Result<ForwardSelect> {
    let target = circuit_target(&ptx_set.target).map_err(|e| format!("{e:#}"));
    let r = resolve_forward(
        args.forward,
        args.model.as_deref(),
        Some(config_json),
        target.as_deref().map_err(String::as_str),
        u64::try_from(args.tp_size)?,
    )?;
    // 2026-10-24: An unset `--forward` that falls back to legacy says why at WARN.
    if r.mode == ForwardArg::Legacy && args.forward.is_none() {
        tracing::warn!("forward: {} ({})", r.mode.name(), r.reason);
    } else {
        tracing::info!("forward: {} ({})", r.mode.name(), r.reason);
    }
    let fusions = match r.mode {
        ForwardArg::Legacy => return Ok(ForwardSelect::Legacy),
        ForwardArg::Circuit => Fusions::All,
        ForwardArg::CircuitReference => Fusions::ReferenceOnly,
    };
    let instance = r
        .instance
        .context("a circuit forward resolves with its instance")?;
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
