// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: `--forward`: the forward selection `serve_load` applies to the built model, with
//! the circuit instance resolved from the checkpoint and the kernel target.
//!
//! Owner: server startup (`met serve`).
//! Invariants:
//! - `circuit` names exactly the instance for this checkpoint on this target, or boot fails.

use anyhow::{Context, Result};
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

/// 2026-09-28: The selection `--forward` asks for; `config_json` is the served checkpoint's.
pub(crate) fn forward_select(
    args: &cli::ServeArgs,
    ptx_set: &metrale_kernels::TargetPtxSet,
    config_json: &str,
) -> Result<ForwardSelect> {
    let fusions = match args.forward {
        ForwardArg::Legacy => return Ok(ForwardSelect::Legacy),
        ForwardArg::Circuit => Fusions::All,
        ForwardArg::CircuitReference => Fusions::ReferenceOnly,
    };
    let checkpoint = args
        .model
        .as_deref()
        .context("--forward circuit needs the checkpoint id as the model argument")?;
    let instance = sources::instance_for(checkpoint, &circuit_target(&ptx_set.target)?)?;
    Ok(ForwardSelect::Circuit {
        instance: Box::new(instance),
        fusions,
        modules: TargetModules(ptx_set.modules.clone()),
        config_json: config_json.to_string(),
    })
}
