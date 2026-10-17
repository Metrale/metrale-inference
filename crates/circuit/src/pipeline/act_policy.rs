// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: The `act` requirement of a projection under the policy's
//! `activation_quantization` (the `--activation-quantization` grammar,
//! `metrale_config::ActivationQuantization`): a fixed format for the projection's family at the
//! plan's row count, the input as instantiated (`declared`), or, under `adaptive` (today's
//! routing, which the rules encode), what the plan's rule states it quantizes to.
//!
//! Owner: metrale-circuit (pipeline).
//! Invariants:
//! - The setting is required: a policy that does not state it has no projection requirement.
//! - A rule that states an activation format another than a fixed policy routes is an error,
//!   never a silent override in either direction.
//! - Roles outside the flag's families (the MTP `fc`, the Mamba2 projections) follow the
//!   adaptive rule: the flag does not route them.

use metrale_config::{ActQuantFormat, ActivationQuantization, ProjFamily};

use crate::format::{Format, Scale};
use crate::ir::{LinearRole, OpKind};

/// 2026-10-02: The flag's family of a projection op; `None` for a role it does not route.
fn family_of(op: &OpKind) -> Option<ProjFamily> {
    match op {
        OpKind::Linear(r) => match r {
            LinearRole::Qkvz | LinearRole::Ba | LinearRole::GdnOut => Some(ProjFamily::Gdn),
            LinearRole::Q | LinearRole::K | LinearRole::V | LinearRole::O => Some(ProjFamily::Attn),
            LinearRole::GateUp | LinearRole::Down => Some(ProjFamily::Ffn),
            LinearRole::SharedGateUp
            | LinearRole::SharedDown
            | LinearRole::SharedGate
            | LinearRole::SharedUp
            | LinearRole::MoeLatentIn
            | LinearRole::MoeLatentOut => Some(ProjFamily::Moe),
            LinearRole::MtpFc | LinearRole::MambaIn | LinearRole::MambaOut => None,
            // 2026-10-08: The GLM-5 projections: the engine's GLM path does not read the flag.
            LinearRole::KdaB
            | LinearRole::KdaFA
            | LinearRole::KdaFB
            | LinearRole::KdaGA
            | LinearRole::KdaGB
            | LinearRole::MlaQA
            | LinearRole::MlaQB
            | LinearRole::MlaKvA
            | LinearRole::IndexQ
            | LinearRole::IndexK
            | LinearRole::IndexWeights
            | LinearRole::IndexGate
            | LinearRole::HcMix => None,
        },
        OpKind::Router | OpKind::ExpertGateUp | OpKind::ExpertDown => Some(ProjFamily::Moe),
        OpKind::LmHead => Some(ProjFamily::LmHead),
        _ => None,
    }
}

/// 2026-10-02: The activation format a fixed rung names for a `weight` projection: FP8 with one
/// scale per row, per 128 columns where the weight is block-scaled; NVFP4 group 16.
fn fixed(rung: ActQuantFormat, edge: Format, weight: Format) -> Format {
    match rung {
        ActQuantFormat::Bf16 => Format::Bf16,
        ActQuantFormat::Fp8 => Format::Fp8E4m3 {
            scale: match weight {
                Format::Fp8E4m3 {
                    scale: Scale::Block(_, c),
                } => Scale::Group(c),
                _ => Scale::PerToken,
            },
        },
        ActQuantFormat::Nvfp4 => Format::Nvfp4 { group: 16 },
        ActQuantFormat::Declared | ActQuantFormat::Adaptive => edge,
    }
}

/// 2026-10-02: The activation format projection `op` must multiply at, `rows` rows, reading
/// `edge`, the rule stating `stated` (if anything), with a `weight` weight.
pub(super) fn required_act(
    settings: &std::collections::BTreeMap<String, String>,
    op: &OpKind,
    rows: u64,
    (edge, stated, weight): (Format, Option<Format>, Format),
) -> Result<Format, String> {
    let text = settings
        .get("activation_quantization")
        .ok_or("the policy states no `activation_quantization`")?;
    let aq = ActivationQuantization::parse(text).map_err(|e| format!("{e:#}"))?;
    let rung = family_of(op).map(|f| (f, aq.route(f, u32::try_from(rows).unwrap_or(u32::MAX))));
    match rung {
        None | Some((_, ActQuantFormat::Adaptive)) => Ok(stated.unwrap_or(edge)),
        Some((fam, r)) => {
            let want = fixed(r, edge, weight);
            match stated {
                Some(s) if s != want => Err(format!(
                    "the rule runs the activation at {s}, but `activation_quantization = {text}` \
                     routes {} at {rows} rows to {}",
                    fam.name(),
                    r.name()
                )),
                _ => Ok(want),
            }
        }
    }
}
