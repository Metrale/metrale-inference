// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: The circuit texts this binary was built from: INSTANCES.toml, the circuits, their
//! block libraries, the precision tables and each hardware's FUSIONS.toml, embedded so a plan
//! matches the kernels compiled in. `met circuit` and the executor both read them here.
//!
//! Owner: model-layers circuit executor.
//! Invariants:
//! - One copy of every text: the CLI and the executor never embed their own.
//! - A lookup of a name the tables lack is an error, never a fallback to another entry.

use anyhow::{Result, anyhow, bail};
use metrale_circuit::{ArchShape, Instance, Policy, PrecisionSpec, Sources};

/// 2026-09-28: kernels/circuits/INSTANCES.toml as built.
pub const INSTANCES: &str = include_str!("../../../../kernels/circuits/INSTANCES.toml");

/// 2026-09-28: Every circuit an instance can name, by arch.
pub const CIRCUITS: [(&str, &str); 3] = [
    (
        "qwen3_5",
        include_str!("../../../../kernels/circuits/qwen3_5.toml"),
    ),
    (
        "qwen3_6_moe",
        include_str!("../../../../kernels/circuits/qwen3_6_moe.toml"),
    ),
    (
        "nemotron_h",
        include_str!("../../../../kernels/circuits/nemotron_h.toml"),
    ),
];

/// 2026-09-28: Every precision table an instance can name.
pub const PRECISION: [(&str, &str); 2] = [
    (
        "qwen3.6-35b-a3b-nvfp4-declared",
        include_str!("../../../../kernels/circuits/precision/qwen3.6-35b-a3b-nvfp4-declared.toml"),
    ),
    (
        "nemotron-3.5-lightning-30b-a3b-nvfp4",
        include_str!(
            "../../../../kernels/circuits/precision/nemotron-3.5-lightning-30b-a3b-nvfp4.toml"
        ),
    ),
];

/// 2026-09-28: Every checkpoint plan fixture an instance can name.
pub const CHECKPOINTS: [(&str, &str); 2] = [
    (
        "unsloth--Qwen3.8-27B-NVFP4",
        include_str!("../../../../kernels/circuits/checkpoints/unsloth--Qwen3.8-27B-NVFP4.toml"),
    ),
    (
        "Qwen--Qwen3.6-35B-A3B-FP8",
        include_str!("../../../../kernels/circuits/checkpoints/Qwen--Qwen3.6-35B-A3B-FP8.toml"),
    ),
];

/// 2026-09-28: Every block library a circuit can include.
pub const BLOCKS: [(&str, &str); 1] = [(
    "qwen3_hybrid",
    include_str!("../../../../kernels/circuits/blocks/qwen3_hybrid.toml"),
)];

/// 2026-09-28: FUSIONS.toml per hardware.
pub const FUSIONS: [(&str, &str); 1] = [(
    "gb10",
    include_str!("../../../../kernels/gb10/common/FUSIONS.toml"),
)];

/// 2026-09-28: The entry `key` of `table`; `what` names the table in the error.
pub fn lookup<'a>(table: &[(&str, &'a str)], key: &str, what: &str) -> Result<&'a str> {
    table
        .iter()
        .find(|(k, _)| *k == key)
        .map(|(_, v)| *v)
        .ok_or_else(|| anyhow!("{what} `{key}` is not built into this binary"))
}

/// 2026-09-28: The embedded texts `instance` is built from.
pub fn sources(instance: &Instance) -> Result<Sources<'static>> {
    let hw = instance.target.split('/').next().unwrap_or_default();
    Ok(Sources {
        circuit: lookup(&CIRCUITS, &instance.arch, "circuit")?,
        precision: match &instance.precision {
            PrecisionSpec::Table(n) => lookup(&PRECISION, n, "precision table")?,
            PrecisionSpec::Policy {
                checkpoint_plan, ..
            } => lookup(&CHECKPOINTS, checkpoint_plan, "checkpoint plan")?,
        },
        rules: lookup(&FUSIONS, hw, "FUSIONS.toml for hardware")?,
        blocks: &BLOCKS,
    })
}

/// 2026-09-28: The instance serving `recipe`.
pub fn instance(recipe: &str) -> Result<Instance> {
    let all = metrale_circuit::parse_instances(INSTANCES)?;
    let known: Vec<String> = all.iter().map(|i| i.recipe.clone()).collect();
    all.into_iter().find(|i| i.recipe == recipe).ok_or_else(|| {
        anyhow!(
            "no circuit instance for recipe `{recipe}`; kernels/circuits/INSTANCES.toml has: {}",
            known.join(", ")
        )
    })
}

/// 2026-09-28: The instances whose checkpoint and kernel target are these, all of one circuit.
/// Several recipes may serve one checkpoint on one target; 2026-10-03 the executor picks among
/// them with the live policy ([`select_instance`]), which it knows only once the model is built
/// (the served `--lm-head-dtype`, say, resolves at load).
pub fn instances_for(checkpoint: &str, target: &str) -> Result<Vec<Instance>> {
    let all = metrale_circuit::parse_instances(INSTANCES)?;
    let hits: Vec<Instance> = all
        .into_iter()
        .filter(|i| i.checkpoint == checkpoint && i.target == target)
        .collect();
    let Some(first) = hits.first() else {
        bail!(
            "no circuit instance serves checkpoint `{checkpoint}` on target `{target}` \
             (kernels/circuits/INSTANCES.toml)"
        );
    };
    if let Some(other) = hits.iter().find(|i| i.arch != first.arch) {
        bail!(
            "recipes `{}` and `{}` both serve `{checkpoint}` on `{target}` with different circuits",
            first.recipe,
            other.recipe
        );
    }
    Ok(hits)
}

/// 2026-10-03: The candidate that serves the `live` policy. Candidates reading one precision
/// source are interchangeable: the executor takes the policy, and a policy precision's tier and
/// kernel capabilities, from the process (`CircuitExec::build`), so the first serves. Otherwise
/// the candidates whose settings agree with `live` on every setting the candidates disagree on
/// (the 35B's `lm_head_dtype`, which changes its head's engine format) must read one source.
pub fn select_instance(candidates: &[Instance], live: &Policy) -> Result<Instance> {
    let Some(first) = candidates.first() else {
        bail!("no circuit instance to select from");
    };
    if candidates
        .iter()
        .all(|i| same_source(&i.precision, &first.precision))
    {
        return Ok(first.clone());
    }
    let differing: Vec<&String> = first
        .policy
        .settings
        .keys()
        .filter(|k| {
            candidates
                .iter()
                .any(|i| i.policy.settings.get(*k) != first.policy.settings.get(*k))
        })
        .collect();
    let served: Vec<&Instance> = candidates
        .iter()
        .filter(|i| {
            differing
                .iter()
                .all(|k| i.policy.settings.get(*k) == live.settings.get(*k))
        })
        .collect();
    let stated = || {
        differing
            .iter()
            .map(|k| {
                format!(
                    "{k}={}",
                    live.settings.get(*k).map_or("(unset)", |v| v.as_str())
                )
            })
            .collect::<Vec<_>>()
            .join(", ")
    };
    let Some(pick) = served.first() else {
        bail!(
            "no circuit instance for `{}` states the served {}; INSTANCES.toml has {}",
            first.checkpoint,
            stated(),
            candidates
                .iter()
                .map(|i| i.recipe.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
    };
    if let Some(other) = served
        .iter()
        .find(|i| !same_source(&i.precision, &pick.precision))
    {
        bail!(
            "recipes `{}` and `{}` both serve `{}` at {} with different precision sources",
            pick.recipe,
            other.recipe,
            first.checkpoint,
            stated()
        );
    }
    Ok((*pick).clone())
}

/// 2026-09-30: `a` and `b` read the same precision table, or the same checkpoint plan with the
/// same engine formats; the tier and capabilities of a policy are the process's own.
fn same_source(a: &PrecisionSpec, b: &PrecisionSpec) -> bool {
    match (a, b) {
        (
            PrecisionSpec::Policy {
                checkpoint_plan: pa,
                engine: ea,
                ..
            },
            PrecisionSpec::Policy {
                checkpoint_plan: pb,
                engine: eb,
                ..
            },
        ) => pa == pb && ea == eb,
        _ => a == b,
    }
}

/// 2026-09-30: The shape a served checkpoint runs: its `config.json` through the circuit's config
/// map (`metrale_circuit::map_checkpoint`, the one config-to-shape mapping), with `vocab` at the
/// served vocabulary: `ModelConfig::vocab_size` after the server caps it to the tokenizer
/// (`cap_vocab_size_to_tokenizer`). [`served_shape`] then checks it against INSTANCES.toml.
pub fn checkpoint_shape(config_json: &str, served_vocab: u64) -> Result<ArchShape> {
    let mut shape = metrale_circuit::map_checkpoint(config_json)?.shape;
    shape.dims.insert("vocab".to_string(), served_vocab);
    Ok(shape)
}

/// 2026-09-28: Every way `from_config` disagrees with the instance's stated shape.
pub fn shape_drift(stated: &ArchShape, from_config: &ArchShape) -> Vec<String> {
    let mut out = Vec::new();
    if stated.layer_kinds != from_config.layer_kinds {
        out.push(format!(
            "layer kinds: INSTANCES.toml has {} layers, config.json {} (or the kinds differ)",
            stated.layer_kinds.len(),
            from_config.layer_kinds.len()
        ));
    }
    for (k, v) in &stated.dims {
        match from_config.dims.get(k) {
            Some(c) if c == v => {}
            other => out.push(format!("{k}: INSTANCES.toml {v}, config.json {other:?}")),
        }
    }
    out
}

/// 2026-09-28: The shape a served model runs, checked against the instance's stated shape. Every
/// dim must agree except `vocab`, which the server caps to the tokenizer's vocabulary
/// (`cap_vocab_size_to_tokenizer`) and so may be smaller; the served value is the one the logits
/// are sized by, so the plan is instantiated with it.
pub fn served_shape(stated: &ArchShape, served: &ArchShape) -> Result<ArchShape> {
    let capped = |k: &str| {
        k == "vocab"
            && matches!(
                (stated.dims.get(k), served.dims.get(k)),
                (Some(s), Some(v)) if v <= s
            )
    };
    let drift: Vec<String> = shape_drift(stated, served)
        .into_iter()
        .filter(|line| !line.split(':').next().is_some_and(capped))
        .collect();
    if !drift.is_empty() {
        bail!(
            "kernels/circuits/INSTANCES.toml disagrees with the loaded model:\n  {}",
            drift.join("\n  ")
        );
    }
    Ok(served.clone())
}

#[cfg(test)]
mod served_shape_tests {
    use super::*;
    use metrale_circuit::LayerKind;

    fn shape(vocab: u64, hidden: u64) -> ArchShape {
        ArchShape {
            layer_kinds: vec![LayerKind::LinearAttention, LayerKind::FullAttention],
            dims: [("vocab".to_string(), vocab), ("hidden".to_string(), hidden)].into(),
        }
    }

    #[test]
    fn a_vocab_capped_to_the_tokenizer_is_served_and_nothing_else_may_move() {
        let stated = shape(248_320, 5120);
        assert_eq!(
            served_shape(&stated, &shape(248_077, 5120)).unwrap().dims["vocab"],
            248_077
        );
        assert_eq!(served_shape(&stated, &stated).unwrap(), stated);
        let wider = served_shape(&stated, &shape(248_321, 5120))
            .unwrap_err()
            .to_string();
        assert!(wider.contains("vocab"), "{wider}");
        let hidden = served_shape(&stated, &shape(248_077, 4096))
            .unwrap_err()
            .to_string();
        assert!(
            hidden.contains("hidden") && !hidden.contains("vocab"),
            "{hidden}"
        );
        let mut kinds = shape(248_320, 5120);
        kinds.layer_kinds.pop();
        assert!(served_shape(&stated, &kinds).is_err());
    }
}

#[cfg(test)]
mod instances_for_tests {
    use super::*;

    /// 2026-09-30: The dense checkpoint has an instance per tier; the executor takes the tier
    /// from the process, so either serves it. Before `same_source` the pair was refused.
    #[test]
    fn two_tiers_of_one_checkpoint_plan_serve_one_checkpoint() {
        let hits = instances_for("unsloth/Qwen3.8-27B-NVFP4", "gb10/qwen3.8-27b/nvfp4").unwrap();
        assert_eq!(hits.len(), 2);
        // 2026-10-03: One precision source: the first serves whatever the live settings say.
        for live in [&hits[0].policy, &hits[1].policy, &Policy::default()] {
            let got = select_instance(&hits, live).unwrap();
            assert_eq!(got.recipe, "qwen3.8/qwen3.8-27b-nvfp4-unsloth");
        }
    }

    /// 2026-10-03: Every golden instance, served at its own policy, resolves to an instance
    /// reading its precision source. The 35B's NVFP4-head instance once made every 35B circuit
    /// boot refuse: `instance_for` required all of a checkpoint's instances to agree.
    #[test]
    fn every_golden_instance_resolves_at_its_own_policy() {
        let all = metrale_circuit::parse_instances(INSTANCES).unwrap();
        for i in all.iter().filter(|i| i.golden) {
            let hits = instances_for(&i.checkpoint, &i.target).unwrap();
            let got =
                select_instance(&hits, &i.policy).unwrap_or_else(|e| panic!("{}: {e:#}", i.recipe));
            assert!(
                same_source(&got.precision, &i.precision),
                "{} resolved to {}",
                i.recipe,
                got.recipe
            );
        }
    }

    /// 2026-10-03: The 35B's two heads: the live `lm_head_dtype` picks the instance, and a head
    /// no instance states is refused by name rather than served with another head's formats.
    #[test]
    fn the_live_head_picks_the_35b_instance() {
        let hits = instances_for("Qwen/Qwen3.6-35B-A3B-FP8", "gb10/qwen3.6-35b-a3b/nvfp4").unwrap();
        let live = |head: &str| {
            let mut p = hits[0].policy.clone();
            p.settings.insert("lm_head_dtype".into(), head.into());
            p
        };
        for (head, recipe) in [
            ("bf16", "qwen3.6/qwen3.6-35b-a3b-fp8-bf16head"),
            ("nvfp4", "qwen3.6/qwen3.6-35b-a3b-fp8-nvfp4head"),
        ] {
            assert_eq!(select_instance(&hits, &live(head)).unwrap().recipe, recipe);
        }
        let err = select_instance(&hits, &live("fp8"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("lm_head_dtype=fp8"), "{err}");
        // 2026-10-03: Order does not decide it.
        let rev: Vec<_> = hits.iter().rev().cloned().collect();
        assert_eq!(
            select_instance(&rev, &live("bf16")).unwrap().recipe,
            "qwen3.6/qwen3.6-35b-a3b-fp8-bf16head"
        );
    }

    #[test]
    fn another_checkpoint_plan_or_engine_format_is_another_source() {
        let all = metrale_circuit::parse_instances(INSTANCES).unwrap();
        let find = |r: &str| {
            all.iter()
                .find(|i| i.recipe == r)
                .unwrap()
                .precision
                .clone()
        };
        let (nvfp4, declared) = (
            find("qwen3.8/qwen3.8-27b-nvfp4-unsloth"),
            find("qwen3.8/qwen3.8-27b-nvfp4-unsloth-declared"),
        );
        assert!(nvfp4 != declared && same_source(&nvfp4, &declared));
        let edit = |f: &dyn Fn(&mut String, &mut Vec<(String, metrale_circuit::LinearFormats)>)| {
            let mut p = declared.clone();
            if let PrecisionSpec::Policy {
                checkpoint_plan,
                engine,
                ..
            } = &mut p
            {
                f(checkpoint_plan, engine);
            }
            p
        };
        assert!(!same_source(&nvfp4, &edit(&|plan, _| plan.push('x'))));
        assert!(!same_source(
            &nvfp4,
            &edit(&|_, engine| {
                engine.pop();
            })
        ));
        assert!(!same_source(
            &nvfp4,
            &PrecisionSpec::Table("unsloth--Qwen3.8-27B-NVFP4".into())
        ));
    }
}
