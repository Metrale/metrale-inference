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
use metrale_circuit::{ArchShape, Instance, LayerKind, Sources};

/// 2026-09-28: kernels/circuits/INSTANCES.toml as built.
pub const INSTANCES: &str = include_str!("../../../../kernels/circuits/INSTANCES.toml");

/// 2026-09-28: Every circuit an instance can name, by arch.
pub const CIRCUITS: [(&str, &str); 2] = [
    (
        "qwen3_5",
        include_str!("../../../../kernels/circuits/qwen3_5.toml"),
    ),
    (
        "qwen3_6_moe",
        include_str!("../../../../kernels/circuits/qwen3_6_moe.toml"),
    ),
];

/// 2026-09-28: Every precision table an instance can name.
pub const PRECISION: [(&str, &str); 2] = [
    (
        "qwen3.8-27b-nvfp4-unsloth",
        include_str!("../../../../kernels/circuits/precision/qwen3.8-27b-nvfp4-unsloth.toml"),
    ),
    (
        "qwen3.6-35b-a3b-fp8-bf16head",
        include_str!("../../../../kernels/circuits/precision/qwen3.6-35b-a3b-fp8-bf16head.toml"),
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
        precision: lookup(&PRECISION, &instance.precision, "precision table")?,
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

/// 2026-09-28: The instance whose checkpoint and kernel target are these. Several recipes may
/// serve one checkpoint on one target; they must then agree on the circuit and the precision
/// table, the only parts the executor takes from an instance (the policy is read live).
pub fn instance_for(checkpoint: &str, target: &str) -> Result<Instance> {
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
    if let Some(other) = hits
        .iter()
        .find(|i| i.arch != first.arch || i.precision != first.precision)
    {
        bail!(
            "recipes `{}` and `{}` both serve `{checkpoint}` on `{target}` with different \
             circuits or precision tables",
            first.recipe,
            other.recipe
        );
    }
    Ok(first.clone())
}

/// 2026-09-28: The arch shape of a model config, in the dim names the circuits read.
pub fn arch_shape(cfg: &metrale_config::ModelConfig) -> Result<ArchShape> {
    let mut layer_kinds = Vec::with_capacity(cfg.num_hidden_layers);
    for i in 0..cfg.num_hidden_layers {
        layer_kinds.push(match cfg.layer_type(i) {
            metrale_config::LayerType::LinearAttention => LayerKind::LinearAttention,
            metrale_config::LayerType::FullAttention => LayerKind::FullAttention,
            other => bail!("layer {i} is {other:?}, which no circuit models"),
        });
    }
    let dims = [
        ("hidden", cfg.hidden_size),
        ("inter", cfg.intermediate_size),
        ("vocab", cfg.vocab_size),
        ("q_heads", cfg.num_attention_heads),
        ("kv_heads", cfg.num_key_value_heads),
        ("head_dim", cfg.head_dim),
        ("lin_k_heads", cfg.linear_num_key_heads),
        ("lin_k_dim", cfg.linear_key_head_dim),
        ("lin_v_heads", cfg.linear_num_value_heads),
        ("lin_v_dim", cfg.linear_value_head_dim),
        ("experts", cfg.num_experts),
        ("top_k", cfg.num_experts_per_tok),
        ("moe_inter", cfg.moe_intermediate_size),
        ("shared_inter", cfg.shared_expert_intermediate_size),
    ]
    .into_iter()
    .filter(|(_, v)| *v > 0)
    .map(|(k, v)| (k.to_string(), v as u64))
    .collect();
    Ok(ArchShape { layer_kinds, dims })
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
