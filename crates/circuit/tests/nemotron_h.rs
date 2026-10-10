// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The Nemotron-H circuit (`nemotron_h`) serves three instances: Nemotron-3 Nano
//! (`hybrid_override_pattern`, no MTP), Nemotron-3 Super (the latent MoE and one MTP layer) and
//! Nemotron-3.5 Lightning (`layers_block_type`, MTP). For each, the config map reproduces the
//! instance's shape from the checkpoint's own `config.json`, the MoE layers take the block the
//! config selects, and the declared precision table gives every node the formats the checkpoint's
//! own quantization metadata declares.
//!
//! Owner: metrale-circuit tests.
//! Invariants: none beyond the types.

mod checkpoint_fixtures;
mod common;

use checkpoint_fixtures::*;
use metrale_circuit::ir::Circuit;
use metrale_circuit::{Format, Instance, Section};

const NANO: (&str, &str) = (
    "nemotron-3-nano/nemotron-3-nano-30b-a3b-nvfp4",
    "nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4",
);
const SUPER: (&str, &str) = (
    "nemotron-3-super/nemotron-3-super-120b-a12b-nvfp4",
    "nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
);
const LIGHTNING: (&str, &str) = (
    "nemotron-3.5/nemotron-3.5-lightning-30b-a3b-nvfp4",
    "nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
);

fn instance(recipe: &str) -> Instance {
    common::instances()
        .into_iter()
        .find(|i| i.recipe == recipe)
        .unwrap_or_else(|| panic!("no instance {recipe}"))
}

fn templates(c: &Circuit, section: Section) -> Vec<&str> {
    c.blocks
        .iter()
        .filter(|b| b.section == section && b.layer.is_some())
        .map(|b| b.template.as_str())
        .collect()
}

#[test]
fn the_config_map_reproduces_every_nemotron_instance_shape() {
    for (recipe, fixture_dir) in [NANO, SUPER, LIGHTNING] {
        let m = metrale_circuit::map_checkpoint(&fixture(fixture_dir).0)
            .unwrap_or_else(|e| panic!("{recipe}: {e}"));
        let inst = instance(recipe);
        assert_eq!(m.arch, inst.arch, "{recipe}");
        assert_eq!(
            m.shape, inst.shape,
            "{recipe}: INSTANCES.toml != config.json"
        );
    }
}

#[test]
fn the_moe_layers_take_the_block_the_config_selects() {
    let nano = common::load(&instance(NANO.0)).circuit;
    let sup = common::load(&instance(SUPER.0)).circuit;
    let count = |c: &Circuit, t: &str| {
        templates(c, Section::Main)
            .iter()
            .filter(|x| **x == t)
            .count()
    };
    assert_eq!((count(&nano, "moe"), count(&nano, "moe_latent")), (23, 0));
    assert_eq!((count(&sup, "moe"), count(&sup, "moe_latent")), (0, 40));
    assert_eq!((count(&nano, "mamba"), count(&sup, "mamba")), (23, 40));
    // 2026-10-10: Nano ships no MTP layer; Super's draft head is attention then the latent MoE.
    assert!(nano.blocks.iter().all(|b| b.section == Section::Main));
    let draft: Vec<&str> = sup
        .blocks
        .iter()
        .filter(|b| b.section == Section::Draft)
        .map(|b| b.template.as_str())
        .collect();
    assert_eq!(draft.first(), Some(&"mtp_in"));
    assert!(draft.contains(&"moe_latent"), "{draft:?}");
    // 2026-10-10: The latent MoE's experts run at the latent width, not the hidden width.
    let up = node(&sup, "l1.moe_latent.experts_up");
    assert_eq!(sup.edges[up.inputs[0]].dim_value, 1024);
}

#[test]
fn the_declared_tables_give_every_node_the_checkpoints_declared_formats() {
    for (recipe, fixture_dir) in [NANO, SUPER] {
        let declared = ok(fixture_dir);
        let table = common::load(&instance(recipe)).circuit;
        assert_eq!(declared.kv_cache, Some(FP8_TENSOR), "{recipe}");
        let ids = |c: &Circuit| c.nodes.iter().map(|n| n.id.clone()).collect::<Vec<_>>();
        assert_eq!(ids(&declared.circuit), ids(&table), "{recipe}");
        for n in &table.nodes {
            assert_eq!(
                node(&declared.circuit, &n.id).weight,
                n.weight,
                "{recipe} {} weight",
                n.id
            );
            if !n.inputs.is_empty() {
                assert_eq!(
                    input_format(&declared.circuit, &n.id),
                    input_format(&table, &n.id),
                    "{recipe} {} input",
                    n.id
                );
            }
        }
    }
}

#[test]
fn the_mixed_precision_exceptions_are_per_layer() {
    let nano = common::load(&instance(NANO.0)).circuit;
    let sup = common::load(&instance(SUPER.0)).circuit;
    // 2026-10-10: Nano's `exclude_modules` keeps in_proj/out_proj of six Mamba2 layers (4, 11,
    // 18, 25, 32, 41) in BF16; the other Mamba2 layers are W4A4.
    let formats = |c: &Circuit, id: &str| (node(c, id).weight, input_format(c, id));
    let fp4 = Some(NVFP4);
    assert_eq!(formats(&nano, "l0.mamba.in_proj"), (fp4, NVFP4));
    assert_eq!(
        formats(&nano, "l4.mamba.out_proj"),
        (Some(Format::Bf16), Format::Bf16)
    );
    assert_eq!(formats(&nano, "l1.moe.experts_up"), (fp4, NVFP4));
    assert_eq!(
        formats(&nano, "l1.moe.router"),
        (Some(Format::Bf16), Format::Bf16)
    );
    assert_eq!(
        formats(&nano, "head.lm_head"),
        (Some(Format::Bf16), Format::Bf16)
    );
    // 2026-10-10: Super: FP8 W8A8 for most in_proj, BF16 for the twelve the metadata leaves out.
    assert_eq!(
        formats(&sup, "l0.mamba.in_proj"),
        (Some(FP8_TENSOR), FP8_TENSOR)
    );
    assert_eq!(
        formats(&sup, "l22.mamba.in_proj"),
        (Some(Format::Bf16), Format::Bf16)
    );
}
