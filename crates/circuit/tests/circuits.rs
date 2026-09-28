// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: The two checked-in circuits instantiate as their checkpoints are built, and the
//! rule set is consistent with the kernel tree, the golden plans and the audit.
//!
//! Owner: metrale-circuit tests.
//! Invariants: none beyond the types.

mod common;

use std::collections::{BTreeMap, BTreeSet};

use metrale_circuit::{Circuit, Format, Instance, LayerKind, Mode, Numerics, Section};

fn instance(recipe: &str) -> Instance {
    common::instances()
        .into_iter()
        .find(|i| i.recipe == recipe)
        .expect("instance")
}

fn dense() -> (Instance, Circuit) {
    let i = instance("qwen3.8/qwen3.8-27b-nvfp4-unsloth");
    let c = common::load(&i).circuit;
    (i, c)
}

fn moe() -> (Instance, Circuit) {
    let i = instance("qwen3.6/qwen3.6-35b-a3b-fp8-bf16head");
    let c = common::load(&i).circuit;
    (i, c)
}

fn weight(c: &Circuit, id: &str) -> Option<Format> {
    c.nodes[c.node(id).unwrap_or_else(|| panic!("no node {id}"))].weight
}

fn blocks_by_template(c: &Circuit, section: Section) -> BTreeMap<String, usize> {
    let mut out = BTreeMap::new();
    for b in c.blocks.iter().filter(|b| b.section == section) {
        *out.entry(b.template.clone()).or_insert(0) += 1;
    }
    out
}

#[test]
fn dense_circuit_has_the_checkpoint_layers_formats_and_draft_head() {
    let (_, c) = dense();
    assert_eq!(c.layer_kinds.len(), 64);
    let attn: Vec<usize> = (0..64)
        .filter(|&i| c.layer_kinds[i] == LayerKind::FullAttention)
        .collect();
    assert_eq!(attn, (0..16).map(|k| 4 * k + 3).collect::<Vec<_>>());
    assert_eq!(
        blocks_by_template(&c, Section::Main),
        BTreeMap::from([
            ("attn".into(), 16),
            ("dense_ffn".into(), 64),
            ("embed".into(), 1),
            ("gdn".into(), 48),
            ("head".into(), 1)
        ])
    );
    assert_eq!(
        blocks_by_template(&c, Section::Draft),
        BTreeMap::from([
            ("attn".into(), 1),
            ("dense_ffn".into(), 1),
            ("mtp_in".into(), 1),
            ("mtp_out".into(), 1)
        ])
    );
    let nvfp4 = Some(Format::Nvfp4 { group: 16 });
    assert_eq!(weight(&c, "l0.gdn.qkvz"), nvfp4);
    assert_eq!(
        weight(&c, "l3.attn.q"),
        nvfp4,
        "FP8 per-channel q is requantized to NVFP4"
    );
    assert_eq!(
        weight(&c, "l60.dense_ffn.down"),
        nvfp4,
        "the FP8 MLPs of 56-63 too"
    );
    assert_eq!(weight(&c, "l0.gdn.ba"), Some(Format::Bf16));
    assert_eq!(weight(&c, "head.lm_head"), Some(Format::Bf16));
    assert_eq!(weight(&c, "draft.attn.q"), Some(Format::Bf16));
    assert_eq!(weight(&c, "draft.mtp_out.lm_head"), nvfp4);
    let binding = &c.nodes[c.node("draft.dense_ffn.gate_up").unwrap()].binding;
    assert_eq!(
        binding,
        &["mtp.layers.0.mlp.gate_proj", "mtp.layers.0.mlp.up_proj"]
    );
    let qkvz = c.edge("l0.gdn.qkvz").unwrap();
    assert_eq!(c.edges[qkvz].dim_value, 16384);
    assert_eq!(c.edges[c.edge("l0.gdn.conv").unwrap()].format, Format::F32);
}

#[test]
fn each_layer_output_edge_is_the_next_layer_input_edge() {
    let (_, c) = dense();
    for i in 0..63 {
        let out = c.edge(&format!("l{i}.dense_ffn.y")).unwrap();
        let next = c.blocks.iter().find(|b| b.layer == Some(i + 1)).unwrap();
        assert_eq!(next.stream_in, Some(out), "layer {}", i + 1);
        let readers: BTreeSet<&str> = c.edges[out]
            .consumers
            .iter()
            .map(|&n| c.nodes[n].local.as_str())
            .collect();
        assert_eq!(readers, BTreeSet::from(["input_norm", "add"]), "layer {i}");
    }
    let last = c.edge("l63.dense_ffn.y").unwrap();
    let readers: Vec<&str> = c.edges[last]
        .consumers
        .iter()
        .map(|&n| c.nodes[n].id.as_str())
        .collect();
    assert_eq!(readers, ["head.final_norm", "draft.mtp_in.hidden_norm"]);
}

#[test]
fn moe_circuit_has_the_checkpoint_layers_formats_and_experts() {
    let (inst, c) = moe();
    assert_eq!(c.layer_kinds.len(), 40);
    assert_eq!(
        c.layer_kinds
            .iter()
            .filter(|k| **k == LayerKind::FullAttention)
            .count(),
        10
    );
    assert_eq!(blocks_by_template(&c, Section::Main)["moe_ffn"], 40);
    let block = Some(Format::parse("fp8/block128x128").unwrap());
    assert_eq!(weight(&c, "l0.gdn.qkvz"), block);
    assert_eq!(weight(&c, "l7.attn.o"), block);
    assert_eq!(weight(&c, "l0.moe_ffn.experts_gate_up"), block);
    assert_eq!(weight(&c, "l0.moe_ffn.router"), Some(Format::Bf16));
    assert_eq!(weight(&c, "l0.moe_ffn.shared_gate"), Some(Format::Bf16));
    assert_eq!(
        weight(&c, "draft.moe_ffn.experts_down"),
        block,
        "the MTP experts stay FP8"
    );
    assert_eq!(weight(&c, "draft.attn.k"), Some(Format::Bf16));
    let egu = c.edge("l0.moe_ffn.egu").unwrap();
    assert_eq!(c.edges[egu].rows.text(), "n*top_k");
    let mut dims = c.dims.clone();
    dims.insert("n".into(), 4);
    assert_eq!(c.edges[egu].rows.eval(&dims), Ok(32));
    assert_eq!(c.edges[egu].dim_value, 1024);
    assert_eq!(
        c.edges[c.edge("l0.moe_ffn.eact").unwrap()].format,
        Format::F32
    );
    assert_eq!(inst.shape.dims["experts"], 256);
}

#[test]
fn shared_blocks_are_identical_in_both_circuits() {
    let dense: toml::Table =
        toml::from_str(&common::read("kernels/circuits/qwen3_5.toml")).unwrap();
    let moe: toml::Table =
        toml::from_str(&common::read("kernels/circuits/qwen3_6_moe.toml")).unwrap();
    for name in ["embed", "gdn", "attn", "head", "mtp_in", "mtp_out"] {
        assert_eq!(
            dense["block"][name], moe["block"][name],
            "block `{name}` drifted between the circuits"
        );
    }
}

#[test]
fn every_rule_kernel_is_compiled_by_a_golden_target() {
    let mut modules = Vec::new();
    let mut rules = Vec::new();
    for inst in common::instances().iter().filter(|i| i.golden) {
        modules.push(common::target_modules(inst));
        rules = common::load(inst).rules;
    }
    for r in &rules {
        for k in &r.kernels {
            assert!(
                modules.iter().any(|m| common::present(m, k)),
                "rule `{}` names {k}, which no golden target compiles",
                r.id
            );
        }
    }
}

#[test]
fn every_reference_rule_is_used_by_a_golden_plan_and_every_lever_moves_one() {
    let plans = common::golden_plans();
    let used: BTreeSet<String> = plans
        .iter()
        .flat_map(|(_, text)| {
            text.split(" rule=")
                .skip(1)
                .map(|t| t.split_whitespace().next().unwrap_or_default().to_string())
                .collect::<Vec<_>>()
        })
        .collect();
    let inst = &common::instances()[0];
    let rules = common::load(inst).rules;
    for r in &rules {
        match &r.numerics {
            Numerics::Differs { .. } => assert!(
                !used.contains(&r.id),
                "`{}` selected without its lever",
                r.id
            ),
            _ => assert!(
                used.contains(&r.id),
                "rule `{}` is used by no golden plan",
                r.id
            ),
        }
    }
    let (d, loaded) = (
        instance("qwen3.8/qwen3.8-27b-nvfp4-unsloth"),
        common::load(&instance("qwen3.8/qwen3.8-27b-nvfp4-unsloth")),
    );
    let avail = common::available(&d, &loaded.rules);
    for (lever, mode, rows) in [
        ("gdn_fused_norm", Mode::Decode, 1),
        ("decode_fused_silu", Mode::Decode, 1),
        ("gdn_fused_verify", Mode::Verify, 2),
        ("w4a4_downcast", Mode::Verify, 4),
    ] {
        let base = metrale_circuit::render_plan(&d, &loaded, &avail, mode, rows).unwrap();
        let mut on = d.clone();
        on.policy.opt_in_levers.insert(lever.to_string());
        let with = metrale_circuit::render_plan(&on, &loaded, &avail, mode, rows).unwrap();
        let body = |t: &str| {
            t.lines()
                .filter(|l| l.starts_with('g'))
                .map(str::to_string)
                .collect::<Vec<_>>()
        };
        assert_ne!(body(&base), body(&with), "lever {lever} changes no group");
        assert!(with.contains(&format!("differs({lever})")), "{lever}");
    }
}

#[test]
fn the_kv_write_runs_before_attention_reads_it() {
    for (name, text) in common::golden_plans() {
        let mut pending: BTreeMap<String, usize> = BTreeMap::new();
        for (i, line) in text.lines().enumerate() {
            let Some(block) = line.split(' ').nth(1) else {
                continue;
            };
            if line.contains("[kv_write]") {
                pending.insert(block.to_string(), i);
            } else if line.contains("[attend]") {
                assert!(
                    pending.contains_key(block),
                    "{name}: {block} attends before its KV write"
                );
            }
        }
    }
}

#[test]
fn every_cited_path_exists_and_holds_the_cited_line() {
    let rules = common::load(&common::instances()[0]).rules;
    let prefixes = [
        ("ml/", "crates/model-layers/src/layers/"),
        ("me/", "crates/model-engine/src/model/trait_impl/"),
        ("mm/", "crates/model-engine/src/model/"),
        ("k/", "kernels/gb10/common/"),
        ("crates/", "crates/"),
        ("kernels/", "kernels/"),
    ];
    let mut checked = 0;
    for r in &rules {
        for token in r
            .cite
            .split(|c: char| c.is_whitespace() || c == ';' || c == '(' || c == ')')
        {
            let Some((path, lines)) = token.split_once(':') else {
                continue;
            };
            let Some((short, long)) = prefixes.iter().find(|(p, _)| path.starts_with(p)) else {
                continue;
            };
            let file = format!("{long}{}", &path[short.len()..]);
            let text = std::fs::read_to_string(common::root().join(&file))
                .unwrap_or_else(|_| panic!("rule `{}` cites {file}, which does not exist", r.id));
            let max = lines
                .split([',', '-'])
                .filter_map(|n| {
                    n.trim_end_matches(|c: char| !c.is_ascii_digit())
                        .parse::<usize>()
                        .ok()
                })
                .max();
            if let Some(max) = max {
                assert!(
                    max <= text.lines().count(),
                    "rule `{}` cites {file}:{max}, past its end",
                    r.id
                );
            }
            checked += 1;
        }
    }
    assert!(
        checked > 100,
        "only {checked} citations parsed; the cite parser is not seeing them"
    );
}

#[test]
fn the_routing_audit_lists_every_rule_with_its_class_and_citation() {
    let audit = common::read("kernels/circuits/ROUTING-AUDIT.md");
    let rules = common::load(&common::instances()[0]).rules;
    for r in &rules {
        let class = match &r.numerics {
            Numerics::Differs { lever } => format!("differs ({lever})"),
            other => other.class().to_string(),
        };
        let row = format!("| `{}` | {class} | {} |", r.id, r.cite);
        assert!(
            audit.contains(&row),
            "ROUTING-AUDIT.md lacks, or has a stale row for:\n{row}"
        );
    }
    let listed = audit
        .lines()
        .skip_while(|l| !l.starts_with("| Rule | Numerics |"))
        .skip(2)
        .take_while(|l| l.starts_with('|'))
        .count();
    assert_eq!(
        listed,
        rules.len(),
        "the audit lists a rule FUSIONS.toml does not have"
    );
}
