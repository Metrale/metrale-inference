// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: The op and role vocabulary round-trips its spellings, the GLM-5 additions keep
//! their qualifier rules, and latent attention's weight is `kv_b_proj`.
//!
//! Owner: metrale-circuit.
//! Invariants: none beyond the types.

use super::*;

#[test]
fn every_spelling_parses_back_to_its_op_and_role() {
    for (op, name) in PLAIN_OPS {
        assert_eq!(OpKind::parse(name, None, None), Ok(op), "{name}");
        assert_eq!(op.name(), name);
    }
    let mut seen = std::collections::BTreeSet::new();
    for name in [
        "q",
        "k",
        "v",
        "o",
        "qkvz",
        "ba",
        "gdn_out",
        "gate_up",
        "down",
        "shared_gate_up",
        "shared_down",
        "shared_gate",
        "mtp_fc",
        "mamba_in",
        "mamba_out",
        "shared_up",
        "moe_latent_in",
        "moe_latent_out",
        "kda_b",
        "kda_f_a",
        "kda_f_b",
        "kda_g_a",
        "kda_g_b",
        "mla_q_a",
        "mla_q_b",
        "mla_kv_a",
        "index_q",
        "index_k",
        "index_weights",
        "index_gate",
        "hc_mix",
    ] {
        let r = LinearRole::parse(name).unwrap_or_else(|| panic!("{name}"));
        assert_eq!(r.name(), name);
        assert!(
            seen.insert(r),
            "{name} parses to a role another spelling has"
        );
    }
    assert_eq!(LinearRole::parse("mla_kv_b"), None);
    assert_eq!(
        OpKind::parse("mla_attention", Some("o"), None),
        Err(OpParseError::StrayQualifier("mla_attention".into()))
    );
    assert_eq!(
        LayerKind::parse("deepseek_sparse_attention"),
        Some(LayerKind::SparseAttention)
    );
    assert_eq!(
        LayerKind::SparseAttention.name(),
        "deepseek_sparse_attention"
    );
}

#[test]
fn the_glm_ops_are_opaque_where_they_scan_or_read_weights() {
    for (op, heavy, weight) in [
        (OpKind::MlaAttention, true, true),
        (OpKind::IndexSelect, true, false),
        (OpKind::HcPre, true, false),
        (OpKind::KpoolCompress, false, false),
        (OpKind::HcPost, false, false),
        (OpKind::SwigluClamp, false, false),
        (OpKind::LayerNorm, false, false),
    ] {
        assert_eq!(op.is_heavy(), heavy, "{op:?}");
        assert_eq!(op.reads_linear_weight(), weight, "{op:?}");
    }
}

fn node(op: OpKind, inputs: Vec<EdgeIdx>, outputs: Vec<EdgeIdx>) -> Node {
    Node {
        id: "n".into(),
        local: "n".into(),
        op,
        inputs,
        outputs,
        weight: Some(Format::Bf16),
        binding: Vec::new(),
        params: BTreeMap::new(),
        layer: Some(0),
        block: "b".into(),
        state: Vec::new(),
    }
}

fn edge(dim: u64) -> Edge {
    Edge {
        id: format!("e{dim}"),
        format: Format::Bf16,
        rows: DimExpr::parse("n").unwrap(),
        dim: DimExpr::parse(&dim.to_string()).unwrap(),
        dim_value: dim,
        producer: None,
        consumers: Vec::new(),
        is_output: false,
        binds: None,
    }
}

#[test]
fn latent_attention_weighs_kv_b_proj_and_a_projection_its_edges() {
    let dims: BTreeMap<String, u64> = [
        ("q_heads", 64),
        ("mla_qk", 256),
        ("mla_v", 256),
        ("kv_lora", 512),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect();
    let c = Circuit {
        arch: "t".into(),
        description: String::new(),
        nodes: Vec::new(),
        edges: vec![edge(16384), edge(512), edge(4096)],
        blocks: Vec::new(),
        layer_kinds: vec![LayerKind::SparseAttention],
        dims: dims.clone(),
        states: Vec::new(),
    };
    let attend = node(OpKind::MlaAttention, vec![0, 1], vec![0]);
    assert_eq!(c.weight_shape(&attend), Some((32768, 512)));
    let proj = node(OpKind::Linear(LinearRole::O), vec![0], vec![2]);
    assert_eq!(c.weight_shape(&proj), Some((4096, 16384)));
    let mut short = c.clone();
    short.dims.remove("kv_lora");
    assert_eq!(short.weight_shape(&attend), None);
    assert_eq!(short.weight_shape(&proj), Some((4096, 16384)));
}
