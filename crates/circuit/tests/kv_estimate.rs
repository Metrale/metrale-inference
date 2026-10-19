// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: The roofline and the footprint size the paged KV cache from the codec registry:
//! on the Qwen3.8-27B instance, switching `kv_cache_dtype` moves the attention node's bytes and
//! the per-token footprint by exactly the registry's per-token difference, for every codec and
//! pair, and a spelling no codec has is refused.
//!
//! Owner: metrale-circuit tests.
//! Invariants: none beyond the types.

mod common;

use std::collections::BTreeMap;

use metrale_circuit::hardware::estimate::footprint;
use metrale_circuit::ir::Circuit;
use metrale_circuit::kv::{KV_SETTING, parse_kv, token_bytes};
use metrale_circuit::venn::roofline::node_cost;
use metrale_circuit::{Instance, Mode, OpKind};

const QWEN: &str = "qwen3.8/qwen3.8-27b-nvfp4-unsloth";

/// 2026-10-18: Every codec and the pairs with the widest K/V spread, by registry spelling.
const PAIRS: &[&str] = &[
    "bf16",
    "fp8",
    "nvfp4",
    "turbo2",
    "turbo3",
    "turbo4",
    "turbo8",
    "bf16k_turbo2v",
    "fp8k_turbo3v",
    "turbo4k_turbo8v",
];

fn instance() -> Instance {
    common::instances()
        .into_iter()
        .find(|i| i.recipe == QWEN)
        .expect("the Qwen3.8-27B instance")
}

fn with_kv(base: &BTreeMap<String, String>, kv: &str) -> BTreeMap<String, String> {
    let mut s = base.clone();
    s.insert(KV_SETTING.to_string(), kv.to_string());
    s
}

fn per_side(c: &Circuit) -> u64 {
    c.dims["kv_heads"] * c.dims["head_dim"]
}

#[test]
fn the_attention_roofline_moves_by_the_registry_token_bytes() {
    let inst = instance();
    let c = common::load(&inst).circuit;
    let r = common::families(&inst).roofline;
    let attend = c
        .nodes
        .iter()
        .find(|n| n.op == OpKind::PagedAttention)
        .expect("an attention node");
    let base = &inst.policy.settings;
    let rows = 4;
    let cost = |kv: &str| {
        node_cost(&c, attend, Mode::MultiSeq, rows, &with_kv(base, kv), &r).map(|c| c.bytes)
    };
    let bf16 = cost("bf16").expect("bf16 sizes");
    let e = per_side(&c);
    let bf16_token = token_bytes(parse_kv("bf16").unwrap(), e).unwrap();
    // 2026-10-18: The pre-registry estimate: 2 sides x 2 B/value.
    assert_eq!(bf16_token, 4 * e);
    let scale = rows as f64 * r.context_tokens as f64;
    for &kv in PAIRS {
        let token = token_bytes(parse_kv(kv).unwrap(), e).unwrap();
        let got = cost(kv).unwrap_or_else(|err| panic!("{kv}: {err}"));
        assert_eq!(
            got - bf16,
            scale * (token as f64 - bf16_token as f64),
            "{kv}"
        );
    }
    let err = cost("fp7").expect_err("no codec is spelled fp7");
    assert!(err.to_string().contains("fp7"), "{err}");
    let mut unset = base.clone();
    unset.remove(KV_SETTING);
    let err = node_cost(&c, attend, Mode::MultiSeq, rows, &unset, &r).expect_err("unset");
    assert!(err.to_string().contains(KV_SETTING), "{err}");
}

#[test]
fn the_footprint_counts_each_attention_layer_at_the_registry_token_bytes() {
    let inst = instance();
    let c = common::load(&inst).circuit;
    let base = &inst.policy.settings;
    let layers = c
        .nodes
        .iter()
        .filter(|n| n.op == OpKind::PagedAttention)
        .count() as f64;
    let e = per_side(&c);
    let at = |kv: &str| footprint(&c, &with_kv(base, kv)).map(|f| f.kv_per_token);
    let bf16 = at("bf16").expect("bf16 sizes");
    let bf16_token = token_bytes(parse_kv("bf16").unwrap(), e).unwrap() as f64;
    for &kv in PAIRS {
        let token = token_bytes(parse_kv(kv).unwrap(), e).unwrap() as f64;
        let got = at(kv).unwrap_or_else(|err| panic!("{kv}: {err}"));
        assert_eq!(got - bf16, layers * (token - bf16_token), "{kv}");
    }
    assert!(at("fp7").expect_err("refused").contains("fp7"));
}
