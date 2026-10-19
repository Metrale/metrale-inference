// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-10-19: The pre-flight charge of each kind of checkpoint tensor at the published shapes,
//! on the middle rank of three (heads 24..48 of 64, output groups 3..6 of 8, expert rows
//! 768..1536 of 2304) and on one rank; and every per-layer name of the publisher's model has a
//! placement.

use std::collections::BTreeSet;

use super::*;
use crate::deepseek_v41_ref::GOLDEN_JSON;

fn rank1() -> Local {
    Local {
        world: 3,
        heads: 24..48,
        groups: 3..6,
        inter: 768..1536,
        head_dim: 512,
        o_rank: 1024,
    }
}

fn one() -> Local {
    Local {
        world: 1,
        ..rank1()
    }
}

use WeightDtype::{BF16, FP8E4M3 as E4M3, FP8E8M0 as E8M0, FP32, UInt8 as I8};

#[test]
fn routed_expert_tables_charge_the_ranks_rows_or_columns() {
    let l = rank1();
    let e = |t: &str| format!("layers.7.ffn.experts.300.{t}");
    assert_eq!(charge(&l, &e("w1.weight"), I8, &[2304, 2560]), 768 * 2560);
    assert_eq!(charge(&l, &e("w3.scale"), E8M0, &[2304, 160]), 768 * 160);
    assert_eq!(charge(&l, &e("w2.weight"), I8, &[5120, 1152]), 5120 * 384);
    assert_eq!(charge(&l, &e("w2.scale"), E8M0, &[5120, 72]), 5120 * 24);
    assert_eq!(charge(&one(), &e("w2.scale"), E8M0, &[5120, 72]), 5120 * 72);
}

#[test]
fn attention_and_shared_projections_charge_their_cut_and_its_scale_tiles() {
    let l = rank1();
    let a = |t: &str| format!("layers.0.attn.{t}");
    assert_eq!(
        charge(&l, &a("wq_b.weight"), E4M3, &[32768, 1280]),
        24 * 512 * 1280
    );
    assert_eq!(charge(&l, &a("wq_b.scale"), E8M0, &[1024, 40]), 384 * 40);
    assert_eq!(
        charge(&l, &a("wo_a.weight"), E4M3, &[8192, 4096]),
        3 * 1024 * 4096
    );
    assert_eq!(
        charge(&l, &a("wo_b.weight"), E4M3, &[5120, 8192]),
        5120 * 3072
    );
    assert_eq!(charge(&l, &a("wo_b.scale"), E8M0, &[160, 256]), 160 * 96);
    assert_eq!(charge(&l, &a("attn_sink"), FP32, &[64]), 24 * 4);
    assert_eq!(
        charge(&l, &a("wq_a.weight"), E4M3, &[1280, 5120]),
        1280 * 5120
    );
    let s = "layers.2.ffn.shared_experts.w2";
    assert_eq!(
        charge(&l, &format!("{s}.weight"), E4M3, &[5120, 2304]),
        5120 * 768
    );
    assert_eq!(
        charge(&l, &format!("{s}.scale"), E8M0, &[160, 72]),
        160 * 24
    );
}

#[test]
fn the_engram_projection_charges_its_bf16_decode_and_host_tensors_nothing() {
    let l = rank1();
    assert_eq!(
        charge(&l, "layers.1.engram.wkv.weight", E4M3, &[25600, 6144]),
        2 * 25600 * 6144
    );
    for (n, d, s) in [
        ("layers.1.engram.wkv.scale", E8M0, vec![800, 192]),
        ("layers.1.engram.embed.weight", E4M3, vec![1000, 256]),
        ("layers.1.engram.embed.scale", E8M0, vec![1000, 8]),
        ("layers.3.ffn.gate.bias_vl", FP32, vec![384]),
        ("mtp.0.attn.wq_a.weight", E4M3, vec![1280, 5120]),
        ("vision.blocks.0.mlp.w1.weight", BF16, vec![64, 64]),
        ("aligner.w1.weight", BF16, vec![5120, 9216]),
        ("image_newline", BF16, vec![5120]),
        ("layers.3.ffn.experts.3.w4.weight", I8, vec![2, 2]),
    ] {
        assert_eq!(charge(&l, n, d, &s), 0, "{n}");
    }
}

#[test]
fn the_embedding_head_and_norm_charge_whole_and_overflow_saturates() {
    for n in WHOLE {
        assert_eq!(
            charge(&rank1(), n, BF16, &[129280, 5120]),
            2 * 129280 * 5120
        );
    }
    assert_eq!(
        charge(&rank1(), "embed.weight", BF16, &[usize::MAX, 2]),
        usize::MAX
    );
}

#[test]
fn every_tensor_is_claimed() {
    let h = defer_hook();
    assert!(h("layers.0.attn.wq_a.weight", E4M3) && h("anything", BF16));
}

/// 2026-10-19: Every per-layer parameter of the publisher's model has a device placement except
/// the engram row tables and the vision-language router bias, which stay on the host.
#[test]
fn every_publisher_layer_parameter_is_placed() {
    let v: serde_json::Value = serde_json::from_str(GOLDEN_JSON).unwrap();
    let mut all = BTreeSet::new();
    fn keys(v: &serde_json::Value, out: &mut BTreeSet<String>) {
        match v {
            serde_json::Value::Object(m) => m.iter().for_each(|(k, x)| {
                out.insert(k.clone());
                keys(x, out)
            }),
            serde_json::Value::Array(a) => a.iter().for_each(|x| keys(x, out)),
            _ => {}
        }
    }
    keys(&v, &mut all);
    let layer: Vec<&String> = all.iter().filter(|k| k.starts_with("layers.")).collect();
    assert!(layer.len() > 100, "{} layer keys", layer.len());
    for k in layer {
        let host = k.contains(".engram.embed.") || k.ends_with(".gate.bias_vl");
        assert_eq!(
            placement(k) == Placement::Host,
            host,
            "{k}: {:?}",
            placement(k)
        );
    }
}
