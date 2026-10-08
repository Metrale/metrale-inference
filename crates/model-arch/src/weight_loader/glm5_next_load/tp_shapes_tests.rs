// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: The per-rank shapes the GLM-5.3 loader's TP plans give every split tensor of the
//! real checkpoint (`glm53-nvfp4-tp-shapes.tsv`), at TP 1, 2 and 3, without a GPU.
//!
//! Owner: model-arch weight loader.
//! Invariants: none beyond the types.
//!
//! Each rank's config goes through the head division serve's topology runs
//! (`ModelConfig::shard_heads_for_tp` with the loader's own `tp_support`), then through the
//! plans `load_layers` builds: `KdaTpPlan::from_config`, `DsaTpPlan::from_config` and
//! `Glm5NextMlpConfig::from_config`.

use metrale_config::{ModelConfig, parse_config};

use super::Glm5NextWeightLoader;
use crate::glm5next_dsa::Glm5NextDsaConfig;
use crate::glm5next_dsa::tp::{DsaShard, DsaTpPlan};
use crate::glm5next_kda::tp::{KdaShard, KdaTpPlan};
use crate::glm5next_mlp::Glm5NextMlpConfig;
use crate::weight_loader::ModelWeightLoader;

const CONFIG: &str =
    include_str!("../../../../model-engine/tests/fixtures/glm53-nvfp4-9e0d74e3-config.json");
const SHAPES: &str =
    include_str!("../../../../model-engine/tests/fixtures/glm53-nvfp4-tp-shapes.tsv");

/// 2026-10-08: One checkpoint tensor: layer-relative name and shape.
struct Disk {
    layer: usize,
    rel: String,
    dims: Vec<usize>,
}

fn disk() -> Vec<Disk> {
    SHAPES
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
        .map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            let rest = f[0].strip_prefix("model.language_model.layers.").unwrap();
            let (layer, rel) = rest.split_once('.').unwrap();
            Disk {
                layer: layer.parse().unwrap(),
                rel: rel.to_string(),
                dims: f[2].split(',').map(|d| d.parse().unwrap()).collect(),
            }
        })
        .collect()
}

fn get<'a>(d: &'a [Disk], layer: usize, rel: &str) -> &'a Disk {
    d.iter()
        .find(|t| t.layer == layer && t.rel == rel)
        .unwrap_or_else(|| panic!("fixture has no layers.{layer}.{rel}"))
}

/// 2026-10-08: The plans of one rank.
struct Rank {
    config: ModelConfig,
    kda: KdaTpPlan,
    dsa: DsaTpPlan,
    mlp: Glm5NextMlpConfig,
}

/// 2026-10-08: World = TP = EP = `tp`, as serve's overlapping groups set it.
fn rank(tp: usize, r: usize, gate_rank: usize) -> Rank {
    let mut c = parse_config(CONFIG).expect("the real checkpoint config parses");
    c.tp_world_size = tp;
    c.tp_rank = r;
    c.ep_world_size = tp;
    c.ep_rank = r;
    c.shard_heads_for_tp(Glm5NextWeightLoader.tp_support())
        .unwrap();
    let dsa_cfg = Glm5NextDsaConfig::from_config(&c).unwrap();
    Rank {
        kda: KdaTpPlan::from_config(&c, gate_rank).unwrap(),
        dsa: DsaTpPlan::from_config(&c, &dsa_cfg).unwrap(),
        mlp: Glm5NextMlpConfig::from_config(&c).unwrap(),
        config: c,
    }
}

/// 2026-10-08: A plan entry in `[rows, row_elems]` terms; `axis` is `None` when replicated,
/// `Some(0)` for a row split and `Some(1)` for a column split.
#[derive(Debug, Clone, Copy)]
struct Placed {
    full: (usize, usize),
    local: (usize, usize),
    offset: (usize, usize),
    axis: Option<usize>,
}

fn kda_placed(p: &KdaTpPlan, name: &str) -> Option<Placed> {
    let t = p.get(name)?;
    Some(Placed {
        full: (t.full_rows, t.full_row_elems),
        local: (t.local_rows, t.local_row_elems),
        offset: (t.src_row_offset, t.src_col_offset),
        axis: match t.kind {
            KdaShard::Replicated => None,
            KdaShard::HeadRows | KdaShard::ChannelRows => Some(0),
            KdaShard::ChannelCols => Some(1),
        },
    })
}

fn dsa_placed(p: &DsaTpPlan, name: &str) -> Option<Placed> {
    let t = p.get(name)?;
    Some(Placed {
        full: (t.full_rows, t.full_row_elems),
        local: (t.local_rows, t.local_row_elems),
        offset: (t.src_row_offset, t.src_col_offset),
        axis: match t.kind {
            DsaShard::Replicated => None,
            DsaShard::HeadRows => Some(0),
            DsaShard::HeadCols => Some(1),
        },
    })
}

/// 2026-10-08: The plan name of a `self_attn.*` checkpoint tensor: the prefix goes, and the
/// `.weight` suffix too unless the plan names it (the indexer's `k_norm.weight` and `.bias`).
fn plan_name(rel: &str, has: impl Fn(&str) -> bool) -> String {
    let n = rel.strip_prefix("self_attn.").unwrap();
    if has(n) {
        return n.to_string();
    }
    n.strip_suffix(".weight").unwrap_or(n).to_string()
}

/// 2026-10-08: Over the ranks, in order: a replicated tensor is whole everywhere; a split one
/// tiles its axis contiguously from 0 with no overlap, keeps the other axis whole, and sums to
/// the full tensor. With `even` (a `tp` that divides the 64 heads), every rank also holds
/// exactly `full / tp` at `rank * full / tp`, the split loaders used before uneven splits.
fn assert_tiles(name: &str, tp: usize, even: bool, placed: &[Placed]) {
    let full = placed[0].full;
    let mut next = 0;
    for (r, p) in placed.iter().enumerate() {
        assert_eq!(p.full, full, "{name}: full shape differs across ranks");
        let Some(axis) = p.axis else {
            assert_eq!(
                (p.local, p.offset),
                (full, (0, 0)),
                "{name} rank {r} replicates"
            );
            continue;
        };
        let (along, offset, keep, keep_full) = if axis == 0 {
            (p.local.0, p.offset.0, p.local.1, full.1)
        } else {
            (p.local.1, p.offset.1, p.local.0, full.0)
        };
        assert_eq!(offset, next, "{name} rank {r}: contiguous, no overlap");
        assert_eq!(keep, keep_full, "{name} rank {r}: the other axis is whole");
        assert!(along > 0, "{name} rank {r} owns nothing");
        let total = if axis == 0 { full.0 } else { full.1 };
        if even {
            assert_eq!(
                (along, offset),
                (total / tp, r * total / tp),
                "{name} rank {r}"
            );
        }
        next += along;
    }
    if placed[0].axis.is_some() {
        let total = if placed[0].axis == Some(0) {
            full.0
        } else {
            full.1
        };
        assert_eq!(next, total, "{name}: the ranks cover the whole axis");
    }
}

fn elems(dims: &[usize]) -> usize {
    dims.iter().product()
}

/// 2026-10-08: Every `self_attn` tensor of the KDA and DSA layers has a plan entry whose full
/// shape is the checkpoint's, and its per-rank slices tile it at TP 1, 2 and 3.
#[test]
fn every_mixer_tensor_tiles_over_the_ranks() {
    let d = disk();
    let gate_rank = get(&d, 0, "self_attn.f_a_proj.weight").dims[0];
    for tp in 1..=3 {
        let ranks: Vec<Rank> = (0..tp).map(|r| rank(tp, r, gate_rank)).collect();
        for (layer, n_plan) in [
            (0, ranks[0].kda.tensors.len()),
            (3, ranks[0].dsa.tensors.len()),
        ] {
            let mixer: Vec<&Disk> = d
                .iter()
                .filter(|t| t.layer == layer && t.rel.starts_with("self_attn."))
                .collect();
            assert_eq!(
                mixer.len(),
                n_plan,
                "layer {layer}: a tensor without a plan"
            );
            for t in mixer {
                let placed: Vec<Placed> = ranks
                    .iter()
                    .map(|rk| {
                        let p = if layer == 0 {
                            let n = plan_name(&t.rel, |n| rk.kda.get(n).is_some());
                            kda_placed(&rk.kda, &n)
                        } else {
                            let n = plan_name(&t.rel, |n| rk.dsa.get(n).is_some());
                            dsa_placed(&rk.dsa, &n)
                        };
                        p.unwrap_or_else(|| panic!("no plan for {}", t.rel))
                    })
                    .collect();
                let full = placed[0].full;
                assert_eq!(full.0 * full.1, elems(&t.dims), "{} full size", t.rel);
                assert_tiles(&format!("tp{tp} {}", t.rel), tp, 64 % tp == 0, &placed);
            }
        }
    }
}

/// 2026-10-08: At TP=3 the heads are 22/21/21 on both mixers and in the config, and rank 0, with
/// the extra head, stores the most mixer bytes.
#[test]
fn tp3_heads_are_22_21_21_everywhere() {
    let gate_rank = get(&disk(), 0, "self_attn.f_a_proj.weight").dims[0];
    let ranks: Vec<Rank> = (0..3).map(|r| rank(3, r, gate_rank)).collect();
    for (rk, want) in ranks.iter().zip([22usize, 21, 21]) {
        assert_eq!(rk.kda.local_heads, want);
        assert_eq!(rk.dsa.local_heads, want);
        assert_eq!(rk.config.num_attention_heads, want);
        assert_eq!(rk.config.linear_num_value_heads, want);
    }
    let bytes: Vec<usize> = ranks
        .iter()
        .map(|rk| rk.kda.local_bytes() + rk.dsa.local_bytes())
        .collect();
    assert!(bytes[0] > bytes[1] && bytes[1] == bytes[2], "{bytes:?}");
}

/// 2026-10-08: The dense FFN and shared-expert widths the checkpoint stores (U8 is packed two
/// per byte) are the ones the MLP config splits, and the ranks' column ranges tile them.
#[test]
fn the_mlp_widths_tile_over_the_ranks() {
    let d = disk();
    let gate_rank = get(&d, 0, "self_attn.f_a_proj.weight").dims[0];
    let dense_gate = &get(&d, 0, "mlp.gate_proj.weight").dims;
    let dense_down = &get(&d, 0, "mlp.down_proj.weight").dims;
    let shared_gate = &get(&d, 3, "mlp.shared_experts.gate_proj.weight").dims;
    let shared_down = &get(&d, 3, "mlp.shared_experts.down_proj.weight").dims;
    for tp in 1..=3 {
        let ranks: Vec<Rank> = (0..tp).map(|r| rank(tp, r, gate_rank)).collect();
        let c = &ranks[0].config;
        assert_eq!(dense_gate[0], c.intermediate_size);
        assert_eq!(dense_down[1] * 2, c.intermediate_size, "packed K");
        assert_eq!(shared_gate[0], c.shared_expert_intermediate_size);
        assert_eq!(shared_down[1], c.shared_expert_intermediate_size);
        for shared in [false, true] {
            let total = if shared {
                c.shared_expert_intermediate_size
            } else {
                c.intermediate_size
            };
            // 2026-10-08: `gate`/`up` split by row and `down` by column over the same range;
            // placed here as `down`'s `[hidden, inter]` column split.
            let placed: Vec<Placed> = ranks
                .iter()
                .map(|rk| {
                    let s = if shared {
                        rk.mlp.shared_slice()
                    } else {
                        rk.mlp.dense_slice()
                    };
                    Placed {
                        full: (c.hidden_size, total),
                        local: (c.hidden_size, s.len),
                        offset: (0, s.start),
                        axis: Some(1),
                    }
                })
                .collect();
            assert_tiles(&format!("tp{tp} mlp width {total}"), tp, tp != 3, &placed);
        }
    }
}
