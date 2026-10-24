// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: GLM-5.3 circuit mode over any device count: the deployment the serve derives for
//! N in 1..=8 (`metrale_circuit::deploy`) is admitted or refused naming the part that cannot
//! split; at every admitted N each rank's circuit holds exactly the engine's own split of every
//! part (the loader's head division, the MLP config's widths), and every rank fuses the shapes
//! the boot installs with no rule refused. The per-family gates (MoE, mHC, KDA, DSA, the draft
//! head) run their plan-versus-legacy recordings at every admitted N.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use metrale_circuit::{AvailableKernels, Mode};
use metrale_error::ErrorCode;
use metrale_model_layers::circuit_exec::rank_shapes::{rank_plan_shapes, speculative_verify_rows};

use super::fixture::{MAX_WORLD, mlp_config_at, model_config_at, ranks_at, served_world, worlds};
use super::install::{DRAFT_GATED, GATED};

/// 2026-10-23: Run `body` in a child process of this test binary that publishes the served
/// `--dense-quantization w4a16` first (the KDA head unit and the shared expert's split unit read
/// the tier).
fn in_w4a16_process(name: &str, body: impl FnOnce()) {
    let publish = || {
        let q = metrale_model_layers::layers::set_dense_quantization_from_cli(
            metrale_model_layers::layers::DenseQuantization::W4a16,
        );
        assert_eq!(q, metrale_model_layers::layers::DenseQuantization::W4a16);
    };
    let path = format!("glm5next_plan::world_tests::{name}");
    super::fixture::in_child_process(&path, publish, body);
}

/// 2026-10-23: One device is the whole model (no rank circuits); 2..=4 are admitted, the served 3
/// among them; 5..=8 are refused by the vocab part, whose degree the draft head's exchange caps
/// at four (`glm5next_mtp_head::head_xchg::MAX_RANKS`).
#[test]
fn the_served_instance_admits_two_to_four_ranks_and_refuses_past_the_vocab_cap() {
    let (one, _, _) = ranks_at(1).unwrap();
    assert!(one.is_empty(), "N = 1 runs the whole model");
    assert_eq!(worlds(), [2, 3, 4]);
    assert!(worlds().contains(&served_world()));
    for n in 5..=MAX_WORLD {
        let e = ranks_at(n).err().unwrap();
        assert_eq!(e.code(), ErrorCode::CircuitRanksRefused, "{e}");
        assert!(
            e.to_string()
                .contains(&format!("{n} ranks: part `vocab` (dim `vocab`)")),
            "{e}"
        );
        assert!(e.to_string().contains("1..=4"), "{e}");
    }
    assert_eq!(
        crate::glm5next_mtp_head::head_xchg::MAX_RANKS,
        4,
        "the circuit's vocab cap is the exchange's"
    );
}

/// 2026-10-23: At every admitted N, rank r's circuit has the engine's split of r under the served
/// tier: the KDA and DSA heads as `ModelConfig::shard_heads_for_tp` divides them under the GLM
/// loader's own support, the routed, shared and dense FFN widths as `Glm5NextMlpConfig` slices
/// them.
#[test]
fn every_rank_circuit_holds_the_engines_split_at_every_world() {
    in_w4a16_process(
        "every_rank_circuit_holds_the_engines_split_at_every_world",
        every_split,
    );
}

fn every_split() {
    use crate::weight_loader::ModelWeightLoader;
    let support = crate::weight_loader::glm5_next_load::Glm5NextWeightLoader.tp_support();
    for world in worlds() {
        let (c, _, _) = ranks_at(world).unwrap();
        for (rank, circuit) in c.iter().enumerate() {
            let mut cfg = model_config_at(rank, world);
            cfg.shard_heads_for_tp(support).unwrap();
            let mlp = mlp_config_at(rank, world);
            let engine = [
                ("lin_k_heads", cfg.linear_num_key_heads),
                ("lin_v_heads", cfg.linear_num_value_heads),
                ("q_heads", cfg.num_attention_heads),
                ("moe_inter", mlp.moe_intermediate),
                ("shared_inter", mlp.local_shared_intermediate),
                ("inter", mlp.local_dense_intermediate),
            ];
            for (dim, want) in engine {
                assert_eq!(
                    circuit.dims[dim], want as u64,
                    "world {world} rank {rank}: `{dim}`"
                );
            }
        }
    }
}

/// 2026-10-23: At every admitted N every rank fuses the boot's shapes at one draft (decode 1,
/// verify 2, draft 1) under every kernel the rules name, no rule refused.
#[test]
fn every_rank_fuses_the_boot_shapes_at_every_world() {
    let installed: Vec<(Mode, u64)> = GATED.iter().chain(&DRAFT_GATED).copied().collect();
    let shapes = rank_plan_shapes(&installed, &speculative_verify_rows(1)).unwrap();
    let mut fused = 0;
    for world in worlds() {
        let (c, rules, policy) = ranks_at(world).unwrap();
        let avail = AvailableKernels::all_named_by(&rules);
        for (rank, circuit) in c.iter().enumerate() {
            for &(mode, rows) in &shapes {
                metrale_circuit::fuse(circuit, &rules, &avail, &policy, mode, rows)
                    .unwrap_or_else(|e| panic!("world {world} rank {rank} {mode:?} {rows}: {e}"));
                fused += 1;
            }
        }
    }
    assert_eq!(fused, (2 + 3 + 4) * 3);
}

/// 2026-10-23: GLM over `world` ranks with every part at `world` or its cap, and the vocab
/// replicated past its cap (a layout the cost model may pick; the serve's derivation refuses it):
/// each rank's circuit and the rules.
pub(super) fn capped_layout(world: u64) -> super::fixture::Ranks {
    use metrale_model_layers::circuit_exec::sources;
    let mut inst = sources::instance(super::fixture::RECIPE).unwrap();
    let src = sources::sources(&inst).unwrap();
    let spec = metrale_circuit::parallel_spec(src.circuit, src.blocks).unwrap();
    let degree = metrale_circuit::parallel::parts(&spec)
        .map(|p| {
            let cap = spec[p].degree.max;
            (
                p.clone(),
                if world > cap && p == "vocab" {
                    1
                } else {
                    world.min(cap)
                },
            )
        })
        .collect();
    inst.parallel = Some(metrale_circuit::Deployment {
        ranks: world,
        degree,
        why: "world_tests".into(),
    });
    let loaded = metrale_circuit::load(&inst, src).unwrap();
    (loaded.ranks, loaded.rules, inst.policy)
}

/// 2026-10-23: At two ranks the draft's plain all-reduces (the MTP attention and MoE partials)
/// are the 2-rank exchange and `bf16_add_inplace`; at the served three, the one-shot rank sum.
#[test]
fn a_two_rank_plan_names_the_two_rank_add() {
    let kernels_at = |world: usize| {
        let (c, rules, policy) = ranks_at(world).unwrap();
        let avail = AvailableKernels::all_named_by(&rules);
        let p = metrale_circuit::fuse(&c[0], &rules, &avail, &policy, Mode::Draft, 1).unwrap();
        p.groups
            .iter()
            .filter(|g| g.rule.starts_with("tp_all_reduce"))
            .map(|g| (g.rule.clone(), g.kernels[0].func.clone()))
            .collect::<Vec<_>>()
    };
    let two = kernels_at(2);
    assert!(!two.is_empty());
    assert!(
        two.iter()
            .all(|(r, k)| r == "tp_all_reduce_2rank" && k == "bf16_add_inplace"),
        "{two:?}"
    );
    let three = kernels_at(3);
    assert!(
        three
            .iter()
            .all(|(r, k)| r == "tp_all_reduce_oneshot" && k == "bf16_add_rank_sum"),
        "{three:?}"
    );
}
