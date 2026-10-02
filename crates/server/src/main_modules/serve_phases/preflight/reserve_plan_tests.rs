// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: The reserve plan is what the process allocates. On the real configs of the dense
//! 27B, the 35B-A3B MoE and Nemotron-3-Nano, at the serve shapes of the certified recipes and of
//! the dense declared tier:
//! - every allocator-backed term is its allocator's bytes exactly (tolerance 0): the SSM pool
//!   (`PoolPlan::total`, which `SsmStatePool::new` allocates, model-engine
//!   `ssm_pool_state_plan_tests.rs`), the f16 staging arena, the carry stash
//!   (`GdnCarrySizes::total`, which `gdn_carry_bind` allocates) and the decode-rollback ring;
//! - the driver terms cover the driver use measured on GB10 for these serves with at most 20%
//!   to spare (`runtime_headroom.rs` carries the measurements);
//! - the inference reserve is the sum of its terms at any slot count.
//!
//! Owner: server startup (`met serve`).
//! Invariants: none beyond the types.

use metrale_config::ModelConfig;
use metrale_model_layers::layers::ops::{
    GdnCarrySizes, gdn_carry_conv_seq_elems, gdn_carry_seq_floats,
};
use metrale_model_layers::ssm_reserve::{
    PoolPlan, PoolShape, SsmRollbackMode, pool_counts, ssm_h_prefill_stage_bytes,
};

use super::super::runtime_headroom::{DRIVER_BUDGET_PER_MILLE, DRIVER_FIXED_BYTES};
use super::*;

const MIB: usize = 1 << 20;

/// 2026-10-01: GB10's total device memory as `cuMemGetInfo` reports it (124,610 MiB), and the
/// util budget at 0.85.
const GB10_TOTAL: usize = 124_610 * MIB;
fn gb10_budget() -> usize {
    (GB10_TOTAL as f64 * 0.85) as usize
}

fn config(name: &str) -> ModelConfig {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../circuit/tests/fixtures/checkpoints")
        .join(name)
        .join("config.json");
    metrale_config::parse_config(&std::fs::read_to_string(path).unwrap())
        .unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// 2026-10-01: One serve shape: the checkpoint, speculation (`None`: off; `Some(drafts)`), the
/// f16 h pool, the slot count and Marconi slots.
struct Serve {
    what: &'static str,
    checkpoint: &'static str,
    drafts: Option<usize>,
    h_f16_pool: bool,
    slots: usize,
}

/// 2026-10-01: The certified recipes' shapes (metrale-recipes `defaults:`) and the dense
/// declared tier ROWINV measured.
const SERVES: [Serve; 6] = [
    Serve {
        what: "dense declared tier (128 slots, f32 h, MTP K=4)",
        checkpoint: "unsloth--Qwen3.8-27B-NVFP4",
        drafts: Some(3),
        h_f16_pool: false,
        slots: 128,
    },
    Serve {
        what: "qwen3.8-27b-nvfp4-throughput (128 slots, f16 pool, K=4)",
        checkpoint: "unsloth--Qwen3.8-27B-NVFP4",
        drafts: Some(3),
        h_f16_pool: true,
        slots: 128,
    },
    Serve {
        what: "qwen3.8-27b-nvfp4-latency (8 slots, f16 pool, K=4)",
        checkpoint: "unsloth--Qwen3.8-27B-NVFP4",
        drafts: Some(3),
        h_f16_pool: true,
        slots: 8,
    },
    Serve {
        what: "qwen3.8-27b-nvfp4-unsloth (1 slot, K=4)",
        checkpoint: "unsloth--Qwen3.8-27B-NVFP4",
        drafts: Some(3),
        h_f16_pool: false,
        slots: 1,
    },
    Serve {
        what: "qwen3.6-35b-a3b-fp8-mtp (2 slots, K=2)",
        checkpoint: "Qwen--Qwen3.6-35B-A3B-FP8",
        drafts: Some(1),
        h_f16_pool: false,
        slots: 2,
    },
    Serve {
        what: "nemotron-3-nano (8 slots, no speculation)",
        checkpoint: "nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4",
        drafts: None,
        h_f16_pool: false,
        slots: 8,
    },
];

fn plan(s: &Serve) -> ReservePlan {
    let c = config(s.checkpoint);
    let blob = c.num_ssm_layers() * (c.ssm_h_state_bytes() + c.ssm_conv_state_bytes());
    ReservePlan {
        spec: s.drafts.is_some(),
        num_drafts: s.drafts.unwrap_or(0),
        uniform_h: false,
        rollback: SsmRollbackMode::Snapshot,
        h_f16_pool: s.h_f16_pool,
        marconi_bytes: 8 * blob,
        gdn_two_phase_bytes: 88 * MIB,
        budget_bytes: gb10_budget(),
        per_sequence_bytes: 0,
        ring_slots: 0,
        per_seq_blob: blob,
        config: c,
    }
}

fn shape(s: &Serve) -> PoolShape {
    PoolShape {
        max_slots: s.slots,
        spec: s.drafts.is_some(),
        num_intermediates: s.drafts.unwrap_or(0) + 1,
        num_drafts: s.drafts.unwrap_or(0),
        uniform_h: false,
        rollback: SsmRollbackMode::Snapshot,
    }
}

#[test]
fn every_allocator_backed_term_is_its_allocators_bytes() {
    for s in &SERVES {
        let p = plan(s);
        let t = p.terms(s.slots).unwrap();
        let counts = pool_counts(&shape(s));
        let pool = PoolPlan::new(&p.config, &counts, s.h_f16_pool).unwrap();
        assert_eq!(t.ssm_pool, pool.total(), "{}: SSM pool", s.what);
        assert_eq!(
            t.ssm_h_stage,
            ssm_h_prefill_stage_bytes(counts.slots, pool.h_f32_unit, s.h_f16_pool),
            "{}: f16 staging arena",
            s.what
        );
        // 2026-10-01: The carry stash: what `gdn_carry_bind` allocates for the MTP slots plus the
        // dummy, on the GDN layers, with an f32 h pool; nothing otherwise.
        let c = &p.config;
        let carry = match (&counts.verify, s.h_f16_pool, c.linear_key_head_dim) {
            (Some(v), false, 128) => GdnCarrySizes::new(
                (0..c.num_hidden_layers)
                    .filter(|&i| c.layer_type(i) == metrale_config::LayerType::LinearAttention)
                    .count(),
                v.slots(),
                gdn_carry_seq_floats(
                    c.linear_num_value_heads,
                    c.linear_key_head_dim,
                    c.linear_value_head_dim,
                ),
                gdn_carry_conv_seq_elems(
                    c.linear_num_key_heads * c.linear_key_head_dim * 2
                        + c.linear_num_value_heads * c.linear_value_head_dim,
                ),
                metrale_model_layers::layer::VERIFY_WY_TABLE_SEQS,
            )
            .total(),
            _ => 0,
        };
        assert_eq!(t.runtime.carry_stash, carry, "{}: carry stash", s.what);
        assert_eq!(
            t.total(),
            t.ssm_pool
                + t.ssm_h_stage
                + t.replay_ring
                + t.marconi
                + t.gdn_two_phase
                + t.runtime.carry_stash
                + t.runtime.driver_fixed
                + t.runtime.driver_bookkeeping
                + t.per_sequence
                + t.decode_ring,
            "{}: the reserve is its terms",
            s.what
        );
        assert_eq!(p.inference_reserve(s.slots).unwrap(), t.total());
    }
}

#[test]
fn the_dense_declared_carry_stash_is_the_one_the_serve_bound() {
    // 2026-10-01: The serve log (GB10, 2026-10-01): "GDN carry: bound 884.8 MB stash (48 GDN
    // layers x 33 slots)", the stash and conv stash in decimal MB.
    let p = plan(&SERVES[0]);
    let t = p.terms(128).unwrap();
    let c = &p.config;
    let sz = GdnCarrySizes::new(
        48,
        33,
        gdn_carry_seq_floats(c.linear_num_value_heads, 128, 128),
        gdn_carry_conv_seq_elems(c.linear_num_key_heads * 128 * 2 + c.linear_num_value_heads * 128),
        metrale_model_layers::layer::VERIFY_WY_TABLE_SEQS,
    );
    assert_eq!(
        ((sz.stash + sz.conv_stash) as f64 / 1e5).round() / 10.0,
        884.8
    );
    assert_eq!(t.runtime.carry_stash, sz.total());
}

#[test]
fn the_driver_terms_cover_every_measured_serve_with_at_most_a_fifth_to_spare() {
    // 2026-10-01: The largest driver use measured on GB10 per serve (`runtime_headroom.rs`),
    // less the small-allocation chunk slack the KV sizing charges from the ledger: the dense
    // default tier's worst case (C128 and prefix restores, 3,126 MiB), the throughput recipe
    // (2,423), the 35B-A3B MoE at 128 slots (2,798), Nemotron-3-Nano no-spec at its recipe flags
    // (2,935) and with prefix caching on (3,499 less its 429 MiB of chunk slack).
    let driver = DRIVER_FIXED_BYTES + gb10_budget() / 1000 * DRIVER_BUDGET_PER_MILLE;
    let measured = [
        ("dense default tier", 3_126 * MIB),
        ("throughput recipe", 2_423 * MIB),
        ("35B-A3B MoE, 128 slots", 2_798 * MIB),
        ("nano", 2_935 * MIB),
        ("nano with prefix caching", (3_499 - 429) * MIB),
    ];
    let largest = measured.iter().map(|m| m.1).max().unwrap();
    for (what, m) in measured {
        assert!(driver >= m, "{what}: {driver} < {m}");
    }
    assert!(
        driver <= largest + largest / 5,
        "over-reserves the largest by more than 20%"
    );
    // 2026-10-01: Speculation does not change it: both serves read the same terms.
    let (spec, nospec) = (plan(&SERVES[0]), plan(&SERVES[5]));
    let (a, b) = (
        spec.terms(8).unwrap().runtime,
        nospec.terms(8).unwrap().runtime,
    );
    assert_eq!(
        (a.driver_fixed, a.driver_bookkeeping),
        (b.driver_fixed, b.driver_bookkeeping)
    );
    assert_eq!(b.carry_stash, 0);
}

#[test]
fn the_slot_scaled_terms_follow_the_slot_count() {
    let p = plan(&SERVES[0]);
    let (t91, t128) = (p.terms(91).unwrap(), p.terms(128).unwrap());
    // 2026-10-01: The main pool shrinks by one f32 blob per slot; the MTP pools (32 slots) and
    // the carry stash (33 slots) are unchanged above 32 slots.
    let blob = p.config.num_ssm_layers()
        * (p.config.ssm_h_state_bytes() + p.config.ssm_conv_state_bytes());
    assert_eq!(t128.ssm_pool - t91.ssm_pool, 37 * blob);
    assert_eq!(t128.runtime, t91.runtime);
    let t8 = p.terms(8).unwrap();
    assert!(t8.runtime.carry_stash < t91.runtime.carry_stash);
}
