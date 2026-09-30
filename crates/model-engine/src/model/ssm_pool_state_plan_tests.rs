// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: M5 parity: the circuit's state plan (`metrale_circuit::state::StatePlan`, from
//! the states the circuit TOMLs declare) sizes what legacy sizes, on the real configs of the
//! two Qwen families and Nemotron-3-Nano (crates/circuit/tests/fixtures/checkpoints/):
//! - one unit of each state equals the legacy per-layer blob (`ssm_h_state_bytes`,
//!   `ssm_conv_state_bytes`, `KvCacheConfig::{k,v}_block_bytes_for_layer`);
//! - the circuit families are sized from their circuits, which agree with the transitional
//!   source;
//! - `SsmStatePool::new` allocates exactly the pool plan the preflight reserve reserves.
//!
//! Owner: model-engine SSM state pool.
//! Invariants: none beyond the types.

use std::collections::BTreeMap;

use metrale_cache::kv_cache::{KvCacheConfig, KvCacheDtype};
use metrale_circuit::state::VerifySteps;
use metrale_circuit::state::{
    Holding, KvInputs, StateDtype, StateInputs, StateKind, StatePlan, VerifyInputs,
};
use metrale_circuit::{Circuit, QuantMetadata, Section, ServePrecision, resolve_checkpoint};
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
use metrale_model_layers::ssm_reserve::{
    PoolPlan, PoolShape, SsmRollbackMode, UnitSource, pool_counts, recurrent_units,
    ssm_h_prefill_stage_bytes, ssm_replay_ring_bytes, ssm_replay_row_bytes,
};

use super::SsmStatePool;

/// 2026-09-30: The M5 acceptance checkpoints.
const CHECKPOINTS: [&str; 3] = [
    "unsloth--Qwen3.8-27B-NVFP4",
    "Qwen--Qwen3.6-35B-A3B-FP8",
    "nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4",
];

fn load(name: &str) -> (ModelConfig, Circuit) {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../circuit/tests/fixtures/checkpoints")
        .join(name);
    let json = std::fs::read_to_string(dir.join("config.json")).unwrap();
    let hq = std::fs::read_to_string(dir.join("hf_quant_config.json")).ok();
    let config = metrale_config::parse_config(&json).unwrap_or_else(|e| panic!("{name}: {e}"));
    let circuit = resolve_checkpoint(
        &json,
        QuantMetadata {
            hf_quant_config: hq.as_deref(),
        },
        &ServePrecision::Declared,
    )
    .unwrap_or_else(|e| panic!("{name}: {e}"))
    .circuit;
    (config, circuit)
}

fn inputs(slots: u64, h: StateDtype, verify: Option<VerifyInputs>) -> StateInputs {
    StateInputs {
        formats: BTreeMap::from([
            ("ssm_h_storage".to_string(), h),
            ("kv_cache_dtype".to_string(), StateDtype::Bf16),
        ]),
        slots,
        verify,
        kv: None,
        draft_kv: None,
    }
}

/// 2026-09-30: The target's recurrent terms (the pool the tests compare with).
fn recurrent(c: &Circuit, p: &StatePlan) -> u64 {
    let main: Vec<&str> = c
        .states
        .iter()
        .filter(|s| s.kind == StateKind::Recurrent && s.section == Section::Main)
        .map(|s| s.id.as_str())
        .collect();
    p.bytes_where(|t| main.contains(&t.state.as_str()))
}

#[test]
fn one_unit_of_every_state_is_the_legacy_blob() {
    for name in CHECKPOINTS {
        let (config, c) = load(name);
        let main = |local: &str| {
            c.states
                .iter()
                .filter(|s| s.section == Section::Main && s.local == local)
                .collect::<Vec<_>>()
        };
        assert_eq!(main("h").len(), config.num_ssm_layers(), "{name}");
        for s in main("h") {
            assert_eq!(
                s.elements as usize * 4,
                config.ssm_h_state_bytes(),
                "{name} {}",
                s.id
            );
        }
        for s in main("conv") {
            assert_eq!(
                s.elements as usize * 4,
                config.ssm_conv_state_bytes(),
                "{name} {}",
                s.id
            );
        }
        let kv = KvCacheConfig {
            block_size: 16,
            num_kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            num_layers: main("k").len(),
            dtype: KvCacheDtype::Bf16,
            layer_dtypes: Vec::new(),
            layer_dims: Vec::new(),
            cache_blocks_per_seq: None,
        };
        assert_eq!(main("k").len(), config.num_attention_layers(), "{name}");
        let p = StatePlan::new(
            &c.states,
            &StateInputs {
                kv: Some(KvInputs {
                    blocks: 1,
                    block_size: 16,
                }),
                ..inputs(0, StateDtype::F32, None)
            },
        )
        .unwrap();
        assert_eq!(
            p.bytes_where(|t| t.holding == Holding::Blocks && !t.state.starts_with("draft.")),
            kv.block_bytes_kv_all_layers() as u64,
            "{name}"
        );
    }
}

/// 2026-09-30: The circuit families are sized from their circuits (not the transitional
/// source), and the two sources agree on them, so the transitional path is safe to keep for
/// the families without a circuit until M3.
#[test]
fn circuit_families_take_the_circuit_source_which_agrees_with_the_transitional_one() {
    for name in CHECKPOINTS {
        let (config, _) = load(name);
        let (decls, source) = recurrent_units(&config).unwrap();
        assert_eq!(
            source,
            UnitSource::Circuit,
            "{name}: model_type {}",
            config.model_type
        );
        let unit = |v: VerifySteps| decls.iter().find(|d| d.verify == Some(v)).unwrap().elements;
        assert_eq!(
            unit(VerifySteps::H) as usize * 4,
            config.ssm_h_state_bytes(),
            "{name}"
        );
        assert_eq!(
            unit(VerifySteps::Conv) as usize * 4,
            config.ssm_conv_state_bytes(),
            "{name}"
        );
    }
}

/// 2026-09-30: One plan: what the pool allocates (mock backend) is the plan's bytes, and the
/// preflight reserve is that same plan (`preflight_reserve` calls `pool_counts` and
/// `PoolPlan::new` with the pool's arguments).
#[test]
fn the_pool_allocates_exactly_the_plan() {
    for name in CHECKPOINTS {
        let (config, _) = load(name);
        for (max_slots, has_mtp, f16, rollback) in [
            (1, false, false, SsmRollbackMode::Snapshot),
            (8, true, false, SsmRollbackMode::Snapshot),
            (64, true, true, SsmRollbackMode::Snapshot),
            (8, true, false, SsmRollbackMode::Replay),
        ] {
            let gpu = MockGpuBackend::new();
            let (ni, nd) = (4, 3);
            SsmStatePool::new(&config, max_slots, has_mtp, ni, nd, f16, rollback, &gpu).unwrap();
            let counts = pool_counts(&PoolShape {
                max_slots,
                spec: has_mtp,
                num_intermediates: ni,
                num_drafts: nd,
                uniform_h: false,
                rollback,
            });
            let plan = PoolPlan::new(&config, &counts, f16).unwrap();
            let stage = ssm_h_prefill_stage_bytes(counts.slots, plan.h_f32_unit, f16);
            let ring = match (&counts.verify, rollback) {
                (Some(v), SsmRollbackMode::Replay) => ssm_replay_ring_bytes(
                    plan.layers,
                    ssm_replay_row_bytes(config.ssm_qkvz_size(), config.linear_num_value_heads),
                    ni,
                    v.slots(),
                ),
                _ => 0,
            };
            assert_eq!(
                plan.total() + stage + ring,
                gpu.live_bytes().unwrap(),
                "{name} slots={max_slots} mtp={has_mtp} f16={f16} {rollback:?}"
            );
        }
    }
}
