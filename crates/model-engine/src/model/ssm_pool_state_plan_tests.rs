// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: M5 parity: the circuit's state plan (`metrale_circuit::state::StatePlan`, from
//! the states the circuit TOMLs declare) sizes what legacy sizes, on the real configs of the
//! two Qwen families and Nemotron-3-Nano (crates/circuit/tests/fixtures/checkpoints/):
//! - one unit of each state equals the legacy per-layer blob (`ssm_h_state_bytes`,
//!   `ssm_conv_state_bytes`, `KvCacheConfig::{k,v}_block_bytes_for_layer`);
//! - with the allocator's unit counts (the padding dummy, the tiered h intermediates), the plan
//!   is the bytes `SsmStatePool::new` allocates;
//! - with preflight's counts (no dummies), it is `ssm_pool_reserve_bytes`.
//!
//! Owner: model-engine SSM state pool.
//! Invariants: none beyond the types.

use std::collections::BTreeMap;

use metrale_cache::kv_cache::{KvCacheConfig, KvCacheDtype};
use metrale_circuit::state::{
    Holding, KvInputs, StateDtype, StateInputs, StateKind, StatePlan, VerifyInputs,
};
use metrale_circuit::{Circuit, QuantMetadata, Section, ServePrecision, resolve_checkpoint};
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
use metrale_model_layers::ssm_reserve::{
    SsmRollbackMode, ssm_h_prefill_stage_bytes, ssm_pool_reserve_bytes, verify_slot_h_intermediates,
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
            &c,
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

#[test]
fn with_the_allocators_counts_the_plan_is_what_the_pool_allocates() {
    for name in CHECKPOINTS {
        let (config, c) = load(name);
        for (max_slots, has_mtp, f16) in [(1, false, false), (8, true, false), (64, true, true)] {
            let gpu = MockGpuBackend::new();
            let (ni, nd) = (4, 3);
            let pool = SsmStatePool::new(
                &config,
                max_slots,
                has_mtp,
                ni,
                nd,
                f16,
                SsmRollbackMode::Snapshot,
                &gpu,
            )
            .unwrap();
            let verify = has_mtp.then(|| VerifyInputs {
                h_steps: pool.h_inter_counts.iter().map(|&n| n as u64).collect(),
                conv_steps: ni as u64,
            });
            let h = if f16 {
                StateDtype::F16
            } else {
                StateDtype::F32
            };
            let p = StatePlan::new(&c, &inputs(max_slots as u64 + 1, h, verify)).unwrap();
            // 2026-09-30: The FP32 prefill staging arena of an f16 pool is one layer's blob per
            // slot, a scratch the circuit does not declare as state.
            let stage = ssm_h_prefill_stage_bytes(max_slots + 1, config.ssm_h_state_bytes(), f16);
            assert_eq!(
                recurrent(&c, &p) as usize + stage,
                gpu.live_bytes().unwrap(),
                "{name} slots={max_slots} mtp={has_mtp} f16={f16}"
            );
        }
    }
}

#[test]
fn with_preflights_counts_the_plan_is_the_preflight_reserve() {
    for name in CHECKPOINTS {
        let (config, c) = load(name);
        let layers = config.num_ssm_layers();
        for (max_batch, spec, f16) in [(1, false, false), (16, true, false), (128, true, true)] {
            let (nd, mtp_slots) = (3, max_batch.min(32));
            let want = ssm_pool_reserve_bytes(
                max_batch,
                config.ssm_h_state_bytes() * layers,
                config.ssm_conv_state_bytes() * layers,
                spec,
                nd,
                mtp_slots,
                false,
                f16,
                SsmRollbackMode::Snapshot,
            );
            let verify = spec.then(|| VerifyInputs {
                h_steps: (0..mtp_slots)
                    .map(|s| verify_slot_h_intermediates(s, nd, false) as u64)
                    .collect(),
                conv_steps: nd as u64 + 1,
            });
            let h = if f16 {
                StateDtype::F16
            } else {
                StateDtype::F32
            };
            let p = StatePlan::new(&c, &inputs(max_batch as u64, h, verify)).unwrap();
            assert_eq!(
                recurrent(&c, &p) as usize,
                want,
                "{name} batch={max_batch} spec={spec} f16={f16}"
            );
        }
    }
}
