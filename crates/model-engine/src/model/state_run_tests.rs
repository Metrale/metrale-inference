// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The bound state programs, run on a mock device against a real pool: the dense
//! circuit's declarations, instantiated over a small GatedDeltaNet geometry, land every copy on
//! the slot, layer and row the pool's own accessors name. Checked byte for byte, because a
//! launch count alone would pass a program that moved the wrong layer.
//!
//! Owner: model-engine.
//! Invariants: none beyond the types.

use metrale_circuit::LayerKind;
use metrale_circuit::state_ops::StateProgramId;
use metrale_config::{LayerType, ModelConfig};
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
use metrale_model_layers::circuit_exec::sources;
use metrale_model_layers::circuit_exec::state_bind::{PoolUnits, StatePool, StatePrograms};
use metrale_model_layers::ssm_reserve::SsmRollbackMode;

use super::*;

/// 2026-10-03: The Qwen3-Next layer pattern with tiny GatedDeltaNet heads (kilobytes per pool).
fn tiny_config() -> ModelConfig {
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.linear_num_key_heads = 2;
    c.linear_key_head_dim = 4;
    c.linear_num_value_heads = 2;
    c.linear_value_head_dim = 4;
    c.linear_conv_kernel_dim = 4;
    c
}

/// 2026-10-03: The dense instance's circuit over `config`'s layers and GatedDeltaNet dims.
fn circuit(config: &ModelConfig) -> metrale_circuit::Circuit {
    let mut inst = sources::instance("qwen3.8/qwen3.8-27b-nvfp4-unsloth").unwrap();
    inst.shape.layer_kinds = (0..config.num_hidden_layers)
        .map(|i| match config.layer_type(i) {
            LayerType::LinearAttention => LayerKind::LinearAttention,
            _ => LayerKind::FullAttention,
        })
        .collect();
    for (k, v) in [
        ("lin_k_heads", config.linear_num_key_heads),
        ("lin_k_dim", config.linear_key_head_dim),
        ("lin_v_heads", config.linear_num_value_heads),
        ("lin_v_dim", config.linear_value_head_dim),
    ] {
        inst.shape.dims.insert(k.into(), v as u64);
    }
    metrale_circuit::load(&inst, sources::sources(&inst).unwrap())
        .unwrap()
        .circuit
}

fn setup() -> (MockGpuBackend, SsmStatePool, StatePrograms) {
    let config = tiny_config();
    let gpu = MockGpuBackend::new();
    let pool = SsmStatePool::new(&config, 4, true, 4, 3, false, SsmRollbackMode::Snapshot, &gpu)
        .unwrap();
    let programs = StatePool {
        units: PoolUnits {
            h_stored: pool.h_stored_bytes,
            conv: pool.conv_bytes,
        },
        h_f16: false,
    }
    .bind(&circuit(&config), &config)
    .unwrap();
    (gpu, pool, programs)
}

/// 2026-10-03: A distinct fill per layer, slot and place.
fn seed(gpu: &MockGpuBackend, p: &SsmStatePool, slot: usize) {
    for l in 0..p.num_ssm_layers {
        let tag = |k: usize| ((l * 131 + slot * 17 + k * 7) % 251 + 1) as u8;
        gpu.copy_h2d(&vec![tag(0); p.h_stored_bytes], p.h_state(l, slot)).unwrap();
        gpu.copy_h2d(&vec![tag(1); p.conv_bytes], p.conv_state(l, slot)).unwrap();
        gpu.copy_h2d(&vec![tag(2); p.h_stored_bytes], p.h_checkpoint(l, slot)).unwrap();
        gpu.copy_h2d(&vec![tag(3); p.conv_bytes], p.conv_checkpoint(l, slot)).unwrap();
        for t in 0..p.h_inter_count(slot) {
            gpu.copy_h2d(&vec![tag(4 + t); p.h_stored_bytes], p.h_intermediate(l, slot, t))
                .unwrap();
        }
        for t in 0..p.num_intermediates {
            gpu.copy_h2d(&vec![tag(9 + t); p.conv_bytes], p.conv_intermediate(l, slot, t))
                .unwrap();
        }
    }
}

fn read(gpu: &MockGpuBackend, ptr: DevicePtr, bytes: usize) -> Vec<u8> {
    let mut v = vec![0u8; bytes];
    gpu.copy_d2h(ptr, &mut v).unwrap();
    v
}

#[test]
fn a_commit_lands_every_layers_accepted_row_in_its_live_slot() {
    let (gpu, pool, programs) = setup();
    let slot = 1;
    seed(&gpu, &pool, slot);
    seed(&gpu, &pool, 2);
    let neighbour = read(&gpu, pool.h_state(3, 2), pool.h_stored_bytes);
    let nodes = programs.nodes(StateProgramId::CommitAccepted).unwrap();
    let p = Places {
        slot,
        step: Some(1),
        cache: None,
    };
    pool.run_copies(nodes, p, Parts::ALL, &gpu, 0).unwrap();
    for l in 0..pool.num_ssm_layers {
        assert_eq!(
            read(&gpu, pool.h_state(l, slot), pool.h_stored_bytes),
            read(&gpu, pool.h_intermediate(l, slot, 1), pool.h_stored_bytes),
            "layer {l} h"
        );
        assert_eq!(
            read(&gpu, pool.conv_state(l, slot), pool.conv_bytes),
            read(&gpu, pool.conv_intermediate(l, slot, 1), pool.conv_bytes),
            "layer {l} conv"
        );
    }
    assert_eq!(read(&gpu, pool.h_state(3, 2), pool.h_stored_bytes), neighbour);
}

/// 2026-10-03: A commit whose h the fold already placed copies conv only.
#[test]
fn a_conv_only_commit_leaves_h_alone() {
    let (gpu, pool, programs) = setup();
    seed(&gpu, &pool, 0);
    let h_before = read(&gpu, pool.h_state(0, 0), pool.h_stored_bytes);
    let p = Places {
        slot: 0,
        step: Some(0),
        cache: None,
    };
    let conv_only = Parts {
        h: false,
        conv: true,
    };
    pool.run_copies(
        programs.nodes(StateProgramId::CommitAccepted).unwrap(),
        p,
        conv_only,
        &gpu,
        0,
    )
    .unwrap();
    assert_eq!(read(&gpu, pool.h_state(0, 0), pool.h_stored_bytes), h_before);
    assert_eq!(
        read(&gpu, pool.conv_state(0, 0), pool.conv_bytes),
        read(&gpu, pool.conv_intermediate(0, 0, 0), pool.conv_bytes)
    );
}

/// 2026-10-03: Checkpoint then rollback round-trips the live state; a commit past the slot's
/// h intermediates is refused before any copy.
#[test]
fn checkpoint_rollback_round_trips_and_an_overlong_commit_is_refused() {
    let (gpu, pool, programs) = setup();
    seed(&gpu, &pool, 2);
    let live = read(&gpu, pool.conv_state(5, 2), pool.conv_bytes);
    let p = Places {
        slot: 2,
        step: None,
        cache: None,
    };
    pool.run_copies(programs.nodes(StateProgramId::VerifyCheckpoint).unwrap(), p, Parts::ALL, &gpu, 0)
        .unwrap();
    gpu.copy_h2d(&vec![0xEE; pool.conv_bytes], pool.conv_state(5, 2)).unwrap();
    pool.run_copies(programs.nodes(StateProgramId::VerifyRollback).unwrap(), p, Parts::ALL, &gpu, 0)
        .unwrap();
    assert_eq!(read(&gpu, pool.conv_state(5, 2), pool.conv_bytes), live);

    let too_far = Places {
        step: Some(pool.h_inter_count(2)),
        ..p
    };
    let before = read(&gpu, pool.conv_state(0, 2), pool.conv_bytes);
    assert!(
        pool.run_copies(programs.nodes(StateProgramId::CommitAccepted).unwrap(), too_far, Parts::ALL, &gpu, 0)
            .is_err()
    );
    assert_eq!(read(&gpu, pool.conv_state(0, 2), pool.conv_bytes), before);
}

#[test]
fn slot_zero_clears_only_that_slot() {
    let (gpu, pool, programs) = setup();
    seed(&gpu, &pool, 0);
    seed(&gpu, &pool, 1);
    let other = read(&gpu, pool.h_state(2, 0), pool.h_stored_bytes);
    pool.run_zero(programs.nodes(StateProgramId::SlotZero).unwrap(), 1, &gpu, 0)
        .unwrap();
    for l in 0..pool.num_ssm_layers {
        assert!(read(&gpu, pool.h_state(l, 1), pool.h_stored_bytes).iter().all(|&b| b == 0));
        assert!(read(&gpu, pool.conv_state(l, 1), pool.conv_bytes).iter().all(|&b| b == 0));
    }
    assert_eq!(read(&gpu, pool.h_state(2, 0), pool.h_stored_bytes), other);
}
