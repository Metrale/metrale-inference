// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: CPU/mock-comm tests for the engine-level DFlash TP readiness
//! rendezvous. These tests run the production vote/gate logic over the same
//! rooted-broadcast gather and one reused 4-byte buffer as the EP command path.

use super::{DFlashProposeReadiness, TransformerModel, dflash_propose_readiness_gate};
use crate::rank_agree::gather_u32_via_broadcast;
use crate::test_pair_comm::{Op, run_pair};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

const READY: DFlashProposeReadiness = DFlashProposeReadiness {
    slot_valid: true,
    proposer_collective: true,
    proposer_state: true,
};

fn worker_readiness(
    sequence_slot_idx: Option<usize>,
    proposer_collective: bool,
    proposer_state: bool,
) -> DFlashProposeReadiness {
    let (slot_valid, proposer_collective, proposer_state) =
        TransformerModel::dflash_worker_readiness(
            1,
            2,
            sequence_slot_idx,
            true,
            proposer_collective,
            proposer_state,
        );
    DFlashProposeReadiness {
        slot_valid,
        proposer_collective,
        proposer_state,
    }
}

/// 2026-10-10: Reuse the rendezvous buffer for a v2-style command preamble
/// after a negative vote, proving the shared failure left command dispatch
/// synchronized and usable.
fn next_ep_command(
    gpu: &dyn GpuBackend,
    comm: &dyn metrale_comm::CommBackend,
    buf: DevicePtr,
) -> anyhow::Result<(u32, u32)> {
    const NEXT_SLOT: u32 = 0;
    const NEXT_CMD: u32 = 0xFFFFFFF1;

    if comm.rank() == 0 {
        gpu.copy_h2d(&NEXT_SLOT.to_le_bytes(), buf)?;
        comm.broadcast(buf.0, 4, 0)?;
    } else {
        comm.recv_command_u32(buf.0, 0)?;
        gpu.synchronize(gpu.default_stream())?;
    }
    let slot = read_word(gpu, comm.rank(), buf, NEXT_SLOT)?;

    if comm.rank() == 0 {
        gpu.copy_h2d(&NEXT_CMD.to_le_bytes(), buf)?;
    }
    comm.broadcast(buf.0, 4, 0)?;
    if comm.rank() != 0 {
        gpu.synchronize(gpu.default_stream())?;
    }
    let cmd = read_word(gpu, comm.rank(), buf, NEXT_CMD)?;
    Ok((slot, cmd))
}

fn read_word(
    gpu: &dyn GpuBackend,
    rank: usize,
    buf: DevicePtr,
    root_value: u32,
) -> anyhow::Result<u32> {
    if rank == 0 {
        Ok(root_value)
    } else {
        let mut bytes = [0u8; 4];
        gpu.copy_d2h(buf, &mut bytes)?;
        Ok(u32::from_le_bytes(bytes))
    }
}

fn run_gate_pair(
    local: [DFlashProposeReadiness; 2],
) -> [(Result<u32, String>, bool, (u32, u32), Vec<Op>); 2] {
    run_pair(move |rank, gpu, comm| {
        let buf = gpu.alloc(4).expect("mock EP control buffer allocation");
        let mut model_arch_entered = false;
        let result = dflash_propose_readiness_gate(
            local[rank],
            |vote| gather_u32_via_broadcast(gpu, comm, buf, 2, vote),
            || {
                model_arch_entered = true;
                Ok(0xDFA5)
            },
        )
        .map_err(|error| format!("{error:#}"));
        let next = next_ep_command(gpu, comm, buf).expect("subsequent EP command");
        gpu.free(buf).expect("mock EP control buffer free");
        (result, model_arch_entered, next)
    })
    .map(|((result, entered, next), ops)| (result, entered, next, ops))
}

fn assert_command_path_survived(results: &[(Result<u32, String>, bool, (u32, u32), Vec<Op>); 2]) {
    for (_, _, next, ops) in results {
        assert_eq!(*next, (0, 0xFFFFFFF1));
        assert_eq!(
            ops.as_slice(),
            &[(false, 0), (false, 1), (false, 0), (false, 0)]
        );
    }
}

#[test]
fn dflash_readiness_all_ready_enters_model_arch_on_both_ranks() {
    let results = run_gate_pair([READY, READY]);
    for (result, entered, _, _) in &results {
        assert_eq!(result.as_ref().unwrap(), &0xDFA5);
        assert!(
            *entered,
            "ready ranks must proceed into the gated operation"
        );
    }
    assert_command_path_survived(&results);
}

#[test]
fn dflash_worker_missing_slot_rejects_both_ranks_before_model_arch() {
    // An absent slot cannot supply proposer state either. Use the same worker
    // vote builder as ep_worker_execute, before its generic slot early return.
    let worker_missing_slot = worker_readiness(None, true, false);
    let results = run_gate_pair([READY, worker_missing_slot]);
    let root_error = results[0].0.as_ref().unwrap_err();
    for (result, entered, _, _) in &results {
        assert!(
            result.is_err(),
            "rank must share the worker's negative vote"
        );
        assert!(!entered, "negative vote must prevent model-arch preflight");
        assert_eq!(result.as_ref().unwrap_err(), root_error);
    }
    assert!(root_error.contains("rank 1 (slot_valid=false"));
    assert_command_path_survived(&results);
}

#[test]
fn dflash_worker_missing_proposer_state_rejects_both_ranks_before_model_arch() {
    let worker_missing_state = worker_readiness(Some(1), true, false);
    let results = run_gate_pair([READY, worker_missing_state]);
    let root_error = results[0].0.as_ref().unwrap_err();
    for (result, entered, _, _) in &results {
        assert!(
            result.is_err(),
            "rank must share the worker's negative vote"
        );
        assert!(!entered, "negative vote must prevent model-arch preflight");
        assert_eq!(result.as_ref().unwrap_err(), root_error);
    }
    assert!(
        root_error
            .contains("rank 1 (slot_valid=true, proposer_collective=true, proposer_state=false")
    );
    assert_command_path_survived(&results);
}

#[test]
fn dflash_worker_noncollective_proposer_rejects_both_ranks_before_model_arch() {
    let worker_noncollective = worker_readiness(Some(1), false, true);
    let results = run_gate_pair([READY, worker_noncollective]);
    let root_error = results[0].0.as_ref().unwrap_err();
    for (result, entered, _, _) in &results {
        assert!(
            result.is_err(),
            "rank must share the worker's negative vote"
        );
        assert!(!entered, "negative vote must prevent model-arch preflight");
        assert_eq!(result.as_ref().unwrap_err(), root_error);
    }
    assert!(root_error.contains("rank 1 (slot_valid=true, proposer_collective=false"));
    assert_command_path_survived(&results);
}
