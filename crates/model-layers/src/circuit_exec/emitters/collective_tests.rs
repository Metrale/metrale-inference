// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: A tensor-parallel rank's program (rank 0 of 2 of the dense circuit, no draft head)
//! over the fixture's bindings at the rank's shape, run on the recording mock with a recording
//! communicator: one in-place sum per row-parallel output, of the step's rows, on the step's
//! stream; and the refusal without a communicator.
//!
//! Owner: model-layers circuit executor (FEATURES workstream).
//! Invariants: none beyond the types.

use std::sync::{Arc, Mutex};

use anyhow::{Result, bail};
use metrale_circuit::parallel::{TpRank, rank_shape, with_reduces};
use metrale_circuit::planner::plan_buffers_with;
use metrale_circuit::{AvailableKernels, Circuit, LayerKind, LinearRole, Mode, OpKind};
use metrale_comm::CommBackend;
use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::super::super::bindings::{CircuitLayer, HeadBinding, MixerFacts};
use super::super::super::compile::{self, Inputs};
use super::super::super::exec_fixture::{
    RECIPE, WORKSPACE, config, dense, fixed, layer_binding, ptr,
};
use super::super::super::kernels::KernelTable;
use super::super::super::program::{GdnState, LaunchKind, Program, StepEnv};
use super::super::super::sources;
use crate::weight_map::DenseWeight;

/// 2026-10-03: A communicator that records every asynchronous all-reduce.
#[derive(Default)]
struct Recording(Mutex<Vec<(u64, usize, u64)>>);

impl CommBackend for Recording {
    fn all_reduce(&self, _ptr: u64, _bytes: usize) -> Result<()> {
        bail!("the circuit reduces asynchronously")
    }
    fn all_reduce_async(&self, ptr: u64, bytes: usize, stream: u64) -> Result<()> {
        self.0.lock().unwrap().push((ptr, bytes, stream));
        Ok(())
    }
    fn all_gather(&self, _: u64, _: u64, _: usize) -> Result<()> {
        bail!("unused")
    }
    fn reduce_scatter(&self, _: u64, _: u64, _: usize) -> Result<()> {
        bail!("unused")
    }
    fn broadcast(&self, _: u64, _: usize, _: usize) -> Result<()> {
        bail!("unused")
    }
    fn barrier(&self) -> Result<()> {
        bail!("unused")
    }
    fn send_to(&self, _: u64, _: usize, _: usize, _: u64) -> Result<()> {
        bail!("unused")
    }
    fn recv_from(&self, _: u64, _: usize, _: usize, _: u64) -> Result<()> {
        bail!("unused")
    }
    fn rank(&self) -> usize {
        0
    }
    fn world_size(&self) -> usize {
        2
    }
}

/// 2026-10-03: The logits buffer: the first allocation of the run's mock (the head zeroes it).
const LOGITS: DevicePtr = DevicePtr(0x1000_0000);

struct Rank {
    circuit: Circuit,
    program: Program,
    layers: Vec<CircuitLayer>,
}

/// 2026-10-03: Rank 0 of 2, no draft head (`mtp = 0`), compiled at `mode` x `rows` with `comm`.
fn build(mode: Mode, rows: u64, comm: Option<Arc<dyn CommBackend>>) -> Result<Rank> {
    let mut inst = sources::instance(RECIPE)?;
    inst.shape.dims.insert("mtp".into(), 0);
    inst.shape = rank_shape(&inst.shape, TpRank { rank: 0, world: 2 })?;
    let mut loaded = metrale_circuit::load(&inst, sources::sources(&inst)?)?;
    loaded.circuit = with_reduces(&loaded.circuit, TpRank { rank: 0, world: 2 })?;
    let avail = AvailableKernels::all_named_by(&loaded.rules);
    let plan = metrale_circuit::fuse(
        &loaded.circuit,
        &loaded.rules,
        &avail,
        &inst.policy,
        mode,
        rows,
    )?;
    let layout = compile::layout(&loaded.circuit, &plan)?;
    let buffers = plan_buffers_with(&loaded.circuit, &plan, rows, &layout)?;
    let gpu = MockGpuBackend::new();
    crate::layers::ops::w4a4_proj::prepare(&gpu)?;
    let mut attn = 0;
    let layers: Vec<CircuitLayer> = (0..loaded.circuit.layer_kinds.len())
        .map(|i| {
            let l = layer_binding(&loaded.circuit, i, attn);
            attn += usize::from(matches!(l.mixer, MixerFacts::Attention(_)));
            l
        })
        .collect();
    let mut fx = fixed(attn);
    fx.draft = None;
    fx.comm = comm;
    fx.logits = LOGITS;
    let head = HeadBinding {
        final_norm: DenseWeight {
            weight: ptr(0x9000_0000),
        },
        lm_head: dense(0x9100_0000),
        unmodelled: Vec::new(),
        batchm_max_rows: 8,
    };
    let cfg = config();
    let program = compile::compile(
        &loaded.circuit,
        &plan,
        &layout,
        &buffers,
        ptr(WORKSPACE),
        &Inputs {
            gpu: &gpu,
            config: &cfg,
            kernels: &KernelTable::resolve(&gpu, &avail),
            fixed: &fx,
            layers: &layers,
            head: &head,
            draft: None,
            arena: None,
        },
    )?;
    Ok(Rank {
        circuit: loaded.circuit,
        program,
        layers,
    })
}

fn run(r: &Rank, rows: usize) -> Vec<metrale_gpu_runtime::gpu::mock::MockLaunch> {
    let gpu = MockGpuBackend::new();
    gpu.set_kernel_n_tile(KernelHandle(0xDEAD), 128);
    let v = r.circuit.dims["vocab"] as usize;
    assert_eq!(gpu.alloc(rows * v * 2).unwrap(), LOGITS);
    let gdn: Vec<Vec<GdnState>> = r
        .layers
        .iter()
        .enumerate()
        .map(|(i, l)| match l.mixer {
            MixerFacts::Gdn(_) => (0..rows)
                .map(|row| {
                    let at = 0xD000_0000 + ((i as u64) << 24) + ((row as u64) << 16);
                    GdnState {
                        h: ptr(at),
                        conv: ptr(at + 0x8000),
                        h_steps: [1, 2, 3].map(|t| ptr(at + 0x1000 * t)),
                        conv_steps: [1, 2, 3].map(|t| ptr(at + 0x8000 + 0x1000 * t)),
                    }
                })
                .collect(),
            MixerFacts::Attention(_) => Vec::new(),
        })
        .collect();
    r.program
        .run(&StepEnv {
            gpu: &gpu,
            stream: 7,
            gdn: &gdn,
            max_blocks_per_seq: 9,
            prefill: None,
        })
        .unwrap();
    gpu.launches_snapshot()
}

#[test]
fn each_row_parallel_output_is_summed_in_place_over_the_steps_rows() {
    for (mode, rows) in [(Mode::Decode, 1usize), (Mode::MultiSeq, 4)] {
        let comm = Arc::new(Recording::default());
        let r = build(mode, rows as u64, Some(comm.clone())).unwrap();
        let launched = run(&r, rows);
        let calls = comm.0.lock().unwrap().clone();
        let reduced = r
            .circuit
            .nodes
            .iter()
            .filter(|n| n.op == OpKind::AllReduce)
            .count();
        assert_eq!(
            reduced,
            r.circuit.layer_kinds.len() + 1,
            "one per layer (o or out_proj) and the head's"
        );
        assert_eq!(calls.len(), reduced, "{mode:?}");
        let h = r.circuit.dims["hidden"] as usize;
        let v = r.circuit.dims["vocab"] as usize;
        let (head, calls): (Vec<_>, Vec<_>) = calls.into_iter().partition(|c| c.1 == rows * v * 2);
        assert_eq!(head.len(), 1, "{mode:?}: the logits are summed once, last");
        assert_eq!(head[0].0, LOGITS.0, "in place, in the logits buffer");
        let kernel_launches: Vec<_> = r
            .program
            .launches
            .iter()
            .filter(|l| l.kind == LaunchKind::Kernel)
            .collect();
        assert_eq!(kernel_launches.len(), launched.len());
        for (p, bytes, stream) in calls {
            assert_eq!((bytes, stream), (rows * h * 2, 7), "{mode:?}");
            let reads = launched
                .iter()
                .filter(|m| m.args.contains(&MockArg::Buffer(DevicePtr(p))))
                .count();
            assert!(
                reads >= 2,
                "{mode:?}: {p:#x} is written by the projection and read after the sum"
            );
        }
    }
}

#[test]
fn the_rank_shards_the_mixers_and_keeps_the_ffn_whole() {
    let r = build(Mode::Decode, 1, Some(Arc::new(Recording::default()))).unwrap();
    assert_eq!(r.circuit.dims["q_heads"], 12);
    assert_eq!(r.circuit.dims["lin_v_heads"], 24);
    let o = r
        .circuit
        .nodes
        .iter()
        .find(|n| n.op == OpKind::Linear(LinearRole::O))
        .unwrap();
    assert_eq!(
        r.circuit.edges[o.inputs[0]].dim_value,
        12 * 256,
        "o reads this rank's heads"
    );
    assert_eq!(
        r.circuit.edges[o.outputs[0]].dim_value, 5120,
        "and writes a whole partial row"
    );
    assert_eq!(r.circuit.layer_kinds[3], LayerKind::FullAttention);
}

#[test]
fn a_rank_plan_without_a_communicator_is_refused() {
    let e = build(Mode::Decode, 1, None).err().unwrap();
    assert!(
        format!("{e:#}").contains("without the model's communicator"),
        "{e:#}"
    );
}

#[test]
fn the_head_runs_its_vocabulary_slice_into_zeroed_logits_and_skips_the_sum_when_wide() {
    let comm = Arc::new(Recording::default());
    let r = build(Mode::Decode, 1, Some(comm.clone())).unwrap();
    let launched = run(&r, 1);
    let v = r.circuit.dims["vocab"] as u64;
    let head = launched
        .iter()
        .filter(|m| m.args.contains(&MockArg::Buffer(LOGITS)))
        .count();
    assert_eq!(head, 1, "rank 0's slice starts at the logits' first row");
    let weight = dense(0x9100_0000);
    let crate::circuit_exec::BoundWeight::Dense(w) = weight else {
        unreachable!()
    };
    assert!(
        launched
            .iter()
            .any(|m| m.args.contains(&MockArg::Buffer(w.weight))),
        "rank 0 reads the head from its first row"
    );
    let wide = build(Mode::MultiSeq, 16, Some(comm.clone())).unwrap();
    let before = comm.0.lock().unwrap().len();
    run(&wide, 16);
    let sums = comm.0.lock().unwrap()[before..].to_vec();
    assert!(
        sums.iter().all(|c| c.1 != 16 * v as usize * 2),
        "no sum of the logits at 16 rows"
    );
}
