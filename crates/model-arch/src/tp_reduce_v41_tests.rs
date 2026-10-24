// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-10-19: The tensor-parallel reduction. On the mock: one rank never gathers and combines
//! its own partials; several ranks gather exactly the call's rows into distinct slots, drain a
//! foreign stream around the gather, and refuse a missing or mismatched communicator and an
//! oversized call. On the GPU: `moe_v41_rank_combine` equals a CPU emulation of the reference's
//! rounding sequence bit for bit.
//!
//! Owner: model-arch, DeepSeek-V4.1.
//! Invariants: none beyond the types.

use super::*;
use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend};
use std::sync::Mutex;

/// 2026-10-19: A communicator that records each all-gather and moves no bytes; the mock's
/// combine is not executed, so only the call's geometry is under test.
pub(crate) struct RecordingComm {
    pub(crate) world: usize,
    pub(crate) gathers: Mutex<Vec<(u64, u64, usize)>>,
}

impl CommBackend for RecordingComm {
    fn all_reduce(&self, _ptr: u64, _bytes: usize) -> Result<()> {
        anyhow::bail!("the reduce must not all-reduce")
    }
    fn all_gather(&self, send_ptr: u64, recv_ptr: u64, bytes: usize) -> Result<()> {
        self.gathers
            .lock()
            .unwrap()
            .push((send_ptr, recv_ptr, bytes));
        Ok(())
    }
    fn reduce_scatter(&self, _s: u64, _r: u64, _bytes: usize) -> Result<()> {
        anyhow::bail!("the reduce must not reduce-scatter")
    }
    fn broadcast(&self, _ptr: u64, _bytes: usize, _root: usize) -> Result<()> {
        anyhow::bail!("the reduce must not broadcast")
    }
    fn send_to(&self, _ptr: u64, _bytes: usize, _peer: usize, _stream: u64) -> Result<()> {
        anyhow::bail!("the reduce must not send")
    }
    fn recv_from(&self, _ptr: u64, _bytes: usize, _peer: usize, _stream: u64) -> Result<()> {
        anyhow::bail!("the reduce must not recv")
    }
    fn barrier(&self) -> Result<()> {
        Ok(())
    }
    fn rank(&self) -> usize {
        0
    }
    fn world_size(&self) -> usize {
        self.world
    }
}

fn u32_arg(a: &MockArg) -> u32 {
    match a {
        MockArg::Bytes(b) => u32::from_le_bytes(b[..4].try_into().unwrap()),
        other => panic!("{other:?}"),
    }
}

fn u64_arg(a: &MockArg) -> u64 {
    match a {
        MockArg::Bytes(b) => u64::from_le_bytes(b[..8].try_into().unwrap()),
        other => panic!("{other:?}"),
    }
}

const DIM: usize = 512;

fn rows(rows: usize, topk: usize, shared_base: usize, m: usize) -> CombineRows {
    CombineRows {
        rows,
        pos: DevicePtr(0x77),
        topk,
        shared_base,
        m,
    }
}

#[test]
fn one_rank_combines_its_own_partials_without_a_gather() {
    let g = MockGpuBackend::new();
    let r = TpReduce::new(&g, 1, 16, DIM).unwrap();
    let out = DevicePtr(0x99);
    r.reduce(&g, None, rows(3, 0, 0, 3), out, 0).unwrap();
    let l = g.launches_snapshot();
    assert_eq!(l.len(), 1);
    assert_eq!(l[0].grid, [2, 3, 1]);
    assert_eq!(
        l[0].args[0],
        MockArg::Buffer(r.partial()),
        "G is the partial buffer"
    );
    assert_eq!(u64_arg(&l[0].args[1]), (3 * DIM) as u64);
    assert_eq!(u32_arg(&l[0].args[2]), 1);
    assert_eq!(
        l[0].args[3],
        MockArg::Buffer(r.partial()),
        "no position list without topk"
    );
    assert_eq!(l[0].args[6], MockArg::Buffer(out));
}

#[test]
fn three_ranks_gather_exactly_the_calls_rows_into_their_own_slots() {
    let g = MockGpuBackend::new();
    let r = TpReduce::new(&g, 3, 64, DIM).unwrap();
    let comm = RecordingComm {
        world: 3,
        gathers: Mutex::new(Vec::new()),
    };
    // 2026-10-19: Two tokens of top-2: four expert rows, then the two shared rows.
    r.reduce(&g, Some(&comm), rows(6, 2, 4, 2), DevicePtr(0x99), 0)
        .unwrap();
    let gathers = comm.gathers.lock().unwrap().clone();
    assert_eq!(gathers.len(), 1);
    let (send, recv, bytes) = gathers[0];
    assert_eq!(send, r.partial().0);
    assert_ne!(recv, send, "the gather lands outside the partials");
    assert_eq!(bytes, 6 * DIM * 4);
    let l = g.launches_snapshot();
    assert_eq!(l[0].args[0], MockArg::Buffer(DevicePtr(recv)));
    assert_eq!(
        u64_arg(&l[0].args[1]),
        (6 * DIM) as u64,
        "slot stride = gathered bytes"
    );
    assert_eq!(u32_arg(&l[0].args[2]), 3);
    assert_eq!(l[0].args[3], MockArg::Buffer(DevicePtr(0x77)));
    assert_eq!((u32_arg(&l[0].args[4]), u32_arg(&l[0].args[5])), (2, 4));
}

#[test]
fn a_foreign_stream_is_drained_around_the_gather() {
    let g = MockGpuBackend::new();
    let r = TpReduce::new(&g, 3, 8, DIM).unwrap();
    let comm = RecordingComm {
        world: 3,
        gathers: Mutex::new(Vec::new()),
    };
    r.reduce(&g, Some(&comm), rows(2, 0, 0, 2), DevicePtr(0x99), 0)
        .unwrap();
    assert_eq!(
        g.sync_count(),
        0,
        "the default stream orders the gather itself"
    );
    r.reduce(&g, Some(&comm), rows(2, 0, 0, 2), DevicePtr(0x99), 5)
        .unwrap();
    assert_eq!(g.sync_count(), 2);
}

#[test]
fn a_missing_or_mismatched_communicator_and_an_oversized_call_are_refused() {
    let g = MockGpuBackend::new();
    let r = TpReduce::new(&g, 3, 8, DIM).unwrap();
    let e = r
        .reduce(&g, None, rows(2, 0, 0, 2), DevicePtr(0x99), 0)
        .unwrap_err();
    assert!(e.to_string().contains("without a communicator"), "{e}");
    let two = RecordingComm {
        world: 2,
        gathers: Mutex::new(Vec::new()),
    };
    let e = r
        .reduce(&g, Some(&two), rows(2, 0, 0, 2), DevicePtr(0x99), 0)
        .unwrap_err();
    assert!(e.to_string().contains("spans 2 ranks"), "{e}");
    let e = r
        .reduce(&g, Some(&two), rows(9, 0, 0, 2), DevicePtr(0x99), 0)
        .unwrap_err();
    assert!(e.to_string().contains("exceeds 8 rows"), "{e}");
    let e = r
        .reduce(&g, Some(&two), rows(4, 2, 3, 2), DevicePtr(0x99), 0)
        .unwrap_err();
    assert!(e.to_string().contains("exceeds"), "{e}");
    assert!(g.launches_snapshot().is_empty());
    assert!(two.gathers.lock().unwrap().is_empty());
}

#[cfg(feature = "cuda")]
mod gpu {
    use super::*;

    fn backend() -> metrale_gpu_runtime::cuda_backend::MetraleCudaBackend {
        let set = metrale_kernels::ptx_for_exact_target("deepseek-v4-flash", "nvfp4")
            .expect("deepseek-v4-flash/nvfp4 not in this build");
        metrale_gpu_runtime::cuda_backend::MetraleCudaBackend::new(0, &set.modules)
            .expect("CUDA backend")
    }

    fn bf16_round(x: f32) -> f32 {
        let b = x.to_bits();
        f32::from_bits(((b.wrapping_add(0x7FFF + ((b >> 16) & 1))) >> 16) << 16)
    }

    #[test]
    #[ignore = "requires a CUDA GB10 + the deepseek-v4-flash kernel target"]
    fn rank_combine_matches_the_reference_rounding_sequence() {
        let gpu = backend();
        let g: &dyn GpuBackend = &gpu;
        let stream = g.default_stream();
        let (world, m, topk, dim) = (3usize, 5usize, 6usize, 300usize);
        let rows = m * topk + m;
        let mut seed = 0x1234_5678u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as u32 as f32 / u32::MAX as f32 - 0.5) * 8.0
        };
        let gath: Vec<f32> = (0..world * rows * dim).map(|_| next()).collect();
        // 2026-10-19: A permutation of the expert rows, so each token's list is not contiguous.
        let pos: Vec<i32> = (0..m * topk)
            .map(|i| ((i * 7) % (m * topk)) as i32)
            .collect();
        let up = |b: &[u8]| {
            let p = g.alloc(b.len()).unwrap();
            g.copy_h2d(b, p).unwrap();
            p
        };
        let gd = up(&gath
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>());
        let pd = up(&pos.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
        let od = g.alloc(m * dim * 2).unwrap();
        let k = g.kernel("moe_v41", "moe_v41_rank_combine").unwrap();
        KernelLaunch::new(g, k)
            .grid([(dim as u32).div_ceil(256), m as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(gd)
            .arg_u64((rows * dim) as u64)
            .arg_u32(world as u32)
            .arg_ptr(pd)
            .arg_u32(topk as u32)
            .arg_u32((m * topk) as u32)
            .arg_ptr(od)
            .arg_u32(dim as u32)
            .launch(stream)
            .unwrap();
        g.synchronize(stream).unwrap();
        let mut ob = vec![0u8; m * dim * 2];
        g.copy_d2h(od, &mut ob).unwrap();
        let at = |r: usize, row: usize, d: usize| gath[r * rows * dim + row * dim + d];
        for t in 0..m {
            for d in 0..dim {
                let mut y = 0f32;
                for j in 0..topk {
                    let row = pos[t * topk + j] as usize;
                    let mut s = 0f32;
                    for r in 0..world {
                        s += at(r, row, d);
                    }
                    y += bf16_round(s);
                }
                let mut s = 0f32;
                for r in 0..world {
                    s += at(r, m * topk + t, d);
                }
                y += bf16_round(s);
                let want = (bf16_round(y).to_bits() >> 16) as u16;
                let i = t * dim + d;
                let got = u16::from_le_bytes([ob[2 * i], ob[2 * i + 1]]);
                assert_eq!(got, want, "out[{t}, {d}]");
            }
        }
        for p in [gd, pd, od] {
            g.free(p).unwrap();
        }
    }
}
