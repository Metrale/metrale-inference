// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The swap runner on the mock backend, which moves real bytes: a swap-out writes the
//! record in legacy's order (computed here from the pools directly), a swap-in into other blocks
//! and slots restores every byte, across several staging chunks; the chunking; the refusals.
//!
//! Owner: model-layers circuit executor (FEATURES workstream).
//! Invariants: none beyond the types.

use metrale_circuit::swap::{KvLayer, RecurrentUnit, Segment, SwapPlan};
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::{SwapBinding, SwapRunner, chunks};

const BLOCKS: usize = 12;

/// 2026-10-03: Two attention layers (K and V blocks of 96 and 64 bytes) and three recurrent
/// units, the largest 200 bytes, so a 400-byte chunk splits a 3-block record several times.
fn plan() -> SwapPlan {
    SwapPlan {
        kv: vec![
            KvLayer {
                layer: 1,
                k_block_bytes: 96,
                v_block_bytes: 96,
            },
            KvLayer {
                layer: 3,
                k_block_bytes: 64,
                v_block_bytes: 64,
            },
        ],
        recurrent: [("l0.gdn.h", 200), ("l0.gdn.conv", 24), ("l2.gdn.h", 200)]
            .iter()
            .map(|(s, b)| RecurrentUnit {
                state: (*s).into(),
                layer: 0,
                local: "h".into(),
                bytes: *b,
            })
            .collect(),
    }
}

/// 2026-10-03: A pool or slot filled with bytes derived from `seed`.
fn filled(gpu: &MockGpuBackend, bytes: usize, seed: u8) -> DevicePtr {
    let p = gpu.alloc(bytes).unwrap();
    let data: Vec<u8> = (0..bytes)
        .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
        .collect();
    gpu.copy_h2d(&data, p).unwrap();
    p
}

struct Model {
    bind: SwapBinding,
    recurrent: Vec<DevicePtr>,
}

fn model(gpu: &MockGpuBackend, seed: u8) -> Model {
    let p = plan();
    let kv =
        p.kv.iter()
            .enumerate()
            .map(|(i, l)| {
                let k = filled(gpu, BLOCKS * l.k_block_bytes as usize, seed + i as u8);
                let v = filled(gpu, BLOCKS * l.v_block_bytes as usize, seed + 10 + i as u8);
                [(k, l.k_block_bytes), (v, l.v_block_bytes)]
            })
            .collect();
    let recurrent = p
        .recurrent
        .iter()
        .enumerate()
        .map(|(i, r)| filled(gpu, r.bytes as usize, seed + 20 + i as u8))
        .collect();
    Model {
        bind: SwapBinding {
            kv,
            recurrent_bytes: p.recurrent.iter().map(|r| r.bytes).collect(),
        },
        recurrent,
    }
}

fn bytes_at(gpu: &MockGpuBackend, p: DevicePtr, n: u64) -> Vec<u8> {
    let mut out = vec![0u8; n as usize];
    gpu.copy_d2h(p, &mut out).unwrap();
    out
}

/// 2026-10-03: Legacy's record, read straight from the pools: each block, each layer, K then V;
/// then the recurrent units.
fn legacy_record(gpu: &MockGpuBackend, m: &Model, table: &[u32]) -> Vec<u8> {
    let mut out = Vec::new();
    for &b in table {
        for [(k, ks), (v, vs)] in &m.bind.kv {
            out.extend(bytes_at(gpu, k.offset(b as usize * *ks as usize), *ks));
            out.extend(bytes_at(gpu, v.offset(b as usize * *vs as usize), *vs));
        }
    }
    for (p, n) in m.recurrent.iter().zip(&m.bind.recurrent_bytes) {
        out.extend(bytes_at(gpu, *p, *n));
    }
    out
}

#[test]
fn a_swap_out_writes_legacys_record_and_a_swap_in_restores_it_elsewhere() {
    let gpu = MockGpuBackend::new();
    let src = model(&gpu, 1);
    let runner = SwapRunner::new(&gpu, plan(), src.bind.clone()).unwrap();
    assert_eq!(runner.staging_bytes(), 2 * 400);
    let table = [5u32, 2, 7];
    let mut file = Vec::new();
    runner
        .swap_out(&gpu, 0, (&table, &src.recurrent), &mut file)
        .unwrap();
    let want = legacy_record(&gpu, &src, &table);
    assert_eq!(file.len() as u64, runner.record_bytes(table.len() as u64));
    assert_eq!(file, want, "the record is legacy's, byte for byte");
    let segs = plan().segments(table.len());
    assert!(
        chunks(&segs, 400).len() >= 3,
        "the record crossed several chunks"
    );

    let dst = model(&gpu, 100);
    let runner_in = SwapRunner::new(&gpu, plan(), dst.bind.clone()).unwrap();
    let restored_table = [0u32, 11, 4];
    runner_in
        .swap_in(
            &gpu,
            0,
            (&restored_table, &dst.recurrent),
            &mut file.as_slice(),
        )
        .unwrap();
    assert_eq!(legacy_record(&gpu, &dst, &restored_table), want);
    let untouched = bytes_at(&gpu, dst.bind.kv[0][0].0.offset(96), 96);
    let fresh = model(&gpu, 100);
    assert_eq!(
        untouched,
        bytes_at(&gpu, fresh.bind.kv[0][0].0.offset(96), 96),
        "a block outside the table is not written"
    );
    runner.free(&gpu).unwrap();
    runner_in.free(&gpu).unwrap();
}

#[test]
fn a_short_record_fails_the_swap_in() {
    let gpu = MockGpuBackend::new();
    let m = model(&gpu, 1);
    let runner = SwapRunner::new(&gpu, plan(), m.bind.clone()).unwrap();
    let short = vec![0u8; runner.record_bytes(3) as usize - 1];
    assert!(
        runner
            .swap_in(&gpu, 0, (&[0, 1, 2], &m.recurrent), &mut short.as_slice())
            .is_err()
    );
}

#[test]
fn chunks_cover_every_segment_in_order_within_the_chunk() {
    let seg = |bytes| Segment {
        piece: metrale_circuit::swap::Piece::Recurrent { index: 0 },
        offset: 0,
        bytes,
    };
    let segs: Vec<Segment> = [100, 100, 150, 400, 10, 10].map(seg).to_vec();
    let groups = chunks(&segs, 300);
    assert_eq!(groups, vec![0..2, 2..3, 3..4, 4..6]);
    assert_eq!(chunks(&[], 300), Vec::<std::ops::Range<usize>>::new());
}

#[test]
fn a_binding_the_plan_misdescribes_is_refused() {
    let gpu = MockGpuBackend::new();
    let mut m = model(&gpu, 1);
    m.bind.kv[1][1].1 = 65;
    let e = SwapRunner::new(&gpu, plan(), m.bind.clone()).err().unwrap();
    assert!(e.to_string().contains("KV layer 1"), "{e}");
    let mut m = model(&gpu, 1);
    m.bind.recurrent_bytes[0] = 400;
    let e = SwapRunner::new(&gpu, plan(), m.bind).err().unwrap();
    assert!(e.to_string().contains("recurrent units"), "{e}");
}
