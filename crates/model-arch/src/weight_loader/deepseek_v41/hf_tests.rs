// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-10-19: The safetensors source reads exactly a rank's rows or columns of a tensor written
//! to a real file, with the matching scale tiles, in the stored precision; the engram projection
//! stays E4M3 and whole; and a NaN scale is refused.
//!
//! The GPU is `MockGpuBackend`, which holds the uploaded bytes in host memory: the source's only
//! device work is uploads, and the test reads back exactly what was uploaded.

use std::io::Write;

use metrale_gpu_runtime::gpu::mock::MockGpuBackend;

use super::*;

/// 2026-10-19: Heads 2..4 of 6 (32 values each), output group 1 of 3 (32 wide).
fn local() -> Local {
    Local {
        world: 3,
        heads: 2..4,
        groups: 1..2,
        inter: 32..64,
        head_dim: 32,
        o_rank: 32,
    }
}

pub(crate) struct File {
    _dir: tempfile::TempDir,
    pub(crate) store: WeightStore,
}

/// 2026-10-19: A shard file holding `tensors` back to back, each deferred at its offset.
pub(crate) fn shard(tensors: Vec<(String, WeightDtype, Vec<usize>, Vec<u8>)>) -> File {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model-00001-of-00001.safetensors");
    let mut f = std::fs::File::create(&path).unwrap();
    let mut store = WeightStore::empty();
    let mut at = 0u64;
    for (n, dtype, shape, bytes) in tensors {
        f.write_all(&bytes).unwrap();
        let d =
            DeferredTensor::at_span(path.clone(), at, bytes.len() as u64, shape, dtype).unwrap();
        store.defer(n, d);
        at += bytes.len() as u64;
    }
    File { _dir: dir, store }
}

pub(crate) fn pattern(n: usize, seed: usize) -> Vec<u8> {
    (0..n).map(|i| ((i * 31 + seed) % 251) as u8).collect()
}

pub(crate) fn back(g: &MockGpuBackend, p: DevicePtr, n: usize) -> Vec<u8> {
    let mut b = vec![0u8; n];
    g.copy_d2h(p, &mut b).unwrap();
    b
}

#[test]
fn an_fp8_row_cut_uploads_its_rows_and_scale_tile_rows() {
    let (rows, cols) = (6 * 32, 64);
    let n = name(Format::Safetensors, 0, T::WqB);
    let (w, s) = (pattern(rows * cols, 1), pattern(6 * 2, 2));
    let f = shard(vec![
        (n.clone(), WeightDtype::FP8E4M3, vec![rows, cols], w.clone()),
        (
            scale_of(&n).unwrap(),
            WeightDtype::FP8E8M0,
            vec![6, 2],
            s.clone(),
        ),
    ]);
    let g = MockGpuBackend::new();
    let ResidentMat::Fp8Block { w: dw, scale, qdq } =
        HfSource::new(&f.store, local()).mat(&g, 0, T::WqB).unwrap()
    else {
        panic!("not FP8")
    };
    assert!(qdq);
    assert_eq!(back(&g, dw, 64 * cols), w[64 * cols..128 * cols]);
    assert_eq!(back(&g, scale, 4), s[4..8]);
}

#[test]
fn an_fp8_column_cut_keeps_its_columns_and_wo_a_stays_unrounded() {
    let (rows, cols) = (64, 96);
    let n = name(Format::Safetensors, 3, T::WoB);
    let (w, s) = (pattern(rows * cols, 3), pattern(2 * 3, 4));
    let wa = name(Format::Safetensors, 3, T::WoA);
    let f = shard(vec![
        (n.clone(), WeightDtype::FP8E4M3, vec![rows, cols], w.clone()),
        (
            scale_of(&n).unwrap(),
            WeightDtype::FP8E8M0,
            vec![2, 3],
            s.clone(),
        ),
        (
            wa.clone(),
            WeightDtype::FP8E4M3,
            vec![96, 32],
            pattern(96 * 32, 5),
        ),
        (
            scale_of(&wa).unwrap(),
            WeightDtype::FP8E8M0,
            vec![3, 1],
            vec![127; 3],
        ),
    ]);
    let g = MockGpuBackend::new();
    let src = HfSource::new(&f.store, local());
    let ResidentMat::Fp8Block { w: dw, scale, .. } = src.mat(&g, 3, T::WoB).unwrap() else {
        panic!("not FP8")
    };
    let want: Vec<u8> = (0..rows)
        .flat_map(|r| w[r * cols + 32..r * cols + 64].to_vec())
        .collect();
    assert_eq!(back(&g, dw, rows * 32), want);
    assert_eq!(back(&g, scale, 2), vec![s[1], s[4]]);
    let ResidentMat::Fp8Block { w: aw, qdq, .. } = src.mat(&g, 3, T::WoA).unwrap() else {
        panic!("not FP8")
    };
    assert!(!qdq, "wo_a reads its input widened");
    assert_eq!(back(&g, aw, 32 * 32), pattern(96 * 32, 5)[32 * 32..64 * 32]);
}

#[test]
fn an_fp8_weight_without_whole_tile_scales_is_refused() {
    let n = name(Format::Safetensors, 0, T::WqA);
    let f = shard(vec![
        (
            n.clone(),
            WeightDtype::FP8E4M3,
            vec![64, 64],
            pattern(64 * 64, 0),
        ),
        (
            scale_of(&n).unwrap(),
            WeightDtype::FP8E8M0,
            vec![64, 2],
            pattern(128, 0),
        ),
    ]);
    let e = HfSource::new(&f.store, local())
        .mat(&MockGpuBackend::new(), 0, T::WqA)
        .err()
        .unwrap()
        .to_string();
    assert!(e.contains("needs an E8M0 [2, 2] scale"), "{e}");
}

#[test]
fn a_sink_widens_its_rank_rows_and_a_router_uploads_once() {
    let sink: Vec<f32> = (0..6).map(|i| i as f32 + 0.5).collect();
    let gate = pattern(8 * 2, 7);
    let f = shard(vec![
        (
            name(Format::Safetensors, 1, T::Sink),
            WeightDtype::FP32,
            vec![6],
            sink.iter().flat_map(|x| x.to_le_bytes()).collect(),
        ),
        (
            name(Format::Safetensors, 1, T::GateW),
            WeightDtype::BF16,
            vec![4, 2],
            gate.clone(),
        ),
    ]);
    let g = MockGpuBackend::new();
    let src = HfSource::new(&f.store, local());
    assert_eq!(src.host_f32(&g, 1, T::Sink).unwrap(), vec![2.5, 3.5]);
    let p = src.bf16(&g, 1, T::GateW).unwrap();
    assert_eq!(src.bf16(&g, 1, T::GateW).unwrap(), p);
    assert_eq!(back(&g, p, 16), gate);
    assert!(src.has(1, T::GateW) && !src.has(2, T::GateW));
    assert!(
        src.bf16(&g, 1, T::Sink).is_err(),
        "an f32 tensor is not bf16"
    );
}

/// 2026-10-19: The engram projection is replicated: every rank uploads the whole E4M3 weight and
/// its whole scale plane as stored (the kernel decodes them), and rounds its rows first.
#[test]
fn the_engram_projection_uploads_its_whole_e4m3_weight_and_scales() {
    let (rows, cols) = (64, 96);
    let n = name(Format::Safetensors, 1, T::EngramWkv);
    let (w, s) = (pattern(rows * cols, 6), pattern(2 * 3, 7));
    let f = shard(vec![
        (n.clone(), WeightDtype::FP8E4M3, vec![rows, cols], w.clone()),
        (
            scale_of(&n).unwrap(),
            WeightDtype::FP8E8M0,
            vec![2, 3],
            s.clone(),
        ),
    ]);
    let g = MockGpuBackend::new();
    let e = HfSource::new(&f.store, local()).engram_wkv(&g, 1).unwrap();
    let ResidentMat::Fp8Block { w: dw, scale, qdq } = e.wkv else {
        panic!("{:?}", e.wkv)
    };
    assert!(qdq && e.wkv_q2k.is_null());
    assert_eq!(back(&g, dw, rows * cols), w);
    assert_eq!(back(&g, scale, 6), s);
}

/// 2026-10-19: 0xFF is E8M0's NaN; a weight carrying it is refused at load, whichever
/// projection reads it.
#[test]
fn an_fp8_scale_holding_the_nan_code_is_refused() {
    for t in [T::WqB, T::EngramWkv] {
        let n = name(Format::Safetensors, 1, t);
        let f = shard(vec![
            (
                n.clone(),
                WeightDtype::FP8E4M3,
                vec![6 * 32, 32],
                pattern(6 * 32 * 32, 1),
            ),
            (
                scale_of(&n).unwrap(),
                WeightDtype::FP8E8M0,
                vec![6, 1],
                vec![127, 127, 127, 0xFF, 127, 127],
            ),
        ]);
        let g = MockGpuBackend::new();
        let e = HfSource::new(&f.store, local())
            .mat(&g, 1, t)
            .unwrap_err()
            .to_string();
        assert!(e.contains("NaN"), "{t:?}: {e}");
    }
}
