// SPDX-License-Identifier: MIT OR Apache-2.0

//! Parity test: FastSafetensorsLoader must produce byte-identical weights
//! to the mmap-based SafetensorsLoader for the same file.
//!
//! Builds a tiny synthetic safetensors file in a tempdir, loads it with both
//! loaders against a MockGpuBackend, and asserts every tensor's bytes match.

#![cfg(unix)]

use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
use metrale_model_weights::fast_weights::FastSafetensorsLoader;
use metrale_model_weights::weights::{SafetensorsLoader, WeightLoader};
use std::io::Write;

/// Build a minimal `model.safetensors` with two BF16 tensors and one U8 tensor.
/// Layout written by hand so the test doesn't depend on the safetensors crate
/// for encoding (decoding is still needed, used by the baseline loader).
fn write_test_safetensors(dir: &std::path::Path) -> std::path::PathBuf {
    // Tensor A: BF16, shape [4, 8] = 64 bytes.
    // Tensor B: BF16, shape [2, 2] = 8 bytes.
    // Tensor C: U8,   shape [16]   = 16 bytes.
    let a_bytes: Vec<u8> = (0..64).map(|i| i as u8).collect();
    let b_bytes: Vec<u8> = (0..8).map(|i| (128 + i) as u8).collect();
    let c_bytes: Vec<u8> = (0..16).map(|i| (200 + i) as u8).collect();

    let header = serde_json::json!({
        "a": { "dtype": "BF16", "shape": [4, 8], "data_offsets": [0, 64] },
        "b": { "dtype": "BF16", "shape": [2, 2], "data_offsets": [64, 72] },
        "c": { "dtype": "U8",   "shape": [16],   "data_offsets": [72, 88] },
    });
    let header_bytes = serde_json::to_vec(&header).unwrap();

    let path = dir.join("model.safetensors");
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(&(header_bytes.len() as u64).to_le_bytes())
        .unwrap();
    f.write_all(&header_bytes).unwrap();
    f.write_all(&a_bytes).unwrap();
    f.write_all(&b_bytes).unwrap();
    f.write_all(&c_bytes).unwrap();
    f.sync_all().unwrap();
    path
}

#[test]
fn fast_and_mmap_loaders_agree() {
    let tmp = tempdir_like();
    write_test_safetensors(&tmp);

    let gpu_base = MockGpuBackend::new();
    let base = SafetensorsLoader::new()
        .load(&tmp, &gpu_base, 0)
        .expect("baseline load");
    assert_eq!(base.len(), 3);

    let gpu_fast = MockGpuBackend::new();
    let mut fast = FastSafetensorsLoader::new();
    // Force the buffered-read path: tmpfs rejects O_DIRECT on most kernels,
    // but we disable it explicitly so the test is deterministic.
    fast.try_direct_io = false;
    let new = fast.load(&tmp, &gpu_fast, 0).expect("fast load");
    assert_eq!(new.len(), 3);

    for name in ["a", "b", "c"] {
        let wb = base.get(name).unwrap();
        let wn = new.get(name).unwrap();
        assert_eq!(wb.shape, wn.shape, "shape mismatch for {name}");
        assert_eq!(wb.dtype, wn.dtype, "dtype mismatch for {name}");
        let bb = gpu_base.read_alloc(wb.ptr).unwrap();
        let bn = gpu_fast.read_alloc(wn.ptr).unwrap();
        assert_eq!(bb, bn, "byte mismatch for {name}");
    }

    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn fast_loader_with_direct_io_if_supported() {
    // Best-effort O_DIRECT test — silently succeeds (by falling back to
    // buffered) if the filesystem rejects O_DIRECT.
    let tmp = tempdir_like();
    write_test_safetensors(&tmp);

    let gpu_base = MockGpuBackend::new();
    let base = SafetensorsLoader::new().load(&tmp, &gpu_base, 0).unwrap();

    let gpu_fast = MockGpuBackend::new();
    let fast = FastSafetensorsLoader::new(); // try_direct_io = true by default
    let new = fast
        .load(&tmp, &gpu_fast, 0)
        .expect("fast load with O_DIRECT attempted");
    assert_eq!(new.len(), 3);

    for name in ["a", "b", "c"] {
        let bb = gpu_base.read_alloc(base.get(name).unwrap().ptr).unwrap();
        let bn = gpu_fast.read_alloc(new.get(name).unwrap().ptr).unwrap();
        assert_eq!(bb, bn, "byte mismatch for {name} (O_DIRECT path)");
    }

    std::fs::remove_dir_all(&tmp).ok();
}

/// Creates a unique temp directory without pulling in the tempfile crate.
fn tempdir_like() -> std::path::PathBuf {
    let pid = std::process::id();
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let p = std::env::temp_dir().join(format!("metrale-fwp-{pid}-{ns}"));
    std::fs::create_dir_all(&p).unwrap();
    p
}

/// 2026-10-17: Write one safetensors file holding `entries` back to back.
fn write_shard(path: &std::path::Path, entries: &[(&str, &str, Vec<usize>, Vec<u8>)]) {
    let mut header = serde_json::Map::new();
    let mut offset = 0usize;
    for (name, dtype, shape, bytes) in entries {
        header.insert(
            name.to_string(),
            serde_json::json!({
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [offset, offset + bytes.len()],
            }),
        );
        offset += bytes.len();
    }
    let header_bytes = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(&(header_bytes.len() as u64).to_le_bytes())
        .unwrap();
    f.write_all(&header_bytes).unwrap();
    for (_, _, _, bytes) in entries {
        f.write_all(bytes).unwrap();
    }
    f.sync_all().unwrap();
}

/// 2026-10-17: A checkpoint whose index routes tensors to a shard in a
/// subdirectory, and whose scale planes carry U16 and U32 components, loads
/// the same tensors with the same bytes through both loaders. The mmap loader
/// is the reference: it already joined index values onto the model
/// directory. Before the fix the fast loader routed nothing to the
/// subdirectory shard and refused the U16/U32 header entries.
#[test]
fn subdirectory_shard_with_u16_u32_components_matches_baseline() {
    let tmp = tempdir_like();
    let u16s: Vec<u8> = [0x0102u16, 0xFFFF, 0, 0x8001]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let u32s: Vec<u8> = [7u32, 0xDEAD_BEEF]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    write_shard(
        &tmp.join("model-00001-of-00001.safetensors"),
        &[("flat", "U8", vec![8], (0..8).collect())],
    );
    write_shard(
        &tmp.join("tensors/side.safetensors"),
        &[
            ("plane.offsets", "U16", vec![4], u16s.clone()),
            ("plane.page_bases", "U32", vec![2], u32s.clone()),
        ],
    );
    let index = serde_json::json!({
        "metadata": {},
        "weight_map": {
            "flat": "model-00001-of-00001.safetensors",
            "plane.offsets": "tensors/side.safetensors",
            "plane.page_bases": "tensors/side.safetensors",
        }
    });
    std::fs::write(tmp.join("model.safetensors.index.json"), index.to_string()).unwrap();

    let gpu_base = MockGpuBackend::new();
    let base = SafetensorsLoader::new()
        .load(&tmp, &gpu_base, 0)
        .expect("baseline load");
    let gpu_fast = MockGpuBackend::new();
    let mut fast = FastSafetensorsLoader::new();
    fast.try_direct_io = false;
    let new = fast.load(&tmp, &gpu_fast, 0).expect("fast load");
    assert_eq!(base.len(), 3);
    assert_eq!(new.len(), 3);

    use metrale_model_weights::weights::WeightDtype;
    for (name, dtype, shape, bytes) in [
        (
            "flat",
            WeightDtype::UInt8,
            vec![8],
            (0..8).collect::<Vec<u8>>(),
        ),
        ("plane.offsets", WeightDtype::UInt16, vec![4], u16s),
        ("plane.page_bases", WeightDtype::UInt32, vec![2], u32s),
    ] {
        for (label, store, gpu) in [("baseline", &base, &gpu_base), ("fast", &new, &gpu_fast)] {
            let w = store.get(name).unwrap();
            assert_eq!(w.dtype, dtype, "{label} dtype for {name}");
            assert_eq!(w.shape, shape, "{label} shape for {name}");
            assert_eq!(w.byte_size(), bytes.len(), "{label} byte size for {name}");
            assert_eq!(
                gpu.read_alloc(w.ptr).unwrap(),
                bytes,
                "{label} bytes for {name}"
            );
        }
    }
    std::fs::remove_dir_all(&tmp).ok();
}
