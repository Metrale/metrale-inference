// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Strict shared diagnostic tensor reader and owned GPU buffers.
use anyhow::{Context, Result, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    io::{Read, Seek, SeekFrom},
    path::Path,
};
pub fn sha(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
pub struct Owned<'a> {
    gpu: &'a dyn GpuBackend,
    pub ptr: DevicePtr,
}
impl<'a> Owned<'a> {
    pub fn upload(gpu: &'a dyn GpuBackend, bytes: &[u8]) -> Result<Self> {
        let value = Self {
            gpu,
            ptr: gpu.alloc(bytes.len())?,
        };
        gpu.copy_h2d(bytes, value.ptr)?;
        Ok(value)
    }
}
impl Drop for Owned<'_> {
    fn drop(&mut self) {
        let _ = self.gpu.free(self.ptr);
    }
}
#[derive(Deserialize)]
pub struct Module {
    pub name: String,
    pub ptx: String,
    pub ptx_sha256: String,
}
#[derive(Deserialize)]
pub struct Modules {
    pub modules: Vec<Module>,
}
#[derive(Deserialize)]
struct TensorHeader {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}
#[allow(dead_code)] // BF16 and FP32 diagnostics share this strict reader.
pub fn read_tensor(
    component: &Path,
    index: &BTreeMap<String, String>,
    name: &str,
    expected_shape: &[usize],
) -> Result<(Vec<u8>, Vec<usize>)> {
    read_typed_tensor(component, index, name, expected_shape, "BF16", 2)
}
#[allow(dead_code)] // Used by the separate FP32 decoder diagnostic.
pub fn read_tensor_f32(
    component: &Path,
    index: &BTreeMap<String, String>,
    name: &str,
    expected_shape: &[usize],
) -> Result<(Vec<u8>, Vec<usize>)> {
    read_typed_tensor(component, index, name, expected_shape, "F32", 4)
}
fn read_typed_tensor(
    component: &Path,
    index: &BTreeMap<String, String>,
    name: &str,
    expected_shape: &[usize],
    dtype: &str,
    element_bytes: usize,
) -> Result<(Vec<u8>, Vec<usize>)> {
    let shard = index.get(name).context("missing tensor index")?;
    ensure!(
        Path::new(shard).components().count() == 1 && !Path::new(shard).is_absolute(),
        "unsafe shard path"
    );
    let mut f = std::fs::File::open(component.join(shard))?;
    let length = f.metadata()?.len();
    let mut prefix = [0u8; 8];
    f.read_exact(&mut prefix)?;
    let header_len = u64::from_le_bytes(prefix);
    ensure!(
        header_len <= 16 * 1024 * 1024 && header_len + 8 <= length,
        "invalid safetensors header length"
    );
    let mut bytes = vec![0u8; header_len as usize];
    f.read_exact(&mut bytes)?;
    let header: serde_json::Value = serde_json::from_slice(&bytes)?;
    let tensor: TensorHeader =
        serde_json::from_value(header.get(name).context("missing tensor header")?.clone())?;
    ensure!(
        tensor.dtype == dtype && tensor.shape == expected_shape,
        "tensor precision/shape differs"
    );
    let count = tensor
        .shape
        .iter()
        .try_fold(element_bytes, |n, d| n.checked_mul(*d))
        .context("tensor size overflow")?;
    let [start, end] = tensor.data_offsets;
    ensure!(
        end >= start && end - start == count as u64 && end <= length - header_len - 8,
        "invalid tensor offsets"
    );
    f.seek(SeekFrom::Start(8 + header_len + start))?;
    let mut payload = vec![0u8; count];
    f.read_exact(&mut payload)?;
    Ok((payload, tensor.shape))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn checkpoint_reader_refuses_dtype_shape_and_payload_corruption() {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory =
            std::env::temp_dir().join(format!("qwen-reader-{}-{stamp}", std::process::id()));
        std::fs::create_dir(&directory).unwrap();
        let index = BTreeMap::from([("x".to_string(), "weights.safetensors".to_string())]);
        for (dtype, shape, offsets, valid) in [
            ("BF16", vec![2], [0, 4], true),
            ("F32", vec![2], [0, 4], false),
            ("BF16", vec![1, 2], [0, 4], false),
            ("BF16", vec![2], [0, 8], false),
            ("BF16", vec![2], [8, 4], false),
        ] {
            let header = serde_json::to_vec(
                &serde_json::json!({"x":{"dtype":dtype,"shape":shape,"data_offsets":offsets}}),
            )
            .unwrap();
            let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
            bytes.extend(header);
            bytes.extend([1, 2, 3, 4]);
            std::fs::write(directory.join("weights.safetensors"), bytes).unwrap();
            let result = read_tensor(&directory, &index, "x", &[2]);
            assert_eq!(result.is_ok(), valid);
            if valid {
                assert_eq!(result.unwrap().0, [1, 2, 3, 4]);
            }
        }
        let bad = BTreeMap::from([("x".to_string(), "../weights.safetensors".to_string())]);
        assert!(read_tensor(&directory, &bad, "x", &[2]).is_err());
        std::fs::remove_dir_all(directory).unwrap();
    }
}
