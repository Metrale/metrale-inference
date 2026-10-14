// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Debug dump of named full-attention prefill tensors to files.
//!
//! `METRALE_OP_DUMP=<dir>` turns it on; unset or empty is a no-op.
//! `METRALE_OP_DUMP_LAYERS` is a comma list of `attn_layer_idx` values (the
//! layer's index among the full-attention layers, not its absolute index) and
//! `METRALE_OP_DUMP_OPS` a comma list of op names; an unset or empty list
//! selects all.
//!
//! Each dump writes `<dir>/metrale_op_L{attn_layer_idx}_{op}.bin` as
//! headerless little-endian f32 (`dump_bf16` widens BF16 on the host). Every
//! call site (the `prefill/` files and `trait_impl/prefill_inner.rs`) passes
//! the last token's row, and each call overwrites the file, so under chunked
//! prefill the last chunk's row remains.
//!
//! 2026-10-14: `tail_rows` taps the mHC highway and the block input/output on every model
//! layer, GDN and attention (`HcWeights::model_layer` labels them), for the last
//! `min(num_tokens, TAIL_ROWS)` rows of the chunk. Under chunked prefill the last chunk
//! remains, so a cold prompt and its warm replay leave the same tail rows to diff.
//! `METRALE_OP_DUMP_GATE=<path>` dumps only while `<path>` exists, so a running server can
//! serve the same request with and without the taps' stream synchronisation.
//!
//! Owner: model-layers (attention).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

fn parse_csv(env: &str) -> Vec<String> {
    std::env::var(env)
        .ok()
        .map(|s| {
            s.split(',')
                .filter(|p| !p.trim().is_empty())
                .map(|p| p.trim().to_string())
                .collect()
        })
        .unwrap_or_default()
}

fn parse_csv_usize(env: &str) -> Vec<usize> {
    std::env::var(env)
        .ok()
        .map(|s| s.split(',').filter_map(|p| p.trim().parse().ok()).collect())
        .unwrap_or_default()
}

/// 2026-09-25: `Some(dir)` when `METRALE_OP_DUMP` is non-empty and both lists
/// admit `layer_idx` and `op`; `None` otherwise.
fn op_dump_dir(layer_idx: usize, op: &str) -> Option<String> {
    let dir = std::env::var("METRALE_OP_DUMP").ok()?;
    if dir.is_empty() {
        return None;
    }
    let layers = parse_csv_usize("METRALE_OP_DUMP_LAYERS");
    if !layers.is_empty() && !layers.contains(&layer_idx) {
        return None;
    }
    let ops = parse_csv("METRALE_OP_DUMP_OPS");
    if !ops.is_empty() && !ops.iter().any(|o| o == op) {
        return None;
    }
    if let Ok(gate) = std::env::var("METRALE_OP_DUMP_GATE")
        && !std::path::Path::new(&gate).exists()
    {
        return None;
    }
    Some(dir)
}

/// 2026-09-25: Synchronises `stream`, copies `n_elements` BF16 values from
/// `ptr + byte_offset`, widens them to little-endian f32 and writes
/// `<METRALE_OP_DUMP>/metrale_op_L{layer_idx}_{op}.bin`. Does nothing when
/// `op_dump_dir` returns `None`.
pub(crate) fn dump_bf16(
    gpu: &dyn GpuBackend,
    ptr: DevicePtr,
    byte_offset: usize,
    n_elements: usize,
    layer_idx: usize,
    op: &str,
    stream: u64,
) -> Result<()> {
    let Some(dir) = op_dump_dir(layer_idx, op) else {
        return Ok(());
    };
    gpu.synchronize(stream)?;
    let mut buf = vec![0u16; n_elements];
    // 2026-09-25: SAFETY: `buf` is `vec![0u16; n_elements]`, so the
    // `n_elements * 2` bytes are exactly its initialised buffer. `bytes` is the
    // only reference derived from `buf` while it is live, and it is unused after
    // the `copy_d2h` below, before `buf.iter()` runs.
    let bytes =
        unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, n_elements * 2) };
    gpu.copy_d2h(ptr.offset(byte_offset), bytes)?;
    let vals: Vec<f32> = buf
        .iter()
        .map(|&b| f32::from_bits((b as u32) << 16))
        .collect();
    let bytes_f32: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    let path = std::path::Path::new(&dir).join(format!("metrale_op_L{layer_idx}_{op}.bin"));
    std::fs::create_dir_all(&dir).ok();
    std::fs::write(&path, &bytes_f32)?;
    tracing::info!(
        "METRALE_OP_DUMP: wrote {} ({n_elements} f32, bf16-source)",
        path.display()
    );
    Ok(())
}

/// 2026-10-14: Rows `tail_rows` keeps: a 20-row warm tail fits whole, a 4,096-row cold chunk
/// keeps its last 64.
const TAIL_ROWS: usize = 64;

/// 2026-10-14: Dump the last `min(num_tokens, TAIL_ROWS)` rows of a `[num_tokens, row]`
/// tensor, f32 when `is_f32`, BF16 otherwise; written as f32 either way.
#[allow(clippy::too_many_arguments)]
pub(crate) fn tail_rows(
    gpu: &dyn GpuBackend,
    ptr: DevicePtr,
    num_tokens: usize,
    row: usize,
    is_f32: bool,
    layer_idx: usize,
    op: &str,
    stream: u64,
) -> Result<()> {
    let rows = num_tokens.min(TAIL_ROWS);
    let elem = if is_f32 { 4 } else { 2 };
    let off = (num_tokens - rows) * row * elem;
    if is_f32 {
        dump_f32(gpu, ptr, off, rows * row, layer_idx, op, stream)
    } else {
        dump_bf16(gpu, ptr, off, rows * row, layer_idx, op, stream)
    }
}

/// 2026-09-25: `dump_bf16` for a tensor already stored as f32 on the device.
pub(crate) fn dump_f32(
    gpu: &dyn GpuBackend,
    ptr: DevicePtr,
    byte_offset: usize,
    n_elements: usize,
    layer_idx: usize,
    op: &str,
    stream: u64,
) -> Result<()> {
    let Some(dir) = op_dump_dir(layer_idx, op) else {
        return Ok(());
    };
    gpu.synchronize(stream)?;
    let mut buf = vec![0f32; n_elements];
    // 2026-09-25: SAFETY: `buf` is `vec![0f32; n_elements]`, so the
    // `n_elements * 4` bytes are exactly its initialised buffer. `bytes` is the
    // only reference derived from `buf` while it is live, and it is unused after
    // the `copy_d2h` below, before `buf.iter()` runs.
    let bytes =
        unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, n_elements * 4) };
    gpu.copy_d2h(ptr.offset(byte_offset), bytes)?;
    let bytes_f32: Vec<u8> = buf.iter().flat_map(|v| v.to_le_bytes()).collect();
    let path = std::path::Path::new(&dir).join(format!("metrale_op_L{layer_idx}_{op}.bin"));
    std::fs::create_dir_all(&dir).ok();
    std::fs::write(&path, &bytes_f32)?;
    tracing::info!(
        "METRALE_OP_DUMP: wrote {} ({n_elements} f32 native)",
        path.display()
    );
    Ok(())
}
