// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-13: The aux-blob section of a swap record: the per-layer host state the
//! layers report through `LayerAuxState` (the PLE and QSA carries), written after
//! the KV blocks and SSM states. Pure encode and decode over `Write` / `Read`; the
//! device copies stay in `collect_aux_states` / `apply_aux_states`.
//!
//! Owner: model-engine.
//! Invariants:
//! - `write_aux_blobs` and `read_aux_blobs` agree on the layout: a u32 blob count,
//!   then per blob a u32 layer index, a u64 byte length and the bytes, all
//!   little-endian.
//! - `read_aux_blobs` refuses a count above `num_layers`, a layer index at or above
//!   `num_layers`, a layer listed twice and a truncated stream, so a record from
//!   another model or a short file fails before any upload.

use std::io::{Read, Write};

use anyhow::{Context, Result, bail};

/// 2026-10-13: Append `blobs` (layer index, blob) to `writer` in the module's layout.
pub(super) fn write_aux_blobs(writer: &mut dyn Write, blobs: &[(u32, Vec<u8>)]) -> Result<()> {
    let count = u32::try_from(blobs.len()).context("aux record: blob count exceeds u32")?;
    writer.write_all(&count.to_le_bytes())?;
    for (layer, blob) in blobs {
        writer.write_all(&layer.to_le_bytes())?;
        writer.write_all(&(blob.len() as u64).to_le_bytes())?;
        writer.write_all(blob)?;
    }
    Ok(())
}

/// 2026-10-13: Read the section [`write_aux_blobs`] wrote, for a model of
/// `num_layers` layers. The blobs come back in the written order.
pub(super) fn read_aux_blobs(
    reader: &mut dyn Read,
    num_layers: usize,
) -> Result<Vec<(u32, Vec<u8>)>> {
    let count = read_u32(reader).context("aux record: blob count")? as usize;
    if count > num_layers {
        bail!("aux record: {count} blobs for a model of {num_layers} layers");
    }
    let mut seen = vec![false; num_layers];
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let layer = read_u32(reader).context("aux record: layer index")?;
        let Some(slot) = seen.get_mut(layer as usize) else {
            bail!("aux record: blob for layer {layer} but the model has {num_layers} layers");
        };
        if std::mem::replace(slot, true) {
            bail!("aux record: layer {layer} listed twice");
        }
        let len = usize::try_from(read_u64(reader).context("aux record: blob length")?)
            .context("aux record: blob length exceeds usize")?;
        let mut blob = vec![0u8; len];
        reader
            .read_exact(&mut blob)
            .with_context(|| format!("aux record: blob of layer {layer} ({len} bytes)"))?;
        out.push((layer, blob));
    }
    Ok(out)
}

fn read_u32(reader: &mut dyn Read) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    reader.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64(reader: &mut dyn Read) -> std::io::Result<u64> {
    let mut b = [0u8; 8];
    reader.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}
