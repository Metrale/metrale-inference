// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Bind a routed expert under the layout the MLP config chose: whole
//! (`bind_expert`) or, under `--moe-expert-layout tp`, this rank's slice of it
//! (`glm5next_mlp::expert_tp`).
//!
//! Under the `tp` layout the defer hook keeps every routed-expert tensor off the device
//! ([`is_routed_expert_tensor`]), so a rank uploads only its slices: the rows of gate and up are
//! read from the shard as one contiguous range, down whole (its columns are strided). A BF16
//! expert (the MTP layer of the NVFP4 export) is quantized whole, as `bind_expert` would, and its
//! packed bytes are sliced, so a slice holds exactly the bytes the whole expert would.
//!
//! Owner: model-arch weight loader.
//! Invariants:
//! - `ExpertShard::Whole` binds through `bind_expert`, unchanged.
//! - A sliced projection keeps the expert's own `weight_scale_2` and `input_scale`.

use super::*;
use crate::glm5next_mlp::expert_tp::{
    ExpertShard, ExpertSlice, pad_expert_rows, slice_expert_cols, slice_expert_rows,
};
use metrale_config::TpSlice;

/// 2026-10-09: Whether a store tensor belongs to a routed expert of any layer (the text layers
/// and the MTP layer): the family the defer hook keeps off the device under the `tp` layout.
pub(super) fn is_routed_expert_tensor(name: &str) -> bool {
    name.strip_prefix("model.language_model.layers.")
        .and_then(|rest| rest.split_once('.'))
        .is_some_and(|(idx, rel)| idx.parse::<usize>().is_ok() && rel.starts_with("mlp.experts."))
}

/// 2026-10-09: Routed expert `id` of `layer` as this rank holds it under `shard`.
pub(super) fn bind_routed_expert(
    gpu: &dyn GpuBackend,
    store: &WeightStore,
    layer: usize,
    id: usize,
    shard: ExpertShard,
) -> Result<Glm5NextExpertWeights> {
    match shard {
        ExpertShard::Whole => bind_expert(gpu, store, layer, id),
        ExpertShard::Sliced(s) => Ok(Glm5NextExpertWeights {
            gate_proj: bind_sliced_proj(gpu, store, layer, id, "gate_proj", &s)?,
            up_proj: bind_sliced_proj(gpu, store, layer, id, "up_proj", &s)?,
            down_proj: bind_sliced_proj(gpu, store, layer, id, "down_proj", &s)?,
        }),
    }
}

/// 2026-10-09: A tensor's on-disk or store dtype and shape, deferred or resident.
fn meta(store: &WeightStore, name: &str) -> Result<(WeightDtype, Vec<usize>)> {
    if let Some(d) = store.deferred(name) {
        return Ok((d.dtype, d.shape.clone()));
    }
    let t = store.get(name)?;
    Ok((t.dtype, t.shape.clone()))
}

/// 2026-10-09: Bytes `[start, start + len)` of a tensor: a ranged shard read when deferred, else
/// read back from the device and cut on the host.
fn read_range(
    gpu: &dyn GpuBackend,
    store: &WeightStore,
    name: &str,
    start: usize,
    len: usize,
) -> Result<Vec<u8>> {
    if let Some(d) = store.deferred(name) {
        return d
            .read_host_range(start, len)
            .with_context(|| format!("{name}: reading the deferred expert from its shard"));
    }
    let all = host_bytes(gpu, store.get(name)?)?;
    all.get(start..start + len)
        .map(<[u8]>::to_vec)
        .with_context(|| format!("{name}: bytes {start}..+{len} past its {} B", all.len()))
}

/// 2026-10-09: A whole tensor's bytes, deferred or resident.
fn read_all(gpu: &dyn GpuBackend, store: &WeightStore, name: &str) -> Result<Vec<u8>> {
    if let Some(d) = store.deferred(name) {
        return d
            .read_host_bytes()
            .with_context(|| format!("{name}: reading the deferred expert from its shard"));
    }
    host_bytes(gpu, store.get(name)?)
}

/// 2026-10-09: The scalar F32 `weight_scale_2` of `base`, deferred or resident.
fn scale_2(gpu: &dyn GpuBackend, store: &WeightStore, layer: usize, base: &str) -> Result<f32> {
    let name = qualify(layer, &format!("{base}.weight_scale_2"));
    let (dtype, _) = meta(store, &name)?;
    let b = read_all(gpu, store, &name)?;
    match (dtype, &b[..]) {
        (WeightDtype::FP32, &[a, b2, c, d]) => Ok(f32::from_le_bytes([a, b2, c, d])),
        _ => bail!("{name} is {dtype:?} of {} bytes; expected one F32", b.len()),
    }
}

/// 2026-10-09: One sliced projection: gate/up (`[full, hidden]`) keep the slice's rows, down
/// (`[hidden, full]`) its columns, both zero-padded to `s.len`; uploaded and adopted by the
/// store's derived-weight ledger.
fn bind_sliced_proj(
    gpu: &dyn GpuBackend,
    store: &WeightStore,
    layer: usize,
    id: usize,
    p: &str,
    s: &ExpertSlice,
) -> Result<Nvfp4Proj> {
    let by_rows = p != "down_proj";
    let base = format!("mlp.experts.{id}.{p}");
    let wname = qualify(layer, &format!("{base}.weight"));
    let (dtype, shape) = meta(store, &wname)?;
    let (packed, scales, s2, input_scale) = match (dtype, &shape[..]) {
        (WeightDtype::UInt8, &[n, half_k]) => {
            let k = half_k * 2;
            let (want_n, want_k) = if by_rows { (s.full, k) } else { (n, s.full) };
            if n != want_n || k != want_k || !k.is_multiple_of(16) {
                bail!(
                    "{base}.weight: packed shape [{n}, {half_k}] does not fit a {} expert of \
                     width {} (tp layout)",
                    if by_rows {
                        "row-sliced"
                    } else {
                        "column-sliced"
                    },
                    s.full
                );
            }
            let sname = qualify(layer, &format!("{base}.weight_scale"));
            let (sd, sshape) = meta(store, &sname)?;
            if sd != WeightDtype::FP8E4M3 || sshape != [n, k / 16] {
                bail!(
                    "{base}.weight_scale is {sd:?} {sshape:?}, expected F8_E4M3 [{n}, {}]",
                    k / 16
                );
            }
            let (packed, scales) = if by_rows {
                let rows = s.real_cols();
                let cut = |name: &str, row_bytes: usize, r: TpSlice| {
                    read_range(gpu, store, name, r.start * row_bytes, r.len * row_bytes)
                };
                pad_expert_rows(cut(&wname, k / 2, rows)?, cut(&sname, k / 16, rows)?, k, s)?
            } else {
                slice_expert_cols(
                    &read_all(gpu, store, &wname)?,
                    &read_all(gpu, store, &sname)?,
                    n,
                    s,
                )?
            };
            (
                packed,
                scales,
                scale_2(gpu, store, layer, &base)?,
                input_scale(gpu, store, layer, &base)?,
            )
        }
        (WeightDtype::BF16, &[rows, cols]) => {
            let values: Vec<f32> = read_all(gpu, store, &wname)?
                .chunks_exact(2)
                .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect();
            let blob = nvfp4_quant::quantize_to_nvfp4(&base, &values, rows, cols)?;
            let (packed, scales) = if by_rows {
                slice_expert_rows(&blob.packed, &blob.scales, cols, s)?
            } else {
                slice_expert_cols(&blob.packed, &blob.scales, rows, s)?
            };
            // 2026-10-09: Quantized here from 16-bit weights: no declared activation scale.
            (packed, scales, blob.scale_2, None)
        }
        _ => bail!(
            "{base}.weight is {dtype:?} {shape:?}; the tp expert layout binds 2-D packed U8 \
             NVFP4 or BF16"
        ),
    };
    Ok(Nvfp4Proj {
        packed: expert_quant::upload_bytes(gpu, store, &packed)?,
        scale: expert_quant::upload_bytes(gpu, store, &scales)?,
        scale_2: s2,
        input_scale,
    })
}

#[cfg(test)]
#[path = "expert_tp_tests.rs"]
mod tests;
