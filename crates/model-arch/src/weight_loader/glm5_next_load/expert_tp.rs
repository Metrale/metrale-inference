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
//! 2026-10-14: A layer's sliced experts are uploaded into one [`ExpertSlab`], not one
//! allocation per buffer: GB10 kept 16.5 GiB per rank of 2 MiB chunks part-empty under the
//! per-buffer uploads (`expert_slab.rs`).
//!
//! Owner: model-arch weight loader.
//! Invariants:
//! - `ExpertShard::Whole` binds through `bind_expert`, unchanged.
//! - A sliced projection keeps the expert's own `weight_scale_2` and `input_scale`.
//! - 2026-10-17: A csf-encoded scale plane is refused sliced (its bytes have no pre-flight
//!   charge under this layout); the whole layout binds it.
//! - 2026-10-21: Under a packing layer ([`ScalePack`], `METRALE_GLM_MOE_SCALE_DICT`) a sliced
//!   projection's block scales are packed on the host before the slab takes them, into the same
//!   bytes of its slot (`scale_dict::pack`); a whole expert is never packed.
//! - 2026-10-24: Under `METRALE_GLM_EXPERT_FAST_LOAD` (default on) a sliced layer's experts are
//!   prepared on the host in parallel before the first upload (`expert_prepare`); the slicing is
//!   [`sliced_proj`] on either path, only the reads differ.

use super::expert_digest::ExpertDigest;
use super::expert_prepare::{Buffered, ExpertBytes, ExpertLoad, Prepared};
use super::expert_scales::has_csf_plane;
use super::expert_slab::{CSF_SCALE_SLAB_LABEL, ExpertSlab, TP_SLAB_LABEL};
use super::scale_pack::ScalePack;
use super::*;
use crate::glm5next_mlp::expert_tp::{
    ExpertShard, ExpertSlice, pad_expert_rows, slice_expert_cols, slice_expert_rows,
};
use crate::glm5next_mlp::scale_dict::ScaleDictPlan;
use metrale_config::TpSlice;

/// 2026-10-09: Whether a store tensor belongs to a routed expert of any layer (the text layers
/// and the MTP layer): the family the defer hook keeps off the device under the `tp` layout.
pub(super) fn is_routed_expert_tensor(name: &str) -> bool {
    name.strip_prefix("model.language_model.layers.")
        .and_then(|rest| rest.split_once('.'))
        .is_some_and(|(idx, rel)| idx.parse::<usize>().is_ok() && rel.starts_with("mlp.experts."))
}

/// 2026-10-14: The `build_moe` expert binder of MoE layer `layer` under `cfg`'s shard:
/// [`bind_routed_expert`] into one [`ExpertSlab`] over `cfg.local_expert_range()`, labelled by
/// [`slab_label`].
/// 2026-10-21: `pack`: the layer's scale packing; `None` keeps every scale plain (the MTP layer,
/// whose 16-bit experts run W4A16).
/// 2026-10-24: Under `load.fast` a sliced layer's experts are prepared in parallel waves as the
/// binder takes them ([`Prepared`]); under `load.digest` every sliced expert's uploaded
/// buffers are digested ([`ExpertDigest`]).
pub(super) fn routed_expert_binder<'a>(
    gpu: &'a dyn GpuBackend,
    store: &'a WeightStore,
    layer: usize,
    cfg: &Glm5NextMlpConfig,
    pack: Option<&'a ScalePack>,
    load: &ExpertLoad,
) -> Result<impl Fn(usize) -> Result<Glm5NextExpertWeights> + use<'a>> {
    let (shard, slab) = (
        cfg.expert_shard,
        ExpertSlab::new(cfg.local_expert_range(), slab_label(cfg.expert_shard)),
    );
    let prepared = match shard {
        ExpertShard::Sliced(s) if load.fast => {
            Some(Prepared::new(store, layer, cfg.local_expert_range(), &s))
        }
        _ => None,
    };
    let digest = load.digest.clone().map(|path| ExpertDigest { path, layer });
    Ok(move |id| {
        let ExpertShard::Sliced(s) = shard else {
            return bind_routed_expert(gpu, store, layer, id, shard, &slab, pack);
        };
        let host = match &prepared {
            Some(p) => p.take(id)?,
            None => host_expert(store, &Buffered { gpu, store }, layer, id, &s)?,
        };
        bind_sliced(gpu, store, id, &s, &slab, pack, host, digest.as_ref())
    })
}

/// 2026-10-17: The ledger label of a layer's slab under `shard`: what that layout places in it.
pub(super) fn slab_label(shard: ExpertShard) -> &'static str {
    match shard {
        ExpertShard::Whole => CSF_SCALE_SLAB_LABEL,
        ExpertShard::Sliced(_) => TP_SLAB_LABEL,
    }
}

/// 2026-10-09: Routed expert `id` of `layer` as this rank holds it under `shard`. 2026-10-14: A
/// sliced expert's six buffers go into `id`'s slot of `slab`. 2026-10-17: A whole one places only
/// block scales decoded from csf planes there ([`bind_expert`]). 2026-10-21: A sliced one's
/// scales packed first under `pack`; a whole one is refused under a packing layer.
pub(super) fn bind_routed_expert(
    gpu: &dyn GpuBackend,
    store: &WeightStore,
    layer: usize,
    id: usize,
    shard: ExpertShard,
    slab: &ExpertSlab,
    pack: Option<&ScalePack>,
) -> Result<Glm5NextExpertWeights> {
    let plain = ScalePack::new(ScaleDictPlan::Off);
    let pack = pack.unwrap_or(&plain);
    let s = match shard {
        ExpertShard::Whole if pack.on() => bail!(
            "glm5_next layer {layer}: the whole expert layout binds the checkpoint's scale \
             tensors in place and cannot pack them (scale_dict::plan keeps it plain)"
        ),
        ExpertShard::Whole => return bind_expert(gpu, store, layer, id, slab),
        ExpertShard::Sliced(s) => s,
    };
    let host = host_expert(store, &Buffered { gpu, store }, layer, id, &s)?;
    bind_sliced(gpu, store, id, &s, slab, Some(pack), host, None)
}

/// 2026-10-24: Expert `id` of `layer` under slice `s` on the host, gate, up and down, from `b`.
pub(super) fn host_expert<B: ExpertBytes + ?Sized>(
    store: &WeightStore,
    b: &B,
    layer: usize,
    id: usize,
    s: &ExpertSlice,
) -> Result<[HostProj; 3]> {
    Ok([
        sliced_proj(store, b, layer, id, "gate_proj", s)?,
        sliced_proj(store, b, layer, id, "up_proj", s)?,
        sliced_proj(store, b, layer, id, "down_proj", s)?,
    ])
}

/// 2026-10-24: Pack (under `pack`) and upload a sliced expert's host projections into `id`'s
/// slot of `slab`; with `digest`, record the uploaded device bytes.
#[allow(clippy::too_many_arguments)]
fn bind_sliced(
    gpu: &dyn GpuBackend,
    store: &WeightStore,
    id: usize,
    s: &ExpertSlice,
    slab: &ExpertSlab,
    pack: Option<&ScalePack>,
    [mut g, mut u, mut d]: [HostProj; 3],
    digest: Option<&ExpertDigest>,
) -> Result<Glm5NextExpertWeights> {
    let plain = ScalePack::new(ScaleDictPlan::Off);
    let pack = pack.unwrap_or(&plain);
    // 2026-10-21: gate/up keep `s.len` rows of hidden / 16 groups; down hidden rows of
    // `s.len / 16` groups.
    let gu_groups = g.1.len() / s.len.max(1);
    let layouts = [
        pack.apply(&mut g, gu_groups),
        pack.apply(&mut u, gu_groups),
        pack.apply(&mut d, s.len / 16),
    ];
    let blobs = [&g.0[..], &g.1, &u.0, &u.1, &d.0, &d.1];
    let [gp, gs, up, us, dp, ds] = slab.place(gpu, store, id, blobs)?;
    if let Some(dg) = digest {
        let scalars = [&g, &u, &d].map(|h| (h.2, h.3));
        dg.record(gpu, id, &blobs, &[gp, gs, up, us, dp, ds], &scalars)?;
    }
    let proj = |h: &HostProj, packed, scale, scale_layout| Nvfp4Proj {
        packed,
        scale,
        scale_2: h.2,
        input_scale: h.3,
        scale_layout,
    };
    Ok(Glm5NextExpertWeights {
        gate_proj: proj(&g, gp, gs, layouts[0]),
        up_proj: proj(&u, up, us, layouts[1]),
        down_proj: proj(&d, dp, ds, layouts[2]),
    })
}

/// 2026-10-09: A tensor's on-disk or store dtype and shape, deferred or resident.
pub(super) fn meta(store: &WeightStore, name: &str) -> Result<(WeightDtype, Vec<usize>)> {
    if let Some(d) = store.deferred(name) {
        return Ok((d.dtype, d.shape.clone()));
    }
    let t = store.get(name)?;
    Ok((t.dtype, t.shape.clone()))
}

/// 2026-10-09: Bytes `[start, start + len)` of a tensor: a ranged shard read when deferred, else
/// read back from the device and cut on the host.
pub(super) fn read_range(
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
pub(super) fn read_all(gpu: &dyn GpuBackend, store: &WeightStore, name: &str) -> Result<Vec<u8>> {
    if let Some(d) = store.deferred(name) {
        return d
            .read_host_bytes()
            .with_context(|| format!("{name}: reading the deferred expert from its shard"));
    }
    host_bytes(gpu, store.get(name)?)
}

/// 2026-10-09: The scalar F32 `weight_scale_2` of `base`, deferred or resident.
fn scale_2<B: ExpertBytes + ?Sized>(
    store: &WeightStore,
    src: &B,
    layer: usize,
    base: &str,
) -> Result<f32> {
    let name = qualify(layer, &format!("{base}.weight_scale_2"));
    let (dtype, _) = meta(store, &name)?;
    let b = src.all(&name)?;
    match (dtype, &b[..]) {
        (WeightDtype::FP32, &[a, b2, c, d]) => Ok(f32::from_le_bytes([a, b2, c, d])),
        _ => bail!("{name} is {dtype:?} of {} bytes; expected one F32", b.len()),
    }
}

/// 2026-10-14: A sliced projection on the host: codes, block scales, global scale, activation
/// scale.
pub(super) type HostProj = (Vec<u8>, Vec<u8>, f32, Option<f32>);

/// 2026-10-09: One sliced projection: gate/up (`[full, hidden]`) keep the slice's rows, down
/// (`[hidden, full]`) its columns, both zero-padded to `s.len`. 2026-10-14: Returned on the host
/// for the layer's slab, no longer uploaded here. 2026-10-24: Its bytes from `src`.
fn sliced_proj<B: ExpertBytes + ?Sized>(
    store: &WeightStore,
    src: &B,
    layer: usize,
    id: usize,
    p: &str,
    s: &ExpertSlice,
) -> Result<HostProj> {
    let by_rows = p != "down_proj";
    let base = format!("mlp.experts.{id}.{p}");
    let wname = qualify(layer, &format!("{base}.weight"));
    let (dtype, shape) = meta(store, &wname)?;
    Ok(match (dtype, &shape[..]) {
        (WeightDtype::UInt8, &[n, half_k]) => {
            let k = half_k
                .checked_mul(2)
                .with_context(|| format!("{base}.weight: packed width {half_k} overflows"))?;
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
            // 2026-10-17: csf-encoded scale planes are refused under the tp layout: its experts
            // are deferred without a pre-flight charge (`defer_charge`), so decoding them here
            // would place bytes the pre-flight never counted. The whole (ep) layout binds them.
            if has_csf_plane(store, layer, &base) {
                bail!(
                    "{base}: csf-encoded block scales are not supported under \
                     --moe-expert-layout tp; use the ep layout for this checkpoint"
                );
            }
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
                    src.range(name, r.start * row_bytes, r.len * row_bytes)
                };
                pad_expert_rows(cut(&wname, k / 2, rows)?, cut(&sname, k / 16, rows)?, k, s)?
            } else {
                let scales = src.all(&sname)?;
                slice_expert_cols(&src.all(&wname)?, &scales, n, s)?
            };
            (
                packed,
                scales,
                scale_2(store, src, layer, &base)?,
                src.input_scale(layer, &base)?,
            )
        }
        (WeightDtype::BF16, &[rows, cols]) => {
            let values: Vec<f32> = src
                .all(&wname)?
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
    })
}

#[cfg(test)]
#[path = "expert_tp_tests.rs"]
pub(super) mod tests;
