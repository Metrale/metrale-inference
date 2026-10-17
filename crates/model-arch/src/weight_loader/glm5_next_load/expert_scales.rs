// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: Routed-expert NVFP4 block scales stored as a `csf_32x2_v1` plane.
//!
//! A checkpoint that declares `weight_scale_encoding: "csf_32x2_v1"` ships each expert
//! projection's `weight_scale` (F8_E4M3 `[n, k/16]`) as five components named
//! `{base}.weight_scale.csf32_*` instead (`metrale_core::csf_blocks`). The defer hook keeps the
//! components off the device ([`is_csf_component`]); the expert binders decode them on the host
//! into the exact bytes a native `weight_scale` holds, so every kernel operand is unchanged.
//!
//! Owner: model-arch weight loader.
//! Invariants:
//! - A projection has either a native `weight_scale` or a complete csf plane, never both.
//! - Component dtypes and shapes are checked before decoding; the decoder checks the rest.

use super::expert_slab::{ExpertSlab, SLAB_ALIGN};
use super::expert_tp::{meta, read_all};
use super::*;
use metrale_core::csf_blocks::{
    CsfGeometry, CsfOffsetWidth, CsfPlaneRef, SUFFIX_BASES, SUFFIX_CODES, SUFFIX_EXCEPTIONS,
    SUFFIX_OFFSETS, SUFFIX_PAGE_BASES, decode_csf_32x2_v1_to_vec,
};

/// 2026-10-17: Whether a store tensor is one component of a csf-encoded scale plane.
pub(super) fn is_csf_component(name: &str) -> bool {
    name.contains(".weight_scale.csf32_")
}

/// 2026-10-17: Whether `{base}.weight_scale` of `layer` is stored as a csf plane (its bases
/// component is present, deferred or resident).
pub(super) fn has_csf_plane(store: &WeightStore, layer: usize, base: &str) -> bool {
    let name = qualify(layer, &format!("{base}.weight_scale{SUFFIX_BASES}"));
    store.deferred(&name).is_some() || store.get(&name).is_ok()
}

/// 2026-10-17: The row-major F8_E4M3 `[n, k/16]` block scales of the packed NVFP4 projection
/// `base` of `layer`, decoded from its csf plane.
pub(super) fn decode_expert_block_scales(
    gpu: &dyn GpuBackend,
    store: &WeightStore,
    layer: usize,
    base: &str,
    n: usize,
    k: usize,
) -> Result<Vec<u8>> {
    if !k.is_multiple_of(16) {
        bail!("{base}: K {k} is not a multiple of the NVFP4 group 16");
    }
    let native = qualify(layer, &format!("{base}.weight_scale"));
    if store.deferred(&native).is_some() || store.get(&native).is_ok() {
        bail!("{native} is present beside its csf plane; the checkpoint declares one of them");
    }
    let geometry = CsfGeometry::new(n, k / 16, CsfOffsetWidth::Nvfp4)
        .with_context(|| format!("{base}: csf geometry of [{n}, {}]", k / 16))?;
    let component = |suffix: &str, dtype: WeightDtype, elems: Option<usize>| -> Result<Vec<u8>> {
        let name = format!("{native}{suffix}");
        let (got, shape) =
            meta(store, &name).with_context(|| format!("{name}: the csf plane is incomplete"))?;
        let ok_shape = match (elems, &shape[..]) {
            (Some(want), &[len]) => len == want,
            (None, &[_]) => true,
            _ => false,
        };
        if got != dtype || !ok_shape {
            bail!("{name} is {got:?} {shape:?}; expected 1-D {dtype:?} of {elems:?} elements");
        }
        read_all(gpu, store, &name)
    };
    let bases = component(SUFFIX_BASES, WeightDtype::UInt8, Some(geometry.rows))?;
    let codes = component(SUFFIX_CODES, WeightDtype::UInt8, Some(geometry.codes_len()))?;
    let page_bases = component(SUFFIX_PAGE_BASES, WeightDtype::UInt32, Some(geometry.pages))?;
    let offsets = component(
        SUFFIX_OFFSETS,
        WeightDtype::UInt16,
        Some(geometry.offsets_entries()),
    )?;
    let exceptions = component(SUFFIX_EXCEPTIONS, WeightDtype::UInt16, None)?;
    decode_csf_32x2_v1_to_vec(&CsfPlaneRef {
        geometry,
        bases: &bases,
        codes: &codes,
        page_bases: &page_bases,
        offsets: &offsets,
        exceptions: &exceptions,
    })
    .with_context(|| format!("{native}: decoding the csf_32x2_v1 plane"))
}

/// 2026-10-17: A whole-bound projection's block scales: a device pointer it already has (the
/// store's native tensor, or the scales a quantising bind produced), or bytes decoded from a csf
/// plane that [`place_block_scales`] still has to upload.
pub(super) enum BlockScales {
    Resident(DevicePtr),
    Decoded(Vec<u8>),
}

/// 2026-10-17: For the whole-expert bind: the block scales of the resident packed projection
/// `w` (`base` of `layer`): the store's own tensor when native, else the decoded csf plane.
pub(super) fn block_scales(
    gpu: &dyn GpuBackend,
    store: &WeightStore,
    layer: usize,
    base: &str,
    w: &WeightTensor,
) -> Result<BlockScales> {
    if !has_csf_plane(store, layer, base) {
        let native = store.get(&qualify(layer, &format!("{base}.weight_scale")))?;
        return Ok(BlockScales::Resident(native.ptr));
    }
    let [n, half_k] = w.shape[..] else {
        bail!(
            "{base}.weight is packed NVFP4 of shape {:?}, not 2-D",
            w.shape
        );
    };
    let k = half_k
        .checked_mul(2)
        .with_context(|| format!("{base}.weight: packed width {half_k} overflows"))?;
    Ok(BlockScales::Decoded(decode_expert_block_scales(
        gpu, store, layer, base, n, k,
    )?))
}

/// 2026-10-17: The gate, up and down block-scale pointers of whole expert `id`. Decoded scales
/// go into `id`'s slot of the layer's `slab`, one allocation per layer that the store's
/// derived-weight ledger owns, never one per projection. An expert whose projections mix decoded
/// and resident scales is refused: the checkpoint encodes an expert's scales one way.
pub(super) fn place_block_scales(
    gpu: &dyn GpuBackend,
    store: &WeightStore,
    slab: &ExpertSlab,
    id: usize,
    scales: [BlockScales; 3],
) -> Result<[DevicePtr; 3]> {
    use BlockScales::{Decoded, Resident};
    match scales {
        [Resident(g), Resident(u), Resident(d)] => Ok([g, u, d]),
        [Decoded(g), Decoded(u), Decoded(d)] => slab.place(gpu, store, id, [&g, &u, &d]),
        _ => bail!(
            "expert {id}: some projections store csf scale planes and some native scales; an \
             expert's scales are encoded one way"
        ),
    }
}

/// 2026-10-17: The loader's pre-flight charge under `config`: [`whole_layout_charge`] under the
/// whole (default) layout. The `tp` layout states none: every routed-expert tensor is deferred
/// there, and its slab was never charged.
pub(super) fn defer_charge(
    config: &ModelConfig,
) -> Option<metrale_model_weights::weights::DeferCharge> {
    if config.moe_expert_layout == metrale_config::MoeExpertLayout::Tp {
        return None;
    }
    Some(std::sync::Arc::new(
        |name: &str, _: WeightDtype, shape: &[usize]| whole_layout_charge(name, shape),
    ))
}

/// 2026-10-17: The device bytes the whole-layout bind allocates for the deferred tensor `name`
/// of `shape`, for the load's pre-flight: the decoded scales of a csf plane, charged on its codes
/// component (two scale bytes per code byte, rounded up to the slab alignment as
/// [`ExpertSlab`] places them). Every other component, and every other deferred tensor, is read
/// on the host and charged nothing. An overflowing shape charges `usize::MAX`, which the
/// pre-flight refuses.
pub(super) fn whole_layout_charge(name: &str, shape: &[usize]) -> usize {
    if !name.ends_with(&format!(".weight_scale{SUFFIX_CODES}")) {
        return 0;
    }
    shape
        .iter()
        .try_fold(2usize, |acc, &d| acc.checked_mul(d))
        .and_then(|b| b.checked_next_multiple_of(SLAB_ALIGN))
        .unwrap_or(usize::MAX)
}

#[cfg(test)]
#[path = "expert_scales_tests.rs"]
mod tests;
