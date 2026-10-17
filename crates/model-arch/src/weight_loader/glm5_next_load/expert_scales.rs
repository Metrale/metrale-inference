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

/// 2026-10-17: For the whole-expert bind: the block scales of the resident packed projection
/// `w` (`base` of `layer`): the store's own tensor when native, else a device copy of the
/// decoded csf plane.
pub(super) fn block_scale_ptr(
    gpu: &dyn GpuBackend,
    store: &WeightStore,
    layer: usize,
    base: &str,
    w: &WeightTensor,
) -> Result<DevicePtr> {
    if !has_csf_plane(store, layer, base) {
        return Ok(store
            .get(&qualify(layer, &format!("{base}.weight_scale")))?
            .ptr);
    }
    let [n, half_k] = w.shape[..] else {
        bail!(
            "{base}.weight is packed NVFP4 of shape {:?}, not 2-D",
            w.shape
        );
    };
    let decoded = decode_expert_block_scales(gpu, store, layer, base, n, half_k * 2)?;
    let p = gpu.alloc(decoded.len())?;
    gpu.copy_h2d(&decoded, p)?;
    Ok(p)
}

#[cfg(test)]
#[path = "expert_scales_tests.rs"]
mod tests;
