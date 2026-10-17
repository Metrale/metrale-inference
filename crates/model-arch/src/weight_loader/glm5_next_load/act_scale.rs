// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: The checkpoint's static NVFP4 activation scales (`*.input_scale`) of the text
//! layers: kept off the device by the defer hook and read on the host at bind time.
//!
//! ModelOpt stores, per quantized linear, `input_scale = amax(x) / (6 * 448)`: the per-tensor
//! global scale of the activations, calibrated once. At run time a row is quantized to E2M1 with
//! one E4M3 scale per 16 values, `s = e4m3(amax16 / 6 / input_scale)`, and the GEMM output is
//! multiplied by `input_scale * weight_scale_2`. One F32 scalar per projection; a routed layer
//! has 864 of them, so uploading each would cost one allocation granule apiece.
//!
//! Owner: model-arch weight loader.
//! Invariants:
//! - A scale is returned only if it is a finite, positive F32 scalar; anything else is an error.

use super::*;

/// 2026-10-08: Whether a tensor is a text layer's activation scale, which the defer hook keeps
/// off the device. Keyed on the store dtype: only an F32 scalar's spelling is deferred.
pub(super) fn is_activation_scale(name: &str, dtype: WeightDtype) -> bool {
    dtype == WeightDtype::FP32
        && name.starts_with("model.language_model.layers.")
        && name.ends_with(".input_scale")
}

/// 2026-10-08: The static activation scale of projection `base` (layer-relative, without the
/// `.input_scale` leaf) of `layer`: read from its shard when deferred, read back when resident,
/// `None` when the checkpoint has none (a projection quantized at load, or a W4A16 export).
pub(super) fn input_scale(
    gpu: &dyn GpuBackend,
    store: &WeightStore,
    layer: usize,
    base: &str,
) -> Result<Option<f32>> {
    let name = qualify(layer, &format!("{base}.input_scale"));
    let values = if let Some(d) = store.deferred(&name) {
        if d.dtype != WeightDtype::FP32 {
            bail!(
                "{name} was deferred as {:?}; an activation scale is F32",
                d.dtype
            );
        }
        d.read_host_bytes()
            .with_context(|| format!("{name}: reading the activation scale from its shard"))?
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect::<Vec<f32>>()
    } else if store.contains(&name) {
        host_f32(gpu, store.get(&name)?, &name)?
    } else {
        return Ok(None);
    };
    let [s] = values[..] else {
        bail!(
            "{name} has {} elements; an activation scale is one scalar",
            values.len()
        );
    };
    if !(s.is_finite() && s > 0.0) {
        bail!("{name} is {s}; an activation scale must be finite and positive");
    }
    Ok(Some(s))
}
