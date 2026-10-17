// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: What `prune_after_load` frees once every binder has run. The body moved verbatim
//! out of `loader.rs` (which was past the file cap) when it gained the MTP layer's widened E4M3
//! originals.
//!
//! Owner: model-arch weight loader.
//! Invariants:
//! - Only tensors whose bytes a binder re-uploaded or quantised are freed; a tensor bound
//!   zero-copy (a U8 routed expert, the MTP layer's `eh_proj`) is never matched.

use super::*;

/// 2026-10-17: Whether store tensor `name` of the MTP layer `mtp_layer` is one half of an E4M3
/// pair `fp8_widen::widen_layer_fp8` replaced on the host: an F8_E4M3 `.weight` with a
/// `.weight_scale` sibling, or that sibling. The drafter's binders upload the widening, so the
/// originals are dead once it is bound; the text layers' are freed with every re-uploaded
/// tensor (`is_reuploaded`), which keeps the MTP layer. `dtype` answers a resident tensor's
/// dtype, `None` when absent.
pub(super) fn is_widened_mtp_original(
    name: &str,
    mtp_layer: usize,
    dtype: impl Fn(&str) -> Option<WeightDtype>,
) -> bool {
    let prefix = format!("model.language_model.layers.{mtp_layer}.");
    let Some(rel) = name.strip_prefix(&prefix) else {
        return false;
    };
    if rel.starts_with("mlp.experts.") {
        return false;
    }
    let weight = name.strip_suffix("_scale").unwrap_or(name);
    weight.ends_with(".weight")
        && dtype(weight) == Some(WeightDtype::FP8E4M3)
        && dtype(&format!("{weight}_scale")).is_some()
}

/// 2026-09-25: Free the store tensors `is_reuploaded` matches, then those
/// `is_quantized_expert_weight` matches. 2026-10-17: Then the MTP layer's widened E4M3
/// originals ([`is_widened_mtp_original`]).
pub(super) fn prune_after_load(
    store: &mut WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<()> {
    let n = config.num_hidden_layers;
    let (count, bytes) = store.free_matching(gpu, |name| is_reuploaded(name, n))?;
    tracing::info!(
        "glm5_next: released {count} store tensors ({:.2} GB) already re-uploaded by the \
         binders; routed experts and the MTP block kept",
        bytes as f64 / 1e9,
    );
    let quantized: std::collections::BTreeSet<String> = store
        .names()
        .filter(|n| {
            store
                .get(n)
                .is_ok_and(|t| is_quantized_expert_weight(n, t.dtype))
        })
        .map(str::to_string)
        .collect();
    if !quantized.is_empty() {
        let (qcount, qbytes) = store.free_matching(gpu, |name| quantized.contains(name))?;
        tracing::info!(
            "glm5_next: released {qcount} full-width BF16 routed-expert tensors \
             ({:.2} GB) quantised to NVFP4 at bind time",
            qbytes as f64 / 1e9,
        );
    }
    let resident = |s: &WeightStore, name: &str| s.get(name).ok().map(|t| t.dtype);
    let widened: std::collections::BTreeSet<String> = store
        .names()
        .filter(|name| is_widened_mtp_original(name, n, |x| resident(store, x)))
        .map(str::to_string)
        .collect();
    if !widened.is_empty() {
        let (wcount, wbytes) = store.free_matching(gpu, |name| widened.contains(name))?;
        tracing::info!(
            "glm5_next: released {wcount} E4M3 MTP-layer tensors ({:.2} GB) widened to BF16 at \
             bind time",
            wbytes as f64 / 1e9,
        );
    }
    Ok(())
}
