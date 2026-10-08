// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-07: Laguna MoE layers whose routed experts are compressed-tensors `pack-quantized`
//! INT4/INT8 (Laguna-XS-2.1-INT4): load them as stored into a [`PackedIntMoeLayer`].
//!
//! Owner: model-arch weight loader (Laguna).
//! Invariants:
//! - A layer takes this path only when its expert 0 `gate_proj.weight_packed` is an I32
//!   tensor; NVFP4 checkpoints store U8 there and never reach it.
//! - Every tensor is checked before the layer is built: expert words I32 and scales BF16 in
//!   the layout the config's declared scheme fixes, router and shared expert BF16 of their
//!   config shapes, the correction bias F32 `[num_experts]`. Nothing is converted.

use anyhow::{Context, Result, ensure};
use metrale_config::ModelConfig;
use metrale_config::precision_plan::packed_int::{PackedIntScheme, routed_expert_scheme};
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_weights::weights::{WeightDtype, WeightStore};

use metrale_model_layers::layers::FfnComponent;
use metrale_model_layers::layers::packed_int_moe::{
    PackedIntExpert, PackedIntMoeLayer, PackedIntMoeWeights, PackedIntTensor,
};
use metrale_model_layers::quant_format::packed_int::check_tensor_pair;
use metrale_model_layers::weight_map::DenseWeight;

/// 2026-10-07: The packed-int scheme of the MoE layer under `mlp` (`model.layers.<i>.mlp`), or
/// `None` when its experts are not stored as I32 words. I32 words without a declared
/// packed-int scheme are refused.
pub(super) fn packed_int_scheme(
    store: &WeightStore,
    config: &ModelConfig,
    mlp: &str,
) -> Result<Option<PackedIntScheme>> {
    let probe = format!("{mlp}.experts.0.gate_proj.weight_packed");
    if !store
        .get(&probe)
        .is_ok_and(|t| t.dtype == WeightDtype::Int32)
    {
        return Ok(None);
    }
    let plan = &config
        .quantization_config
        .as_ref()
        .with_context(|| format!("{probe} is I32 but config.json declares no quantization"))?
        .precision;
    let scheme = routed_expert_scheme(plan, mlp, config.num_experts)?
        .with_context(|| format!("{probe} is I32 but {mlp} declares no packed-int experts"))?;
    Ok(Some(scheme))
}

/// 2026-10-07: A tensor of `dtype` and `shape`, as a device pointer.
fn checked(
    store: &WeightStore,
    name: &str,
    dtype: WeightDtype,
    shape: &[usize],
) -> Result<DenseWeight> {
    let t = store.get(name)?;
    ensure!(
        t.dtype == dtype && t.shape == shape,
        "{name}: {:?} {:?}, expected {dtype:?} {shape:?}",
        t.dtype,
        t.shape
    );
    Ok(DenseWeight { weight: t.ptr })
}

/// 2026-10-07: One projection's `weight_packed` / `weight_scale` for an `[n, k]` weight.
fn packed(
    store: &WeightStore,
    scheme: PackedIntScheme,
    prefix: &str,
    n: usize,
    k: usize,
) -> Result<PackedIntTensor> {
    let words = store.get(&format!("{prefix}.weight_packed"))?;
    let scales = store.get(&format!("{prefix}.weight_scale"))?;
    ensure!(
        words.dtype == WeightDtype::Int32 && scales.dtype == WeightDtype::BF16,
        "{prefix}: weight_packed {:?} / weight_scale {:?}, expected Int32 / BF16",
        words.dtype,
        scales.dtype
    );
    check_tensor_pair(scheme, n, k, ("I32", &words.shape), ("BF16", &scales.shape))
        .with_context(|| prefix.to_string())?;
    Ok(PackedIntTensor {
        words: words.ptr,
        scales: scales.ptr,
    })
}

/// 2026-10-07: The packed-int MoE FFN of the layer under `mlp`.
pub(super) fn load_packed_int_moe(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    mlp: &str,
    scheme: PackedIntScheme,
) -> Result<FfnComponent> {
    ensure!(
        config.ep_world_size <= 1,
        "packed-int Laguna experts have no expert-parallel path (ep_world_size {})",
        config.ep_world_size
    );
    let (h, inter, si, e) = (
        config.hidden_size,
        config.moe_intermediate_size,
        config.shared_expert_intermediate_size,
        config.num_experts,
    );
    let experts = (0..e)
        .map(|x| {
            let ep = format!("{mlp}.experts.{x}");
            Ok(PackedIntExpert {
                gate_proj: packed(store, scheme, &format!("{ep}.gate_proj"), inter, h)?,
                up_proj: packed(store, scheme, &format!("{ep}.up_proj"), inter, h)?,
                down_proj: packed(store, scheme, &format!("{ep}.down_proj"), h, inter)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let bf16 = WeightDtype::BF16;
    let shared = format!("{mlp}.shared_expert");
    let weights = PackedIntMoeWeights {
        scheme,
        gate: checked(store, &format!("{mlp}.gate.weight"), bf16, &[e, h])?,
        correction_bias: checked(
            store,
            &format!("{mlp}.experts.e_score_correction_bias"),
            WeightDtype::FP32,
            &[e],
        )?,
        shared_gate: checked(store, &format!("{shared}.gate_proj.weight"), bf16, &[si, h])?,
        shared_up: checked(store, &format!("{shared}.up_proj.weight"), bf16, &[si, h])?,
        shared_down: checked(store, &format!("{shared}.down_proj.weight"), bf16, &[h, si])?,
        experts,
    };
    Ok(FfnComponent::PackedIntMoe(PackedIntMoeLayer::new(
        weights, config, gpu,
    )?))
}

#[cfg(test)]
#[path = "packed_int_tests.rs"]
mod tests;
