// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `Glm5NextWeightLoader`: builds the GLM-5.3 text stack from a
//! `WeightStore`, plus the helpers the MTP loader shares.
//!
//! `Glm5NextTextSkeleton::from_config` fixes each layer's mixer (KDA or DSA)
//! and MLP (dense or routed MoE) from the config's `layer_types` and
//! `mlp_only_layers`; the loader builds what it lists.
//!
//! Under TP, DSA shards through `DsaTpPlan`, the MLP through the rank passed to
//! `glm5next_mlp::build`, and KDA through [`KdaShardedSource`], which slices
//! the host bytes before `bind_kda_weights` sees them. At TP > 1 both mixer
//! plans report `needs_output_all_reduce`, and the layer all-reduces the mixer
//! output (`Glm5NextLayer::mixer_all_reduce`).
//!
//! Owner: model-arch weight loader.
//! Invariants: none beyond the types.

use anyhow::{Context, Result, bail};
use metrale_cache::kv_cache::KvCacheDtype;
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_weights::weights::{WeightDtype, WeightStore, WeightTensor};

use super::ModelWeightLoader;
use crate::glm5next_dsa::build::build_dsa_weights;
use crate::glm5next_dsa::layer::{Glm5NextDsaLayer, Glm5NextDsaLayerKernels};
use crate::glm5next_dsa::{Glm5NextDsaConfig, Glm5NextDsaKernels};
use crate::glm5next_kda::binding::{KdaDtype, KdaTensorSource, RawTensor, bind_kda_weights};
use crate::glm5next_kda::tp::KdaTpPlan;
use crate::glm5next_kda::tp_bind::KdaShardedSource;
use crate::glm5next_kda::{Glm5NextKdaConfig, Glm5NextKdaKernels, Glm5NextKdaLayer};
use crate::glm5next_layer::{Glm5NextLayer, Glm5NextMhc, Glm5NextMixer, Glm5NextMlpSite};
use crate::glm5next_mhc::{Glm5NextMhcKernels, Glm5NextMhcSiteWeights, mix_hc};
use crate::glm5next_mlp::weights::{Glm5NextExpertWeights, Nvfp4Proj, ScaleLayout};
use crate::glm5next_mlp::{Glm5NextMlpConfig, Glm5NextMlpKernels, build as mlp_build};
use crate::glm5next_skeleton::{Glm5NextTextSkeleton, Mixer, Mlp};
use metrale_model_layers::layer::TransformerLayer;
use metrale_model_layers::weight_map::DenseWeight;

mod act_scale;
#[cfg(test)]
mod defer_hook_tests;
mod dense_site;
pub(crate) mod dense_tiers;
mod dsa_stack;
mod expert_digest;
mod expert_prepare;
mod expert_quant;
mod expert_scales;
mod expert_slab;
mod expert_tp;
mod expert_whole;
#[cfg(test)]
mod export_layout_tests;
mod fp8_head;
mod fp8_widen;
mod layer_stream;
#[cfg(test)]
mod layer_stream_tests;
mod loader;
mod mlp_precision;
mod nvfp4_dequant;
mod nvfp4_quant;
#[cfg(test)]
mod plan_cast_tests;
mod plan_dtype;
mod prune;
mod scale_pack;
#[cfg(test)]
mod tp_shapes_tests;

use act_scale::input_scale;
use act_scale::is_activation_scale;
use expert_quant::{quantize_deferred_expert_proj, quantize_expert_proj};
use expert_whole::bind_expert;

pub struct Glm5NextWeightLoader;

/// 2026-09-25: The full name of layer `layer`'s tensor `leaf`, under
/// `model.language_model.layers.`.
fn qualify(layer: usize, leaf: &str) -> String {
    format!("model.language_model.layers.{layer}.{leaf}")
}

/// 2026-09-25: Whether a store tensor belongs to a text layer (index below
/// `num_layers`) and is not a routed expert. `load_layers` copies those to the
/// host (`LayerSource::collect`) and the binders upload their own buffers, so
/// `prune_after_load` frees the store's copies.
///
/// Two families are never matched:
/// - `mlp.experts.*`: `bind_expert` binds U8 experts zero-copy from the store's
///   pointers.
/// - layer `num_layers`, the MTP layer: `glm5_next_mtp::load_glm5next_mtp_module`
///   reads it.
fn is_reuploaded(name: &str, num_layers: usize) -> bool {
    let Some(rest) = name.strip_prefix("model.language_model.layers.") else {
        return false;
    };
    let Some((idx, rel)) = rest.split_once('.') else {
        return false;
    };
    let Ok(idx) = idx.parse::<usize>() else {
        return false;
    };
    idx < num_layers && !rel.starts_with("mlp.experts.")
}

/// 2026-09-25: Whether a store tensor is a BF16 routed-expert projection
/// weight, which [`bind_expert`] quantised to NVFP4 rather than binding. The
/// dtype is the whole test: a U8 expert is the kernel's operand and is never
/// matched. `prune_after_load` frees the matches; `factory::build` calls it
/// after the MTP module is bound. With a store that honoured
/// [`Glm5NextWeightLoader::defer_predicate`], the MTP layer's BF16 experts are
/// never resident.
fn is_quantized_expert_weight(name: &str, dtype: WeightDtype) -> bool {
    dtype == WeightDtype::BF16
        && name.starts_with("model.language_model.layers.")
        && name.contains(".mlp.experts.")
        && name.ends_with("_proj.weight")
}

/// 2026-09-25: Whether a store tensor is a BF16 `mlp.experts.*_proj.weight` of
/// layer `num_layers`, the MTP layer: the family
/// [`Glm5NextWeightLoader::defer_predicate`] keeps off the device. The shared
/// expert (`mlp.shared_experts.*`) does not match; it is read through
/// [`LayerSource`].
fn is_full_width_mtp_expert(name: &str, dtype: WeightDtype, num_layers: usize) -> bool {
    if dtype != WeightDtype::BF16 || !name.ends_with("_proj.weight") {
        return false;
    }
    let Some(rest) = name.strip_prefix("model.language_model.layers.") else {
        return false;
    };
    let Some((idx, rel)) = rest.split_once('.') else {
        return false;
    };
    idx.parse::<usize>().ok() == Some(num_layers) && rel.starts_with("mlp.experts.")
}

/// 2026-09-25: Read a device tensor back as host bytes.
fn host_bytes(gpu: &dyn GpuBackend, t: &WeightTensor) -> Result<Vec<u8>> {
    let mut b = vec![0u8; t.byte_size()];
    gpu.copy_d2h(t.ptr, &mut b)?;
    Ok(b)
}

/// 2026-09-25: Read a BF16 or FP32 device tensor back as host `f32`, by the
/// tensor's own dtype; any other dtype is an error.
fn host_f32(gpu: &dyn GpuBackend, t: &WeightTensor, what: &str) -> Result<Vec<f32>> {
    let b = host_bytes(gpu, t)?;
    match t.dtype {
        WeightDtype::BF16 => Ok(b
            .chunks_exact(2)
            .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect()),
        WeightDtype::FP32 => Ok(b
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        other => bail!(
            "{what}: dtype {other:?} cannot be read as f32 without a conversion this loader refuses to guess"
        ),
    }
}

fn upload_f32(gpu: &dyn GpuBackend, v: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = gpu.alloc(b.len().max(1))?;
    gpu.copy_h2d(&b, p)?;
    Ok(p)
}

/// 2026-09-25: One layer's tensors under layer-relative names, copied to the
/// host (routed experts by name only). 2026-10-16: Names sorted, so every run sees one order.
pub(super) struct LayerSource {
    names: std::collections::BTreeSet<String>,
    tensors: fp8_widen::HostTensors,
    /// 2026-09-25: Copies, at the width the binders bind at, of the tensors
    /// stored at another width (see [`plan_dtype`]). Only
    /// [`KdaTensorSource::get`] reads them; [`Self::f32`] reads the stored bytes.
    plan_cast: std::collections::BTreeMap<String, (WeightDtype, Vec<u8>)>,
}

impl LayerSource {
    pub(super) fn collect(gpu: &dyn GpuBackend, store: &WeightStore, layer: usize) -> Result<Self> {
        let prefix = format!("model.language_model.layers.{layer}.");
        let mut names = std::collections::BTreeSet::new();
        let mut tensors = fp8_widen::HostTensors::new();
        let rels: std::collections::BTreeSet<String> = store
            .names()
            .filter_map(|n| n.strip_prefix(&prefix).map(|r| r.to_string()))
            .collect();
        for rel in rels {
            names.insert(rel.clone());
            // 2026-09-25: Routed experts are listed but not copied: `bind_expert` reads
            // them from the store.
            if rel.starts_with("mlp.experts.") {
                continue;
            }
            let t = store.get(&format!("{prefix}{rel}"))?;
            tensors.insert(rel, (t.dtype, t.shape.clone(), host_bytes(gpu, t)?));
        }
        // 2026-10-16: The tensors `METRALE_GLM_LAYER_STREAM` kept off the device, from their shard.
        for (name, d) in store.deferred_sorted() {
            let Some(rel) = name.strip_prefix(&prefix) else {
                continue;
            };
            if !layer_stream::collected_from_shard(rel, name, d.dtype) {
                continue;
            }
            names.insert(rel.to_string());
            let bytes = d
                .read_host_bytes()
                .with_context(|| format!("{name}: reading the streamed layer tensor"))?;
            tensors.insert(rel.to_string(), (d.dtype, d.shape.clone(), bytes));
        }
        Self::from_host(names, tensors)
    }

    /// 2026-10-17: The source over one layer's listed names and host-copied tensors (`collect`
    /// without the device reads). Block-scaled E4M3 linears are replaced by their BF16 widening
    /// and their consumed scales leave both lists, so the binders see a 16-bit layer; then the
    /// plan-width copies are made.
    pub(super) fn from_host(
        mut names: std::collections::BTreeSet<String>,
        mut tensors: fp8_widen::HostTensors,
    ) -> Result<Self> {
        for scale in fp8_widen::widen_layer_fp8(&mut tensors)? {
            names.remove(&scale);
        }
        let mut plan_cast = std::collections::BTreeMap::new();
        for (rel, (dtype, _, bytes)) in &tensors {
            if let Some(cast) = plan_dtype::cast_to_plan_dtype(rel, *dtype, bytes)? {
                plan_cast.insert(rel.clone(), cast);
            }
        }
        Ok(Self {
            names,
            tensors,
            plan_cast,
        })
    }

    /// 2026-10-08: A tensor's stored dtype, shape and bytes, copied (the packed NVFP4 dense MLP
    /// the W4A4 path slices).
    pub(super) fn raw(&self, name: &str) -> Result<(WeightDtype, Vec<usize>, Vec<u8>)> {
        let (dtype, shape, bytes) = self
            .tensors
            .get(name)
            .with_context(|| format!("missing tensor {name}"))?;
        Ok((*dtype, shape.clone(), bytes.clone()))
    }

    pub(super) fn f32(&self, name: &str) -> Result<Vec<f32>> {
        let (dtype, shape, bytes) = self
            .tensors
            .get(name)
            .with_context(|| format!("missing tensor {name}"))?;
        match dtype {
            WeightDtype::BF16 => Ok(bytes
                .chunks_exact(2)
                .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect()),
            WeightDtype::FP32 => Ok(bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()),
            // 2026-09-25: A packed-NVFP4 tensor is dequantized, so `f32` accepts
            // it as well as BF16 and FP32.
            WeightDtype::UInt8 => self.dequant_packed_nvfp4(name, shape, bytes),
            other => bail!("{name}: dtype {other:?} is not a plain float tensor"),
        }
    }

    /// 2026-09-25: One packed-NVFP4 `.weight` of this layer as `f32`, scaled by
    /// its `.weight_scale` (F8_E4M3 blocks) and scalar `.weight_scale_2`
    /// siblings. A missing or differently typed sibling is an error. Reached
    /// only from the `UInt8` arm of [`Self::f32`].
    fn dequant_packed_nvfp4(&self, name: &str, shape: &[usize], bytes: &[u8]) -> Result<Vec<f32>> {
        let base = name.strip_suffix(".weight").with_context(|| {
            format!("{name}: packed NVFP4 must be a `.weight`, with scale siblings beside it")
        })?;
        let (scale_dtype, _, scale_bytes) = self
            .tensors
            .get(&format!("{base}.weight_scale"))
            .with_context(|| format!("{name} is packed NVFP4 but {base}.weight_scale is absent"))?;
        if *scale_dtype != WeightDtype::FP8E4M3 {
            bail!("{base}.weight_scale is {scale_dtype:?}, expected F8_E4M3 block scales");
        }
        let s2 = self.f32(&format!("{base}.weight_scale_2"))?;
        let [scale_2] = s2[..] else {
            bail!("{base}.weight_scale_2 is not a scalar");
        };
        nvfp4_dequant::dequant_nvfp4_to_f32(name, bytes, shape, scale_bytes, scale_2)
    }
}

impl KdaTensorSource for LayerSource {
    fn get(&self, name: &str) -> Option<RawTensor<'_>> {
        let (dtype, shape, bytes) = match self.plan_cast.get(name) {
            // 2026-09-25: A plan-width copy, when one exists, replaces the stored bytes.
            Some((dtype, bytes)) => (dtype, &self.tensors.get(name)?.1, bytes),
            None => {
                let (dtype, shape, bytes) = self.tensors.get(name)?;
                (dtype, shape, bytes)
            }
        };
        // 2026-09-25: Only BF16 and FP32 tensors are offered; any other dtype reads
        // as absent rather than cast.
        let dtype = match dtype {
            WeightDtype::BF16 => KdaDtype::Bf16,
            WeightDtype::FP32 => KdaDtype::F32,
            _ => return None,
        };
        Some(RawTensor {
            dtype,
            shape: shape.clone(),
            bytes,
        })
    }
    fn names(&self) -> Vec<String> {
        self.names.iter().cloned().collect()
    }
}

/// 2026-09-25: Bind one mHC site (`attn` or `ffn`) of a layer. `hc_{site}_fn`
/// is uploaded as BF16 with `hc_fn_bf16` set; `base` and `scale` as F32. Each
/// tensor's length is checked first.
fn bind_mhc_site(
    gpu: &dyn GpuBackend,
    src: &LayerSource,
    site: &str,
    hc_mult: usize,
    hidden: usize,
    mix_tokens: usize,
) -> Result<Glm5NextMhcSiteWeights> {
    let f = src.f32(&format!("hc_{site}_fn"))?;
    let want = mix_hc(hc_mult) * hc_mult * hidden;
    if f.len() != want {
        bail!(
            "hc_{site}_fn has {} elements, expected mix_hc({hc_mult}) * {hc_mult} * {hidden} = {want}",
            f.len()
        );
    }
    let base = src.f32(&format!("hc_{site}_base"))?;
    if base.len() != mix_hc(hc_mult) {
        bail!(
            "hc_{site}_base has {} entries, expected mix_hc({hc_mult}) = {}",
            base.len(),
            mix_hc(hc_mult)
        );
    }
    let scale = src.f32(&format!("hc_{site}_scale"))?;
    if scale.len() != 3 {
        bail!(
            "hc_{site}_scale has {} entries, expected 3 (pre, post, comb)",
            scale.len()
        );
    }
    Ok(Glm5NextMhcSiteWeights {
        hc_fn: upload_f32_as_bf16(gpu, &f)?,
        hc_fn_bf16: true,
        hc_scale: upload_f32(gpu, &scale)?,
        hc_base: upload_f32(gpu, &base)?,
        // 2026-09-25: The `hc_mix` to `hc_finish` scratch; one per site, so a
        // layer's two sites never share it.
        mix: gpu.alloc(mix_tokens * mix_hc(hc_mult) * 4)?,
        // 2026-10-09: `glm5next_hc_pre_norm` requires its counters zero before the first launch.
        done: {
            let bytes = mix_tokens * 4;
            let p = gpu.alloc(bytes)?;
            gpu.memset(p, 0, bytes)?;
            p
        },
        mix_tokens,
    })
}

fn dense(store: &WeightStore, name: &str) -> Result<DenseWeight> {
    Ok(DenseWeight {
        weight: store.get(name)?.ptr,
    })
}

pub(super) fn upload_f32_as_bf16(gpu: &dyn GpuBackend, v: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = v
        .iter()
        .flat_map(|x| half::bf16::from_f32(*x).to_le_bytes())
        .collect();
    let p = gpu.alloc(b.len().max(1))?;
    gpu.copy_h2d(&b, p)?;
    Ok(p)
}

#[cfg(test)]
mod prune_tests;

/// 2026-09-25: [`LayerSource::collect`], for the MTP loader.
pub(super) fn layer_source(
    gpu: &dyn GpuBackend,
    store: &WeightStore,
    layer: usize,
) -> Result<LayerSource> {
    LayerSource::collect(gpu, store, layer)
}

/// 2026-09-25: [`bind_expert`], for the MTP loader. 2026-10-09: Under the MLP config's
/// `shard` (`expert_tp::bind_routed_expert`). 2026-10-14: As the binder of all of `layer`'s
/// experts (`expert_tp::routed_expert_binder`), one slab under the `tp` layout. 2026-10-24:
/// Read as `settings` say (`expert_prepare::ExpertLoad`).
pub(super) fn expert_binder_at<'a>(
    gpu: &'a dyn GpuBackend,
    store: &'a WeightStore,
    layer: usize,
    cfg: &Glm5NextMlpConfig,
    settings: &metrale_settings::InferenceServerSettings,
) -> Result<impl Fn(usize) -> Result<Glm5NextExpertWeights> + use<'a>> {
    let load = expert_prepare::ExpertLoad::from_settings(settings);
    expert_tp::routed_expert_binder(gpu, store, layer, cfg, None, &load)
}

/// 2026-10-08: The routed-expert precision plan of the MTP layer `layer`, for the MTP loader.
/// 2026-10-11: Register the MTP block's mixer and shared-expert projections under the published
/// `--dense-quantization` and `--dsa-quantization`, as `loader.rs` registers a text layer's, so
/// the drafter's DSA q_a / absorbed q / absorbed o follow the DSA tier and the rest of its dense
/// projections the dense tier. Nothing to do when both are `declared`.
pub(super) fn register_mtp_dense_tiers(
    gpu: &dyn GpuBackend,
    mixer: &mut crate::glm5next_layer::Glm5NextMixer,
    mlp: &mut crate::glm5next_layer::Glm5NextMlpSite,
    mlp_cfg: &crate::glm5next_mlp::Glm5NextMlpConfig,
    dense: &crate::glm5next_dense_levers::DenseLevers,
    idx: usize,
) -> Result<()> {
    // 2026-10-11: Opt-in (`METRALE_GLM_MTP_DRAFTER_TIERS=1`): the first measurement with the
    // drafter on the serve's tiers saw p1 fall (0.64-0.70 -> 0.53-0.61), not yet separated from
    // the device exchange; the drafter stays BF16 until it is.
    if !dense.mtp_drafter_tiers {
        return Ok(());
    }
    if !dense_tiers::any_below_declared(
        metrale_model_layers::layers::dense_quantization(),
        metrale_model_layers::layers::dsa_quantization(),
    ) {
        return Ok(());
    }
    dense_tiers::register_dense_tiers(gpu, mixer, mlp, mlp_cfg, dense, idx)
}

pub(super) fn mtp_expert_precision(
    config: &ModelConfig,
    kernels: &Glm5NextMlpKernels,
    layer: usize,
    cfg: &Glm5NextMlpConfig,
    has_scales: bool,
) -> Result<crate::glm5next_mlp::precision::GroupPrecision> {
    mlp_precision::group_precision(
        config,
        kernels,
        layer,
        crate::glm5next_mlp::precision::MlpGroup::RoutedExperts,
        cfg.local_expert_range().start,
        has_scales,
    )
}

/// 2026-09-25: `upload_f32_as_bf16`, for the MTP loader.
pub fn upload_bf16(gpu: &dyn GpuBackend, v: &[f32]) -> Result<DevicePtr> {
    upload_f32_as_bf16(gpu, v)
}

#[cfg(test)]
mod vision_capability_tests;
