// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: Install the W8A8 decode weights on every layer whose FP8 projections the
//! declared-precision policy wants run W8A8 (`wants(module)`, module = the tensor prefix,
//! e.g. `model.language_model.layers.56.mlp.gate_proj`).
//!
//! Each projection stays the checkpoint's per-row E4M3 bytes, read in place from the store
//! (`rowwise_fp8::load_fp8_per_row`; its BF16 `[N,1]` scale is widened to F32 into a new
//! buffer). A layer's group installs only whole: attention Q, K, V and O; GDN in_proj_qkv,
//! in_proj_z and out_proj; FFN gate, up and down. The loader's other copies (NVFP4 requant)
//! are still built: prefill and launches wider than the W8A8 family use them.
//!
//! Owner: model-arch weight loader (Qwen3.5 dense).
//! Invariants:
//! - TP 1 only (the per-row loader does not shard); at TP > 1, or when the W8A8 kernels are
//!   not compiled into the target, nothing is installed.
//! - Every W8A8 projection of the model shares one `W8a8Ctx`, so one scratch.

use anyhow::{Context, Result};
use metrale_config::{LayerType, ModelConfig};
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_layers::layer::TransformerLayer;
use metrale_model_layers::layers::ops::{W8a8Scale, W8a8Weight};
use metrale_model_layers::layers::qwen3_attention::Qwen3AttentionLayer;
use metrale_model_layers::layers::qwen3_ssm::Qwen3SsmLayer;
use metrale_model_layers::layers::{W8a8Ctx, W8a8Ffn, W8a8Mixer};
use metrale_model_weights::weights::WeightStore;

use super::rowwise_fp8::{load_fp8_per_row, proj_is_fp8_per_row};
use super::served_formats::{Group, ServedFormats};

/// 2026-09-28: Projections installed, by group.
#[derive(Debug, Default, PartialEq, Eq)]
pub(super) struct W8a8Installed {
    pub attention: usize,
    pub gdn: usize,
    pub ffn: usize,
}

/// 2026-09-28: The widest K a W8A8 projection of this model reads.
fn max_k(config: &ModelConfig) -> u32 {
    let q_dim = config.num_attention_heads * config.head_dim;
    let value_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    config
        .hidden_size
        .max(config.intermediate_size)
        .max(q_dim)
        .max(value_dim) as u32
}

/// 2026-09-28: One W8A8 weight stacked from `names` under `prefix`, or `None` unless every
/// one is wanted and stored as per-row FP8.
fn stacked(
    store: &WeightStore,
    gpu: &dyn GpuBackend,
    prefix: &str,
    names: &[&str],
    wants: &dyn Fn(&str) -> bool,
) -> Result<Option<W8a8Weight>> {
    let modules: Vec<String> = names.iter().map(|n| format!("{prefix}.{n}")).collect();
    if !modules
        .iter()
        .all(|m| wants(m) && proj_is_fp8_per_row(store, m))
    {
        return Ok(None);
    }
    let segs = modules
        .iter()
        .map(|m| load_fp8_per_row(store, m, gpu))
        .collect::<Result<Vec<_>>>()?;
    W8a8Weight::new(&segs).map(Some)
}

/// 2026-09-28: The loader's entry: install every projection the declared-precision policy
/// wants run W8A8 (`WeightQuantPolicy::fp8_decode_act(module) == Some(Fp8)`: the `declared`
/// tier, FP8 weights and activations declared, `kernel_caps().w8a8_decode`).
pub(super) fn install_declared(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    layer_types: &[LayerType],
    layers: &mut [Box<dyn TransformerLayer>],
    served: &mut ServedFormats,
) -> Result<W8a8Installed> {
    let policy = super::load_cx::weight_quant_policy(config);
    if !policy.follows_plan() {
        return Ok(W8a8Installed::default());
    }
    let wants = |m: &str| {
        policy.fp8_decode_act(m) == Some(metrale_config::weight_quantization::ActFormat::Fp8)
    };
    install_w8a8_decode(store, config, gpu, layer_types, layers, &wants, served)
}

/// 2026-09-28: Install on `layers` (index = model layer) and return what was installed.
pub(super) fn install_w8a8_decode(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    layer_types: &[LayerType],
    layers: &mut [Box<dyn TransformerLayer>],
    wants: &dyn Fn(&str) -> bool,
    served: &mut ServedFormats,
) -> Result<W8a8Installed> {
    let mut done = W8a8Installed::default();
    if config.tp_world_size > 1 {
        return Ok(done);
    }
    let ctx = W8a8Ctx::new(gpu, max_k(config))?;
    if !ctx.kernels.resolved(W8a8Scale::PerRow) {
        tracing::info!(
            "W8A8 decode kernels not in this target; FP8-declared layers keep their other arms"
        );
        return Ok(done);
    }
    // 2026-10-05: The attention prefill's W8A8 projections run on the prefill stream, so they
    // get a scratch of their own (`Qwen3AttentionLayer::set_w8a8_prefill_ctx`).
    let prefill_ctx = W8a8Ctx::new(gpu, max_k(config))?;
    let (h, inter) = (config.hidden_size as u32, config.intermediate_size as u32);
    for (i, (lt, layer)) in layer_types.iter().zip(layers.iter_mut()).enumerate() {
        let lp = config.layer_prefix(i);
        let mlp = format!("{lp}.mlp");
        let ffn = match (
            stacked(store, gpu, &mlp, &["gate_proj"], wants)?,
            stacked(store, gpu, &mlp, &["up_proj"], wants)?,
            stacked(store, gpu, &mlp, &["down_proj"], wants)?,
        ) {
            (Some(gate), Some(up), Some(down)) => Some(W8a8Ffn {
                ctx,
                gate,
                up,
                down,
            }),
            _ => None,
        };
        let any = layer
            .as_any_mut()
            .with_context(|| format!("layer {i}: no downcast hook"))?;
        match lt {
            LayerType::FullAttention => {
                let l = any
                    .downcast_mut::<Qwen3AttentionLayer>()
                    .with_context(|| format!("layer {i}: not a Qwen3 attention layer"))?;
                let at = format!("{lp}.self_attn");
                let qkv = stacked(store, gpu, &at, &["q_proj", "k_proj", "v_proj"], wants)?;
                let o = stacked(store, gpu, &at, &["o_proj"], wants)?;
                if let (Some(input), Some(output)) = (qkv, o) {
                    l.set_w8a8_decode_weights(
                        W8a8Mixer { ctx, input, output },
                        input.n(),
                        h,
                        output.k(),
                    )?;
                    l.set_w8a8_prefill_ctx(prefill_ctx);
                    done.attention += 4;
                    served.upgrade_w8a8(Group::Attention, i)?;
                }
                if let Some(f) = ffn {
                    l.set_w8a8_ffn_weights(f, h, inter)?;
                    done.ffn += 3;
                    served.upgrade_w8a8(Group::Ffn, i)?;
                }
            }
            LayerType::LinearAttention => {
                let l = any
                    .downcast_mut::<Qwen3SsmLayer>()
                    .with_context(|| format!("layer {i}: not a Qwen3 GDN layer"))?;
                let la = format!("{lp}.linear_attn");
                let qkvz = stacked(store, gpu, &la, &["in_proj_qkv", "in_proj_z"], wants)?;
                let out = stacked(store, gpu, &la, &["out_proj"], wants)?;
                if let (Some(input), Some(output)) = (qkvz, out) {
                    l.set_w8a8_decode_weights(
                        W8a8Mixer { ctx, input, output },
                        input.n(),
                        h,
                        output.k(),
                    )?;
                    done.gdn += 3;
                    served.upgrade_w8a8(Group::Gdn, i)?;
                }
                if let Some(f) = ffn {
                    l.set_w8a8_ffn_weights(f, h, inter)?;
                    done.ffn += 3;
                    served.upgrade_w8a8(Group::Ffn, i)?;
                }
            }
            _ => {}
        }
    }
    tracing::info!(
        "W8A8 decode (declared FP8 W8A8, per-row E4M3 read in place): {} attention, {} GDN, {} FFN projections",
        done.attention,
        done.gdn,
        done.ffn
    );
    Ok(done)
}
