// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `impl ModelWeightLoader for Glm5NextWeightLoader`.
//!
//! Owner: model-arch weight loader.
//! Invariants: none beyond the types.

use super::*;
use crate::glm5next_mlp::Glm5NextDenseSite;
use crate::glm5next_mlp::build_w4a4::{act_scales_of, build_dense_nvfp4};
use crate::glm5next_mlp::precision::{MlpGroup, MlpKernel};

impl ModelWeightLoader for Glm5NextWeightLoader {
    /// 2026-09-25: True only when `metrale_config::glm_vision_enabled()`
    /// (`METRALE_GLM_VISION`). The GLM-5.3 config parser reads the same function
    /// before it parses `vision_config`; this method takes no config, so the
    /// gate is the environment.
    fn binds_vision_encoder(&self) -> bool {
        metrale_config::glm_vision_enabled()
    }

    /// 2026-09-25: Bind the `model.visual.*` tower; see
    /// [`crate::weight_loader::glm5_next_vision`].
    fn load_vision_encoder(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<Option<metrale_model_layers::layers::VisionTower>> {
        crate::weight_loader::glm5_next_vision::load_glm5_next_vision(store, config, gpu)
    }

    /// 2026-09-25: Keep the MTP layer's BF16 routed-expert weights off the
    /// device (`is_full_width_mtp_expert`); `bind_expert` quantises them from
    /// disk. The layer is `num_hidden_layers`. A checkpoint whose MTP experts
    /// are U8 defers nothing. 2026-10-08: Also every text layer's F32
    /// `*.input_scale` (`is_activation_scale`): one scalar per projection, read
    /// on the host by `act_scale::input_scale` rather than given an allocation
    /// granule each.
    fn defer_predicate(
        &self,
        config: &ModelConfig,
    ) -> Option<metrale_model_weights::weights::DeferHook> {
        let num_layers = config.num_hidden_layers;
        Some(std::sync::Arc::new(
            move |name: &str, dtype: WeightDtype| {
                is_full_width_mtp_expert(name, dtype, num_layers)
                    || is_activation_scale(name, dtype)
            },
        ))
    }

    /// 2026-09-25: DSA, KDA and the MLP all shard under TP (see `glm5_next_load.rs`);
    /// routed experts are also split by EP (`local_expert_range`). 2026-10-08: The head and
    /// width splits come from `metrale_config::tp_split`, so they need not divide evenly.
    fn tp_support(&self) -> metrale_config::TpSupport {
        metrale_config::TpSupport::Uneven
    }

    fn load_layers(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        _layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        let skeleton = Glm5NextTextSkeleton::from_config(config)?;
        let kda_cfg = Glm5NextKdaConfig::from_model_config(config);
        kda_cfg.validate()?;
        // 2026-09-25: `gate_rank` is the row count of layer 0's `f_a_proj`; the
        // config has no such key.
        let gate_rank = {
            let n = qualify(0, "self_attn.f_a_proj.weight");
            let t = store.get(&n).with_context(|| {
                format!("glm5_next: {n} is needed to size the KDA gate bottleneck")
            })?;
            *t.shape.first().context("f_a_proj has no rows")?
        };
        let kda_plan = KdaTpPlan::from_config(config, gate_rank)?;
        let dsa_cfg = Glm5NextDsaConfig::from_config(config)?;
        let mlp_cfg = Glm5NextMlpConfig::from_config(config)?;

        let kda_kernels = Glm5NextKdaKernels::resolve(gpu)?;
        let dsa_kernels = Glm5NextDsaKernels::resolve(gpu)?;
        let dsa_layer_kernels = Glm5NextDsaLayerKernels::resolve(gpu)?;
        let mlp_kernels = Glm5NextMlpKernels::resolve(gpu)?;
        let mhc_kernels_probe = Glm5NextMhcKernels::resolve(gpu)?;
        let rms_norm_k = gpu.kernel("rms_norm_vanilla", "rms_norm_vanilla")?;
        // 2026-09-25: Only a layer without mHC (the MTP layer) uses `add_k`, and it
        // fails there if the kernel is missing; `try_kernel` lets a target without
        // it still build the text stack.
        let add_k = metrale_model_layers::layers::try_kernel(gpu, "bf16_add", "bf16_add_inplace");

        // 2026-09-25: Every KDA layer shares one workspace (one `kda_cfg`). The
        // workspaces are sized for `verify_k` rows, the largest of the batched
        // GEMV's `DENSE_GEMV_BATCHM_MAX_M`, `PREFILL_ROWS` and `prefill_rows()`
        // (`METRALE_GLM_PREFILL_ROWS`).
        let verify_k = (metrale_model_layers::layers::ops::DENSE_GEMV_BATCHM_MAX_M as usize)
            .max(crate::glm5next_layer::PREFILL_ROWS)
            .max(crate::glm5next_layer::prefill_rows());
        let kda_ws = std::sync::Arc::new(crate::glm5next_kda::Glm5NextKdaWorkspace::new(
            gpu, &kda_cfg, verify_k,
        )?);

        // 2026-09-25: Unless `METRALE_GLM_MLP_WS_SHARED=0`, one MLP workspace serves
        // every layer; otherwise each layer allocates its own. Either way it is
        // allocated here, at load, before the KV pool is sized.
        let mlp_ws_bytes = crate::glm5next_mlp::forward::mlp_ws_total_bytes(&mlp_cfg, verify_k);
        let shared_mlp_ws = if crate::glm5next_mlp::forward::mlp_ws_shared() {
            tracing::info!(
                "GLM MLP workspace: SHARED, 1 x {:.1} MB for {} layers at {verify_k} rows \
                 (per-layer would be {:.1} MB)",
                mlp_ws_bytes as f64 / 1e6,
                skeleton.layers.len(),
                (mlp_ws_bytes * skeleton.layers.len()) as f64 / 1e6,
            );
            Some(std::sync::Arc::new(
                crate::glm5next_mlp::forward::Glm5NextMlpWorkspace::new(gpu, &mlp_cfg, verify_k)?,
            ))
        } else {
            tracing::warn!(
                "GLM MLP workspace: PER-LAYER, {} x {:.1} MB at {verify_k} rows",
                skeleton.layers.len(),
                mlp_ws_bytes as f64 / 1e6,
            );
            None
        };

        let dsa_plan = crate::glm5next_dsa::tp::DsaTpPlan::from_config(config, &dsa_cfg)?;
        let last = skeleton.layers.len() - 1;
        let mut out: Vec<Box<dyn TransformerLayer>> = Vec::with_capacity(skeleton.layers.len());

        // 2026-09-25: The KV pool has `num_attention_layers()` slots, which counts
        // the sparse-attention layers only, so a DSA layer addresses it by its
        // ordinal among DSA layers, not by its model index.
        let mut attn_layer_idx = 0usize;

        for sl in &skeleton.layers {
            let idx = sl.index;
            let t_layer = std::time::Instant::now();
            let src = LayerSource::collect(gpu, store, idx)
                .with_context(|| format!("glm5_next: collecting layer {idx}"))?;
            let t_collect = t_layer.elapsed();

            let mixer = match sl.mixer {
                Mixer::Kda => {
                    let sharded = KdaShardedSource::new(&src, &kda_plan)?;
                    let (w, _report) = bind_kda_weights(gpu, &kda_cfg, idx, &sharded)?;
                    Glm5NextMixer::Kda {
                        layer: Box::new(Glm5NextKdaLayer::new(idx, kda_cfg, w, kda_kernels)?),
                        ws: kda_ws.clone(),
                        cfg: kda_cfg,
                    }
                }
                Mixer::Dsa => {
                    let load = |n: &str| src.f32(n);
                    let w = build_dsa_weights(gpu, &dsa_cfg, &dsa_plan, &load)?;
                    Glm5NextMixer::Dsa(Box::new(Glm5NextDsaLayer {
                        persist_bt: std::env::var("METRALE_GLM_DSA_ALLOC_PER_STEP").as_deref()
                            != Ok("1"),
                        cfg: dsa_cfg,
                        weights: w,
                        kernels: dsa_layer_kernels,
                        select_kernels: dsa_kernels,
                        decode_kernel:
                            crate::glm5next_dsa::attend::Glm5NextDsaDecodeKernel::resolve(gpu)?,
                        workspace: crate::glm5next_dsa::layer::Glm5NextDsaWorkspace::new(
                            gpu, &dsa_cfg, verify_k,
                        )?,
                        layer_idx: idx,
                        attn_layer_idx: {
                            let a = attn_layer_idx;
                            attn_layer_idx += 1;
                            a
                        },
                        rms_eps: config.rms_norm_eps as f32,
                        kv_scale: 1.0,
                    }))
                }
            };

            let t_mixer = t_layer.elapsed();

            let load = |n: &str| src.f32(n);
            let mlp = match sl.mlp {
                Mlp::Dense => Glm5NextMlpSite::Dense(Box::new(build_dense_site(
                    gpu,
                    store,
                    config,
                    &mlp_cfg,
                    &mlp_kernels,
                    &src,
                    idx,
                    verify_k,
                )?)),
                Mlp::RoutedMoe => {
                    let expert = |id: usize| bind_expert(gpu, store, idx, id);
                    let precision = |has_scales: bool| {
                        mlp_precision::group_precision(
                            config,
                            &mlp_kernels,
                            idx,
                            MlpGroup::RoutedExperts,
                            mlp_cfg.local_expert_range().start,
                            has_scales,
                        )
                    };
                    let moe = mlp_build::build_moe(
                        gpu,
                        &mlp_cfg,
                        config.shared_expert_intermediate_size,
                        &load,
                        &expert,
                        &precision,
                        verify_k,
                    )?;
                    mlp_precision::announce(&moe.precision, verify_k);
                    Glm5NextMlpSite::Moe(Box::new(moe))
                }
            };

            if crate::glm5next_fp8_dense::enabled() {
                register_fp8_dense(gpu, &mixer, &mlp, &mlp_cfg, idx)?;
            }

            let t_mlp = t_layer.elapsed();
            tracing::info!(
                "glm5_next layer {idx} built: collect {:.2}s mixer {:.2}s mlp {:.2}s \
                 (mixer={:?} mlp={:?})",
                t_collect.as_secs_f64(),
                (t_mixer - t_collect).as_secs_f64(),
                (t_mlp - t_mixer).as_secs_f64(),
                sl.mixer,
                sl.mlp,
            );

            let mhc = if sl.hyper_connection {
                Some(Glm5NextMhc {
                    kernels: mhc_kernels_probe,
                    attn: bind_mhc_site(gpu, &src, "attn", config.hc_mult, config.hidden_size)?,
                    ffn: bind_mhc_site(gpu, &src, "ffn", config.hc_mult, config.hidden_size)?,
                    hc_mult: config.hc_mult,
                    sinkhorn_iters: config.hc_sinkhorn_iters,
                    hc_eps: config.hc_eps,
                })
            } else {
                None
            };

            out.push(Box::new(Glm5NextLayer {
                layer_idx: idx,
                mixer,
                mlp,
                mlp_cfg,
                mlp_kernels,
                mlp_ws: match &shared_mlp_ws {
                    Some(ws) => ws.clone(),
                    None => std::sync::Arc::new(
                        crate::glm5next_mlp::forward::Glm5NextMlpWorkspace::new(
                            gpu, &mlp_cfg, verify_k,
                        )?,
                    ),
                },
                mhc,
                input_norm: upload_f32_as_bf16(gpu, &src.f32("input_layernorm.weight")?)?,
                post_attn_norm: upload_f32_as_bf16(
                    gpu,
                    &src.f32("post_attention_layernorm.weight")?,
                )?,
                rms_norm_k,
                add_k,
                rms_eps: config.rms_norm_eps as f32,
                hidden: config.hidden_size,
                mixer_all_reduce: match sl.mixer {
                    Mixer::Kda => kda_plan.needs_output_all_reduce(),
                    Mixer::Dsa => dsa_plan.needs_output_all_reduce(),
                },
                is_first: idx == 0,
                is_last: idx == last,
            }));
        }
        if crate::glm5next_fp8_dense::enabled() {
            let (count, bytes) = crate::glm5next_fp8_dense::registered();
            tracing::warn!(
                "glm5_next --dense-quantization fp8: {count} BF16 dense projections also held as \
                 FP8 per-channel ({:.2} GB) and decoded W8A8, BELOW the checkpoint's declared \
                 BF16; the router, the indexer's wq_b and weights_proj stay BF16",
                bytes as f64 / 1e9
            );
        }
        Ok(out)
    }

    fn load_embedding(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense(store, "model.language_model.embed_tokens.weight")
    }

    fn load_final_norm(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense(store, "model.language_model.norm.weight")
    }

    fn load_lm_head(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense(store, "lm_head.weight")
    }

    /// 2026-09-25: `None`: the GLM-5.3 MTP layer is loaded by
    /// `glm5_next_mtp::load_glm5next_mtp_module`, not as `MtpWeights`.
    fn load_mtp_weights(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<Option<crate::weight_loader::MtpWeights>> {
        Ok(None)
    }

    /// 2026-09-25: Free the store tensors `is_reuploaded` matches, then those
    /// `is_quantized_expert_weight` matches.
    fn prune_after_load(
        &self,
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
        Ok(())
    }
}

/// 2026-10-08: One dense MLP site: its precision plan, then the weight forms the plan reaches
/// within `max_rows`: the checkpoint's packed NVFP4, TP-sliced, for W4A4, and the BF16
/// dequantization for the 16-bit path (both when a ladder mixes them).
#[allow(clippy::too_many_arguments)]
fn build_dense_site(
    gpu: &dyn GpuBackend,
    store: &WeightStore,
    config: &ModelConfig,
    mlp_cfg: &Glm5NextMlpConfig,
    kernels: &Glm5NextMlpKernels,
    src: &LayerSource,
    idx: usize,
    max_rows: usize,
) -> Result<Glm5NextDenseSite> {
    let act = ["gate_proj", "up_proj", "down_proj"]
        .map(|p| input_scale(gpu, store, idx, &format!("mlp.{p}")))
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
    let act = [act[0], act[1], act[2]];
    let precision = mlp_precision::group_precision(
        config,
        kernels,
        idx,
        MlpGroup::DenseMlp,
        0,
        act.iter().all(Option::is_some),
    )?;
    mlp_precision::announce(&precision, max_rows);
    let nvfp4 = if precision.reaches(MlpKernel::W4a4Static, max_rows) {
        let scales = act_scales_of(act, &format!("layer {idx} dense MLP"))?;
        let raw = |n: &str| src.raw(n);
        let w = build_dense_nvfp4(
            gpu,
            mlp_cfg.hidden,
            config.intermediate_size,
            mlp_cfg.dense_slice(),
            "mlp",
            &raw,
            scales,
        )?;
        Some((w, scales))
    } else {
        None
    };
    let bf16 = if precision.reaches(MlpKernel::Bf16, max_rows) {
        let load = |n: &str| src.f32(n);
        Some(mlp_build::build_dense_mlp(
            gpu,
            mlp_cfg,
            config.intermediate_size,
            mlp_cfg.dense_slice(),
            "mlp",
            &load,
        )?)
    } else {
        None
    };
    Ok(Glm5NextDenseSite {
        bf16,
        nvfp4,
        precision,
    })
}

/// 2026-10-09: `--dense-quantization fp8`: register the FP8 copy of every BF16 projection of
/// one layer's mixer and shared expert (`glm5next_fp8_dense`), with the shapes their forwards
/// launch.
fn register_fp8_dense(
    gpu: &dyn GpuBackend,
    mixer: &Glm5NextMixer,
    mlp: &Glm5NextMlpSite,
    mlp_cfg: &Glm5NextMlpConfig,
    idx: usize,
) -> Result<()> {
    let mut projs = match mixer {
        Glm5NextMixer::Kda { layer, .. } => layer.dense_projections(),
        Glm5NextMixer::Dsa(layer) => layer.dense_projections(),
    };
    if let Glm5NextMlpSite::Moe(w) = mlp {
        let (h, s) = (mlp_cfg.hidden, mlp_cfg.local_shared_intermediate);
        projs.push((w.shared.gate_proj, s, h, "shared_experts.gate_proj"));
        projs.push((w.shared.up_proj, s, h, "shared_experts.up_proj"));
        projs.push((w.shared.down_proj, h, s, "shared_experts.down_proj"));
    }
    let max_k = projs.iter().map(|p| p.2).max().unwrap_or(0);
    crate::glm5next_fp8_dense::prepare(gpu, max_k)?;
    let quantize = gpu.kernel("gemv_fp8w", "quantize_bf16_to_fp8")?;
    for (w, n, k, name) in projs {
        crate::glm5next_fp8_dense::register(
            gpu,
            quantize,
            w,
            n,
            k,
            &format!("layer {idx} {name}"),
            0,
        )?;
    }
    Ok(())
}
