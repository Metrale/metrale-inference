// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `impl ModelWeightLoader for Glm5NextWeightLoader`.
//!
//! Owner: model-arch weight loader.
//! Invariants: none beyond the types.

use super::*;
use crate::glm5next_layer::l2_warm;
use crate::glm5next_mlp::precision::MlpGroup;

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
    /// granule each. 2026-10-09: Under `--moe-expert-layout tp`, also every
    /// routed-expert tensor of every layer (`expert_tp::is_routed_expert_tensor`):
    /// each rank uploads only its slices of them.
    /// 2026-10-16: Under `METRALE_GLM_LAYER_STREAM=1`, also every text layer's non-expert
    /// tensor (`layer_stream::streams`): `LayerSource::collect` reads them from their shard.
    /// 2026-10-17: Also csf scale planes and an 8-bit head with its scales (`expert_scales`,
    /// `fp8_head`): decoded or widened on the host.
    fn defer_predicate(
        &self,
        config: &ModelConfig,
    ) -> Option<metrale_model_weights::weights::DeferHook> {
        let num_layers = config.num_hidden_layers;
        let sliced = config.moe_expert_layout == metrale_config::MoeExpertLayout::Tp;
        // 2026-10-16: `load_layers` refuses an invalid `METRALE_GLM_LAYER_STREAM` value.
        let stream = matches!(layer_stream::enabled(), Ok(true));
        Some(std::sync::Arc::new(
            move |name: &str, dtype: WeightDtype| {
                is_full_width_mtp_expert(name, dtype, num_layers)
                    || is_activation_scale(name, dtype)
                    || expert_scales::is_csf_component(name)
                    || fp8_head::is_fp8_head_tensor(name, dtype)
                    || (sliced && expert_tp::is_routed_expert_tensor(name))
                    || (stream && layer_stream::streams(name, dtype, num_layers))
            },
        ))
    }

    /// 2026-10-17: The decoded csf scales of the whole layout (`expert_scales::defer_charge`).
    fn defer_charge(
        &self,
        config: &ModelConfig,
    ) -> Option<metrale_model_weights::weights::DeferCharge> {
        expert_scales::defer_charge(config)
    }

    /// 2026-10-09: The MLP plan slices every routed expert over TP under
    /// `--moe-expert-layout tp` (`glm5next_mlp::expert_tp`).
    fn slices_experts_over_tp(&self) -> bool {
        true
    }

    /// 2026-09-25: DSA, KDA and the MLP all shard under TP (see `glm5_next_load.rs`);
    /// routed experts are also split by EP (`local_expert_range`). 2026-10-08: The head and
    /// width splits come from `metrale_config::tp_split`, so they need not divide evenly.
    /// 2026-10-09: The KDA heads split in the published dense tier's channel unit
    /// (`glm5next_w4a16_dense::kda_channel_unit`), the one `KdaTpPlan::from_config` gets.
    fn tp_support(&self) -> metrale_config::TpSupport {
        metrale_config::TpSupport::Uneven {
            linear_channel_unit: crate::glm5next_w4a16_dense::kda_channel_unit(),
        }
    }

    fn load_layers(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        _layer_kv_dtypes: &[KvCacheDtype],
        lc: metrale_model_layers::load_context::LoadContext<'_>,
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        let skeleton = Glm5NextTextSkeleton::from_config(config)?;
        // 2026-10-17: The GLM levers, resolved once and stored on every layer built here.
        let levers = crate::glm5next_layer::GlmLevers::from_settings(lc.settings)?;
        levers.announce();
        crate::glm5next_pdl::apply(gpu, levers.pdl)?;
        levers.dense.check_tiers(
            crate::glm5next_w4a16_dense::enabled(),
            crate::glm5next_fp8_dense::enabled(),
        )?;
        let kda_cfg = Glm5NextKdaConfig::from_model_config(config);
        kda_cfg.validate()?;
        // 2026-10-16: Refuses a bad `METRALE_GLM_LAYER_STREAM` before any layer is built.
        layer_stream::enabled()?;
        // 2026-09-25: `gate_rank` is the row count of layer 0's `f_a_proj`; the
        // config has no such key. 2026-10-16: Deferred under `METRALE_GLM_LAYER_STREAM`.
        let gate_rank = {
            let n = qualify(0, "self_attn.f_a_proj.weight");
            let (_, shape) = expert_tp::meta(store, &n).with_context(|| {
                format!("glm5_next: {n} is needed to size the KDA gate bottleneck")
            })?;
            *shape.first().context("f_a_proj has no rows")?
        };
        let kda_plan = KdaTpPlan::from_config(
            config,
            gate_rank,
            crate::glm5next_w4a16_dense::kda_channel_unit(),
        )?;
        let mlp_cfg = Glm5NextMlpConfig::from_config(config)?;

        let kda_kernels = Glm5NextKdaKernels::resolve(gpu)?;
        let mlp_kernels = Glm5NextMlpKernels::resolve(gpu, levers)?;
        let mhc_kernels_probe = Glm5NextMhcKernels::resolve(gpu, levers.mhc)?;
        let rms_norm_k = gpu.kernel("rms_norm_vanilla", "rms_norm_vanilla")?;
        // 2026-09-25: Only a layer without mHC (the MTP layer) uses `add_k`, and it
        // fails there if the kernel is missing; `try_kernel` lets a target without
        // it still build the text stack.
        let add_k = metrale_model_layers::layers::try_kernel(gpu, "bf16_add", "bf16_add_inplace");

        // 2026-09-25: Every KDA layer shares one workspace (one `kda_cfg`). The
        // workspaces are sized for `verify_k` rows, the largest of the batched
        // GEMV's `DENSE_GEMV_BATCHM_MAX_M`, `PREFILL_ROWS` and `levers.prefill_rows`
        // (`METRALE_GLM_PREFILL_ROWS`).
        let verify_k = (metrale_model_layers::layers::ops::DENSE_GEMV_BATCHM_MAX_M as usize)
            .max(crate::glm5next_layer::PREFILL_ROWS)
            .max(levers.prefill_rows)
            // 2026-10-09: `METRALE_GLM_ROW_GROUP`, the batched decode/verify group width.
            .max(levers.row_group);
        let kda_ws = std::sync::Arc::new(crate::glm5next_kda::Glm5NextKdaWorkspace::new(
            gpu, &kda_cfg, verify_k,
        )?);

        // 2026-09-25: Unless `METRALE_GLM_MLP_WS_SHARED=0`, one MLP workspace serves
        // every layer; otherwise each layer allocates its own. Either way it is
        // allocated here, at load, before the KV pool is sized.
        let mlp_ws_bytes = crate::glm5next_mlp::forward::mlp_ws_total_bytes(
            &mlp_cfg,
            verify_k,
            levers.mlp.w4a4_grouped,
        );
        let shared_mlp_ws = if levers.mlp_ws_shared {
            tracing::info!(
                "GLM MLP workspace: SHARED, 1 x {:.1} MB for {} layers at {verify_k} rows \
                 (per-layer would be {:.1} MB)",
                mlp_ws_bytes as f64 / 1e6,
                skeleton.layers.len(),
                (mlp_ws_bytes * skeleton.layers.len()) as f64 / 1e6,
            );
            Some(std::sync::Arc::new(
                crate::glm5next_mlp::forward::Glm5NextMlpWorkspace::new(
                    gpu,
                    &mlp_cfg,
                    verify_k,
                    &levers.mlp,
                )?,
            ))
        } else {
            tracing::warn!(
                "GLM MLP workspace: PER-LAYER, {} x {:.1} MB at {verify_k} rows",
                skeleton.layers.len(),
                mlp_ws_bytes as f64 / 1e6,
            );
            None
        };

        let dsa_layers = skeleton
            .layers
            .iter()
            .filter(|l| l.mixer == Mixer::Dsa)
            .count();
        let dsa = dsa_stack::DsaStack::new(gpu, config, lc.settings, levers, verify_k, dsa_layers)?;
        let last = skeleton.layers.len() - 1;
        let mix_tokens = crate::glm5next_mhc::mhc_mix_max_tokens(levers.prefill_rows);
        let mut out: Vec<Box<dyn TransformerLayer>> = Vec::with_capacity(skeleton.layers.len());

        // 2026-09-25: The KV pool has `num_attention_layers()` slots, which counts
        // the sparse-attention layers only, so a DSA layer addresses it by its
        // ordinal among DSA layers, not by its model index.
        let mut attn_layer_idx = 0usize;
        // 2026-10-10: `METRALE_GLM_AR_L2_PREFETCH`: each layer's warmable ranges, planned after
        // the loop (`l2_warm::plan_next`).
        let l2 = levers
            .l2_warm_budget
            .map(|b| l2_warm::L2Warm::new(gpu, b).map(std::sync::Arc::new))
            .transpose()?;
        let mut l2_ranges: Vec<Vec<l2_warm::Range>> = Vec::new();
        let dump = crate::glm5next_layer_dump::config(lc.settings)?.map(std::sync::Arc::new);
        if let Some(d) = &dump {
            crate::glm5next_layer_dump::announce(d);
        }

        for sl in &skeleton.layers {
            let idx = sl.index;
            let t_layer = std::time::Instant::now();
            let src = LayerSource::collect(gpu, store, idx)
                .with_context(|| format!("glm5_next: collecting layer {idx}"))?;
            let t_collect = t_layer.elapsed();

            let mut mixer = match sl.mixer {
                Mixer::Kda => {
                    let sharded = KdaShardedSource::new(&src, &kda_plan)?;
                    let (w, _report) = bind_kda_weights(gpu, &kda_cfg, idx, &sharded)?;
                    Glm5NextMixer::Kda {
                        layer: Box::new(Glm5NextKdaLayer::new(
                            idx,
                            kda_cfg,
                            w,
                            kda_kernels,
                            levers,
                        )?),
                        ws: kda_ws.clone(),
                        cfg: kda_cfg,
                    }
                }
                Mixer::Dsa => {
                    attn_layer_idx += 1;
                    Glm5NextMixer::Dsa(Box::new(dsa.layer(gpu, &src, idx, attn_layer_idx - 1)?))
                }
            };

            let t_mixer = t_layer.elapsed();

            let load = |n: &str| src.f32(n);
            let mut mlp = match sl.mlp {
                Mlp::Dense => Glm5NextMlpSite::Dense(Box::new(dense_site::build_dense_site(
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
                    let expert = expert_tp::routed_expert_binder(gpu, store, idx, &mlp_cfg);
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

            if dense_tiers::any_below_declared(
                metrale_model_layers::layers::dense_quantization(),
                metrale_model_layers::layers::dsa_quantization(),
            ) {
                dense_tiers::register_dense_tiers(
                    gpu,
                    &mut mixer,
                    &mut mlp,
                    &mlp_cfg,
                    &levers.dense,
                    idx,
                )?;
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
                    attn: bind_mhc_site(
                        gpu,
                        &src,
                        "attn",
                        config.hc_mult,
                        config.hidden_size,
                        mix_tokens,
                    )?,
                    ffn: bind_mhc_site(
                        gpu,
                        &src,
                        "ffn",
                        config.hc_mult,
                        config.hidden_size,
                        mix_tokens,
                    )?,
                    hc_mult: config.hc_mult,
                    sinkhorn_iters: config.hc_sinkhorn_iters,
                    hc_eps: config.hc_eps,
                    fold_streams: if levers.mhc_post_fold {
                        gpu.alloc(config.hc_mult * config.hidden_size * 4)?
                    } else {
                        DevicePtr::NULL
                    },
                })
            } else {
                None
            };

            // 2026-10-12: `METRALE_GLM_SP_MHC=1`: the reduce-scatter workspace for the widest
            // prefill sub-chunk at this TP width (a no-op once reserved).
            crate::glm5next_sp_mhc::reserve(
                gpu,
                levers.prefill_rows,
                config.hidden_size,
                mlp_cfg.tp_world_size,
                &levers.mhc,
            )?;
            if l2.is_some() {
                anyhow::ensure!(idx == l2_ranges.len(), "layer {idx} out of order");
                l2_ranges.push(l2_warm::mixer_ranges(&mixer));
            }
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
                            gpu,
                            &mlp_cfg,
                            verify_k,
                            &levers.mlp,
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
                    Mixer::Dsa => dsa.plan.needs_output_all_reduce(),
                },
                is_first: idx == 0,
                is_last: idx == last,
                l2_warm: l2.clone(),
                keep_slot_graphs: lc.levers.glm_keep_slot_graphs,
                dump: dump.clone(),
                levers,
            }));
        }
        if let Some(l2) = &l2 {
            let plan = l2_warm::plan_next(&l2_ranges, l2.budget());
            l2_warm::check_plan(&plan, out.len())?;
            let warmed: usize = plan.iter().flatten().map(|r| r.1).sum();
            tracing::warn!(
                "glm5_next {}: {:.1} MB of next-layer W4A16 weights warmed into L2 per step",
                l2_warm::L2_WARM_ENV,
                warmed as f64 / 1e6
            );
            l2.set_plan(plan)?;
        }
        if crate::glm5next_fp8_dense::enabled() {
            let (count, bytes) = crate::glm5next_fp8_dense::registered();
            tracing::warn!(
                "glm5_next --dense-quantization {} --dsa-quantization {}: {count} BF16 dense \
                 projections also held as FP8 per-channel ({:.2} GB) and decoded W8A8, BELOW the \
                 checkpoint's declared BF16; the router, the indexer's wq_b and weights_proj \
                 stay BF16",
                metrale_model_layers::layers::dense_quantization().name(),
                metrale_model_layers::layers::dsa_quantization().name(),
                bytes as f64 / 1e9
            );
        }
        if crate::glm5next_w4a16_dense::registry_in_use() {
            let (count, nvfp4, bf16) = crate::glm5next_w4a16_dense::registered();
            tracing::warn!(
                "glm5_next --dense-quantization {} --dsa-quantization {}: {count} BF16 dense \
                 projections replaced by NVFP4 ({:.2} GB; {:.2} GB of BF16 freed) and decoded \
                 W4A16, FURTHER BELOW the checkpoint's declared BF16",
                metrale_model_layers::layers::dense_quantization().name(),
                metrale_model_layers::layers::dsa_quantization().name(),
                nvfp4 as f64 / 1e9,
                bf16 as f64 / 1e9
            );
        }
        crate::glm5next_dequant_persist::log_readback(levers.dense.persist);
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
        gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        fp8_head::head_weight(gpu, store)
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

    /// 2026-10-18: The DSA indexer keeps its rows in the KV blocks, so it takes the pooled FP8
    /// rows (`IndexerRowFormat::Fp8`).
    fn indexer_cache_dtypes(&self) -> &'static [metrale_config::IndexerCacheDtype] {
        &[metrale_config::IndexerCacheDtype::Fp8]
    }

    /// 2026-10-16: The FP8 indexer cache resizes the V block; see `IndexerRowFormat`.
    fn kv_v_block_bytes(
        &self,
        config: &ModelConfig,
        block_size: usize,
        settings: &metrale_settings::InferenceServerSettings,
    ) -> Result<Option<usize>> {
        crate::glm5next_dsa::fp8_rows::IndexerRowFormat::text_stack_v_block_bytes(
            config,
            block_size,
            crate::glm5next_dsa::paged::text_stack_indexer_cache(settings),
            settings.kernels.glm_dsa_pool_keys,
        )
    }

    /// 2026-09-25: Free what the binders re-uploaded or quantised. 2026-10-17: See
    /// `prune::prune_after_load`.
    fn prune_after_load(
        &self,
        store: &mut WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        _lc: metrale_model_layers::load_context::LoadContext<'_>,
    ) -> Result<()> {
        prune::prune_after_load(store, config, gpu)
    }
}
