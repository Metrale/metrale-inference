// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: What the text stack's DSA layers share, built once per load ([`DsaStack::new`]),
//! and the build of one DSA layer from it ([`DsaStack::layer`]). Moved out of `loader.rs`
//! unchanged (that file passed 500 lines) but for the levers resolved at load.
//!
//! Owner: model-arch weight loader.
//! Invariants: every DSA layer of one load gets the same kernels, levers and (when
//! `levers.dsa_ws_shared`) workspace; `attn_layer_idx` is the layer's ordinal among DSA layers.

use std::sync::Arc;

use super::*;
use crate::glm5next_dsa::attend::Glm5NextDsaDecodeKernel;
use crate::glm5next_dsa::fp8_rows::IndexerRowFormat;
use crate::glm5next_dsa::layer::Glm5NextDsaWorkspace;
use crate::glm5next_dsa::levers::DsaOwnedLevers;
use crate::glm5next_dsa::paged::IndexerCache;
use crate::glm5next_dsa::tp::DsaTpPlan;
use crate::glm5next_layer::GlmLevers;

/// 2026-10-17: The per-load parts of the text stack's DSA layers.
pub(super) struct DsaStack {
    pub(super) cfg: Glm5NextDsaConfig,
    pub(super) plan: DsaTpPlan,
    layer_kernels: Glm5NextDsaLayerKernels,
    kernels: Glm5NextDsaKernels,
    decode_kernel: Glm5NextDsaDecodeKernel,
    owned: Arc<DsaOwnedLevers>,
    shared_ws: Option<Arc<Glm5NextDsaWorkspace>>,
    indexer_cache: IndexerCache,
    indexer_format: IndexerRowFormat,
    verify_k: usize,
    rms_eps: f32,
    levers: GlmLevers,
    absorb: crate::mla_absorb::AbsorbThreads,
}

impl DsaStack {
    /// 2026-10-17: Resolve the DSA kernels and levers of one load, and the shared workspace
    /// for `verify_k` rows unless `METRALE_GLM_DSA_WS_SHARED=0`; `dsa_layers` is only logged.
    pub(super) fn new(
        gpu: &dyn GpuBackend,
        config: &ModelConfig,
        settings: &metrale_settings::InferenceServerSettings,
        levers: GlmLevers,
        verify_k: usize,
        dsa_layers: usize,
    ) -> Result<Self> {
        let cfg = Glm5NextDsaConfig::from_config(config)?;
        let kernels = Glm5NextDsaKernels::resolve(gpu)?;
        let indexer_cache = crate::glm5next_dsa::paged::text_stack_indexer_cache(settings);
        let indexer_format = IndexerRowFormat::text_stack(config, &kernels.fp8, indexer_cache)?;
        let owned = Arc::new(DsaOwnedLevers::from_settings(settings)?);
        owned.announce();
        // 2026-10-12: One DSA workspace for the stack unless `METRALE_GLM_DSA_WS_SHARED=0`
        // (`levers.dsa_ws_shared`); sized like the KDA and MLP ones, for `verify_k` rows.
        let shared_ws = if levers.dsa_ws_shared {
            tracing::info!(
                "GLM DSA workspace: SHARED, 1 for {dsa_layers} DSA layers at {verify_k} rows"
            );
            Some(Arc::new(Glm5NextDsaWorkspace::new(
                gpu, &cfg, verify_k, &levers,
            )?))
        } else {
            None
        };
        Ok(Self {
            plan: DsaTpPlan::from_config(config, &cfg)?,
            layer_kernels: Glm5NextDsaLayerKernels::resolve(gpu)?,
            decode_kernel: Glm5NextDsaDecodeKernel::resolve(gpu, levers.dsa.decode_hb)?,
            cfg,
            kernels,
            owned,
            shared_ws,
            indexer_cache,
            indexer_format,
            verify_k,
            rms_eps: config.rms_norm_eps as f32,
            levers,
            absorb: crate::mla_absorb::AbsorbThreads::from_settings(settings),
        })
    }

    /// 2026-10-17: The DSA layer `idx` (`attn_layer_idx` among the DSA layers) from `src`.
    pub(super) fn layer(
        &self,
        gpu: &dyn GpuBackend,
        src: &LayerSource,
        idx: usize,
        attn_layer_idx: usize,
    ) -> Result<Glm5NextDsaLayer> {
        let load = |n: &str| src.f32(n);
        let levers = self.levers;
        Ok(Glm5NextDsaLayer {
            persist_bt: !levers.dsa_alloc_per_step,
            levers,
            cfg: self.cfg,
            weights: build_dsa_weights(gpu, &self.cfg, &self.plan, &load, self.absorb)?,
            kernels: self.layer_kernels,
            select_kernels: self.kernels,
            decode_kernel: self.decode_kernel,
            dsa_owned: self.owned.clone(),
            workspace: match &self.shared_ws {
                Some(ws) => ws.clone(),
                None => Arc::new(Glm5NextDsaWorkspace::new(
                    gpu,
                    &self.cfg,
                    self.verify_k,
                    &levers,
                )?),
            },
            layer_idx: idx,
            attn_layer_idx,
            rms_eps: self.rms_eps,
            kv_scale: 1.0,
            indexer_cache: self.indexer_cache,
            indexer_format: self.indexer_format,
        })
    }
}
