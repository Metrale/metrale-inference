// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Native GPT-OSS decode with explicit experimental chunk prefill.
//! No factory registration. Host expert-ID readback vetoes CUDA graphs; this is
//! not a throughput-qualified route. Weight pointers stay owned by the model store.
use super::GptOssLayerWeights;
use anyhow::{Context, Result, ensure};
use metrale_cache::kv_cache::{KvCacheDtype, PagedKvCache};
use metrale_config::{LayerType, ModelConfig};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::circuit_exec::CircuitBindings;
use metrale_model_layers::{layer::*, layers::ops, weight_map::DenseWeight};
mod diagnostics;
pub use diagnostics::DiagnosticTensor;
mod expert_plan;
mod forward;
mod prefill;
mod prefill_experts;
mod prefill_scratch;
mod serving_prefill;
mod tc_plan;
mod tc_scratch;
pub use prefill_scratch::PrefillScratch;
mod kernels;
mod state;
mod weights;
use kernels::Kernels;
use state::State;
use weights::Weights;

/// 2026-10-07: Native BF16 attention and packed-MXFP4 MoE, eager C1 only.
pub struct GptOssLayer {
    weights: Weights,
    kernels: Kernels,
    yarn: ops::GptOssYarn,
    index: usize,
    window: u32,
    max_positions: usize,
    eps: f32,
    chunk_prefill_tokens: Option<usize>,
}
impl GptOssLayer {
    /// 2026-10-07: Snapshot validated pointers; keep the source store alive.
    pub fn new(
        bound: &GptOssLayerWeights<'_>,
        config: &ModelConfig,
        index: usize,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        ensure!(
            config.hidden_size == 2880
                && config.head_dim == 64
                && config.num_attention_heads == 64
                && config.num_key_value_heads == 8
                && config.num_experts == 32
                && config.moe_intermediate_size == 2880
                && config.num_hidden_layers == 24
                && config.num_experts_per_tok == 4
                && config.sliding_window == 128
                && config.swiglu_limit == 7.0
                && config.rms_norm_eps == 1e-5,
            "GPT runtime requires pinned 20B geometry"
        );
        ensure!(
            index < 24 && config.tp_world_size.max(1) == 1 && config.tp_rank == 0,
            "GPT runtime requires C1 single-device geometry"
        );
        ensure!(
            config.max_position_embeddings == 131072,
            "GPT runtime requires pinned context limit"
        );
        ensure!(
            config.layer_types.get(index) == Some(&bound.kind),
            "GPT layer kind differs from config"
        );
        Ok(Self {
            weights: Weights::from(bound)?,
            kernels: Kernels::new(gpu)?,
            yarn: ops::GptOssYarn::from_config(config)?,
            index,
            window: if bound.kind == LayerType::SlidingAttention {
                128
            } else {
                0
            },
            max_positions: config.max_position_embeddings,
            eps: config.rms_norm_eps as f32,
            chunk_prefill_tokens: None,
        })
    }
}
impl GptOssLayer {
    /// 2026-10-07: Run one token without scheduler construction. The caller owns
    /// a complete BF16 hidden row and retains the checkpoint store. This eager
    /// route performs host expert-ID readback and is not graph-safe.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_token(
        &self,
        hidden: DevicePtr,
        state: &mut dyn LayerState,
        cache: &mut PagedKvCache,
        position: usize,
        blocks: &mut Vec<u32>,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        ensure!(position < self.max_positions, "GPT context exhausted");
        ensure!(
            self.index < cache.num_layers(),
            "GPT cache has too few layers"
        );
        ensure!(
            cache.dtype_for_layer(self.index) == KvCacheDtype::Bf16,
            "GPT sink attention requires BF16 KV"
        );
        let state = state
            .as_any_mut()
            .downcast_mut::<State>()
            .context("GPT layer state type")?;
        ensure!(!state.allocation.is_null(), "GPT layer state was released");
        ensure!(
            cache.config().dims_for_layer(self.index) == (8, 64),
            "GPT cache geometry must be 8 KV heads of width 64"
        );
        ensure!(
            cache.config().cache_blocks_per_seq.is_none(),
            "GPT eager route does not support KV window eviction"
        );
        ensure!(
            !hidden.is_null() && hidden.0.is_multiple_of(4),
            "GPT hidden row is null or misaligned"
        );
        // 2026-10-07: A state owns one contiguous cache prefix. Reset/rewind and
        // restored-cache adoption require a fresh state; never read unwritten KV.
        ensure!(!state.failed, "GPT state failed; allocate a fresh state");
        ensure!(
            position == state.next_position,
            "GPT requires sequential positions; reset/rewind unsupported"
        );
        ensure!(
            blocks.starts_with(&state.prefix_blocks),
            "GPT cache prefix mapping changed"
        );
        // 2026-10-06: Distinct logical blocks must never alias and overwrite earlier tokens.
        let unique_blocks: std::collections::HashSet<_> = blocks.iter().collect();
        ensure!(
            unique_blocks.len() == blocks.len(),
            "GPT physical blocks alias"
        );
        let pools = (cache.k_pool_ptr(self.index), cache.v_pool_ptr(self.index));
        ensure!(
            state.cache_pools.is_none_or(|previous| previous == pools),
            "GPT cache pool changed"
        );
        state.failed = true;
        self.forward(hidden, state, cache, position, blocks, gpu, stream)?;
        state.next_position += 1;
        state.prefix_blocks.clone_from(blocks);
        state.cache_pools = Some(pools);
        state.failed = false;
        Ok(())
    }
}
impl LayerCapabilities for GptOssLayer {
    fn decode_graph_unsupported(&self) -> bool {
        true
    }
    fn decode_multi_seq_unsupported(&self) -> bool {
        true
    }
    fn decode_verify_multi_unsupported(&self) -> bool {
        true
    }
}
impl LayerWeightSetup for GptOssLayer {}
impl LayerWriteOnAccept for GptOssLayer {}
impl LayerGraphHooks for GptOssLayer {}
impl LayerAuxState for GptOssLayer {}
impl LayerSplitPrefill for GptOssLayer {}
impl CircuitBindings for GptOssLayer {}
impl TransformerLayer for GptOssLayer {
    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn LayerState>> {
        let state = State::new(gpu, self.max_positions)?;
        if let Err(error) = ops::gpt_oss_yarn_frequencies(
            gpu,
            self.kernels.frequencies,
            state.frequencies,
            &self.yarn,
            gpu.default_stream(),
        ) {
            let _ = gpu.free(state.allocation);
            return Err(error);
        }
        if let Err(error) = gpu.synchronize(gpu.default_stream()) {
            let _ = gpu.free(state.allocation);
            return Err(error);
        }
        Ok(Box::new(state))
    }
    fn release_state(&self, state: &mut dyn LayerState, gpu: &dyn GpuBackend) -> Result<()> {
        let s = state
            .as_any_mut()
            .downcast_mut::<State>()
            .context("GPT layer state type")?;
        if let Some(scratch) = &mut s.chunk_scratch {
            scratch.release_bound(gpu)?;
        }
        if !s.allocation.is_null() {
            gpu.free(s.allocation)?;
            s.allocation = DevicePtr::NULL;
        }
        Ok(())
    }
    fn prefill(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        cache: &mut PagedKvCache,
        start: usize,
        blocks: &mut Vec<u32>,
        disk_ids: &mut Vec<u32>,
        disk_offloaded: &mut Vec<u32>,
        kv_write_start: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.serving_prefill(
            hidden,
            residual,
            num_tokens,
            state,
            cache,
            start,
            blocks,
            disk_ids,
            disk_offloaded,
            kv_write_start,
            ctx,
            stream,
        )
    }
    fn decode(
        &self,
        hidden: DevicePtr,
        _residual: DevicePtr,
        state: &mut dyn LayerState,
        cache: &mut PagedKvCache,
        position: usize,
        blocks: &mut Vec<u32>,
        disk_ids: &mut Vec<u32>,
        disk_offloaded: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            !ctx.graph_capture,
            "GPT correctness route does not support CUDA graphs"
        );
        ensure!(
            disk_ids.is_empty() && disk_offloaded.iter().all(|n| *n == 0),
            "GPT runtime does not support disk KV swapping"
        );
        self.forward_token(hidden, state, cache, position, blocks, ctx.gpu, stream)
    }
}

#[cfg(test)]
mod tests;
