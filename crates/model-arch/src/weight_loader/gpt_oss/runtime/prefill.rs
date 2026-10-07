// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Bounded C1 chunks used by explicit experimental serving admission.
use super::prefill_scratch::PrefillScratch;
use super::*;
use weights::Linear;
impl GptOssLayer {
    /// 2026-10-07: One-sequence chunk; scalar prefill remains the default serving policy.
    /// hidden holds rows contiguous BF16 vectors. Scratch is exclusively borrowed.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_chunk(
        &self,
        hidden: DevicePtr,
        state: &mut dyn LayerState,
        cache: &mut PagedKvCache,
        start: usize,
        rows: usize,
        blocks: &mut Vec<u32>,
        scratch: &mut PrefillScratch,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        ensure!((1..=16).contains(&rows), "GPT chunk rows outside1..=16");
        let end = start
            .checked_add(rows)
            .context("GPT chunk position overflow")?;
        ensure!(
            end <= self.max_positions && self.index < cache.num_layers(),
            "GPT chunk context/layer bound"
        );
        ensure!(
            cache.dtype_for_layer(self.index) == KvCacheDtype::Bf16
                && cache.config().dims_for_layer(self.index) == (8, 64),
            "GPT chunk requires BF16 8x64 cache"
        );
        ensure!(
            cache.config().cache_blocks_per_seq.is_none(),
            "GPT chunk rejects cache eviction"
        );
        ensure!(
            !hidden.is_null()
                && hidden.0.is_multiple_of(4)
                && hidden.0.checked_add((rows * 5760) as u64).is_some(),
            "GPT chunk hidden pointer"
        );
        let size = cache.block_size();
        ensure!(
            size > 0 && u32::try_from(size).is_ok(),
            "GPT chunk block size"
        );
        let needed = end.div_ceil(size);
        scratch.admit(rows, needed, stream)?;
        let s = state
            .as_any_mut()
            .downcast_mut::<State>()
            .context("GPT chunk state type")?;
        ensure!(
            !s.allocation.is_null() && !s.failed && s.next_position == start,
            "GPT chunk state is not live sequential prefix"
        );
        ensure!(
            needed <= s.max_blocks && blocks.starts_with(&s.prefix_blocks),
            "GPT chunk prefix changed"
        );
        validate_pages(blocks, cache.num_blocks())?;
        let pools = (cache.k_pool_ptr(self.index), cache.v_pool_ptr(self.index));
        ensure!(
            s.cache_pools.is_none_or(|v| v == pools),
            "GPT chunk cache pool changed"
        );
        s.failed = true;
        while blocks.len() < needed {
            blocks.push(cache.alloc_block()?);
        }
        validate_pages(blocks, cache.num_blocks())?;
        self.chunk_compute(hidden, s, cache, start, rows, blocks, scratch, gpu, stream)?;
        s.next_position = end;
        s.prefix_blocks.clone_from(blocks);
        s.cache_pools = Some(pools);
        s.failed = false;
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    fn chunk_linear(
        &self,
        gpu: &dyn GpuBackend,
        kernel: metrale_gpu_runtime::gpu::KernelHandle,
        w: Linear,
        input: DevicePtr,
        output: DevicePtr,
        s: &PrefillScratch,
        rows: u32,
        stream: u64,
    ) -> Result<()> {
        ops::dense_gemv_batchm_fp32(
            gpu, kernel, input, w.weight, s.accum, rows, w.rows, w.cols, w.rows, stream,
        )?;
        ops::projection_bias_bf16(
            gpu,
            self.kernels.projection_bias,
            s.accum,
            w.bias,
            output,
            rows,
            w.rows,
            stream,
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn chunk_compute(
        &self,
        hidden: DevicePtr,
        state: &mut State,
        cache: &PagedKvCache,
        start: usize,
        rows: usize,
        blocks: &[u32],
        s: &PrefillScratch,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let m = rows as u32;
        let block_size = cache.block_size();
        let needed = (start + rows).div_ceil(block_size);
        let kernel = gpu.kernel("dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm_fp32out")?;
        let positions: Vec<u8> = (start..start + rows)
            .flat_map(|p| (p as u32).to_le_bytes())
            .collect();
        let lengths: Vec<u8> = (start + 1..=start + rows)
            .flat_map(|p| (p as u32).to_le_bytes())
            .collect();
        let slots: Vec<u8> = (start..start + rows)
            .flat_map(|p| {
                ((blocks[p / block_size] as usize * block_size + p % block_size) as i64)
                    .to_le_bytes()
            })
            .collect();
        // 2026-10-07: Each query intentionally shares the same unique page prefix.
        // Per-query causal lengths exclude later KV even though the chunk writes it first.
        let tables: Vec<u8> = (0..rows)
            .flat_map(|_| blocks[..needed].iter().flat_map(|b| b.to_le_bytes()))
            .collect();
        for (bytes, ptr) in [
            (&positions, s.positions),
            (&lengths, s.lengths),
            (&slots, s.slots),
            (&tables, s.tables),
        ] {
            gpu.copy_h2d_async(bytes, ptr, stream)?;
        }
        ops::rms_norm(
            gpu,
            self.kernels.norm,
            hidden,
            &DenseWeight {
                weight: self.weights.input_norm,
            },
            s.norm,
            m,
            2880,
            self.eps,
            stream,
        )?;
        self.chunk_linear(gpu, kernel, self.weights.q, s.norm, s.q, s, m, stream)?;
        self.chunk_linear(gpu, kernel, self.weights.k, s.norm, s.k, s, m, stream)?;
        self.chunk_linear(gpu, kernel, self.weights.v, s.norm, s.v, s, m, stream)?;
        ops::gpt_oss_rope_bf16(
            gpu,
            self.kernels.rope,
            [s.q, s.k, s.positions, state.frequencies],
            m,
            64,
            8,
            &self.yarn,
            stream,
        )?;
        ops::reshape_and_cache(
            gpu,
            self.kernels.cache,
            s.k,
            s.v,
            cache.k_pool_ptr(self.index),
            cache.v_pool_ptr(self.index),
            s.slots,
            m,
            8,
            64,
            block_size as u32,
            512,
            512,
            cache.block_stride_bytes_for_layer(self.index) as u64,
            stream,
        )?;
        ops::paged_decode_attn_bf16_sink(
            gpu,
            self.kernels.attention,
            [
                s.q,
                cache.k_pool_ptr(self.index),
                cache.v_pool_ptr(self.index),
                s.attn,
                s.tables,
                s.lengths,
                self.weights.sinks,
            ],
            &ops::PagedSinkGeometry {
                sequences: m,
                max_blocks: needed as u32,
                q_heads: 64,
                kv_heads: 8,
                head_dim: 64,
                block_size: block_size as u32,
                scale: 0.125,
                q_stride: 4096,
                window: self.window,
            },
            stream,
        )?;
        self.chunk_linear(
            gpu,
            kernel,
            self.weights.o,
            s.attn,
            s.projection,
            s,
            m,
            stream,
        )?;
        ops::residual_add(
            gpu,
            self.kernels.residual,
            hidden,
            s.projection,
            m * 2880,
            stream,
        )?;
        ops::rms_norm(
            gpu,
            self.kernels.norm,
            hidden,
            &DenseWeight {
                weight: self.weights.post_norm,
            },
            s.norm,
            m,
            2880,
            self.eps,
            stream,
        )?;
        self.chunk_linear(
            gpu,
            kernel,
            self.weights.router,
            s.norm,
            s.logits,
            s,
            m,
            stream,
        )?;
        ops::gpt_oss_router_bf16(
            gpu,
            self.kernels.router,
            s.logits,
            s.ids,
            s.scores,
            m,
            stream,
        )?;
        let mut ids = vec![0u8; rows * 16];
        gpu.copy_d2h_on_stream(s.ids, &mut ids, stream)?;
        let ids: Vec<_> = ids
            .chunks_exact(4)
            .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
            .collect();
        let plan = super::expert_plan::ExpertTokenPlan::new(&ids)?;
        if Self::uses_expert_reuse(rows as u32) {
            gpu.copy_h2d_async(&plan.bytes(), s.expert_plan, stream)?;
        }
        self.chunk_experts(hidden, rows as u32, s, gpu, stream)?;
        Ok(())
    }
}
fn validate_pages(blocks: &[u32], pool: usize) -> Result<()> {
    let unique: std::collections::HashSet<_> = blocks.iter().collect();
    ensure!(
        unique.len() == blocks.len() && blocks.iter().all(|&b| (b as usize) < pool),
        "GPT chunk physical pages alias or exceed pool"
    );
    Ok(())
}
#[cfg(test)]
mod tests {
    #[test]
    fn pages_are_unique_within_sequence_not_across_query_rows() {
        assert!(super::validate_pages(&[3, 1, 7], 8).is_ok());
        assert!(super::validate_pages(&[3, 1, 3], 8).is_err());
        assert!(super::validate_pages(&[3, 8], 8).is_err());
    }
}
