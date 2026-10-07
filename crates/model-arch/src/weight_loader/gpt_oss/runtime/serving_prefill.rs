// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Explicit chunk admission; the unselected route retains scalar decode.
use super::*;
impl GptOssLayer {
    /// 2026-10-07: Resolve the extra kernel before admitting a chunk-enabled loader.
    pub(crate) fn set_chunk_prefill(&mut self, enabled: bool, gpu: &dyn GpuBackend) -> Result<()> {
        if enabled {
            gpu.kernel("dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm_fp32out")?;
            gpu.kernel("gpt_oss_mxfp4_gemv", "gpt_oss_mxfp4_selected_tokens_bf16")?;
            gpu.kernel("gpt_oss_expert_ops", "gpt_oss_selected_bias_tokens_bf16")?;
        }
        self.chunk_prefill = enabled;
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn serving_prefill(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        count: usize,
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
        ensure!(!ctx.graph_capture, "GPT prefill rejects graph capture");
        ensure!(
            kv_write_start == 0,
            "GPT prefill cannot adopt cached prefix tokens"
        );
        ensure!(
            disk_ids.is_empty() && disk_offloaded.iter().all(|n| *n == 0),
            "GPT prefill rejects disk KV swapping"
        );
        let end = start
            .checked_add(count)
            .context("GPT prefill position overflow")?;
        let bytes = count
            .checked_mul(5760)
            .context("GPT prefill extent overflow")?;
        ensure!(
            end <= self.max_positions
                && !hidden.is_null()
                && hidden.0.checked_add(bytes as u64).is_some(),
            "GPT prefill extent/context bound"
        );
        if !self.chunk_prefill {
            for t in 0..count {
                self.decode(
                    hidden.offset(t * 5760),
                    residual.offset(t * 5760),
                    state,
                    cache,
                    start + t,
                    blocks,
                    disk_ids,
                    disk_offloaded,
                    ctx,
                    stream,
                )?;
            }
            return Ok(());
        }
        if count == 0 {
            return Ok(());
        }
        let s = state
            .as_any_mut()
            .downcast_mut::<State>()
            .context("GPT prefill state type")?;
        ensure!(
            !s.allocation.is_null() && !s.failed && s.next_position == start,
            "GPT prefill state is not live sequential prefix"
        );
        ensure!(cache.block_size() > 0, "GPT prefill zero block size");
        if s.chunk_scratch.is_none() {
            // 2026-10-07: Persistent per-sequence storage, sized from actual page geometry.
            match PrefillScratch::new(ctx.gpu, 16, self.max_positions.div_ceil(cache.block_size()))
            {
                Ok(scratch) => s.chunk_scratch = Some(scratch),
                Err(error) => {
                    s.failed = true;
                    return Err(error);
                }
            }
        }
        let mut scratch = s
            .chunk_scratch
            .take()
            .context("GPT prefill scratch missing")?;
        let result = (|| {
            for offset in (0..count).step_by(16) {
                self.forward_chunk(
                    hidden.offset(offset * 5760),
                    state,
                    cache,
                    start + offset,
                    (count - offset).min(16),
                    blocks,
                    &mut scratch,
                    ctx.gpu,
                    stream,
                )?;
            }
            Ok(())
        })();
        // 2026-10-07: Retain scratch even on failure so teardown drains queued work.
        state
            .as_any_mut()
            .downcast_mut::<State>()
            .context("GPT prefill state changed")?
            .chunk_scratch = Some(scratch);
        result
    }
}
