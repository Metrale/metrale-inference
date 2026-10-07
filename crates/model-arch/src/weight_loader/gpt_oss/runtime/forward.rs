// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Explicit staged BF16 boundaries and sink/window cache policy.
use super::*;
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use weights::{Linear, Packed};
impl GptOssLayer {
    #[allow(clippy::too_many_arguments)]
    fn linear(
        &self,
        gpu: &dyn GpuBackend,
        w: Linear,
        input: DevicePtr,
        out: DevicePtr,
        accum: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        ops::dense_gemv(
            gpu,
            self.kernels.gemv,
            input,
            &DenseWeight { weight: w.weight },
            accum,
            w.rows,
            w.cols,
            stream,
        )?;
        ops::projection_bias_bf16(
            gpu,
            self.kernels.projection_bias,
            accum,
            w.bias,
            out,
            1,
            w.rows,
            stream,
        )
    }
    fn packed(
        &self,
        gpu: &dyn GpuBackend,
        w: Packed,
        input: DevicePtr,
        out: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, self.kernels.mxfp4)
            .grid([w.rows.div_ceil(4), 1, 1])
            .block([128, 1, 1])
            .arg_ptr(w.blocks)
            .arg_ptr(w.scales)
            .arg_ptr(input)
            .arg_ptr(out)
            .arg_u32(w.rows)
            .arg_u32(w.cols)
            .launch(stream)
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward(
        &self,
        hidden: DevicePtr,
        s: &mut State,
        cache: &mut PagedKvCache,
        position: usize,
        blocks: &mut Vec<u32>,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let block_size = cache.block_size();
        ensure!(
            block_size > 0 && u32::try_from(block_size).is_ok(),
            "GPT cache block size outside kernel ABI"
        );
        let logical = position / block_size;
        ensure!(logical < s.max_blocks, "GPT block table exceeds allocation");
        while blocks.len() <= logical {
            blocks.push(cache.alloc_block()?);
        }
        ensure!(
            blocks.iter().all(|&b| (b as usize) < cache.num_blocks()),
            "GPT physical block outside pool"
        );
        let slot = (blocks[logical] as usize * block_size + position % block_size) as i64;
        let table: Vec<u8> = blocks[..=logical]
            .iter()
            .flat_map(|n| n.to_le_bytes())
            .collect();
        // 2026-10-07: Build scalar metadata from the argument, never reuse prefill row0.
        gpu.copy_h2d_async(&table, s.table, stream)?;
        gpu.copy_h2d_async(&(position as u32).to_le_bytes(), s.position, stream)?;
        gpu.copy_h2d_async(&((position + 1) as u32).to_le_bytes(), s.length, stream)?;
        gpu.copy_h2d_async(&slot.to_le_bytes(), s.slot, stream)?;
        ops::rms_norm(
            gpu,
            self.kernels.norm,
            hidden,
            &DenseWeight {
                weight: self.weights.input_norm,
            },
            s.norm,
            1,
            2880,
            self.eps,
            stream,
        )?;
        self.linear(gpu, self.weights.q, s.norm, s.q, s.accum, stream)?;
        self.linear(gpu, self.weights.k, s.norm, s.k, s.accum, stream)?;
        self.linear(gpu, self.weights.v, s.norm, s.v, s.accum, stream)?;
        ops::gpt_oss_rope_bf16(
            gpu,
            self.kernels.rope,
            [s.q, s.k, s.position, s.frequencies],
            1,
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
            s.slot,
            1,
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
                s.table,
                s.length,
                self.weights.sinks,
            ],
            &ops::PagedSinkGeometry {
                sequences: 1,
                max_blocks: (logical + 1) as u32,
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
        self.linear(gpu, self.weights.o, s.attn, s.projection, s.accum, stream)?;
        ops::residual_add(
            gpu,
            self.kernels.residual,
            hidden,
            s.projection,
            2880,
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
            1,
            2880,
            self.eps,
            stream,
        )?;
        self.linear(gpu, self.weights.router, s.norm, s.logits, s.accum, stream)?;
        ops::gpt_oss_router_bf16(
            gpu,
            self.kernels.router,
            s.logits,
            s.ids,
            s.scores,
            1,
            stream,
        )?;
        let mut bytes = [0u8; 16];
        gpu.copy_d2h_on_stream(s.ids, &mut bytes, stream)?;
        let ids: Vec<usize> = bytes
            .chunks_exact(4)
            .map(|x| u32::from_le_bytes(x.try_into().expect("four bytes")) as usize)
            .collect();
        ensure!(
            ids.iter().all(|&i| i < 32),
            "GPT router produced invalid expert"
        );
        ensure!(
            ids.iter().enumerate().all(|(i, id)| !ids[..i].contains(id)),
            "GPT router produced duplicate experts"
        );
        for (slot, &expert) in ids.iter().enumerate() {
            self.packed(gpu, self.weights.gate_up[expert], s.norm, s.gate_up, stream)?;
            ops::gpt_oss_expert_bias_bf16(
                gpu,
                self.kernels.bias,
                s.gate_up,
                self.weights.gate_up_bias.offset(expert * 5760 * 2),
                1,
                5760,
                stream,
            )?;
            ops::gpt_oss_swiglu_bf16(
                gpu,
                self.kernels.activation,
                s.gate_up,
                s.activation,
                2880,
                stream,
            )?;
            let out = s.selected.offset(slot * 2880 * 2);
            self.packed(gpu, self.weights.down[expert], s.activation, out, stream)?;
            ops::gpt_oss_expert_bias_bf16(
                gpu,
                self.kernels.bias,
                out,
                self.weights.down_bias.offset(expert * 2880 * 2),
                1,
                2880,
                stream,
            )?;
        }
        ops::gpt_oss_expert_reduce_bf16(
            gpu,
            self.kernels.reduce,
            s.selected,
            s.scores,
            s.ids,
            s.moe,
            1,
            2880,
            stream,
        )?;
        ops::residual_add(gpu, self.kernels.residual, hidden, s.moe, 2880, stream)
    }
}
