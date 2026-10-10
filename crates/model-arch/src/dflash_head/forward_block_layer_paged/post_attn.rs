// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The post-attention call of the paged drafter layer: o_proj, the attention
//! residual, post_attention_layernorm and the SwiGLU MLP, with the DFlash2 conv applications.
//!
//! Owner: model-arch (DFlash drafter).
//! Invariants: none beyond the types.

use anyhow::Result;

use super::PagedLayerArgs;
use crate::dflash_head::{BlockDiffusionDraftHead, DflashLayer};
use metrale_model_layers::layer::ForwardContext;

impl BlockDiffusionDraftHead {
    /// 2026-09-26: The body of `forward_block_layer_post_attn`.
    pub(super) fn post_attn_layer(
        &self,
        layer: &DflashLayer,
        args: &PagedLayerArgs,
        ctx: &ForwardContext,
    ) -> Result<()> {
        use metrale_model_layers::layers::ops;

        let PagedLayerArgs {
            h,
            q_dim,
            inter,
            stream,
            ..
        } = *args;
        let gpu = ctx.gpu;
        // 2026-09-25: All rows: o_proj and the MLP are weight-bearing, so they span every
        // sequence in the batch.
        let g = self.block_g() as u32 * args.n_seq.max(1);

        // 2026-09-25: The same FP8/BF16 GEMM choice as `forward_block_layer_pre_attn`.
        let use_fp8 = matches!(
            self.quant,
            crate::dflash_head::DflashQuantization::Fp8Weights
        );
        let gemm_swap = |w_bf16: &metrale_model_layers::weight_map::DenseWeight,
                         w_fp8: &Option<metrale_model_layers::weight_map::Fp8DenseWeight>,
                         src: metrale_gpu_runtime::gpu::DevicePtr,
                         dst: metrale_gpu_runtime::gpu::DevicePtr,
                         n_out: u32,
                         k_in: u32|
         -> Result<()> {
            if use_fp8 && let Some(fp8) = w_fp8 {
                return self.fp8_rows(gpu, src, fp8, dst, g, n_out, k_in, stream);
            }
            ops::dense_gemm_bf16_pipelined(
                gpu,
                self.kernels.dense_gemm_pipelined,
                src,
                w_bf16,
                dst,
                g,
                n_out,
                k_in,
                stream,
            )
        };

        // 2026-09-25: stream_acc = o_proj(attn_out). stream_buf still holds the layer
        // input: pre_attn read it into norm_buf and did not write it.
        gemm_swap(
            &layer.o_proj,
            &layer.o_proj_fp8,
            self.scratch.attn_out,
            self.scratch.stream_acc,
            h,
            q_dim,
        )?;
        // 2026-10-10: Tensor-parallel: each rank's o_proj covers its heads; sum the partials
        // (the conv finish below is linear in its input).
        self.tp_all_reduce(ctx, self.scratch.stream_acc, (g * h) as usize * 2, stream)?;

        // 2026-09-25: DFlash2 attention conv finish on the o_proj output, with the
        // dynamic slice computed in pre_attn.
        let attn_res_src = self.conv_finish(
            layer,
            crate::dflash_head::dflash2::ConvSite::Attention,
            self.scratch.stream_acc,
            ctx,
            args.n_seq.max(1),
            stream,
        )?;

        // 2026-09-25: stream_buf += attn_res_src (the o_proj output, convolved when
        // DFlash2 is active).
        ops::residual_add(
            gpu,
            self.kernels.residual_add,
            self.scratch.stream_buf,
            attn_res_src,
            g * h,
            stream,
        )?;

        // 2026-09-25: stream_buf is both the MLP's residual and the input to
        // post_attention_layernorm, whose output goes to norm_buf.
        ops::rms_norm(
            gpu,
            self.kernels.rms_norm,
            self.scratch.stream_buf,
            &layer.post_attention_layernorm,
            self.scratch.norm_buf,
            g,
            h,
            self.rms_norm_eps,
            stream,
        )?;

        // 2026-09-25: DFlash2 MLP conv prepare. It overwrites scratch.conv_dyn, which
        // the attention conv finish above has already read.
        let mlp_src = self.conv_prepare(
            layer,
            crate::dflash_head::dflash2::ConvSite::Mlp,
            self.scratch.norm_buf,
            ctx,
            args.n_seq.max(1),
            stream,
        )?;

        // 2026-09-25: MLP: down(silu(gate(x)) * up(x)) with x = mlp_src. silu_mul writes
        // into mlp_intermediate, down_proj into stream_acc.
        gemm_swap(
            &layer.gate_proj,
            &layer.gate_proj_fp8,
            mlp_src,
            self.scratch.mlp_intermediate,
            inter,
            h,
        )?;
        gemm_swap(
            &layer.up_proj,
            &layer.up_proj_fp8,
            mlp_src,
            self.scratch.mlp_up,
            inter,
            h,
        )?;
        ops::silu_mul(
            gpu,
            self.kernels.silu_mul,
            self.scratch.mlp_intermediate,
            self.scratch.mlp_up,
            self.scratch.mlp_intermediate,
            g * inter,
            stream,
        )?;
        gemm_swap(
            &layer.down_proj,
            &layer.down_proj_fp8,
            self.scratch.mlp_intermediate,
            self.scratch.stream_acc,
            h,
            inter,
        )?;
        // 2026-10-10: Tensor-parallel: each rank's down covers its slice of the intermediate.
        self.tp_all_reduce(ctx, self.scratch.stream_acc, (g * h) as usize * 2, stream)?;

        // 2026-09-25: DFlash2 MLP conv finish on the down_proj output.
        let mlp_res_src = self.conv_finish(
            layer,
            crate::dflash_head::dflash2::ConvSite::Mlp,
            self.scratch.stream_acc,
            ctx,
            args.n_seq.max(1),
            stream,
        )?;

        // 2026-09-25: stream_buf += mlp_res_src. stream_buf is then the layer output,
        // read by the next layer or the tail.
        ops::residual_add(
            gpu,
            self.kernels.residual_add,
            self.scratch.stream_buf,
            mlp_res_src,
            g * h,
            stream,
        )?;

        if args.block_dump {
            self.block_dump_buf(
                ctx,
                self.scratch.stream_buf,
                args.layer_idx,
                "layer_out",
                g,
                h,
                stream,
            )?;
        }

        Ok(())
    }
}
