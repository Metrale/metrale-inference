// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The router step of the grouped FP8 MoE decode
//! (`forward_fp8_grouped_decode_routed`) and its admission per routing.
//!
//! The grouped expert kernels reproduce the per-row FP8 expert and blend
//! kernels bit for bit given the same routing
//! (`model-arch/examples/fp8_moe_grouped_decode_microtest.rs`). The routing is
//! what differed: the batched gate GEMM and `moe_topk_softmax_batched` may pick
//! differently from the per-row router on a near-tie. The two exact routings
//! repeat the router of the path they replace, so the whole MoE output is the
//! same bytes:
//! - `PerRow`: `MoeLayer::forward` once per row. `dense_gemv_bf16_batchm`
//!   (row-for-row `dense_gemv_bf16`, at most 16 rows a launch) and
//!   `moe_topk_softmax_rows` (the `moe_topk_softmax` body per row).
//! - `PerToken`: `MoeLayer::forward_batched`. The router GEMM of
//!   `batched_gate_logits` (`router_gemm_bf16`, the bits of `dense_gemm_bf16`)
//!   over all rows, then `moe_topk_softmax_rows`.
//!
//! Owner: model-layers (MoE).
//! Invariants: `grouped_route` launches nothing unless
//! `fp8_grouped_routing_ok` holds for its `m` and `routing`.

use super::*;

/// 2026-09-26: Which router arithmetic `forward_fp8_grouped_decode_routed` runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GroupedRouting {
    /// 2026-09-27: The batched gate GEMM (NVFP4 or BF16) and the batched softmax or sigmoid
    /// top-k: the MTP drafter and the `METRALE_FP8_MOE_GROUPED_DECODE` arm.
    Batched,
    /// 2026-09-27: The router of `MoeLayer::forward`, row by row.
    PerRow,
    /// 2026-09-27: The router of `MoeLayer::forward_batched` with BF16 logits.
    PerToken,
}

/// 2026-09-27: Rows per block row of the per-row router GEMV
/// (`dense_gemv_batchm_split`); at most `DENSE_GEMV_BATCHM_MAX_M`.
const ROUTER_ROWS_PER_BLOCK: u32 = 4;

/// 2026-09-29: Block rows of the per-row router GEMV for `n` rows: `ROUTER_ROWS_PER_BLOCK` rows
/// each up to 16 rows (4 block rows, 256 CTAs), then as many rows per block row as keep 4 block
/// rows, up to `DENSE_GEMV_BATCHM_MAX_M`. Every block row reads the whole 1 MB router, so at a
/// 256-row verify 64 block rows read it 64 times (7.8 ms per step on dgx2). A row's result does
/// not depend on how rows are grouped (`dense_gemv_bf16_batchm.cu`).
fn router_block_rows(n: u32) -> u32 {
    let per_block = n
        .div_ceil(4)
        .clamp(ROUTER_ROWS_PER_BLOCK, ops::DENSE_GEMV_BATCHM_MAX_M);
    n.div_ceil(per_block)
}

impl MoeLayer {
    /// 2026-09-26: Whether the grouped decode serves `m` rows with `routing`.
    /// `Batched` is [`Self::fp8_grouped_decode_ok`]. The exact routings also need
    /// a BF16 softmax router with no correction bias, pre-router norm or FP32
    /// gate, a layer `forward` does not reroute (DFlash capture), and their
    /// kernels.
    pub fn fp8_grouped_routing_ok(
        &self,
        m: usize,
        routing: GroupedRouting,
        ctx: &ForwardContext,
    ) -> bool {
        if !self.fp8_grouped_decode_ok(m, ctx) {
            return false;
        }
        let exact_router = self.gate_nvfp4.is_none()
            && self.correction_bias_dev.is_none()
            && self.weights.router_pre_norm.is_none()
            && !ctx.levers.fp32_gate
            && !(self.is_dflash_capture_layer && ctx.levers.frankenstein_decode_via_prefill)
            && self.moe_topk_softmax_rows_k.0 != 0;
        match routing {
            GroupedRouting::Batched => true,
            GroupedRouting::PerToken => exact_router,
            GroupedRouting::PerRow => {
                exact_router
                    && self.router_gemv_batchm_k.0 != 0
                    && ctx.config.hidden_size.is_multiple_of(8)
            }
        }
    }

    /// 2026-09-26: Router logits into `gate_logits()` and top-k into
    /// `indices_dev` / `weights_dev` (`[m * top_k]` u32 and f32) for `m` rows of
    /// `input`, with the arithmetic `routing` names.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn grouped_route(
        &self,
        input: DevicePtr,
        m: usize,
        routing: GroupedRouting,
        indices_dev: DevicePtr,
        weights_dev: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size as u32;
        let num_experts = ctx.config.num_experts as u32;
        let top_k = ctx.config.num_experts_per_tok as u32;
        let n = m as u32;
        let router_in = self.router_input(input, n, h, ctx, stream)?;
        let gate_logits = ctx.buffers.gate_logits();
        match routing {
            GroupedRouting::PerRow => {
                // 2026-09-27: One launch, ROUTER_ROWS_PER_BLOCK rows per block row: the
                // router's 256 columns give only 64 blocks per row group.
                ops::dense_gemv_batchm_split(
                    ctx.gpu,
                    self.router_gemv_batchm_k,
                    router_in,
                    &self.weights.gate,
                    gate_logits,
                    n,
                    router_block_rows(n),
                    num_experts,
                    h,
                    num_experts,
                    stream,
                )?;
            }
            GroupedRouting::PerToken => {
                self.router_gemm_bf16(router_in, gate_logits, n, num_experts, h, ctx, stream)?;
            }
            GroupedRouting::Batched => {
                return self.grouped_route_batched(
                    router_in,
                    n,
                    indices_dev,
                    weights_dev,
                    ctx,
                    stream,
                );
            }
        }
        ops::moe_topk_softmax_batched(
            ctx.gpu,
            self.moe_topk_softmax_rows_k,
            gate_logits,
            indices_dev,
            weights_dev,
            num_experts,
            top_k,
            ctx.config.norm_topk_prob,
            n,
            stream,
        )
    }

    /// 2026-09-25: The `Batched` router: `[n, H] x [H, E]` -> `gate_logits [n, E]`
    /// (NVFP4 or BF16 gate), then the batched sigmoid (with a correction bias)
    /// or softmax top-k.
    fn grouped_route_batched(
        &self,
        router_in: DevicePtr,
        n: u32,
        indices_dev: DevicePtr,
        weights_dev: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size as u32;
        let num_experts = ctx.config.num_experts as u32;
        let top_k = ctx.config.num_experts_per_tok as u32;
        let gate_logits = ctx.buffers.gate_logits();
        if let Some(ref nvfp4) = self.gate_nvfp4 {
            ops::w4a16_gemm(
                ctx.gpu,
                self.w4a16_gemm,
                router_in,
                nvfp4,
                gate_logits,
                n,
                num_experts,
                h,
                stream,
            )?;
        } else {
            self.router_gemm_bf16(router_in, gate_logits, n, num_experts, h, ctx, stream)?;
        }
        if let Some(bias) = self.correction_bias_dev {
            ops::moe_topk_sigmoid_batched(
                ctx.gpu,
                self.moe_topk_sigmoid_batched_k,
                gate_logits,
                bias,
                indices_dev,
                weights_dev,
                num_experts,
                top_k,
                ctx.config.norm_topk_prob,
                ctx.config.routed_scaling_factor as f32,
                n,
                stream,
            )
        } else {
            ops::moe_topk_softmax_batched(
                ctx.gpu,
                self.moe_topk_batched,
                gate_logits,
                indices_dev,
                weights_dev,
                num_experts,
                top_k,
                ctx.config.norm_topk_prob,
                n,
                stream,
            )
        }
    }
}

#[cfg(test)]
mod router_block_rows_tests {
    use super::router_block_rows;

    /// 2026-09-29: Up to 16 rows the grouping is the previous one (4 rows per block row); wider
    /// batches keep 4 block rows until a block row holds 16, and no block row exceeds 16.
    #[test]
    fn narrow_batches_keep_four_rows_per_block_and_wide_ones_cap_at_sixteen() {
        for n in 1..=16u32 {
            assert_eq!(router_block_rows(n), n.div_ceil(4), "n={n}");
        }
        assert_eq!(router_block_rows(64), 4);
        assert_eq!(router_block_rows(128), 8);
        assert_eq!(router_block_rows(256), 16);
        for n in 1..=512u32 {
            let y = router_block_rows(n);
            assert!(
                n.div_ceil(y) <= 16,
                "n={n}: {} rows per block row",
                n.div_ceil(y)
            );
        }
    }
}
