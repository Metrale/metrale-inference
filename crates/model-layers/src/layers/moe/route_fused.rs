// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: The routing step of the grouped NVFP4 decode: the exact per-row router
//! (`GroupedRouting::PerRow`) and the slot sort, from the Hopper fused launch
//! (`ops::moe_route_fused`) when `METRALE_MOE_ROUTE_FUSED=1`, else from the three launches.
//!
//! Owner: model-layers (MoE).
//! Invariants:
//! - Both forms write the same logits, picks, weights and sort outputs
//!   (`ops/moe_route_fused_tests.rs`); the fused one is taken only where its module exists
//!   (`try_target_kernel`: Hopper) and `ops::moe_route_fused_shape_ok` holds.
//! - The fused launch's completion counter is per buffer arena (keyed by its `gate_logits`
//!   address): two forwards that run at once use different arenas, as they already must for
//!   `gate_logits`, so no two launches in flight share a counter. An arena past the
//!   `COUNTER_SLOTS` this layer holds takes the three launches.

use std::sync::Mutex;

use super::*;
use crate::layers::try_target_kernel;

/// 2026-10-14: Buffer arenas one layer keeps a counter for.
const COUNTER_SLOTS: usize = 8;

/// 2026-10-14: `METRALE_MOE_ROUTE_FUSED` equal to `"1"`, read once per process.
fn moe_route_fused_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("METRALE_MOE_ROUTE_FUSED").as_deref() == Ok("1"))
}

/// 2026-10-14: The fused router kernel and its zeroed completion counters; a zero handle when
/// the lever is off or the module is absent.
pub(super) struct RouteFused {
    kernel: KernelHandle,
    counters: DevicePtr,
    arenas: Mutex<[u64; COUNTER_SLOTS]>,
}

impl RouteFused {
    pub(super) fn resolve(gpu: &dyn GpuBackend) -> Result<Self> {
        let kernel = if moe_route_fused_enabled() {
            try_target_kernel(gpu, "moe_route_fused_hopper", "moe_route_fused_hopper")
        } else {
            KernelHandle(0)
        };
        let counters = if kernel.0 != 0 {
            let p = gpu.alloc(COUNTER_SLOTS * 4)?;
            gpu.memset(p, 0, COUNTER_SLOTS * 4)?;
            p
        } else {
            DevicePtr::NULL
        };
        Ok(Self {
            kernel,
            counters,
            arenas: Mutex::new([0; COUNTER_SLOTS]),
        })
    }

    /// 2026-10-14: The counter of the arena whose `gate_logits` is `arena`, claiming a free slot
    /// on first use; `None` when every slot belongs to another arena.
    fn counter_for(&self, arena: DevicePtr) -> Option<DevicePtr> {
        let mut arenas = self.arenas.lock().unwrap_or_else(|e| e.into_inner());
        let slot = match arenas.iter().position(|&a| a == arena.0) {
            Some(s) => s,
            None => {
                let s = arenas.iter().position(|&a| a == 0)?;
                arenas[s] = arena.0;
                s
            }
        };
        Some(self.counters.offset(slot * 4))
    }
}

impl MoeLayer {
    /// 2026-10-14: Routes the `m` rows of `input` (picks to `indices_dev`, weights to
    /// `weights_dev`) and sorts the slots into `out`, which overlaps `gate_logits`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn grouped_route_sorted(
        &self,
        input: DevicePtr,
        m: usize,
        indices_dev: DevicePtr,
        weights_dev: DevicePtr,
        out: ops::Fp8GroupedSortOut,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size as u32;
        let num_experts = ctx.config.num_experts as u32;
        let top_k = ctx.config.num_experts_per_tok as u32;
        let n = m as u32;
        let gate_logits = ctx.buffers.gate_logits();
        let fused = self.route_fused.kernel.0 != 0
            && ops::moe_route_fused_shape_ok(n, h, num_experts, top_k);
        if let Some(counter) = fused.then(|| self.route_fused.counter_for(gate_logits)).flatten() {
            let router_in = self.router_input(input, n, h, ctx, stream)?;
            return ops::moe_route_fused(
                ctx.gpu,
                self.route_fused.kernel,
                router_in,
                self.weights.gate.weight,
                gate_logits,
                indices_dev,
                weights_dev,
                out,
                counter,
                n,
                num_experts,
                h,
                top_k,
                ctx.config.norm_topk_prob,
                stream,
            );
        }
        self.grouped_route(
            input,
            m,
            GroupedRouting::PerRow,
            indices_dev,
            weights_dev,
            ctx,
            stream,
        )?;
        // 2026-09-27: The sort outputs reuse `gate_logits`, which top-k has read earlier on
        // this stream (the layout of `grouped_decode_buffer_need`).
        ops::moe_fp8_grouped_sort(
            ctx.gpu,
            self.moe_fp8_grouped_sort_k,
            out,
            indices_dev,
            n * top_k,
            num_experts,
            top_k,
            stream,
        )
    }
}
