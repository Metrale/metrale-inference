// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: `MoeLayer::pair_nvfp4_experts`: permute a declared-NVFP4 layer's routed and shared
//! expert weights in place into MMA-paired nibble order (`nvfp4_repack_mma_pairs`), which the
//! tensor-core `_tc_r` pair decodes with two logic ops per BF16 pair instead of byte tables
//! (kernels/gb10/common/tc_weight_formats.cuh `Nvfp4G16R`). Same products and sums as the
//! row-major pair; fewer ALU instructions per weight, so less power at the same bytes per second.
//!
//! Owner: model-layers (MoE).
//! Invariants: after pairing, only the `_tc_r` pair reads the permuted weights: the layer's
//! grouped NVFP4 decode selects it, and every row-major NVFP4 decode arm refuses a paired layer.
//! The prefill copies (transposed, FP8) are built from the row-major order before pairing.

use metrale_gpu_runtime::kernel_args::KernelLaunch;

use super::*;

/// 2026-10-02: `METRALE_MOE_NVFP4_PAIRED` (presence): pair a declared-NVFP4 MoE's expert weights at
/// load. Read once per process.
pub fn nvfp4_paired_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("METRALE_MOE_NVFP4_PAIRED").is_some())
}

impl MoeLayer {
    /// 2026-10-02: Pair this layer's routed and shared NVFP4 expert weights in place. Errors unless
    /// the experts are the checkpoint's declared NVFP4 (`set_declared_nvfp4_experts`), the `_tc_r`
    /// pair and the permutation kernel resolved, and the projections fit the tensor-core tiles.
    pub fn pair_nvfp4_experts(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &metrale_config::ModelConfig,
        stream: u64,
    ) -> Result<()> {
        let k = &self.nvfp4_grouped;
        let (h, inter) = (
            config.hidden_size as u32,
            config.moe_intermediate_size as u32,
        );
        anyhow::ensure!(
            k.declared_experts
                && !k.paired
                && k.gate_up_tc_r.0 != 0
                && k.down_tc_r.0 != 0
                && k.repack.0 != 0
                && ops::nvfp4_grouped_tc_shape_ok(inter, h, ops::NVFP4_GROUPED_GATE_UP_TC)
                && ops::nvfp4_grouped_tc_shape_ok(h, inter, ops::NVFP4_GROUPED_DOWN_TC),
            "pair_nvfp4_experts: needs declared NVFP4 experts, the paired kernels and tile shapes"
        );
        let repack = k.repack;
        let launch = |w: DevicePtr, n: u32, kk: u32| -> Result<()> {
            if w.is_null() {
                return Ok(());
            }
            let words = n as u64 * kk as u64 / 8;
            KernelLaunch::new(gpu, repack)
                .grid([words.div_ceil(256).min(4096) as u32, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(w)
                .arg_u64(words)
                .launch(stream)
        };
        for e in &self.weights.experts {
            launch(e.gate_proj.weight, inter, h)?;
            launch(e.up_proj.weight, inter, h)?;
            launch(e.down_proj.weight, h, inter)?;
        }
        let sh = &self.weights.shared_expert;
        let shared_inter = config.shared_expert_intermediate_size as u32;
        launch(sh.gate_proj.weight, shared_inter, h)?;
        launch(sh.up_proj.weight, shared_inter, h)?;
        launch(sh.down_proj.weight, h, shared_inter)?;
        gpu.synchronize(stream)?;
        self.nvfp4_grouped.paired = true;
        Ok(())
    }

    /// 2026-10-02: Whether the experts are the checkpoint's declared NVFP4
    /// (`set_declared_nvfp4_experts`).
    pub fn declared_nvfp4_experts(&self) -> bool {
        self.nvfp4_grouped.declared_experts
    }

    /// 2026-10-02: Refuse a row-major NVFP4 decode arm on a paired layer (its weights are permuted).
    pub(super) fn refuse_if_paired(&self, arm: &str) -> Result<()> {
        anyhow::ensure!(
            !self.nvfp4_grouped.paired,
            "{arm}: this layer's NVFP4 experts are MMA-paired; only the grouped tensor-core decode \
             reads them (the grouped decode declined this call)"
        );
        Ok(())
    }
}
