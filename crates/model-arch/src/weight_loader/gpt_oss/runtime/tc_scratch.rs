// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Opt-in packed TC scratch; BF16 weights are decoded only inside the shared kernel.
use super::tc_plan::TcPlan;
use super::*;
use metrale_gpu_runtime::gpu::KernelHandle;
pub(super) struct TcScratch {
    allocation: DevicePtr,
    project: KernelHandle,
    reorder: KernelHandle,
    temp: DevicePtr,
    blocks: DevicePtr,
    scales: DevicePtr,
    scale2: DevicePtr,
    offsets: DevicePtr,
    gate_rows: DevicePtr,
    down_rows: DevicePtr,
    inverse: DevicePtr,
}
impl TcScratch {
    const SIZES: [usize; 8] = [
        4 * 128 * 5760 * 2,
        256,
        256,
        128,
        144,
        4 * 128 * 4,
        4 * 128 * 4,
        4 * 128 * 4,
    ];
    pub fn required_bytes() -> usize {
        Self::SIZES.iter().sum()
    }
    pub fn new(gpu: &dyn GpuBackend) -> Result<Self> {
        let project = gpu.kernel(
            "moe_w4a16_grouped_gemm",
            "moe_w4a16_grouped_gemm_ptrtable_e8m0_gpt",
        )?;
        let reorder = gpu.kernel("moe_v41", "moe_v41_gather_rows")?;
        let bytes = Self::required_bytes();
        ensure!(
            gpu.device_free_memory()? >= bytes + gpu.total_memory()?.div_ceil(100) * 15,
            "TC diagnostic scratch would exceed85% device memory"
        );
        let allocation = gpu.alloc(bytes)?;
        let mut at = 0;
        let p: Vec<_> = Self::SIZES
            .iter()
            .map(|&n| {
                let p = allocation.offset(at);
                at += n;
                p
            })
            .collect();
        Ok(Self {
            allocation,
            project,
            reorder,
            temp: p[0],
            blocks: p[1],
            scales: p[2],
            scale2: p[3],
            offsets: p[4],
            gate_rows: p[5],
            down_rows: p[6],
            inverse: p[7],
        })
    }
    pub fn upload_plan(&self, gpu: &dyn GpuBackend, plan: &TcPlan, stream: u64) -> Result<()> {
        for (words, ptr) in [
            (&plan.offsets, self.offsets),
            (&plan.gate_rows, self.gate_rows),
            (&plan.down_rows, self.down_rows),
            (&plan.inverse, self.inverse),
        ] {
            let bytes: Vec<_> = words.iter().flat_map(|v| v.to_le_bytes()).collect();
            gpu.copy_h2d_async(&bytes, ptr, stream)?;
        }
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    pub fn projection(
        &self,
        gpu: &dyn GpuBackend,
        weight: weights::Packed,
        input: DevicePtr,
        output: DevicePtr,
        max_rows: u32,
        per_slot: bool,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            matches!(weight.rows, 2880 | 5760) && weight.cols == 2880,
            "TC diagnostic weight shape"
        );
        let block_stride = weight.rows as usize * weight.cols as usize / 2;
        let scale_stride = weight.rows as usize * weight.cols as usize / 32;
        for (base, stride, dest) in [
            (weight.blocks, block_stride, self.blocks),
            (weight.scales, scale_stride, self.scales),
        ] {
            let end = (stride * 32) as u64;
            ensure!(
                !base.is_null() && base.0.checked_add(end).is_some(),
                "TC expert pointer-table overflow"
            );
            let bytes: Vec<_> = (0..32)
                .flat_map(|e| base.offset(e * stride).0.to_le_bytes())
                .collect();
            gpu.copy_h2d_async(&bytes, dest, stream)?;
        }
        let scale2: Vec<_> = (0..32).flat_map(|_| 1.0f32.to_le_bytes()).collect();
        gpu.copy_h2d_async(&scale2, self.scale2, stream)?;
        ops::gpt_oss_packed_tc::gpt_oss_packed_tc(
            gpu,
            self.project,
            input,
            self.blocks,
            self.scales,
            self.scale2,
            self.temp,
            self.offsets,
            if per_slot {
                self.down_rows
            } else {
                self.gate_rows
            },
            128,
            weight.rows,
            weight.cols,
            max_rows,
            per_slot,
            stream,
        )?;
        ops::gpt_oss_packed_tc::gpt_oss_packed_tc_reorder(
            gpu,
            self.reorder,
            self.temp,
            self.inverse,
            output,
            128,
            weight.rows,
            stream,
        )
    }
    pub fn release(&mut self, gpu: &dyn GpuBackend) -> Result<()> {
        if !self.allocation.is_null() {
            gpu.free(self.allocation)?;
            self.allocation = DevicePtr::NULL;
        }
        Ok(())
    }
}
