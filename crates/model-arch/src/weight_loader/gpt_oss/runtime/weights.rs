// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Validated pointer snapshots; the model's WeightStore retains ownership.
use super::super::{GptOssLayerWeights, GptOssLinear};
use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

#[derive(Clone, Copy)]
pub(super) struct Linear {
    pub weight: DevicePtr,
    pub bias: DevicePtr,
    pub rows: u32,
    pub cols: u32,
}
impl Linear {
    fn from(w: &GptOssLinear<'_>) -> Self {
        Self {
            weight: w.weight.ptr(),
            bias: w.bias.ptr(),
            rows: w.weight.shape()[0] as u32,
            cols: w.weight.shape()[1] as u32,
        }
    }
}
#[derive(Clone, Copy)]
pub(super) struct Packed {
    pub blocks: DevicePtr,
    pub scales: DevicePtr,
    pub rows: u32,
    pub cols: u32,
}
pub(super) struct Weights {
    pub input_norm: DevicePtr,
    pub post_norm: DevicePtr,
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub o: Linear,
    pub router: Linear,
    pub sinks: DevicePtr,
    pub gate_up: Packed,
    pub down: Packed,
    pub gate_up_bias: DevicePtr,
    pub down_bias: DevicePtr,
}
impl Weights {
    pub fn from(w: &GptOssLayerWeights<'_>) -> Result<Self> {
        let snapshot=|pack:&metrale_model_layers::weight_map::PackedMxfp4Experts<'_>|->Result<Vec<Packed>> {
            (0..pack.expert_count()).map(|i| {let e=pack.expert(i)?;Ok(Packed{blocks:e.weight(),scales:e.scales(),rows:e.rows() as u32,cols:e.cols() as u32})}).collect()
        };
        Ok(Self {
            input_norm: w.input_norm.ptr(),
            post_norm: w.post_attention_norm.ptr(),
            q: Linear::from(&w.q),
            k: Linear::from(&w.k),
            v: Linear::from(&w.v),
            o: Linear::from(&w.o),
            router: Linear::from(&w.router),
            sinks: w.sinks.ptr(),
            gate_up: contiguous_base(&snapshot(&w.gate_up)?)?,
            down: contiguous_base(&snapshot(&w.down)?)?,
            gate_up_bias: w.gate_up_bias.ptr(),
            down_bias: w.down_bias.ptr(),
        })
    }
}

// 2026-10-07: Device expert selection requires the exact contiguous checkpoint
// layout; reject pointer snapshots that cannot justify derived expert strides.
fn contiguous_base(pack: &[Packed]) -> Result<Packed> {
    anyhow::ensure!(pack.len() == 32, "GPT selected kernel requires 32 experts");
    let base = pack[0];
    anyhow::ensure!(
        base.rows > 0
            && base.cols > 0
            && base.cols.is_multiple_of(32)
            && !base.blocks.is_null()
            && !base.scales.is_null(),
        "invalid GPT packed shape"
    );
    let weight_stride = u64::from(base.rows) * u64::from(base.cols / 2);
    let scale_stride = u64::from(base.rows) * u64::from(base.cols / 32);
    for (i, p) in pack.iter().enumerate() {
        anyhow::ensure!(
            p.rows == base.rows
                && p.cols == base.cols
                && weight_stride
                    .checked_mul(i as u64)
                    .and_then(|n| base.blocks.0.checked_add(n))
                    == Some(p.blocks.0)
                && scale_stride
                    .checked_mul(i as u64)
                    .and_then(|n| base.scales.0.checked_add(n))
                    == Some(p.scales.0),
            "GPT selected experts require contiguous validated storage"
        );
    }
    Ok(base)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn packed() -> Vec<Packed> {
        (0..32)
            .map(|i| Packed {
                blocks: DevicePtr(1024 + i * 64),
                scales: DevicePtr(8192 + i * 4),
                rows: 2,
                cols: 64,
            })
            .collect()
    }
    #[test]
    fn selected_storage_rejects_noncontiguous_or_wrong_geometry() {
        let good = packed();
        assert!(contiguous_base(&good).is_ok());
        assert!(contiguous_base(&good[..31]).is_err());
        let mut bad = packed();
        bad[7].blocks.0 += 1;
        assert!(contiguous_base(&bad).is_err());
        let mut bad = packed();
        bad[31].scales.0 += 1;
        assert!(contiguous_base(&bad).is_err());
        let mut bad = packed();
        bad[3].rows = 4;
        assert!(contiguous_base(&bad).is_err());
        let mut bad = packed();
        bad[0].blocks = DevicePtr(u64::MAX - 1);
        assert!(contiguous_base(&bad).is_err());
        let mut bad = packed();
        bad[0].blocks = DevicePtr(0);
        assert!(contiguous_base(&bad).is_err());
    }
}
