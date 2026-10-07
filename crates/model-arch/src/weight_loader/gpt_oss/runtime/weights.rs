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
    pub gate_up: Vec<Packed>,
    pub down: Vec<Packed>,
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
            gate_up: snapshot(&w.gate_up)?,
            down: snapshot(&w.down)?,
            gate_up_bias: w.gate_up_bias.ptr(),
            down_bias: w.down_bias.ptr(),
        })
    }
}
