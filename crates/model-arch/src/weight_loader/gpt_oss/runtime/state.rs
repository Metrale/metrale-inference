// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: One allocation per layer/sequence; explicit teardown owns it.
use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::layer::LayerState;
use std::any::Any;

pub(super) struct State {
    pub allocation: DevicePtr,
    pub frequencies: DevicePtr,
    pub accum: DevicePtr,
    pub norm: DevicePtr,
    pub q: DevicePtr,
    pub k: DevicePtr,
    pub v: DevicePtr,
    pub attn: DevicePtr,
    pub projection: DevicePtr,
    pub logits: DevicePtr,
    pub scores: DevicePtr,
    pub ids: DevicePtr,
    pub gate_up: DevicePtr,
    pub activation: DevicePtr,
    pub selected: DevicePtr,
    pub moe: DevicePtr,
    pub position: DevicePtr,
    pub slot: DevicePtr,
    pub length: DevicePtr,
    pub table: DevicePtr,
    pub max_blocks: usize,
}
impl State {
    pub fn new(gpu: &dyn GpuBackend, max_positions: usize) -> Result<Self> {
        // 2026-10-07: Worst case block size one; avoids cache geometry assumptions.
        let widths = [
            5760,
            8192,
            1024,
            1024,
            8192,
            5760,
            64,
            64,
            16,
            11520,
            5760,
            23040,
            5760,
            8,
            8,
            8,
            max_positions * 4,
            128,
            16384,
        ];
        let allocation = gpu.alloc(widths.iter().sum())?;
        let mut offset = 0;
        let p: Vec<_> = widths
            .iter()
            .map(|bytes| {
                let ptr = allocation.offset(offset);
                offset += bytes;
                ptr
            })
            .collect();
        Ok(Self {
            allocation,
            norm: p[0],
            q: p[1],
            k: p[2],
            v: p[3],
            attn: p[4],
            projection: p[5],
            logits: p[6],
            scores: p[7],
            ids: p[8],
            gate_up: p[9],
            activation: p[10],
            selected: p[11],
            moe: p[12],
            position: p[13],
            slot: p[14],
            length: p[15],
            table: p[16],
            max_blocks: max_positions,
            frequencies: p[17],
            accum: p[18],
        })
    }
}
impl LayerState for State {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
