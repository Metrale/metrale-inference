// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: The argument packer of the fused KV write and decode kernels: one launcher per
//! operation for every codec and pair, built from the two sides' `KvSideArgs`. It replaces the
//! per-pair `ops::*` launchers, whose argument order it reproduces.
//!
//! Owner: model-layers.
//! Invariants: the order is the `.cu` entry points' declaration order:
//! - write: the five pointers, the five shape `u32`s, a `DenseScaled` K's scale, then per side
//!   its block stride and, for a `BlockScaled` side, its data-section bytes;
//! - decode: the six pointers, the five shape `u32`s, the softmax scale, a `DenseScaled` K's
//!   scale, the Q stride, then per `BlockScaled` side its block stride and data-section bytes,
//!   and for a split pair the sliding window.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use super::KvSideArgs;

/// 2026-10-18: How many per-side tails a fused kernel takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tails {
    /// 2026-10-18: A symmetric cache: K's tail once, which the kernel applies to both pools.
    Shared,
    /// 2026-10-18: A pair: each side's own tail (and, for decode, the sliding window).
    PerSide,
}

/// 2026-10-18: The arguments of a fused cache write besides the sides.
#[derive(Clone, Copy, Debug)]
pub struct WriteCommon {
    pub key: DevicePtr,
    pub value: DevicePtr,
    pub slot_mapping: DevicePtr,
    pub num_tokens: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub block_size: u32,
    pub key_stride: u32,
    pub value_stride: u32,
}

/// 2026-10-18: The arguments of a fused decode besides the sides.
#[derive(Clone, Copy, Debug)]
pub struct DecodeCommon {
    pub q: DevicePtr,
    pub output: DevicePtr,
    pub block_tables: DevicePtr,
    pub seq_lens: DevicePtr,
    pub max_blocks_per_seq: u32,
    pub num_seqs: u32,
    pub num_q_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub block_size: u32,
    pub inv_sqrt_d: f32,
    pub q_stride: u32,
    /// 2026-10-18: Positions attended behind the newest one, 0 for all; a split pair's kernels
    /// take it, a symmetric cache's do not.
    pub sliding_window: u32,
}

/// 2026-10-18: `launch` with the K side's out-of-block scale, when it has one.
fn k_scale(launch: KernelLaunch<'_>, k: KvSideArgs) -> KernelLaunch<'_> {
    match k {
        KvSideArgs::DenseScaled { scale, .. } => launch.arg_f32(scale),
        KvSideArgs::Dense { .. } | KvSideArgs::BlockScaled { .. } | KvSideArgs::Trellis { .. } => {
            launch
        }
    }
}

/// 2026-10-18: The sides whose tails a kernel takes.
fn tailed(k: KvSideArgs, v: KvSideArgs, tails: Tails) -> impl Iterator<Item = KvSideArgs> {
    let v = match tails {
        Tails::Shared => None,
        Tails::PerSide => Some(v),
    };
    core::iter::once(k).chain(v)
}

/// 2026-10-18: Write `num_tokens` K/V rows into the cache through the fused write `kernel`, one
/// 256-thread block per token.
pub fn launch_fused_write(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    c: &WriteCommon,
    k: KvSideArgs,
    v: KvSideArgs,
    tails: Tails,
    stream: u64,
) -> Result<()> {
    let launch = KernelLaunch::new(gpu, kernel)
        .grid([c.num_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(c.key)
        .arg_ptr(c.value)
        .arg_ptr(k.pool())
        .arg_ptr(v.pool())
        .arg_ptr(c.slot_mapping)
        .arg_u32(c.num_kv_heads)
        .arg_u32(c.head_dim)
        .arg_u32(c.block_size)
        .arg_u32(c.key_stride)
        .arg_u32(c.value_stride);
    tailed(k, v, tails)
        .fold(k_scale(launch, k), |launch, side| match side {
            KvSideArgs::Dense { block_stride, .. }
            | KvSideArgs::DenseScaled { block_stride, .. }
            | KvSideArgs::Trellis { block_stride, .. } => launch.arg_u64(block_stride),
            KvSideArgs::BlockScaled {
                block_stride,
                data_bytes,
                ..
            } => launch.arg_u64(block_stride).arg_u64(data_bytes),
        })
        .launch(stream)
}

/// 2026-10-18: Paged decode attention through the fused decode `kernel`, one 256-thread block
/// per (query head, sequence).
pub fn launch_fused_decode(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    c: &DecodeCommon,
    k: KvSideArgs,
    v: KvSideArgs,
    tails: Tails,
    stream: u64,
) -> Result<()> {
    let launch = KernelLaunch::new(gpu, kernel)
        .grid([c.num_q_heads, c.num_seqs, 1])
        .block([256, 1, 1])
        .arg_ptr(c.q)
        .arg_ptr(k.pool())
        .arg_ptr(v.pool())
        .arg_ptr(c.output)
        .arg_ptr(c.block_tables)
        .arg_ptr(c.seq_lens)
        .arg_u32(c.max_blocks_per_seq)
        .arg_u32(c.num_q_heads)
        .arg_u32(c.num_kv_heads)
        .arg_u32(c.head_dim)
        .arg_u32(c.block_size)
        .arg_f32(c.inv_sqrt_d);
    let launch = k_scale(launch, k).arg_u32(c.q_stride);
    let launch = tailed(k, v, tails).fold(launch, |launch, side| match side {
        KvSideArgs::Dense { .. } | KvSideArgs::DenseScaled { .. } | KvSideArgs::Trellis { .. } => {
            launch
        }
        KvSideArgs::BlockScaled {
            block_stride,
            data_bytes,
            ..
        } => launch.arg_u64(block_stride).arg_u64(data_bytes),
    });
    match tails {
        Tails::Shared => launch,
        Tails::PerSide => launch.arg_u32(c.sliding_window),
    }
    .launch(stream)
}
