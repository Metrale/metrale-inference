// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: The KV ABI: launch-argument families (`AbiId`), one zero-sized type each, and the
//! one dispatch from a codec's family to its launcher (`dispatch::with_abi`). Executors read a
//! codec's spec and its family, never its id, so a codec that fits an existing family needs no
//! executor code.
//!
//! Owner: model-layers.
//! Invariants:
//! - `dispatch.rs` holds the only match on `AbiId`, with no wildcard arm.
//! - The fused-kernel argument order is reproduced by `pack.rs` from the side arguments alone.

#![deny(clippy::wildcard_enum_match_arm)]

mod dispatch;
mod pack;
#[cfg(test)]
mod pack_tests;

use metrale_error::{Error, ErrorCode, error};
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_kv_codec::{AbiId, KvCodecSpec};

pub use dispatch::{AbiFn, decode, side_args, with_abi, write};
pub use pack::{DecodeCommon, Tails, WriteCommon, launch_fused_decode, launch_fused_write};

/// 2026-10-18: One side's launch arguments, built by its family's `KvAbi::side_args`. Plain data:
/// the packer turns it into kernel arguments in a fixed order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum KvSideArgs {
    /// 2026-10-18: Values only.
    Dense {
        /// 2026-10-18: The side's pool.
        pool: DevicePtr,
        /// 2026-10-18: Bytes per block of this side.
        block_stride: u64,
    },
    /// 2026-10-18: Values plus one scale held outside the block.
    DenseScaled {
        /// 2026-10-18: The side's pool.
        pool: DevicePtr,
        /// 2026-10-18: Bytes per block of this side.
        block_stride: u64,
        /// 2026-10-18: The side's dequantization scale.
        scale: f32,
    },
    /// 2026-10-18: Packed values, then the in-block side tensors.
    BlockScaled {
        /// 2026-10-18: The side's pool.
        pool: DevicePtr,
        /// 2026-10-18: Bytes per block of this side.
        block_stride: u64,
        /// 2026-10-18: Bytes of packed values per block, where the side tensors start.
        data_bytes: u64,
    },
}

impl KvSideArgs {
    /// 2026-10-18: The side's pool.
    pub fn pool(&self) -> DevicePtr {
        match *self {
            KvSideArgs::Dense { pool, .. }
            | KvSideArgs::DenseScaled { pool, .. }
            | KvSideArgs::BlockScaled { pool, .. } => pool,
        }
    }
}

/// 2026-10-18: Everything a layer knows about one side of its cache; each family keeps the
/// fields its launchers read.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SideSource {
    /// 2026-10-18: The side's pool.
    pub pool: DevicePtr,
    /// 2026-10-18: Bytes per block of this side.
    pub block_stride: u64,
    /// 2026-10-18: Bytes of packed values per block of this side's codec.
    pub data_bytes: u64,
    /// 2026-10-18: The side's dequantization scale (the fp8 per-tensor scale).
    pub scale: f32,
}

/// 2026-10-18: The per-family operations of one attention layer. A layer implements it once;
/// `dispatch::decode` and `dispatch::write` pick the method from the codec's family.
pub trait KvExecutor {
    /// 2026-10-18: The arguments of one decode call.
    type Decode<'a>;
    /// 2026-10-18: The arguments of one cache write.
    type Write<'a>;
    /// 2026-10-18: Decode attention over a `Dense` cache.
    fn decode_dense(&self, a: &Self::Decode<'_>) -> anyhow::Result<()>;
    /// 2026-10-18: Decode attention over a `DenseScaled` cache.
    fn decode_dense_scaled(&self, a: &Self::Decode<'_>) -> anyhow::Result<()>;
    /// 2026-10-18: Decode attention over a `BlockScaled` cache.
    fn decode_block_scaled(&self, a: &Self::Decode<'_>) -> anyhow::Result<()>;
    /// 2026-10-18: Write into a `Dense` cache.
    fn write_dense(&self, a: &Self::Write<'_>) -> anyhow::Result<()>;
    /// 2026-10-18: Write into a `DenseScaled` cache.
    fn write_dense_scaled(&self, a: &Self::Write<'_>) -> anyhow::Result<()>;
    /// 2026-10-18: Write into a `BlockScaled` cache.
    fn write_block_scaled(&self, a: &Self::Write<'_>) -> anyhow::Result<()>;
}

/// 2026-10-18: A launch-argument family. Implemented by one zero-sized type per `AbiId`.
pub trait KvAbi {
    /// 2026-10-18: The family this type stands for.
    const ID: AbiId;
    /// 2026-10-18: The side arguments of a side stored as `spec`.
    fn side_args(spec: &'static KvCodecSpec, src: SideSource) -> Result<KvSideArgs, Error>;
    /// 2026-10-18: Decode attention over a symmetric cache of this family.
    fn decode<E: KvExecutor + ?Sized>(e: &E, a: &E::Decode<'_>) -> anyhow::Result<()>;
    /// 2026-10-18: Write into a symmetric cache of this family.
    fn write<E: KvExecutor + ?Sized>(e: &E, a: &E::Write<'_>) -> anyhow::Result<()>;
}

/// 2026-10-18: `AbiId::Dense`.
pub struct Dense;
/// 2026-10-18: `AbiId::DenseScaled`.
pub struct DenseScaled;
/// 2026-10-18: `AbiId::BlockScaled`.
pub struct BlockScaled;
/// 2026-10-18: `AbiId::Trellis`; declared, no codec row uses it yet.
pub struct Trellis;
/// 2026-10-18: `AbiId::Latent`; declared, no codec row uses it yet.
pub struct Latent;

impl KvAbi for Dense {
    const ID: AbiId = AbiId::Dense;
    fn side_args(_spec: &'static KvCodecSpec, src: SideSource) -> Result<KvSideArgs, Error> {
        Ok(KvSideArgs::Dense {
            pool: src.pool,
            block_stride: src.block_stride,
        })
    }
    fn decode<E: KvExecutor + ?Sized>(e: &E, a: &E::Decode<'_>) -> anyhow::Result<()> {
        e.decode_dense(a)
    }
    fn write<E: KvExecutor + ?Sized>(e: &E, a: &E::Write<'_>) -> anyhow::Result<()> {
        e.write_dense(a)
    }
}

impl KvAbi for DenseScaled {
    const ID: AbiId = AbiId::DenseScaled;
    fn side_args(_spec: &'static KvCodecSpec, src: SideSource) -> Result<KvSideArgs, Error> {
        Ok(KvSideArgs::DenseScaled {
            pool: src.pool,
            block_stride: src.block_stride,
            scale: src.scale,
        })
    }
    fn decode<E: KvExecutor + ?Sized>(e: &E, a: &E::Decode<'_>) -> anyhow::Result<()> {
        e.decode_dense_scaled(a)
    }
    fn write<E: KvExecutor + ?Sized>(e: &E, a: &E::Write<'_>) -> anyhow::Result<()> {
        e.write_dense_scaled(a)
    }
}

impl KvAbi for BlockScaled {
    const ID: AbiId = AbiId::BlockScaled;
    fn side_args(_spec: &'static KvCodecSpec, src: SideSource) -> Result<KvSideArgs, Error> {
        Ok(KvSideArgs::BlockScaled {
            pool: src.pool,
            block_stride: src.block_stride,
            data_bytes: src.data_bytes,
        })
    }
    fn decode<E: KvExecutor + ?Sized>(e: &E, a: &E::Decode<'_>) -> anyhow::Result<()> {
        e.decode_block_scaled(a)
    }
    fn write<E: KvExecutor + ?Sized>(e: &E, a: &E::Write<'_>) -> anyhow::Result<()> {
        e.write_block_scaled(a)
    }
}

/// 2026-10-18: The refusal of a declared family that has no launcher yet.
fn unimplemented(abi: &'static str, op: &'static str) -> Error {
    error!(ErrorCode::KvAbiUnimplemented, abi, op)
}

impl KvAbi for Trellis {
    const ID: AbiId = AbiId::Trellis;
    fn side_args(_spec: &'static KvCodecSpec, _src: SideSource) -> Result<KvSideArgs, Error> {
        Err(unimplemented("Trellis", "side-argument"))
    }
    fn decode<E: KvExecutor + ?Sized>(_e: &E, _a: &E::Decode<'_>) -> anyhow::Result<()> {
        Err(unimplemented("Trellis", "decode").into())
    }
    fn write<E: KvExecutor + ?Sized>(_e: &E, _a: &E::Write<'_>) -> anyhow::Result<()> {
        Err(unimplemented("Trellis", "write").into())
    }
}

impl KvAbi for Latent {
    const ID: AbiId = AbiId::Latent;
    fn side_args(_spec: &'static KvCodecSpec, _src: SideSource) -> Result<KvSideArgs, Error> {
        Err(unimplemented("Latent", "side-argument"))
    }
    fn decode<E: KvExecutor + ?Sized>(_e: &E, _a: &E::Decode<'_>) -> anyhow::Result<()> {
        Err(unimplemented("Latent", "decode").into())
    }
    fn write<E: KvExecutor + ?Sized>(_e: &E, _a: &E::Write<'_>) -> anyhow::Result<()> {
        Err(unimplemented("Latent", "write").into())
    }
}
