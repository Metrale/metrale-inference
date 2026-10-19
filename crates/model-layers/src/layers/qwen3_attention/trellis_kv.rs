// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: The device state and launches of a `Trellis`-family KV cache (the 128-group
//! rotated codecs; design credit spiritbuun/buun-llama-cpp, MIT; the bitshift trellis per QTIP,
//! arXiv:2406.11235; see CITATIONS.md). The layer builds one `TrellisKv` when its codec is of that family: the
//! codebooks and write midpoints live on the device for the layer's lifetime, and the write,
//! decode and chunked-prefill launches read them from here.
//!
//! Owner: model-layers (attention).
//! Invariants:
//! - Books, midpoints and V decode scales come from `metrale-kv-tq128`, found by the registry
//!   row's name; the kernel names come from the codec kernel table. Nothing here names a codec.
//! - Each launch's argument order is its `.cu` entry's declaration order
//!   (`reshape_and_cache_tcq.cu`, `reshape_and_cache_turbo4_n128.cu`,
//!   `paged_decode_attn_tq128.cu`); `trellis_kv_tests.rs` pins them.
//! - A sliding window and a split pair are refused with a coded error; the kernels have neither.

use anyhow::Result;
use metrale_error::{ErrorCode, error};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_kv_codec::{AbiId, KernelName, KvPair, ResolvedKvKernels, spec};
use metrale_kv_tq128::{GROUP, KvSide, Tq128Codec};

use crate::kv_abi::{DecodeCommon, KvSideArgs, WriteCommon};

/// 2026-10-18: Warps per write block; one warp codes one (token, head, group). The `.cu` files'
/// `WARPS`.
const WRITE_WARPS: u64 = 4;
/// 2026-10-18: Threads per attention block (the `.cu` files' `NUM_WARPS * 32`).
const ATTN_THREADS: u32 = 256;

/// 2026-10-18: A layer's `Trellis` cache state.
pub(super) struct TrellisKv {
    codec: Tq128Codec,
    k_book: DevicePtr,
    v_book: DevicePtr,
    mids: Option<DevicePtr>,
    prefill: KernelHandle,
}

/// 2026-10-18: One chunked-prefill attention call over a single sequence's paged cache.
pub(super) struct TrellisPrefill {
    /// 2026-10-18: `[n, nq, hd]` bf16 queries.
    pub q: DevicePtr,
    /// 2026-10-18: `[n, nq, hd]` bf16 output.
    pub out: DevicePtr,
    /// 2026-10-18: The sequence's block table (`i32` physical block ids).
    pub block_table: DevicePtr,
    /// 2026-10-18: Tokens cached before this chunk; token `t` attends `[0, seq_len_start + t]`.
    pub seq_len_start: u32,
    pub n: u32,
    pub nq: u32,
    pub nkv: u32,
    pub hd: u32,
    pub bs: u32,
    pub inv_sqrt_d: f32,
    pub sliding_window: u32,
    pub k: KvSideArgs,
    pub v: KvSideArgs,
}

/// 2026-10-18: `values` as little-endian `f32` bytes in a fresh device buffer.
fn upload(gpu: &dyn GpuBackend, values: &[f32]) -> Result<DevicePtr> {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let ptr = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(&bytes, ptr)?;
    Ok(ptr)
}

impl TrellisKv {
    /// 2026-10-18: The state of a layer holding `pair` at `head_dim`: `None` when neither side is
    /// of the `Trellis` family; refused when only one is, or when the two sides differ.
    pub(super) fn for_pair(
        gpu: &dyn GpuBackend,
        pair: KvPair,
        head_dim: usize,
        names: &ResolvedKvKernels<KernelName>,
    ) -> Result<Option<Self>> {
        let (k, v) = (spec(pair.k()), spec(pair.v()));
        if k.abi != AbiId::Trellis && v.abi != AbiId::Trellis {
            return Ok(None);
        }
        if pair.is_asymmetric() {
            return Err(error!(ErrorCode::KvPairUnsupported, k.name, v.name, head_dim).into());
        }
        let codec = Tq128Codec::named(k.name).ok_or_else(|| {
            error!(
                ErrorCode::KvCodecFeatureUnsupported,
                k.name, "the trellis launches (kv-tq128 has no format of that name)"
            )
        })?;
        let prefill = names
            .prefill
            .ok_or_else(|| error!(ErrorCode::KvAbiUnimplemented, "Trellis", "prefill"))?;
        Ok(Some(Self {
            codec,
            k_book: upload(gpu, codec.codebook(KvSide::Key))?,
            v_book: upload(gpu, codec.codebook(KvSide::Value))?,
            mids: codec.midpoints().map(|m| upload(gpu, m)).transpose()?,
            prefill: gpu.kernel(prefill.module, prefill.func)?,
        }))
    }

    /// 2026-10-18: The pool and block stride of `side`, which this state's pair stores in the
    /// `Trellis` family.
    fn side(&self, side: KvSideArgs) -> Result<(DevicePtr, u64)> {
        side.trellis().ok_or_else(|| {
            error!(
                ErrorCode::KvCodecFeatureUnsupported,
                self.codec.name(),
                "a side stored by another launch family"
            )
            .into()
        })
    }

    /// 2026-10-18: Refuses a sliding window, which the trellis attention kernels do not apply.
    fn no_window(&self, sliding_window: u32) -> Result<()> {
        if sliding_window != 0 {
            return Err(error!(
                ErrorCode::KvCodecFeatureUnsupported,
                self.codec.name(),
                "a sliding attention window"
            )
            .into());
        }
        Ok(())
    }

    /// 2026-10-18: Encodes `c.num_tokens` K and V rows into the cache through the write
    /// `kernel`: one launch per side, each with that side's book (and the write midpoints, for
    /// a scalar codec).
    pub(super) fn write(
        &self,
        gpu: &dyn GpuBackend,
        kernel: KernelHandle,
        c: &WriteCommon,
        k: KvSideArgs,
        v: KvSideArgs,
        stream: u64,
    ) -> Result<()> {
        let groups = u64::from(c.num_tokens)
            * u64::from(c.num_kv_heads)
            * u64::from(c.head_dim / GROUP as u32);
        if groups == 0 {
            return Ok(());
        }
        let grid = u32::try_from(groups.div_ceil(WRITE_WARPS))?;
        let sides = [
            (c.key, c.key_stride, self.side(k)?, self.k_book),
            (c.value, c.value_stride, self.side(v)?, self.v_book),
        ];
        for (src, src_stride, (pool, block_stride), book) in sides {
            let launch = KernelLaunch::new(gpu, kernel)
                .grid([grid, 1, 1])
                .block([(WRITE_WARPS * 32) as u32, 1, 1])
                .arg_ptr(src)
                .arg_ptr(pool)
                .arg_ptr(c.slot_mapping)
                .arg_ptr(book);
            let launch = match self.mids {
                Some(mids) => launch.arg_ptr(mids),
                None => launch,
            };
            launch
                .arg_u32(c.num_tokens)
                .arg_u32(c.num_kv_heads)
                .arg_u32(c.head_dim)
                .arg_u32(c.block_size)
                .arg_u32(src_stride)
                .arg_u64(block_stride)
                .launch(stream)?;
        }
        Ok(())
    }

    /// 2026-10-18: Paged decode attention through the decode `kernel`, one block per (query
    /// head, sequence), with each side's book and decode scale.
    pub(super) fn decode(
        &self,
        gpu: &dyn GpuBackend,
        kernel: KernelHandle,
        c: &DecodeCommon,
        k: KvSideArgs,
        v: KvSideArgs,
        stream: u64,
    ) -> Result<()> {
        self.no_window(c.sliding_window)?;
        let ((k_pool, k_stride), (v_pool, v_stride)) = (self.side(k)?, self.side(v)?);
        if c.num_seqs == 0 {
            return Ok(());
        }
        KernelLaunch::new(gpu, kernel)
            .grid([c.num_q_heads, c.num_seqs, 1])
            .block([ATTN_THREADS, 1, 1])
            .arg_ptr(c.q)
            .arg_ptr(k_pool)
            .arg_ptr(v_pool)
            .arg_ptr(c.output)
            .arg_ptr(c.block_tables)
            .arg_ptr(c.seq_lens)
            .arg_ptr(self.k_book)
            .arg_ptr(self.v_book)
            .arg_u32(c.max_blocks_per_seq)
            .arg_u32(c.num_q_heads)
            .arg_u32(c.num_kv_heads)
            .arg_u32(c.head_dim)
            .arg_u32(c.block_size)
            .arg_f32(c.inv_sqrt_d)
            .arg_f32(self.codec.decode_scale(KvSide::Key))
            .arg_f32(self.codec.decode_scale(KvSide::Value))
            .arg_u32(c.q_stride)
            .arg_u64(k_stride)
            .arg_u64(v_stride)
            .launch(stream)
    }

    /// 2026-10-18: Causal chunked-prefill attention of `p.n` tokens over the paged cache, one
    /// block per (query head, token).
    pub(super) fn prefill(
        &self,
        gpu: &dyn GpuBackend,
        p: &TrellisPrefill,
        stream: u64,
    ) -> Result<()> {
        self.no_window(p.sliding_window)?;
        let ((k_pool, k_stride), (v_pool, v_stride)) = (self.side(p.k)?, self.side(p.v)?);
        if p.n == 0 {
            return Ok(());
        }
        KernelLaunch::new(gpu, self.prefill)
            .grid([p.nq, p.n, 1])
            .block([ATTN_THREADS, 1, 1])
            .arg_ptr(p.q)
            .arg_ptr(k_pool)
            .arg_ptr(v_pool)
            .arg_ptr(p.out)
            .arg_ptr(p.block_table)
            .arg_ptr(self.k_book)
            .arg_ptr(self.v_book)
            .arg_u32(p.seq_len_start)
            .arg_u32(p.nq)
            .arg_u32(p.nkv)
            .arg_u32(p.hd)
            .arg_u32(p.bs)
            .arg_f32(p.inv_sqrt_d)
            .arg_f32(self.codec.decode_scale(KvSide::Key))
            .arg_f32(self.codec.decode_scale(KvSide::Value))
            .arg_u32(p.nq * p.hd)
            .arg_u64(k_stride)
            .arg_u64(v_stride)
            .launch(stream)
    }
}

#[cfg(test)]
#[path = "trellis_kv_tests.rs"]
mod tests;
