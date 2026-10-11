// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Launchers for the device-side argmax kernels (single row,
//! batched, with top-1 log-probability, and the feed-cell variant), the embedding
//! gathers, and the device token feed resolver.
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.

#![allow(unused_imports)]

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::moe;
use crate::weight_map::{DenseWeight, Fp8DenseWeight, Fp8Weight, QuantizedWeight};

use super::*;

/// 2026-09-25: Index of the largest of `vocab_size` BF16 logits, written as one
/// u32 to `out`, by a single block.
pub fn argmax_bf16(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    logits: DevicePtr,
    out: DevicePtr,
    vocab_size: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([1, 1, 1])
        .block([1024, 1, 1])
        .arg_ptr(logits)
        .arg_ptr(out)
        .arg_u32(vocab_size)
        .launch(stream)
}

/// 2026-09-25: [`argmax_bf16`] for `n_rows` rows `row_stride` apart, one block
/// per row. Each block runs the per-row body of `argmax_bf16`
/// (`argmax_bf16.cu`), so a row's index equals what `argmax_bf16` returns for it.
#[allow(clippy::too_many_arguments)]
pub fn argmax_bf16_batch(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    logits: DevicePtr,
    out: DevicePtr,
    vocab_size: u32,
    n_rows: u32,
    row_stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([n_rows, 1, 1])
        .block([1024, 1, 1])
        .arg_ptr(logits)
        .arg_ptr(out)
        .arg_u32(vocab_size)
        .arg_u32(row_stride)
        .launch(stream)
}

/// 2026-09-25: [`argmax_bf16_batch`] that also writes each row's top-1
/// log-probability, `out_logprob[row] = log softmax(row)[argmax]` in FP32,
/// from an online softmax in the same pass over the row. D-Cut ranks
/// verification depths by prefix sums of these values.
#[allow(clippy::too_many_arguments)]
pub fn argmax_bf16_batch_lp(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    logits: DevicePtr,
    out: DevicePtr,
    out_logprob: DevicePtr,
    vocab_size: u32,
    n_rows: u32,
    row_stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([n_rows, 1, 1])
        .block([1024, 1, 1])
        .arg_ptr(logits)
        .arg_ptr(out)
        .arg_ptr(out_logprob)
        .arg_u32(vocab_size)
        .arg_u32(row_stride)
        .launch(stream)
}

/// 2026-09-25: Read the token id at `argmax_out`, copy its BF16 row of
/// `embed_table` to `embed_out`, and copy the id to `token_id_out`, all on the
/// device.
pub fn embed_from_argmax(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    argmax_out: DevicePtr,
    embed_table: DevicePtr,
    embed_out: DevicePtr,
    token_id_out: DevicePtr,
    hidden_size: u32,
    stream: u64,
) -> Result<()> {
    let grid_x = hidden_size.div_ceil(256);
    KernelLaunch::new(gpu, kernel)
        .grid([grid_x, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(argmax_out)
        .arg_ptr(embed_table)
        .arg_ptr(embed_out)
        .arg_ptr(token_id_out)
        .arg_u32(hidden_size)
        .launch(stream)
}

/// 2026-09-25: Gather the BF16 embedding rows of `num_tokens` token ids
/// (`token_ids_dev`, device `[num_tokens]` u32) into `output`, one block per
/// token.
pub fn batched_embed(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    token_ids_dev: DevicePtr,
    embed_table: DevicePtr,
    output: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(token_ids_dev)
        .arg_ptr(embed_table)
        .arg_ptr(output)
        .arg_u32(hidden_size)
        .launch(stream)
}

/// 2026-09-25: [`batched_embed`] over an FP8 E4M3 table with one f32 scale per
/// row (the `quantize_bf16_to_fp8` layout); rows are dequantized to BF16 on
/// read.
pub fn batched_embed_fp8(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    token_ids_dev: DevicePtr,
    embed_table: DevicePtr,
    row_scale: DevicePtr,
    output: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(token_ids_dev)
        .arg_ptr(embed_table)
        .arg_ptr(row_scale)
        .arg_ptr(output)
        .arg_u32(hidden_size)
        .launch(stream)
}

/// 2026-09-25: Bit of a [`feed_resolve`] source word that marks an inline host
/// token id in the low 31 bits; without it the word indexes the previous
/// step's feed cell (`argmax_feed.cu`).
pub const FEED_HOST_BIT: u32 = 0x8000_0000;

/// 2026-09-25: Batched argmax into the feed `cells` (device u32 per row) with a
/// per-row pair of masked ids `masks` (`[m0, m1]` per row, `u32::MAX` = none):
/// when the plain argmax is a masked id, the row gets the argmax with both ids
/// excluded. `argmax_feed.cu` states the tie rules. One block per row.
pub fn argmax_bf16_batch_feed(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    logits: DevicePtr,
    masks: DevicePtr,
    cells: DevicePtr,
    vocab_size: u32,
    n_rows: u32,
    row_stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([n_rows, 1, 1])
        .block([1024, 1, 1])
        .arg_ptr(logits)
        .arg_ptr(masks)
        .arg_ptr(cells)
        .arg_u32(vocab_size)
        .arg_u32(row_stride)
        .launch(stream)
}

/// 2026-09-25: `ids_out[i]` is the inline id when `sources[i]` has
/// [`FEED_HOST_BIT`] set, else `cells[sources[i]]`.
pub fn feed_resolve(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    sources: DevicePtr,
    cells: DevicePtr,
    ids_out: DevicePtr,
    n_rows: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([1, 1, 1])
        .block([128, 1, 1])
        .arg_ptr(sources)
        .arg_ptr(cells)
        .arg_ptr(ids_out)
        .arg_u32(n_rows)
        .launch(stream)
}

/// 2026-10-10: Threads per block of [`argmax_key_shard_bf16`].
const ARGMAX_KEY_BLOCK: u32 = 256;

/// 2026-10-10: Raise `keys` (`[p, q]` u64, zeroed before the first slice) to the sharded feed
/// argmax keys of `n_local` BF16 logits at `shard`, whose first entry is global index `base`;
/// `masks` is the row's `[m0, m1]` (`argmax_key.cu`, host spec
/// `metrale_sampling::feed_argmax_key`). One block per 1024 entries, at most 128.
#[allow(clippy::too_many_arguments)]
pub fn argmax_key_shard_bf16(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    shard: DevicePtr,
    masks: DevicePtr,
    keys: DevicePtr,
    base: u32,
    n_local: u32,
    stream: u64,
) -> Result<()> {
    let blocks = n_local.div_ceil(1024).clamp(1, 128);
    KernelLaunch::new(gpu, kernel)
        .grid([blocks, 1, 1])
        .block([ARGMAX_KEY_BLOCK, 1, 1])
        .arg_ptr(shard)
        .arg_ptr(masks)
        .arg_ptr(keys)
        .arg_u32(base)
        .arg_u32(n_local)
        .launch(stream)
}

/// 2026-10-10: `cell[0]` = the token named by the max of `keys` and `n_peers` peer key pairs at
/// `peers + slot * slot_bytes` (a one-shot exchange's slots; `n_peers` 0 when `keys` holds every
/// slice), under `masks`.
/// 2026-10-11: The K=2 MTP verdict on the device (`mtp_k2_verdict.cu`, host specification
/// `speculative::k2_device_verdict`): from the verify's two argmaxes and its token ids (row 1 is
/// the draft) into `out`'s `K2_DEVICE_VERDICT_WORDS` words.
pub fn mtp_k2_verdict(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    argmax: DevicePtr,
    token_ids: DevicePtr,
    out: DevicePtr,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([1, 1, 1])
        .block([32, 1, 1])
        .arg_ptr(argmax)
        .arg_ptr(token_ids)
        .arg_ptr(out)
        .launch(stream)
}

#[allow(clippy::too_many_arguments)]
pub fn argmax_key_resolve(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    keys: DevicePtr,
    peers: DevicePtr,
    slot_bytes: u64,
    n_peers: u32,
    masks: DevicePtr,
    cell: DevicePtr,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([1, 1, 1])
        .block([32, 1, 1])
        .arg_ptr(keys)
        .arg_ptr(peers)
        .arg_u64(slot_bytes)
        .arg_u32(n_peers)
        .arg_ptr(masks)
        .arg_ptr(cell)
        .launch(stream)
}
