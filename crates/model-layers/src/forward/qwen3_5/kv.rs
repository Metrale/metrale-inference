// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: The KV cache of the Metal Qwen3.5 forward: the contiguous per-sequence buffers of
//! one full-attention layer, the kernels each codec pair runs (from the codec crate's Metal
//! table), and the append-then-attend step. The pair's specs decide every branch (bf16 storage,
//! rotation, scale buffers); nothing here names a codec.
//!
//! Owner: model-layers.
//! Invariants:
//! - A cache is allocated only for a pair the Metal kernel table serves.
//! - A side is WHT-rotated before the append exactly when its spec rotates; Q is rotated when K
//!   is, and the attention output gets the inverse WHT when V is.
//! - A side has a scale buffer exactly when its spec has side tensors.

use anyhow::Result;
use metrale_error::{ErrorCode, error};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelArg, KernelHandle};
use metrale_kv_codec::{
    KvPair, ResolvedKvKernels, contiguous_kv_kernel_names, contiguous_kv_pairs, spec,
};

/// 2026-10-18: The Metal KV kernels of every pair the Metal table serves, resolved once at
/// startup so a missing kernel fails there, whichever pair the run uses.
pub struct MetalKvKernels {
    pairs: Vec<(KvPair, ResolvedKvKernels<KernelHandle>)>,
}

impl MetalKvKernels {
    /// 2026-10-18: Looks up the append, decode and WHT kernels of every served pair.
    pub fn resolve(gpu: &dyn GpuBackend) -> Result<Self> {
        let pairs = contiguous_kv_pairs()
            .map(|pair| {
                let names = contiguous_kv_kernel_names(pair)?;
                Ok((pair, names.try_map(|n| gpu.kernel(n.module, n.func))?))
            })
            .collect::<Result<_>>()?;
        Ok(Self { pairs })
    }

    /// 2026-10-18: The kernels of `pair`; a pair the Metal table does not serve is refused with
    /// `KvCapabilityMissing`.
    pub fn get(&self, pair: KvPair) -> Result<&ResolvedKvKernels<KernelHandle>> {
        let found = self.pairs.iter().find(|(p, _)| *p == pair);
        let missing = || {
            error!(
                ErrorCode::KvCapabilityMissing,
                pair.name(),
                "Metal KV kernels"
            )
        };
        Ok(&found.ok_or_else(missing)?.1)
    }
}

/// 2026-10-18: Bytes of `(K data, V data, K scales, V scales)` for `max_seq` tokens of `kv_dim`
/// elements per side, from each side's spec.
pub fn layer_kv_bytes(pair: KvPair, max_seq: u32, kv_dim: u32) -> (u64, u64, u64, u64) {
    let n = u64::from(max_seq) * u64::from(kv_dim);
    let (k, v) = (spec(pair.k()), spec(pair.v()));
    (
        k.data_bytes(n),
        v.data_bytes(n),
        k.side_bytes(n),
        v.side_bytes(n),
    )
}

/// 2026-09-25: KV cache of one full-attention layer, for one sequence.
///
/// 2026-10-18: `pair` selects the storage format. A quantized side holds packed data in `k`/`v`
/// and its group scales in `k_scales`/`v_scales`, sized by [`layer_kv_bytes`].
pub struct LayerKvCache {
    pub k: DevicePtr,
    pub v: DevicePtr,
    /// 2026-09-25: Capacity in tokens: the `max_seq` given to [`Self::alloc`].
    #[allow(dead_code)]
    pub capacity: u32,
    pub pair: KvPair,
    /// 2026-10-18: Group-scale buffers, `Some` exactly for a side whose spec has side tensors.
    pub k_scales: Option<DevicePtr>,
    pub v_scales: Option<DevicePtr>,
}

impl LayerKvCache {
    /// 2026-10-18: Allocates a cache of `max_seq` tokens by `kv_dim` elements holding `pair`.
    /// Refuses a pair the Metal table does not serve, and a quantized side whose group does not
    /// divide `kv_dim`.
    pub fn alloc(gpu: &dyn GpuBackend, pair: KvPair, max_seq: u32, kv_dim: u32) -> Result<Self> {
        contiguous_kv_kernel_names(pair)?;
        for side in [pair.k(), pair.v()] {
            let s = spec(side);
            anyhow::ensure!(
                s.group == 0 || kv_dim.is_multiple_of(s.group),
                "{} KV needs KV_DIM divisible by {}, got {kv_dim}",
                s.name,
                s.group
            );
        }
        let (kb, vb, ksb, vsb) = layer_kv_bytes(pair, max_seq, kv_dim);
        let alloc = |bytes: u64| -> Result<DevicePtr> { gpu.alloc(usize::try_from(bytes)?) };
        let alloc_opt = |bytes: u64| -> Result<Option<DevicePtr>> {
            Ok(if bytes > 0 { Some(alloc(bytes)?) } else { None })
        };
        Ok(Self {
            k: alloc(kb)?,
            v: alloc(vb)?,
            capacity: max_seq,
            pair,
            k_scales: alloc_opt(ksb)?,
            v_scales: alloc_opt(vsb)?,
        })
    }
}

/// 2026-10-18: The per-token buffers the KV step reads and writes: post-norm Q, the post-RoPE K,
/// V, and the attention output.
pub(super) struct KvStepIo {
    pub q: DevicePtr,
    pub k: DevicePtr,
    pub v: DevicePtr,
    pub out: DevicePtr,
}

/// 2026-10-18: The shape and position of one KV step.
pub(super) struct KvStepDims {
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    /// 2026-10-18: The position the append writes.
    pub cache_pos: u32,
    /// 2026-10-18: The cache length attention reads.
    pub seq_len_attn: u32,
    pub scale: f32,
    /// 2026-09-25: The quantized attention kernels skip V rows whose unnormalized weight
    /// exp(score - max) is at or below it.
    pub sparse_v: f32,
}

/// 2026-10-18: Appends this token's K and V to `kv` and runs decode attention into `io.out`. A
/// cache whose sides both store bf16 runs the plain append and attention; any other pair rotates
/// its rotated sides, appends with its scale buffers, rotates Q when K is rotated, attends, and
/// inverts the rotation of the output when V is rotated.
pub(super) fn append_and_attend(
    gpu: &dyn GpuBackend,
    kernels: &ResolvedKvKernels<KernelHandle>,
    kv: &LayerKvCache,
    io: &KvStepIo,
    d: &KvStepDims,
    stream: u64,
) -> Result<()> {
    let (ks, vs) = (spec(kv.pair.k()), spec(kv.pair.v()));
    let lead = [
        d.num_kv_heads.to_le_bytes(),
        d.head_dim.to_le_bytes(),
        d.cache_pos.to_le_bytes(),
    ];
    let attn_lead = [
        d.seq_len_attn.to_le_bytes(),
        d.num_heads.to_le_bytes(),
        d.num_kv_heads.to_le_bytes(),
        d.head_dim.to_le_bytes(),
        d.scale.to_le_bytes(),
    ];
    let mut append: Vec<KernelArg<'_>> = lead.iter().map(|b| KernelArg::Bytes(b)).collect();
    append.extend([io.k, io.v, kv.k, kv.v].map(KernelArg::Buffer));
    let mut attn: Vec<KernelArg<'_>> = attn_lead.iter().map(|b| KernelArg::Bytes(b)).collect();
    if ks.stores_bf16() && vs.stores_bf16() {
        gpu.launch_typed(
            kernels.write,
            [d.head_dim, d.num_kv_heads, 1],
            [1, 1, 1],
            0,
            stream,
            &append,
        )?;
        attn.extend([io.q, kv.k, kv.v, io.out].map(KernelArg::Buffer));
        return gpu.launch_typed(
            kernels.decode,
            [d.num_heads, 1, 1],
            [32, 1, 1],
            0,
            stream,
            &attn,
        );
    }
    let hd = d.head_dim.to_le_bytes();
    let (fwd, inv) = match kernels.wht {
        Some(wht) => (Some(wht.0), Some(wht.1)),
        None => (None, None),
    };
    let wht = |handle: Option<KernelHandle>, rows: u32, x: DevicePtr| -> Result<()> {
        let missing = || error!(ErrorCode::KvCapabilityMissing, kv.pair.name(), "the WHT");
        let args = [KernelArg::Bytes(&hd), KernelArg::Buffer(x)];
        gpu.launch_typed(
            handle.ok_or_else(missing)?,
            [rows, 1, 1],
            [32, 1, 1],
            0,
            stream,
            &args,
        )
    };
    if ks.rotates() {
        wht(fwd, d.num_kv_heads, io.k)?;
    }
    if vs.rotates() {
        wht(fwd, d.num_kv_heads, io.v)?;
    }
    let scales = [kv.k_scales, kv.v_scales].into_iter().flatten();
    append.extend(scales.clone().map(KernelArg::Buffer));
    let group = ks.group.max(vs.group);
    anyhow::ensure!(group > 0, "{} has no quantization group", kv.pair.name());
    let append_grid = [(d.num_kv_heads * d.head_dim / group).div_ceil(64), 1, 1];
    gpu.launch_typed(kernels.write, append_grid, [64, 1, 1], 0, stream, &append)?;
    if ks.rotates() {
        wht(fwd, d.num_heads, io.q)?;
    }
    let sparse_v = d.sparse_v.to_le_bytes();
    attn.push(KernelArg::Bytes(&sparse_v));
    attn.extend([io.q, kv.k, kv.v].map(KernelArg::Buffer));
    attn.extend(scales.map(KernelArg::Buffer));
    attn.push(KernelArg::Buffer(io.out));
    gpu.launch_typed(
        kernels.decode,
        [d.num_heads, 1, 1],
        [32, 1, 1],
        0,
        stream,
        &attn,
    )?;
    if vs.rotates() {
        wht(inv, d.num_heads, io.out)?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "kv_tests.rs"]
mod tests;
