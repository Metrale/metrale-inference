// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The paged KV cache: the `KvCacheDtype` storage formats, the
//! per-layer block geometry in `KvCacheConfig`, and the `PagedKvCache` block
//! pool with its per-block reference counts.
//!
//! Owner: cache.
//! Invariants:
//! - Until `release`, every attention layer `i < num_layers` has its own K pool
//!   and V pool of `num_blocks` blocks, sized by that layer's K-side and V-side
//!   block bytes.
//! - A block index is pushed back onto the free list only when a decrement
//!   (`dec_ref`, `return_evicted_block`) takes its count from 1 to 0. A
//!   decrement at 0 never pushes: it is logged, and `dec_ref` also panics in
//!   a debug build.

use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_kv_codec::{BlockGeometry, KvCodecId, KvPair, spec};

/// 2026-09-25: KV cache storage format. A symmetric variant uses one format
/// for K and V; a `*K*V` variant names the K format, then the V format
/// (`kv_pair`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvCacheDtype {
    /// 2026-09-25: BF16, 2 bytes per element.
    Bf16,
    /// 2026-09-25: FP8 E4M3, 1 byte per element, with one scale per tensor
    /// (the `k_scale`/`v_scale` arguments of `reshape_and_cache_flash_fp8`).
    Fp8,
    /// 2026-09-25: E2M1 values, two per byte, plus one FP8 E4M3 scale byte per
    /// 16-element group: 4.5 bits per element.
    Nvfp4,
    /// 2026-09-25: WHT-rotated 16-level Lloyd-Max codebook (`TURBO4_LUT`) in
    /// the `Nvfp4` byte layout.
    Turbo4,
    /// 2026-09-25: WHT-rotated 8-level Lloyd-Max codebook (`TURBO3_LUT`), 8
    /// values in 3 bytes plus one FP8 scale byte per group: 3.5 bits per
    /// element, 22% smaller than `Turbo4`.
    Turbo3,
    /// 2026-09-25: WHT-rotated 4-level codebook, 4 values per byte plus one FP8
    /// scale byte per group: 2.5 bits per element, 6.4x smaller than `Bf16`.
    /// Its registry row (`AutoBoundary::FourFifths`) gives this dtype a larger
    /// high-precision boundary than the other symmetric turbo dtypes.
    Turbo2,
    /// 2026-09-25: WHT-rotated FP8 E4M3 data plus one BF16 scale per group: 9
    /// bits per element, 12.5% more than `Fp8`.
    Turbo8,
    /// 2026-09-25: K `Turbo4`, V `Turbo3`: 8 bits per K+V element pair, against
    /// 9 for `Turbo4`.
    Turbo4KTurbo3V,
    /// 2026-09-25: K `Turbo4`, V `Turbo8`: 13.5 bits per K+V element pair,
    /// against 18 for `Turbo8`.
    Turbo4KTurbo8V,
    /// 2026-09-25: K `Turbo3`, V `Turbo8`.
    Turbo3KTurbo8V,
    /// 2026-09-25: K `Bf16`, V `Turbo4`.
    Bf16KTurbo4V,
    /// 2026-09-25: K `Bf16`, V `Turbo3`.
    Bf16KTurbo3V,
    /// 2026-09-25: K `Fp8`, V `Turbo4`.
    Fp8KTurbo4V,
    /// 2026-09-25: K `Fp8`, V `Turbo3`.
    Fp8KTurbo3V,
    /// 2026-09-25: K `Bf16`, V `Turbo2`.
    Bf16KTurbo2V,
    /// 2026-09-25: K `Fp8`, V `Turbo2`.
    Fp8KTurbo2V,
}

impl std::fmt::Display for KvCacheDtype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // 2026-09-25: The spelling lives only in `name` (catalog.rs), whose
        // match has no wildcard arm.
        f.write_str(self.name())
    }
}

impl KvCacheDtype {
    /// 2026-10-18: The registry codec of each side. This is the only place a legacy variant
    /// meets the registry; every property below is read from the codecs' specs.
    pub const fn pair(self) -> (KvCodecId, KvCodecId) {
        match self {
            KvCacheDtype::Bf16 => (KvCodecId::Bf16, KvCodecId::Bf16),
            KvCacheDtype::Fp8 => (KvCodecId::Fp8, KvCodecId::Fp8),
            KvCacheDtype::Nvfp4 => (KvCodecId::Nvfp4, KvCodecId::Nvfp4),
            KvCacheDtype::Turbo4 => (KvCodecId::Turbo4, KvCodecId::Turbo4),
            KvCacheDtype::Turbo3 => (KvCodecId::Turbo3, KvCodecId::Turbo3),
            KvCacheDtype::Turbo2 => (KvCodecId::Turbo2, KvCodecId::Turbo2),
            KvCacheDtype::Turbo8 => (KvCodecId::Turbo8, KvCodecId::Turbo8),
            KvCacheDtype::Turbo4KTurbo3V => (KvCodecId::Turbo4, KvCodecId::Turbo3),
            KvCacheDtype::Turbo4KTurbo8V => (KvCodecId::Turbo4, KvCodecId::Turbo8),
            KvCacheDtype::Turbo3KTurbo8V => (KvCodecId::Turbo3, KvCodecId::Turbo8),
            KvCacheDtype::Bf16KTurbo4V => (KvCodecId::Bf16, KvCodecId::Turbo4),
            KvCacheDtype::Bf16KTurbo3V => (KvCodecId::Bf16, KvCodecId::Turbo3),
            KvCacheDtype::Fp8KTurbo4V => (KvCodecId::Fp8, KvCodecId::Turbo4),
            KvCacheDtype::Fp8KTurbo3V => (KvCodecId::Fp8, KvCodecId::Turbo3),
            KvCacheDtype::Bf16KTurbo2V => (KvCodecId::Bf16, KvCodecId::Turbo2),
            KvCacheDtype::Fp8KTurbo2V => (KvCodecId::Fp8, KvCodecId::Turbo2),
        }
    }

    /// 2026-10-18: The legacy variant whose `pair()` is `(k, v)`, or `None` when the pair has
    /// no legacy variant.
    pub fn from_pair(k: KvCodecId, v: KvCodecId) -> Option<Self> {
        Self::ALL.into_iter().find(|d| d.pair() == (k, v))
    }

    /// 2026-09-25: The (K, V) formats. A symmetric variant returns
    /// `(self, self)`. 2026-10-18: Derived from `pair()`.
    pub fn kv_pair(self) -> (KvCacheDtype, KvCacheDtype) {
        let (k, v) = self.pair();
        // 2026-10-18: Each side of a legacy variant is a legacy codec, which has a symmetric
        // variant (`catalog.rs` `registry_tests` checks every one).
        let side =
            |c| Self::from_pair(c, c).expect("a legacy side codec has a symmetric legacy variant");
        (side(k), side(v))
    }

    /// 2026-09-25: True for the symmetric turbo formats, whose cache contents
    /// are in the WHT-rotated basis: `write_kv_cache.rs` runs
    /// `wht_bf16_inplace` on K and V before the turbo write. The decode
    /// attention calls it on each side of `kv_pair()`, applying the WHT to Q
    /// when K is rotated and the inverse WHT to the output when V is rotated.
    /// A `*K*V` variant itself returns false, so call it on a side.
    /// 2026-10-18: Read from the codec spec's rotation.
    pub fn is_wht_rotated(self) -> bool {
        let (k, v) = self.pair();
        k == v && spec(k).rotates()
    }

    /// 2026-09-25: True if K and V use different formats.
    pub fn is_asymmetric(self) -> bool {
        let (k, v) = self.pair();
        k != v
    }
}

impl std::str::FromStr for KvCacheDtype {
    type Err = metrale_error::Error;
    /// 2026-10-18: Parsed by the registry (`KvPair`), whose error lists every accepted
    /// spelling.
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        let pair: KvPair = s.parse()?;
        Self::from_pair(pair.k(), pair.v()).ok_or_else(|| {
            metrale_error::error!(
                metrale_error::ErrorCode::KvCodecUnknown,
                s.to_owned(),
                metrale_kv_codec::accepted()
            )
        })
    }
}

/// 2026-09-25: Geometry and formats of the paged KV cache.
pub struct KvCacheConfig {
    /// 2026-09-25: Tokens per block.
    pub block_size: usize,
    /// 2026-09-25: KV heads per layer, where `layer_dims` does not override it.
    pub num_kv_heads: usize,
    /// 2026-09-25: Head dimension, where `layer_dims` does not override it.
    pub head_dim: usize,
    /// 2026-09-25: Attention layers that get a K pool and a V pool.
    pub num_layers: usize,
    /// 2026-09-25: Format of every layer that `layer_dtypes` does not cover.
    pub dtype: KvCacheDtype,
    /// 2026-09-25: Per-layer format: layer `i` uses `layer_dtypes[i]` when
    /// `i < layer_dtypes.len()`, else `dtype`.
    pub layer_dtypes: Vec<KvCacheDtype>,
    /// 2026-09-25: Per-layer `(num_kv_heads, head_dim)`, by the same rule as
    /// `layer_dtypes`, for models whose attention layers differ in shape (the
    /// Gemma-4 loader's `kv_layer_dims`). Pool sizes and block strides use it.
    pub layer_dims: Vec<(usize, usize)>,
    /// 2026-09-25: Per-sequence cap on HBM-resident blocks under
    /// `--high-speed-swap`. With `Some(N)` the model engine slides a
    /// sequence's window at `N` blocks (`ensure_blocks_through_decode`);
    /// `None` means no cap.
    pub cache_blocks_per_seq: Option<u32>,
    /// 2026-10-16: V-side bytes per block of every layer, overriding the size its format and
    /// dims give; `None` keeps that size. A model whose V side holds something other than V
    /// rows sets it (the FP8 sparse-attention indexer layout).
    pub v_block_bytes: Option<usize>,
}

impl KvCacheConfig {
    /// 2026-09-25: The format of attention layer `layer_idx`.
    pub fn dtype_for_layer(&self, layer_idx: usize) -> KvCacheDtype {
        if layer_idx < self.layer_dtypes.len() {
            self.layer_dtypes[layer_idx]
        } else {
            self.dtype
        }
    }

    /// 2026-09-25: Bytes of one side of one block for `dtype` and
    /// `(nkv, hd)`. A `*K*V` variant gives its K-side size.
    fn block_bytes_dims(&self, dtype: KvCacheDtype, nkv: usize, hd: usize) -> usize {
        // 2026-10-18: Derived from the K-side codec's registry spec; the kv-codec layout golden
        // test holds every codec to the sizes the per-variant match here used to compute.
        side_bytes(dtype.pair().0, self.block_size, nkv, hd)
    }

    /// 2026-09-25: V-side bytes per block: `block_bytes_dims` of
    /// `dtype.kv_pair().1`.
    #[allow(dead_code)]
    pub fn v_block_bytes_dims(&self, dtype: KvCacheDtype, nkv: usize, hd: usize) -> usize {
        let (_, v) = dtype.kv_pair();
        self.block_bytes_dims(v, nkv, hd)
    }

    /// 2026-09-25: K-side bytes per block: `block_bytes_dims` of
    /// `dtype.kv_pair().0` (for `Bf16KTurbo3V`, the BF16 size).
    #[allow(dead_code)]
    pub fn k_block_bytes_dims(&self, dtype: KvCacheDtype, nkv: usize, hd: usize) -> usize {
        let (k, _) = dtype.kv_pair();
        self.block_bytes_dims(k, nkv, hd)
    }

    /// 2026-09-25: K-side bytes per block of attention layer `layer_idx`, from
    /// its dims and the K side of its format. `PagedKvCache::new` sizes the K
    /// pool with it.
    pub fn k_block_bytes_for_layer(&self, layer_idx: usize) -> usize {
        let (nkv, hd) = self.dims_for_layer(layer_idx);
        let (k, _) = self.dtype_for_layer(layer_idx).kv_pair();
        self.block_bytes_dims(k, nkv, hd)
    }

    /// 2026-09-25: V-side bytes per block of attention layer `layer_idx`.
    /// Equals `k_block_bytes_for_layer` for a symmetric format. 2026-10-16: `v_block_bytes`
    /// when set.
    pub fn v_block_bytes_for_layer(&self, layer_idx: usize) -> usize {
        if let Some(bytes) = self.v_block_bytes {
            return bytes;
        }
        let (nkv, hd) = self.dims_for_layer(layer_idx);
        let (_, v) = self.dtype_for_layer(layer_idx).kv_pair();
        self.block_bytes_dims(v, nkv, hd)
    }

    /// 2026-09-25: `block_bytes_dims` with the global `num_kv_heads` and
    /// `head_dim`, ignoring `layer_dims`.
    fn block_bytes_for_dtype(&self, dtype: KvCacheDtype) -> usize {
        self.block_bytes_dims(dtype, self.num_kv_heads, self.head_dim)
    }

    /// 2026-09-25: Bytes of one side of one block with the uniform `dtype` and
    /// the global dims. For a `*K*V` dtype this is the K side.
    pub fn block_bytes(&self) -> usize {
        self.block_bytes_for_dtype(self.dtype)
    }

    /// 2026-09-25: `(num_kv_heads, head_dim)` of attention layer `layer_idx`:
    /// `layer_dims[layer_idx]` when present, else the global values.
    pub fn dims_for_layer(&self, layer_idx: usize) -> (usize, usize) {
        if layer_idx < self.layer_dims.len() {
            self.layer_dims[layer_idx]
        } else {
            (self.num_kv_heads, self.head_dim)
        }
    }

    /// 2026-09-25: Bytes of one side of one block of attention layer
    /// `layer_idx`, from its own dims and format. For a `*K*V` format this is
    /// the K side.
    pub fn block_bytes_for_layer(&self, layer_idx: usize) -> usize {
        let (nkv, hd) = self.dims_for_layer(layer_idx);
        self.block_bytes_dims(self.dtype_for_layer(layer_idx), nkv, hd)
    }

    /// 2026-09-25: Twice `block_bytes()`. That is K + V for a symmetric
    /// uniform dtype, but twice the K side for a `*K*V` dtype.
    pub fn block_bytes_kv(&self) -> usize {
        self.block_bytes() * 2
    }

    /// 2026-09-25: K plus V bytes of one block index summed over all
    /// layers, per layer's own dims and K and V formats.
    pub fn block_bytes_kv_all_layers(&self) -> usize {
        (0..self.num_layers)
            .map(|i| self.k_block_bytes_for_layer(i) + self.v_block_bytes_for_layer(i))
            .sum()
    }

    /// 2026-09-25: Elements per block and side, from the global dims:
    /// `block_size * num_kv_heads * head_dim`.
    pub fn cache_stride_elements(&self) -> usize {
        self.block_size * self.num_kv_heads * self.head_dim
    }

    /// 2026-10-18: Values per block and side with the global dims.
    fn values(&self) -> u64 {
        geometry(self.block_size, self.num_kv_heads, self.head_dim).values()
    }

    /// 2026-10-18: Packed-value bytes per block of `codec`, from the global dims.
    fn data_bytes_of(&self, codec: KvCodecId) -> usize {
        to_usize(spec(codec).data_bytes(self.values()))
    }

    /// 2026-10-18: Side-tensor bytes per block of `codec`, from the global dims.
    fn scale_bytes_of(&self, codec: KvCodecId) -> usize {
        to_usize(spec(codec).side_bytes(self.values()))
    }

    /// 2026-09-25: NVFP4 data bytes per block (two E2M1 values per byte),
    /// from the global dims.
    pub fn nvfp4_data_bytes(&self) -> usize {
        self.data_bytes_of(KvCodecId::Nvfp4)
    }

    /// 2026-09-25: NVFP4 scale bytes per block (one FP8 byte per group), from
    /// the global dims.
    pub fn nvfp4_scale_bytes(&self) -> usize {
        self.scale_bytes_of(KvCodecId::Nvfp4)
    }

    /// 2026-09-25: Turbo4 data bytes per block, equal to the NVFP4 value.
    pub fn turbo4_data_bytes(&self) -> usize {
        self.data_bytes_of(KvCodecId::Turbo4)
    }

    /// 2026-09-25: Turbo4 scale bytes per block, equal to the NVFP4 value.
    pub fn turbo4_scale_bytes(&self) -> usize {
        self.scale_bytes_of(KvCodecId::Turbo4)
    }

    /// 2026-09-25: Turbo3 data bytes per block (8 values in 3 bytes).
    pub fn turbo3_data_bytes(&self) -> usize {
        self.data_bytes_of(KvCodecId::Turbo3)
    }

    /// 2026-09-25: Turbo3 scale bytes per block, equal to the NVFP4 value.
    pub fn turbo3_scale_bytes(&self) -> usize {
        self.scale_bytes_of(KvCodecId::Turbo3)
    }

    /// 2026-09-25: Turbo2 data bytes per block (4 values per byte).
    pub fn turbo2_data_bytes(&self) -> usize {
        self.data_bytes_of(KvCodecId::Turbo2)
    }

    /// 2026-09-25: Turbo2 scale bytes per block, equal to the NVFP4 value.
    pub fn turbo2_scale_bytes(&self) -> usize {
        self.scale_bytes_of(KvCodecId::Turbo2)
    }

    /// 2026-09-25: Turbo8 data bytes per block (one FP8 byte per element).
    pub fn turbo8_data_bytes(&self) -> usize {
        self.data_bytes_of(KvCodecId::Turbo8)
    }

    /// 2026-09-25: Turbo8 scale bytes per block: one 2-byte BF16 scale per
    /// group, twice the NVFP4 value.
    pub fn turbo8_scale_bytes(&self) -> usize {
        self.scale_bytes_of(KvCodecId::Turbo8)
    }
}

/// 2026-10-18: Bytes of one side of one block of `codec` at `(block_size, nkv, hd)`, from the
/// codec's registry spec.
pub fn side_bytes(codec: KvCodecId, block_size: usize, nkv: usize, hd: usize) -> usize {
    to_usize(spec(codec).block_bytes(&geometry(block_size, nkv, hd)))
}

/// 2026-10-18: The registry geometry of `(block_size, nkv, hd)`. A dimension past `u32` is a
/// configuration no model has, so it is an invariant violation, not a mode.
fn geometry(block_size: usize, nkv: usize, hd: usize) -> BlockGeometry {
    let dim = |d: usize| u32::try_from(d).expect("a KV block dimension fits u32");
    BlockGeometry {
        block_size: dim(block_size),
        num_kv_heads: dim(nkv),
        head_dim: dim(hd),
    }
}

/// 2026-10-18: A byte count as `usize`; it indexes host memory, so it fits on any host that
/// could allocate it.
fn to_usize(bytes: u64) -> usize {
    usize::try_from(bytes).expect("a KV block byte count fits usize")
}

/// 2026-09-25: One attention layer's K and V pools.
struct LayerPool {
    k_pool: DevicePtr,
    v_pool: DevicePtr,
    /// 2026-09-25: Bytes between K blocks; differs from V for a `*K*V` format.
    k_block_stride: usize,
    /// 2026-09-25: Bytes between V blocks.
    v_block_stride: usize,
    /// 2026-09-25: This layer's format.
    dtype: KvCacheDtype,
}

/// 2026-09-25: Paged KV cache across all attention layers: the pools and
/// the block free list with per-block reference counts.
pub struct PagedKvCache {
    layers: Vec<LayerPool>,
    num_blocks: usize,
    free_blocks: Vec<u32>,
    /// 2026-09-25: Per-block reference count, so a block can be shared (the
    /// prefix cache holds one). Set to 1 on alloc; the block returns to the
    /// free list when a decrement reaches 0.
    block_ref_counts: Vec<u32>,
    config: KvCacheConfig,
    /// 2026-09-25: Per-block refcount event history (`METRALE_KV_TRACE=1`;
    /// empty otherwise).
    trace: block_trace::BlockTrace,
}

mod block_trace;
mod catalog;
mod paged_impl;
/// 2026-09-25: Release both pools of every layer, and clear the free list and
/// ref counts with them, so a released cache cannot hand out a block of freed
/// memory: `alloc_block` then fails. Every pool is freed even after an error;
/// the first error is returned.
impl metrale_core::scope::ModelResource<dyn metrale_gpu_runtime::gpu::GpuBackend> for PagedKvCache {
    fn label(&self) -> &'static str {
        "kv cache"
    }

    fn release(&mut self, gpu: &dyn metrale_gpu_runtime::gpu::GpuBackend) -> anyhow::Result<()> {
        let mut first_error = None;
        for layer in self.layers.drain(..) {
            for ptr in [layer.k_pool, layer.v_pool] {
                if let Err(e) = gpu.free(ptr)
                    && first_error.is_none()
                {
                    first_error = Some(e);
                }
            }
        }
        self.free_blocks.clear();
        self.block_ref_counts.clear();
        self.num_blocks = 0;
        match first_error {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod tests_tq_plus;
