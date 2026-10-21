// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-16: The FP8 indexer cache (`--indexer-cache-dtype fp8`): its block layout, how many
//! open pools its BF16 ring keeps, and its kernels (`kernels/gb10/common/dsa_indexer_fp8.cu`).
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - [`Fp8IndexerLayout`]'s offsets are the kernel's `DsaFp8Block` offsets: one DSA layer's V
//!   block of `bs` tokens holds `E = bs / KP` E4M3 pool entries of `D` bytes, then `E` f32
//!   scales, then the ring's `S x (KP-1)` BF16 key rows and as many gate rows.
//! - A complete pool's entry is written by the store of the token that completes it; a token
//!   of an open pool sits in ring set `(x / KP) % S`. The ring therefore holds the open rows
//!   of the newest `S` pools, so a rewind that reopens a pool older than that loses rows:
//!   [`rewind_keeps_tail`] says which rewinds are safe and the state refuses the others.
//! - [`IndexerRowFormat::Bf16`] is the layout every serve has always used; nothing in this
//!   module runs for it.

use anyhow::{Context, Result, bail, ensure};
use metrale_config::{IndexerCacheDtype, ModelConfig};
use metrale_gpu_runtime::gpu::{GpuBackend, KernelHandle};
use metrale_kv_codec::{IndexerPlan, KvCodecId, spec};

use super::paged::IndexerCache;

/// 2026-10-16: Module of the FP8 cache's kernels: `dsa_indexer_fp8.cu`, named by its file stem.
pub const DSA_FP8_MODULE: &str = "dsa_indexer_fp8";

/// 2026-10-18: The registry codec of the FP8 indexer rows.
const FP8_ROW: KvCodecId = match IndexerCacheDtype::Fp8.codec() {
    Some(id) => id,
    None => panic!("--indexer-cache-dtype fp8 names no registry codec"),
};

/// 2026-10-16: The channel count the FP8 kernels are compiled for (`DSA_FP8_D`): one thread per
/// channel, a 128-point rotation. 2026-10-18: The one head dim the `fp8_row` spec serves.
pub const FP8_INDEX_HEAD_DIM: usize = spec(FP8_ROW).head_dims[0] as usize;

/// 2026-10-16: How a DSA layer stores its indexer rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexerRowFormat {
    /// 2026-10-16: Per-token BF16 key and gate rows (`PagedIndexerLayout`, or a flat buffer).
    Bf16,
    /// 2026-10-21: [`Self::Bf16`]'s paged rows plus each complete pool's f32 key, persisted in
    /// the block when its last row is stored (`paged::pooled_block_bytes`), so the decode's
    /// selection reads them instead of recompressing every pool (`METRALE_GLM_DSA_POOL_KEYS`).
    Bf16Pooled,
    /// 2026-10-16: Pooled E4M3 keys with a ring of `tail_sets` open pools.
    Fp8 { tail_sets: usize },
}

impl IndexerRowFormat {
    /// 2026-10-16: The text stack's format from `--indexer-cache-dtype`, its ring sized for
    /// `config.indexer_rewind_rows`. FP8 lives in the KV blocks, so a flat cache refuses it.
    /// 2026-10-21: BF16 rows of a paged cache keep their pool keys when `pool_keys`
    /// (`METRALE_GLM_DSA_POOL_KEYS`); a flat cache never does.
    pub fn for_text_stack(
        config: &ModelConfig,
        cache: IndexerCache,
        pool_keys: bool,
    ) -> Result<Self> {
        match config.indexer_cache_dtype {
            IndexerCacheDtype::Bf16 if pool_keys && cache == IndexerCache::Paged => {
                Ok(Self::Bf16Pooled)
            }
            IndexerCacheDtype::Bf16 => Ok(Self::Bf16),
            IndexerCacheDtype::Fp8 => {
                ensure!(
                    cache == IndexerCache::Paged,
                    "--indexer-cache-dtype fp8 needs the paged indexer cache; \
                     METRALE_GLM_DSA_INDEXER_FLAT=1 keeps the rows in a per-sequence buffer"
                );
                Ok(Self::Fp8 {
                    tail_sets: tail_sets_for(config.indexer_rewind_rows, config.index_kpool)?,
                })
            }
        }
    }

    /// 2026-10-16: The bytes one DSA layer's V block needs in this format, or `None` for BF16
    /// (the KV cache's own V block, as always).
    pub fn v_block_bytes(&self, block_size: usize, d: usize, kp: usize) -> Result<Option<usize>> {
        match *self {
            Self::Bf16 => Ok(None),
            Self::Bf16Pooled => Ok(Some(super::paged::pooled_block_bytes(block_size, d, kp)?)),
            Self::Fp8 { tail_sets } => Ok(Some(Fp8IndexerLayout::block_bytes(
                block_size, d, kp, tail_sets,
            )?)),
        }
    }

    /// 2026-10-16: The text stack's format, refusing FP8 on a target built without its kernels.
    /// 2026-10-17: `cache` is `paged::text_stack_indexer_cache` of the load's settings.
    /// 2026-10-21: `pool_keys_ready`: whether the pool-key kernels resolved; the pooled format
    /// refuses without them (the loader sized the V blocks for it already).
    pub fn text_stack(
        config: &ModelConfig,
        kernels: &DsaFp8Kernels,
        cache: super::paged::IndexerCache,
        pool_keys: bool,
        pool_keys_ready: bool,
    ) -> Result<Self> {
        let format = Self::for_text_stack(config, cache, pool_keys)?;
        ensure!(
            !format.is_fp8() || kernels.ready(),
            "--indexer-cache-dtype fp8: this build has no {DSA_FP8_MODULE} kernels"
        );
        ensure!(
            format != Self::Bf16Pooled || pool_keys_ready,
            "METRALE_GLM_DSA_POOL_KEYS: this build has no dsa_pool_keys_store_rows / \
             dsa_index_scores_warp_paged kernels; set METRALE_GLM_DSA_POOL_KEYS=0"
        );
        Ok(format)
    }

    /// 2026-10-16: The text stack's V block bytes for KV blocks of `block_size` tokens (the
    /// loader's `kv_v_block_bytes`): `None` for BF16.
    pub fn text_stack_v_block_bytes(
        config: &ModelConfig,
        block_size: usize,
        cache: super::paged::IndexerCache,
        pool_keys: bool,
    ) -> Result<Option<usize>> {
        Self::for_text_stack(config, cache, pool_keys)?.v_block_bytes(
            block_size,
            config.index_head_dim,
            config.index_kpool,
        )
    }

    /// 2026-10-21: The MTP drafter's format: its flat cache keeps persisted pool keys under
    /// `pool_keys` (`METRALE_GLM_DSA_POOL_KEYS`), so its captured draft selection scores them
    /// instead of recompressing every pool per draft; refused without the pool-key kernels.
    pub fn drafter(pool_keys: bool, pool_keys_ready: bool) -> Result<Self> {
        if !pool_keys {
            return Ok(Self::Bf16);
        }
        ensure!(
            pool_keys_ready,
            "METRALE_GLM_DSA_POOL_KEYS: this build has no dsa_pool_keys_store_rows / \
             dsa_index_scores_warp_paged kernels for the MTP drafter; set \
             METRALE_GLM_DSA_POOL_KEYS=0"
        );
        Ok(Self::Bf16Pooled)
    }

    pub fn is_fp8(&self) -> bool {
        matches!(self, Self::Fp8 { .. })
    }

    /// 2026-10-16: The ring a state in this format guards, for pools of `kp` tokens.
    pub fn ring(&self, kp: usize) -> Option<Fp8Ring> {
        match *self {
            Self::Bf16 | Self::Bf16Pooled => None,
            Self::Fp8 { tail_sets } => Some(Fp8Ring { kp, tail_sets }),
        }
    }
}

/// 2026-10-16: Ring sets that keep every open row a rewind of up to `rewind_rows` rows can
/// reopen: `S = (R + KP - 2) / KP + 1` (R 0 or 1 gives 1). A rewind of R rows from length L
/// reopens pools from `(L - R) / KP` to `(L - 1) / KP`, at most `S` of them.
pub fn tail_sets_for(rewind_rows: usize, kp: usize) -> Result<usize> {
    ensure!(
        kp >= 2,
        "FP8 indexer cache: index_kpool {kp} must be at least 2"
    );
    Ok((rewind_rows + kp).saturating_sub(2) / kp + 1)
}

/// 2026-10-16: Whether rewinding a sequence from `len` rows to `n` keeps every row of the pool
/// `n` reopens: growing, a pool boundary, or a reopened pool among the newest `sets`. Blocks
/// are ignored: a pool `sets` newer in the next block does not share the slot, but is refused
/// anyway (conservative; the serve sizes `sets` so its rewinds stay within the newest pools).
pub fn rewind_keeps_tail(len: usize, n: usize, kp: usize, sets: usize) -> bool {
    n >= len || n.is_multiple_of(kp) || (len - 1) / kp - n / kp < sets
}

/// 2026-10-16: What a sequence's FP8 state needs to refuse a rewind or an adoption that would
/// read a ring row that is no longer there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fp8Ring {
    pub kp: usize,
    pub tail_sets: usize,
}

impl Fp8Ring {
    /// 2026-10-16: Refuses a rewind from `len` to `n` that [`rewind_keeps_tail`] rejects.
    pub fn check_rewind(&self, len: usize, n: usize) -> Result<()> {
        if !rewind_keeps_tail(len, n, self.kp, self.tail_sets) {
            bail!(
                "FP8 DSA indexer: rewinding from {len} to {n} rows reopens pool {} but the ring \
                 keeps the open rows of the newest {} pools only; the serve sized it for a \
                 shorter rewind (--indexer-cache-dtype fp8 follows the drafter's verify width)",
                n / self.kp,
                self.tail_sets
            );
        }
        Ok(())
    }

    /// 2026-10-16: Refuses adopting rows `[0, seq_len)` into an empty state when `seq_len`
    /// ends inside a pool: that pool's open rows would be another sequence's ring slots, which
    /// it may have reused since.
    pub fn check_adopt(&self, len: usize, seq_len: usize) -> Result<()> {
        if len == 0 && seq_len > 0 && !seq_len.is_multiple_of(self.kp) {
            bail!(
                "FP8 DSA indexer: a shared prefix of {seq_len} rows ends inside a pool of {}; \
                 its open rows are not in this sequence's ring",
                self.kp
            );
        }
        Ok(())
    }
}

/// 2026-10-16: The FP8 rows inside one DSA layer's V block (see the module invariants).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fp8IndexerLayout {
    pub block_size: usize,
    /// 2026-10-16: Bytes between V blocks of this layer (`v_block_stride_bytes_for_layer`).
    pub block_stride_bytes: usize,
    pub index_head_dim: usize,
    pub kp: usize,
    pub tail_sets: usize,
}

impl Fp8IndexerLayout {
    /// 2026-10-16: `E * (D + 4) + 2 * S * (KP - 1) * D * 2` bytes; refuses a geometry the
    /// kernels do not take (D other than 128, a block that is not whole pools).
    /// 2026-10-18: The bytes and the pool and ring rules are the registry's `fp8_row`
    /// [`IndexerPlan`], so the indexer is sized from the codec spec like the attention cache.
    pub fn block_bytes(block_size: usize, d: usize, kp: usize, sets: usize) -> Result<usize> {
        if d != FP8_INDEX_HEAD_DIM {
            bail!(
                "--indexer-cache-dtype fp8: index_head_dim is {d}, but dsa_indexer_fp8.cu is \
                 compiled for {FP8_INDEX_HEAD_DIM} channels; use --indexer-cache-dtype bf16"
            );
        }
        let plan = IndexerPlan::new(
            FP8_ROW,
            u32::try_from(d)?,
            u32::try_from(kp)?,
            u32::try_from(sets)?,
        )?;
        let bytes = plan
            .block_bytes(u32::try_from(block_size)?)
            .context("--indexer-cache-dtype fp8: a KV block holds whole pools")?;
        Ok(usize::try_from(bytes)?)
    }

    /// 2026-10-16: Refuses a V block smaller than [`Self::block_bytes`] or a stride the f32
    /// scales cannot align to.
    pub fn new(
        block_size: usize,
        block_stride_bytes: usize,
        index_head_dim: usize,
        kp: usize,
        tail_sets: usize,
    ) -> Result<Self> {
        let need = Self::block_bytes(block_size, index_head_dim, kp, tail_sets)?;
        if block_stride_bytes < need || !block_stride_bytes.is_multiple_of(4) {
            bail!(
                "FP8 DSA indexer: a KV block's V side is {block_stride_bytes} B but {block_size} \
                 tokens with {tail_sets} ring sets need {need} B (a multiple of 4)"
            );
        }
        Ok(Self {
            block_size,
            block_stride_bytes,
            index_head_dim,
            kp,
            tail_sets,
        })
    }

    fn entries(&self) -> usize {
        self.block_size / self.kp
    }

    /// 2026-10-16: In-block byte offset of the E4M3 entry of token `x`'s pool.
    pub fn entry_off(&self, x: usize) -> usize {
        ((x % self.block_size) / self.kp) * self.index_head_dim
    }

    /// 2026-10-16: In-block byte offset of the f32 scale of token `x`'s pool.
    pub fn scale_off(&self, x: usize) -> usize {
        self.entries() * self.index_head_dim + 4 * ((x % self.block_size) / self.kp)
    }

    /// 2026-10-16: In-block byte offset of token `x`'s ring key row (`x % KP < KP - 1`).
    pub fn ring_key_off(&self, x: usize) -> usize {
        let slot = ((x / self.kp) % self.tail_sets) * (self.kp - 1) + x % self.kp;
        self.entries() * (self.index_head_dim + 4) + slot * self.index_head_dim * 2
    }

    /// 2026-10-16: In-block byte offset of token `x`'s ring gate row.
    pub fn ring_gate_off(&self, x: usize) -> usize {
        self.ring_key_off(x) + self.tail_sets * (self.kp - 1) * self.index_head_dim * 2
    }
}

/// 2026-10-16: The entries of `dsa_indexer_fp8.cu`, resolved with `try_kernel`; all
/// `KernelHandle(0)` on a target without them ([`Self::ready`] is then false and an FP8 serve
/// refuses to start).
#[derive(Clone, Copy, Debug)]
pub struct DsaFp8Kernels {
    pub store: KernelHandle,
    pub compress: KernelHandle,
    pub pool_scores: KernelHandle,
}

impl DsaFp8Kernels {
    /// 2026-10-16: No kernels, for fixtures that never run an FP8 layer.
    pub const NONE: Self = Self {
        store: KernelHandle(0),
        compress: KernelHandle(0),
        pool_scores: KernelHandle(0),
    };

    pub fn resolve(gpu: &dyn GpuBackend) -> Self {
        let k = |name: &str| metrale_model_layers::layers::try_kernel(gpu, DSA_FP8_MODULE, name);
        Self {
            store: k("dsa_indexer_store_fp8"),
            compress: k("dsa_kpool_compress_fp8"),
            pool_scores: k("dsa_pool_scores_rows_fp8"),
        }
    }

    pub fn ready(&self) -> bool {
        [self.store, self.compress, self.pool_scores]
            .iter()
            .all(|k| k.0 != 0)
    }
}

#[cfg(test)]
#[path = "fp8_rows_tests.rs"]
mod tests;
