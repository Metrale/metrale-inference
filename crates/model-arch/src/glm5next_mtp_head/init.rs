// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `Glm5NextMtpHead::new`: the drafter's private KV pool and its vocab-sharded draft head.
//!
//! Owner: model-arch (GLM-5.3 MTP drafter).
//! Invariants:
//! - A constructed head's `max_seq_len` is at most `max_dsa_context` of its DSA block.
//! - At `tp_world_size` in `2..=head_xchg::MAX_RANKS` the ranks' `[head_v0, head_v0 + head_n)`
//!   partition the vocab (2026-10-11: any such world, uneven splits included); otherwise
//!   `head_n` is `vocab_size` from row 0 ([`draft_head_shard`]). The range always lies inside
//!   the vocab.

use super::*;

/// 2026-10-08: `(head_v0, head_n)`, the vocab rows rank `rank` of `world` sweeps. The sweep
/// splits only where the exchange can combine the ranks' results. Elsewhere every rank sweeps
/// the whole vocab from row 0. Before 2026-10-08, TP=3 kept `head_n = vocab` but offset rank
/// `r` by `r * vocab` rows, past the end of `lm_head`.
///
/// 2026-10-11: Any world of `2..=head_xchg::MAX_RANKS` splits, not only two ranks over an even
/// vocab: the first `vocab % world` ranks take one row more. Until then TP=3 swept all 154,856
/// BF16 rows (1.27 GB) on every rank per draft, and the FP8 copy was built but never read.
pub(super) fn draft_head_shard(vocab: usize, world: usize, rank: usize) -> (usize, usize) {
    if (2..=super::head_xchg::MAX_RANKS).contains(&world) && rank < world && vocab >= world {
        let (base, extra) = (vocab / world, vocab % world);
        (
            rank * base + rank.min(extra),
            base + usize::from(rank < extra),
        )
    } else {
        (0, vocab)
    }
}

/// 2026-10-11: `METRALE_GLM_MTP_EH_TP=1`: each rank of a tensor-parallel propose computes its
/// slice of the drafter's `eh_proj` output rows and one all-reduce of the hidden row assembles
/// them (the other ranks' rows are zero, so the sum is exact). The `[hidden, 2 * hidden]` BF16
/// projection (67 MB on GLM-5.3) was read whole on every rank. Every rank must agree (it adds a
/// collective); `comm_setup` checks it. Read once.
pub fn eh_tp_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("METRALE_GLM_MTP_EH_TP").as_deref() == Ok("1"))
}

/// 2026-10-11: Output rows of `eh_proj` per shard unit: a rank's first row and row count are
/// multiples of it, so its output slice and weight rows start aligned for the GEMV's vector
/// stores (a block computes four rows).
const EH_SHARD_ROWS: usize = 64;

/// 2026-10-11: `(r0, n)`, the `eh_proj` output rows rank `rank` of `world` computes: whole
/// [`EH_SHARD_ROWS`] units split as evenly as they go (the first ranks one unit more), the last
/// rank taking any remainder rows. `None` for one rank or a world past the unit count.
pub(super) fn eh_shard(hidden: usize, world: usize, rank: usize) -> Option<(usize, usize)> {
    let units = hidden / EH_SHARD_ROWS;
    if world < 2 || rank >= world || units < world {
        return None;
    }
    let (base, extra) = (units / world, units % world);
    let r0 = (rank * base + rank.min(extra)) * EH_SHARD_ROWS;
    let mut n = (base + usize::from(rank < extra)) * EH_SHARD_ROWS;
    if rank == world - 1 {
        n = hidden - r0;
    }
    Some((r0, n))
}

/// 2026-10-11: `METRALE_GLM_MTP_XCHG_DEVICE=1`: run the vocab-sharded draft head's exchange on
/// the device (`glm5next_mtp_xchg.cu`) instead of through the host (two reads, a write and four
/// syncs per draft); same draft (the kernels compute `head_xchg::pack` / `pick`). Read once.
fn xchg_device_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("METRALE_GLM_MTP_XCHG_DEVICE").as_deref() == Ok("1"))
}

/// 2026-10-11: The device exchange's kernels when the lever asks for them and the build has both.
fn xchg_kernels(gpu: &dyn GpuBackend) -> Option<(KernelHandle, KernelHandle)> {
    if !xchg_device_enabled() {
        return None;
    }
    let k = |f| metrale_model_layers::layers::try_kernel(gpu, "glm5next_mtp_xchg", f);
    let (pack, pick) = (k("glm5next_mtp_xchg_pack"), k("glm5next_mtp_xchg_pick"));
    (pack.0 != 0 && pick.0 != 0).then_some((pack, pick))
}

impl Glm5NextMtpHead {
    pub fn new(
        module: Glm5NextMtpModule,
        embed_tokens: DenseWeight,
        lm_head: DenseWeight,
        config: &metrale_config::ModelConfig,
        gpu: &dyn GpuBackend,
        max_seq_len: usize,
    ) -> Result<Self> {
        let dsa = match &module.layer.mixer {
            crate::glm5next_layer::Glm5NextMixer::Dsa(l) => l,
            _ => bail!("GLM MTP block is not a DSA layer"),
        };
        // 2026-09-25: The drafter block is a DSA layer, so it can never reach a position past
        // `max_dsa_context`: that is the indexer cache `Glm5NextDsaState::alloc` reserves, and
        // `advance` refuses to grow past it. Everything sized from `max_seq_len` here uses the
        // capped value: the private KV pool below, the block table claimed in `alloc_state`,
        // the bounds checks in `forward_one` and `rows_impl`, and, through
        // `prefill_hidden_rows`, the model's `mtp_prefill_hidden` capture.
        let max_seq_len = drafter_context_rows(max_seq_len, &dsa.cfg);
        // 2026-09-25: One KV head of `kv_lora_rank` per token, the absorbed-MLA latent that
        // the block's `latent_write` and paged gather address.
        let kv_config = KvCacheConfig {
            block_size: 16,
            num_kv_heads: 1,
            head_dim: dsa.cfg.kv_lora_rank,
            num_layers: 1,
            dtype: KvCacheDtype::Fp8,
            layer_dtypes: vec![],
            layer_dims: vec![],
            cache_blocks_per_seq: None,
        };
        let blocks = max_seq_len / kv_config.block_size + 2;
        let kv_cache = PagedKvCache::new(kv_config, blocks, gpu)?;
        // 2026-09-25: The draft `lm_head` sweep is most of the drafter's time. Measured
        // 2026-08-29 with `METRALE_GLM_MTP_SKIP=head`: propose 8.84 ms -> 1.52 ms. It reads
        // `[vocab, hidden]` BF16, 1.27 GB at GLM-5.3's 154,880 x 4,096.
        //
        // So the sweep is split by vocab across the ranks that run propose together: each
        // rank sweeps its `head_n` rows over the full hidden dimension, and the ranks exchange
        // (max, argmax) through one 16-byte all-reduce (`forward_one`).
        //
        // Vocab, not hidden: rows of `[vocab, hidden]` are contiguous, so a vocab shard is a
        // base-pointer offset and a smaller `n`. A hidden shard would need a row stride, and
        // the gemv takes none; it assumes rows packed at K.
        let head_world = config.tp_world_size.max(1);
        let head_rank = config.tp_rank;
        let (head_v0, head_n) = draft_head_shard(config.vocab_size, head_world, head_rank);
        // 2026-09-25: Quantise only the rows this rank sweeps: `head_n * hidden` bytes, not
        // the whole vocab. A failure here is not fatal; the head falls back to the BF16 sweep.
        let gemv_fp8w_k =
            metrale_model_layers::layers::try_kernel(gpu, "gemv_fp8w", "dense_gemv_fp8w");
        let head_fp8 = if std::env::var("METRALE_GLM_MTP_HEAD_FP8").as_deref() == Ok("0")
            || gemv_fp8w_k.0 == 0
        {
            None
        } else {
            let shard = DenseWeight {
                weight: lm_head.weight.offset(head_v0 * config.hidden_size * 2),
            };
            match gpu
                .kernel("gemv_fp8w", "quantize_bf16_to_fp8")
                .and_then(|qk| {
                    metrale_model_layers::weight_map::quantize_to_fp8(
                        &shard,
                        head_n,
                        config.hidden_size,
                        gpu,
                        qk,
                        gpu.default_stream(),
                    )
                }) {
                Ok(q) => {
                    tracing::info!(
                        "GLM MTP: draft lm_head shard quantised to FP8 ({} rows x {}, {} MB)",
                        head_n,
                        config.hidden_size,
                        head_n * config.hidden_size / (1024 * 1024),
                    );
                    Some(q)
                }
                Err(e) => {
                    tracing::warn!("GLM MTP: FP8 draft head unavailable ({e:#}); staying BF16");
                    None
                }
            }
        };

        Ok(Self {
            module,
            embed_tokens,
            lm_head,
            kv_cache: Mutex::new(kv_cache),
            rms_norm_k: gpu.kernel("rms_norm_vanilla", "rms_norm_vanilla")?,
            gemv_k: gpu.kernel("gemv", "dense_gemv_bf16")?,
            argmax_k: gpu.kernel("argmax", "argmax_bf16")?,
            hidden: config.hidden_size,
            vocab: config.vocab_size,
            max_seq_len,
            head_world,
            head_rank,
            eh_shard: if eh_tp_enabled() {
                eh_shard(config.hidden_size, head_world, head_rank)
            } else {
                None
            },
            head_v0,
            head_n,
            head_fp8,
            gemv_fp8w_k,
            xchg_k: xchg_kernels(gpu),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::draft_head_shard;

    /// 2026-10-08: GLM-5.3's vocab.
    const VOCAB: usize = 154_880;

    /// 2026-10-11: The `eh_proj` shards partition the hidden rows, start on whole units, and
    /// are refused for one rank or more ranks than units.
    #[test]
    fn eh_shards_partition_the_hidden_rows() {
        use super::eh_shard;
        for hidden in [4096usize, 4096 + 7, 6144, 128] {
            for world in 2..=4 {
                let Some(_) = eh_shard(hidden, world, 0) else {
                    assert!(hidden / 64 < world);
                    continue;
                };
                let mut next = 0;
                for r in 0..world {
                    let (r0, n) = eh_shard(hidden, world, r).unwrap();
                    assert_eq!(r0, next, "hidden {hidden} world {world} rank {r}");
                    assert_eq!(r0 % 64, 0);
                    assert!(n > 0);
                    next = r0 + n;
                }
                assert_eq!(next, hidden);
            }
        }
        assert_eq!(eh_shard(4096, 3, 0), Some((0, 1408)));
        assert_eq!(eh_shard(4096, 3, 1), Some((1408, 1344)));
        assert_eq!(eh_shard(4096, 3, 2), Some((2752, 1344)));
        assert_eq!(eh_shard(4096, 1, 0), None);
        assert_eq!(eh_shard(4096, 3, 3), None);
        assert_eq!(eh_shard(128, 3, 0), None);
    }

    /// 2026-10-08: Two ranks halve the vocab, as before.
    #[test]
    fn two_ranks_halve_the_vocab() {
        assert_eq!(draft_head_shard(VOCAB, 2, 0), (0, VOCAB / 2));
        assert_eq!(draft_head_shard(VOCAB, 2, 1), (VOCAB / 2, VOCAB / 2));
    }

    /// 2026-10-11: Every admitted world partitions the vocab exactly: contiguous, in rank order,
    /// no row twice or missing, shards differing by at most one row, never past the end.
    /// TP=3 over GLM-5.3's served vocab (154,856, which 3 does not divide) is the case that
    /// used to sweep the whole head on every rank.
    #[test]
    fn admitted_worlds_partition_the_vocab() {
        for vocab in [VOCAB, 154_856, 154_881, 7, 4] {
            for world in 2..=super::super::head_xchg::MAX_RANKS {
                let shards: Vec<(usize, usize)> = (0..world)
                    .map(|r| draft_head_shard(vocab, world, r))
                    .collect();
                let mut next = 0;
                for (r, &(v0, n)) in shards.iter().enumerate() {
                    assert_eq!(v0, next, "vocab {vocab} world {world} rank {r}");
                    assert!(n >= vocab / world && n <= vocab / world + 1);
                    next = v0 + n;
                }
                assert_eq!(next, vocab, "vocab {vocab} world {world}");
            }
        }
        assert_eq!(draft_head_shard(154_856, 3, 0), (0, 51_619));
        assert_eq!(draft_head_shard(154_856, 3, 1), (51_619, 51_619));
        assert_eq!(draft_head_shard(154_856, 3, 2), (103_238, 51_618));
    }

    /// 2026-10-08: One rank, or more ranks than the exchange holds, sweeps the whole vocab from
    /// row 0 on every rank, and the range never leaves the vocab.
    #[test]
    fn other_worlds_sweep_the_whole_vocab_inside_it() {
        let over = super::super::head_xchg::MAX_RANKS + 1;
        for (world, vocab) in [(1, VOCAB), (over, VOCAB), (over, 154_881), (3, 2)] {
            for rank in 0..world {
                assert_eq!(
                    draft_head_shard(vocab, world, rank),
                    (0, vocab),
                    "world {world} rank {rank}"
                );
            }
        }
    }
}
