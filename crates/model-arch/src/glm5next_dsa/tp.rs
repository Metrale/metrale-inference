// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: GLM-5.3-Flash DSA tensor-parallel shard plan: MLA heads sharded, the indexer
//! replicated.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - [`DsaTpPlan::new`] refuses `tp_rank >= tp_size`, a config that fails
//!   `Glm5NextDsaConfig::validate`, and a `cfg.local_heads` that is not this rank's share.
//! - 2026-10-08: This rank's heads are `metrale_config::tp_split(full_heads, tp_size, tp_rank, 1)`
//!   (64 over three ranks: 22/21/21); every head-sharded tensor slices whole heads of that range.
//! - Every indexer tensor, `kv_a_proj_with_mqa`, and the low-rank down-projections and their
//!   norms are replicated; `q_b_proj` and `kv_b_proj` shard by head; `o_proj` is row-parallel.
//!
//! Pure data, as in [`crate::glm5next_kda::tp`], so the per-rank arithmetic is tested without a
//! GPU.
//!
//! # Why the indexer is replicated
//!
//! The indexer emits a token selection, not a partial sum. `index_scores` adds
//! `weights[h] * relu(scale * dot)` over the indexer heads, so a head-sharded indexer gives each
//! rank a partial score, and a per-rank top-k over partial scores can select different tokens on
//! different ranks. Sharding is correct only with an all-reduce of the scores before the top-k.
//! Scores are `[q_rows, n_pools]` with `n_pools = seq / kpool`: at a 262,144-token context
//! (`max_position_embeddings` in MODEL.toml) that is 65,536 x 4 B = 256 KiB per decode token per
//! DSA layer, against a replicated indexer of about 15 MB per layer (the `indexer.*` sizes in
//! `tp/tests.rs`). An all-reduce would also change the order of the per-head sum, which can
//! reorder pools whose scores tie at the cutoff.
//!
//! * `kv_a_proj_with_mqa` produces the shared latent every head decompresses from; it has no
//!   head axis.
//! * `q_b_proj` shards at `qk_head_dim` rows per head and `kv_b_proj` at
//!   `qk_nope_head_dim + v_head_dim`; the two strides differ.
//! * `o_proj` is row-parallel on `heads * v_head_dim`.

use anyhow::{Result, bail};
use metrale_config::ModelConfig;

use super::Glm5NextDsaConfig;

/// 2026-09-25: How one DSA tensor maps onto TP ranks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DsaShard {
    /// 2026-09-25: Every rank holds the whole tensor: the indexer, the latent KV projection, and
    /// the low-rank down-projections and norms.
    Replicated,
    /// 2026-09-25: Leading dim is `heads * per_head`: slice by this rank's head range.
    HeadRows,
    /// 2026-09-25: Trailing (input) dim is `heads * v_head_dim`: a row-parallel GEMM, sliced on the
    /// input dim, whose output is all-reduced.
    HeadCols,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DsaTensorPlan {
    pub name: &'static str,
    pub kind: DsaShard,
    pub elem_bytes: usize,
    pub full_rows: usize,
    pub full_row_elems: usize,
    pub local_rows: usize,
    pub local_row_elems: usize,
    pub src_row_offset: usize,
    pub src_col_offset: usize,
}

impl DsaTensorPlan {
    pub fn local_bytes(&self) -> usize {
        self.local_rows * self.local_row_elems * self.elem_bytes
    }
    pub fn full_bytes(&self) -> usize {
        self.full_rows * self.full_row_elems * self.elem_bytes
    }
}

const BF16: usize = 2;

/// 2026-09-25: Per-rank shard plan for one DSA block.
#[derive(Debug, Clone)]
pub struct DsaTpPlan {
    pub tp_rank: usize,
    pub tp_size: usize,
    pub full_heads: usize,
    pub local_heads: usize,
    /// 2026-10-08: The first head this rank owns.
    pub head_start: usize,
    pub tensors: Vec<DsaTensorPlan>,
}

impl DsaTpPlan {
    /// 2026-10-08: The plan for `config`'s rank: the full head count is the recorded pre-shard
    /// one (`ModelConfig::pre_shard_heads`), and `cfg` is `Glm5NextDsaConfig::from_config` of
    /// the same config.
    pub fn from_config(config: &ModelConfig, cfg: &Glm5NextDsaConfig) -> Result<Self> {
        Self::new(
            config.tp_rank,
            config.tp_world_size.max(1),
            config.pre_shard_heads()?.num_attention_heads,
            cfg,
        )
    }

    /// 2026-09-25: `cfg.local_heads` is already per rank. 2026-10-08: The full head count is
    /// passed in rather than rebuilt as `local_heads * tp_size`, which an uneven split breaks;
    /// `cfg.local_heads` must equal this rank's share of it.
    pub fn new(
        tp_rank: usize,
        tp_size: usize,
        full_heads: usize,
        cfg: &Glm5NextDsaConfig,
    ) -> Result<Self> {
        if tp_rank >= tp_size {
            bail!("tp_rank {tp_rank} >= tp_size {tp_size}");
        }
        cfg.validate()?;
        let heads = metrale_config::tp_split(full_heads, tp_size, tp_rank, 1)?;
        if heads.len != cfg.local_heads {
            bail!(
                "DSA TP: rank {tp_rank} of {tp_size} owns {} of {full_heads} heads, but the \
                 layer config holds {}",
                heads.len,
                cfg.local_heads
            );
        }
        let (local_heads, head_start) = (heads.len, heads.start);

        let h = cfg.hidden;
        let qk = cfg.qk_head_dim();
        let kvb_per_head = cfg.qk_nope_head_dim + cfg.v_head_dim;
        let ihd = cfg.index_head_dim;

        // 2026-10-08: The sharded axis holds `full_heads` heads of equal width, so a rank's
        // slice is its head range scaled by that width.
        let mk = |name, kind, full_rows: usize, full_row_elems: usize| {
            let (local_rows, local_row_elems, src_row_offset, src_col_offset) = match kind {
                DsaShard::Replicated => (full_rows, full_row_elems, 0, 0),
                DsaShard::HeadRows => {
                    let per_head = full_rows / full_heads;
                    (
                        local_heads * per_head,
                        full_row_elems,
                        head_start * per_head,
                        0,
                    )
                }
                DsaShard::HeadCols => {
                    let per_head = full_row_elems / full_heads;
                    (full_rows, local_heads * per_head, 0, head_start * per_head)
                }
            };
            DsaTensorPlan {
                name,
                kind,
                elem_bytes: BF16,
                full_rows,
                full_row_elems,
                local_rows,
                local_row_elems,
                src_row_offset,
                src_col_offset,
            }
        };

        let tensors = vec![
            mk("q_a_proj", DsaShard::Replicated, cfg.q_lora_rank, h),
            mk("q_a_layernorm", DsaShard::Replicated, cfg.q_lora_rank, 1),
            mk(
                "q_b_proj",
                DsaShard::HeadRows,
                full_heads * qk,
                cfg.q_lora_rank,
            ),
            // 2026-09-25: The shared latent has no head axis, so it is replicated.
            mk(
                "kv_a_proj_with_mqa",
                DsaShard::Replicated,
                cfg.kv_cache_dim(),
                h,
            ),
            mk("kv_a_layernorm", DsaShard::Replicated, cfg.kv_lora_rank, 1),
            mk(
                "kv_b_proj",
                DsaShard::HeadRows,
                full_heads * kvb_per_head,
                cfg.kv_lora_rank,
            ),
            mk("o_proj", DsaShard::HeadCols, h, full_heads * cfg.v_head_dim),
            // 2026-09-25: The indexer, replicated (see the module doc).
            mk(
                "indexer.wq_b",
                DsaShard::Replicated,
                cfg.index_heads * ihd,
                cfg.q_lora_rank,
            ),
            mk("indexer.wk", DsaShard::Replicated, ihd, h),
            mk("indexer.k_norm.weight", DsaShard::Replicated, ihd, 1),
            // 2026-09-25: `k_norm` is a LayerNorm, so it has a bias.
            mk("indexer.k_norm.bias", DsaShard::Replicated, ihd, 1),
            mk(
                "indexer.weights_proj",
                DsaShard::Replicated,
                cfg.index_heads,
                h,
            ),
            mk(
                "indexer.index_kpool_compress_gate",
                DsaShard::Replicated,
                ihd,
                h,
            ),
            mk(
                "indexer.index_kpool_compress_ape",
                DsaShard::Replicated,
                cfg.index_kpool,
                ihd,
            ),
        ];

        Ok(Self {
            tp_rank,
            tp_size,
            full_heads,
            local_heads,
            head_start,
            tensors,
        })
    }

    pub fn get(&self, name: &str) -> Option<&DsaTensorPlan> {
        self.tensors.iter().find(|t| t.name == name)
    }
    pub fn local_bytes(&self) -> usize {
        self.tensors.iter().map(|t| t.local_bytes()).sum()
    }
    pub fn full_bytes(&self) -> usize {
        self.tensors.iter().map(|t| t.full_bytes()).sum()
    }
    /// 2026-09-25: `o_proj` is row-parallel, so above `tp_size = 1` its output needs the reduce.
    pub fn needs_output_all_reduce(&self) -> bool {
        self.tp_size > 1
    }
    /// 2026-09-25: Bytes every rank holds in full.
    pub fn replicated_bytes(&self) -> usize {
        self.tensors
            .iter()
            .filter(|t| t.kind == DsaShard::Replicated)
            .map(|t| t.full_bytes())
            .sum()
    }
}

#[cfg(test)]
mod tests;
