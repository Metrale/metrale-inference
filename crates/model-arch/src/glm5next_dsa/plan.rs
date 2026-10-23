// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: A DSA mixer's captured pass run from its plan groups instead of the layer's
//! levers: the latent write, the indexer key norm, the indexer row's placement (store, pool key,
//! geometry), the selection over the persisted pool keys and the attend, as typed steps over plan
//! kernel ids ([`DsaSteps`], built by `glm5next_plan::dsa`), bound to this device's handles with
//! the context ceiling they launch at ([`DsaPlan`]), and launched through the launchers the
//! hand-written path uses (`Glm5NextDsaLayer::decode_k_plan`). The projections (W8A8 groups, the
//! RMS norm, the FP32-out selector GEMVs) are declared legacy sites: the layer's own code.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - A plan runs only a captured pass with row metadata (`plan_rows`): an eager pass, a prefill
//!   piece or a pass without metadata is the layer's own code, declared so.
//! - [`entry`] is the one source of the `(module, func)` of every kernel a plan step names; the
//!   layer resolves the same entries from it.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{GpuBackend, KernelHandle};
use metrale_model_layers::layer::{AttnMetadataDev, ForwardContext};
use parking_lot::RwLock;

use super::select::PlannedSelect;

/// 2026-10-23: The `(module, func)` of each kernel a DSA plan step names.
pub mod entry {
    use super::super::attend::{DSA_DECODE_HB_MODULE, DSA_DECODE_MODULE};
    use super::super::{
        DSA_MODULE, DSA_ROWS_MODULE, DSA_SCORES_HEADS_WINDOW_MODULE, DSA_SCORES_WARP_MODULE,
        DSA_TOPK_MC_MODULE, LAYERNORM_MODULE,
    };

    pub type Entry = (&'static str, &'static str);
    pub const LATENT_WRITE: Entry = ("glm5next_mla_latent_write", "glm5next_mla_latent_write_fp8");
    pub const K_NORM: Entry = (LAYERNORM_MODULE, "nllb_layernorm_bf16");
    pub const STORE: Entry = (DSA_MODULE, "dsa_indexer_store");
    pub const POOL_KEYS: Entry = (DSA_SCORES_WARP_MODULE, "dsa_pool_keys_store_rows");
    pub const WRITE_GEOM: Entry = (DSA_MODULE, "dsa_write_geom");
    pub const SCORES_HEADS_PAGED: Entry = (DSA_SCORES_WARP_MODULE, "dsa_index_scores_heads_paged");
    pub const SCORES_HEADS_WINDOW: Entry = (
        DSA_SCORES_HEADS_WINDOW_MODULE,
        "dsa_index_scores_heads_window",
    );
    pub const TOPK_MC_PASS: Entry = (DSA_TOPK_MC_MODULE, "dsa_topk_mc_pass");
    pub const TOPK_MC_COLLECT: Entry = (DSA_TOPK_MC_MODULE, "dsa_topk_mc_collect");
    pub const EXPAND_ROWS: Entry = (DSA_ROWS_MODULE, "dsa_expand_selection_rows");
    pub const PER_HEAD: Entry = (DSA_DECODE_MODULE, "glm5next_dsa_mla_decode_fp8");
    pub const DECODE_HB: Entry = (DSA_DECODE_HB_MODULE, "glm5next_dsa_mla_decode_hb_fp8");
    pub const HB_MERGE: Entry = (DSA_DECODE_HB_MODULE, "glm5next_dsa_mla_merge");
}

/// 2026-10-23: `entry` resolved with `try_kernel` (`KernelHandle(0)` when the target lacks it).
pub fn try_entry(gpu: &dyn GpuBackend, (module, func): entry::Entry) -> KernelHandle {
    metrale_model_layers::layers::try_kernel(gpu, module, func)
}

/// 2026-10-23: The indexer key norm of a planned pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KNormStep<K> {
    /// 2026-10-23: Per row, between the row's key and gate projections (one row).
    Row(K),
    /// 2026-10-23: Once over the staged rows of a verify whose projections ran merged.
    Rows(K),
}

/// 2026-10-23: The pool keys a planned selection scores. 2026-10-23: only FP32 here; the GPU
/// track's `METRALE_GLM_DSA_POOL_KEYS_BF16` and `_DECODE_SCORES_MMA` (default off, not on this
/// branch) join as variants with their kernels when they land.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PoolKeys {
    F32,
}

/// 2026-10-23: The scores kernel of a planned selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScoresStep<K> {
    /// 2026-10-23: `dsa_index_scores_heads_paged` (one row).
    HeadsPaged(K),
    /// 2026-10-23: `dsa_index_scores_heads_window` (a verify's rows in one pass).
    HeadsWindow(K),
}

/// 2026-10-23: A planned selection: the scores, the multi-CTA radix top-k
/// (`select::TOPK_MC_PASSES` launches of `topk_pass`, then `topk_collect`), the expansion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SelectStep<K> {
    pub pool_keys: PoolKeys,
    pub scores: ScoresStep<K>,
    pub topk_pass: K,
    pub topk_collect: K,
    pub expand: K,
}

/// 2026-10-23: The attend of a planned pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttendStep<K> {
    PerHead(K),
    /// 2026-10-23: The head-batched entry and its merge at `splits` (`glm_dsa_decode_hb`).
    HeadBatched {
        decode: K,
        merge: K,
        splits: usize,
    },
}

/// 2026-10-23: A DSA mixer's plan at `rows` rows: one row selected per row (`rows == 1`), or a
/// verify's rows selected in one pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DsaSteps<K> {
    pub rows: usize,
    pub latent_write: K,
    pub k_norm: KNormStep<K>,
    pub store: K,
    pub pool_keys: K,
    pub write_geom: K,
    pub select: SelectStep<K>,
    pub attend: AttendStep<K>,
}

impl<K> DsaSteps<K> {
    /// 2026-10-23: Every kernel through `f`.
    pub fn map<L>(self, mut f: impl FnMut(K) -> Result<L>) -> Result<DsaSteps<L>> {
        let s = self.select;
        Ok(DsaSteps {
            rows: self.rows,
            latent_write: f(self.latent_write)?,
            k_norm: match self.k_norm {
                KNormStep::Row(k) => KNormStep::Row(f(k)?),
                KNormStep::Rows(k) => KNormStep::Rows(f(k)?),
            },
            store: f(self.store)?,
            pool_keys: f(self.pool_keys)?,
            write_geom: f(self.write_geom)?,
            select: SelectStep {
                pool_keys: s.pool_keys,
                scores: match s.scores {
                    ScoresStep::HeadsPaged(k) => ScoresStep::HeadsPaged(f(k)?),
                    ScoresStep::HeadsWindow(k) => ScoresStep::HeadsWindow(f(k)?),
                },
                topk_pass: f(s.topk_pass)?,
                topk_collect: f(s.topk_collect)?,
                expand: f(s.expand)?,
            },
            attend: match self.attend {
                AttendStep::PerHead(k) => AttendStep::PerHead(f(k)?),
                AttendStep::HeadBatched {
                    decode,
                    merge,
                    splits,
                } => AttendStep::HeadBatched {
                    decode: f(decode)?,
                    merge: f(merge)?,
                    splits,
                },
            },
        })
    }
}

/// 2026-10-23: A bound DSA plan and the context ceiling its captured selection launches at.
#[derive(Clone, Copy, Debug)]
pub struct DsaPlan {
    pub steps: DsaSteps<KernelHandle>,
    /// 2026-10-23: The ceiling's pool count, the top-k CTAs at it and the scores' grid (one
    /// block per SM), fixed at installation from the layer's shape and the device.
    pub max_pools: usize,
    pub topk_ctas: usize,
    pub sm_count: u32,
}

impl DsaPlan {
    /// 2026-10-23: The selection's kernels and ceiling.
    pub fn select(&self) -> PlannedSelect {
        let s = self.steps.select;
        PlannedSelect {
            scores: match s.scores {
                ScoresStep::HeadsPaged(k) | ScoresStep::HeadsWindow(k) => k,
            },
            sm_count: self.sm_count,
            topk_pass: s.topk_pass,
            topk_collect: s.topk_collect,
            topk_ctas: self.topk_ctas,
            expand: s.expand,
            max_pools: self.max_pools,
        }
    }
}

/// 2026-10-23: The row metadata of a pass a plan runs: a captured pass that is not a prefill
/// piece, with metadata for its rows (one row on a decode step, `k` rows otherwise), as
/// `decode_k_with` reads it; `None` for every pass the layer's own code runs.
pub fn plan_rows<'a>(
    ctx: &'a ForwardContext<'_>,
    k: usize,
    is_prefill: bool,
) -> Option<&'a AttnMetadataDev> {
    if is_prefill || !ctx.graph_capture {
        return None;
    }
    let m = ctx.attn_metadata.as_ref()?;
    match k {
        1 => ctx.decode_step.then_some(m),
        _ => (m.num_seqs as usize == k).then_some(m),
    }
}

/// 2026-10-23: The DSA plans one layer runs, by row count, with their executed sites.
#[derive(Default)]
pub struct InstalledDsa {
    by_rows: RwLock<Vec<(DsaPlan, Vec<String>)>>,
}

impl std::fmt::Debug for InstalledDsa {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InstalledDsa")
            .field("sites", &self.sites().len())
            .finish()
    }
}

impl InstalledDsa {
    /// 2026-10-23: Replace the installed plans (empty clears).
    pub fn set(&self, plans: Vec<(DsaPlan, Vec<String>)>) {
        *self.by_rows.write() = plans;
    }

    /// 2026-10-23: The plan at `rows` rows.
    pub fn get(&self, rows: usize) -> Option<DsaPlan> {
        self.by_rows
            .read()
            .iter()
            .find(|(p, _)| p.steps.rows == rows)
            .map(|(p, _)| *p)
    }

    /// 2026-10-23: The installed sites.
    pub fn sites(&self) -> Vec<String> {
        self.by_rows
            .read()
            .iter()
            .flat_map(|(_, s)| s.clone())
            .collect()
    }
}
