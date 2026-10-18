// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: DFlash drafter levers, resolved from the startup settings once per head and carried on it.
//!
//! Owner: model-arch (DFlash drafter).
//! Invariants:
//! - `DFlashLevers::from_settings` is called only while a head is built (`from_weights.rs`),
//!   so a head's levers are fixed for its lifetime.
//! - Each field reads one settings field, except `any_diagnostic_armed`, which asks whether any
//!   name in `GRAPH_SUPPRESSING_DIAGNOSTICS` was set (the settings provenance).
//!
//! The levers live on the head rather than in a process-wide `OnceLock`, so a second
//! model built in the same process with other settings resolves its own.

use metrale_settings::InferenceServerSettings;
use metrale_settings::provenance::Source;

/// 2026-09-25: Diagnostic and A/B levers for one loaded DFlash drafter.
// 2026-09-25: No `Eq`: `conf_tau` is an `f32`.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct DFlashLevers {
    /// 2026-09-25: A variable in `GRAPH_SUPPRESSING_DIAGNOSTICS` is set, at any
    /// value. `forward_block` then runs eagerly, without CUDA graphs. This is
    /// presence, not truth: `METRALE_DFLASH_BLOCK_DUMP=0` suppresses graphs and
    /// arms no dump.
    pub any_diagnostic_armed: bool,

    /// 2026-09-25: `METRALE_DFLASH_DEBUG_DUMP=1`: log the first 10 BF16 values
    /// of key `forward_block` intermediates.
    pub debug_dump: bool,
    /// 2026-09-25: `METRALE_DFLASH_DEBUG_DUMP_FULL=1`: one-shot whole-tensor dumps
    /// to `/tmp`, and the first propose's drafts to the log.
    pub debug_dump_full: bool,
    /// 2026-09-25: `METRALE_DFLASH_LOG_DRAFTS=1`: log the first propose's drafts
    /// (one-shot per model).
    pub log_drafts: bool,
    /// 2026-09-25: `METRALE_DFLASH_BLOCK_DUMP=1`: one-shot `/tmp` dumps of the
    /// block logits and per-layer intermediates, gated by `block_dump_armed_at`.
    pub block_dump: bool,
    /// 2026-09-25: `METRALE_DFLASH_BLOCK_DUMP_AT_POS=<n>`: arm the block dump only
    /// at decode position >= n. Unset or unparseable is 0.
    pub block_dump_at_pos: usize,
    /// 2026-09-25: `METRALE_DFLASH_OPTION_B_DIAG=1`: one-shot read-back of the
    /// K row layer 0 just wrote to the paged drafter cache, compared with its
    /// source.
    pub option_b_diag: bool,

    /// 2026-09-25: `METRALE_DFLASH_DEBUG_FORCE_PATTERN=1`: overwrite the first ctx
    /// slot `forward_block` reads with a deterministic pattern.
    pub force_pattern: bool,
    /// 2026-09-25: `METRALE_DFLASH_DEBUG_FORCE_NOISE_PATTERN=1`: the same, for the
    /// noise rows.
    pub force_noise_pattern: bool,
    /// 2026-09-25: `METRALE_DFLASH_DEBUG_CTX_OFF=1`: drop ctx conditioning
    /// (`eff_ctx = 0`).
    pub force_no_ctx: bool,
    /// 2026-09-25: `METRALE_DFLASH_DEBUG_CTX_USED=<n>`: cap `eff_ctx` at n (it
    /// stays bounded by the ctx rows present and `ctx_window`).
    pub force_ctx_used: Option<usize>,

    /// 2026-09-25: `METRALE_DFLASH_PRECOMPUTE=1`: also run the ctx K/V precompute
    /// from `forward_block`; `propose_drafts` runs it either way.
    pub precompute: bool,
    /// 2026-09-25: `METRALE_DFLASH_PRECOMPUTE_COMMIT=1`: let that diagnostic run
    /// write to the paged cache (its `commit` argument).
    pub precompute_commit: bool,

    /// 2026-09-25: `METRALE_DFLASH_PROPOSE_WARMUP_N=<n>`: eager passes per block
    /// width before graph capture. Unset or unparseable is 2.
    pub propose_warmup_n: usize,

    /// 2026-09-25: The paged drafter cache for ctx K/V ("Option B"). On unless
    /// `METRALE_DFLASH_OPTION_B=0`; resolved by `option_b_from` in `dflash_head.rs`.
    pub option_b: bool,
    /// 2026-09-25: `METRALE_DFLASH_OPTION_B_NO_CTX=1`: pass `ctx_count = 0`, so
    /// paged attention sees only the γ K/V written in-layer.
    pub option_b_no_ctx: bool,
    /// 2026-09-25: The DFlash2 conv and selector path. Runs when the checkpoint
    /// carries its components, unless `METRALE_DFLASH2=0`.
    pub dflash2: bool,
    /// 2026-09-25: `METRALE_DFLASH_BATCH_PROPOSE=<width>` caps the cross-sequence
    /// batch. Unset is `usize::MAX` (as wide as the scratch bands allow); `0` or
    /// `1` makes `propose_batch_max` return 1.
    pub batch_propose_width: usize,
    /// 2026-09-25: `METRALE_DFLASH_DRAFT_CAP=<n>`: return at most n drafts per
    /// propose. `None` caps at the block width in flight (`block_g`).
    pub draft_cap: Option<usize>,
    /// 2026-10-09: `METRALE_DFLASH_FULL_BLOCK=1`: draft at the head's full block width
    /// (gamma) whatever the step's draft count, and return that many drafts from the front of
    /// the block. A drafter with non-causal attention predicts each mask row from every row
    /// of the block, so a narrower block is a different input from the one it was trained on;
    /// the reference serve drafts the full block and verifies a prefix. Off by default.
    pub full_block: bool,

    /// 2026-09-25: `METRALE_DFLASH_VERIFY_TRACE=1`: log every propose's drafts
    /// before the row-0 drop and the cap.
    pub verify_trace: bool,
    /// 2026-09-25: `METRALE_DFLASH_PRECOMPUTE_DUMP=1`: one-shot `/tmp` dumps of
    /// the ctx K/V precompute intermediates.
    pub precompute_dump: bool,
    /// 2026-09-25: `METRALE_DFLASH_CTX_PARITY_DUMP=1`: one-shot `/tmp` dump of the
    /// accumulated ctx hidden rows. 2026-10-09: With their positions, after the decode-append,
    /// at the first propose at or past `block_dump_at_pos` (the block dump's propose).
    pub ctx_parity_dump: bool,
    /// 2026-09-25: `METRALE_DFLASH_DEBUG_NO_DECODE_APPEND=1`: skip the ctx append
    /// at the start of `propose_drafts`.
    pub no_decode_append: bool,
    /// 2026-09-25: `METRALE_DFLASH_DEBUG_FULL_PRECOMPUTE=1`: the non-batched
    /// propose recomputes the whole ctx prefix (`committed = 0`) each step.
    pub full_precompute: bool,
    /// 2026-09-25: `METRALE_DFLASH_CTXLEN_PROBE=1`: warn when `ctx_positions` is
    /// not strictly increasing, and log `ctx_len` against the position when the
    /// position is a multiple of 16.
    pub ctxlen_probe: bool,

    /// 2026-09-25: The sequential Markov fixup. Runs when the drafter carries
    /// the head, unless `METRALE_DSPARK_MARKOV=0`, which takes the plain
    /// per-row argmax loop.
    pub dspark_markov: bool,
    /// 2026-09-25: `METRALE_DSPARK_CONF_TAU=<t>`: sigmoid-space acceptance
    /// threshold for the confidence head. Unset or unparseable is `0.0`, which
    /// leaves the head off.
    pub conf_tau: f32,
    /// 2026-09-25: `METRALE_DSPARK_SHIFT=1|0` forces the shifted-row convention
    /// on or off; any other value (`None`) defers to the drafter config.
    pub dspark_shift: Option<bool>,
    /// 2026-09-25: Row 0 carries the Markov bias. On unless
    /// `METRALE_DSPARK_ANCHOR_BIAS=0`. Confidence truncation also requires it:
    /// an unbiased row never writes its confidence slot.
    pub dspark_anchor_bias: bool,
    /// 2026-09-25: `METRALE_DSPARK_CONF_TRACE=1`: log the confidence logits and
    /// sigmoids.
    pub dspark_conf_trace: bool,
    /// 2026-10-10: The DFlash2 top-16 runs in two stages over many blocks
    /// (`dflash2_topk16_part` + `dflash2_topk16_merge`, the same outputs and writes as
    /// `dflash2_topk16`). On unless `METRALE_DFLASH2_TOPK_SPLIT=0`.
    pub topk16_split: bool,
    /// 2026-10-10: `METRALE_DFLASH_TP_GRAPHS=1`: a tensor-parallel drafter's propose runs on its
    /// CUDA graphs, collectives captured (`tp.rs`). Opt-in; otherwise it runs eagerly.
    pub tp_graphs: bool,
    /// 2026-10-17: The batched ctx precompute (one fc + fused-KV pass over every sequence's
    /// uncommitted ctx rows) runs on the batched propose path, unless
    /// `METRALE_DFLASH_NO_BATCHED_PRECOMPUTE=1`, which runs the per-sequence loop.
    pub batched_precompute: bool,
    /// 2026-10-17: The rt2 FP8 GEMVs may run; off under `METRALE_NO_DFLASH_FP8_RT=1`. Fixed for
    /// the head's lifetime, so the kernel choice stays the same across CUDA-graph captures.
    pub fp8_rt: bool,
    /// 2026-10-17: `METRALE_DFLASH_FP8_GEMM_P2=1`: the row-scaled FP8 GEMM takes only the
    /// 2-stage original.
    pub fp8_gemm_p2: bool,
    /// 2026-10-17: `METRALE_DFLASH_FP8_GEMM_P4=1`: the row-scaled FP8 GEMM skips `_k64`.
    pub fp8_gemm_p4: bool,
    /// 2026-10-17: The per-sequence ctx accumulator bound ([`super::dflash_ctx_cap`]); `0` is
    /// uncapped.
    pub ctx_cap: usize,
}

/// 2026-09-25: Variables whose presence, at any value, sets `any_diagnostic_armed`
/// and so keeps `forward_block` off CUDA graphs.
const GRAPH_SUPPRESSING_DIAGNOSTICS: [&str; 11] = [
    "METRALE_DFLASH_PROPOSE_NO_GRAPH",
    "METRALE_DFLASH_DEBUG_DUMP_FULL",
    "METRALE_DFLASH_OPTION_B_DIAG",
    "METRALE_DFLASH_PRECOMPUTE_DUMP",
    "METRALE_DFLASH_VERIFY_TRACE",
    "METRALE_DFLASH_LOG_DRAFTS",
    "METRALE_DFLASH_DEBUG_FORCE_PATTERN",
    "METRALE_DFLASH_DEBUG_FORCE_NOISE_PATTERN",
    "METRALE_DFLASH_DEBUG_CTX_OFF",
    "METRALE_DFLASH_DEBUG_CTX_USED",
    "METRALE_DFLASH_BLOCK_DUMP",
];

/// 2026-10-17: The value of a numeric text lever; unset or unparseable is `None`, as before.
fn parsed<T: core::str::FromStr>(value: Option<&str>) -> Option<T> {
    value.and_then(|v| v.parse().ok())
}

/// 2026-10-17: A lever in `GRAPH_SUPPRESSING_DIAGNOSTICS` was given a value by any source, so a
/// resolved `0` still keeps `forward_block` off CUDA graphs (presence, not truth).
fn diagnostic_set(s: &InferenceServerSettings) -> bool {
    GRAPH_SUPPRESSING_DIAGNOSTICS.iter().any(|var| {
        s.provenance
            .source_of(var)
            .is_some_and(|src| src != Source::Default)
    })
}

impl DFlashLevers {
    /// 2026-10-17: Resolve from the startup settings. The head calls it once, in
    /// `from_weights.rs`; per-step code reads the head's `levers` field.
    pub fn from_settings(s: &InferenceServerSettings) -> Self {
        let (sp, rt, tm) = (&s.speculative, &s.runtime, &s.telemetry);
        Self {
            any_diagnostic_armed: diagnostic_set(s),

            debug_dump: tm.dflash_debug_dump,
            debug_dump_full: tm.dflash_debug_dump_full,
            log_drafts: tm.dflash_log_drafts,
            block_dump: tm.dflash_block_dump,
            block_dump_at_pos: parsed(tm.dflash_block_dump_at_pos.as_deref()).unwrap_or(0),
            option_b_diag: tm.dflash_option_b_diag,

            force_pattern: tm.dflash_debug_force_pattern,
            force_noise_pattern: tm.dflash_debug_force_noise_pattern,
            force_no_ctx: tm.dflash_debug_ctx_off,
            force_ctx_used: parsed(tm.dflash_debug_ctx_used.as_deref()),

            precompute: sp.dflash_precompute,
            precompute_commit: sp.dflash_precompute_commit,

            option_b: super::option_b_from(sp.dflash_option_b.as_deref()),
            option_b_no_ctx: sp.dflash_option_b_no_ctx,
            dflash2: sp.dflash2,
            topk16_split: sp.dflash2_topk_split,
            tp_graphs: sp.dflash_tp_graphs,
            batch_propose_width: parsed(sp.dflash_batch_propose.as_deref()).unwrap_or(usize::MAX),
            draft_cap: parsed(sp.dflash_draft_cap.as_deref()),
            full_block: sp.dflash_full_block,

            verify_trace: tm.dflash_verify_trace,
            precompute_dump: tm.dflash_precompute_dump,
            ctx_parity_dump: tm.dflash_ctx_parity_dump,
            no_decode_append: tm.dflash_debug_no_decode_append,
            full_precompute: tm.dflash_debug_full_precompute,
            ctxlen_probe: sp.dflash_ctxlen_probe,

            dspark_markov: sp.dspark_markov,
            conf_tau: parsed(rt.dspark_conf_tau.as_deref()).unwrap_or(0.0),

            propose_warmup_n: parsed(sp.dflash_propose_warmup_n.as_deref()).unwrap_or(2),

            dspark_shift: match rt.dspark_shift.as_deref() {
                Some("1") => Some(true),
                Some("0") => Some(false),
                _ => None,
            },
            dspark_anchor_bias: rt.dspark_anchor_bias,
            dspark_conf_trace: tm.dspark_conf_trace,

            batched_precompute: sp.dflash_no_batched_precompute.as_deref() != Some("1"),
            fp8_rt: !sp.no_dflash_fp8_rt,
            fp8_gemm_p2: sp.dflash_fp8_gemm_p2,
            fp8_gemm_p4: sp.dflash_fp8_gemm_p4,
            ctx_cap: super::dflash_ctx_cap(s),
        }
    }

    /// 2026-09-25: What `from_settings` resolves to with none of these variables set
    /// (checked by `nothing_set_resolves_to_defaults`).
    pub fn defaults() -> Self {
        Self {
            // 2026-09-25: Every field whose unset value is not its type's
            // `Default` is spelled out here.
            option_b: true,
            dflash2: true,
            topk16_split: true,
            dspark_markov: true,
            batch_propose_width: usize::MAX,
            dspark_anchor_bias: true,
            propose_warmup_n: 2,
            batched_precompute: true,
            fp8_rt: true,
            ctx_cap: 16384,
            ..Self::default()
        }
    }

    /// 2026-09-25: The block dump is armed for this decode position.
    pub fn block_dump_armed_at(&self, position: usize) -> bool {
        self.block_dump && position >= self.block_dump_at_pos
    }
}

#[cfg(test)]
#[path = "levers_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "levers_settings_tests.rs"]
mod settings_tests;
