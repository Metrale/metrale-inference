// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: The serve side of `met circuit memory`: the settings `met serve` would run with (a
//! recipe plus `met serve` flags, through the serve's own parser and validation) and the counts
//! the engine derives from them, each by the engine's own function: the SSM pool's unit counts
//! (`ssm_reserve::pool_counts_with`), the MTP verify slots, the Marconi and decode-ring slots, the
//! carried-state verify's slots, and the legacy buffer arena (`BufferSizes::from_config`).
//!
//! Owner: server CLI.
//! Invariants:
//! - No count here is computed by a formula of this file's own where the engine has a function;
//!   the one exception, the GDN two-phase prefill buffers, cites its allocation site and is
//!   checked against boot ledgers (`circuit_memory_ledger_tests.rs`).
//! - The MODEL.toml `[behavior]` defaults (`default_num_drafts`, `mtp_max_seqs`,
//!   `default_kv_dtype`) are read from the device class's target directory; an absent file or key
//!   takes the engine default the serve would take, and the header says so.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use metrale_config::ModelConfig;
use metrale_model_layers::ssm_reserve::{
    self, PoolCounts, PoolShape, SsmRollbackMode, mtp_state_slots_with, pool_counts_with,
    verify_slot_drafts_with,
};

use crate::cli::ServeArgs;

/// 2026-10-02: The target's MODEL.toml `[behavior]` defaults; 0 / empty when absent.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct Behavior {
    pub default_num_drafts: u32,
    pub mtp_max_seqs: u32,
    pub default_kv_dtype: String,
    /// 2026-10-02: Where they were read, for the header.
    pub source: String,
}

impl Behavior {
    /// 2026-10-02: Parse a MODEL.toml text's `[behavior]`.
    pub(crate) fn parse(text: &str, source: String) -> Result<Self> {
        let t: toml::Table = toml::from_str(text).context("MODEL.toml")?;
        let b = t.get("behavior").and_then(|b| b.as_table());
        let int = |k: &str| {
            b.and_then(|b| b.get(k))
                .and_then(|v| v.as_integer())
                .map_or(0, |v| v as u32)
        };
        Ok(Self {
            default_num_drafts: int("default_num_drafts"),
            mtp_max_seqs: int("mtp_max_seqs"),
            default_kv_dtype: b
                .and_then(|b| b.get("default_kv_dtype"))
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string(),
            source,
        })
    }
}

/// 2026-10-02: `argv` with every flag `extra` names (and the value after it, if any) removed,
/// then `extra` appended: a later flag replaces the recipe's, as a run-time override does.
fn override_flags(argv: Vec<String>, extra: &[String]) -> Vec<String> {
    let named: std::collections::BTreeSet<&str> = extra
        .iter()
        .filter(|f| f.starts_with("--"))
        .map(|f| f.split('=').next().unwrap_or(f))
        .collect();
    let mut out = Vec::with_capacity(argv.len() + extra.len());
    let mut skip_value = false;
    for a in argv {
        if skip_value && !a.starts_with("--") {
            skip_value = false;
            continue;
        }
        skip_value = false;
        if a.starts_with("--") && named.contains(a.split('=').next().unwrap_or(&a)) {
            skip_value = !a.contains('=');
            continue;
        }
        out.push(a);
    }
    out.extend(extra.iter().cloned());
    out
}

/// 2026-10-02: The `met serve` arguments: `recipe` rendered as the serve renders it, then
/// `extra` flags replacing the recipe's, parsed and validated as `met serve` does. Without a
/// recipe the serve's own defaults apply.
pub(crate) fn serve_args(
    root: &Path,
    checkpoint: &str,
    recipe: Option<&str>,
    extra: &[String],
) -> Result<ServeArgs> {
    use clap::Parser as _;
    let mut argv = match recipe {
        Some(id) => {
            let path = root.join("recipes").join(format!("{id}.yaml"));
            let text = std::fs::read_to_string(&path)
                .with_context(|| format!("reading recipe {}", path.display()))?;
            let r = crate::recipe::Recipe::parse(id, &text)?;
            if r.model != checkpoint {
                bail!(
                    "recipe {id} serves {}, not --checkpoint {checkpoint}",
                    r.model
                );
            }
            r.argv(&BTreeMap::new())?
        }
        None => vec!["met".into(), "serve".into(), checkpoint.to_string()],
    };
    argv = override_flags(argv, extra);
    let cli = crate::cli::Cli::try_parse_from(&argv)
        .with_context(|| format!("parsing the serve flags {:?}", &argv[2..]))?;
    let crate::cli::Command::Serve(args) = cli.command else {
        bail!("the serve flags did not parse as `met serve`");
    };
    crate::cli::validate_serve_args(&args).map_err(anyhow::Error::msg)?;
    Ok(args)
}

/// 2026-10-02: The counts the engine derives from a serve's settings.
#[derive(Debug, Clone)]
pub(crate) struct EngineFacts {
    /// 2026-10-02: Sequence slots (`--max-batch-size`, or the `--slots` override).
    pub slots: usize,
    /// 2026-10-02: A speculative proposer is on.
    pub spec: bool,
    /// 2026-10-02: Drafts per verify (`resolve_num_drafts`).
    pub num_drafts: usize,
    /// 2026-10-02: The MTP dispatch cap (`resolve_mtp_max_seqs`).
    pub mtp_max_seqs: usize,
    /// 2026-10-02: `bf16` | `fp8`, as `resolve_kv_dtype_str` resolves it.
    pub kv_dtype: String,
    /// 2026-10-02: `--ssm-h-dtype f16-pool`.
    pub h_f16_pool: bool,
    /// 2026-10-02: The SSM pool's unit counts.
    pub pool: PoolCounts,
    /// 2026-10-02: Marconi snapshot slots.
    pub marconi_slots: usize,
    /// 2026-10-02: Decode-rollback ring slots per sequence slot (the requested depth).
    pub ring_slots: usize,
    /// 2026-10-02: Carried-state verify slots (dummy included) and table rows; 0 without.
    pub carry: (u64, u64),
    /// 2026-10-02: Verify-table rows (WY tables, the accepted-hidden stash); 0 without.
    pub verify_rows: u64,
    /// 2026-10-02: The legacy buffer arena plus the GDN two-phase prefill buffers, bytes.
    pub legacy_arena: u64,
    /// 2026-10-02: Of it, the GDN two-phase prefill buffers.
    pub gdn_two_phase: u64,
    /// 2026-10-02: The runtime `max_batch_tokens` that sized it.
    pub max_batch_tokens: usize,
}

/// 2026-10-02: The decode prefill chunk of an SSM model, the rule `preflight_reserve` applies
/// (serve_phases/preflight.rs:116-127): `--max-prefill-tokens` when set to anything but 8192,
/// else 8192, at most `--max-seq-len`; 0 without SSM layers.
fn ssm_prefill_chunk(args: &ServeArgs, config: &ModelConfig) -> usize {
    if config.num_ssm_layers() == 0 {
        return 0;
    }
    let chunk = if args.max_prefill_tokens != 8192 && args.max_prefill_tokens > 0 {
        args.max_prefill_tokens
    } else {
        8192
    };
    args.max_seq_len.min(chunk)
}

/// 2026-10-02: The GDN two-phase prefill buffers `TransformerModel::new` allocates
/// (model-engine model/impl_a1_init.rs:26-46): qkv BF16, gate and beta F32, out and z BF16, at
/// `min(max_batch_tokens, max_seq_len)` rows; 0 without GDN layers.
fn gdn_two_phase_bytes(config: &ModelConfig, rows: usize) -> usize {
    let key_dim = config.linear_num_key_heads * config.linear_key_head_dim;
    let value_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let conv_dim = key_dim * 2 + value_dim;
    if conv_dim == 0 || config.num_ssm_layers() == 0 {
        return 0;
    }
    rows * conv_dim * 2 + rows * config.linear_num_value_heads * 2 * 4 + 2 * rows * value_dim * 2
}

/// 2026-10-02: The engine's counts for `args` over `config`, with `slots` sequence slots.
/// `tree_nodes` (a planned token tree, DFlash2) verifies like a uniform `K = nodes` verify: every
/// node keeps its recurrent state, as DFlash's γ + 1 rows do (`uniform_h`).
pub(crate) fn engine_facts(
    args: &ServeArgs,
    config: &ModelConfig,
    behavior: &Behavior,
    slots: usize,
    tree_nodes: Option<usize>,
) -> Result<EngineFacts> {
    let mut args = args.clone();
    args.max_batch_size = slots;
    let spec = args.speculative_proposer_requested();
    let (num_drafts, _) = crate::main_modules::serve_phases::config::resolve_num_drafts(
        args.num_drafts,
        behavior.default_num_drafts,
    );
    let uniform = args.dflash || tree_nodes.is_some();
    let num_drafts = match (tree_nodes, args.dflash) {
        (Some(n), _) => n.saturating_sub(1).max(1),
        (None, true) => args.serve_dflash_gamma(),
        (None, false) => num_drafts,
    };
    args.num_drafts = Some(num_drafts);
    let (mtp_max_seqs, _) = metrale_model_layers::speculative::resolve_mtp_max_seqs(
        args.mtp_max_seqs,
        behavior.mtp_max_seqs,
        false,
    )?;
    let h_f16_pool = args.ssm_h_dtype.as_deref() == Some("f16-pool");
    let rollback = match args.ssm_rollback_mode.as_str() {
        "replay" => SsmRollbackMode::Replay,
        _ => SsmRollbackMode::Snapshot,
    };
    let shape = PoolShape {
        max_slots: slots,
        spec,
        num_intermediates: num_drafts + 1,
        num_drafts,
        uniform_h: uniform,
        rollback,
    };
    let mtp_slots = mtp_state_slots_with(slots, mtp_max_seqs, false);
    let pool = pool_counts_with(&shape, mtp_slots, |s| match uniform {
        true => num_drafts,
        false => verify_slot_drafts_with(s, mtp_max_seqs, num_drafts, |n| {
            metrale_model_layers::speculative::mtp_ladder_drafts(n, num_drafts)
        }),
    });
    let (kv_dtype, _) = crate::main_modules::serve_phases::kv_cache::resolve_kv_dtype_str(
        args.kv_cache_dtype.as_deref(),
        &behavior.default_kv_dtype,
    );
    let marconi_slots = ssm_reserve::marconi_snapshot_slots(
        args.ssm_cache_slots,
        ssm_reserve::prefix_caching_active(
            args.prefix_caching_enabled(),
            config.kv_only_prefix_cache_is_safe(),
        ),
    )
    .slots;
    let ring_slots = ssm_reserve::decode_rollback_ring_slots(
        config.num_ssm_layers(),
        args.speculative || args.dflash,
    )
    .slots;
    let gdn_128 = config.linear_key_head_dim == 128 && config.linear_value_head_dim == 128;
    let carry = match spec && !h_f16_pool && config.linear_num_value_heads > 0 && gdn_128 {
        true => (
            mtp_slots as u64 + 1,
            metrale_model_layers::layer::VERIFY_WY_TABLE_SEQS as u64,
        ),
        false => (0, 0),
    };
    let verify_rows = match spec {
        true => metrale_model_layers::layer::VERIFY_WY_TABLE_SEQS as u64,
        false => 0,
    };
    let budget = crate::main_modules::serve_phases::kv_cache::resolve_prefill_budget(
        &args,
        ssm_prefill_chunk(&args, config),
    );
    let m = budget.max_batch_tokens;
    let gdn = gdn_two_phase_bytes(config, m.min(args.max_seq_len));
    let arena = metrale_gpu_runtime::buffers::BufferSizes::from_config(
        config,
        m,
        args.max_seq_len,
        args.block_size,
        slots,
    )
    .total_bytes()
        + gdn;
    Ok(EngineFacts {
        slots,
        spec,
        num_drafts,
        mtp_max_seqs,
        kv_dtype,
        h_f16_pool,
        pool,
        marconi_slots,
        ring_slots,
        carry,
        verify_rows,
        legacy_arena: arena as u64,
        gdn_two_phase: gdn as u64,
        max_batch_tokens: m,
    })
}

#[cfg(test)]
mod override_tests {
    use super::override_flags;

    fn v(s: &str) -> Vec<String> {
        s.split_whitespace().map(String::from).collect()
    }

    /// 2026-10-02: A later flag replaces the recipe's, value and all; a boolean flag stays.
    #[test]
    fn a_later_flag_replaces_the_recipes_value() {
        let argv = v("met serve m --gpu-memory-utilization 0.90 --speculative --max-seq-len 8192");
        assert_eq!(
            override_flags(argv.clone(), &v("--gpu-memory-utilization 0.85")),
            v("met serve m --speculative --max-seq-len 8192 --gpu-memory-utilization 0.85")
        );
        assert_eq!(
            override_flags(argv, &v("--speculative --max-seq-len=4096")),
            v("met serve m --gpu-memory-utilization 0.90 --speculative --max-seq-len=4096")
        );
    }
}

#[cfg(test)]
#[path = "circuit_memory_parity_tests.rs"]
mod circuit_memory_parity_tests;
