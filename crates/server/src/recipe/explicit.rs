// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: Every recipe's `defaults:` must name every recipe-settable serve flag
//! explicitly — nothing may resolve from an implicit `met serve` default.
//!
//! Owner directive, 2026-10-05: the night before, `qwen3.6/qwen3.6-35b-a3b-nvfp4-declared`
//! named neither `max_batch_size` nor `max_num_seqs`, so it silently inherited
//! `--max-batch-size`'s clap default of 8 and throttled an official measurement above C8.
//! "Recipes need to explicitly define each and every variable so nothing is left to
//! defaults." This is the checker that enforces it.
//!
//! Owner: server (recipe).
//! Invariants:
//! - [`required_keys`] is derived from `cli::manifest::build()`, i.e. from
//!   `ServeArgs::command()` itself — clap is the single source of truth for the flag
//!   surface (`schema.rs`'s invariant), so this file never lists a flag by name to
//!   require it. A new clap flag is required automatically; nothing here needs editing
//!   when one is added.
//! - [`EXCLUDED_NAMED`] is the only hand-maintained opt-out, and every entry says why.
//!   The other two exclusions (presence-only flags, flags with no clap default) are
//!   categorical: derived from clap metadata, not named.
//! - [`missing_keys`] takes its required set as a parameter rather than calling
//!   [`required_keys`] itself, so a test can hand it a synthetic set without touching
//!   `ServeArgs` (`explicit_tests::a_new_flag_without_a_recipe_value_fails`).

use std::collections::BTreeSet;

use super::Recipe;

/// 2026-10-05: Flags with a concrete clap default that are still not a recipe's decision.
/// Every other required flag is derived from `ServeArgs::command()`; these five are the
/// only names this file hard-codes, each with why it is launch-time/topology rather than
/// model configuration.
const EXCLUDED_NAMED: &[(&str, &str)] = &[
    (
        "gpu_ordinal",
        "which physical GPU this process binds to is a launch-time placement decision \
         (which box/slot the operator starts the server on), not a property of the model \
         being served; the same recipe is reused across GPUs and boxes.",
    ),
    (
        "rank",
        "this process's role (head/worker) in a multi-node launch; the same recipe starts \
         every rank, with --rank passed at launch time, never pinned in the recipe.",
    ),
    (
        "world_size",
        "derived from the recipe's own top-level `min_nodes` key and pushed onto argv by \
         `Recipe::argv_edited` itself (see mod.rs); a `defaults:` entry would be a second, \
         possibly conflicting, source of truth for the same decision.",
    ),
    (
        "master_addr",
        "the NCCL bootstrap (head node) address is cluster topology, decided at launch \
         time, not a model-recipe property.",
    ),
    (
        "master_port",
        "the NCCL bootstrap port is a launch-time connection detail, not a model-recipe \
         property.",
    ),
    (
        "prompt_lookup_shared_cache_mb",
        "clap `requires = \"prompt_lookup_decoding\"` (serve_args_prompt_lookup.rs): giving \
         this flag at all — even its own default value, 0 — without also giving the \
         presence-only `--prompt-lookup-decoding` is refused at parse. Pinning it on every \
         recipe would force every recipe to also turn prompt-lookup decoding on, which is \
         not this pass's business; it inherits the same exemption as a presence-only flag \
         that is off.",
    ),
];

/// 2026-10-05: Why a presence-only flag (clap `ArgAction::SetTrue`) needs no explicit key.
/// `recipe::schema::check_recipe_default` already refuses a recipe that writes `key: false`
/// for one of these: the flag's absence IS the enforced, canonical spelling of "off".
/// Requiring the key here would ask every recipe to write a value the engine refuses, so
/// these flags are complete by construction: present (`key: true`) to turn a feature on,
/// absent to leave it off.
pub const PRESENCE_ONLY_REASON: &str = "presence-only (clap ArgAction::SetTrue): schema::check_recipe_default already refuses \
     a redundant `false`, so omission is the one enforced spelling of off.";

/// 2026-10-05: Why a flag with no clap default needs no explicit key (yet).
/// `Option<T>`/`Vec<T>` fields with no `default_value` resolve through their own documented
/// precedence chain (flag -> env lever -> MODEL.toml -> built-in constant), and several of
/// them change which code path runs merely by being GIVEN, independent of the value given
/// (e.g. `ssm_h_dtype`, `mtp_gate`, `ssm_batched_recurrent on|off`: `KernelFlagPlan::from_args`
/// treats "any GDN flag given" as owning the whole GDN cell, so pinning one is a real
/// behavior change, not a no-op explicitness fix). Each needs its own per-model
/// measurement before it can be pinned — tracked as follow-up, out of scope for this pass,
/// whose target is the class of bug `--max-batch-size` was: a flat clap default with no
/// model-aware fallback at all.
pub const NO_CLAP_DEFAULT_REASON: &str = "clap gives this flag no default (Option<T>/Vec<T>, no default_value): its unset state \
     is a deliberate 'defer to MODEL.toml/env/checkpoint' signal, not a flat wrong fallback. \
     Pinning one changes whether the flag reads as given downstream, so it needs a per-model \
     measurement, not a mechanical fill. Follow-up, out of scope for this pass.";

/// 2026-10-05: The recipe `defaults:` keys every recipe must set.
///
/// Built from `cli::manifest::build()`'s flags (itself built from `ServeArgs::command()`):
/// every flag that is not presence-only, carries a concrete clap default, and is not in
/// [`EXCLUDED_NAMED`]. When a flag has a recipe-spelling alias (`schema::RENAMES`, e.g.
/// `host` for `--bind`, `max_model_len` for `--max-seq-len`), the alias is required — it is
/// the spelling every recipe in the tree already uses.
pub fn required_keys() -> BTreeSet<String> {
    let excluded: BTreeSet<&str> = EXCLUDED_NAMED.iter().map(|(k, _)| *k).collect();
    crate::cli::manifest::build()
        .flags
        .into_iter()
        .filter(|f| !f.presence_only)
        .filter(|f| f.default.is_some())
        .filter(|f| !excluded.contains(f.key.as_str()))
        .map(|f| f.recipe_aliases.into_iter().next().unwrap_or(f.key))
        .collect()
}

/// 2026-10-05: The keys of `required` that `recipe.defaults` does not set.
///
/// Takes `required` as a parameter (rather than calling [`required_keys`] itself) so a test
/// can hand it a synthetic set that includes a key no real flag has, proving the checker
/// catches an addition to the required surface without editing `ServeArgs`
/// (`explicit_tests::a_new_flag_without_a_recipe_value_fails`).
pub fn missing_keys(required: &BTreeSet<String>, recipe: &Recipe) -> BTreeSet<String> {
    required
        .iter()
        .filter(|k| !recipe.defaults.contains_key(k.as_str()))
        .cloned()
        .collect()
}

#[cfg(test)]
#[path = "explicit_tests.rs"]
mod tests;
