// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: Every recipe's `defaults:` must name every recipe-settable serve flag
//! explicitly — nothing may resolve from an implicit `met serve` default, an env lever,
//! or a MODEL.toml fallback that the recipe doesn't name itself.
//!
//! Owner directive, 2026-10-05: the night before, `qwen3.6/qwen3.6-35b-a3b-nvfp4-declared`
//! named neither `max_batch_size` nor `max_num_seqs`, so it silently inherited
//! `--max-batch-size`'s clap default of 8 and throttled an official measurement above C8.
//! "Recipes need to explicitly define each and every variable so nothing is left to
//! defaults" — including presence-only booleans (state `true` or `false`, not just
//! "true or absent") and flags that resolve through MODEL.toml/env precedence chains
//! (pin the value the chain resolves to today, not just the flags with a flat clap
//! default). "Not applicable to this model" is explicitly not an exclusion either:
//! `src_lang`/`tgt_lang` are NLLB/M2M-only and read for no recipe in this tree, but they
//! are still required and pinned (to a value proven inert for every other `model_type`,
//! not merely omitted) — [`EXCLUDED_NAMED`] holds only flags with no value that can be
//! pinned at all, never flags a given model simply doesn't exercise.
//!
//! Owner: server (recipe).
//! Invariants:
//! - [`required_keys`] is derived from `cli::manifest::build()`, i.e. from
//!   `ServeArgs::command()` itself — clap is the single source of truth for the flag
//!   surface (`schema.rs`'s invariant), so this file never lists a flag by name to
//!   require it. A new clap flag is required automatically; nothing here needs editing
//!   when one is added.
//! - [`EXCLUDED_NAMED`] is the only opt-out, and every entry says why. An exclusion must
//!   be structural (a secret, a launch-time/topology fact, a value clap itself refuses to
//!   accept, a flag a repeatable/Vec schema cannot express, or a flag that provably
//!   changes which code path runs merely by being given — see the GDN-family comment on
//!   `ssm_h_dtype`, which is NOT excluded, for the contrast). "Not applicable to this
//!   model" and "no recipe currently needs this" are not valid reasons by themselves.
//! - [`missing_keys`] takes its required set as a parameter rather than calling
//!   [`required_keys`] itself, so a test can hand it a synthetic set without touching
//!   `ServeArgs` (`explicit_tests::a_new_flag_without_a_recipe_value_fails`).

use std::collections::BTreeSet;

use super::Recipe;

/// 2026-10-05: Flags excluded from [`required_keys`], each with why. Two families:
///
/// 1. Flags WITH a concrete clap default that are still not a recipe's decision
///    (`gpu_ordinal` .. `master_port`): launch-time/topology facts, not model config.
/// 2. Flags with NO clap default (`Option<T>`/`Vec<T>`) that cannot safely be pinned to a
///    pretend "resolved value": secrets, box-specific paths, a value clap refuses outright,
///    a Vec/repeatable field the `defaults:` schema (one scalar string per key) cannot
///    represent zero-or-many occurrences of, or a flag that demonstrably changes which
///    code path executes merely by being given (verified against the engine's own
///    resolution code, not assumed from its doc text — `mtp_k_ladder` and
///    `mtp_dcut_ratio` replace an ADAPTIVE depth-selection process with a pinned table the
///    moment either is given at all, per `publish_mtp_k_ladder`/`set_mtp_k_ladder`).
///
/// `ssm_h_dtype` is deliberately NOT here even though it has the same "given changes the
/// code path" property as `mtp_k_ladder` (`KernelFlagPlan::from_args`'s `gdn_given`): unlike
/// the MTP ladder, its downstream GDN cell is a finite, fully-typed bundle (h-dtype,
/// fused-norm, batched-recurrent, exact-verify) that can be bundled and verified bit-for-bit
/// against today's env-read resolution (`explicit_tests::the_pinned_gdn_bundle_matches_the_
/// documented_unsealed_default`), not an open-ended adaptive process. See [`required_keys`]'s
/// doc for the pinning rule.
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
         presence-only `--prompt-lookup-decoding` is refused at parse. Every recipe now \
         pins `prompt_lookup_decoding: false` (it is a required presence-only flag), so \
         this one structurally cannot be given at all while that stays false.",
    ),
    (
        "auth_token",
        "a bearer-token secret; must never be committed to a recipe file.",
    ),
    (
        "auth_tokens_file",
        "path to a secrets file, box-specific; never committed to a recipe.",
    ),
    (
        "cache_dir",
        "local HuggingFace cache directory path, box-specific, not a model-recipe fact.",
    ),
    (
        "model_from_path",
        "an ad hoc local path override of `model`; the recipe's own top-level `model:` \
         field already names the checkpoint, and this flag exists for local testing \
         outside the recipe mechanism entirely.",
    ),
    (
        "dump",
        "a debug request/response dump file path: a diagnostic opt-in tied to one \
         investigation session, not a model-recipe fact.",
    ),
    (
        "kernel_target",
        "disambiguates among compiled kernel targets only when more than one matches the \
         checkpoint; which targets exist is a property of the BINARY BUILD, not the \
         recipe, so pinning a specific target id would make the recipe refuse to serve \
         against any build that doesn't carry that exact id.",
    ),
    (
        "warmup_prompt",
        "giving this flag ANY value is refused at startup ('not implemented: rejected'); \
         there is no value to pin, only absence is valid.",
    ),
    (
        "spec_cost_table",
        "only consulted under --spec-cost-model measured, which every recipe pins `off` \
         (a clap-default, required field); requires a real file from `met benchmark \
         spec-cost-table --out`, which does not exist for any current recipe.",
    ),
    (
        "spec_cost_calibration",
        "same as spec_cost_table: requires a real generated calibration file no current \
         recipe has.",
    ),
    (
        "spec_cost_recipe",
        "only meaningful paired with spec_cost_table/spec_cost_calibration, neither of \
         which exists for any current recipe.",
    ),
    (
        "spec_cost_slack",
        "only consulted under --spec-cost-model measured, which every recipe pins `off`.",
    ),
    (
        "lora_adapter",
        "repeatable (`Vec<(name, path)>`); the `defaults:` schema carries one scalar \
         string per key, so it structurally cannot encode zero-or-many occurrences. \
         Omission already means zero adapters, the only state the schema can express.",
    ),
    (
        "lora_stageable",
        "repeatable; the same structural limit as --lora-adapter.",
    ),
    (
        "lora_stageable_disk",
        "repeatable; the same structural limit as --lora-adapter.",
    ),
    (
        "max_lora_rank",
        "meaningful only relative to configured --lora-adapter/--lora-stageable* \
         adapters, which no recipe configures (see those three keys, excluded above for \
         a structural reason, not an applicability one) — no adapters means no rank to \
         pin against.",
    ),
    (
        "draft_model",
        "the DFlash drafter's own checkpoint id; read only under --dflash. The one \
         DFlash recipe in the tree already pins its own draft_model explicitly; for every \
         other recipe --dflash is false and this flag is never read, with no fallback \
         value the engine would pick instead (unlike ssm_h_dtype's f32, there is no \
         default drafter).",
    ),
    (
        "dflash_gamma",
        "derived from the DFlash drafter's own trained block size when --dflash is on; \
         for every recipe but the DFlash one (which may pin its own) --dflash is false \
         and this is never read.",
    ),
    (
        "draft_confidence_stop",
        "a confidence threshold TAU in the open interval (0,1) with no default; 'off' \
         cannot be expressed as a TAU value distinct from a real threshold (every value \
         in range IS a real, different threshold), so absence is the only way to say 'no \
         confidence-stop'.",
    ),
    (
        "mtp_k_ladder",
        "giving this flag AT ALL pins the per-batch-width draft ladder \
         (`serve_phases::config::publish_mtp_k_ladder` -> `metrale_speculative::spec::\
         set_mtp_k_ladder`), replacing the engine's adaptive depth-selection process with \
         a fixed table — a real code-path change even when the fixed table's numbers \
         happen to match the adaptive process's typical output on one model, not a \
         notational default. Needs a per-model ladder-vs-adaptive measurement before it \
         can safely be pinned; out of scope for this pass.",
    ),
    (
        "mtp_dcut_ratio",
        "same family as mtp_k_ladder: consumed directly into `SchedLevers::from_env` \
         alongside the adaptive gate state, with no confirmed value-only equivalence to \
         its absence. Needs the same per-model verification before it can be pinned; out \
         of scope for this pass.",
    ),
    (
        "default_chat_template_kwargs",
        "a JSON object merged against MODEL.toml's own default AND the request body, at \
         three different precedence levels (an explicit client request still wins); \
         pinning a static blob here risks silently overriding client-sent thinking \
         preferences on every request that doesn't set its own — a correctness risk, not \
         a performance one. Needs per-model review; out of scope for this pass.",
    ),
    (
        "tool_call_parser",
        "mostly pinnable (MODEL.toml behavior.tool_call_parser, else tool_defaults.toml's \
         model_type mapping, else no parser) — EXCEPT `validate_serve_args` accepts a \
         narrower value set for the CLI flag than `tool_defaults.toml` can resolve \
         internally: the deepseek_v4/deepseek_v41 model_type family maps to the internal \
         format \"deepseek_v4\", which --tool-call-parser itself refuses ('is not a valid \
         value'; verified against the real clap validator, not assumed). A recipe whose \
         model_type hits that gap cannot pin this flag at all without breaking its own \
         tool calling, and a per-recipe carve-out would contradict this checker's one-\
         key-one-rule design, so the key is excluded uniformly rather than for one recipe.",
    ),
    (
        "high_speed_swap_dir",
        "`validate_serve_args` refuses giving this without --high-speed-swap (verified: \
         '--high-speed-swap-dir, --high-speed-swap-gb, --high-speed-swap-resident-blocks \
         set without --high-speed-swap'); the feature is off (false) for every recipe, so \
         there is no way to pin even its documented fallback value without also turning \
         the feature on.",
    ),
    (
        "high_speed_swap_gb",
        "same `validate_serve_args` refusal as high_speed_swap_dir.",
    ),
    (
        "high_speed_swap_resident_blocks",
        "same `validate_serve_args` refusal as high_speed_swap_dir.",
    ),
    (
        "max_thinking_budget",
        "mostly pinnable (MODEL.toml behavior.max_thinking_budget, else the engine's \
         DEFAULT_MAX_THINKING_BUDGET build-time constant) — EXCEPT `validate_serve_args` \
         refuses giving it together with `--disable-thinking` ('there is nothing for the \
         budget to cap'; verified against the real validator). Several recipes in this \
         tree pin `disable_thinking: true`, so there is no universal resolved value; same \
         one-key-one-rule reasoning as `tool_call_parser` above, not a per-recipe carve-out.",
    ),
];

/// 2026-10-05: The recipe `defaults:` keys every recipe must set.
///
/// Built from `cli::manifest::build()`'s flags (itself built from `ServeArgs::command()`):
/// every flag not in [`EXCLUDED_NAMED`]. When a flag has a recipe-spelling alias
/// (`schema::RENAMES`, e.g. `host` for `--bind`, `max_model_len` for `--max-seq-len`), the
/// alias is required — it is the spelling every recipe in the tree already uses.
///
/// Every flag is required, including presence-only booleans (pin `true` or `false`
/// explicitly — `recipe::schema::check_recipe_default` accepts both since 2026-10-05) and
/// flags with no clap default (pin the value the flag's own documented precedence chain
/// resolves to today — MODEL.toml, then an engine built-in constant, verified per flag
/// against the engine's own resolution function, never guessed from its doc text alone).
/// The only exemption is [`EXCLUDED_NAMED`].
pub fn required_keys() -> BTreeSet<String> {
    let excluded: BTreeSet<&str> = EXCLUDED_NAMED.iter().map(|(k, _)| *k).collect();
    crate::cli::manifest::build()
        .flags
        .into_iter()
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
