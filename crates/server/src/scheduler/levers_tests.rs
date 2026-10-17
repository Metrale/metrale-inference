// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Scheduler lever resolution tests: lever polarities in
//! `defaults()` and in `from_settings()` over an injected environment, the
//! command-line layer's precedence, and the guard against environment reads on
//! the per-token path.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use metrale_io_host_mem::MapEnv;
use metrale_settings::sources::{CliLayer, RecipeLayer};

use super::*;

/// 2026-10-17: `from_settings` over `env` and the command-line pairs `cli`, as `met serve`
/// resolves a load. No process environment is read or written.
pub(super) fn resolved(env: &[(&str, &str)], cli: &[(&str, &str)]) -> anyhow::Result<SchedLevers> {
    let env = MapEnv::new(env.iter().copied());
    let cli = CliLayer::from_pairs(cli.iter().map(|&(k, v)| (k, v.to_string())))?;
    let settings = metrale_settings::resolve(&env, &cli, &RecipeLayer::empty())?;
    SchedLevers::from_settings(&settings, None)
}

fn live() -> SchedLevers {
    resolved(&[], &[]).expect("an empty environment resolves")
}

#[test]
fn the_five_opt_out_levers_ship_on() {
    // 2026-09-25: Each is spelled as a negative env var; an opt-in resolver
    // would turn all five off.
    let d = SchedLevers::defaults();
    assert!(d.fast_greedy_grammar, "METRALE_DISABLE_FAST_GREEDY");
    assert!(d.fast_masked, "METRALE_DISABLE_FAST_MASKED");
    assert!(d.mtp_minp, "METRALE_NO_MTP_MINP");
    assert!(d.mtp_verify_sample, "METRALE_NO_MTP_VERIFY_SAMPLE");
    assert!(d.forced_token_fastpath, "METRALE_DISABLE_FORCED_TOKEN");
}

/// 2026-09-25: The two turn-termination levers ship ON in the resolver the
/// server uses, not just in `defaults()` (a hand-written literal), and `0` or
/// `false` turns each off.
#[test]
fn the_turn_termination_levers_ship_on_in_the_live_resolver() {
    let live = live();
    assert!(live.tool_response_stop, "METRALE_TOOL_RESPONSE_STOP");
    assert!(live.tool_eos_escape, "METRALE_TOOL_EOS_ESCAPE");
    assert!(SchedLevers::defaults().tool_response_stop);
    assert!(SchedLevers::defaults().tool_eos_escape);
    for off in ["0", "false", "FALSE", "  0  "] {
        let l = resolved(
            &[
                ("METRALE_TOOL_RESPONSE_STOP", off),
                ("METRALE_TOOL_EOS_ESCAPE", off),
            ],
            &[],
        )
        .expect("an off spelling resolves");
        assert!(!l.tool_response_stop && !l.tool_eos_escape, "{off:?}");
    }
}

#[test]
fn every_opt_in_lever_ships_off() {
    let d = SchedLevers::defaults();
    assert!(!d.force_temp_zero);
    assert!(!d.dflash_masked_verify && !d.dflash_adaptive && !d.dflash_spec_think);
    assert!(!d.disable_watchdogs);
    assert!(!d.decode_timing && !d.mtp_timing && !d.adadec_diagnostic);
}

/// 2026-09-25: `defaults()` is a hand-written struct literal; `from_settings()`
/// is what `met serve` calls, so this asserts the resolver itself, with no
/// variable set. `mtp_gate::spec_dispatch_eligible` reads `dflash_spec_think`
/// as
///
///     if inside_thinking && !spec_think { return false; }
///
/// for both lanes, so defaulting it on would let plain MTP speculate inside
/// `<think>`.
#[test]
fn spec_think_is_off_in_the_resolver_the_server_actually_uses() {
    let live = live();
    assert!(
        !live.dflash_spec_think,
        "METRALE_DFLASH_SPEC_THINK must stay OPT-IN: from_settings() resolved it ON. \
         It is the one lever here that is not gated behind dflash_verify_raw_argmax, \
         so defaulting it on changes plain-MTP serving and deterministically \
         damages agentic trajectories. See mtp_gate::spec_dispatch_eligible."
    );
    assert!(
        live.dflash_masked_verify,
        "masked_verify is intentionally default-ON"
    );
    assert!(
        live.dflash_seam_serial,
        "seam_serial is intentionally default-ON"
    );
}

#[test]
fn the_loop_watchdog_is_toggleable_at_runtime() {
    // 2026-09-25: The one lever that changes mid-run: the TUI ops REPL
    // flips it.
    let d = SchedLevers::defaults();
    assert!(!d.loop_watchdog());
    d.set_loop_watchdog(true);
    assert!(d.loop_watchdog());
    d.set_loop_watchdog(false);
    assert!(!d.loop_watchdog());
}

/// 2026-10-17: `--mtp-gate` reaches the levers through the command-line layer
/// (`serve_settings`), which outranks `METRALE_MTP_GATE_FORCE`; with no flag
/// the variable decides.
#[test]
fn the_mtp_gate_flag_outranks_the_legacy_variable() {
    let gate = |env: &[(&str, &str)], cli: &[(&str, &str)]| {
        resolved(env, cli).expect("resolves").mtp_gate_force
    };
    assert!(!gate(&[], &[]), "unset: auto");
    assert!(gate(&[("METRALE_MTP_GATE_FORCE", "1")], &[]));
    assert!(gate(&[], &[("METRALE_MTP_GATE_FORCE", "on")]));
    assert!(
        !gate(
            &[("METRALE_MTP_GATE_FORCE", "1")],
            &[("METRALE_MTP_GATE_FORCE", "off")]
        ),
        "an explicit `--mtp-gate auto` overrides the variable"
    );
}

/// 2026-10-17: `--mtp-dcut-ratio` outranks `METRALE_MTP_DCUT_RATIO`; either
/// value snaps to a bucket, and a value that is not a finite number stops the
/// load instead of falling back to the default.
#[test]
fn the_dcut_ratio_resolves_flag_then_variable_then_default_and_snaps() {
    let ratio = |env: &[(&str, &str)], cli: &[(&str, &str)]| {
        resolved(env, cli).expect("resolves").dcut_ratio
    };
    assert_eq!(ratio(&[], &[]), 0.75);
    assert_eq!(ratio(&[("METRALE_MTP_DCUT_RATIO", "0.3")], &[]), 0.25);
    assert_eq!(
        ratio(
            &[("METRALE_MTP_DCUT_RATIO", "0.3")],
            &[("METRALE_MTP_DCUT_RATIO", "0.9")]
        ),
        1.0
    );
    for junk in ["junk", "", "inf", "NaN"] {
        assert!(
            resolved(&[("METRALE_MTP_DCUT_RATIO", junk)], &[]).is_err(),
            "{junk:?}"
        );
    }
}

/// 2026-10-17: The D-Cut width cap's declared default is the constant the
/// pruning code documents.
#[test]
fn the_dcut_width_cap_default_is_the_documented_constant() {
    assert_eq!(
        live().dcut_width_cap,
        crate::scheduler::mtp_dcut::DCUT_WIDTH_CAP_DEFAULT
    );
    assert_eq!(
        SchedLevers::defaults().dcut_width_cap,
        crate::scheduler::mtp_dcut::DCUT_WIDTH_CAP_DEFAULT
    );
}

/// 2026-10-17: A resolved integer that does not fit its field stops the load:
/// a negative width or a millisecond window past `u64` is not clamped.
#[test]
fn an_integer_lever_out_of_its_field_range_is_an_error() {
    assert!(resolved(&[("METRALE_MTP_DCUT_MAX_SEQS", "-1")], &[]).is_err());
    assert!(resolved(&[("METRALE_DFLASH_ADAPTIVE_REPROBE", "-1")], &[]).is_err());
    assert_eq!(
        resolved(&[("METRALE_MTP_DCUT_MAX_SEQS", " 16 ")], &[])
            .expect("trimmed")
            .dcut_width_cap,
        16
    );
}

#[test]
fn two_runs_hold_independent_levers() {
    let a = SchedLevers::defaults();
    let b = SchedLevers {
        dflash_adaptive: true,
        ..SchedLevers::defaults()
    };
    assert!(!a.dflash_adaptive && b.dflash_adaptive);
    a.set_loop_watchdog(true);
    assert!(!b.loop_watchdog(), "and independent runtime state");
}

/// 2026-09-25: The scheduler thread must not read the environment per token.
///
/// `emit_step` and `decode_logits_step` run once per generated token per
/// sequence. An `std::env::var` call there allocates and takes the
/// process-wide environment lock, once per token per sequence instead of
/// once per run. The levers this path needs are `SchedLevers` fields.
///
/// A source-level check because the property is "who may read the
/// environment", which no runtime assertion can observe. The sibling table
/// for `metrale-model-layers` lives in `layers/ops/hot_path_env_guards.rs`.
#[test]
fn the_per_token_scheduler_path_does_not_read_the_environment() {
    // 2026-09-26: (module, functions still allowed to read). A module is
    // its `<module>.rs` plus every non-test `.rs` file under `<module>/`, so
    // code that a split moves into a child file stays scanned.
    const GUARDED: [(&str, &[&str]); 7] = [
        ("emit_step", &[]),
        // 2026-09-25: Per verify step.
        ("verify_dflash_step", &[]),
        ("verify_k2_step", &[]),
        // 2026-09-25: Per prefill chunk.
        ("prefill_a_step", &[]),
        (
            // 2026-09-25: Per sequence per decode step.
            "decode_logits_seq",
            &["process_seq_logits"],
        ),
        // 2026-09-25: Every lever this module needs is a `SchedLevers`
        // field, so the whole module is on the no-read list.
        ("decode_logits_step", &[]),
        // 2026-10-17: `WatchdogParams::from_behavior` takes its operator
        // overrides as arguments now, so nothing here reads.
        ("helpers", &[]),
    ];
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/scheduler");
    let mut offenders = Vec::new();
    let mut scanned = 0usize;
    for (module, allowed) in GUARDED {
        for path in module_files(&dir, module) {
            scanned += 1;
            let file = path.strip_prefix(&dir).unwrap_or(&path).display();
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("{} is a guarded path: {e}", path.display()));
            let mut current = "<file scope>".to_string();
            for (i, line) in text.lines().enumerate() {
                let trimmed = line.trim_start();
                for prefix in ["pub(crate) fn ", "pub(super) fn ", "pub fn ", "fn "] {
                    if let Some(rest) = trimmed.strip_prefix(prefix) {
                        current = rest
                            .split(['(', '<'])
                            .next()
                            .unwrap_or("?")
                            .trim()
                            .to_string();
                        break;
                    }
                }
                // 2026-09-25: Comments name these variables throughout; only
                // code counts.
                let code = line.split("//").next().unwrap_or("");
                if code.contains("std::env::var") && !allowed.contains(&current.as_str()) {
                    offenders.push(format!("{file}:{} in `{current}`", i + 1));
                }
            }
        }
    }
    // 2026-09-26: 16 files hold these modules today. A lower count means a
    // module moved out from under this table, not that the path got smaller.
    assert!(scanned >= 16, "only {scanned} guarded files were found");
    assert!(
        offenders.is_empty(),
        "environment reads on the per-token scheduler path. Resolve the \
         variable ONCE into `SchedLevers` and read `sched.levers` instead — \
         and reuse the parser in `helpers`, do not re-spell the rule: \
         {offenders:?}"
    );
}

/// 2026-09-25: The verify-step levers' polarities. `dflash_eagle_fix` is
/// asserted against `from_settings()` as well as `defaults()`, for the reason
/// `spec_think_is_off_in_the_resolver_the_server_actually_uses` exists.
#[test]
fn the_verify_step_levers_hold_their_polarities() {
    let d = SchedLevers::defaults();
    assert!(d.dflash_eagle_fix, "the EAGLE append fix ships ON");
    assert!(!d.dflash_step_timing);
    assert!(!d.vision_timing);
    assert!(
        live().dflash_eagle_fix,
        "METRALE_DFLASH_EAGLE_FIX must stay DEFAULT-ON in the resolver the \
         server actually uses — `defaults()` is a hand-written literal and \
         cannot catch a change here"
    );
}

/// 2026-09-26: `<module>.rs` and every `.rs` file below `<module>/` whose
/// name does not end in `tests.rs`. The module file itself must exist.
fn module_files(dir: &std::path::Path, module: &str) -> Vec<std::path::PathBuf> {
    let root = dir.join(format!("{module}.rs"));
    assert!(root.is_file(), "{} is a guarded path", root.display());
    let mut files = vec![root];
    let mut pending = vec![dir.join(module)];
    while let Some(d) = pending.pop() {
        if !d.is_dir() {
            continue;
        }
        for entry in std::fs::read_dir(&d).expect("readable scheduler directory") {
            let path = entry.expect("readable scheduler directory").path();
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if path.is_dir() {
                pending.push(path);
            } else if name.ends_with(".rs") && !name.ends_with("tests.rs") {
                files.push(path);
            }
        }
    }
    files
}

#[path = "levers_equivalence_tests.rs"]
mod equivalence;
