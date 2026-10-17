// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: Tests for `serve_settings`: each lever flag outranks its variable and is
//! recorded as `cli`, an absent flag leaves the variable in charge, a malformed variable stops
//! the load, and the watchdog levers resolve as their old readers did except the recorded
//! changes.
//!
//! Owner: server startup.
//! Invariants: none beyond the types.

use metrale_io_host_mem::MapEnv;
use metrale_settings::provenance::Source;

use super::*;

fn parse(extra: &[&str]) -> cli::ServeArgs {
    let mut argv = vec!["met", "serve", "dummy/model", "--model-name", "dummy"];
    argv.extend_from_slice(extra);
    match <cli::Cli as clap::Parser>::try_parse_from(argv)
        .expect("a valid argv")
        .command
    {
        cli::Command::Serve(a) => a,
        _ => unreachable!("a serve command"),
    }
}

fn with_env(env: &[(&str, &str)], flags: &[&str]) -> Result<Arc<InferenceServerSettings>> {
    resolve(&MapEnv::new(env.iter().copied()), &parse(flags))
}

/// 2026-10-17: Every flag `cli_layer` forwards, the variable it names, a flag value, an
/// environment value it must beat, and how to read the resolved lever back.
type Read = fn(&InferenceServerSettings) -> String;

fn forwarded() -> Vec<(
    &'static str,
    &'static [&'static str],
    &'static str,
    Read,
    &'static str,
)> {
    vec![
        (
            "METRALE_MTP_GATE_FORCE",
            &["--mtp-gate", "auto"],
            "1",
            |s| s.speculative.mtp_gate_force.to_string(),
            "false",
        ),
        (
            "METRALE_MTP_DCUT_RATIO",
            &["--mtp-dcut-ratio", "0.5"],
            "0.25",
            |s| s.speculative.mtp_dcut_ratio.to_string(),
            "0.5",
        ),
        (
            "METRALE_MTP_K_LADDER",
            &["--mtp-k-ladder", "4:2,8:1"],
            "16:1",
            |s| format!("{:?}", s.speculative.mtp_k_ladder),
            "Some(\"4:2,8:1\")",
        ),
        (
            "METRALE_CONTENT_LOOP_WATCHDOG",
            &["--content-loop-watchdog", "off"],
            "1",
            |s| format!("{:?}", s.scheduler.content_loop_watchdog),
            "Some(false)",
        ),
        (
            "METRALE_CONTENT_LOOP_MIN_REPEATS",
            &["--content-loop-min-repeats", "5"],
            "9",
            |s| format!("{:?}", s.scheduler.content_loop_min_repeats),
            "Some(5)",
        ),
        (
            "METRALE_MAX_INTER_TOOL_PROSE",
            &["--max-inter-tool-prose", "40"],
            "90",
            |s| format!("{:?}", s.scheduler.max_inter_tool_prose),
            "Some(40)",
        ),
        (
            "METRALE_HERMETIC",
            &["--hermetic"],
            "0",
            |s| format!("{:?}", s.server.hermetic),
            "Some(\"1\")",
        ),
        (
            "METRALE_SSM_TAIL_MIDCHUNK",
            &["--no-ssm-tail-midchunk"],
            "1",
            |s| format!("{:?}", s.scheduler.ssm_tail_midchunk),
            "Some(\"0\")",
        ),
        (
            "METRALE_VISION_MAX_PIXELS",
            &["--vision-max-pixels", "1024"],
            "4096",
            |s| s.model.vision_max_pixels.to_string(),
            "1024",
        ),
    ]
}

#[test]
fn every_forwarded_flag_outranks_its_variable_and_is_recorded_as_cli() {
    for (var, flag, env_value, read, want) in forwarded() {
        let s = with_env(&[(var, env_value)], flag).expect("resolves");
        assert_eq!(read(&s), want, "{var}");
        assert_eq!(s.provenance.source_of(var), Some(Source::Cli), "{var}");
    }
}

#[test]
fn without_its_flag_each_variable_still_decides() {
    for (var, _, env_value, read, _) in forwarded() {
        let s = with_env(&[(var, env_value)], &[]).expect("resolves");
        let from_env = read(&s);
        let default = read(&with_env(&[], &[]).expect("resolves"));
        assert_ne!(
            from_env, default,
            "{var}: the variable did not reach the lever"
        );
        assert_eq!(s.provenance.source_of(var), Some(Source::Env), "{var}");
    }
}

/// 2026-10-17: The flag spellings that mean "not given" add nothing to the CLI layer.
#[test]
fn the_unset_spellings_of_a_flag_leave_the_variable_in_charge() {
    for (var, flag, env_value) in [
        (
            "METRALE_CONTENT_LOOP_WATCHDOG",
            ["--content-loop-watchdog", "auto"],
            "1",
        ),
        (
            "METRALE_VISION_MAX_PIXELS",
            ["--vision-max-pixels", "0"],
            "4096",
        ),
    ] {
        let s = with_env(&[(var, env_value)], &flag).expect("resolves");
        assert_eq!(s.provenance.source_of(var), Some(Source::Env), "{var}");
    }
    let s =
        with_env(&[("METRALE_MTP_GATE_FORCE", "1")], &["--mtp-gate", "auto"]).expect("resolves");
    assert!(
        !s.speculative.mtp_gate_force,
        "an explicit `--mtp-gate auto` is a value, not an absence"
    );
}

#[test]
fn a_malformed_variable_stops_the_load() {
    for (var, junk) in [
        ("METRALE_MTP_GATE_FORCE", "maybe"),
        ("METRALE_MTP_DCUT_RATIO", "wide"),
        ("METRALE_CONTENT_LOOP_WATCHDOG", ""),
        ("METRALE_CONTENT_LOOP_MIN_REPEATS", "three"),
        ("METRALE_MAX_INTER_TOOL_PROSE", "lots"),
    ] {
        assert!(with_env(&[(var, junk)], &[]).is_err(), "{var}={junk:?}");
    }
}

#[test]
fn a_flag_shadows_its_variable_without_parsing_it() {
    // 2026-10-17: The resolver stops at the first layer that holds a value, so a variable under a
    // flag is never parsed. The old readers ignored it too: no behaviour change.
    let s = with_env(
        &[("METRALE_MTP_GATE_FORCE", "maybe")],
        &["--mtp-gate", "force"],
    )
    .expect("the flag decides");
    assert!(s.speculative.mtp_gate_force);
    assert_eq!(
        s.provenance.source_of("METRALE_MTP_GATE_FORCE"),
        Some(Source::Cli)
    );
}

const VALUES: [Option<&str>; 12] = [
    None,
    Some("1"),
    Some("true"),
    Some("yes"),
    Some("0"),
    Some("false"),
    Some("off"),
    Some(""),
    Some(" 7 "),
    Some("garbage"),
    Some("7"),
    Some("-1"),
];

/// 2026-10-17: The pre-settings readers, frozen: `None` is "keeps the lower-precedence value".
fn old_watchdog(v: Option<&str>) -> Option<bool> {
    let v = v?.trim();
    if v == "1" || v.eq_ignore_ascii_case("true") {
        Some(true)
    } else if v == "0" || v.eq_ignore_ascii_case("false") {
        Some(false)
    } else {
        None
    }
}

fn old_min_repeats(v: Option<&str>) -> Option<u32> {
    v?.trim().parse().ok()
}

fn old_inter_tool_prose(v: Option<&str>) -> Option<u32> {
    v?.parse().ok()
}

/// 2026-10-17: Every (variable, value) whose outcome changed, as `Debug` of the new outcome
/// (`Err` when the load now stops). Reviewed: `yes`/`off` now reach the watchdog, the prose cap
/// trims, and every value the old readers ignored now stops the load.
const CHANGES: &[(&str, &str, &str)] = &[
    ("METRALE_CONTENT_LOOP_MIN_REPEATS", "true", "Err"),
    ("METRALE_MAX_INTER_TOOL_PROSE", "true", "Err"),
    ("METRALE_CONTENT_LOOP_WATCHDOG", "yes", "Some(true)"),
    ("METRALE_CONTENT_LOOP_MIN_REPEATS", "yes", "Err"),
    ("METRALE_MAX_INTER_TOOL_PROSE", "yes", "Err"),
    ("METRALE_CONTENT_LOOP_MIN_REPEATS", "false", "Err"),
    ("METRALE_MAX_INTER_TOOL_PROSE", "false", "Err"),
    ("METRALE_CONTENT_LOOP_WATCHDOG", "off", "Some(false)"),
    ("METRALE_CONTENT_LOOP_MIN_REPEATS", "off", "Err"),
    ("METRALE_MAX_INTER_TOOL_PROSE", "off", "Err"),
    ("METRALE_CONTENT_LOOP_WATCHDOG", "", "Err"),
    ("METRALE_CONTENT_LOOP_MIN_REPEATS", "", "Err"),
    ("METRALE_MAX_INTER_TOOL_PROSE", "", "Err"),
    ("METRALE_CONTENT_LOOP_WATCHDOG", " 7 ", "Err"),
    ("METRALE_MAX_INTER_TOOL_PROSE", " 7 ", "Some(7)"),
    ("METRALE_CONTENT_LOOP_WATCHDOG", "garbage", "Err"),
    ("METRALE_CONTENT_LOOP_MIN_REPEATS", "garbage", "Err"),
    ("METRALE_MAX_INTER_TOOL_PROSE", "garbage", "Err"),
    ("METRALE_CONTENT_LOOP_WATCHDOG", "7", "Err"),
    ("METRALE_CONTENT_LOOP_WATCHDOG", "-1", "Err"),
    ("METRALE_CONTENT_LOOP_MIN_REPEATS", "-1", "Err"),
    ("METRALE_MAX_INTER_TOOL_PROSE", "-1", "Err"),
];

#[test]
fn the_watchdog_levers_resolve_as_their_old_readers_did_except_the_recorded_changes() {
    let mut actual: Vec<(&str, &str, String)> = Vec::new();
    for value in VALUES {
        for var in [
            "METRALE_CONTENT_LOOP_WATCHDOG",
            "METRALE_CONTENT_LOOP_MIN_REPEATS",
            "METRALE_MAX_INTER_TOOL_PROSE",
        ] {
            let env: Vec<(&str, &str)> = value.map(|v| (var, v)).into_iter().collect();
            let settings = with_env(&env, &[]);
            let (was, new) = match var {
                "METRALE_CONTENT_LOOP_WATCHDOG" => (
                    format!("{:?}", old_watchdog(value)),
                    settings.map(|s| format!("{:?}", s.scheduler.content_loop_watchdog)),
                ),
                "METRALE_CONTENT_LOOP_MIN_REPEATS" => (
                    format!("{:?}", old_min_repeats(value)),
                    settings.and_then(|s| {
                        crate::lever_value::narrow_opt::<u32>(
                            var,
                            s.scheduler.content_loop_min_repeats,
                        )
                        .map(|n| format!("{n:?}"))
                    }),
                ),
                _ => (
                    format!("{:?}", old_inter_tool_prose(value)),
                    settings.and_then(|s| {
                        crate::lever_value::narrow_opt::<u32>(var, s.scheduler.max_inter_tool_prose)
                            .map(|n| format!("{n:?}"))
                    }),
                ),
            };
            let new = new.unwrap_or_else(|_| "Err".to_string());
            if new != was {
                actual.push((var, value.unwrap_or("<unset>"), new));
            }
        }
    }
    let report: String = actual
        .iter()
        .map(|(var, v, o)| format!("    ({var:?}, {v:?}, {o:?}),\n"))
        .collect();
    let expected: Vec<(&str, &str, String)> = CHANGES
        .iter()
        .map(|&(var, v, o)| (var, v, o.to_string()))
        .collect();
    assert_eq!(actual, expected, "actual changes:\n{report}");
}

/// 2026-10-17: The device cells' old readers, frozen: the flag published a value that won,
/// otherwise `METRALE_HERMETIC == "1"` / `METRALE_SSM_TAIL_MIDCHUNK != "0"`. Applied to the
/// resolved text, the same rules must give the same answer for every flag and value: the CLI
/// layer changes no outcome.
#[test]
fn the_device_flags_compose_with_their_variables_as_the_published_cells_did() {
    let old_hermetic = |flag: bool, v: Option<&str>| flag || v == Some("1");
    let old_midchunk = |flag: bool, v: Option<&str>| !flag && v != Some("0");
    for flag in [false, true] {
        for value in VALUES {
            let h_env: Vec<(&str, &str)> =
                value.map(|v| ("METRALE_HERMETIC", v)).into_iter().collect();
            let h = with_env(&h_env, if flag { &["--hermetic"][..] } else { &[] }).expect("text");
            assert_eq!(
                h.server.hermetic.as_deref() == Some("1"),
                old_hermetic(flag, value),
                "--hermetic={flag} METRALE_HERMETIC={value:?}"
            );
            let m_env: Vec<(&str, &str)> = value
                .map(|v| ("METRALE_SSM_TAIL_MIDCHUNK", v))
                .into_iter()
                .collect();
            let m = with_env(
                &m_env,
                if flag {
                    &["--no-ssm-tail-midchunk"][..]
                } else {
                    &[]
                },
            )
            .expect("text");
            assert_eq!(
                m.scheduler.ssm_tail_midchunk.as_deref() != Some("0"),
                old_midchunk(flag, value),
                "--no-ssm-tail-midchunk={flag} METRALE_SSM_TAIL_MIDCHUNK={value:?}"
            );
        }
    }
}
