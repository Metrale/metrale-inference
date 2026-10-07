// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Golden-trace tests: every scenario's live trace must equal the file under `golden/`.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.
//!
//! Regenerate with `cargo test -p metrale-server trace_harness --
//! --ignored write_goldens`.

use std::path::PathBuf;
use std::sync::Mutex;

use super::runner::{Scenario, run_scenario};
use super::scenarios;

#[test]
fn masked_greedy_unsupported_model_retains_exact_host_rows() {
    use super::model::{ModelCfg, RecordingModel};
    use crate::scheduler::io::{DecodeRows, Readback, sync_device::execute_readback};
    use metrale_model_engine::traits::ModelLogits;
    let model = RecordingModel::new(ModelCfg::default());
    model.state.lock().unwrap().rows = vec![3, 7];
    let ptr = model.row_ptr(0);
    let masks = vec![vec![3], vec![7]];
    assert_eq!(model.argmax_batch_masked(ptr, 2, &masks, 0).unwrap(), None);
    let mut expected = vec![0; model.vocab_size() * 2 * 2];
    model.copy_logits_to_host(ptr, &mut expected).unwrap();
    let mut actual = vec![99; 5];
    let result = execute_readback(
        &model,
        ptr,
        2,
        Readback::MaskedGreedy {
            masks,
            into: &mut actual,
        },
    )
    .unwrap();
    assert!(matches!(result, DecodeRows::HostLogits { elem_bytes: 2 }));
    assert_eq!(actual, expected);
}

/// 2026-09-25: The scheduler keys its spill directory on the process id
/// (`core/mod.rs`), so scenarios must not run concurrently.
pub(super) static SERIAL: Mutex<()> = Mutex::new(());

pub(super) fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/scheduler/trace_harness/golden")
}

fn scenario(name: &str) -> Scenario {
    scenarios::all()
        .into_iter()
        .find(|s| s.name == name)
        .unwrap_or_else(|| panic!("no scenario {name}"))
}

fn check(name: &str) {
    check_against_golden(&scenario(name));
}

/// 2026-10-03: `sc`'s live trace must equal the golden of the scenario it is named after.
fn check_against_golden(sc: &Scenario) {
    let name = sc.name;
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let live = run_scenario(sc).join("\n") + "\n";
    let path = golden_dir().join(format!("{name}.trace"));
    let golden = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("golden {} unreadable: {e}", path.display()));
    if live != golden {
        let first = live
            .lines()
            .zip(golden.lines())
            .position(|(a, b)| a != b)
            .unwrap_or(live.lines().count().min(golden.lines().count()));
        panic!(
            "trace for {name} differs from {} at line {}:\n  live:   {}\n  golden: {}",
            path.display(),
            first + 1,
            live.lines().nth(first).unwrap_or("<end>"),
            golden.lines().nth(first).unwrap_or("<end>"),
        );
    }
}

#[test]
#[ignore = "regenerates the golden traces; run explicitly"]
fn write_goldens() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    std::fs::create_dir_all(golden_dir()).unwrap();
    for sc in scenarios::all() {
        let text = run_scenario(&sc).join("\n") + "\n";
        std::fs::write(golden_dir().join(format!("{}.trace", sc.name)), text).unwrap();
    }
}

macro_rules! golden_tests {
    ($($fn_name:ident => $scenario:literal),* $(,)?) => {
        $( #[test] fn $fn_name() { check($scenario); } )*
    };
}

golden_tests! {
    trace_plain_decode => "plain_decode",
    trace_chunked_prefill => "chunked_prefill",
    trace_mixed_prefill => "mixed_prefill",
    trace_batched_prefill_waves => "batched_prefill_waves",
    trace_batched_mixed => "batched_mixed",
    trace_verify_k2 => "verify_k2",
    trace_verify_k3 => "verify_k3",
    trace_verify_k4 => "verify_k4",
    trace_batched_verify_k4 => "batched_verify_k4",
    trace_dflash => "dflash",
    trace_dflash_batched => "dflash_batched",
    trace_ngram => "ngram",
    trace_self_spec => "self_spec",
    trace_beam => "beam",
    trace_preempt_requeue => "preempt_requeue",
    trace_spill_swap_out_in => "spill_swap_out_in",
    trace_cancel_mid_stream => "cancel_mid_stream",
    trace_request_timeout => "request_timeout",
    trace_lora_rotation_at_quiescence => "lora_rotation_at_quiescence",
    trace_slai_policy => "slai_policy",
    trace_watchdog_rollback => "watchdog_rollback",
    trace_host_logits_paths => "host_logits_paths",
}

/// 2026-10-03: D15 (CI run 36999630203). The inbox closes while the loop is applying the
/// rotation, so the close is visible to that tick's shutdown check; the harness used to close
/// it only after the rotation ack, which races the same check. A tick that applied a rotation
/// must not end the loop, so the trace is the golden one either way.
#[test]
fn trace_lora_rotation_with_the_inbox_closed_during_the_rotation() {
    let mut sc = scenario("lora_rotation_at_quiescence");
    sc.opts.close_inbox_at = Some("set_active_lora(");
    check_against_golden(&sc);
}

#[test]
fn every_scenario_has_a_golden_test() {
    let names: Vec<&str> = scenarios::all().iter().map(|s| s.name).collect();
    for n in &names {
        assert!(
            golden_dir().join(format!("{n}.trace")).exists(),
            "scenario {n} has no golden trace"
        );
    }
    assert_eq!(
        names.len(),
        22,
        "add a golden_tests! entry for every scenario"
    );
}
