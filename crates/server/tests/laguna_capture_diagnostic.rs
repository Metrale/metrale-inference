// SPDX-License-Identifier: MIT OR Apache-2.0
#![cfg(feature = "laguna-diagnostic-capture")]
#[allow(dead_code)]
#[path = "../src/scheduler/io/laguna_capture.rs"]
mod capture;
use capture::{Capture, Plan, REVISION, Row, prompt_digest};
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicUsize, Ordering};
static NEXT: AtomicUsize = AtomicUsize::new(0);
fn fixture() -> (Plan, Row) {
    let hash = prompt_digest(&[11, 22, 33]);
    let path = std::env::temp_dir().join(format!(
        "laguna-capture-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    (
        Plan {
            output: path,
            model_revision: REVISION.into(),
            asserted_source_commit: "a".repeat(40),
            prompt_sha256: BTreeSet::from([hash.clone()]),
            generated_positions: BTreeSet::from([1]),
            max_records: 1,
        },
        Row {
            slot: 7,
            seq_len: 4,
            prompt_len: 3,
            prompt_sha256: hash,
            prefix_lookup_skip: false,
        },
    )
}
#[test]
fn skips_unknown_membership_positions_and_exhausted_cap_without_readback() {
    let (plan, row) = fixture();
    let dir = plan.output.clone();
    let mut cap = Capture::create(plan).unwrap();
    let mut other = row.clone();
    other.slot = 8;
    other.prompt_sha256 = "b".repeat(64);
    assert!(
        !cap.record(
            0,
            &[row.clone(), other],
            Some(&[9, 10]),
            100352,
            false,
            |_| panic!("unknown batch read")
        )
        .unwrap()
    );
    let mut later = row.clone();
    later.seq_len = 5;
    assert!(
        !cap.record(0, &[later], Some(&[9]), 100352, false, |_| panic!(
            "unselected position read"
        ))
        .unwrap()
    );
    assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0);
    assert!(
        cap.record(
            0,
            std::slice::from_ref(&row),
            Some(&[9]),
            100352,
            false,
            |b| {
                b.fill(1);
                Ok(())
            }
        )
        .unwrap()
    );
    assert!(
        !cap.record(1, &[row], Some(&[9]), 100352, false, |_| panic!(
            "exhausted read"
        ))
        .unwrap()
    );
    std::fs::remove_dir_all(dir).unwrap();
}
#[test]
fn freezes_same_step_bits_and_preserves_native_ids_and_router_order() {
    let (mut plan, row) = fixture();
    let dir = plan.output.clone();
    let mut second = row.clone();
    second.slot = 3;
    second.prefix_lookup_skip = true;
    plan.max_records = 2;
    let mut cap = Capture::create(plan).unwrap();
    let ids = vec![1025, 2];
    let original = ids.clone();
    let mut device = vec![0x7b; 100352 * 2 * 2];
    cap.record(12, &[row, second], Some(&ids), 100352, false, |b| {
        b.copy_from_slice(&device);
        Ok(())
    })
    .unwrap();
    device.fill(0); // Simulated next forward cannot alter the saved same-step bytes.
    assert_eq!(
        std::fs::read(dir.join("000-ticket-12.bf16")).unwrap(),
        vec![0x7b; device.len()]
    );
    let receipt: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("000-ticket-12.json")).unwrap()).unwrap();
    assert_eq!(receipt["selected_ids"], serde_json::json!(original));
    assert_eq!(ids, original);
    assert_eq!(receipt["rows"][0]["slot"], 7);
    assert_eq!(receipt["rows"][1]["slot"], 3);
    assert!(!receipt.to_string().contains("prompt_tokens"));
    std::fs::remove_dir_all(dir).unwrap();
}
#[test]
fn refuses_wrong_precision_host_route_duplicate_slots_and_bad_ids_before_copy() {
    let (plan, row) = fixture();
    let dir = plan.output.clone();
    let mut cap = Capture::create(plan).unwrap();
    assert!(
        cap.record(
            0,
            std::slice::from_ref(&row),
            Some(&[0]),
            100352,
            true,
            |_| panic!()
        )
        .is_err()
    );
    assert!(
        cap.record(
            0,
            std::slice::from_ref(&row),
            None,
            100352,
            false,
            |_| panic!()
        )
        .is_err()
    );
    assert!(
        cap.record(
            0,
            std::slice::from_ref(&row),
            Some(&[100352]),
            100352,
            false,
            |_| panic!()
        )
        .is_err()
    );
    assert!(
        cap.record(
            0,
            &[row.clone(), row],
            Some(&[0, 1]),
            100352,
            false,
            |_| panic!()
        )
        .is_err()
    );
    assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0);
    std::fs::remove_dir_all(dir).unwrap();
}
#[test]
fn plan_bounds_and_existing_directory_refuse() {
    let (mut plan, _) = fixture();
    plan.max_records = 65;
    assert!(plan.validate().is_err());
    plan.max_records = 1;
    plan.generated_positions = BTreeSet::from([0]);
    assert!(plan.validate().is_err());
    plan.generated_positions = BTreeSet::from([1]);
    std::fs::create_dir(&plan.output).unwrap();
    let dir = plan.output.clone();
    assert!(Capture::create(plan).is_err());
    std::fs::remove_dir(dir).unwrap();
}
#[test]
fn disabled_capture_does_not_invoke_callback() {
    capture::with_capture(|_| panic!("uninitialized capture must remain inert")).unwrap();
}

#[test]
fn unsupported_execution_modes_refuse_instead_of_falling_back() {
    use capture::validate_mode;
    assert!(validate_mode(true, 1, 4, false, false, true).is_ok());
    for (sync, ranks, batch, spec, codispatch, no_mix) in [
        (false, 1, 4, false, false, true),
        (true, 2, 4, false, false, true),
        (true, 1, 5, false, false, true),
        (true, 1, 4, true, false, true),
        (true, 1, 4, false, true, true),
        (true, 1, 4, false, false, false),
    ] {
        assert!(validate_mode(sync, ranks, batch, spec, codispatch, no_mix).is_err());
    }
}
