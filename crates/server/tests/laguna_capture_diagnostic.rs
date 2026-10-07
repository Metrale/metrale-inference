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
            allocation_generation: 41,
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
    second.allocation_generation = 42;
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
    device.fill(0); // 2026-10-07: Simulated next forward cannot alter the saved same-step bytes.
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
    for variant in 0..4 {
        let (plan, row) = fixture();
        let dir = plan.output.clone();
        let mut cap = Capture::create(plan).unwrap();
        let rows = if variant == 3 {
            vec![row.clone(), row]
        } else {
            vec![row]
        };
        let ids = if variant == 2 {
            vec![100352]
        } else {
            vec![0; rows.len()]
        };
        let selected = if variant == 1 {
            None
        } else {
            Some(ids.as_slice())
        };
        assert!(
            cap.record(0, &rows, selected, 100352, variant == 0, |_| panic!())
                .is_err()
        );
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0);
        std::fs::remove_dir_all(dir).unwrap();
    }
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

#[test]
fn reused_slot_and_prompt_are_distinguished_by_allocation_generation() {
    let (mut plan, row) = fixture();
    let dir = plan.output.clone();
    plan.max_records = 2;
    let mut cap = Capture::create(plan).unwrap();
    let mut reused = row.clone();
    reused.allocation_generation += 1;
    for (ticket, member) in [(1, row), (2, reused)] {
        assert!(
            cap.record(ticket, &[member], Some(&[9]), 100352, false, |b| {
                b.fill(0);
                Ok(())
            })
            .unwrap()
        );
    }
    let a: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("000-ticket-1.json")).unwrap()).unwrap();
    let b: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("001-ticket-2.json")).unwrap()).unwrap();
    assert_eq!(a["rows"][0]["slot"], b["rows"][0]["slot"]);
    assert_eq!(a["rows"][0]["prompt_sha256"], b["rows"][0]["prompt_sha256"]);
    assert_ne!(
        a["rows"][0]["allocation_generation"],
        b["rows"][0]["allocation_generation"]
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn absent_or_duplicate_allocation_identity_refuses_before_readback() {
    for variant in 0..3 {
        let (plan, row) = fixture();
        let dir = plan.output.clone();
        let mut cap = Capture::create(plan).unwrap();
        let mut bad = row.clone();
        let rows = match variant {
            0 => {
                bad.allocation_generation = 0;
                vec![bad]
            }
            1 => {
                bad.slot = usize::MAX;
                vec![bad]
            }
            _ => {
                bad.slot = 8;
                vec![row, bad]
            }
        };
        let ids = vec![0; rows.len()];
        assert!(
            cap.record(0, &rows, Some(&ids), 100352, false, |_| panic!())
                .is_err()
        );
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0);
        std::fs::remove_dir_all(dir).unwrap();
    }
}

#[test]
fn read_failure_permanently_refuses_later_readbacks() {
    let (plan, row) = fixture();
    let dir = plan.output.clone();
    let mut cap = Capture::create(plan).unwrap();
    assert!(
        cap.record(
            0,
            std::slice::from_ref(&row),
            Some(&[9]),
            100352,
            false,
            |_| Err(anyhow::anyhow!("injected GPU read failure"))
        )
        .is_err()
    );
    assert!(
        cap.record(1, &[row], Some(&[9]), 100352, false, |_| panic!(
            "retry read after failure"
        ))
        .is_err()
    );
    assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn file_failure_is_sticky_and_never_overwrites_existing_bytes() {
    let (plan, row) = fixture();
    let dir = plan.output.clone();
    let mut cap = Capture::create(plan).unwrap();
    std::fs::write(dir.join("000-ticket-0.bf16"), b"sentinel").unwrap();
    assert!(
        cap.record(
            0,
            std::slice::from_ref(&row),
            Some(&[9]),
            100352,
            false,
            |bytes| {
                bytes.fill(3);
                Ok(())
            }
        )
        .is_err()
    );
    assert!(
        cap.record(1, &[row], Some(&[9]), 100352, false, |_| panic!(
            "retry after I/O failure"
        ))
        .is_err()
    );
    assert_eq!(
        std::fs::read(dir.join("000-ticket-0.bf16")).unwrap(),
        b"sentinel"
    );
    assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 1);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn plan_reader_stops_at_limit_even_for_an_unbounded_source() {
    struct Endless<'a>(&'a std::cell::Cell<usize>);
    impl std::io::Read for Endless<'_> {
        fn read(&mut self, bytes: &mut [u8]) -> std::io::Result<usize> {
            bytes.fill(b' ');
            self.0.set(self.0.get() + bytes.len());
            Ok(bytes.len())
        }
    }
    let count = std::cell::Cell::new(0);
    assert!(capture::read_plan(Endless(&count)).is_err());
    assert_eq!(count.get(), 32769);
    assert_eq!(
        capture::hash_reader(&b"abc"[..]).unwrap(),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
}

#[test]
fn regular_plan_file_parses_but_devices_and_fifos_refuse() {
    let (plan, _) = fixture();
    let dir = plan.output.clone();
    std::fs::create_dir(&dir).unwrap();
    let path = dir.join("plan.json");
    let value = serde_json::json!({"output":dir.join("out"), "model_revision":plan.model_revision,
        "asserted_source_commit":plan.asserted_source_commit, "prompt_sha256":plan.prompt_sha256,
        "generated_positions":plan.generated_positions,"max_records":1});
    std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
    assert!(capture::read_plan_file(&path).is_ok());
    assert!(capture::read_plan_file(&dir).is_err());
    #[cfg(unix)]
    {
        assert!(capture::read_plan_file(std::path::Path::new("/dev/zero")).is_err());
        let fifo = dir.join("pipe");
        assert!(
            std::process::Command::new("mkfifo")
                .arg(&fifo)
                .status()
                .unwrap()
                .success()
        );
        assert!(capture::read_plan_file(&fifo).is_err());
    }
    std::fs::remove_dir_all(dir).unwrap();
}
