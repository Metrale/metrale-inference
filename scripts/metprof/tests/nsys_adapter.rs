// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: the nsys export adapter against small real sqlite files made
//! by the scripts in `fixtures/` (see each `.sql`).

mod common;

use std::path::PathBuf;

use common::assert_err;
use metprof::adapters::nsys::{NsysSource, NsysTrace};
use metprof::core::capture::{
    device_copies, device_kernels, devices, graph_node_traced, ordered_calls,
};
use metprof::ports::{Trace, TraceSource};

fn fixture(name: &str) -> String {
    format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))
}

/// A fresh, empty directory unique to this test process and `tag`.
fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("metprof-nsys-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).expect("scratch dir");
    d
}

fn small() -> NsysTrace {
    NsysTrace::open(&fixture("nsys_small.sqlite")).expect("fixture opens")
}

#[test]
fn kernels_are_device_filtered_named_and_ordered() {
    let rows = small().kernel_rows().unwrap();
    assert_eq!(devices(&rows), vec![0, 1]);
    let ks = device_kernels("c", rows, 0).unwrap();
    let names: Vec<&str> = ks.iter().map(|k| k.name.as_str()).collect();
    assert_eq!(names, ["gemm_a", "gemm_b"]);
    let b = &ks[1];
    assert_eq!(
        (b.start, b.end, b.grid, b.graph_node, b.correlation),
        (50_000, 60_000, (4, 2, 1), Some(9), 77)
    );
    assert_eq!(b.stream, 3);
    assert_eq!(ks[0].graph_node, None);
}

#[test]
fn absent_device_is_an_error_naming_the_devices_present() {
    assert_err(
        device_kernels("c", small().kernel_rows().unwrap(), 3),
        r"device 3 .*\[0, 1\]",
    );
}

#[test]
fn copies_and_api_calls() {
    let t = small();
    let kinds = |d| {
        device_copies(t.copy_rows().unwrap(), d)
            .into_iter()
            .map(|c| c.kind)
            .collect::<Vec<_>>()
    };
    assert_eq!(kinds(0), ["HtoD", "DtoH"]);
    assert_eq!(kinds(1), ["DtoD"]);
    let calls = ordered_calls(t.call_rows().unwrap());
    assert_eq!(
        (calls[0].name.as_str(), calls[0].start, calls[0].correlation),
        ("cudaStreamSynchronize_v3020", 61_000, 5)
    );
}

#[test]
fn graph_node_trace_detection() {
    assert!(graph_node_traced(&small().kernel_rows().unwrap()));
    let rows = NsysTrace::open(&fixture("nsys_nonode.sqlite"))
        .unwrap()
        .kernel_rows()
        .unwrap();
    assert!(!graph_node_traced(&rows));
    assert!(rows.iter().all(|r| r.kernel.graph_node.is_none()));
    assert_eq!(rows.len(), 3);
}

#[test]
fn missing_path_is_an_error_and_creates_nothing() {
    let d = scratch("missing");
    let p = d.join("absent.sqlite");
    let r = NsysSource.open(p.to_str().unwrap()).map(|_| ());
    assert_err(r, "not found");
    assert!(!p.exists());
    assert_eq!(std::fs::read_dir(&d).unwrap().count(), 0);
}

#[test]
fn not_a_database() {
    let d = scratch("junk");
    let p = d.join("junk.sqlite");
    std::fs::write(&p, "this is not sqlite ".repeat(100)).unwrap();
    assert_err(
        NsysTrace::open(p.to_str().unwrap()).map(|_| ()),
        "not an sqlite database",
    );
}

#[test]
fn export_without_kernel_table() {
    assert_err(
        NsysTrace::open(&fixture("nsys_nokernel.sqlite")).map(|_| ()),
        "CUPTI_ACTIVITY_KIND_KERNEL",
    );
}

#[test]
fn capture_is_only_read() {
    let path = fixture("nsys_small.sqlite");
    let before = std::fs::read(&path).unwrap();
    let t = small();
    t.kernel_rows().unwrap();
    t.copy_rows().unwrap();
    t.call_rows().unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), before);
    for side in ["-wal", "-shm", "-journal"] {
        assert!(
            !PathBuf::from(format!("{path}{side}")).exists(),
            "{side} created"
        );
    }
}

#[test]
fn a_name_id_absent_from_stringids_is_an_error_not_a_dropped_row() {
    let t = NsysTrace::open(&fixture("nsys_orphan.sqlite")).unwrap();
    assert_err(t.kernel_rows(), "string id 99 is missing from StringIds");
}
