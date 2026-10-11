// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: A family's sources are the union of its points' kernel files, sorted; the digest
//! is the SPEC's (path bytes then file bytes, in order); a changed file makes exactly its
//! families stale.

use std::collections::BTreeMap;

use sha2::{Digest, Sha256};

use super::*;
use crate::envelope::schedules::SCHEMA;

const FAMILIES: &str = r#"
schema = 1
hardware = "hw"
[roofline]
dram_gbps = 249.0
bf16_tflops = 123.7
fp8_tflops = 243.6
nvfp4_tflops = 490.8
context_tokens = 4096

[[family]]
id = "two_points"
description = "x"
compute = "memory"
kernels = ["argmax::argmax_bf16", "argmax::argmax_bf16_batch"]
rows = [1, 64]
pipeline.argmax = { in = ["bf16"], compare = "bf16", out = ["i32"] }
op = [{ op = "argmax" }]
[[family.param]]
name = "form"
kind = "policy"
from = { argmax = "param:form" }
absent = "a"
[[family.point]]
values = { form = "a" }
how = "instantiation"
files = ["kernels/hw/common/x.cu", "crates/engine/src/host.rs"]
[[family.point]]
values = { form = "b" }
how = "copy"
files = ["kernels/hw/common/w.cu", "kernels/hw/common/x.cu"]

[[family]]
id = "other"
description = "y"
compute = "memory"
kernels = ["argmax2::argmax_bf16"]
rows = [1, 64]
pipeline.argmax = { in = ["bf16"], compare = "bf16", out = ["i32"] }
op = [{ op = "argmax" }]
[[family.point]]
values = {}
how = "instantiation"
files = ["kernels/hw/common/y.cu"]

[[family]]
id = "host_only"
description = "z"
compute = "memory"
kernels = ["argmax3::argmax_bf16"]
rows = [1, 64]
pipeline.argmax = { in = ["bf16"], compare = "bf16", out = ["i32"] }
op = [{ op = "argmax" }]
[[family.point]]
values = {}
how = "instantiation"
files = ["crates/engine/src/host.rs"]
"#;

struct Mem(BTreeMap<String, String>);

impl Repo for Mem {
    fn read(&self, rel: &str) -> Result<String, String> {
        self.0
            .get(rel)
            .cloned()
            .ok_or_else(|| format!("{rel}: missing"))
    }
    fn list(&self, rel: &str) -> Result<Vec<String>, String> {
        Ok(self
            .0
            .keys()
            .filter(|k| k.starts_with(rel))
            .cloned()
            .collect())
    }
}

fn repo() -> Mem {
    Mem([
        ("kernels/hw/common/KERNEL_FAMILIES.toml", FAMILIES),
        ("kernels/hw/common/x.cu", "x body"),
        ("kernels/hw/common/w.cu", "w body"),
        ("kernels/hw/common/y.cu", "y body"),
        ("crates/engine/src/host.rs", "host"),
    ]
    .iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect())
}

#[test]
fn sources_are_the_sorted_kernel_files_and_the_spec_digest() {
    let got = sources_of(&repo(), "hw", &["two_points"]).unwrap();
    let src = &got["two_points"];
    assert_eq!(
        src.files,
        vec!["kernels/hw/common/w.cu", "kernels/hw/common/x.cu"]
    );
    let want: String = Sha256::digest(b"kernels/hw/common/w.cuw bodykernels/hw/common/x.cux body")
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(src.sha256, want);
}

#[test]
fn a_changed_source_makes_exactly_its_families_stale() {
    let fams = ["two_points", "other"];
    let then = sources_of(&repo(), "hw", &fams).unwrap();
    let file = Schedules {
        schema: SCHEMA,
        hardware: "hw".into(),
        generated_by: "t".into(),
        sources: then.clone(),
        schedule: Vec::new(),
    };
    assert!(stale(&then, &file).is_empty());
    let mut r = repo();
    r.0.insert("kernels/hw/common/w.cu".into(), "w body, edited".into());
    // A host file outside kernels/ is not a kernel source: editing it changes nothing.
    r.0.insert("crates/engine/src/host.rs".into(), "host, edited".into());
    let now = sources_of(&r, "hw", &fams).unwrap();
    assert_eq!(stale(&now, &file), vec!["two_points".to_string()]);
    let mut gone = now.clone();
    gone.remove("other");
    gone.insert("two_points".into(), then["two_points"].clone());
    assert_eq!(
        stale(&gone, &file),
        vec!["other".to_string()],
        "a vanished family is stale"
    );
}

#[test]
fn unknown_and_host_only_families_are_refused() {
    assert!(sources_of(&repo(), "hw", &["nope"]).is_err());
    assert!(sources_of(&repo(), "hw", &["host_only"]).is_err());
    let mut r = repo();
    r.0.remove("kernels/hw/common/y.cu");
    assert!(matches!(
        sources_of(&r, "hw", &["other"]),
        Err(SchedulesError::Load(_))
    ));
}
