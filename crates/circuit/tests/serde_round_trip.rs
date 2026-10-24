// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The circuit's serde derives (the blueprint body encodes the circuit and its plans
//! with them) read back what they write: every instance's circuit, every plan arm it lists
//! fused under every named kernel, and checkpoint-resolved circuits. A `DimExpr` or enum that
//! serialized lossily would fail here (the `KvPair` spelling is checked in `kv.rs`).
//!
//! Owner: metrale-circuit tests.
//! Invariants: the inputs are the checked-in instances and checkpoint fixtures.

mod checkpoint_fixtures;
mod common;

use metrale_circuit::{AvailableKernels, Circuit, FusionPlan, fuse};

fn round_trip<T: serde::Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug>(
    what: &str,
    v: &T,
) {
    let text = serde_json::to_string(v).unwrap();
    let back: T = serde_json::from_str(&text).unwrap_or_else(|e| panic!("{what}: {e}"));
    assert!(back == *v, "{what} does not read back what it wrote");
    assert_eq!(
        serde_json::to_string(&back).unwrap(),
        text,
        "{what}: not canonical"
    );
}

#[test]
fn every_instance_circuit_and_plan_reads_back_what_it_wrote() {
    let mut plans = 0;
    for inst in common::instances() {
        let loaded = common::load(&inst);
        round_trip::<Circuit>(&inst.recipe, &loaded.circuit);
        let avail = AvailableKernels::all_named_by(&loaded.rules);
        for (mode, rows) in &inst.plans {
            for &r in rows {
                let Ok(p) = fuse(
                    &loaded.circuit,
                    &loaded.rules,
                    &avail,
                    &inst.policy,
                    *mode,
                    r,
                ) else {
                    continue;
                };
                round_trip::<FusionPlan>(&format!("{} {} n{r}", inst.recipe, mode.name()), &p);
                plans += 1;
            }
        }
    }
    assert!(plans > 10, "only {plans} plans fused");
}

#[test]
fn checkpoint_resolved_circuits_read_back() {
    for name in ["nvidia--GLM-5.3-Flash-NVFP4", "unsloth--Qwen3.8-27B-NVFP4"] {
        round_trip::<Circuit>(name, &checkpoint_fixtures::ok(name).circuit);
    }
}

/// 2026-10-22: A negative: a dimension expression that does not parse is refused on read, so a
/// decoded `DimExpr` always agrees with its text.
#[test]
fn a_malformed_dimension_text_is_refused() {
    let err = serde_json::from_str::<metrale_circuit::dims::DimExpr>("\"n*\"");
    assert!(err.is_err());
    let ok: metrale_circuit::dims::DimExpr = serde_json::from_str("\"n * 2\"").unwrap();
    assert_eq!(ok.text(), "n*2");
}
