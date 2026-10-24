// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-10-19: The safetensors path admits the default levers and refuses each lever that drives
//! the GGUF expert cache, by name.

use metrale_model_layers::lever_equivalence::defaults;

use super::*;

fn at_defaults() -> Ds41LoadLevers {
    Ds41LoadLevers::from_settings(&defaults())
}

#[test]
fn the_default_levers_are_admitted() {
    refuse_cache_levers(&at_defaults()).unwrap();
}

#[test]
fn each_cache_lever_is_refused_by_name() {
    let cases: [(&str, fn(&mut Ds41LoadLevers)); 7] = [
        ("METRALE_DS41_GRAPH ", |l| l.run.graph_mode = GraphMode::On),
        ("METRALE_DS41_GRAPH_ORACLE", |l| {
            l.run.graph_mode = GraphMode::Oracle
        }),
        ("METRALE_DS41_STEP_GRAPH", |l| l.run.step_graph = true),
        ("METRALE_DS41_PREFETCH_K", |l| l.run.prefetch_k = 1),
        ("METRALE_DS41_READER_POOL", |l| l.reader_pool = true),
        ("METRALE_DS41_ROUTE_TRACE", |l| {
            l.route_trace = Some("t".into())
        }),
        ("METRALE_DS41_PREDICT_TRACE", |l| {
            l.predict_trace = Some("t".into())
        }),
    ];
    for (lever, arm) in cases {
        let mut l = at_defaults();
        arm(&mut l);
        let e = refuse_cache_levers(&l).unwrap_err().to_string();
        assert!(e.contains(lever), "{lever}: {e}");
        // 2026-10-19: The two graph switches share one entry; every other lever is named alone.
        let graph = l.run.graph_mode != GraphMode::Off;
        assert_eq!(
            e.matches("METRALE_DS41_").count(),
            if graph { 2 } else { 1 },
            "{e}"
        );
    }
}

#[test]
fn every_armed_cache_lever_is_named_at_once() {
    let mut l = at_defaults();
    l.run.step_graph = true;
    l.reader_pool = true;
    let e = refuse_cache_levers(&l).unwrap_err().to_string();
    assert!(
        e.contains("METRALE_DS41_STEP_GRAPH, METRALE_DS41_READER_POOL"),
        "{e}"
    );
}

#[test]
fn the_q2k_engram_lever_is_not_a_cache_lever() {
    let mut l = at_defaults();
    l.engram_q2k = true;
    refuse_cache_levers(&l).unwrap();
}
