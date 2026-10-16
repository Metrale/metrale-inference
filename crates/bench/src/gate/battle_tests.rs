// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-16: Tests for `battle`. Each refusal and each verdict rule has a control on its
//! other side, and the real tree is loaded so the committed file and targets stay cells.

use std::collections::BTreeMap;

use super::*;
use crate::gate::model_scope::Subject;

const GATES: &[&str] = &[
    "ttft-cold-glm-64k",
    "ttft-warm-glm-64k",
    "high-isl-ttft-cold-glm-64k",
    "concurrency-sweep-glm",
    "decode-floor",
];

const FILE: &str = r#"
schema = 1

[[battles]]
id = "ttft-battle"
name = "TTFT battle"
gate_prefixes = ["ttft-cold-", "ttft-warm-", "high-isl-ttft-cold-"]
metrics = [
  { pattern = "median_ms", better = "lower" },
  { pattern = "median_ms_{n}", better = "lower" },
]

[[battles]]
id = "decode-battle"
name = "Decode battle"
gate_prefixes = ["concurrency-sweep"]
metrics = [{ pattern = "c{n}_aggregate_tok_s", better = "higher" }]
"#;

fn battles() -> Vec<Battle> {
    parse(FILE, GATES).expect("the fixture parses")
}

fn entry(gate: &str, hw: &str, targets: &[(&str, f64)]) -> Entry {
    Entry {
        gate: gate.into(),
        hw: hw.into(),
        model: "glm".into(),
        checkpoint: "org/GLM".into(),
        bounds: BTreeMap::new(),
        targets: targets.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
    }
}

fn cell(better: Better, target: f64, bound: Option<Bound>) -> Cell {
    Cell {
        gate: "g".into(),
        hw: "gb10".into(),
        model: "glm".into(),
        checkpoint: "org/GLM".into(),
        metric: "m".into(),
        better,
        target,
        bound,
    }
}

fn seen(value: f64) -> Observed {
    Observed {
        git_sha: "abc".into(),
        recorded_at: 7,
        metrics: [("m".to_string(), value)].into_iter().collect(),
    }
}

#[test]
fn the_fixture_parses_and_each_gate_lands_in_its_battle() {
    let b = battles();
    assert_eq!(
        battle_of(&b, "ttft-warm-glm-64k").unwrap().id,
        "ttft-battle"
    );
    assert_eq!(
        battle_of(&b, "concurrency-sweep-glm").unwrap().id,
        "decode-battle"
    );
    assert!(battle_of(&b, "decode-floor").is_none());
}

#[test]
fn a_file_of_another_schema_is_refused() {
    let err = parse(&FILE.replace("schema = 1", "schema = 2"), GATES).unwrap_err();
    assert!(err.contains("schema 2"), "{err}");
}

#[test]
fn an_unknown_key_is_refused() {
    let err = parse(
        &FILE.replace("name = \"Decode battle\"", "name = \"D\"\nfloor = 3"),
        GATES,
    )
    .unwrap_err();
    assert!(err.contains("floor"), "{err}");
}

#[test]
fn a_battle_id_that_is_a_gate_id_is_refused() {
    let err = parse(
        &FILE.replace("id = \"decode-battle\"", "id = \"decode-floor\""),
        GATES,
    )
    .unwrap_err();
    assert!(err.contains("is a gate id"), "{err}");
}

#[test]
fn a_gate_two_battles_claim_is_refused() {
    let both = FILE.replace(
        "[\"concurrency-sweep\"]",
        "[\"concurrency-sweep\", \"ttft-cold-\"]",
    );
    let err = parse(&both, GATES).unwrap_err();
    assert!(
        err.contains("ttft-cold-glm-64k belongs to more than one battle"),
        "{err}"
    );
}

#[test]
fn a_battle_no_registered_gate_belongs_to_is_refused() {
    let err = parse(
        &FILE.replace("[\"concurrency-sweep\"]", "[\"concurency-sweep\"]"),
        GATES,
    )
    .unwrap_err();
    assert!(
        err.contains("no registered gate belongs to battle decode-battle"),
        "{err}"
    );
}

#[test]
fn a_pattern_with_a_second_placeholder_is_refused() {
    let err = parse(&FILE.replace("c{n}_aggregate", "c{n}_{n}_aggregate"), GATES).unwrap_err();
    assert!(err.contains("one `{n}`"), "{err}");
}

#[test]
fn a_placeholder_matches_digits_only_and_a_literal_only_itself() {
    let b = battles();
    let ttft = &b[0];
    assert_eq!(ttft.better("median_ms_256"), Some(Better::Lower));
    assert_eq!(ttft.better("median_ms"), Some(Better::Lower));
    for not in [
        "median_ms_",
        "median_ms_2x",
        "p90_ms_256",
        "median_ms_256_x",
        "median_msx",
    ] {
        assert_eq!(ttft.better(not), None, "{not}");
    }
    let decode = &b[1];
    assert_eq!(decode.better("c12_aggregate_tok_s"), Some(Better::Higher));
    assert_eq!(decode.better("peak_aggregate_tok_s"), None);
    assert_eq!(decode.better("c16_gpu_rail_joules_per_token"), None);
}

#[test]
fn every_target_must_be_a_cell_of_some_battle() {
    let b = battles();
    let clean = [entry(
        "ttft-cold-glm-64k",
        "gb10",
        &[("median_ms_256", 488.0)],
    )];
    assert!(problems(&b, &clean).is_empty());
    let orphan = [entry("decode-floor", "gb10", &[("decode_tps", 40.0)])];
    assert!(problems(&b, &orphan)[0].contains("is in no battle"));
    let no_rule = [entry("ttft-cold-glm-64k", "gb10", &[("p90_ms", 1.0)])];
    assert!(problems(&b, &no_rule)[0].contains("matches no metric rule"));
    let zero = [entry("ttft-cold-glm-64k", "gb10", &[("median_ms", 0.0)])];
    assert!(problems(&b, &zero)[0].contains("finite and positive"));
    let none = [entry("decode-floor", "gb10", &[])];
    assert!(
        problems(&b, &none).is_empty(),
        "an entry with no targets is no cell"
    );
}

#[test]
fn cells_are_the_member_targets_in_prompt_length_order() {
    let b = battles();
    let entries = [
        entry(
            "ttft-cold-glm-64k",
            "gb10",
            &[
                ("median_ms_4096", 1705.0),
                ("median_ms_256", 488.0),
                ("median_ms_1024", 679.0),
            ],
        ),
        entry(
            "concurrency-sweep-glm",
            "gb10",
            &[("c1_aggregate_tok_s", 40.11)],
        ),
    ];
    let ttft: Vec<String> = cells(&b[0], &entries)
        .into_iter()
        .map(|c| c.metric)
        .collect();
    assert_eq!(ttft, ["median_ms_256", "median_ms_1024", "median_ms_4096"]);
    let decode = cells(&b[1], &entries);
    assert_eq!(decode.len(), 1);
    assert_eq!(decode[0].better, Better::Higher);
}

#[test]
fn lower_is_better_wins_under_the_target_and_loses_over_it() {
    let c = cell(Better::Lower, 100.0, None);
    let win = judge(&c, Some(&seen(80.0)));
    assert_eq!(win.verdict, Verdict::Win);
    assert!((win.margin_pct.unwrap() - 20.0).abs() < 1e-9);
    let loss = judge(&c, Some(&seen(125.0)));
    assert_eq!(loss.verdict, Verdict::Loss);
    assert!((loss.margin_pct.unwrap() + 25.0).abs() < 1e-9);
    assert_eq!(judge(&c, Some(&seen(100.0))).verdict, Verdict::Tie);
}

#[test]
fn higher_is_better_wins_over_the_target_and_loses_under_it() {
    let c = cell(Better::Higher, 100.0, None);
    let win = judge(&c, Some(&seen(110.0)));
    assert_eq!(win.verdict, Verdict::Win);
    assert!((win.margin_pct.unwrap() - 10.0).abs() < 1e-9);
    assert_eq!(judge(&c, Some(&seen(90.0))).verdict, Verdict::Loss);
}

#[test]
fn no_record_and_a_missing_metric_are_never_numbers() {
    let c = cell(Better::Lower, 100.0, None);
    let none = judge(&c, None);
    assert_eq!(
        (none.verdict, none.ours, none.margin_pct),
        (Verdict::NoRecord, None, None)
    );
    assert_eq!(none.ratchet, Ratchet::NoValue);
    let mut other = seen(1.0);
    other.metrics = [("other".to_string(), 1.0)].into_iter().collect();
    let silent = judge(&c, Some(&other));
    assert_eq!((silent.verdict, silent.ours), (Verdict::NotReported, None));
    assert_eq!(silent.record_sha.as_deref(), Some("abc"));
    let nan = judge(&c, Some(&seen(f64::NAN)));
    assert_eq!(nan.verdict, Verdict::NotReported);
}

#[test]
fn a_cell_can_pass_its_ratchet_and_still_lose() {
    let ceiling = Bound {
        min: None,
        max: Some(11446.9),
        noise: None,
    };
    let c = cell(Better::Lower, 10479.0, Some(ceiling));
    let j = judge(&c, Some(&seen(11103.9)));
    assert_eq!(
        (j.verdict, j.ratchet.clone()),
        (Verdict::Loss, Ratchet::Within)
    );
    let over = judge(&c, Some(&seen(11500.0)));
    assert!(
        matches!(over.ratchet, Ratchet::Breach(_)),
        "{:?}",
        over.ratchet
    );
    let unbounded = judge(&cell(Better::Lower, 10479.0, None), Some(&seen(9000.0)));
    assert_eq!(
        (unbounded.verdict, unbounded.ratchet),
        (Verdict::Win, Ratchet::Unbounded)
    );
}

#[test]
fn all_cells_win_needs_a_cell_and_every_cell_a_win() {
    let b = &battles()[0];
    let c = cell(Better::Lower, 100.0, None);
    assert!(
        !report(b, &[], &|_| None).all_cells_win,
        "no cell is not a win"
    );
    assert!(report(b, &[c.clone(), c.clone()], &|_| Some(seen(50.0))).all_cells_win);
    let mut other = c.clone();
    other.gate = "h".into();
    let one_unmeasured = report(b, &[c.clone(), other], &|x| {
        (x.gate == "g").then(|| seen(50.0))
    });
    assert!(!one_unmeasured.all_cells_win);
    assert!(
        !report(b, &[c], &|_| Some(seen(100.0))).all_cells_win,
        "a tie is not a win"
    );
}

#[test]
fn open_members_are_the_open_gates_of_this_hardware_each_once() {
    let b = battles();
    let entries = [
        entry(
            "ttft-cold-glm-64k",
            "gb10",
            &[("median_ms_256", 1.0), ("median_ms_1024", 1.0)],
        ),
        entry("ttft-warm-glm-64k", "gb10", &[("median_ms_256", 1.0)]),
        entry("high-isl-ttft-cold-glm-64k", "h100", &[("median_ms", 1.0)]),
    ];
    let c = cells(&b[0], &entries);
    let open = open_members(&c, "gb10", &|c| c.gate == "ttft-warm-glm-64k");
    assert_eq!(
        open,
        ["ttft-cold-glm-64k"],
        "once, though two of its cells are open"
    );
    assert_eq!(
        open_members(&c, "gb10", &|_| false).len(),
        2,
        "h100 is another box"
    );
    let one_cell = open_members(&c, "gb10", &|c| c.metric != "median_ms_1024");
    assert_eq!(
        one_cell,
        ["ttft-cold-glm-64k"],
        "one open cell opens its gate"
    );
    assert!(open_members(&c, "gb10", &|_| true).is_empty());
}

#[test]
fn a_battle_is_owed_by_the_model_whose_cell_subject_is_owed() {
    let b = battles();
    let c = cells(
        &b[0],
        &[entry(
            "ttft-cold-glm-64k",
            "gb10",
            &[("median_ms_256", 1.0)],
        )],
    );
    let owed = |checkpoint: &str| Owed {
        gate: "ttft-cold-glm-64k".into(),
        subject: Subject {
            hw: "gb10".into(),
            model: "glm".into(),
            checkpoint: checkpoint.into(),
        },
    };
    let pairs = [(b[0].clone(), c)];
    let got = owed_battles(&pairs, &[owed("org/GLM")]);
    assert_eq!(got["glm"].iter().collect::<Vec<_>>(), ["ttft-battle"]);
    assert!(owed_battles(&pairs, &[owed("org/Other")]).is_empty());
}

/// 2026-10-16: The committed file and every committed `targets` value, as the CLI loads them.
#[test]
fn the_tree_battles_load_and_hold_the_glm_cells() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let loaded = crate::gate::battle_load::load(&root).expect("the tree's battles load");
    let count = |id: &str| {
        loaded
            .iter()
            .find(|(b, _)| b.id == id)
            .map(|(_, c)| {
                c.iter()
                    .filter(|c| c.checkpoint == "nvidia/GLM-5.3-Flash-NVFP4" && c.hw == "gb10")
                    .count()
            })
            .unwrap_or(0)
    };
    assert_eq!(
        count("ttft-battle"),
        24,
        "cold and warm x 4 prompts x 3 contexts"
    );
    assert_eq!(count("decode-battle"), 6, "C = 1, 2, 4, 8, 12, 16");
}
