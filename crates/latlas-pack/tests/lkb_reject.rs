// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: A whole LKB file its reader refuses is a fatal finding (KgLkbFileRejected), not a
//! row-level KgBadRow: megakernel once added a FUSIONS.toml field an older reader rejected, and
//! every rule silently went missing behind one non-fatal finding.

use latlas_core::SourceSet;
use latlas_core::bundle::Builder;
use metrale_latlas_pack::lkb_derive::{LkbFacts, LkbSources, derive};

const LKB: &str = r#"
repo = "r"
hardware = "gb10"
families = "F.toml"
fusions = "R.toml"
handwritten = "H.toml"
measurements = "M.toml"
instances = "I.toml"
circuits_dir = "c"
plans_dir = "c/plans"
headroom_dir = "c/plans/headroom"
critical_pairs = "c/CRITICAL_PAIRS.toml"
inventory = "kp.py"
remote = "origin"
branch = "megakernel"
"#;

fn sealed(fusions: &str, pairs: &str) -> latlas_core::Bundle {
    let cfg = LkbSources::parse(&toml::from_str(LKB).unwrap()).unwrap();
    let mut src = SourceSet::default();
    for (p, t) in [
        ("r/F.toml", ""),
        ("r/R.toml", fusions),
        ("r/H.toml", ""),
        ("r/M.toml", ""),
        ("r/I.toml", ""),
        ("r/c/CRITICAL_PAIRS.toml", pairs),
    ] {
        src.files.insert(p.into(), t.into());
    }
    let facts = LkbFacts {
        head: "h".into(),
        inventory: Ok(r#"{"kernels":[],"problems":[]}"#.into()),
        file_last: Default::default(),
        commit_date: Default::default(),
    };
    let mut b = Builder::default();
    derive(&mut b, &cfg, &src, &facts).unwrap();
    b.seal(Vec::new())
}

fn rejected_paths(fusions: &str) -> Vec<String> {
    rejected_in(&sealed(fusions, ""))
}

fn rejected_in(bundle: &latlas_core::Bundle) -> Vec<String> {
    bundle
        .fatal()
        .iter()
        .filter(|f| f.code == "LatlasLkbFileRejected")
        .map(|f| f.at.clone())
        .collect()
}

#[test]
fn a_fusions_file_its_reader_refuses_is_fatal() {
    for text in ["= not toml", "schema = 2"] {
        assert!(
            rejected_paths(text).contains(&"r/R.toml".to_string()),
            "{text}"
        );
    }
}

#[test]
fn a_fusions_file_its_reader_accepts_is_not_rejected() {
    assert!(!rejected_paths("schema = 1\nrule = []").contains(&"r/R.toml".to_string()));
}

const RULES: &str = r#"schema = 1

[[rule]]
id = "embed_row_copy"
pattern = [{ op = "embed" }]
kernels = []
repeat = "once"
emitter = "embed_copy"
rows = [1, 128]
modes = ["decode"]
numerics = "reference"
priority = 10
cite = "c"

[[rule]]
id = "input_norm_residual"
pattern = [{ op = "rms_norm", local = "input_norm" }]
kernels = []
repeat = "once"
emitter = "rms_norm_residual"
rows = [1, 128]
modes = ["decode"]
numerics = "reference"
priority = 10
cite = "c"
"#;

const PAIRS: &str = r#"[[pair]]
rules = ["input_norm_residual", "embed_row_copy"]
why = "the fused rule covers the union"
"#;

#[test]
fn each_allowlisted_pair_is_a_priority_critical_pair_overlapping_both_rules() {
    use latlas_core::NodeId;
    use metrale_latlas_pack::rels;
    let b = sealed(RULES, PAIRS);
    let cp =
        NodeId::parse::<metrale_latlas_pack::Metrale>("cpair:embed_row_copy~input_norm_residual")
            .unwrap();
    let n = b.node(&cp).expect("the pair, ids sorted");
    assert_eq!(n.p["resolved_by"], "priority");
    assert_eq!(n.p["allowlisted"], true);
    assert_eq!(n.src, "r/c/CRITICAL_PAIRS.toml:2");
    let to: Vec<String> = b
        .out(&cp, rels::Overlaps)
        .map(|e| e.to.to_string())
        .collect();
    assert_eq!(to, ["rule:embed_row_copy", "rule:input_norm_residual"]);
    assert!(!rejected_in(&b).contains(&"r/c/CRITICAL_PAIRS.toml".to_string()));
    assert!(
        !rejected_in(&b).contains(&"r/R.toml".to_string()),
        "{:?}",
        b.fatal()
    );
}

#[test]
fn a_pair_naming_a_rule_no_fusions_file_defines_is_dangling() {
    use latlas_core::NodeId;
    use metrale_latlas_pack::rels;
    let b = sealed(RULES, &PAIRS.replace("input_norm_residual", "gone"));
    let cp = NodeId::parse::<metrale_latlas_pack::Metrale>("cpair:embed_row_copy~gone").unwrap();
    let to: Vec<String> = b
        .out(&cp, rels::Overlaps)
        .map(|e| e.to.to_string())
        .collect();
    assert_eq!(to, ["rule:embed_row_copy", "unresolved:rule:gone"]);
}

#[test]
fn a_critical_pairs_file_its_reader_refuses_is_fatal() {
    let b = sealed(
        RULES,
        "[[pair]]\nrules = [\"embed_row_copy\", \"input_norm_residual\"]\n",
    );
    assert!(rejected_in(&b).contains(&"r/c/CRITICAL_PAIRS.toml".to_string()));
    assert!(
        b.node(
            &latlas_core::NodeId::parse::<metrale_latlas_pack::Metrale>(
                "cpair:embed_row_copy~input_norm_residual"
            )
            .unwrap()
        )
        .is_none()
    );
}
