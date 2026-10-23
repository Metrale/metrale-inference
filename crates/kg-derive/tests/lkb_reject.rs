// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: A whole LKB file its reader refuses is a fatal finding (KgLkbFileRejected), not a
//! row-level KgBadRow: megakernel once added a FUSIONS.toml field an older reader rejected, and
//! every rule silently went missing behind one non-fatal finding.

use metrale_kg_derive::SourceSet;
use metrale_kg_derive::bundle::Builder;
use metrale_kg_derive::lkb_derive::{LkbFacts, LkbSources, derive};

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
inventory = "kp.py"
remote = "origin"
branch = "megakernel"
"#;

fn rejected_paths(fusions: &str) -> Vec<String> {
    let cfg = LkbSources::parse(&toml::from_str(LKB).unwrap()).unwrap();
    let mut src = SourceSet::default();
    for (p, t) in [
        ("r/F.toml", ""),
        ("r/R.toml", fusions),
        ("r/H.toml", ""),
        ("r/M.toml", ""),
        ("r/I.toml", ""),
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
    let bundle = b.seal(Vec::new());
    bundle
        .fatal()
        .iter()
        .filter(|f| f.code == "KgLkbFileRejected")
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
