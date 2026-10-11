// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: A spec's named files (routing profile, calibration, value statistics) are read
//! relative to the spec's directory, and a missing one is refused by its path.
//!
//! Owner: server CLI.
//! Invariants: none beyond the types.

use super::*;

const PROFILE: &str =
    r#"{"schema":1,"source":"toy/model","experts":8,"top_k":2,"layers":[[1,2,3,4,5,6,7,8]]}"#;
const STATS: &str = r#"{"schema":1,"source":"toy/model","classes":{"c|BF16":{"kind":"halves","counts":[[16256,3]]}}}"#;

fn spec_text(calibration: &str) -> String {
    format!(
        "schema = 1\nseed = 1\n[layers]\nper_signature = 1\n[experts]\nkeep = \"all\"\n[vocab]\nkeep = \"all\"\n\
         [mtp]\nkeep = true\n[vision]\nkeep = true\n[capacity]\nkv = \"free\"\n[routing]\nmode = \"histogram\"\n\
         histogram = \"p.json\"\ncalibration = \"{calibration}\"\n[values]\nmode = \"stats\"\nstats = \"s.json\"\n\
         [speculative]\naccept = \"natural\"\n"
    )
}

#[test]
fn named_files_are_read_beside_the_spec() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("specs");
    std::fs::create_dir(&dir).unwrap();
    let cal = RoutingCalibration::to_text(&[(0usize, 0.75f32)].into());
    std::fs::write(dir.join("p.json"), PROFILE).unwrap();
    std::fs::write(dir.join("c.json"), &cal).unwrap();
    std::fs::write(dir.join("s.json"), STATS).unwrap();
    std::fs::write(dir.join("m.toml"), spec_text("c.json")).unwrap();
    let f = read_spec(&dir.join("m.toml")).unwrap();
    assert_eq!(
        f.profile.unwrap().layers,
        vec![vec![1, 2, 3, 4, 5, 6, 7, 8]]
    );
    assert_eq!(
        f.calibration.unwrap(),
        RoutingCalibration::parse(&cal).unwrap()
    );
    assert_eq!(f.stats.unwrap(), ValueStats::parse(STATS).unwrap());

    std::fs::write(dir.join("n.toml"), spec_text("none")).unwrap();
    assert!(
        read_spec(&dir.join("n.toml"))
            .unwrap()
            .calibration
            .is_none()
    );
}

#[test]
fn a_missing_named_file_is_refused_by_its_path() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("p.json"), PROFILE).unwrap();
    std::fs::write(tmp.path().join("m.toml"), spec_text("none")).unwrap();
    let e = format!("{:#}", read_spec(&tmp.path().join("m.toml")).unwrap_err());
    assert!(
        e.contains("value statistics") && e.contains("s.json"),
        "{e}"
    );
    std::fs::write(tmp.path().join("s.json"), STATS).unwrap();
    std::fs::write(tmp.path().join("m.toml"), spec_text("gone.json")).unwrap();
    let e = format!("{:#}", read_spec(&tmp.path().join("m.toml")).unwrap_err());
    assert!(
        e.contains("routing calibration") && e.contains("gone.json"),
        "{e}"
    );
}
