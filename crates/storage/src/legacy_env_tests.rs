// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: Each storage lever projected through [`StorageLinks::from_settings`] against
//! the old reader kept in [`legacy_env::storage_links_from`], over the same values; every
//! difference must be listed in the lever's `changes` table.

use metrale_io_host_mem::MapEnv;
use metrale_settings::sources::{CliLayer, RecipeLayer};

use crate::legacy_env::storage_links_from;
use crate::links::{RailLink, StorageLinks};

/// 2026-10-17: A lever value as a tier used it; `Err` stops the process at startup.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Out {
    B(bool),
    N(u64),
    S(Option<String>),
    Err,
}

type Get = fn(&StorageLinks) -> Out;

/// 2026-10-17: One lever: its variable, the values it is checked over, the variables set
/// beside it (rail 1 exists only with its dual-rail switch on), its projection, and the
/// values whose outcome changed.
struct Row {
    var: &'static str,
    values: &'static [Option<&'static str>],
    base: &'static [(&'static str, &'static str)],
    get: Get,
    changes: Vec<(&'static str, Out)>,
}

const SWITCH: &[Option<&str>] = &[
    None,
    Some("1"),
    Some("true"),
    Some("TRUE"),
    Some("yes"),
    Some("on"),
    Some("0"),
    Some("false"),
    Some(""),
    Some(" 1 "),
    Some("garbage"),
    Some("-1"),
];

const NUM: &[Option<&str>] = &[
    None,
    Some("0"),
    Some("1"),
    Some("3"),
    Some(" 5 "),
    Some(""),
    Some("garbage"),
    Some("-1"),
    Some("100"),
    Some("200"),
    Some("4095"),
    Some("4096"),
    Some("2097152"),
    Some("4294967296"),
];

const TEXT: &[Option<&str>] = &[
    None,
    Some(""),
    Some("mlx5_0"),
    Some(" mlx5_1 "),
    Some("peer-host:7000"),
    Some("1"),
];

fn env_of(base: &[(&str, &str)], var: &str, value: Option<&str>) -> Vec<(String, String)> {
    base.iter()
        .copied()
        .chain(value.map(|v| (var, v)))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

fn new_links(env: &[(String, String)]) -> Option<StorageLinks> {
    let s = metrale_settings::resolve(
        &MapEnv::new(env.iter().cloned()),
        &CliLayer::empty(),
        &RecipeLayer::empty(),
    )
    .ok()?;
    StorageLinks::from_settings(&s).ok()
}

fn old_links(env: &[(String, String)]) -> StorageLinks {
    storage_links_from(|name| env.iter().find(|(k, _)| k == name).map(|(_, v)| v.clone()))
}

/// 2026-10-17: Resolve every row over its values; panic with every row whose changed values
/// differ from its `changes` table.
fn assert_old_readers_hold(rows: Vec<Row>) {
    let mut wrong = Vec::new();
    for row in rows {
        let actual: Vec<(&str, Out)> = row
            .values
            .iter()
            .filter_map(|&v| {
                let env = env_of(row.base, row.var, v);
                let got = new_links(&env).map_or(Out::Err, |l| (row.get)(&l));
                (got != (row.get)(&old_links(&env))).then(|| (v.unwrap_or("<unset>"), got))
            })
            .collect();
        if actual != row.changes {
            wrong.push(format!("{}: actual changes {actual:?}", row.var));
        }
    }
    assert!(
        wrong.is_empty(),
        "changes tables differ:\n{}",
        wrong.join("\n")
    );
}

fn rail0_dev(l: &StorageLinks) -> Out {
    Out::S(Some(l.rail0.dev.clone()))
}
fn rail0_gid(l: &StorageLinks) -> Out {
    Out::N(l.rail0.gid.into())
}
fn kv1(l: &StorageLinks) -> Option<&RailLink> {
    l.kv_rail1.as_ref()
}
fn ex1(l: &StorageLinks) -> Option<&RailLink> {
    l.expert_rail1.as_ref()
}

const KV_DUAL: &[(&str, &str)] = &[("METRALE_KV_DUAL_RAIL", "1")];
const EXPERT_DUAL: &[(&str, &str)] = &[("METRALE_EXPERT_DUAL_RAIL", "1")];

fn switch_changes() -> Vec<(&'static str, Out)> {
    vec![
        ("true", Out::B(true)),
        ("TRUE", Out::B(true)),
        ("yes", Out::B(true)),
        ("on", Out::B(true)),
        ("", Out::Err),
        (" 1 ", Out::B(true)),
        ("garbage", Out::Err),
        ("-1", Out::Err),
    ]
}

#[test]
fn the_switches_take_the_switch_grammar() {
    // 2026-10-17: The old readers were `== "1"`.
    let row = |var, get: Get| Row {
        var,
        values: SWITCH,
        base: &[],
        get,
        changes: switch_changes(),
    };
    assert_old_readers_hold(vec![
        row("METRALE_KV_DUAL_RAIL", |l| Out::B(l.kv_rail1.is_some())),
        row("METRALE_EXPERT_DUAL_RAIL", |l| {
            Out::B(l.expert_rail1.is_some())
        }),
        row("METRALE_KV_ZERO_COPY", |l| Out::B(l.kv_zero_copy)),
        row("METRALE_SSM_STAGING", |l| Out::B(l.ssm_staging)),
        row("METRALE_PLE_SERIAL_FAULT", |l| Out::B(l.ple_serial_fault)),
    ]);
}

#[test]
fn a_gid_that_is_not_an_index_is_an_error() {
    // 2026-10-17: The old reader parsed a `u32`, untrimmed, and fell back to GID 3.
    let changes = || {
        vec![
            (" 5 ", Out::N(5)),
            ("", Out::Err),
            ("garbage", Out::Err),
            ("-1", Out::Err),
            ("4294967296", Out::Err),
        ]
    };
    let row = |var, base, get: Get| Row {
        var,
        values: NUM,
        base,
        get,
        changes: changes(),
    };
    assert_old_readers_hold(vec![
        row("METRALE_EXPERT_RDMA_GID", &[], rail0_gid),
        row("METRALE_KV_RAIL2_GID", KV_DUAL, |l| {
            kv1(l).map_or(Out::Err, |r| Out::N(r.gid.into()))
        }),
        row("METRALE_EXPERT_RAIL2_GID", EXPERT_DUAL, |l| {
            ex1(l).map_or(Out::Err, |r| Out::N(r.gid.into()))
        }),
    ]);
}

#[test]
fn the_pipeline_depths_and_the_chunk_reject_what_the_tiers_ignored() {
    // 2026-10-17: The KV depth parsed a `u32` and the snapshot depth a `usize`, both
    // untrimmed, falling back to 16, then clamped to 1..=128. The chunk parsed a `usize` and
    // fell back to 1 MiB when it did not parse or was below 4096.
    let depth = |var, get: Get, big: Option<Out>| Row {
        var,
        values: NUM,
        base: &[],
        get,
        changes: [
            (" 5 ", Out::N(5)),
            ("", Out::Err),
            ("garbage", Out::Err),
            ("-1", Out::Err),
        ]
        .into_iter()
        .chain(big.map(|o| ("4294967296", o)))
        .collect(),
    };
    let below = |v| (v, Out::Err);
    assert_old_readers_hold(vec![
        depth(
            "METRALE_KV_PIPELINE_DEPTH",
            |l| Out::N(l.kv_pipeline_depth as u64),
            Some(Out::N(128)),
        ),
        // 2026-10-17: A `usize` parse already accepted the value; only the trim and the
        // errors differ.
        depth(
            "METRALE_SSM_PIPELINE_DEPTH",
            |l| Out::N(l.ssm_pipeline_depth as u64),
            None,
        ),
        Row {
            var: "METRALE_SSM_CHUNK_BYTES",
            values: NUM,
            base: &[],
            get: |l| Out::N(l.ssm_chunk_bytes as u64),
            changes: vec![
                below("0"),
                below("1"),
                below("3"),
                below(" 5 "),
                below(""),
                below("garbage"),
                below("-1"),
                below("100"),
                below("200"),
                below("4095"),
            ],
        },
    ]);
}

#[test]
fn the_device_peer_and_paging_values_stay_verbatim() {
    let row = |var, base, get: Get| Row {
        var,
        values: TEXT,
        base,
        get,
        changes: Vec::new(),
    };
    assert_old_readers_hold(vec![
        row("METRALE_EXPERT_RDMA_DEV", &[], rail0_dev),
        row("METRALE_KV_RAIL2_DEV", KV_DUAL, |l| {
            Out::S(kv1(l).map(|r| r.dev.clone()))
        }),
        row("METRALE_EXPERT_RAIL2_DEV", EXPERT_DUAL, |l| {
            Out::S(ex1(l).map(|r| r.dev.clone()))
        }),
        row("METRALE_EXPERT_PEER", &[], |l| {
            Out::S(l.expert_peer.clone())
        }),
        row("METRALE_KV_PAGING", &[], |l| {
            Out::S(l.kv_paging.selected.clone())
        }),
        row("METRALE_KV_PAGING_ARENA_GB", &[], |l| {
            Out::S(l.kv_paging.arena_gb.clone())
        }),
        row("METRALE_KV_PAGING_NS", &[], |l| {
            Out::S(l.kv_paging.ns.clone())
        }),
        row("METRALE_KV_PAGING_SALT", &[], |l| {
            Out::S(l.kv_paging.salt.clone())
        }),
    ]);
}

#[test]
fn the_declared_links_are_the_old_built_in_values_with_rail_zero_first() {
    let none: Vec<(String, String)> = Vec::new();
    let declared = new_links(&none).expect("the declared links resolve");
    assert_eq!(declared, old_links(&none));
    assert_eq!(declared.kv_rails(), vec![declared.rail0.clone()]);
    let both = env_of(EXPERT_DUAL, "METRALE_KV_DUAL_RAIL", Some("1"));
    let dual = new_links(&both).expect("dual rails resolve");
    for rails in [dual.kv_rails(), dual.expert_rails()] {
        assert_eq!(rails.len(), 2);
        assert_eq!(rails[0], dual.rail0);
    }
    assert_eq!(dual.kv_rails()[1], *kv1(&dual).expect("kv rail 1"));
    assert_eq!(dual.expert_rails()[1], *ex1(&dual).expect("expert rail 1"));
}
