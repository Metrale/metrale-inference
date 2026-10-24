// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: A published schema-2 body stays readable across `metrale_circuit::Group::path`:
//! the path is derived from the group's rule (`collective::derive_paths`), never stored, so the
//! body's bytes are the ones written before the field existed.

mod common;

use metrale_blueprint::types::Blueprint;
use metrale_circuit::CollectivePath;

/// 2026-10-23: The fixture blueprint's body as the build before `Group::path` encoded it
/// (megakernel 45686fd25, `postcard::to_allocvec(&common::blueprint())`).
const V2: &[u8] = include_bytes!("data/v2_body_before_path.postcard");

/// 2026-10-23: The pre-change body decodes to the fixture, re-encodes to its own bytes, and
/// states schema 2; a group with a path set encodes to the same bytes (the field writes nothing).
#[test]
fn a_v2_body_written_before_the_path_field_decodes_unchanged() {
    let bp: Blueprint = postcard::from_bytes(V2).unwrap();
    assert_eq!(bp, common::blueprint());
    assert_eq!(bp.schema, 2);
    assert_eq!(metrale_blueprint::BODY_SCHEMA, 2);
    assert_eq!(postcard::to_allocvec(&bp).unwrap(), V2);
    let mut with = common::blueprint();
    for g in &mut with.plans[0].groups {
        g.path = Some(CollectivePath::OneShot);
    }
    assert_eq!(postcard::to_allocvec(&with).unwrap(), V2);
    let back: Blueprint = postcard::from_bytes(V2).unwrap();
    assert!(back.plans[0].groups.iter().all(|g| g.path.is_none()));
}
