// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `MapEnv` lookups, ordering and the duplicate-key guard.

use metrale_abstract_io::Env;

use super::MapEnv;

#[test]
fn set_names_return_their_exact_value() {
    let env = MapEnv::new([("B", " padded "), ("A", "")]);
    assert_eq!(env.var("A"), Ok(Some(String::new())));
    assert_eq!(env.var("B"), Ok(Some(" padded ".to_owned())));
}

#[test]
fn unset_name_is_ok_none() {
    let env = MapEnv::new([("A", "1")]);
    assert_eq!(env.var("a"), Ok(None));
    assert_eq!(env.var("AB"), Ok(None));
}

#[test]
fn names_are_sorted_in_byte_order() {
    let env = MapEnv::new([("b", "1"), ("B", "2"), ("A_", "3"), ("A", "4")]);
    assert_eq!(env.names(), ["A", "A_", "B", "b"]);
}

#[test]
#[should_panic(expected = "duplicate key K")]
fn duplicate_key_panics() {
    let _ = MapEnv::new([("K", "1"), ("K", "2")]);
}
