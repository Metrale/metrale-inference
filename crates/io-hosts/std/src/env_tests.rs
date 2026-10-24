// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `ProcessEnv` against the real process environment, read-only
//! (`set_var` is unsafe in edition 2024 and races parallel tests).

use metrale_abstract_io::Env;

use super::ProcessEnv;

#[test]
fn absent_name_is_ok_none() {
    let names = ProcessEnv.names();
    let absent = (0u32..)
        .map(|n| format!("IO_HOST_STD_ABSENT_{n}"))
        .find(|candidate| !names.contains(candidate))
        .expect("an unused name exists");
    assert_eq!(ProcessEnv.var(&absent), Ok(None));
}

#[test]
fn names_are_sorted_and_unique() {
    let names = ProcessEnv.names();
    assert!(
        names.windows(2).all(|pair| pair[0] < pair[1]),
        "names must be strictly ascending: {names:?}"
    );
}
