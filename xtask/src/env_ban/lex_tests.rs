// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The lexer on each read shape and each place a read must not be seen. Every
//! negative sits next to a real read, so a lexer that blanks too much fails as well.

use super::{Read, reads};

fn calls(src: &str) -> Vec<(usize, String)> {
    reads(src)
        .into_iter()
        .map(|Read { line, call }| (line, call))
        .collect()
}

fn one(line: usize, call: &str) -> Vec<(usize, String)> {
    vec![(line, call.to_owned())]
}

#[test]
fn qualified_and_spaced_paths_are_reads() {
    let src = "let a = std::env::var(\"A\");\nlet b = env :: var_os(\"B\");\nfor _ in ::std::env::vars() {}\nlet d = env::\n    vars_os();\n";
    assert_eq!(
        calls(src),
        vec![
            (1, "env::var".to_owned()),
            (2, "env::var_os".to_owned()),
            (3, "env::vars".to_owned()),
            (5, "env::vars_os".to_owned()),
        ]
    );
}

#[test]
fn levers_reads_are_reads() {
    let src = "x(metrale_config::levers::var(\"A\"));\ny(levers::var_os(\"B\"));\nuse crate::levers::{all, var};\n";
    assert_eq!(
        calls(src),
        vec![
            (1, "levers::var".to_owned()),
            (2, "levers::var_os".to_owned()),
            (3, "levers::var".to_owned()),
        ]
    );
}

#[test]
fn brace_lists_count_direct_items_only() {
    assert_eq!(calls("use std::env::{self, var};"), one(1, "env::var"));
    assert_eq!(
        calls("use std::env::{var_os as get};"),
        one(1, "env::var_os")
    );
    assert_eq!(
        calls("use std::env::{\n    args,\n    vars,\n};"),
        one(3, "env::vars")
    );
    assert!(calls("use std::env::{consts::var, VarError, variable};").is_empty());
}

#[test]
fn other_receivers_and_lookalikes_are_not_reads() {
    let src = "Env::var(e, \"A\");\nself.env.var(\"A\");\nmy_env::var(\"A\");\nenv!(\"CARGO\");\noption_env!(\"X\");\nenv::variable();\nenv::args();\n";
    assert!(calls(src).is_empty(), "{:?}", calls(src));
}

#[test]
fn comments_are_blanked_including_nested_blocks() {
    let src = "// std::env::var(\"A\")\n/// env::var\n/* env::var /* nested */ env::var */ std::env::var(\"B\");\n";
    assert_eq!(calls(src), one(3, "env::var"));
}

#[test]
fn string_literals_are_blanked_in_every_form() {
    let src = concat!(
        "let a = \"std::env::var(\\\"A\\\")\";\n",
        "let b = r#\"env::var(\"B\")\"#;\n",
        "let c = br\"env::var\";\n",
        "let d = b\"env::var\";\n",
        "let e = r##\"a \"# env::var\"##; std::env::var(\"E\");\n",
    );
    assert_eq!(calls(src), one(5, "env::var"));
}

#[test]
fn char_literals_do_not_open_strings_and_lifetimes_do_not_open_chars() {
    let src = "let q = '\"'; std::env::var(\"A\");\nlet e = '\\''; std::env::var(\"B\");\nfn f<'a>(x: &'a str) -> &'a str { std::env::var(\"C\"); x }\n";
    assert_eq!(
        calls(src),
        vec![
            (1, "env::var".to_owned()),
            (2, "env::var".to_owned()),
            (3, "env::var".to_owned()),
        ]
    );
}

#[test]
fn raw_identifiers_are_code() {
    assert_eq!(
        calls("let r#type = std::env::var(\"A\");"),
        one(1, "env::var")
    );
}

#[test]
fn renamed_modules_resolve_to_their_reads() {
    assert_eq!(
        calls("use std::env as e;\nfn f() { e::var(\"A\"); e::args(); }\n"),
        one(2, "env::var")
    );
    assert_eq!(
        calls("use std::{env as e, fs};\nlet _ = e::vars();\n"),
        one(2, "env::vars")
    );
    assert_eq!(
        calls("use std::env::{self as e};\nlet _ = e::var_os(\"A\");\n"),
        one(2, "env::var_os")
    );
    assert_eq!(
        calls("use std::env as a;\nuse a as b;\nlet _ = b::var(\"A\");\n"),
        one(3, "env::var")
    );
    assert_eq!(
        calls("use metrale_config::levers as l;\nl::var(\"A\");\n"),
        one(2, "levers::var")
    );
}

#[test]
fn renamed_functions_are_reads_at_the_import_and_at_each_call() {
    assert_eq!(
        calls("use std::env::var as get;\nlet _ = get(\"A\");\n"),
        vec![(1, "env::var".to_owned()), (2, "env::var".to_owned())]
    );
    assert_eq!(calls("use std::env::{var as _};\n"), one(1, "env::var"));
}

#[test]
fn importing_or_renaming_a_module_alone_is_not_a_read() {
    let src = "use std::env as e;\nuse std::env;\nuse std::env::{self, consts};\nlet _ = e::args();\nlet _ = env::current_dir();\n";
    assert!(calls(src).is_empty(), "{:?}", calls(src));
}

#[test]
fn gpu_sys_env_items_are_reads_in_every_form() {
    assert_eq!(
        calls("let _ = metrale_gpu_sys::env::first_set(&[\"A\"]);\n"),
        one(1, "metrale_gpu_sys::env::first_set")
    );
    assert_eq!(
        calls("use metrale_gpu_sys::env::{first_set};\nfirst_set(&[\"A\"]);\n"),
        vec![
            (1, "metrale_gpu_sys::env::first_set".to_owned()),
            (2, "metrale_gpu_sys::env::first_set".to_owned()),
        ]
    );
    assert_eq!(
        calls("use metrale_gpu_sys::env as keys;\nkeys::flag(\"A\");\n"),
        vec![
            (1, "metrale_gpu_sys::env".to_owned()),
            (2, "metrale_gpu_sys::env::flag".to_owned()),
        ]
    );
    assert_eq!(
        calls("use ::metrale_gpu_sys::{env::*, Device};\n"),
        one(1, "metrale_gpu_sys::env::*")
    );
    assert!(calls("use metrale_gpu_sys::device::Env;\nmetrale_gpu_sys::envelope();\n").is_empty());
}

#[test]
fn glob_imports_of_a_banned_module_are_reads() {
    assert_eq!(calls("use std::env::*;\n"), one(1, "env::*"));
    assert_eq!(
        calls("use crate::{levers::*, other};\n"),
        one(1, "levers::*")
    );
    assert!(calls("use std::env::consts::*;\nuse std::*;\n").is_empty());
}

#[test]
fn attributes_in_groups_and_raw_use_do_not_derail_the_import() {
    assert_eq!(
        calls("use std::env::{\n    #[cfg(unix)] var,\n    args,\n};\n"),
        one(2, "env::var")
    );
    assert_eq!(
        calls("let r#use = 1; let _ = std::env::var(\"A\");\n"),
        one(1, "env::var")
    );
}

#[test]
fn a_rename_onto_a_banned_name_still_matches_as_written() {
    assert_eq!(
        calls("use my::config as env;\nenv::var(\"A\");\n"),
        one(2, "env::var")
    );
}

#[test]
fn a_glob_of_a_local_env_module_is_a_re_export_unless_env_is_std() {
    assert!(calls("mod env;\npub use env::*;\nuse crate::env::*;\n").is_empty());
    assert_eq!(calls("use std::env;\nuse env::*;\n"), one(2, "env::*"));
}
