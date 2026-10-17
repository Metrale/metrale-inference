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
