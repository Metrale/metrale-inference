// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: Rendering: forms fill their `{}` placeholders verbatim, `{{`/`}}` escapes stay
//! literal, equality and hashing see the code only, and `Error` implements
//! `std::error::Error`.

use std::collections::HashSet;

use metrale_error::{ErrArg, Error, ErrorCode, error};

#[test]
fn form_fills_placeholders() {
    let e = error!(ErrorCode::NotImplemented, "memcpy_2d_async", "SCALE");
    assert_eq!(
        e.to_string(),
        "memcpy_2d_async is not yet implemented on SCALE"
    );
}

#[test]
fn argument_braces_are_verbatim_not_rescanned() {
    let e = error!(ErrorCode::Generic, "a {} b {{x}}");
    assert_eq!(e.to_string(), "a {} b {{x}}");
}

#[test]
fn primitive_arguments_render() {
    assert_eq!(
        error!(ErrorCode::InvalidArgument, "pitch", 7u32).to_string(),
        "invalid argument pitch: 7",
    );
    assert_eq!(
        error!(ErrorCode::InvalidArgument, "pitch", -3i64).to_string(),
        "invalid argument pitch: -3",
    );
    assert_eq!(
        error!(ErrorCode::InvalidArgument, "pitch", true).to_string(),
        "invalid argument pitch: true",
    );
    assert_eq!(
        error!(ErrorCode::InvalidArgument, "pitch", String::from("wide")).to_string(),
        "invalid argument pitch: wide",
    );
}

#[test]
fn code_argument_renders_name_and_number() {
    let e = error!(ErrorCode::InvalidArgument, "code", ErrorCode::Io);
    assert_eq!(e.to_string(), "invalid argument code: Io (0x0001)");
}

#[test]
fn nested_error_renders_its_message() {
    let inner = error!(ErrorCode::Unavailable, "gpu");
    let outer = error!(ErrorCode::Generic, inner);
    assert_eq!(outer.to_string(), "gpu is unavailable");
}

#[test]
fn origin_is_the_call_site_file_and_line() {
    let (line, e) = (line!(), error!(ErrorCode::Unavailable, "x"));
    assert_eq!(e.origin(), format!("{}:{}", file!(), line));
}

#[test]
fn args_expose_the_form_arguments_in_order() {
    assert_eq!(error!(ErrorCode::NotImplemented, "a", "b").args().len(), 2);
    assert_eq!(error!(ErrorCode::Generic, "x").args().len(), 1);
    assert!(matches!(
        error!(ErrorCode::Generic, "x").args(),
        [ErrArg::Str("x")]
    ));
}

#[test]
fn debug_shows_code_origin_and_message() {
    let text = format!("{:?}", error!(ErrorCode::NotImplemented, "a", "b"));
    assert!(
        text.starts_with("Error { code: 0x0003 NotImplemented, origin: "),
        "unexpected Debug: {text}",
    );
    assert!(
        text.contains("message: \"a is not yet implemented on b\""),
        "{text}"
    );
}

#[test]
fn equality_and_hashing_see_the_code_only() {
    let a = error!(ErrorCode::NotImplemented, "a", "z");
    let b = error!(ErrorCode::NotImplemented, "b", "c");
    assert_eq!(a, b);
    let mut set = HashSet::new();
    set.insert(a);
    set.insert(b);
    assert_eq!(set.len(), 1);
    assert_ne!(error!(ErrorCode::Generic, "x"), error!(ErrorCode::Io, "y"));
}

#[test]
fn every_form_placeholder_count_matches_a_literal_scan() {
    for &c in ErrorCode::ALL {
        let stripped = c.form().replace("{{", "").replace("}}", "");
        assert_eq!(
            stripped.matches("{}").count(),
            c.placeholder_count(),
            "{}: placeholder count mismatch",
            c.name(),
        );
    }
}

#[test]
fn implements_std_error() {
    fn takes<E: std::error::Error>(_: &E) {}
    takes(&error!(ErrorCode::Generic, "x"));
}

#[test]
fn hidden_constructor_renders_missing_placeholders_literally() {
    let e = Error::__from_parts(ErrorCode::NotImplemented, "o", None);
    assert_eq!(e.to_string(), "{} is not yet implemented on {}");
}
