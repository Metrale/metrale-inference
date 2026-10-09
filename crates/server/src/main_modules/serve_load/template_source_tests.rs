// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: `template_source`: the chat-template source each flag combination
//! selects, and clap refusing `--chat-template` with `--disable-template-overrides`.
//!
//! Owner: server startup (`met serve`).
//! Invariants: none beyond the types.

use clap::Parser;

use super::template_source;
use crate::cli::ServeArgs;
use crate::tokenizer::TemplateSource;

fn parse(extra: &[&str]) -> Result<ServeArgs, clap::Error> {
    ServeArgs::try_parse_from(std::iter::once("serve").chain(extra.iter().copied()))
}

#[test]
fn each_flag_combination_selects_its_source() {
    let args = parse(&[]).expect("parses");
    assert_eq!(template_source(&args), TemplateSource::OverrideDir);
    let args = parse(&["--disable-template-overrides"]).expect("parses");
    assert_eq!(template_source(&args), TemplateSource::Checkpoint);
    let args = parse(&["--chat-template", "/t/served.jinja"]).expect("parses");
    assert_eq!(
        template_source(&args),
        TemplateSource::File(std::path::Path::new("/t/served.jinja"))
    );
}

#[test]
fn a_template_file_and_disabling_overrides_are_refused_together() {
    let err = parse(&[
        "--chat-template",
        "/t/served.jinja",
        "--disable-template-overrides",
    ])
    .expect_err("conflicting template choices");
    assert_eq!(err.kind(), clap::error::ErrorKind::ArgumentConflict);
}
