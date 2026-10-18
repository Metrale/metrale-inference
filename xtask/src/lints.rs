// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `cargo xtask lints`: every pure crate's library root denies the silent integer
//! casts (BU64: a `usize` is 32 bits on wasm32, so a cast that truncates there must be spelled
//! `try_from`). A per-crate `[lints]` table cannot extend the workspace one, so the crate-level
//! attribute is the carrier. A lint named only in a comment, under `warn`, or under `cfg_attr`
//! does not count. Offenders are keyed by crate in `xtask/lints-allow.txt`.

use crate::allow::{Finding, Findings};
use crate::rust_text;

/// 2026-10-17: The lints every pure library root must `deny` (or `forbid`).
pub const CAST_LINTS: [&str; 4] = [
    "clippy::cast_possible_truncation",
    "clippy::cast_lossless",
    "clippy::cast_possible_wrap",
    "clippy::cast_sign_loss",
];

/// 2026-10-17: Every lint `src` names in a crate-level `#![deny(..)]` or `#![forbid(..)]`.
pub fn denied(src: &str) -> Vec<String> {
    let code = rust_text::strip_comments(src);
    let squeezed: String = code.chars().filter(|c| !c.is_whitespace()).collect();
    let mut out = Vec::new();
    let mut rest = squeezed.as_str();
    while let Some(start) = rest.find("#![") {
        rest = &rest[start + 3..];
        let Some(body) = rest
            .strip_prefix("deny(")
            .or_else(|| rest.strip_prefix("forbid("))
        else {
            continue;
        };
        let Some(end) = body.find(')') else {
            break;
        };
        out.extend(
            body[..end]
                .split(',')
                .filter(|l| !l.is_empty())
                .map(str::to_owned),
        );
        rest = &body[end..];
    }
    out
}

/// 2026-10-17: The `CAST_LINTS` that `src` does not deny.
pub fn missing(src: &str) -> Vec<&'static str> {
    let have = denied(src);
    CAST_LINTS
        .into_iter()
        .filter(|l| !have.iter().any(|h| h == l))
        .collect()
}

/// 2026-10-17: The offending pure crates among `roots` (crate, library root path, source).
pub fn findings<'a>(roots: impl IntoIterator<Item = (&'a str, &'a str, &'a str)>) -> Findings {
    let mut out = Findings::new();
    for (krate, path, src) in roots {
        let lacking = missing(src);
        if lacking.is_empty() {
            continue;
        }
        out.insert(
            krate.to_owned(),
            Finding {
                count: None,
                why: format!(
                    "{path} does not carry #![deny({})] for the pure crate `{krate}`",
                    lacking.join(", ")
                ),
            },
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::missing;

    #[test]
    fn all_four_casts_denied_in_any_split_passes() {
        let one = "#![no_std]\n#![deny(\n    clippy::cast_possible_truncation,\n    clippy::cast_lossless,\n    clippy::cast_possible_wrap,\n    clippy::cast_sign_loss,\n)]\n";
        assert!(missing(one).is_empty());
        let split = "#![forbid(unsafe_code, clippy::cast_lossless)]\n#![deny(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]\n#![deny(clippy::cast_sign_loss)]";
        assert!(missing(split).is_empty());
    }

    #[test]
    fn comments_warn_and_cfg_attr_do_not_count() {
        let src = "// #![deny(clippy::cast_sign_loss)]\n#![warn(clippy::cast_possible_wrap)]\n#![cfg_attr(test, deny(clippy::cast_lossless))]\n#![deny(clippy::cast_possible_truncation)]";
        assert_eq!(
            missing(src),
            vec![
                "clippy::cast_lossless",
                "clippy::cast_possible_wrap",
                "clippy::cast_sign_loss"
            ]
        );
    }
}
