// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: `cargo xtask kv-codec-ban` (xPEAR, FACE, TIDE): codec decisions live in the
//! `metrale-kv-codec` registry and its one ABI dispatch, never in the engine. Every workspace
//! `.rs` file outside [`EXEMPT`] is scanned, and each of these counts once:
//! - the identifier `KvCacheDtype` (the legacy enum, read as code only);
//! - a `KvCodecId::<Variant>` path (an associated constant such as `KvCodecId::ALL` is not one);
//! - a `match` whose top-level arm patterns name `KvCodecId::` or `AbiId::`;
//! - a string literal that is a KV kernel name: one token (letters, digits, `_`, `{`, `}`, `:`)
//!   containing `paged_decode`, `reshape_and_cache` or `prefill_paged`, or starting `wht_`. This
//!   reads literal text, so prose in a log message (it has spaces) never counts.
//!
//! Offenders are keyed by path with their count in `xtask/kv-codec-allow.txt`, which may only
//! shrink. Files no workspace member owns are not scanned, as in `vendor-ban`.

use crate::allow::{Finding, Findings};
use crate::rust_text;

/// 2026-10-18: Path prefixes the scan skips: the registry crate, the one ABI dispatch, and this
/// gate's own crate (its rules and fixtures hold the forbidden forms as data).
pub const EXEMPT: [&str; 3] = [
    "crates/kv-codec/",
    "crates/model-layers/src/kv_abi/dispatch.rs",
    "xtask/",
];

/// 2026-10-18: Fragments that make a one-token literal a KV kernel name.
const KERNEL_FRAGMENTS: [&str; 3] = ["paged_decode", "reshape_and_cache", "prefill_paged"];

/// 2026-10-18: Whether the scan reads `path`.
pub fn in_scope(path: &str) -> bool {
    path.ends_with(".rs") && !EXEMPT.iter().any(|e| path.starts_with(e))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tok<'a> {
    Ident(&'a str),
    PathSep,
    FatArrow,
    Open(char),
    Close(char),
    Comma,
    Other,
}

/// 2026-10-18: The tokens of comment- and literal-blanked code, each with its 1-based line.
fn tokens(code: &str) -> Vec<(usize, Tok<'_>)> {
    let mut out = Vec::new();
    let mut line = 1_usize;
    let bytes = code.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let c = char::from(bytes[i]);
        let next = bytes.get(i + 1).copied().map(char::from);
        if c == '\n' {
            line = line.saturating_add(1);
            i += 1;
        } else if c.is_ascii_whitespace() || !c.is_ascii() {
            i += 1;
        } else if c.is_ascii_alphabetic() || c == '_' {
            let start = i;
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            out.push((line, Tok::Ident(&code[start..i])));
        } else if c.is_ascii_digit() {
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            out.push((line, Tok::Other));
        } else {
            let (tok, len) = match (c, next) {
                (':', Some(':')) => (Tok::PathSep, 2),
                ('=', Some('>')) => (Tok::FatArrow, 2),
                ('(' | '[' | '{', _) => (Tok::Open(c), 1),
                (')' | ']' | '}', _) => (Tok::Close(c), 1),
                (',', _) => (Tok::Comma, 1),
                _ => (Tok::Other, 1),
            };
            out.push((line, tok));
            i += len;
        }
    }
    out
}

/// 2026-10-18: Whether `ident` is a CamelCase variant name (not a SCREAMING_CASE constant).
fn is_variant(ident: &str) -> bool {
    ident.starts_with(|c: char| c.is_ascii_uppercase())
        && ident.chars().any(|c| c.is_ascii_lowercase())
}

/// 2026-10-18: Whether the tokens at `i` spell `KvCodecId::` or `AbiId::`.
fn names_codec_type(toks: &[(usize, Tok<'_>)], i: usize) -> bool {
    matches!(toks.get(i), Some((_, Tok::Ident("KvCodecId" | "AbiId"))))
        && matches!(toks.get(i + 1), Some((_, Tok::PathSep)))
}

/// 2026-10-18: The index just past the group opened at `open` (its matching close).
fn skip_group(toks: &[(usize, Tok<'_>)], open: usize) -> usize {
    let mut depth = 0_usize;
    let mut i = open;
    while let Some((_, t)) = toks.get(i) {
        match t {
            Tok::Open(_) => depth += 1,
            Tok::Close(_) => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return i + 1;
                }
            }
            Tok::Ident(_) | Tok::PathSep | Tok::FatArrow | Tok::Comma | Tok::Other => {}
        }
        i += 1;
    }
    i
}

/// 2026-10-18: Whether the `match` at `at` has a top-level arm pattern naming a codec type.
fn match_on_codec(toks: &[(usize, Tok<'_>)], at: usize) -> bool {
    let mut i = at + 1;
    while let Some((_, t)) = toks.get(i) {
        match t {
            Tok::Open('{') => break,
            Tok::Open(_) => i = skip_group(toks, i),
            Tok::Close(_) => return false,
            Tok::Ident(_) | Tok::PathSep | Tok::FatArrow | Tok::Comma | Tok::Other => i += 1,
        }
    }
    let end = skip_group(toks, i).saturating_sub(1);
    i += 1;
    while i < end {
        let mut named = false;
        while i < end && !matches!(toks[i].1, Tok::FatArrow) {
            named |= names_codec_type(toks, i);
            i = match toks[i].1 {
                Tok::Open(_) => skip_group_naming(toks, i, &mut named),
                Tok::Ident(_)
                | Tok::PathSep
                | Tok::FatArrow
                | Tok::Comma
                | Tok::Close(_)
                | Tok::Other => i + 1,
            };
        }
        if named {
            return true;
        }
        i += 1;
        if matches!(toks.get(i), Some((_, Tok::Open('{')))) {
            i = skip_group(toks, i);
            if matches!(toks.get(i), Some((_, Tok::Comma))) {
                i += 1;
            }
            continue;
        }
        while i < end && !matches!(toks[i].1, Tok::Comma) {
            i = match toks[i].1 {
                Tok::Open(_) => skip_group(toks, i),
                Tok::Ident(_)
                | Tok::PathSep
                | Tok::FatArrow
                | Tok::Comma
                | Tok::Close(_)
                | Tok::Other => i + 1,
            };
        }
        i += 1;
    }
    false
}

/// 2026-10-18: [`skip_group`] for a pattern group, noting a codec type named inside it.
fn skip_group_naming(toks: &[(usize, Tok<'_>)], open: usize, named: &mut bool) -> usize {
    let end = skip_group(toks, open);
    *named |= (open..end).any(|j| names_codec_type(toks, j));
    end
}

/// 2026-10-18: Whether the literal text `lit` is a KV kernel name.
fn is_kernel_literal(lit: &str) -> bool {
    let one_token = !lit.is_empty()
        && lit
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '{' | '}' | ':'));
    one_token && (lit.starts_with("wht_") || KERNEL_FRAGMENTS.iter().any(|f| lit.contains(f)))
}

/// 2026-10-18: Each string literal of `src` with its 1-based line: the text between a pair of
/// quotes that survive `code_only`, read from `strip_comments` at the same character offsets.
fn literals(src: &str) -> Vec<(usize, String)> {
    let code: Vec<char> = rust_text::code_only(src).chars().collect();
    let text: Vec<char> = rust_text::strip_comments(src).chars().collect();
    let mut out = Vec::new();
    let mut line = 1_usize;
    let mut i = 0;
    while i < code.len() {
        match code[i] {
            '\n' => line = line.saturating_add(1),
            '"' => {
                let open = i;
                i += 1;
                while i < code.len() && code[i] != '"' {
                    i += 1;
                }
                let lit: String = text.get(open + 1..i).unwrap_or_default().iter().collect();
                out.push((line, lit.clone()));
                line = line.saturating_add(lit.matches('\n').count());
            }
            _ => {}
        }
        i += 1;
    }
    out
}

/// 2026-10-18: Each counted form in `src` with its 1-based line, in line order.
pub fn hits(src: &str) -> Vec<(usize, String)> {
    let code = rust_text::code_only(src);
    let toks = tokens(&code);
    let mut out = Vec::new();
    for (i, (line, tok)) in toks.iter().enumerate() {
        match tok {
            Tok::Ident("KvCacheDtype") => out.push((*line, "KvCacheDtype".to_owned())),
            Tok::Ident("KvCodecId") => {
                if let (Some((_, Tok::PathSep)), Some((_, Tok::Ident(v)))) =
                    (toks.get(i + 1), toks.get(i + 2))
                    && is_variant(v)
                {
                    out.push((*line, format!("KvCodecId::{v}")));
                }
            }
            Tok::Ident("match") if match_on_codec(&toks, i) => {
                out.push((*line, "match on a codec type".to_owned()));
            }
            Tok::Ident(_)
            | Tok::PathSep
            | Tok::FatArrow
            | Tok::Open(_)
            | Tok::Close(_)
            | Tok::Comma
            | Tok::Other => {}
        }
    }
    for (line, lit) in literals(src) {
        if is_kernel_literal(&lit) {
            out.push((line, format!("\"{lit}\"")));
        }
    }
    out.sort_by_key(|(line, _)| *line);
    out
}

/// 2026-10-18: The offending files among `files` (path, source).
pub fn findings<'a>(files: impl IntoIterator<Item = (&'a str, &'a str)>) -> Findings {
    let mut out = Findings::new();
    for (path, src) in files {
        let found = hits(src);
        let Some((line, form)) = found.first() else {
            continue;
        };
        let count = u64::try_from(found.len()).unwrap_or(u64::MAX);
        out.insert(
            path.to_owned(),
            Finding {
                count: Some(count),
                why: format!(
                    "{count} KV codec decision(s) outside the registry (first: line {line} `{form}`); ask `metrale-kv-codec` (spec, plan, kernel key) instead"
                ),
            },
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{hits, in_scope};

    fn forms(src: &str) -> Vec<String> {
        hits(src).into_iter().map(|(_, f)| f).collect()
    }

    #[test]
    fn each_counted_form_counts_once_with_its_line() {
        let src = "fn f(d: KvCacheDtype) {\n    let k = KvCodecId::Turbo4;\n    match k {\n        KvCodecId::Bf16 => g(\"paged_decode_bf16\"),\n        _ => g(\"wht_bf16_fwd\"),\n    }\n}\n";
        assert_eq!(
            hits(src),
            vec![
                (1, "KvCacheDtype".to_owned()),
                (2, "KvCodecId::Turbo4".to_owned()),
                (3, "match on a codec type".to_owned()),
                (4, "KvCodecId::Bf16".to_owned()),
                (4, "\"paged_decode_bf16\"".to_owned()),
                (5, "\"wht_bf16_fwd\"".to_owned()),
            ]
        );
    }

    #[test]
    fn prose_comments_constants_and_arm_bodies_do_not_count() {
        let src = "// KvCacheDtype::Fp8 in a comment\n/* match x { KvCodecId::Bf16 => 0 } */\nlet all = KvCodecId::ALL;\nlet s = \"paged_decode failed for this request\";\nlet t = \"kv dtype: {}\";\nmatch abi { Abi::Dense => KvCodecId::ALL, _ => [] }\nmatch x { 0 => { let y = AbiId::Dense; } _ => {} }\n";
        assert_eq!(forms(src), Vec::<String>::new());
    }

    #[test]
    fn a_codec_pattern_in_any_arm_or_nested_group_marks_the_match() {
        for src in [
            "match x { 1 => a, (KvCodecId::Fp8, _) => b, _ => c }",
            "match (k, v) { (_, _) if z => { q() } AbiId::Trellis | AbiId::Latent => r, }",
            "match f(KvCodecId::ALL) { Some(AbiId::Dense) => 1, _ => 0 }",
        ] {
            assert!(
                forms(src).contains(&"match on a codec type".to_owned()),
                "{src}"
            );
        }
    }

    #[test]
    fn templated_kernel_names_count_and_the_exemptions_hold() {
        assert_eq!(
            forms(
                "let n = format!(\"paged_decode_{d}_512\"); let m = \"attn_prefill_paged_64\"; let r = r#\"reshape_and_cache_fp8\"#;"
            ),
            vec![
                "\"paged_decode_{d}_512\"".to_owned(),
                "\"attn_prefill_paged_64\"".to_owned(),
                "\"reshape_and_cache_fp8\"".to_owned(),
            ]
        );
        assert!(in_scope("crates/server/src/main.rs"));
        assert!(!in_scope("crates/kv-codec/src/rows.rs"));
        assert!(!in_scope("crates/model-layers/src/kv_abi/dispatch.rs"));
        assert!(in_scope("crates/model-layers/src/kv_abi/write.rs"));
        assert!(!in_scope("xtask/src/kv_codec_ban.rs"));
        assert!(!in_scope("crates/server/Cargo.toml"));
    }
}
