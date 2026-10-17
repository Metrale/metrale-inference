// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: Rust source as the text scans read it. Comments (line, nested block, doc) are
//! blanked with spaces and newlines kept, so line numbers survive and a word in a comment never
//! counts. `code_only` also blanks the contents of string and char literals, so only code counts.

/// 2026-10-17: `src` with every comment blanked; literals kept.
pub fn strip_comments(src: &str) -> String {
    scrub(src, false)
}

/// 2026-10-17: `src` with every comment and every literal's contents blanked; the quotes stay.
pub fn code_only(src: &str) -> String {
    scrub(src, true)
}

fn blank(c: char) -> char {
    if c == '\n' { '\n' } else { ' ' }
}

fn scrub(src: &str, blank_literals: bool) -> String {
    let chars: Vec<char> = src.chars().collect();
    let mut out = String::with_capacity(src.len());
    let lit = |c: char| if blank_literals { blank(c) } else { c };
    let at = |i: usize| chars.get(i).copied();
    let mut i = 0;
    while let Some(c) = at(i) {
        match (c, at(i + 1)) {
            ('/', Some('/')) => {
                while let Some(c) = at(i).filter(|c| *c != '\n') {
                    out.push(blank(c));
                    i += 1;
                }
            }
            ('/', Some('*')) => {
                let mut depth = 0_u32;
                while let Some(c) = at(i) {
                    let pair = (c, at(i + 1));
                    if pair == ('/', Some('*')) || pair == ('*', Some('/')) {
                        depth = if c == '/' { depth + 1 } else { depth - 1 };
                        out.push_str("  ");
                        i += 2;
                        if depth == 0 {
                            break;
                        }
                    } else {
                        out.push(blank(c));
                        i += 1;
                    }
                }
            }
            ('r', Some('#' | '"')) if raw_string_hashes(&chars, i).is_some() => {
                let hashes = raw_string_hashes(&chars, i).unwrap_or_default();
                let close: Vec<char> = std::iter::once('"')
                    .chain(std::iter::repeat_n('#', hashes))
                    .collect();
                let open_end = i + 2 + hashes;
                out.extend(&chars[i..open_end]);
                i = open_end;
                while i < chars.len() && !chars[i..].starts_with(&close) {
                    out.push(lit(chars[i]));
                    i += 1;
                }
                let end = (i + close.len()).min(chars.len());
                out.extend(&chars[i..end]);
                i = end;
            }
            ('"', _) => {
                out.push('"');
                i += 1;
                while let Some(c) = at(i) {
                    i += 1;
                    if c == '"' {
                        out.push('"');
                        break;
                    }
                    out.push(lit(c));
                    if c == '\\' {
                        if let Some(next) = at(i) {
                            out.push(lit(next));
                            i += 1;
                        }
                    }
                }
            }
            ('\'', Some('\\')) => {
                out.push('\'');
                let escaped_end = (i + 3).min(chars.len());
                out.extend(chars[i + 1..escaped_end].iter().map(|c| lit(*c)));
                i = escaped_end;
                while let Some(c) = at(i) {
                    i += 1;
                    if c == '\'' {
                        out.push('\'');
                        break;
                    }
                    out.push(lit(c));
                }
            }
            ('\'', Some(ch)) if at(i + 2) == Some('\'') => {
                out.push('\'');
                out.push(lit(ch));
                out.push('\'');
                i += 3;
            }
            _ => {
                out.push(c);
                i += 1;
            }
        }
    }
    out
}

/// 2026-10-17: If a raw string literal (`r"…"`, `r#"…"#`) opens at `i`, its number of hashes.
fn raw_string_hashes(chars: &[char], i: usize) -> Option<usize> {
    let ident_before = i
        .checked_sub(1)
        .and_then(|p| chars.get(p))
        .is_some_and(|c| c.is_alphanumeric() || *c == '_');
    if ident_before {
        return None;
    }
    let hashes = chars
        .get(i + 1..)?
        .iter()
        .take_while(|c| **c == '#')
        .count();
    (chars.get(i + 1 + hashes) == Some(&'"')).then_some(hashes)
}

#[cfg(test)]
mod tests {
    use super::{code_only, strip_comments};

    #[test]
    fn comments_go_and_code_and_literals_stay_line_for_line() {
        let src =
            "let a = cuda; // cuda here\n/* outer /* inner cuda */ still */ b\n/// doc cuda\nc";
        let got = strip_comments(src);
        assert_eq!(got.lines().count(), src.lines().count());
        assert!(got.starts_with("let a = cuda;"));
        assert_eq!(got.matches("cuda").count(), 1, "{got}");
        assert!(got.contains(" b\n"));
        assert!(got.ends_with('c'));
    }

    #[test]
    fn comment_markers_inside_literals_are_not_comments() {
        for src in [
            r#"let s = "// cuda"; x"#,
            r##"let s = r#"/* cuda "quoted" */"#; x"##,
            r#"let s = "esc \" // cuda"; x"#,
            "let c = '\"'; let s = \"// cuda\"; x",
            "let c = '\\''; let s = \"// cuda\"; x",
            "fn f<'a>(s: &'a str) { \"// cuda\" } x",
        ] {
            let kept = strip_comments(src);
            assert!(kept.contains("cuda"), "{src} -> {kept}");
            assert!(kept.ends_with('x'), "{src} -> {kept}");
            let code = code_only(src);
            assert!(!code.contains("cuda"), "{src} -> {code}");
            assert!(code.ends_with('x'), "{src} -> {code}");
            assert_eq!(code.chars().count(), src.chars().count(), "{src} -> {code}");
        }
    }
}
