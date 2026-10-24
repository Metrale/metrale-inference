// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: Finds runtime environment reads in Rust source text. Comments, string literals
//! and char literals are blanked first, so a read named in a doc comment or a message is not a
//! read. The rest is split into tokens, and `resolve` decides which paths are reads. Pure: text
//! in, findings out.

/// 2026-10-17: One read: the 1-based line and the call it resolves to, e.g. `env::var`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Read {
    pub line: usize,
    pub call: String,
}

/// 2026-10-17: A token of blanked source: an identifier or keyword, a `::`, or any other
/// character.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Tok {
    Ident(String),
    Sep,
    Punct(char),
}

/// 2026-10-17: A token and the char index it starts at.
pub struct Token {
    pub kind: Tok,
    pub pos: usize,
}
fn is_ident(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

fn blank_range(out: &mut [char], src: &[char], from: usize, to: usize) {
    let to = to.min(src.len());
    for (o, s) in out[from..to].iter_mut().zip(&src[from..to]) {
        if *s != '\n' {
            *o = ' ';
        }
    }
}

/// 2026-10-17: The end (exclusive) of a `"…"` literal whose opening quote is at `open`.
fn string_end(c: &[char], open: usize) -> usize {
    let mut j = open + 1;
    while j < c.len() {
        match c[j] {
            '\\' => j += 2,
            '"' => return j + 1,
            _ => j += 1,
        }
    }
    c.len()
}

/// 2026-10-17: The end (exclusive) of a raw string whose `r` is at `r_at`, or `None` when the
/// `r` does not start one (a raw identifier `r#name`, or a plain identifier).
fn raw_string_end(c: &[char], r_at: usize) -> Option<usize> {
    let mut j = r_at + 1;
    let mut hashes = 0usize;
    while c.get(j) == Some(&'#') {
        hashes += 1;
        j += 1;
    }
    if c.get(j) != Some(&'"') {
        return None;
    }
    j += 1;
    while j < c.len() {
        if c[j] == '"' && (1..=hashes).all(|h| c.get(j + h) == Some(&'#')) {
            return Some(j + 1 + hashes);
        }
        j += 1;
    }
    Some(c.len())
}

/// 2026-10-17: The end (exclusive) of a nested `/* … */` comment opening at `open`.
fn block_comment_end(c: &[char], open: usize) -> usize {
    let (mut j, mut depth) = (open + 2, 1usize);
    while j < c.len() && depth > 0 {
        if c[j] == '/' && c.get(j + 1) == Some(&'*') {
            depth += 1;
            j += 2;
        } else if c[j] == '*' && c.get(j + 1) == Some(&'/') {
            depth -= 1;
            j += 2;
        } else {
            j += 1;
        }
    }
    j
}

/// 2026-10-17: The end (exclusive) of a char literal opening at `open`, or `None` for a
/// lifetime or label (`'a`, `'outer:`).
fn char_end(c: &[char], open: usize) -> Option<usize> {
    if c.get(open + 1) == Some(&'\\') {
        let mut j = open + 3;
        while j < c.len() && c[j] != '\'' && c[j] != '\n' {
            j += 1;
        }
        return Some((j + 1).min(c.len()));
    }
    (c.get(open + 2) == Some(&'\'')).then_some(open + 3)
}

/// 2026-10-17: `src` with every comment, string and char literal replaced by spaces; newlines
/// are kept, so a position in the result has the same line as in `src`.
pub fn blank(src: &str) -> Vec<char> {
    let c: Vec<char> = src.chars().collect();
    let mut out = c.clone();
    let mut i = 0;
    while i < c.len() {
        let boundary = i == 0 || !is_ident(c[i - 1]);
        let end = match c[i] {
            '/' if c.get(i + 1) == Some(&'/') => Some(
                c[i..]
                    .iter()
                    .position(|&x| x == '\n')
                    .map_or(c.len(), |p| i + p),
            ),
            '/' if c.get(i + 1) == Some(&'*') => Some(block_comment_end(&c, i)),
            '"' => Some(string_end(&c, i)),
            '\'' => char_end(&c, i),
            'r' if boundary => raw_string_end(&c, i),
            'b' | 'c' if boundary && c.get(i + 1) == Some(&'r') => raw_string_end(&c, i + 1),
            'b' | 'c' if boundary && c.get(i + 1) == Some(&'"') => Some(string_end(&c, i + 1)),
            _ => None,
        };
        match end {
            Some(end) => {
                blank_range(&mut out, &c, i, end);
                i = end.max(i + 1);
            }
            None => i += 1,
        }
    }
    out
}

/// 2026-10-17: The tokens of blanked source, whitespace dropped.
pub fn tokens(c: &[char]) -> Vec<Token> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < c.len() {
        let (kind, len) = match c[i] {
            x if x.is_whitespace() => {
                i += 1;
                continue;
            }
            x if is_ident(x) => {
                let len = c[i..].iter().take_while(|&&x| is_ident(x)).count();
                (Tok::Ident(c[i..i + len].iter().collect()), len)
            }
            ':' if c.get(i + 1) == Some(&':') => (Tok::Sep, 2),
            x => (Tok::Punct(x), 1),
        };
        out.push(Token { kind, pos: i });
        i += len;
    }
    out
}

/// 2026-10-17: Every read in `src`, in source order.
pub fn reads(src: &str) -> Vec<Read> {
    let c = blank(src);
    let newlines: Vec<usize> = (0..c.len()).filter(|&i| c[i] == '\n').collect();
    super::resolve::reads(&tokens(&c))
        .into_iter()
        .map(|(pos, call)| Read {
            line: 1 + newlines.partition_point(|&n| n < pos),
            call,
        })
        .collect()
}

#[cfg(test)]
#[path = "lex_tests.rs"]
mod tests;
