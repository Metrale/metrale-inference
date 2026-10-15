// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: column lists from the `CREATE TABLE` text stored in
//! `sqlite_schema`. Only what a reader needs: column names in declared order,
//! which column (if any) aliases the rowid, and which columns carry a DEFAULT.
//! WITHOUT ROWID tables and generated columns are refused: their storage is
//! different and this reader does not implement it.

use crate::core::error::Result;
use crate::fail;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Column {
    pub name: String,
    pub rowid_alias: bool,
    pub has_default: bool,
    /// REAL affinity: SQLite stores whole-valued reals as integers and converts them back on read.
    pub real_affinity: bool,
}

/// Remove `-- ...` and `/* ... */` comments outside quoted text.
fn strip_comments(sql: &str) -> String {
    let b: Vec<char> = sql.chars().collect();
    let (mut out, mut i, mut quote) = (String::new(), 0, None::<char>);
    while i < b.len() {
        let c = b[i];
        if let Some(q) = quote {
            out.push(c);
            if c == q {
                quote = None;
            }
            i += 1;
            continue;
        }
        match (c, b.get(i + 1)) {
            ('-', Some('-')) => {
                while i < b.len() && b[i] != '\n' {
                    i += 1;
                }
            }
            ('/', Some('*')) => {
                i += 2;
                while i < b.len() && !(b[i] == '*' && b.get(i + 1) == Some(&'/')) {
                    i += 1;
                }
                i += 2;
                out.push(' ');
            }
            _ => {
                if matches!(c, '"' | '\'' | '`') {
                    quote = Some(c);
                } else if c == '[' {
                    quote = Some(']');
                }
                out.push(c);
                i += 1;
            }
        }
    }
    out
}

/// Split at commas that are outside parentheses and quotes.
fn split_top(body: &str) -> Vec<String> {
    let (mut parts, mut cur, mut depth, mut quote) =
        (Vec::new(), String::new(), 0i32, None::<char>);
    for c in body.chars() {
        match quote {
            Some(q) => {
                if c == q {
                    quote = None;
                }
            }
            None => match c {
                '"' | '\'' | '`' => quote = Some(c),
                '[' => quote = Some(']'),
                '(' => depth += 1,
                ')' => depth -= 1,
                ',' if depth == 0 => {
                    parts.push(std::mem::take(&mut cur));
                    continue;
                }
                _ => {}
            },
        }
        cur.push(c);
    }
    parts.push(cur);
    parts
        .into_iter()
        .map(|p| p.trim().to_string())
        .filter(|p| !p.is_empty())
        .collect()
}

/// The first token of a definition (unquoted) and the rest.
fn first_token(def: &str) -> (String, &str) {
    let mut chars = def.char_indices();
    let close = match chars.next() {
        Some((_, '"')) => Some('"'),
        Some((_, '`')) => Some('`'),
        Some((_, '[')) => Some(']'),
        Some((_, '\'')) => Some('\''),
        _ => None,
    };
    if let Some(q) = close
        && let Some(end) = def[1..].find(q)
    {
        return (def[1..1 + end].to_string(), &def[2 + end..]);
    }
    let end = def
        .find(|c: char| c.is_whitespace() || c == '(')
        .unwrap_or(def.len());
    (def[..end].to_string(), &def[end..])
}

fn words(s: &str) -> Vec<String> {
    s.split(|c: char| c.is_whitespace() || c == '(' || c == ')' || c == ',')
        .filter(|w| !w.is_empty())
        .map(str::to_ascii_uppercase)
        .collect()
}

const CONSTRAINTS: [&str; 5] = ["CONSTRAINT", "PRIMARY", "UNIQUE", "CHECK", "FOREIGN"];

/// Columns of a `CREATE TABLE` statement in declared order.
pub fn table_columns(sql: &str) -> Result<Vec<Column>> {
    let sql = strip_comments(sql);
    let (Some(open), Some(close)) = (sql.find('('), sql.rfind(')')) else {
        fail!("CREATE TABLE without a column list: {sql}");
    };
    if open >= close {
        fail!("CREATE TABLE with a malformed column list: {sql}");
    }
    if words(&sql[close + 1..]).iter().any(|w| w == "WITHOUT") {
        fail!("WITHOUT ROWID tables are not supported by this reader");
    }
    let mut cols: Vec<Column> = Vec::new();
    let mut int_typed: Vec<bool> = Vec::new();
    let mut table_pk: Option<Vec<String>> = None;
    for def in split_top(&sql[open + 1..close]) {
        let (name, rest) = first_token(&def);
        let w = words(rest);
        if CONSTRAINTS.contains(&name.to_ascii_uppercase().as_str()) {
            let all = words(&def);
            if all.iter().any(|x| x == "PRIMARY") {
                let inside = def
                    .find('(')
                    .zip(def.find(')'))
                    .filter(|(a, b)| a < b)
                    .map(|(a, b)| &def[a + 1..b])
                    .unwrap_or("");
                table_pk = Some(split_top(inside).iter().map(|c| first_token(c).0).collect());
            }
            continue;
        }
        if w.iter().any(|x| x == "GENERATED")
            || w.windows(2).any(|p| p[0] != "DEFAULT" && p[1] == "AS")
        {
            fail!("generated column {name} is not supported by this reader");
        }
        let type_words: Vec<&String> = w.iter().take_while(|x| !is_constraint_word(x)).collect();
        let is_int = type_words.len() == 1 && type_words[0] == "INTEGER";
        let pk = w.windows(2).any(|p| p[0] == "PRIMARY" && p[1] == "KEY");
        let desc = w.iter().any(|x| x == "DESC");
        cols.push(Column {
            name,
            rowid_alias: is_int && pk && !desc,
            has_default: w.iter().any(|x| x == "DEFAULT"),
            real_affinity: real_affinity(&type_words),
        });
        int_typed.push(is_int);
    }
    if let Some(pk) = table_pk
        && let [only] = pk.as_slice()
        && let Some(i) = cols.iter().position(|c| c.name.eq_ignore_ascii_case(only))
    {
        cols[i].rowid_alias = int_typed[i];
    }
    if cols.is_empty() {
        fail!("CREATE TABLE without columns: {sql}");
    }
    Ok(cols)
}

/// SQLite's affinity rules (datatype3.html §3.1), in their order: only REAL matters to a reader.
fn real_affinity(type_words: &[&String]) -> bool {
    let t: String = type_words
        .iter()
        .map(|w| w.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    let has = |s: &str| t.contains(s);
    !has("INT")
        && !has("CHAR")
        && !has("CLOB")
        && !has("TEXT")
        && !has("BLOB")
        && !t.is_empty()
        && (has("REAL") || has("FLOA") || has("DOUB"))
}

fn is_constraint_word(w: &str) -> bool {
    matches!(
        w,
        "CONSTRAINT"
            | "PRIMARY"
            | "NOT"
            | "NULL"
            | "UNIQUE"
            | "CHECK"
            | "DEFAULT"
            | "COLLATE"
            | "REFERENCES"
            | "GENERATED"
            | "AS"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nsys_style_ddl_with_comments_and_a_rowid_alias() {
        let sql = "CREATE TABLE StringIds (\n  -- Consolidation of repetitive string values.\n  id INTEGER NOT NULL PRIMARY KEY, -- ID reference value.\n  value TEXT NOT NULL /* the, string */\n)";
        let cols = table_columns(sql).unwrap();
        assert_eq!(
            cols.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(),
            ["id", "value"]
        );
        assert!(cols[0].rowid_alias && !cols[1].rowid_alias);
    }

    #[test]
    fn quoted_names_table_constraints_and_defaults() {
        let sql = r#"CREATE TABLE "t x" ("a b" INT, [c] TEXT DEFAULT 'q,r', `d` REAL, k INTEGER, PRIMARY KEY (k), UNIQUE (a))"#;
        let cols = table_columns(sql).unwrap();
        let names: Vec<&str> = cols.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, ["a b", "c", "d", "k"]);
        assert!(cols[1].has_default && !cols[0].has_default);
        assert!(
            cols[3].rowid_alias,
            "a single-column INTEGER table primary key aliases the rowid"
        );
        assert!(!table_columns("CREATE TABLE t (a INT PRIMARY KEY)").unwrap()[0].rowid_alias);
        assert!(
            !table_columns("CREATE TABLE t (a INTEGER PRIMARY KEY DESC)").unwrap()[0].rowid_alias
        );
    }

    #[test]
    fn real_affinity_follows_sqlite_rule_order() {
        let aff =
            |ty: &str| table_columns(&format!("CREATE TABLE t (a {ty})")).unwrap()[0].real_affinity;
        for ty in ["REAL", "DOUBLE PRECISION", "FLOAT", "real NOT NULL"] {
            assert!(aff(ty), "{ty}");
        }
        // INT wins over REAL (rule 1 first); CHAR/BLOB/empty/NUMERIC are not REAL.
        for ty in [
            "FLOATING POINT",
            "INTEGER",
            "VARCHAR(10)",
            "BLOB",
            "",
            "NUMERIC",
            "DECIMAL(10,5)",
        ] {
            assert!(!aff(ty), "{ty}");
        }
    }

    #[test]
    fn unsupported_storage_is_refused() {
        assert!(table_columns("CREATE TABLE t (a INTEGER PRIMARY KEY) WITHOUT ROWID").is_err());
        assert!(
            table_columns("CREATE TABLE t (a INT, b INT GENERATED ALWAYS AS (a + 1))").is_err()
        );
        assert!(table_columns("CREATE TABLE t (a INT, b INT AS (a + 1) STORED)").is_err());
        assert!(table_columns("CREATE TABLE t").is_err());
    }
}
