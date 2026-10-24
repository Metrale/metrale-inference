// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: Parsed manifests as values, and where their dependency tables sit. Nothing here
//! reads a file: `shell.rs` loads the text and hands it over, so the rules judge plain values.

use toml::{Table, Value};

/// 2026-10-17: One manifest: its path relative to the workspace root, which every message
/// names, and its parsed contents.
pub struct Manifest {
    pub file: String,
    pub table: Table,
}

/// 2026-10-17: A workspace: the root manifest and every member's, in `members` order.
pub struct Workspace {
    pub root: Manifest,
    pub members: Vec<Manifest>,
}

/// 2026-10-17: One dependency table of a manifest: its header as written in TOML (for example
/// `target.'cfg(unix)'.dev-dependencies`) and its entries.
pub struct DepTable<'a> {
    pub header: String,
    pub entries: &'a Table,
}

/// 2026-10-17: Every dependency table name cargo reads, the underscore spellings included, so a
/// table cannot escape the gate by its spelling.
const KINDS: [&str; 5] = [
    "dependencies",
    "dev-dependencies",
    "build-dependencies",
    "dev_dependencies",
    "build_dependencies",
];

/// 2026-10-17: Parse `text` as the manifest at `file`.
pub fn parse(file: &str, text: &str) -> Result<Manifest, String> {
    let table =
        toml::from_str::<Table>(text).map_err(|e| format!("{file}: not valid TOML: {e}"))?;
    Ok(Manifest {
        file: file.to_owned(),
        table,
    })
}

fn workspace_table(root: &Manifest) -> Result<&Table, String> {
    root.table
        .get("workspace")
        .and_then(Value::as_table)
        .ok_or_else(|| format!("{}: no [workspace] table", root.file))
}

/// 2026-10-17: The `[workspace] members` directories, as written. A glob is refused: the gate
/// reads an explicit list rather than guessing which directories a pattern matches.
pub fn member_dirs(root: &Manifest) -> Result<Vec<String>, String> {
    let members = workspace_table(root)?
        .get("members")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{}: [workspace] has no `members` array", root.file))?;
    members
        .iter()
        .map(|m| {
            let dir = m.as_str().ok_or_else(|| {
                format!(
                    "{}: a `members` entry is a {}, not a string",
                    root.file,
                    m.type_str()
                )
            })?;
            if dir.contains(['*', '?', '[']) {
                return Err(format!(
                    "{}: member `{dir}` is a glob; `cargo xtask deps` reads an explicit list",
                    root.file
                ));
            }
            Ok(dir.trim_end_matches('/').to_owned())
        })
        .collect()
}

/// 2026-10-17: The root's `[workspace.dependencies]`: the one place a dependency is declared.
pub fn workspace_deps(root: &Manifest) -> Result<&Table, String> {
    workspace_table(root)?
        .get("dependencies")
        .ok_or_else(|| format!("{}: no [workspace.dependencies] table", root.file))?
        .as_table()
        .ok_or_else(|| format!("{}: [workspace.dependencies] is not a table", root.file))
}

/// 2026-10-17: Every dependency table of `m`: the top-level ones, then each `[target.<cfg>]`'s.
pub fn dep_tables(m: &Manifest) -> Result<Vec<DepTable<'_>>, String> {
    let mut out = Vec::new();
    push_kinds(&m.file, &m.table, "", &mut out)?;
    if let Some(target) = m.table.get("target") {
        let target = target
            .as_table()
            .ok_or_else(|| format!("{}: `target` is not a table", m.file))?;
        for (cfg, body) in target {
            let prefix = format!("target.{}.", quote_key(cfg));
            let body = body
                .as_table()
                .ok_or_else(|| format!("{}: [{prefix}] is not a table", m.file))?;
            push_kinds(&m.file, body, &prefix, &mut out)?;
        }
    }
    Ok(out)
}

fn push_kinds<'a>(
    file: &str,
    scope: &'a Table,
    prefix: &str,
    out: &mut Vec<DepTable<'a>>,
) -> Result<(), String> {
    for kind in KINDS {
        if let Some(value) = scope.get(kind) {
            let entries = value
                .as_table()
                .ok_or_else(|| format!("{file}: [{prefix}{kind}] is not a table"))?;
            out.push(DepTable {
                header: format!("{prefix}{kind}"),
                entries,
            });
        }
    }
    Ok(())
}

/// 2026-10-17: A key as TOML would need it written: bare when it can be, quoted otherwise.
fn quote_key(key: &str) -> String {
    let bare = !key.is_empty()
        && key
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
    if bare {
        key.to_owned()
    } else if key.contains('\'') {
        format!("\"{}\"", key.replace('\\', "\\\\").replace('"', "\\\""))
    } else {
        format!("'{key}'")
    }
}
