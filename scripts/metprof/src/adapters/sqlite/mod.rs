// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: a small read-only reader of the SQLite database file format,
//! in pure Rust (no C library). It reads rollback-journal databases only: a
//! WAL-mode header or a `-wal` file next to the database is refused, because
//! committed rows could live in the WAL and a reader that ignored it would
//! silently return stale data. Tables are found by name in `sqlite_schema`;
//! columns are decoded by their declared position. Nothing is ever written.

mod btree;
pub mod record;
mod schema;

use std::path::Path;

use crate::adapters::sqlite::btree::{Pages, be16, be32};
pub use crate::adapters::sqlite::record::Value;
use crate::adapters::sqlite::schema::{Column, table_columns};
use crate::core::error::{Error, Result};
use crate::fail;

const MAGIC: &[u8; 16] = b"SQLite format 3\0";
const HEADER: usize = 100;

/// An opened database image.
pub struct Db {
    label: String,
    data: Vec<u8>,
    page_size: usize,
    usable: usize,
    pages: u32,
}

/// A table found in the schema.
#[derive(Debug, Clone)]
pub struct Table {
    pub name: String,
    root: u32,
    columns: Vec<Column>,
}

impl Table {
    pub fn has_column(&self, name: &str) -> bool {
        self.columns
            .iter()
            .any(|c| c.name.eq_ignore_ascii_case(name))
    }
}

fn header(data: &[u8]) -> Result<(usize, usize, u32)> {
    if data.len() < HEADER || &data[..16] != MAGIC {
        fail!("not an sqlite database: the file does not start with the SQLite header");
    }
    let page_size = match be16(data, 16)? {
        1 => 65536,
        n if n >= 512 && n.is_power_of_two() => usize::from(n),
        n => fail!("not an sqlite database: invalid page size {n}"),
    };
    match (data[18], data[19]) {
        (1, 1) => {}
        (2, _) | (_, 2) => {
            fail!("the database is in WAL mode; this reader only reads rollback-journal databases")
        }
        (w, r) => fail!("not an sqlite database: unknown file format versions {w}/{r}"),
    }
    let usable = page_size - usize::from(data[20]);
    if usable < 480 {
        fail!("not an sqlite database: usable page size {usable} is below 480");
    }
    if (data[21], data[22], data[23]) != (64, 32, 32) {
        fail!("not an sqlite database: invalid payload fractions");
    }
    match be32(data, 56)? {
        0 | 1 => {}
        e => fail!("text encoding {e} is not UTF-8; this reader only reads UTF-8 databases"),
    }
    let in_header = be32(data, 28)?;
    let valid = be32(data, 24)? == be32(data, 92)?;
    let from_size = u32::try_from(data.len() / page_size).unwrap_or(u32::MAX);
    let pages = if valid && in_header > 0 {
        in_header
    } else {
        from_size
    };
    if (pages as usize).saturating_mul(page_size) > data.len() {
        fail!(
            "the file is truncated: the header declares {pages} pages of {page_size} bytes, the file has {} bytes",
            data.len()
        );
    }
    Ok((page_size, usable, pages))
}

impl Db {
    /// Read the file at `path` (refusing a WAL sidecar) and check its header.
    pub fn open(path: &Path) -> Result<Db> {
        let label = path.display().to_string();
        let mut wal = path.as_os_str().to_owned();
        wal.push("-wal");
        if Path::new(&wal).exists() {
            fail!("{label}: a -wal file exists next to the database; checkpoint it before reading");
        }
        let data = std::fs::read(path).map_err(|e| Error(format!("cannot read {label}: {e}")))?;
        Db::from_bytes(&label, data)
    }

    /// Check the header of an in-memory database image.
    pub fn from_bytes(label: &str, data: Vec<u8>) -> Result<Db> {
        let (page_size, usable, pages) = header(&data).map_err(|e| e.context(label))?;
        Ok(Db {
            label: label.to_string(),
            data,
            page_size,
            usable,
            pages,
        })
    }

    fn tree(&self) -> Pages<'_> {
        Pages {
            data: &self.data,
            page_size: self.page_size,
            usable: self.usable,
            count: self.pages,
        }
    }

    /// The table named `name` (case-insensitive), or `None`.
    pub fn table(&self, name: &str) -> Result<Option<Table>> {
        let mut found: Option<(u32, String)> = None;
        let mut visit = |_: i64, rec: &[u8]| -> Result<()> {
            let v = record::columns(rec, &[0, 1, 3, 4])?;
            let text = |i: usize| match &v[i] {
                Some(Value::Text(s)) => Some(s.as_str()),
                _ => None,
            };
            if text(0) == Some("table") && text(1).is_some_and(|n| n.eq_ignore_ascii_case(name)) {
                let root = match &v[2] {
                    Some(Value::Int(r)) => u32::try_from(*r)
                        .map_err(|_| Error(format!("table {name}: bad root page")))?,
                    _ => fail!("table {name}: no root page in the schema"),
                };
                found = Some((root, text(3).unwrap_or_default().to_string()));
            }
            Ok(())
        };
        self.tree()
            .scan(1, &mut visit)
            .map_err(|e| e.context(&self.label))?;
        let Some((root, sql)) = found else {
            return Ok(None);
        };
        let columns =
            table_columns(&sql).map_err(|e| e.context(&format!("{}: table {name}", self.label)))?;
        Ok(Some(Table {
            name: name.to_string(),
            root,
            columns,
        }))
    }

    /// Visit every row of `table`, decoding the named columns in that order.
    pub fn scan(
        &self,
        table: &Table,
        names: &[&str],
        f: &mut dyn FnMut(&[Value]) -> Result<()>,
    ) -> Result<()> {
        let mut pos = Vec::with_capacity(names.len());
        for n in names {
            match table
                .columns
                .iter()
                .position(|c| c.name.eq_ignore_ascii_case(n))
            {
                Some(p) => pos.push(p),
                None => fail!("{}: table {} has no column {n}", self.label, table.name),
            }
        }
        let mut row = Vec::with_capacity(names.len());
        let mut visit = |rowid: i64, rec: &[u8]| -> Result<()> {
            row.clear();
            for (v, &p) in record::columns(rec, &pos)?.into_iter().zip(&pos) {
                let col = &table.columns[p];
                row.push(match v {
                    _ if col.rowid_alias => Value::Int(rowid),
                    Some(Value::Int(i)) if col.real_affinity => Value::Real(i as f64),
                    Some(v) => v,
                    None if col.has_default => fail!(
                        "column {} relies on a DEFAULT this reader does not evaluate",
                        col.name
                    ),
                    None => Value::Null,
                });
            }
            f(&row)
        };
        self.tree()
            .scan(table.root, &mut visit)
            .map_err(|e| e.context(&format!("{}: table {}", self.label, table.name)))
    }
}
