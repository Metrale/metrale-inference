// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: an nsys sqlite export (`nsys export --type sqlite`) as a
//! `Trace`. The file is only ever read; a wrong path is an error, never a new
//! empty database. Names are joined from `StringIds`; a kernel or API call
//! whose name id is absent from it is an error, never a silently dropped row.

use std::collections::HashMap;
use std::path::Path;

use crate::adapters::sqlite::{Db, Table, Value};
use crate::core::capture::{CopyRow, KernelRow};
use crate::core::error::Result;
use crate::core::model::{ApiCall, Kernel, Launch};
use crate::fail;
use crate::ports::{Trace, TraceSource};

pub const KERNEL_TABLE: &str = "CUPTI_ACTIVITY_KIND_KERNEL";
pub const MEMCPY_TABLE: &str = "CUPTI_ACTIVITY_KIND_MEMCPY";
pub const RUNTIME_TABLE: &str = "CUPTI_ACTIVITY_KIND_RUNTIME";
pub const STRINGS_TABLE: &str = "StringIds";

pub struct NsysSource;

pub struct NsysTrace {
    path: String,
    db: Db,
    kernels: Table,
    strings: Table,
}

impl TraceSource for NsysSource {
    fn open(&self, path: &str) -> Result<Box<dyn Trace>> {
        Ok(Box::new(NsysTrace::open(path)?))
    }
}

fn int(v: &Value, what: &str) -> Result<i64> {
    match v {
        Value::Int(i) => Ok(*i),
        other => fail!("{what} is not an integer: {other:?}"),
    }
}

fn opt_int(v: &Value, what: &str) -> Result<Option<i64>> {
    match v {
        Value::Null => Ok(None),
        v => int(v, what).map(Some),
    }
}

impl NsysTrace {
    pub fn open(path: &str) -> Result<NsysTrace> {
        if !Path::new(path).is_file() {
            fail!("capture not found: {path}");
        }
        let db = Db::open(Path::new(path))?;
        let kernels = Self::require(&db, path, KERNEL_TABLE)?;
        let strings = Self::require(&db, path, STRINGS_TABLE)?;
        Ok(NsysTrace {
            path: path.to_string(),
            db,
            kernels,
            strings,
        })
    }

    fn require(db: &Db, path: &str, name: &str) -> Result<Table> {
        match db.table(name)? {
            Some(t) => Ok(t),
            None => fail!("{path}: table {name} is missing from the export"),
        }
    }

    /// The strings for exactly the ids asked for; an absent id is an error.
    fn names(&self, ids: impl IntoIterator<Item = i64>) -> Result<HashMap<i64, String>> {
        let mut want: HashMap<i64, Option<String>> = ids.into_iter().map(|i| (i, None)).collect();
        self.db.scan(&self.strings, &["id", "value"], &mut |r| {
            if let Some(slot) = want.get_mut(&int(&r[0], "StringIds.id")?) {
                match &r[1] {
                    Value::Text(s) => *slot = Some(s.clone()),
                    other => fail!("StringIds.value is not text: {other:?}"),
                }
            }
            Ok(())
        })?;
        let mut out = HashMap::with_capacity(want.len());
        for (id, v) in want {
            match v {
                Some(s) => out.insert(id, s),
                None => fail!(
                    "{}: string id {id} is missing from {STRINGS_TABLE}",
                    self.path
                ),
            };
        }
        Ok(out)
    }
}

impl Trace for NsysTrace {
    fn kernel_rows(&self) -> Result<Vec<KernelRow>> {
        let node = self.kernels.has_column("graphNodeId");
        let mut cols = vec![
            "start",
            "end",
            "deviceId",
            "streamId",
            "correlationId",
            "shortName",
            "gridX",
            "gridY",
            "gridZ",
            "demangledName",
            "blockX",
            "blockY",
            "blockZ",
            "registersPerThread",
            "staticSharedMemory",
            "dynamicSharedMemory",
        ];
        if node {
            cols.push("graphNodeId");
        }
        let mut raw: Vec<(i64, Kernel, i64, i64)> = Vec::new();
        self.db.scan(&self.kernels, &cols, &mut |r| {
            let kernel = Kernel {
                start: int(&r[0], "start")?,
                end: int(&r[1], "end")?,
                stream: int(&r[3], "streamId")?,
                correlation: int(&r[4], "correlationId")?,
                graph_node: if node {
                    opt_int(&r[16], "graphNodeId")?
                } else {
                    None
                },
                name: String::new(),
                demangled: String::new(),
                grid: (
                    int(&r[6], "gridX")?,
                    int(&r[7], "gridY")?,
                    int(&r[8], "gridZ")?,
                ),
                launch: Launch {
                    block: (
                        int(&r[10], "blockX")?,
                        int(&r[11], "blockY")?,
                        int(&r[12], "blockZ")?,
                    ),
                    registers_per_thread: int(&r[13], "registersPerThread")?,
                    static_shared_bytes: int(&r[14], "staticSharedMemory")?,
                    dynamic_shared_bytes: int(&r[15], "dynamicSharedMemory")?,
                },
            };
            raw.push((
                int(&r[2], "deviceId")?,
                kernel,
                int(&r[5], "shortName")?,
                int(&r[9], "demangledName")?,
            ));
            Ok(())
        })?;
        let names = self.names(raw.iter().flat_map(|r| [r.2, r.3]))?;
        Ok(raw
            .into_iter()
            .map(|(device, mut kernel, short, full)| {
                kernel.name = names[&short].clone();
                kernel.demangled = names[&full].clone();
                KernelRow { device, kernel }
            })
            .collect())
    }

    fn copy_rows(&self) -> Result<Vec<CopyRow>> {
        let t = Self::require(&self.db, &self.path, MEMCPY_TABLE)?;
        let mut out = Vec::new();
        self.db.scan(
            &t,
            &["start", "end", "deviceId", "copyKind", "bytes"],
            &mut |r| {
                out.push(CopyRow {
                    start: int(&r[0], "start")?,
                    end: int(&r[1], "end")?,
                    device: int(&r[2], "deviceId")?,
                    kind: int(&r[3], "copyKind")?,
                    nbytes: int(&r[4], "bytes")?,
                });
                Ok(())
            },
        )?;
        Ok(out)
    }

    fn call_rows(&self) -> Result<Vec<ApiCall>> {
        let t = Self::require(&self.db, &self.path, RUNTIME_TABLE)?;
        let mut raw: Vec<(ApiCall, i64)> = Vec::new();
        self.db
            .scan(&t, &["start", "end", "correlationId", "nameId"], &mut |r| {
                let call = ApiCall {
                    start: int(&r[0], "start")?,
                    end: int(&r[1], "end")?,
                    name: String::new(),
                    correlation: int(&r[2], "correlationId")?,
                };
                raw.push((call, int(&r[3], "nameId")?));
                Ok(())
            })?;
        let names = self.names(raw.iter().map(|r| r.1))?;
        Ok(raw
            .into_iter()
            .map(|(mut c, id)| {
                c.name = names[&id].clone();
                c
            })
            .collect())
    }
}
