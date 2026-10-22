// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: JSON Schema from the export types: each type is declared once through
//! [`json_struct!`] / [`json_enum!`], which emit both the serde type and its schema, so the
//! committed `schemas/kbp-*.v1.json` cannot drift from what `met blueprint export` writes.
//!
//! Owner: blueprint.
//! Invariants:
//! - Every struct is `additionalProperties: false` with every field required (an `Option`
//!   field is required and nullable): a consumer sees exactly the fields the exporter writes.
//! - Output is deterministic: `$defs` is a `BTreeMap`, properties are in declaration order.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use serde_json::{Map, Value, json};

/// 2026-10-22: A type that can state its JSON Schema. Named types register themselves in
/// `defs` and return a `$ref`.
pub trait Describe {
    /// 2026-10-22: The schema of `Self`.
    fn describe(defs: &mut BTreeMap<String, Value>) -> Value;
}

macro_rules! leaf {
    ($($t:ty => $v:expr),* $(,)?) => {
        $(impl Describe for $t {
            fn describe(_: &mut BTreeMap<String, Value>) -> Value {
                $v
            }
        })*
    };
}

leaf! {
    String => json!({"type": "string"}),
    bool => json!({"type": "boolean"}),
    u32 => json!({"minimum": 0, "type": "integer"}),
    u64 => json!({"minimum": 0, "type": "integer"}),
    i64 => json!({"type": "integer"}),
    f64 => json!({"type": "number"}),
}

impl<T: Describe> Describe for Option<T> {
    fn describe(defs: &mut BTreeMap<String, Value>) -> Value {
        json!({"anyOf": [T::describe(defs), {"type": "null"}]})
    }
}

impl<T: Describe> Describe for Vec<T> {
    fn describe(defs: &mut BTreeMap<String, Value>) -> Value {
        json!({"items": T::describe(defs), "type": "array"})
    }
}

impl<T: Describe> Describe for BTreeMap<String, T> {
    fn describe(defs: &mut BTreeMap<String, Value>) -> Value {
        json!({"additionalProperties": T::describe(defs), "type": "object"})
    }
}

/// 2026-10-22: A field's schema function.
pub type Describer = fn(&mut BTreeMap<String, Value>) -> Value;

/// 2026-10-22: Register an object schema named `name` (once) and return its `$ref`. Keys are
/// inserted in sorted order, so the text is the same whether or not serde_json's
/// `preserve_order` is unified into the build.
pub fn object(
    defs: &mut BTreeMap<String, Value>,
    name: &str,
    fields: &[(&str, Describer)],
) -> Value {
    if !defs.contains_key(name) {
        defs.insert(String::from(name), Value::Null);
        let mut sorted: Vec<&(&str, Describer)> = fields.iter().collect();
        sorted.sort_by_key(|(f, _)| *f);
        let mut props = Map::new();
        for (f, d) in sorted {
            props.insert(String::from(*f), d(defs));
        }
        let required: Vec<Value> = fields.iter().map(|(f, _)| Value::from(*f)).collect();
        defs.insert(
            String::from(name),
            json!({"additionalProperties": false, "properties": props, "required": required, "type": "object"}),
        );
    }
    json!({"$ref": alloc::format!("#/$defs/{name}")})
}

/// 2026-10-22: The schema document of `T`: draft 2020-12, the root a `$ref` into `$defs`.
pub fn document<T: Describe>(id: &str, title: &str) -> Value {
    let mut defs = BTreeMap::new();
    let root = T::describe(&mut defs);
    let defs: Map<String, Value> = defs.into_iter().collect();
    json!({
        "$defs": defs,
        "$id": id,
        "$ref": root["$ref"],
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": title,
    })
}

/// 2026-10-22: Declare an export struct: the serde type and its [`Describe`] impl.
macro_rules! json_struct {
    ($(#[$m:meta])* pub struct $name:ident { $($(#[$fm:meta])* pub $f:ident : $t:ty),* $(,)? }) => {
        $(#[$m])*
        #[derive(Clone, Debug, PartialEq, serde::Serialize)]
        pub struct $name { $($(#[$fm])* pub $f: $t),* }

        impl $crate::export::schema::Describe for $name {
            fn describe(defs: &mut alloc::collections::BTreeMap<alloc::string::String, serde_json::Value>) -> serde_json::Value {
                $crate::export::schema::object(defs, stringify!($name), &[
                    $((stringify!($f), <$t as $crate::export::schema::Describe>::describe as $crate::export::schema::Describer)),*
                ])
            }
        }
    };
}

/// 2026-10-22: Declare an export string enum: serialized as its spelling, described as a JSON
/// Schema `enum`.
macro_rules! json_enum {
    ($(#[$m:meta])* pub enum $name:ident { $($(#[$vm:meta])* $v:ident => $s:literal),* $(,)? }) => {
        $(#[$m])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub enum $name { $($(#[$vm])* $v),* }

        impl $name {
            /// 2026-10-22: The spelling.
            pub const fn name(self) -> &'static str {
                match self { $($name::$v => $s),* }
            }
        }

        impl serde::Serialize for $name {
            fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
                s.serialize_str(self.name())
            }
        }

        impl $crate::export::schema::Describe for $name {
            fn describe(defs: &mut alloc::collections::BTreeMap<alloc::string::String, serde_json::Value>) -> serde_json::Value {
                let name = stringify!($name);
                if !defs.contains_key(name) {
                    defs.insert(alloc::string::String::from(name), serde_json::json!({"enum": [$($s),*], "type": "string"}));
                }
                serde_json::json!({"$ref": alloc::format!("#/$defs/{name}")})
            }
        }
    };
}

pub(crate) use {json_enum, json_struct};
