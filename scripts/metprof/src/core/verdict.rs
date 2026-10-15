// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: A/B verdict from a record (pure). Every field is required;
//! nothing defaults.
//!
//! Rules, in order:
//! - not armed -> invalid
//! - bit-identical lever and the bits differ -> parity-broken
//! - bit-changing lever and the bits are identical -> unproven (the lever did
//!   not touch the path it claims to)
//! - improvement > band -> win if engaged, else unproven
//! - worsening > band -> loss
//! - within the band -> neutral if engaged, else unproven
//!
//! The band is in the metric's units (the control spread or the MDE for the n
//! run). improvement = control - treatment when lower is better, else the
//! reverse.

use serde_json::{Map, Value, json};

use crate::core::error::Result;
use crate::core::repr::{quote_list, toml_value};
use crate::core::tables::into_map;
use crate::fail;

#[derive(Clone, Copy)]
enum Kind {
    Str,
    Bool,
    Num,
}

const FIELDS: [(&str, Kind); 9] = [
    ("metric", Kind::Str),
    ("lower_is_better", Kind::Bool),
    ("control", Kind::Num),
    ("treatment", Kind::Num),
    ("band", Kind::Num),
    ("armed", Kind::Bool),
    ("engaged", Kind::Bool),
    ("lever_parity", Kind::Str),
    ("bits", Kind::Str),
];
const OPTIONAL: [&str; 1] = ["notes"];
const PARITY: [&str; 2] = ["bit-identical", "bit-changing"];
const BITS: [&str; 2] = ["identical", "differ"];

fn num(rec: &toml::Table, key: &str) -> Result<f64> {
    let v = &rec[key];
    let f = match v {
        toml::Value::Integer(i) => Some(*i as f64),
        toml::Value::Float(f) => Some(*f),
        _ => None,
    };
    match f.filter(|x| x.is_finite()) {
        Some(x) => Ok(x),
        None => fail!(
            "record field {key} must be a finite number, got {}",
            toml_value(v)
        ),
    }
}

fn text<'a>(rec: &'a toml::Table, key: &str) -> Result<&'a str> {
    match rec[key].as_str() {
        Some(s) => Ok(s),
        None => fail!(
            "record field {key} must be str, got {}",
            toml_value(&rec[key])
        ),
    }
}

fn flag(rec: &toml::Table, key: &str) -> Result<bool> {
    match rec[key].as_bool() {
        Some(b) => Ok(b),
        None => fail!(
            "record field {key} must be bool, got {}",
            toml_value(&rec[key])
        ),
    }
}

fn one_of<'a>(value: &'a str, key: &str, allowed: &[&str]) -> Result<&'a str> {
    if !allowed.contains(&value) {
        let inner: Vec<String> = allowed.iter().map(|a| format!("'{a}'")).collect();
        fail!(
            "record field {key} must be one of ({}), got '{value}'",
            inner.join(", ")
        );
    }
    Ok(value)
}

pub fn decide(rec: &toml::Table) -> Result<Map<String, Value>> {
    let names: Vec<&str> = FIELDS.iter().map(|f| f.0).collect();
    let mut missing: Vec<&str> = names
        .iter()
        .copied()
        .filter(|k| !rec.contains_key(*k))
        .collect();
    let mut unknown: Vec<&str> = rec
        .keys()
        .map(String::as_str)
        .filter(|k| !names.contains(k) && !OPTIONAL.contains(k))
        .collect();
    missing.sort_unstable();
    unknown.sort_unstable();
    if !missing.is_empty() || !unknown.is_empty() {
        fail!(
            "record: missing fields {}, unknown fields {}",
            quote_list(&missing),
            quote_list(&unknown)
        );
    }
    for (key, kind) in FIELDS {
        match kind {
            Kind::Str => text(rec, key).map(|_| ())?,
            Kind::Bool => flag(rec, key).map(|_| ())?,
            Kind::Num => num(rec, key).map(|_| ())?,
        }
    }
    if let Some(notes) = rec.get("notes")
        && !notes.is_str()
    {
        fail!("record field notes must be str, got {}", toml_value(notes));
    }
    let (control, treatment, band) = (
        num(rec, "control")?,
        num(rec, "treatment")?,
        num(rec, "band")?,
    );
    if band <= 0.0 {
        fail!("record field band must be positive, got {band}");
    }
    let parity = one_of(text(rec, "lever_parity")?, "lever_parity", &PARITY)?;
    let bits = one_of(text(rec, "bits")?, "bits", &BITS)?;
    let (armed, engaged) = (flag(rec, "armed")?, flag(rec, "engaged")?);
    let delta = treatment - control;
    let improvement = if flag(rec, "lower_is_better")? {
        -delta
    } else {
        delta
    };
    let (verdict, why) = if !armed {
        ("invalid", "the lever was not proven armed")
    } else if parity == "bit-identical" && bits == "differ" {
        (
            "parity-broken",
            "a bit-identical lever changed the output bits",
        )
    } else if parity == "bit-changing" && bits == "identical" {
        ("unproven", "a bit-changing lever left the bits identical")
    } else if improvement > band {
        if engaged {
            ("win", "better by more than the band")
        } else {
            (
                "unproven",
                "better by more than the band, but the lever was not proven engaged",
            )
        }
    } else if -improvement > band {
        ("loss", "worse by more than the band")
    } else if engaged {
        ("neutral", "within the band")
    } else {
        (
            "unproven",
            "within the band and the lever was not proven engaged",
        )
    };
    let ratio = if control != 0.0 {
        Some(treatment / control)
    } else {
        None
    };
    Ok(into_map(json!({
        "verdict": verdict,
        "reason": why,
        "metric": text(rec, "metric")?,
        "control": control,
        "treatment": treatment,
        "delta": delta,
        "ratio": ratio,
        "improvement": improvement,
        "band": band,
    })))
}
