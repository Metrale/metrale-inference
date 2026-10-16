// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-16: per-row cost (pure): how one cost moves as the decode batch
//! widens, from the same cost measured at three or more widths (rows per
//! step).
//!
//! Between two consecutive widths the marginal is the added ms per step
//! divided by the added rows. With three or more widths the marginals
//! themselves can be compared: a marginal that falls as the batch widens is a
//! cost shared across rows (weight bytes read once per step), a flat one is a
//! cost paid again by every row (per-row loops, launches, state), and a rising
//! one is a cliff. Each change between consecutive marginals is classified
//! against a band from the config (`[slope] flat_band_ms_per_row`, no
//! default): within the band, inclusive, is FLAT; above it RISING; below its
//! negative FALLING. The whole series is RISING when any change is (a cliff
//! dominates), else FALLING when any change is, else FLAT.
//!
//! A cost absent at a width is 0 there (it did not run), and the width is
//! listed in `absent_at`, so an absence is never mistaken for a measurement.

use std::collections::BTreeMap;

use crate::core::error::Result;
use crate::core::families::{FamilyTable, check_keys};
use crate::core::model::ms;
use crate::core::repr::{quote, quote_list, toml_value};
use crate::core::segment::per_unit;
use crate::fail;

/// A marginal needs two widths and a change in marginal needs three.
pub const MIN_WIDTHS: usize = 3;
/// What a family's per-row cost is declared to come from.
pub const DRIVERS: [&str; 4] = [
    "weight_bytes",
    "state_bytes",
    "launches_per_row",
    "host_sync",
];
/// The cost driver of a family the config does not declare one for.
pub const UNDECLARED: &str = "undeclared";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Class {
    Falling,
    Flat,
    Rising,
}

impl Class {
    pub const ALL: [Class; 3] = [Class::Falling, Class::Flat, Class::Rising];

    pub fn as_str(self) -> &'static str {
        match self {
            Class::Falling => "FALLING",
            Class::Flat => "FLAT",
            Class::Rising => "RISING",
        }
    }
}

/// The class of one change in marginal ms per row. The band is inclusive:
/// a change of exactly `band` either way is FLAT.
pub fn class_of(delta_ms_per_row: f64, band: f64) -> Class {
    if delta_ms_per_row > band {
        Class::Rising
    } else if delta_ms_per_row < -band {
        Class::Falling
    } else {
        Class::Flat
    }
}

/// One cost at one width, per pinned decode step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    pub rows: u64,
    pub ms_per_step: f64,
    pub launches_per_step: f64,
}

impl Point {
    /// The cost of `steps` steps, `total_ns` of kernel time over `launches`.
    pub fn of(rows: u64, steps: usize, total_ns: i64, launches: u64) -> Point {
        Point {
            rows,
            ms_per_step: ms(per_unit(total_ns as f64, steps)),
            launches_per_step: per_unit(launches as f64, steps),
        }
    }

    pub fn launches_per_row(&self) -> f64 {
        self.launches_per_step / self.rows as f64
    }
}

/// The added cost per added row from one width to the next.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Marginal {
    pub from_rows: u64,
    pub to_rows: u64,
    pub ms_per_row: f64,
    pub launches_per_row: f64,
}

/// The change between the marginals on either side of the width `at_rows`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Change {
    pub at_rows: u64,
    pub delta_ms_per_row: f64,
    pub class: Class,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Analysis {
    pub marginals: Vec<Marginal>,
    pub changes: Vec<Change>,
    pub class: Class,
    /// Widths at which the cost did not run (no launches).
    pub absent_at: Vec<u64>,
}

/// The order that sorts `rows` ascending. Fewer than `MIN_WIDTHS` widths, a
/// width below 1 row and a width given twice are refused.
pub fn order_widths(rows: &[u64]) -> Result<Vec<usize>> {
    if rows.len() < MIN_WIDTHS {
        fail!(
            "slope needs at least {MIN_WIDTHS} widths (one --at ROWS=CAPTURE each), got {}: a marginal \
             needs two widths and a change in marginal needs three",
            rows.len()
        );
    }
    if rows.contains(&0) {
        fail!("slope: a width is at least 1 row, got 0");
    }
    let mut order: Vec<usize> = (0..rows.len()).collect();
    order.sort_by_key(|&i| rows[i]);
    if let Some(w) = order.windows(2).find(|w| rows[w[0]] == rows[w[1]]) {
        fail!(
            "slope: width {} rows is given twice; each width is one capture",
            rows[w[0]]
        );
    }
    Ok(order)
}

/// Marginals, changes and the class of `points`, which must be ordered by
/// strictly increasing rows (`order_widths`) and number at least `MIN_WIDTHS`.
pub fn analyze(points: &[Point], band: f64) -> Result<Analysis> {
    let rows: Vec<u64> = points.iter().map(|p| p.rows).collect();
    let order = order_widths(&rows)?;
    if order.iter().enumerate().any(|(i, &o)| i != o) {
        fail!("slope: points must be ordered by increasing rows, got {rows:?}");
    }
    let marginals: Vec<Marginal> = points
        .windows(2)
        .map(|w| {
            let dr = (w[1].rows - w[0].rows) as f64;
            Marginal {
                from_rows: w[0].rows,
                to_rows: w[1].rows,
                ms_per_row: (w[1].ms_per_step - w[0].ms_per_step) / dr,
                launches_per_row: (w[1].launches_per_step - w[0].launches_per_step) / dr,
            }
        })
        .collect();
    let changes: Vec<Change> = marginals
        .windows(2)
        .map(|m| {
            let delta = m[1].ms_per_row - m[0].ms_per_row;
            Change {
                at_rows: m[0].to_rows,
                delta_ms_per_row: delta,
                class: class_of(delta, band),
            }
        })
        .collect();
    let has = |c: Class| changes.iter().any(|x| x.class == c);
    let class = if has(Class::Rising) {
        Class::Rising
    } else if has(Class::Falling) {
        Class::Falling
    } else {
        Class::Flat
    };
    let absent_at = points
        .iter()
        .filter(|p| p.launches_per_step == 0.0)
        .map(|p| p.rows)
        .collect();
    Ok(Analysis {
        marginals,
        changes,
        class,
        absent_at,
    })
}

/// The `[slope]` section of the family table file.
#[derive(Debug, Clone, PartialEq)]
pub struct SlopeConfig {
    /// Half-width of the FLAT band on a change in marginal ms per row.
    pub band_ms_per_row: f64,
    /// Declared cost drivers by family (each one of `DRIVERS`).
    pub drivers: BTreeMap<String, &'static str>,
}

impl SlopeConfig {
    /// The declared driver of `family`, or `UNDECLARED`.
    pub fn driver(&self, family: &str) -> &'static str {
        self.drivers.get(family).copied().unwrap_or(UNDECLARED)
    }
}

/// A band is a finite number of ms per row, at least 0.
pub fn check_band(place: &str, band: f64) -> Result<f64> {
    if !band.is_finite() || band < 0.0 {
        fail!("{place}: flat_band_ms_per_row must be a finite number >= 0, got {band}");
    }
    Ok(band)
}

/// `[slope]`: `flat_band_ms_per_row` (required) and `[slope.cost_driver]`
/// (optional; `family = "<driver>"` for declared families only).
pub fn slope_config(doc: &toml::Table, table: &FamilyTable) -> Result<SlopeConfig> {
    const PLACE: &str = "families [slope]";
    let Some(raw) = doc.get("slope") else {
        fail!("families: [slope] is required by `slope` (flat_band_ms_per_row has no default)");
    };
    let Some(s) = raw.as_table() else {
        fail!("{PLACE}: must be a table");
    };
    check_keys(PLACE, s, &["flat_band_ms_per_row"], &["cost_driver"])?;
    let band = match &s["flat_band_ms_per_row"] {
        toml::Value::Integer(i) => *i as f64,
        toml::Value::Float(f) => *f,
        other => fail!(
            "{PLACE}: flat_band_ms_per_row must be a number, got {}",
            toml_value(other)
        ),
    };
    let band_ms_per_row = check_band(PLACE, band)?;
    let mut drivers = BTreeMap::new();
    if let Some(v) = s.get("cost_driver") {
        let Some(d) = v.as_table() else {
            fail!("families [slope.cost_driver]: must be a table");
        };
        let fams = table.families();
        for (family, driver) in d {
            if !fams.contains(&family.as_str()) {
                fail!(
                    "families [slope.cost_driver]: {} is not a declared family",
                    quote(family)
                );
            }
            let Some(known) = driver
                .as_str()
                .and_then(|x| DRIVERS.iter().find(|k| **k == x))
            else {
                fail!(
                    "families [slope.cost_driver]: {} must be one of {}, got {}",
                    quote(family),
                    quote_list(&DRIVERS),
                    toml_value(driver)
                );
            };
            drivers.insert(family.clone(), *known);
        }
    }
    Ok(SlopeConfig {
        band_ms_per_row,
        drivers,
    })
}
