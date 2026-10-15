// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: `METRALE_GLM_DSA_SP_SELECT`: the measured split table of the sequence-parallel DSA
//! selection, read once.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - Unset or `0` is off. Any other value must be a [`MinPoolsTable`] spelled `k:min_pools`
//!   rows joined by `,` (`k` strictly increasing); anything else, the empty string included,
//!   is an error at the first read (startup, through the rank-agreement check), never a
//!   silent off. There is no built-in table: the value is the measured table.
//! - [`agreement_value`] is 0 when off and otherwise a hash of the canonical table, so two
//!   ranks that would take different split branches disagree at startup.

use std::sync::OnceLock;

use anyhow::{Result, anyhow};

use super::MinPoolsTable;

/// 2026-10-15: The lever's name.
pub const ENV: &str = "METRALE_GLM_DSA_SP_SELECT";

/// 2026-10-15: The table a lever value names: `None` when off (unset or `0`).
pub fn parse(v: Option<&str>) -> Result<Option<MinPoolsTable>, String> {
    let s = match v {
        None | Some("0") => return Ok(None),
        Some(s) => s,
    };
    let bad = |why: &str| {
        format!("{ENV}={s:?}: {why}; expected `0` or measured `k:min_pools` rows joined by `,`")
    };
    let rows = s
        .split(',')
        .map(|row| {
            let (k, p) = row
                .split_once(':')
                .ok_or_else(|| bad(&format!("row {row:?} has no `:`")))?;
            let num = |x: &str| {
                x.trim()
                    .parse::<usize>()
                    .map_err(|_| bad(&format!("{x:?} is not a row or pool count")))
            };
            let (k, p) = (num(k)?, num(p)?);
            if k == 0 {
                return Err(bad("a row of k = 0"));
            }
            Ok((k, p))
        })
        .collect::<Result<Vec<_>, String>>()?;
    MinPoolsTable::new(rows)
        .map(Some)
        .map_err(|e| bad(&e.to_string()))
}

/// 2026-10-15: The lever as read once from the environment, with its one-time readback line
/// (`DSA SP select: sp_select=<0|1> table=<canonical|->`), or the parse error.
pub fn table() -> Result<Option<&'static MinPoolsTable>> {
    static T: OnceLock<Result<Option<MinPoolsTable>, String>> = OnceLock::new();
    let t = T.get_or_init(|| {
        let t = parse(std::env::var(ENV).ok().as_deref());
        if let Ok(on) = &t {
            tracing::info!(
                "DSA SP select: sp_select={} table={}",
                u8::from(on.is_some()),
                on.as_ref()
                    .map_or_else(|| "-".to_string(), |t| t.canonical())
            );
        }
        t
    });
    t.as_ref().map(Option::as_ref).map_err(|e| anyhow!("{e}"))
}

/// 2026-10-15: The rank-agreement value of a table: 0 when off, else the 64-bit FNV-1a of its
/// canonical spelling (never 0 for a table, which is never empty).
pub fn agreement_of(t: Option<&MinPoolsTable>) -> u64 {
    t.map_or(0, |t| {
        t.canonical()
            .bytes()
            .fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
                (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
            })
    })
}

/// 2026-10-15: [`agreement_of`] the lever as read.
pub fn agreement_value() -> Result<u64> {
    Ok(agreement_of(table()?))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-10-15: Off spellings, a table and its canonical form, and every malformed value
    /// refused (none silently off).
    #[test]
    fn the_lever_parses_tables_and_refuses_everything_else() {
        assert_eq!(parse(None), Ok(None));
        assert_eq!(parse(Some("0")), Ok(None));
        let t = parse(Some("4096:1024, 8208 :2052")).unwrap().unwrap();
        assert_eq!(t.canonical(), "4096:1024,8208:2052");
        assert_eq!(parse(Some(&t.canonical())).unwrap().unwrap(), t);
        assert!(t.split(8208, 2052) && !t.split(8208, 2051) && t.split(5000, 1024));
        for bad in [
            "",
            "1",
            "on",
            "8208",
            "8208:",
            ":2052",
            "8208:-1",
            "0:5",
            "8208:2052,4096:1024",
            "8208:2052,8208:4000",
            "8208:2052,",
        ] {
            assert!(parse(Some(bad)).is_err(), "{bad:?} must be refused");
        }
    }

    /// 2026-10-15: The agreement value tells off from on and one table from another, and is
    /// the same for the same table however it was spelled.
    #[test]
    fn the_agreement_value_separates_tables() {
        let t = |s: &str| parse(Some(s)).unwrap();
        let a = agreement_of(t("8208:2052").as_ref());
        assert_eq!(agreement_of(None), 0);
        assert_ne!(a, 0);
        assert_eq!(a, agreement_of(t(" 8208 : 2052").as_ref()));
        assert_ne!(a, agreement_of(t("8208:2053").as_ref()));
        assert_ne!(a, agreement_of(t("4096:2052").as_ref()));
        assert_ne!(a, agreement_of(t("4096:1024,8208:2052").as_ref()));
    }
}
