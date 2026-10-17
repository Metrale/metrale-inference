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
//! - 2026-10-15: `METRALE_GLM_DSA_SP_SELECT_SEL` picks the gathered payload of a split pass:
//!   unset or `0` the expanded token rows, `1` the folded selected pools (`select::folded`),
//!   which needs the table on; any other value, or `1` without a table, is a startup error.

use std::sync::OnceLock;

use anyhow::{Result, anyhow};

use super::{MIN_SUBPASS_ROWS, MinPoolsTable};

/// 2026-10-15: The lever's name.
pub const ENV: &str = "METRALE_GLM_DSA_SP_SELECT";

/// 2026-10-15: The payload sub-lever's name.
pub const ENV_SEL: &str = "METRALE_GLM_DSA_SP_SELECT_SEL";

/// 2026-10-15: What a split pass gathers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Payload {
    /// 2026-10-15: The expanded `[rows, out_width]` token rows.
    Tokens,
    /// 2026-10-15: The folded `[rows, select_k]` selected pools; every rank expands all rows.
    Selected,
}

/// 2026-10-15: The payload a sub-lever value names, given whether the table is on.
pub fn parse_payload(v: Option<&str>, table_on: bool) -> Result<Payload, String> {
    match v {
        None | Some("0") => Ok(Payload::Tokens),
        Some("1") if table_on => Ok(Payload::Selected),
        Some("1") => Err(format!("{ENV_SEL}=1 needs {ENV} set to a table")),
        Some(s) => Err(format!("{ENV_SEL}={s:?}: expected `0` or `1`")),
    }
}

/// 2026-10-15: The payload as read once, with its one-time readback line
/// (`DSA SP select payload: sel=<0|1> row_bytes=<select_k*4|out_width*4> fold=launch
/// expand=replicated`), or the error.
pub fn payload() -> Result<Payload> {
    static P: OnceLock<Result<Payload, String>> = OnceLock::new();
    let p = P.get_or_init(|| {
        let on = table().map_err(|e| e.to_string())?.is_some();
        let p = parse_payload(std::env::var(ENV_SEL).ok().as_deref(), on);
        if let Ok(p) = p {
            let sel = p == Payload::Selected;
            tracing::info!(
                "DSA SP select payload: sel={} row_bytes={} fold={} expand=replicated",
                u8::from(sel),
                if sel { "select_k*4" } else { "out_width*4" },
                if sel { "launch" } else { "-" }
            );
        }
        p
    });
    p.clone().map_err(|e| anyhow!("{e}"))
}

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
/// (`DSA SP select: sp_select=<0|1> table=<canonical|-> prefill_only=1 min_subpass_rows=2`), or
/// the parse error. 2026-10-15: the last two fields name the bit-identity guards of
/// `run::plan`: decode never splits, nor does a pass with a 1-row sub-pass.
pub fn table() -> Result<Option<&'static MinPoolsTable>> {
    static T: OnceLock<Result<Option<MinPoolsTable>, String>> = OnceLock::new();
    let t = T.get_or_init(|| {
        let t = parse(std::env::var(ENV).ok().as_deref());
        if let Ok(on) = &t {
            tracing::info!(
                "DSA SP select: sp_select={} table={} prefill_only=1 min_subpass_rows={}",
                u8::from(on.is_some()),
                on.as_ref()
                    .map_or_else(|| "-".to_string(), |t| t.canonical()),
                MIN_SUBPASS_ROWS
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

/// 2026-10-15: [`agreement_of`] the lever as read, and the payload in the low bit (the
/// table's hash with bit 0 flipped for the selected-pools payload, which needs a table, so
/// off stays 0).
pub fn agreement_value() -> Result<u64> {
    Ok(agreement_of(table()?) ^ u64::from(payload()? == Payload::Selected))
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

    /// 2026-10-15: The payload sub-lever: off spellings give the token rows, `1` the selected
    /// pools with a table, and `1` without a table or any other value is refused.
    #[test]
    fn the_payload_lever_needs_a_table_and_refuses_everything_else() {
        for on in [false, true] {
            assert_eq!(parse_payload(None, on), Ok(Payload::Tokens));
            assert_eq!(parse_payload(Some("0"), on), Ok(Payload::Tokens));
            for bad in ["", "2", "on", "true", " 1"] {
                assert!(parse_payload(Some(bad), on).is_err(), "{bad:?} on {on}");
            }
        }
        assert_eq!(parse_payload(Some("1"), true), Ok(Payload::Selected));
        assert!(parse_payload(Some("1"), false).is_err());
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
