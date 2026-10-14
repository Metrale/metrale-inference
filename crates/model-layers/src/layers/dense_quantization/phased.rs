// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: A quantization flag's value per forward phase: `X` serves both phases at `X`;
//! `decode=X,prefill=Y` serves decode (one row per sequence, speculative verify) at `X` and the
//! prompt pass at `Y`. Pure: the tier names come from the caller, so `--dense-quantization` and
//! `--dsa-quantization` share one parser and one set of tests.
//!
//! Owner: model-layers (quantization dispatch).
//! Invariants:
//! - The plain form is exactly `decode=X,prefill=X`.
//! - Both phases are named exactly once; an unknown key, an unknown tier, a duplicate, a missing
//!   phase or surrounding whitespace is an error naming what was wrong, never a guess.

/// 2026-10-14: One tier per phase.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Phased<T> {
    pub decode: T,
    pub prefill: T,
}

impl<T: Copy + PartialEq> Phased<T> {
    /// 2026-10-14: The same tier in both phases (the plain flag form).
    pub fn both(t: T) -> Self {
        Self {
            decode: t,
            prefill: t,
        }
    }

    /// 2026-10-14: Whether the phases differ.
    pub fn is_split(&self) -> bool {
        self.decode != self.prefill
    }
}

/// 2026-10-14: The flag value back from its tiers' names: the plain form when the phases agree.
pub fn phased_name<T: Copy + PartialEq>(p: Phased<T>, name: impl Fn(T) -> &'static str) -> String {
    if p.is_split() {
        format!("decode={},prefill={}", name(p.decode), name(p.prefill))
    } else {
        name(p.decode).to_string()
    }
}

/// 2026-10-14: Parse `s` as `X` or `decode=X,prefill=Y` (either order), each tier through `one`.
/// `names` lists the accepted tiers for the error message.
pub fn parse_phased<T: Copy>(
    s: &str,
    one: impl Fn(&str) -> Option<T>,
    names: &str,
) -> Result<Phased<T>, String> {
    let tier = |v: &str| one(v).ok_or_else(|| format!("unknown tier {v:?} (expected {names})"));
    if !s.contains('=') && !s.contains(',') {
        let t = tier(s)?;
        return Ok(Phased {
            decode: t,
            prefill: t,
        });
    }
    let (mut decode, mut prefill) = (None, None);
    for part in s.split(',') {
        let Some((key, value)) = part.split_once('=') else {
            return Err(format!(
                "{part:?} is not key=value (expected `X` or `decode=X,prefill=Y`)"
            ));
        };
        let slot = match key {
            "decode" => &mut decode,
            "prefill" => &mut prefill,
            _ => {
                return Err(format!(
                    "unknown phase {key:?} (expected decode or prefill)"
                ));
            }
        };
        if slot.is_some() {
            return Err(format!("phase {key:?} given twice"));
        }
        *slot = Some(tier(value)?);
    }
    match (decode, prefill) {
        (Some(decode), Some(prefill)) => Ok(Phased { decode, prefill }),
        (None, _) => Err("missing decode=<tier>".to_string()),
        (_, None) => Err("missing prefill=<tier>".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NAMES: [&str; 3] = ["declared", "fp8", "w4a16"];

    fn one(s: &str) -> Option<usize> {
        NAMES.iter().position(|n| *n == s)
    }

    fn parse(s: &str) -> Result<Phased<usize>, String> {
        parse_phased(s, one, "declared, fp8, w4a16")
    }

    /// 2026-10-14: The plain form is both phases at the tier, the same value as its explicit
    /// spelling; the split form reads in either order and prints back canonically.
    #[test]
    fn plain_is_both_phases_and_split_round_trips() {
        for (i, n) in NAMES.iter().enumerate() {
            let p = parse(n).unwrap();
            assert_eq!(p, Phased::both(i));
            assert!(!p.is_split());
            assert_eq!(parse(&format!("decode={n},prefill={n}")).unwrap(), p);
            assert_eq!(phased_name(p, |t| NAMES[t]), *n);
        }
        let split = parse("decode=w4a16,prefill=fp8").unwrap();
        assert_eq!(
            split,
            Phased {
                decode: 2,
                prefill: 1
            }
        );
        assert!(split.is_split());
        assert_eq!(parse("prefill=fp8,decode=w4a16").unwrap(), split);
        let printed = phased_name(split, |t| NAMES[t]);
        assert_eq!(printed, "decode=w4a16,prefill=fp8");
        assert_eq!(parse(&printed).unwrap(), split);
    }

    /// 2026-10-14: Every malformed value is refused with the reason, never read as a nearby one.
    #[test]
    fn malformed_values_are_refused_with_the_reason() {
        for (s, why) in [
            ("w4a8", "unknown tier"),
            ("", "unknown tier"),
            ("decode=w4a16", "missing prefill"),
            ("prefill=fp8", "missing decode"),
            ("decode=w4a16,decode=fp8", "given twice"),
            ("decode=w4a16,prefill=fp8,prefill=fp8", "given twice"),
            ("decode=w4a16,verify=fp8", "unknown phase"),
            ("decode=w4a16,prefill=fp9", "unknown tier"),
            ("decode=w4a16,prefill", "not key=value"),
            ("decode=w4a16, prefill=fp8", "unknown phase"),
            ("decode=w4a16,prefill=fp8,", "not key=value"),
            (" fp8", "unknown tier"),
        ] {
            let e = parse(s).expect_err(s);
            assert!(e.contains(why), "{s:?}: {e}");
        }
    }
}
