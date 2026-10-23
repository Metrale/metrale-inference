// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The trace text refuses what it cannot represent exactly.

use alloc::format;
use alloc::string::ToString;

use metrale_error::ErrorCode;

use crate::synth_tests::BASE;
use crate::trace::{HEADER, PowerTrace};

#[test]
fn a_missing_header_unknown_record_or_bad_number_is_refused_with_its_line() {
    assert_eq!(
        PowerTrace::decode("meta,a,b").unwrap_err().code(),
        ErrorCode::PowerTraceFormat
    );
    let e = PowerTrace::decode(&format!("{HEADER}\nbogus,1")).unwrap_err();
    assert!(e.to_string().contains("line 2"), "{e}");
    let e = PowerTrace::decode(&format!("{HEADER}\nlaunches,x,1")).unwrap_err();
    assert_eq!(e.code(), ErrorCode::PowerTraceFormat);
}

#[test]
fn a_sample_for_an_undeclared_field_is_refused_on_decode() {
    let text = BASE.record().encode();
    let bad = text.replacen(";sm=2435", ";sm=2435;tm=40.0", 1);
    assert_ne!(bad, text);
    assert_eq!(
        PowerTrace::decode(&bad).unwrap_err().code(),
        ErrorCode::PowerFieldUndeclared
    );
}

#[test]
fn an_idle_window_with_launches_is_refused() {
    let text = format!("{HEADER}\nwindow,idle,0,0,1,2,5");
    assert_eq!(
        PowerTrace::decode(&text).unwrap_err().code(),
        ErrorCode::PowerTraceFormat
    );
}

/// 2026-10-22: Found by the first live run (spark-28c2): nvidia-smi's reason for its missing
/// instant power reads "…moving average, not an instant reading", and the comma split the
/// declaration, so a trace the harness wrote could not be recomputed.
#[test]
fn an_unavailable_reason_with_commas_round_trips() {
    let text = BASE.record().encode();
    let at = text.find("\nfield,").expect("BASE declares a field");
    let id = text[at + 7..].split(',').next().unwrap().to_string();
    let reason = "the driver's moving average, not an instant reading, by design";
    let line = format!("\nunavailable,{id},mem_clock,{reason}");
    let with = format!("{}{line}{}", &text[..at], &text[at..]);
    let t = PowerTrace::decode(&with).unwrap();
    let again = t.encode();
    assert!(again.contains(&line), "{again}");
    assert_eq!(PowerTrace::decode(&again).unwrap().encode(), again);
}
