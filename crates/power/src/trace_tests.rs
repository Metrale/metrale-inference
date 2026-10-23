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
