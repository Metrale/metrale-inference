// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: `recorded_at`: Unix seconds as RFC 3339 UTC (whole seconds,
//! `Z`), by the proleptic Gregorian civil-from-days conversion, and the
//! system clock behind the `Clock` port.

/// `unix_s` as `YYYY-MM-DDTHH:MM:SSZ`.
pub fn rfc3339_utc(unix_s: u64) -> String {
    let days = unix_s / 86_400;
    let secs = unix_s % 86_400;
    // Days since 0000-03-01, in 400-year eras of 146097 days.
    let z = days + 719_468;
    let era = z / 146_097;
    let doe = z % 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + u64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        secs / 3_600,
        secs % 3_600 / 60,
        secs % 60
    )
}

/// The system wall clock.
#[cfg(not(target_arch = "wasm32"))]
pub struct SystemClock;

#[cfg(not(target_arch = "wasm32"))]
impl crate::port::Clock for SystemClock {
    fn now_rfc3339(&self) -> crate::Result<String> {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| rfc3339_utc(d.as_secs()))
            .map_err(|e| crate::Error(format!("clock: {e}")))
    }
}
