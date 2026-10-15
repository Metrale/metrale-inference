// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: `recorded_at` against known instants, across the leap rules
//! (divisible by 4, not by 100, unless by 400) and day and year ends.

use metprof_kb::time::rfc3339_utc;

#[test]
fn unix_seconds_render_as_rfc3339_utc() {
    for (unix_s, want) in [
        (0, "1970-01-01T00:00:00Z"),
        (951_782_400, "2000-02-29T00:00:00Z"),
        (1_709_251_199, "2024-02-29T23:59:59Z"),
        (1_767_225_599, "2025-12-31T23:59:59Z"),
        (4_107_542_399, "2100-02-28T23:59:59Z"),
        (4_107_542_400, "2100-03-01T00:00:00Z"),
    ] {
        assert_eq!(rfc3339_utc(unix_s), want, "{unix_s}");
    }
}
