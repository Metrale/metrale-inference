// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: The process lines of `GET /metrics` (investor-mvp #28, #37): which build
//! answers, when the process started (uptime = scrape time − start) and the startup phase
//! it last entered. Present at every telemetry level, since none of them needs a device.
//!
//! A fact the process does not have is left out, never written as 0: no start time before
//! `main` records one, no phase before the first, and an unstamped build is labelled
//! `commit="unstamped"`.
//!
//! Owner: telemetry.
//! Invariants: no I/O; one series per metric name.

use super::prometheus_text::{escape, head, one};

/// 2026-10-11: What the process knows about itself.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProcessFacts<'a> {
    pub version: &'a str,
    /// 2026-10-11: Full lower-case hex commit, or `None` for an unstamped build.
    pub commit: Option<&'a str>,
    pub start_unix_s: Option<f64>,
    pub phase: Option<u8>,
}

/// 2026-10-11: Render the process lines.
pub fn render(f: &ProcessFacts<'_>) -> String {
    let mut out = String::new();
    head(
        &mut out,
        "metrale_build_info",
        "gauge",
        "The build answering: version and commit (`unstamped` without METRALE_BUILD_COMMIT)",
    );
    out.push_str(&format!(
        "metrale_build_info{{version=\"{}\",commit=\"{}\"}} 1\n",
        escape(f.version),
        escape(f.commit.unwrap_or("unstamped")),
    ));
    if let Some(start) = f.start_unix_s {
        one(
            &mut out,
            "metrale_process_start_time_seconds",
            "gauge",
            "Start time of the process since the Unix epoch, in seconds",
            start,
        );
    }
    if let Some(phase) = f.phase {
        one(
            &mut out,
            "metrale_startup_phase",
            "gauge",
            "Index of the last startup phase entered (11 = listening)",
            phase,
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_facts_are_rendered() {
        let text = render(&ProcessFacts {
            version: "1.0.0",
            commit: Some("99b2613072a69a8708048347f1b74d9cc7880ed7"),
            start_unix_s: Some(1_760_000_000.5),
            phase: Some(11),
        });
        assert!(text.contains(
            "metrale_build_info{version=\"1.0.0\",commit=\"99b2613072a69a8708048347f1b74d9cc7880ed7\"} 1\n"
        ));
        assert!(text.contains("metrale_process_start_time_seconds 1760000000.5\n"));
        assert!(text.contains("metrale_startup_phase 11\n"));
    }

    #[test]
    fn unknown_facts_are_absent_not_zero() {
        let text = render(&ProcessFacts {
            version: "1.0.0",
            commit: None,
            start_unix_s: None,
            phase: None,
        });
        assert!(text.contains("commit=\"unstamped\""));
        assert!(!text.contains("metrale_process_start_time_seconds"));
        assert!(!text.contains("metrale_startup_phase"));
    }
}
