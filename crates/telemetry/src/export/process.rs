// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: The process lines of `GET /metrics` (investor-mvp #28, #37): which build
//! answers, when the process started (uptime = scrape time − start) and the startup phase
//! it last entered. Present at every telemetry level, since none of them needs a device.
//!
//! A fact the process does not have is left out, never written as 0: no start time before
//! `main` records one, no phase before the first, no model line before a model is loaded,
//! no revision line for a checkpoint without a known revision, no recipe line without a
//! launch recipe, and an unstamped build is labelled `commit="unstamped"`.
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
    /// 2026-10-11: The loaded model, when one is.
    pub model: Option<ModelFacts<'a>>,
}

/// 2026-10-11: What identifies the loaded model beyond its served name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelFacts<'a> {
    pub served_model: &'a str,
    /// 2026-10-11: The checkpoint's 40-hex revision, when the loader knows it.
    pub revision: Option<&'a str>,
    /// 2026-10-11: SHA-256 of the launch recipe's canonical JSON, when one was given.
    pub recipe: Option<&'a str>,
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
    if let Some(m) = &f.model {
        for (name, help, label, value) in [
            (
                "metrale_model_revision_info",
                "The loaded checkpoint's revision (a full commit)",
                "revision",
                m.revision,
            ),
            (
                "metrale_model_recipe_info",
                "SHA-256 of the canonical JSON of the ModelRecipe this process was launched from",
                "recipe",
                m.recipe,
            ),
        ] {
            if let Some(value) = value {
                head(&mut out, name, "gauge", help);
                out.push_str(&format!(
                    "{name}{{model=\"{}\",{label}=\"{}\"}} 1\n",
                    escape(m.served_model),
                    escape(value),
                ));
            }
        }
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
            model: Some(ModelFacts {
                served_model: "openai/gpt-oss-20b",
                revision: Some("ad31ec163396d5cb44d8deb87c596cecb0f55bf8"),
                recipe: Some("17"),
            }),
        });
        assert!(text.contains(
            "metrale_build_info{version=\"1.0.0\",commit=\"99b2613072a69a8708048347f1b74d9cc7880ed7\"} 1\n"
        ));
        assert!(text.contains("metrale_process_start_time_seconds 1760000000.5\n"));
        assert!(text.contains("metrale_startup_phase 11\n"));
        assert!(text.contains(
            "metrale_model_revision_info{model=\"openai/gpt-oss-20b\",revision=\"ad31ec163396d5cb44d8deb87c596cecb0f55bf8\"} 1\n"
        ));
        assert!(
            text.contains(
                "metrale_model_recipe_info{model=\"openai/gpt-oss-20b\",recipe=\"17\"} 1\n"
            )
        );
    }

    #[test]
    fn unknown_facts_are_absent_not_zero() {
        let text = render(&ProcessFacts {
            version: "1.0.0",
            commit: None,
            start_unix_s: None,
            phase: None,
            model: Some(ModelFacts {
                served_model: "m",
                revision: None,
                recipe: None,
            }),
        });
        assert!(text.contains("commit=\"unstamped\""));
        assert!(!text.contains("metrale_model_"));
        assert!(!text.contains("metrale_process_start_time_seconds"));
        assert!(!text.contains("metrale_startup_phase"));
    }
}
