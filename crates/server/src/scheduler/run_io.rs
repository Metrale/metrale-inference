// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: [`RunIoSettings`]: the stream-send deadline and the diagnostic sink paths one
//! scheduler run is given. Resolved once from the load's settings (ROTI) and carried in
//! `SchedulerConfig`, so the scheduler thread reads no environment (SBIO).
//!
//! Owner: scheduler.
//! Invariants: the paths are the settings' values verbatim; the deadline is the declared
//! default only when `METRALE_STREAM_SEND_DEADLINE_MS` is unset.

use std::time::Duration;

use metrale_settings::InferenceServerSettings;

use crate::lever_value::text_int;

/// 2026-10-17: The deadline when `METRALE_STREAM_SEND_DEADLINE_MS` is unset. A default
/// exists because an unbounded send would never time out.
pub const STREAM_SEND_DEADLINE_MS: u64 = 5000;

/// 2026-10-17: What the scheduler's transport and diagnostic sinks are configured with.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunIoSettings {
    /// 2026-10-17: How long a scheduler-thread send waits on a full stream channel
    /// (`mod_helpers::bounded_stream_send`, `spawn_terminal_send`).
    pub stream_send_deadline: Duration,
    /// 2026-10-17: `METRALE_LOGIT_DUMP`: the JSONL file `logit_dump::record` appends to.
    pub logit_dump: Option<String>,
    /// 2026-10-17: `METRALE_ADADEC_DIAGNOSTIC`: the directory of the entropy trace.
    pub adadec_diagnostic: Option<String>,
    /// 2026-10-17: `METRALE_DUMP_LOGITS_PATH`: the directory of the raw FP32 logits rows.
    pub dump_logits_path: Option<String>,
}

impl RunIoSettings {
    /// 2026-10-17: This run's values, read from `settings`.
    ///
    /// # Errors
    /// `METRALE_STREAM_SEND_DEADLINE_MS` set to a value that is not a non-negative integer.
    pub fn from_settings(settings: &InferenceServerSettings) -> anyhow::Result<Self> {
        let telemetry = &settings.telemetry;
        let deadline_ms = text_int::<u64>(
            "METRALE_STREAM_SEND_DEADLINE_MS",
            settings.runtime.stream_send_deadline_ms.as_deref(),
        )?
        .unwrap_or(STREAM_SEND_DEADLINE_MS);
        Ok(Self {
            stream_send_deadline: Duration::from_millis(deadline_ms),
            logit_dump: telemetry.logit_dump.clone(),
            adadec_diagnostic: telemetry.adadec_diagnostic.clone(),
            dump_logits_path: telemetry.dump_logits_path.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    //! 2026-10-17: [`RunIoSettings::from_settings`] against the readers it replaced
    //! (`send::stream_send_deadline`, `RunDumps::from_env`), resolved over `MapEnv`.

    use std::borrow::Cow;

    use super::{RunIoSettings, STREAM_SEND_DEADLINE_MS};
    use crate::lever_equivalence::{Old, Out, Row, assert_old_readers_hold, num, settings_from};

    fn resolved(env: &[(&str, &str)]) -> anyhow::Result<RunIoSettings> {
        RunIoSettings::from_settings(&settings_from(env)?)
    }

    #[test]
    fn an_empty_environment_gives_the_default_deadline_and_no_sinks() {
        let r = resolved(&[]).expect("defaults resolve");
        assert_eq!(
            r.stream_send_deadline.as_millis(),
            u128::from(STREAM_SEND_DEADLINE_MS)
        );
        assert_eq!(
            (r.logit_dump, r.adadec_diagnostic, r.dump_logits_path),
            (None, None, None)
        );
    }

    #[test]
    fn the_sink_paths_are_kept_verbatim() {
        for v in ["/srv/dumps/logits.jsonl", " ", "relative/dir"] {
            let r = resolved(&[
                ("METRALE_LOGIT_DUMP", v),
                ("METRALE_ADADEC_DIAGNOSTIC", v),
                ("METRALE_DUMP_LOGITS_PATH", v),
            ])
            .expect("any text resolves");
            assert_eq!(r.logit_dump.as_deref(), Some(v));
            assert_eq!(r.adadec_diagnostic.as_deref(), Some(v));
            assert_eq!(r.dump_logits_path.as_deref(), Some(v));
        }
    }

    fn rows() -> Vec<Row<RunIoSettings>> {
        vec![(
            "METRALE_STREAM_SEND_DEADLINE_MS",
            Old::NumUint(STREAM_SEND_DEADLINE_MS),
            |r| num(r.stream_send_deadline.as_millis()),
        )]
    }

    /// 2026-10-17: The deadline was `parse().ok()` or 5000 ms; it now uses the settings
    /// crate's integer grammar, so it is trimmed and anything that is not a non-negative
    /// integer stops the load instead of silently becoming 5000 ms.
    const CHANGES: &[(&str, &str, Out)] = &[
        ("uint", "true", Out::Err),
        ("uint", "TRUE", Out::Err),
        ("uint", "yes", Out::Err),
        ("uint", "false", Out::Err),
        ("uint", "off", Out::Err),
        ("uint", "no", Out::Err),
        ("uint", "", Out::Err),
        ("uint", " 1 ", Out::N(Cow::Borrowed("1"))),
        ("uint", "garbage", Out::Err),
        ("uint", "-1", Out::Err),
    ];

    #[test]
    fn the_deadline_matches_the_old_reader_except_the_recorded_changes() {
        assert_old_readers_hold(&rows(), resolved, CHANGES);
    }
}
