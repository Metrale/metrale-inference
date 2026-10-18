// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: [`RunDumps`]: the diagnostic file sinks one scheduler run writes to.
//!
//! Serving builds it with [`RunDumps::open`] over the run's `RunIoSettings` (in the
//! scheduler core); `RunDumps::default()` opens nothing.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use std::fs::{File, OpenOptions};
use std::io::BufWriter;
use std::sync::Mutex;

use crate::scheduler::run_io::RunIoSettings;

/// 2026-09-25: Diagnostic file sinks for one scheduler run.
#[derive(Debug, Default)]
pub struct RunDumps {
    /// 2026-09-25: `METRALE_LOGIT_DUMP=path`: the JSONL file
    /// `logit_dump::record` appends to.
    pub logits: Option<Mutex<BufWriter<File>>>,
    /// 2026-09-25: `METRALE_ADADEC_DIAGNOSTIC=dir`: `dir/adadec_entropy.jsonl`,
    /// the adaptive-decode entropy trace.
    pub adadec: Option<Mutex<File>>,
    /// 2026-09-25: `METRALE_DUMP_LOGITS_PATH=dir`: raw FP32 logits rows are
    /// appended to `logits_seq.bin` and `logits_stok.bin` under it. `None` =
    /// no raw dump.
    pub raw_logits_dir: Option<std::path::PathBuf>,
}

impl RunDumps {
    /// 2026-09-25: Open whichever sinks this run's settings ask for (2026-10-17: the paths
    /// come from `RunIoSettings`, resolved at load, not from the environment).
    pub fn open(paths: &RunIoSettings) -> Self {
        Self {
            raw_logits_dir: paths
                .dump_logits_path
                .as_deref()
                .map(std::path::PathBuf::from),
            logits: Self::open_append("METRALE_LOGIT_DUMP", paths.logit_dump.as_deref(), |p| {
                p.to_path_buf()
            })
            .map(|f| Mutex::new(BufWriter::new(f))),
            adadec: Self::open_append(
                "METRALE_ADADEC_DIAGNOSTIC",
                paths.adadec_diagnostic.as_deref(),
                |p| p.join("adadec_entropy.jsonl"),
            )
            .map(Mutex::new),
        }
    }

    /// 2026-09-25: Run `to_file` over the path `raw` and open the result for append. `None`
    /// when `raw` is unset or empty; an open failure is logged under `var` and treated as
    /// unset, so a bad diagnostic path never fails a serve.
    fn open_append(
        var: &str,
        raw: Option<&str>,
        to_file: impl FnOnce(&std::path::Path) -> std::path::PathBuf,
    ) -> Option<File> {
        let raw = raw?;
        if raw.is_empty() {
            return None;
        }
        let base = std::path::Path::new(raw);
        let path = to_file(base);
        if let Some(dir) = path.parent()
            && dir != std::path::Path::new("")
            && raw == base.to_string_lossy()
            && path != base
        {
            let _ = std::fs::create_dir_all(dir);
        }
        match OpenOptions::new().create(true).append(true).open(&path) {
            Ok(f) => Some(f),
            Err(e) => {
                tracing::error!("{var}: cannot open {}: {e}", path.display());
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lever_equivalence::default_settings;

    fn paths(logit: Option<&str>, adadec: Option<&str>, raw: Option<&str>) -> RunIoSettings {
        RunIoSettings {
            logit_dump: logit.map(str::to_string),
            adadec_diagnostic: adadec.map(str::to_string),
            dump_logits_path: raw.map(str::to_string),
            ..RunIoSettings::from_settings(&default_settings()).expect("defaults resolve")
        }
    }

    #[test]
    fn default_settings_open_nothing() {
        let d = RunDumps::open(&paths(None, None, None));
        assert!(d.logits.is_none() && d.adadec.is_none() && d.raw_logits_dir.is_none());
    }

    #[test]
    fn each_path_opens_its_own_sink() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let logit = tmp.path().join("logits.jsonl");
        let adadec = tmp.path().join("not_yet");
        let (l, a) = (logit.to_str().unwrap(), adadec.to_str().unwrap());
        let d = RunDumps::open(&paths(Some(l), Some(a), Some("raw/dir")));
        assert!(d.logits.is_some() && logit.is_file());
        // 2026-10-17: The adadec directory is created, then the trace file under it.
        assert!(d.adadec.is_some() && adadec.join("adadec_entropy.jsonl").is_file());
        assert_eq!(
            d.raw_logits_dir.as_deref(),
            Some(std::path::Path::new("raw/dir"))
        );
    }

    #[test]
    fn an_empty_or_unopenable_path_opens_nothing_and_does_not_fail() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let dir = tmp.path().to_str().unwrap();
        let d = RunDumps::open(&paths(Some(""), Some(""), None));
        assert!(d.logits.is_none() && d.adadec.is_none());
        // 2026-10-17: A directory cannot be opened for append as the logit file.
        assert!(
            RunDumps::open(&paths(Some(dir), None, None))
                .logits
                .is_none()
        );
    }
}
