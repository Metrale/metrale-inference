// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: the fake KB behind `Mem` (the `KbOpen` and `Clock` ports):
//! the kb.toml every `Mem::run` records through, and what the sink does
//! with a run.

use metprof_kb::config::EmbeddingConfig;
use metprof_kb::kb_toml::KbConfig;
use metprof_kb::port::{Clock, Kb, KbOpen, KbReport, KbSink, Recorded, RunRecord, Verify};

use super::{Capture, Mem};

/// The kb.toml every `Mem::run` records through unless argv names one.
pub const KB_TOML: &str = "/kb/kb.toml";
pub const EMBEDDING_TOML: &str = "/kb/embedding.toml";
/// The fake clock's one instant.
pub const NOW: &str = "2026-01-01T00:00:00Z";

pub fn kb_toml(seal: &str) -> String {
    format!(
        "schema = 1\nspool_dir = \"/kb/spool\"\ndurable_dir = \"/kb/durable\"\n\
         export_dir = \"/kb/export\"\n{seal}\nembedding = \"{EMBEDDING_TOML}\"\n\
         [identity]\nbox_alias = \"box-a\"\ncampaign = \"campaign-a\"\n\
         hardware_class = \"class-a\"\n"
    )
}

pub const HASHED: &str = "[embedder]\nkind = \"hashed\"\n[reranker]\nkind = \"none\"\n";

/// What the fake KB sink does with a run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sink {
    /// Spooled and materialised.
    Materialises,
    /// Spooled only (another writer holds the lease).
    SpoolsOnly,
    /// Not spooled: the write fails.
    Fails,
}

/// The fake KB: it keeps the runs in the `Mem` and has no maintenance.
struct FakeKb<'a>(&'a Mem);

impl KbSink for FakeKb<'_> {
    fn record(&self, run: &RunRecord) -> metprof_kb::Result<Recorded> {
        run.check()?;
        let spool_path = "/kb/spool/run.json".to_string();
        match self.0.sink {
            Sink::Fails => return Err(metprof_kb::Error("disk full (synthetic)".into())),
            Sink::Materialises | Sink::SpoolsOnly => self.0.records.borrow_mut().push(run.clone()),
        }
        Ok(match self.0.sink {
            Sink::SpoolsOnly => Recorded {
                spool_path,
                materialized: false,
                reason: Some("another writer holds the lease (synthetic)".into()),
            },
            _ => Recorded {
                spool_path,
                materialized: true,
                reason: None,
            },
        })
    }
}

impl Kb for FakeKb<'_> {
    fn selfcheck(&self, _: &RunRecord) -> metprof_kb::Result<KbReport> {
        Err(metprof_kb::Error("the fake KB has no maintenance".into()))
    }
    fn replay(&self) -> metprof_kb::Result<KbReport> {
        Err(metprof_kb::Error("the fake KB has no maintenance".into()))
    }
    fn export(&self) -> metprof_kb::Result<KbReport> {
        Err(metprof_kb::Error("the fake KB has no maintenance".into()))
    }
    fn seal(&self) -> metprof_kb::Result<KbReport> {
        Err(metprof_kb::Error("the fake KB has no maintenance".into()))
    }
    fn verify(&self) -> metprof_kb::Result<Verify> {
        Err(metprof_kb::Error("the fake KB has no maintenance".into()))
    }
}

impl KbOpen for Mem {
    fn open<'a>(
        &'a self,
        _: &KbConfig,
        _: &EmbeddingConfig,
    ) -> metprof_kb::Result<Box<dyn Kb + 'a>> {
        Ok(Box::new(FakeKb(self)))
    }
}

impl Clock for Mem {
    fn now_rfc3339(&self) -> metprof_kb::Result<String> {
        Ok(NOW.to_string())
    }
}

/// The command the KB tests record: a summary of `capture()`.
pub const SUMMARIZE: [&str; 10] = [
    "summarize",
    "c.sqlite",
    "--device",
    "0",
    "--families",
    "f.toml",
    "--gap-ms",
    "1",
    "--top",
    "2",
];

/// A two-kernel capture.
pub fn capture() -> Capture {
    Capture::new()
        .kernel("gemm_a", 0.0, 100.0)
        .kernel("norm_x", 3000.0, 3010.0)
}
