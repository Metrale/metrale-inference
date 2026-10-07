// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-07: Bounded, opt-in synchronous-router diagnostic. It changes timing, never sampling.
//! Records device argmax candidates, not necessarily subsequently emitted tokens.

use std::collections::BTreeSet;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const REVISION: &str = "d32afde8b09af1539b49ff96ff5551c674485f8e";
const VOCAB: usize = 100352;
const MAX_RECORDS: usize = 64;
static CAPTURE: OnceLock<Mutex<Option<Capture>>> = OnceLock::new();

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Plan {
    pub output: PathBuf,
    pub model_revision: String,
    /// 2026-10-07: Operator assertion, not a runtime proof of source or checkpoint identity.
    pub asserted_source_commit: String,
    pub prompt_sha256: BTreeSet<String>,
    /// 2026-10-07: Number of generated tokens already processed, after this decode forward.
    pub generated_positions: BTreeSet<usize>,
    pub max_records: usize,
}

#[derive(Clone, Serialize)]
pub struct Row {
    pub slot: usize,
    /// 2026-10-07: Per-model allocation ticket, not HTTP identity across reallocation.
    pub allocation_generation: u64,
    pub seq_len: usize,
    pub prompt_len: usize,
    pub prompt_sha256: String,
    pub prefix_lookup_skip: bool,
}

pub fn prompt_digest(tokens: &[u32]) -> String {
    let mut hash = Sha256::new();
    for token in tokens {
        hash.update(token.to_le_bytes());
    }
    encode_hex(&hash.finalize())
}

fn encode_hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    bytes
        .iter()
        .flat_map(|b| {
            [
                char::from(DIGITS[(b >> 4) as usize]),
                char::from(DIGITS[(b & 15) as usize]),
            ]
        })
        .collect()
}

fn hex(value: &str, len: usize) -> bool {
    value.len() == len && value.bytes().all(|c| c.is_ascii_hexdigit())
}

impl Plan {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.model_revision == REVISION, "wrong checkpoint revision");
        ensure!(
            hex(&self.asserted_source_commit, 40),
            "source must be full commit"
        );
        ensure!(self.output.is_absolute(), "output must be absolute");
        ensure!(
            (1..=MAX_RECORDS).contains(&self.max_records),
            "record cap 1..64"
        );
        ensure!(
            (1..=8).contains(&self.prompt_sha256.len()),
            "allow 1..8 fixtures"
        );
        ensure!(
            self.prompt_sha256.iter().all(|s| hex(s, 64)),
            "invalid prompt SHA256"
        );
        ensure!(
            !self.generated_positions.is_empty()
                && self.generated_positions.len() <= 16
                && self
                    .generated_positions
                    .iter()
                    .all(|n| (1..=128).contains(n)),
            "positions 1..128, at most16"
        );
        Ok(())
    }
}

pub struct Capture {
    plan: Plan,
    written: usize,
    failed: bool,
}

impl Capture {
    pub fn create(plan: Plan) -> Result<Self> {
        plan.validate()?;
        let mut builder = fs::DirBuilder::new();
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            builder.mode(0o700);
        }
        builder
            .create(&plan.output)
            .context("capture output must be new")?;
        Ok(Self {
            plan,
            written: 0,
            failed: false,
        })
    }

    /// 2026-10-07: All live rows must be allowlisted. No callback (and therefore no GPU copy)
    /// occurs for an unrelated batch, an unselected position, or an exhausted cap.
    pub fn record(
        &mut self,
        ticket: u32,
        rows: &[Row],
        selected: Option<&[u32]>,
        vocab: usize,
        fp32: bool,
        read: impl FnOnce(&mut [u8]) -> Result<()>,
    ) -> Result<bool> {
        ensure!(
            !self.failed,
            "capture failed previously; restart the diagnostic process"
        );
        let result = self.record_inner(ticket, rows, selected, vocab, fp32, read);
        if result.is_err() {
            self.failed = true;
        }
        result
    }

    fn record_inner(
        &mut self,
        ticket: u32,
        rows: &[Row],
        selected: Option<&[u32]>,
        vocab: usize,
        fp32: bool,
        read: impl FnOnce(&mut [u8]) -> Result<()>,
    ) -> Result<bool> {
        if self.written == self.plan.max_records || rows.is_empty() {
            return Ok(false);
        }
        if rows
            .iter()
            .any(|r| !self.plan.prompt_sha256.contains(&r.prompt_sha256))
        {
            return Ok(false);
        }
        ensure!(
            rows.len() <= 4 && vocab == VOCAB && !fp32,
            "only <=4 rows of Laguna BF16 logits"
        );
        ensure!(
            rows.iter()
                .all(|r| r.prompt_len > 0 && r.prompt_len <= 4096 && r.seq_len > r.prompt_len),
            "invalid sequence prefix"
        );
        if !rows.iter().any(|r| {
            self.plan
                .generated_positions
                .contains(&(r.seq_len - r.prompt_len))
        }) {
            return Ok(false);
        }
        let ids = selected.context("diagnostic refuses host/masked sampling")?;
        ensure!(
            ids.len() == rows.len() && ids.iter().all(|id| (*id as usize) < vocab),
            "invalid device IDs"
        );
        ensure!(
            rows.iter().map(|r| r.slot).collect::<BTreeSet<_>>().len() == rows.len(),
            "duplicate physical slot"
        );
        ensure!(
            rows.iter()
                .all(|r| r.slot != usize::MAX && r.allocation_generation != 0),
            "capture requires allocated live sequence identities"
        );
        ensure!(
            rows.iter()
                .map(|r| r.allocation_generation)
                .collect::<BTreeSet<_>>()
                .len()
                == rows.len(),
            "duplicate allocation generation"
        );
        let mut bytes = vec![0; rows.len() * vocab * 2];
        read(&mut bytes)?;
        let stem = format!("{:03}-ticket-{ticket}", self.written);
        let digest = encode_hex(&Sha256::digest(&bytes));
        let mut raw = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(self.plan.output.join(format!("{stem}.bf16")))?;
        raw.write_all(&bytes)?;
        raw.sync_all()?;
        let receipt = serde_json::json!({
            "schema":1, "route":"sync_decode_batch/native_argmax/post_argmax_copy",
            "timing_intervention":true, "device_candidates_not_final_tokens":true,
            "ticket":ticket, "rows":rows,
            "identity_scope":"model allocation within this process; not cross-reallocation HTTP identity", "selected_ids":ids, "vocab":vocab,
            "dtype":"BF16-little-endian", "logits_sha256":digest,
            "model_revision_asserted":self.plan.model_revision,
            "source_commit_asserted":self.plan.asserted_source_commit,
            "generated_positions_requested":self.plan.generated_positions,
            "padded_rows":"not inferred; receipt contains only live router rows"
        });
        let mut output = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(self.plan.output.join(format!("{stem}.json")))?;
        serde_json::to_writer_pretty(&mut output, &receipt)?;
        output.sync_all()?;
        self.written += 1;
        Ok(true)
    }
}

/// 2026-10-07: Read at most32KiB plus one refusal byte, even if a file grows after metadata.
pub fn read_plan(reader: impl Read) -> Result<Plan> {
    let mut bytes = Vec::with_capacity(32769);
    reader.take(32769).read_to_end(&mut bytes)?;
    ensure!(bytes.len() <= 32768, "capture plan too large");
    let plan: Plan = serde_json::from_slice(&bytes)?;
    plan.validate()?;
    Ok(plan)
}

/// 2026-10-07: Refuse devices/FIFOs and open nonblocking before descriptor metadata validation.
pub fn read_plan_file(path: &Path) -> Result<Plan> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NONBLOCK);
    }
    let file = options.open(path)?;
    ensure!(
        file.metadata()?.is_file(),
        "capture plan must be a regular file"
    );
    read_plan(file)
}

/// 2026-10-07: Binary identity hashing uses fixed memory, independent of executable size.
pub fn hash_reader(mut reader: impl Read) -> Result<String> {
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(encode_hex(&hash.finalize()))
}

pub fn validate_mode(
    sync: bool,
    ranks: usize,
    batch: usize,
    spec: bool,
    codispatch: bool,
    no_mix: bool,
) -> Result<()> {
    ensure!(
        sync && ranks == 1 && (1..=4).contains(&batch) && !spec && !codispatch && no_mix,
        "Laguna capture requires explicit sync, single rank, batch1..4, no speculation/codispatch, METRALE_BISECT_NO_MIX=1 plus METRALE_BISECT_Q12_DISABLE=1"
    );
    Ok(())
}

/// 2026-10-07: Called before scheduler startup. Unsupported modes are errors, never silent fallbacks.
pub fn initialize(
    sync: bool,
    ranks: usize,
    batch: usize,
    spec: bool,
    codispatch: bool,
) -> Result<()> {
    let Some(path) = std::env::var_os("METRALE_LAGUNA_CAPTURE_PLAN") else {
        return Ok(());
    };
    validate_mode(
        sync,
        ranks,
        batch,
        spec,
        codispatch,
        matches!(
            std::env::var("METRALE_BISECT_NO_MIX").as_deref(),
            Ok("1" | "true")
        ) && matches!(
            std::env::var("METRALE_BISECT_Q12_DISABLE").as_deref(),
            Ok("1" | "true")
        ),
    )?;
    let plan = read_plan_file(Path::new(&path))?;
    let capture = Capture::create(plan)?;
    let binary_sha256 = hash_reader(File::open(std::env::current_exe()?)?)?;
    fs::write(capture.plan.output.join("binary-sha256.txt"), binary_sha256)?;
    CAPTURE
        .set(Mutex::new(Some(capture)))
        .map_err(|_| anyhow::anyhow!("capture already initialized"))?;
    Ok(())
}

pub fn with_capture(f: impl FnOnce(&mut Capture) -> Result<()>) -> Result<()> {
    let Some(lock) = CAPTURE.get() else {
        return Ok(());
    };
    let mut guard = lock
        .lock()
        .map_err(|_| anyhow::anyhow!("capture poisoned"))?;
    if let Some(capture) = guard.as_mut() {
        f(capture)?;
    }
    Ok(())
}
