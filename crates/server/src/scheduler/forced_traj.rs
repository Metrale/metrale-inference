// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The forced-trajectory mode of the paired A/B instrument
//! (`METRALE_PAIRED_AB_TRAJ`): a serve either records the token trajectory of its single-sequence
//! MTP steps (`record`), or replays one recorded by another serve (`replay:<rank-0 log>`), so two
//! serves whose levers change the numerics (a `bounded` lever) still run the SAME token sequence
//! through the SAME steps at the SAME KV lengths, and their step-time difference measures only
//! the kernels (`met benchmark forced-ab`).
//!
//! Replay of one K=2 verify: the verify's row 1 is the recorded draft (not this serve's own), and
//! the verdict, the emitted tokens and so the KV growth are the recorded ones. The verify, the
//! propose and every graph still run at the served shapes. What this serve's numerics WOULD have
//! done is logged beside it (`PAIRED_TRAJ replay` lines): its own draft against its own pick at
//! row 0 (its acceptance on this trajectory) and whether its picks agree with the record. A
//! bootstrap step (one decode, then a propose) replays its recorded pick the same way.
//!
//! Owner: scheduler (instrument; model-neutral).
//! Invariants:
//! - Off unless the setting is present; resolved once at serve start (the replay file is read
//!   there, by the host, never here).
//! - A replayed request follows its record step by step: each step's kind, output index and KV
//!   length must be the record's next step, or the step fails with a desync error (never a
//!   silent partial force). The token a step FEEDS (`last`) is never in the KV before that step,
//!   so a differing one (the prefill's first token) is replaced, and counted, not refused.
//! - A request is keyed by an FNV-1a hash of its prompt tokens; a key the record lacks is refused.

use std::cell::RefCell;
use std::collections::HashMap;

use anyhow::{Context, Result, bail};

/// 2026-10-23: What one recorded step did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecKind {
    /// 2026-10-23: A K=2 verify of `[last, draft]`: accepted, and the tokens it emitted (`e1`
    /// only when accepted).
    K2 {
        draft: u32,
        acc: bool,
        e0: u32,
        e1: u32,
    },
    /// 2026-10-23: A bootstrap decode of `last` that picked `tok`.
    Boot { tok: u32 },
}

/// 2026-10-23: One recorded step: the output length and KV length before it, the token it fed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rec {
    pub out: usize,
    pub kv: usize,
    pub last: u32,
    pub kind: RecKind,
}

/// 2026-10-23: A recorded serve: each request's steps in order, by prompt key.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Trajectory {
    pub reqs: HashMap<u64, Vec<Rec>>,
}

/// 2026-10-23: The value of `key=` in `line` (after the marker).
fn field<'a>(rest: &'a str, k: &str, line: &str) -> Result<&'a str> {
    rest.split_whitespace()
        .find_map(|kv| kv.strip_prefix(k).and_then(|v| v.strip_prefix('=')))
        .with_context(|| format!("PAIRED_TRAJ line without `{k}=`: {line}"))
}

impl Trajectory {
    /// 2026-10-23: The `PAIRED_TRAJ rec` lines of a recording serve's log, in log order. Other
    /// lines are ignored; a malformed record line is an error.
    pub fn parse(text: &str) -> Result<Self> {
        let mut t = Self::default();
        for line in text.lines() {
            let Some(at) = line.find("PAIRED_TRAJ rec ") else {
                continue;
            };
            let rest = &line[at + "PAIRED_TRAJ rec ".len()..];
            let f = |k: &str| field(rest, k, line);
            let u = |k: &str| -> Result<u32> { Ok(f(k)?.parse()?) };
            let key = u64::from_str_radix(f("key")?, 16)?;
            let kind = if rest.starts_with("k2 ") {
                RecKind::K2 {
                    draft: u("draft")?,
                    acc: f("acc")? == "1",
                    e0: u("e0")?,
                    e1: u("e1")?,
                }
            } else if rest.starts_with("boot ") {
                RecKind::Boot { tok: u("tok")? }
            } else {
                bail!("PAIRED_TRAJ rec line of no known kind: {line}");
            };
            t.reqs.entry(key).or_default().push(Rec {
                out: f("out")?.parse()?,
                kv: f("kv")?.parse()?,
                last: u("last")?,
                kind,
            });
        }
        if t.reqs.is_empty() {
            bail!("no `PAIRED_TRAJ rec` line: the log is not a recording serve's");
        }
        Ok(t)
    }
}

/// 2026-10-23: The `PAIRED_TRAJ rec` line of `rec` (what [`Trajectory::parse`] reads).
pub fn rec_line(key: u64, rec: &Rec) -> String {
    let head = format!(
        "key={key:016x} out={} kv={} last={}",
        rec.out, rec.kv, rec.last
    );
    match rec.kind {
        RecKind::K2 { draft, acc, e0, e1 } => format!(
            "PAIRED_TRAJ rec k2 {head} draft={draft} acc={} e0={e0} e1={e1}",
            u8::from(acc)
        ),
        RecKind::Boot { tok } => format!("PAIRED_TRAJ rec boot {head} tok={tok}"),
    }
}

/// 2026-10-23: FNV-1a over the prompt tokens (little-endian bytes): a request's key.
pub fn prompt_key(prompt: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for t in prompt {
        for b in t.to_le_bytes() {
            h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
        }
    }
    h
}

/// 2026-10-23: Record or replay.
#[derive(Debug)]
pub enum TrajMode {
    Record,
    Replay(Trajectory),
}

/// 2026-10-23: One step's coordinates: the request key, its output length and KV length before
/// the step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct At {
    pub key: u64,
    pub out: usize,
    pub kv: usize,
}

/// 2026-10-23: The running mode of one serve.
#[derive(Debug)]
pub struct ForcedTraj {
    mode: TrajMode,
    /// 2026-10-23: Per replayed request, the index of its next recorded step.
    cursor: RefCell<HashMap<u64, usize>>,
    /// 2026-10-23: Prompt keys by (slot, request start), so a request is hashed once.
    keys: RefCell<HashMap<(usize, std::time::Instant), u64>>,
}

impl ForcedTraj {
    pub fn new(mode: TrajMode) -> Self {
        Self {
            mode,
            cursor: RefCell::default(),
            keys: RefCell::default(),
        }
    }

    pub fn replaying(&self) -> bool {
        matches!(self.mode, TrajMode::Replay(_))
    }

    /// 2026-10-23: The key of the request in `slot` that started at `start`, hashing `prompt`
    /// only the first time.
    pub fn key(&self, slot: usize, start: std::time::Instant, prompt: &[u32]) -> u64 {
        *self
            .keys
            .borrow_mut()
            .entry((slot, start))
            .or_insert_with(|| prompt_key(prompt))
    }

    /// 2026-10-23: The recorded step a replayed step at `at` must be (the invariant), not yet
    /// consumed.
    fn next(&self, at: At, k2: bool) -> Result<Rec> {
        let TrajMode::Replay(t) = &self.mode else {
            bail!("forced trajectory: not replaying");
        };
        let steps = t
            .reqs
            .get(&at.key)
            .with_context(|| format!("forced trajectory: no recorded request {:016x}", at.key))?;
        let i = *self.cursor.borrow().get(&at.key).unwrap_or(&0);
        let rec = steps.get(i).with_context(|| {
            format!(
                "forced trajectory desync: request {:016x} ran past its {} recorded steps",
                at.key,
                steps.len()
            )
        })?;
        let kind_ok = matches!(rec.kind, RecKind::K2 { .. }) == k2;
        if !kind_ok || rec.out != at.out || rec.kv != at.kv {
            bail!(
                "forced trajectory desync: request {:016x} step {i}: this serve runs a {} at \
                 out {} kv {}, the record a {:?} at out {} kv {}",
                at.key,
                if k2 { "K=2 verify" } else { "bootstrap" },
                at.out,
                at.kv,
                rec.kind,
                rec.out,
                rec.kv
            );
        }
        Ok(*rec)
    }

    fn advance(&self, key: u64) {
        *self.cursor.borrow_mut().entry(key).or_insert(0) += 1;
    }

    /// 2026-10-23: Before a step feeds `last`: under replay, `last` becomes the recorded one
    /// (returns whether it changed). Recording changes nothing.
    pub fn feed(&self, at: At, k2: bool, last: &mut u32) -> Result<bool> {
        if !self.replaying() {
            return Ok(false);
        }
        let rec = self.next(at, k2)?;
        let repaired = *last != rec.last;
        *last = rec.last;
        Ok(repaired)
    }

    /// 2026-10-23: The draft a K=2 verify at `at` verifies: the recorded one under replay, else
    /// `own`.
    pub fn k2_draft(&self, at: At, own: u32) -> Result<u32> {
        if !self.replaying() {
            return Ok(own);
        }
        match self.next(at, true)?.kind {
            RecKind::K2 { draft, .. } => Ok(draft),
            RecKind::Boot { .. } => unreachable!("next() checked the kind"),
        }
    }

    /// 2026-10-23: After a K=2 verify at `at` that fed `last` and `draft` and picked `picks`
    /// (`own` = the draft this serve proposed): the `(draft, v0, v1)` the step goes on with, so
    /// that `accepted = draft == v0`. Recording logs the step and returns its own values;
    /// replay logs what this serve would have done and returns the record's.
    pub fn k2_verdict(
        &self,
        at: At,
        last: u32,
        draft: u32,
        own: u32,
        picks: [u32; 2],
    ) -> Result<[u32; 3]> {
        let [v0, v1] = picks;
        let TrajMode::Replay(_) = &self.mode else {
            let rec = Rec {
                out: at.out,
                kv: at.kv,
                last,
                kind: RecKind::K2 {
                    draft,
                    acc: draft == v0,
                    e0: v0,
                    e1: v1,
                },
            };
            tracing::info!("{}", rec_line(at.key, &rec));
            return Ok([draft, v0, v1]);
        };
        let rec = self.next(at, true)?;
        let RecKind::K2 {
            draft: rd,
            acc,
            e0,
            e1,
        } = rec.kind
        else {
            unreachable!("next() checked the kind");
        };
        if draft != rd || last != rec.last {
            bail!(
                "forced trajectory: the verify fed ({last}, {draft}), the record ({}, {rd})",
                rec.last
            );
        }
        tracing::info!(
            "PAIRED_TRAJ replay k2 key={:016x} out={} kv={} own_draft={own} v0={v0} v1={v1} \
             forced_acc={} own_acc={} agree0={} agree1={}",
            at.key,
            at.out,
            at.kv,
            u8::from(acc),
            u8::from(own == v0),
            u8::from(v0 == e0),
            if acc {
                (v1 == e1).to_string()
            } else {
                "-".into()
            }
        );
        self.advance(at.key);
        Ok(if acc { [rd, rd, e1] } else { [rd, e0, v1] })
    }

    /// 2026-10-23: After a bootstrap decode at `at` of `last` picked `natural`: the token the
    /// step goes on with (recorded, or the record's under replay).
    pub fn boot_pick(&self, at: At, last: u32, natural: u32) -> Result<u32> {
        let TrajMode::Replay(_) = &self.mode else {
            let rec = Rec {
                out: at.out,
                kv: at.kv,
                last,
                kind: RecKind::Boot { tok: natural },
            };
            tracing::info!("{}", rec_line(at.key, &rec));
            return Ok(natural);
        };
        let rec = self.next(at, false)?;
        let RecKind::Boot { tok } = rec.kind else {
            unreachable!("next() checked the kind");
        };
        tracing::info!(
            "PAIRED_TRAJ replay boot key={:016x} out={} kv={} tok={tok} natural={natural} agree={}",
            at.key,
            at.out,
            at.kv,
            u8::from(natural == tok)
        );
        self.advance(at.key);
        Ok(tok)
    }
}

#[cfg(test)]
#[path = "forced_traj_tests.rs"]
mod tests;
