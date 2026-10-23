// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The paired in-serve A/B instrument (`METRALE_PAIRED_AB`): one serve runs every
//! single-sequence verify step on one of a few lever arms, in a palindromic order (A B B A for
//! two arms), each arm through its own CUDA graph, and times each step's forward on the device.
//! Adjacent steps of different arms see the same request at almost the same KV length, so the
//! paired differences cancel the drift a between-serve A/B pays for. `met benchmark paired-ab`
//! analyses the `PAIRED_AB` lines.
//!
//! Owner: model-layers ops (instrument; family-neutral: a model maps the lever names).
//! Invariants:
//! - Off (`NONE` levers, nothing timed) unless the setting is present; resolved once per model.
//! - The spec grammar: `aa` (two arms, both A: the A/A control), `aa:graphs` (both A, each through
//!   its own graph: the control for a lever arm's cost of switching graphs), or `n1,n2,..` (arms A and
//!   all-on), with an optional `:attr` suffix adding, per lever, the arm with only that lever
//!   off. At most [`AbLevers::MAX`] names; duplicates and empty names are refused.
//! - The arm of step `i` is [`PairedAbSpec::arm_at`]: cycle `i / order.len()` runs `order` (arms
//!   ascending then descending) or its mirror (descending then ascending), chosen by a fixed
//!   hash of the cycle index, so a position effect inside the cycle (measured 2026-10-23: the
//!   second step of a four-step cycle ran ~45 us faster, arm-blind) lands on every arm equally
//!   often instead of biasing the inner arms. A pure function of `i`: every rank of a
//!   tensor-parallel serve that counts the same verify calls runs the same arm.
//! - A step is logged only when its pair of events completed; one still pending when the next
//!   step starts is dropped with a warning, never waited for (a verify step ends in a host
//!   readback, so its end has completed by then).

use std::sync::{Mutex, OnceLock};

use anyhow::{Result, bail};
use metrale_gpu_runtime::timing::StepTimer;

use crate::layer::AbLevers;

/// 2026-10-23: A parsed `METRALE_PAIRED_AB`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PairedAbSpec {
    /// 2026-10-23: The lever names, by position (empty for `aa`).
    pub names: Vec<String>,
    /// 2026-10-23: The arms' lever sets; arm 0 is A (`NONE`).
    pub arms: Vec<AbLevers>,
}

impl PairedAbSpec {
    /// 2026-10-23: Parse the setting's text (grammar in the module doc).
    pub fn parse(text: &str) -> Result<Self> {
        let text = text.trim();
        if text == "aa" {
            return Ok(Self {
                names: Vec::new(),
                arms: vec![AbLevers::NONE, AbLevers::NONE],
            });
        }
        // 2026-10-23: A/A over two graphs: arm 1 holds a position no name owns, so a model
        // (which reads only its named positions) runs A, but the step's graph key differs.
        if text == "aa:graphs" {
            return Ok(Self {
                names: Vec::new(),
                arms: vec![AbLevers::NONE, AbLevers::only(AbLevers::MAX - 1)],
            });
        }
        let (list, attr) = match text.strip_suffix(":attr") {
            Some(l) => (l, true),
            None => (text, false),
        };
        let names: Vec<String> = list.split(',').map(|n| n.trim().to_string()).collect();
        if names.iter().any(String::is_empty) {
            bail!("METRALE_PAIRED_AB `{text}`: an empty lever name");
        }
        if names.len() > AbLevers::MAX {
            bail!(
                "METRALE_PAIRED_AB: {} levers, at most {}",
                names.len(),
                AbLevers::MAX
            );
        }
        for (i, n) in names.iter().enumerate() {
            if names[..i].contains(n) {
                bail!("METRALE_PAIRED_AB: lever `{n}` named twice");
            }
        }
        let all = AbLevers::first(names.len());
        let mut arms = vec![AbLevers::NONE, all];
        if attr {
            arms.extend((0..names.len()).map(|i| all.without(i)));
        }
        Ok(Self { names, arms })
    }

    /// 2026-10-23: The palindromic arm order: arms ascending, then descending.
    pub fn order(&self) -> Vec<usize> {
        let up: Vec<usize> = (0..self.arms.len()).collect();
        up.iter().copied().chain(up.iter().rev().copied()).collect()
    }

    /// 2026-10-23: The arm of step `step` (the module invariant): position `step % len` of the
    /// cycle's orientation, the mirror where the hash of the cycle index is odd.
    pub fn arm_at(&self, step: u64) -> usize {
        let len = 2 * self.arms.len() as u64;
        let (cycle, pos) = (step / len, (step % len) as usize);
        let up = pos < self.arms.len();
        let i = if up {
            pos
        } else {
            2 * self.arms.len() - 1 - pos
        };
        if mirror(cycle) {
            self.arms.len() - 1 - i
        } else {
            i
        }
    }

    /// 2026-10-23: Every lever set an arm uses, without repeats (a model prebuilds one kernel
    /// set per entry).
    pub fn lever_sets(&self) -> Vec<AbLevers> {
        let mut v = self.arms.clone();
        v.sort();
        v.dedup();
        v
    }
}

/// 2026-10-23: Whether cycle `c` runs the mirrored order: the low bit of the SplitMix64
/// finalizer of `c` (a fixed, seedless hash, so every rank agrees).
fn mirror(c: u64) -> bool {
    let mut z = c.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (z ^ (z >> 31)) & 1 == 1
}

/// 2026-10-23: The running instrument of one model.
pub struct PairedAbRun {
    spec: PairedAbSpec,
    timer: StepTimer,
    state: Mutex<RunState>,
}

#[derive(Default)]
struct RunState {
    step: u64,
    /// 2026-10-23: The step whose events are recorded and not yet read:
    /// (step, arm, slot, KV tokens before the step).
    pending: Option<(u64, usize, usize, usize)>,
}

/// 2026-10-23: The arm one step runs and the timer slot it records into.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PairedStep {
    pub step: u64,
    pub arm: usize,
    pub levers: AbLevers,
    slot: usize,
    kv: usize,
}

impl PairedAbRun {
    fn new(spec: PairedAbSpec) -> Result<Self> {
        tracing::info!(
            "PAIRED_AB on: levers [{}], {} arms {:?}, order {:?} or its mirror per cycle",
            spec.names.join(","),
            spec.arms.len(),
            spec.arms.iter().map(|a| a.bits()).collect::<Vec<_>>(),
            spec.order()
        );
        Ok(Self {
            spec,
            timer: StepTimer::new(2)?,
            state: Mutex::new(RunState::default()),
        })
    }

    /// 2026-10-23: The spec this run follows.
    pub fn spec(&self) -> &PairedAbSpec {
        &self.spec
    }

    /// 2026-10-23: Log the previous step, then open the span of the next one, a forward over a
    /// sequence of `kv` tokens, on `stream`.
    pub fn begin(&self, kv: usize, stream: u64) -> Result<PairedStep> {
        let mut st = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("PAIRED_AB lock"))?;
        self.harvest(&mut st)?;
        let step = st.step;
        st.step += 1;
        let arm = self.spec.arm_at(step);
        let slot = (step % 2) as usize;
        self.timer.start(slot, stream)?;
        Ok(PairedStep {
            step,
            arm,
            levers: self.spec.arms[arm],
            slot,
            kv,
        })
    }

    /// 2026-10-23: Close `s`'s span on `stream`; it is logged at the next [`Self::begin`].
    pub fn end(&self, s: PairedStep, stream: u64) -> Result<()> {
        let mut st = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("PAIRED_AB lock"))?;
        self.timer.end(s.slot, stream)?;
        st.pending = Some((s.step, s.arm, s.slot, s.kv));
        Ok(())
    }

    fn harvest(&self, st: &mut RunState) -> Result<()> {
        let Some((step, arm, slot, kv)) = st.pending.take() else {
            return Ok(());
        };
        let Some(us) = self.timer.elapsed_us(slot)? else {
            tracing::warn!(
                "PAIRED_AB step={step} arm={arm} not complete at the next step: dropped"
            );
            return Ok(());
        };
        tracing::info!(
            "PAIRED_AB step={step} arm={arm} levers={:#x} kv={kv} us={us:.3}",
            self.spec.arms[arm].bits()
        );
        Ok(())
    }
}

/// 2026-10-23: The model's instrument slot: resolved once, at the first step that asks.
#[derive(Default)]
pub struct PairedAb {
    run: OnceLock<Option<PairedAbRun>>,
}

impl std::fmt::Debug for PairedAb {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PairedAb")
            .field("on", &self.run.get().map(Option::is_some))
            .finish()
    }
}

impl PairedAb {
    /// 2026-10-23: The running instrument for setting `text` (`None`: off), built on first use.
    pub fn get(&self, text: Option<&str>) -> Result<Option<&PairedAbRun>> {
        if let Some(r) = self.run.get() {
            return Ok(r.as_ref());
        }
        let built = match text {
            None => None,
            Some(t) => Some(PairedAbRun::new(PairedAbSpec::parse(t)?)?),
        };
        Ok(self.run.get_or_init(|| built).as_ref())
    }
}

#[cfg(test)]
#[path = "paired_ab_tests.rs"]
mod tests;
