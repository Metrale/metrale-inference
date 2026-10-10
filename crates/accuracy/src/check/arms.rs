// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The mutation arms of a check (split from `check.rs` under the 500-line rule): each
//! runs one mutation and measures it against the UNMUTATED case's reference (derived) or the
//! sibling's bytes (bit_identical), and marks it inert when its output equals the good arm's.
//!
//! Owner: metrale-accuracy.
//! Invariants: as `check.rs`.

use super::{Arm, Job, Verdict, vacuous};
use crate::case::Case;
use crate::compare;
use crate::inputs::SplitMix64;
use crate::mutation::Mutation;
use crate::plan::Plan;
use crate::refs::Reference;
use crate::runner::{KernelRunner, RunError, decode};

pub(crate) fn derived_mutation(
    job: &Job<'_>,
    reference: Reference,
    plan: &Plan,
    case: &Case,
    idx: &[usize],
    m: &Mutation,
    out: crate::elem::Elem,
    good: &[u8],
    runner: &mut dyn KernelRunner,
) -> Result<Arm, Verdict> {
    let err = |e: String| Verdict::Error(format!("{}: {e}", m.name()));
    let mut mutated = case.clone();
    let mut rng = SplitMix64::keyed(
        job.seed,
        &[
            &case.family,
            job.kernel,
            &job.key(),
            job.input.name(),
            &m.name(),
        ],
    );
    let mut at: Vec<usize> = idx.to_vec();
    let got = match m {
        Mutation::Accumulate(e) => {
            // 2026-10-09: The emulated arm runs every legitimate bracketing and keeps the one
            // that strays furthest: a narrower accumulator in one row's single reduction (a norm
            // at one row) can land near the exact sum by chance in one order, and the arm proves
            // the contract can see the narrower accumulator, not that one order is unlucky.
            let want = reference.reference(case, plan, &at).map_err(err)?;
            let mut worst: Option<Arm> = None;
            let mut visible = false;
            for v in 0..crate::emulate::VARIANTS {
                let em = reference
                    .emulate(case, plan, Some(*e), v, &at)
                    .map_err(err)?;
                let conforming = reference.emulate(case, plan, None, v, &at).map_err(err)?;
                visible |= em
                    .iter()
                    .zip(&conforming)
                    .any(|(a, b)| a.to_bits() != b.to_bits());
                let a = arm(m, true, &em, &want, out)?;
                if worst.as_ref().is_none_or(|w| a.ratio > w.ratio) {
                    worst = Some(a);
                }
            }
            let mut w = worst.ok_or_else(|| err("no bracketing ran".into()))?;
            w.inert = !visible;
            return Ok(w);
        }
        Mutation::Symbol(s) => {
            mutated.kernel = s.clone();
            match runner.run(&mutated) {
                Ok(b) => b,
                Err(e) => return faulted(m, e),
            }
        }
        _ => {
            at.extend(reference.mutate(&mut mutated, m, &mut rng).map_err(err)?);
            at.sort_unstable();
            at.dedup();
            match runner.run(&mutated) {
                Ok(b) => b,
                Err(e) => return faulted(m, e),
            }
        }
    };
    let want = reference.reference(case, plan, &at).map_err(err)?;
    let mutated_vals = decode(case, &got, &at).map_err(err)?;
    let good_vals = decode(case, good, &at).map_err(err)?;
    let mut a = arm(m, false, &mutated_vals, &want, out)?;
    a.inert = mutated_vals
        .iter()
        .zip(&good_vals)
        .all(|(x, y)| x.to_bits() == y.to_bits());
    Ok(a)
}

/// 2026-10-09: A mutation arm whose launch failed: a fault is a loud detection (infinite
/// ratio, the fault named); an unavailable launch is a setup error of the contract.
fn faulted(m: &Mutation, e: RunError) -> Result<Arm, Verdict> {
    match e {
        RunError::Fault(why) => Ok(Arm {
            name: format!("{} (fault: {why})", m.name()),
            emulated: false,
            ratio: f64::INFINITY,
            max_err: f64::INFINITY,
            compared: 0,
            misrounded: 1.0,
            inert: false,
        }),
        RunError::Unavailable(why) => Err(Verdict::Error(format!("{}: {why}", m.name()))),
    }
}

fn arm(
    m: &Mutation,
    emulated: bool,
    got: &[f64],
    want: &[crate::bounded::Bounded],
    out: crate::elem::Elem,
) -> Result<Arm, Verdict> {
    let b = compare::bounded(got, want, out).map_err(vacuous)?;
    Ok(Arm {
        name: m.name(),
        emulated,
        ratio: b.max_ratio,
        max_err: b.max_err,
        compared: b.compared,
        misrounded: b.misrounded as f64 / b.compared.max(1) as f64,
        inert: false,
    })
}

pub(crate) fn identical_mutation(
    job: &Job<'_>,
    reference: Reference,
    case: &Case,
    base: &[u8],
    good: &[u8],
    m: &Mutation,
    runner: &mut dyn KernelRunner,
) -> Result<Arm, Verdict> {
    let err = |e: String| Verdict::Error(format!("{}: {e}", m.name()));
    let mut mutated = case.clone();
    let mut rng = SplitMix64::keyed(
        job.seed,
        &[
            &case.family,
            job.kernel,
            &job.key(),
            job.input.name(),
            &m.name(),
        ],
    );
    match m {
        Mutation::Symbol(s) => mutated.kernel = s.clone(),
        Mutation::Accumulate(_) => {
            return Err(err("an emulated arm has no bytes to compare".into()));
        }
        _ => {
            reference.mutate(&mut mutated, m, &mut rng).map_err(err)?;
        }
    }
    let got = match runner.run(&mutated) {
        Ok(b) => b,
        Err(e) => return faulted(m, e),
    };
    let b = compare::bytes(&got, base).map_err(vacuous)?;
    Ok(Arm {
        name: m.name(),
        emulated: false,
        ratio: b.differing as f64,
        max_err: 0.0,
        compared: b.compared,
        misrounded: b.differing as f64 / b.compared.max(1) as f64,
        inert: got.as_slice() == good,
    })
}
