// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: One check: a contract at one point and input class. The good arm runs the
//! kernel on seeded operands and compares it with the bounded reference (derived) or the named
//! sibling (bit_identical); a conforming emulation gives the noise floor; every mutation arm
//! must fail. The verdict is a pass only when all of that holds.
//!
//! Owner: metrale-accuracy.
//! Invariants:
//! - Pure orchestration: the kernel is reached only through [`KernelRunner`].
//! - A mutation arm is compared against the reference of the UNMUTATED case: it is caught when
//!   it leaves the bound (derived) or changes a byte (bit_identical). Mutation arms run on the
//!   gaussian class; the adversarial classes run the good arm and the floor (at a denormal or
//!   all-equal input a mutation can be invisible, which says nothing about the contract).
//! - The drift alarm sits at the geometric midpoint between the calibrated good arm and the
//!   nearest calibrated mutation, so an unchanged kernel cannot trip it and a kernel drifting
//!   toward a mutation trips it before the bound.

use sha2::{Digest, Sha256};

use metrale_circuit::venn::families::{Family, Values};

use crate::case::Case;
use crate::compare::{self, Vacuous};
use crate::contract::{Class, Contract};
use crate::inputs::{InputClass, SplitMix64};
use crate::plan::{self, Plan};
use crate::points::Shape;
use crate::refs::Reference;
use crate::refs::linear_mutate::shards;
use crate::runner::{KernelRunner, decode};

mod arms;
use arms::{derived_mutation, identical_mutation};

/// 2026-10-09: One arm's measurement.
#[derive(Debug, Clone, PartialEq)]
pub struct Arm {
    /// 2026-10-09: `good`, `floor`, or the mutation's spelling.
    pub name: String,
    /// 2026-10-09: Ran on the emulation, not the kernel.
    pub emulated: bool,
    /// 2026-10-09: Max err/bound (derived) or differing bytes (bit_identical).
    pub ratio: f64,
    /// 2026-10-09: Max absolute error (derived); 0 for bit_identical.
    pub max_err: f64,
    /// 2026-10-09: Elements (or bytes) compared.
    pub compared: usize,
    /// 2026-10-09: Share of compared elements not correctly rounded from the reference value
    /// (derived), or of differing bytes (bit_identical): the drift statistic.
    pub misrounded: f64,
    /// 2026-10-09: A mutation arm whose output equals the good arm's at every compared element:
    /// the mutation is not observable at this input (a narrower accumulator that lands on the
    /// same roundings), so it says nothing about the contract here. Never true for `good` or
    /// `floor`. A run fails when a contract's mutation is inert at every point it checks.
    pub inert: bool,
}

/// 2026-10-09: The verdict of one check.
#[derive(Debug, Clone, PartialEq)]
pub enum Verdict {
    /// 2026-10-09: Good arm inside the contract, every mutation caught.
    Pass,
    /// 2026-10-09: The good arm leaves the derived bound (or changes bytes).
    FailBound,
    /// 2026-10-09: The good arm is inside the bound but beyond the calibrated drift threshold.
    FailDrift {
        /// 2026-10-09: The threshold.
        threshold: f64,
    },
    /// 2026-10-09: A mutation stayed inside the contract: the contract is too loose.
    FailMutationPassed(String),
    /// 2026-10-09: A comparison could not support a verdict.
    FailVacuous(String),
    /// 2026-10-09: The check could not run.
    Error(String),
}

impl Verdict {
    /// 2026-10-09: Record spelling.
    pub fn name(&self) -> String {
        match self {
            Verdict::Pass => "pass".into(),
            Verdict::FailBound => "fail:bound".into(),
            Verdict::FailDrift { .. } => "fail:drift".into(),
            Verdict::FailMutationPassed(m) => format!("fail:mutation-passed:{m}"),
            Verdict::FailVacuous(w) => format!("fail:vacuous:{w}"),
            Verdict::Error(e) => format!("error:{e}"),
        }
    }
}

/// 2026-10-09: The outcome of one check.
#[derive(Debug, Clone, PartialEq)]
pub struct Outcome {
    /// 2026-10-09: Family.
    pub family: String,
    /// 2026-10-09: Entry point.
    pub kernel: String,
    /// 2026-10-09: Case key (point values and shape).
    pub key: String,
    /// 2026-10-09: Input class.
    pub input: InputClass,
    /// 2026-10-09: The good arm.
    pub good: Option<Arm>,
    /// 2026-10-09: The noise floor (derived only).
    pub floor: Option<Arm>,
    /// 2026-10-09: The mutation arms.
    pub mutations: Vec<Arm>,
    /// 2026-10-09: SHA-256 of the good arm's output bytes (cross-box byte parity).
    pub output_sha256: String,
    /// 2026-10-09: Verdict.
    pub verdict: Verdict,
}

/// 2026-10-09: What one check needs.
pub struct Job<'a> {
    /// 2026-10-09: The contract.
    pub contract: &'a Contract,
    /// 2026-10-09: Its family.
    pub family: &'a Family,
    /// 2026-10-09: The entry point checked.
    pub kernel: &'a str,
    /// 2026-10-09: Compile-time and policy values.
    pub point: &'a Values,
    /// 2026-10-09: The launch shape.
    pub shape: &'a Shape,
    /// 2026-10-09: Input class.
    pub input: InputClass,
    /// 2026-10-09: Corpus seed.
    pub seed: u64,
}

impl Job<'_> {
    /// 2026-10-09: The case key: kernel, point and shape.
    pub fn key(&self) -> String {
        let vals = self
            .point
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(",");
        let s = self.shape;
        format!(
            "{}[{vals}] {} rows={} k={} n={}",
            self.kernel, s.op, s.rows, s.in_dim, s.out_dim
        )
    }
}

fn hex(d: &[u8]) -> String {
    d.iter().map(|b| format!("{b:02x}")).collect()
}

/// 2026-10-09: Run one check.
pub fn run(job: &Job<'_>, runner: &mut dyn KernelRunner) -> Outcome {
    let mut o = Outcome {
        family: job.family.id.clone(),
        kernel: job.kernel.to_string(),
        key: job.key(),
        input: job.input,
        good: None,
        floor: None,
        mutations: Vec::new(),
        output_sha256: String::new(),
        verdict: Verdict::Pass,
    };
    if let Err(e) = arms(job, runner, &mut o) {
        o.verdict = e;
    }
    o
}

/// 2026-10-10: The case `job` launches (its canonical operands for the job's input class), as
/// the check builds it: the envelope sweep times and digests candidates on one shared case.
pub fn case_of(job: &Job<'_>) -> Result<Case, String> {
    prepare(job).map(|(_, _, c)| c).map_err(|v| v.name())
}

fn prepare(job: &Job<'_>) -> Result<(Reference, Plan, Case), Verdict> {
    let c = job.contract;
    let err = |e: String| Verdict::Error(e);
    let reference = Reference::parse(&c.reference)
        .ok_or_else(|| err(format!("reference `{}`", c.reference)))?;
    let pipeline = plan::declared(job.family, job.kernel, &c.op, job.point).map_err(err)?;
    let lens = reference.lens(job.shape, &pipeline);
    let plan = plan::plan(c, pipeline, &lens, job.point).map_err(err)?;
    let mut case = Case {
        family: job.family.id.clone(),
        kernel: job.kernel.to_string(),
        launcher: job.kernel.to_string(),
        op: c.op.clone(),
        tensors: Default::default(),
        scalars: Default::default(),
        out: (Vec::new(), crate::case::Enc::Bf16),
        split: Vec::new(),
    };
    reference
        .fill(&mut case, &plan, job.shape, job.input, job.seed, &job.key())
        .map_err(err)?;
    if let Some(s) = c.split {
        case.split = shards(case.out.0[1], s);
    }
    Ok((reference, plan, case))
}

pub(crate) fn vacuous(e: Vacuous) -> Verdict {
    Verdict::FailVacuous(e.to_string())
}

fn arms(job: &Job<'_>, runner: &mut dyn KernelRunner, o: &mut Outcome) -> Result<(), Verdict> {
    let (reference, plan, case) = prepare(job)?;
    let err = |e: String| Verdict::Error(e);
    let mut rng = SplitMix64::keyed(
        job.seed,
        &[
            &case.family,
            job.kernel,
            &job.key(),
            job.input.name(),
            "sample",
        ],
    );
    let idx = reference.sample(&case, &mut rng);
    let out = case
        .out
        .1
        .elem()
        .ok_or_else(|| err("an output with no rounding model".into()))?;
    let got = runner.run(&case).map_err(|e| err(e.to_string()))?;
    o.output_sha256 = hex(&Sha256::digest(&got));
    match &job.contract.class {
        Class::Derived => {
            let want = reference.reference(&case, &plan, &idx).map_err(err)?;
            let g = compare::bounded(&decode(&case, &got, &idx).map_err(err)?, &want, out)
                .map_err(vacuous)?;
            o.good = Some(Arm {
                name: "good".into(),
                emulated: false,
                ratio: g.max_ratio,
                max_err: g.max_err,
                compared: g.compared,
                misrounded: g.misrounded as f64 / g.compared.max(1) as f64,
                inert: false,
            });
            let mut floor = Arm {
                name: "floor".into(),
                emulated: true,
                ratio: 0.0,
                max_err: 0.0,
                compared: 0,
                misrounded: 0.0,
                inert: false,
            };
            for v in 0..crate::emulate::VARIANTS {
                let fl = reference
                    .emulate(&case, &plan, None, v, &idx)
                    .map_err(err)?;
                let f = compare::bounded(&fl, &want, out).map_err(vacuous)?;
                floor.ratio = floor.ratio.max(f.max_ratio);
                floor.max_err = floor.max_err.max(f.max_err);
                floor.compared = f.compared;
                floor.misrounded = floor
                    .misrounded
                    .max(f.misrounded as f64 / f.compared.max(1) as f64);
            }
            o.floor = Some(floor);
            if job.input == InputClass::Gaussian {
                for m in &job.contract.mutations {
                    o.mutations.push(derived_mutation(
                        job, reference, &plan, &case, &idx, m, out, &got, runner,
                    )?);
                }
            }
            if g.max_ratio > 1.0 {
                return Err(Verdict::FailBound);
            }
            let observed = g.misrounded as f64 / g.compared.max(1) as f64;
            if let Some(t) = drift_threshold(job.contract, &o.key, job.input, g.compared)
                && observed > t
            {
                return Err(Verdict::FailDrift { threshold: t });
            }
        }
        Class::BitIdentical { against } => {
            let mut sib = case.clone();
            sib.kernel = against.clone();
            sib.launcher = against.clone();
            let base = runner.run(&sib).map_err(|e| err(e.to_string()))?;
            let b = compare::bytes(&got, &base).map_err(vacuous)?;
            o.good = Some(Arm {
                name: "good".into(),
                emulated: false,
                ratio: b.differing as f64,
                max_err: 0.0,
                compared: b.compared,
                misrounded: b.differing as f64 / b.compared.max(1) as f64,
                inert: false,
            });
            if job.input == InputClass::Gaussian {
                for m in &job.contract.mutations {
                    o.mutations.push(identical_mutation(
                        job, reference, &case, &base, &got, m, runner,
                    )?);
                }
            }
            if b.differing > 0 {
                return Err(Verdict::FailBound);
            }
        }
    }
    // 2026-10-09: An inert arm (output identical to the good arm's) cannot be judged here; a
    // visible one inside the contract means the contract is too loose. Inert arms are
    // accounted for across the run ([`unobserved`]).
    let derived = !matches!(job.contract.class, Class::BitIdentical { .. });
    if let Some(m) = o
        .mutations
        .iter()
        .find(|m| !m.inert && ((derived && m.ratio <= 1.0) || m.ratio == 0.0))
    {
        return Err(Verdict::FailMutationPassed(m.name.clone()));
    }
    Ok(())
}

/// 2026-10-09: The drift threshold of a calibrated (point, input), on the share of misrounded
/// outputs: the geometric midpoint between the expected spread (the larger of the calibrated
/// good share, the noise floor's share over the legitimate bracketings, and one element in
/// `compared`) and the nearest mutation's share. A legitimate reordering flips a few elements
/// and stays under it; a kernel moving toward a mutation crosses it before the bound.
pub fn drift_threshold(c: &Contract, key: &str, input: InputClass, compared: usize) -> Option<f64> {
    let cal = c
        .calibration
        .iter()
        .find(|r| r.point == key && r.input == input.name())?;
    let spread = cal
        .misrounded
        .max(cal.floor_misrounded)
        .max(1.0 / compared.max(1) as f64);
    Some((spread * cal.mutation_min_misrounded.min(1.0)).sqrt())
}

/// 2026-10-09: The (family, kernel, mutation) a run never observed: every arm of it was inert,
/// so no checked point proves the contract catches it. A non-empty answer fails the run.
pub fn unobserved(outcomes: &[Outcome]) -> Vec<(String, String, String)> {
    use std::collections::BTreeMap;
    let mut seen: BTreeMap<(String, String, String), bool> = BTreeMap::new();
    for o in outcomes {
        for m in &o.mutations {
            let base = m
                .name
                .split(" (fault")
                .next()
                .unwrap_or(&m.name)
                .to_string();
            let e = seen
                .entry((o.family.clone(), o.kernel.clone(), base))
                .or_insert(false);
            *e |= !m.inert;
        }
    }
    seen.into_iter()
        .filter(|(_, v)| !v)
        .map(|(k, _)| k)
        .collect()
}
