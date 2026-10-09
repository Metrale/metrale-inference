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
use crate::mutation::Mutation;
use crate::plan::{self, Plan};
use crate::points::Shape;
use crate::refs::Reference;
use crate::refs::linear_mutate::shards;
use crate::runner::{KernelRunner, RunError, decode};

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

fn vacuous(e: Vacuous) -> Verdict {
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
            });
            let mut floor = Arm {
                name: "floor".into(),
                emulated: true,
                ratio: 0.0,
                max_err: 0.0,
                compared: 0,
                misrounded: 0.0,
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
                        job, reference, &plan, &case, &idx, m, out, runner,
                    )?);
                }
            }
            if g.max_ratio > 1.0 {
                return Err(Verdict::FailBound);
            }
            let observed = g.misrounded as f64 / g.compared.max(1) as f64;
            if let Some(t) = drift_threshold(job.contract, &o.key, job.input, g.compared) {
                if observed > t {
                    return Err(Verdict::FailDrift { threshold: t });
                }
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
            });
            if job.input == InputClass::Gaussian {
                for m in &job.contract.mutations {
                    o.mutations
                        .push(identical_mutation(job, reference, &case, &base, m, runner)?);
                }
            }
            if b.differing > 0 {
                return Err(Verdict::FailBound);
            }
        }
    }
    if let Some(m) = o
        .mutations
        .iter()
        .find(|m| m.ratio <= 1.0 && !matches!(job.contract.class, Class::BitIdentical { .. }))
    {
        return Err(Verdict::FailMutationPassed(m.name.clone()));
    }
    if let Some(m) = o.mutations.iter().find(|m| m.ratio == 0.0) {
        return Err(Verdict::FailMutationPassed(m.name.clone()));
    }
    Ok(())
}

fn derived_mutation(
    job: &Job<'_>,
    reference: Reference,
    plan: &Plan,
    case: &Case,
    idx: &[usize],
    m: &Mutation,
    out: crate::elem::Elem,
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
            let want = reference.reference(case, plan, &at).map_err(err)?;
            let em = reference
                .emulate(case, plan, Some(*e), 0, &at)
                .map_err(err)?;
            return arm(m, true, &em, &want, out);
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
    arm(m, false, &decode(case, &got, &at).map_err(err)?, &want, out)
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
    })
}

fn identical_mutation(
    job: &Job<'_>,
    reference: Reference,
    case: &Case,
    base: &[u8],
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
    })
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
