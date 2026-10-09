// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Both arms of the tensor-core projection contracts (dense_bf16_tc, w4a16_tc, tc_rows,
//! w8a16_gemm), proven on the CPU against the conforming emulation of the contracts as committed
//! in kernels/gb10/common/ACCURACY.toml: a conforming "kernel" passes every input class with a
//! margin; every mutation is caught; a kernel that breaks its declaration fails the bound; a
//! contract made too loose fails because a mutation passes. One representative per family.
//!
//! The wrong-symbol arm on a GPU launches another entry point on the contract's own grid, which
//! covers only part of the output; its CPU stand-in writes the conforming values into that part
//! and leaves the rest at the sentinel, as the GPU output buffer is left.

mod common;

use common::{Behaviour, Emu, Tree, families};
use metrale_accuracy::case::Case;
use metrale_accuracy::check::{Job, Outcome, Verdict, run};
use metrale_accuracy::contract::{Contract, Level, parse_contracts};
use metrale_accuracy::elem::BF16;
use metrale_accuracy::inputs::InputClass;
use metrale_accuracy::points::Shape;
use metrale_accuracy::runner::{KernelRunner, RunError, SENTINEL};
use metrale_circuit::venn::Repo;
use metrale_circuit::venn::families::Values;

/// 2026-10-09: The corpus seed of the committed contracts.
const SEED: u64 = 20261009;

fn committed(family: &str, kernel: &str, op: &str) -> Contract {
    let text = Tree
        .read("kernels/gb10/common/ACCURACY.toml")
        .expect("ACCURACY.toml");
    parse_contracts(&text)
        .expect("ACCURACY.toml parses")
        .contracts
        .into_iter()
        .find(|c| c.family == family && c.op == op && c.kernels.iter().any(|k| k == kernel))
        .unwrap_or_else(|| panic!("no committed contract for {family} {kernel} {op}"))
}

fn shape(op: &str, rows: u64, k: u64, n: u64) -> Shape {
    Shape {
        op: op.into(),
        weight: None,
        activation: None,
        output: None,
        in_dim: k,
        out_dim: n,
        rows,
        runtime: Default::default(),
    }
}

/// 2026-10-09: The conforming emulation, plus the wrong-symbol stand-in: `symbol` writes only the
/// first `cols / cover` output columns.
struct Tc {
    emu: Emu,
    symbol: Option<(String, usize)>,
}

impl KernelRunner for Tc {
    fn run(&mut self, case: &Case) -> Result<Vec<u8>, RunError> {
        let Some((symbol, cover)) = self.symbol.clone().filter(|(s, _)| *s == case.kernel) else {
            return self.emu.run(case);
        };
        let mut own = case.clone();
        own.kernel = case.launcher.clone();
        let mut out = self.emu.run(&own)?;
        let (rows, cols) = (case.out.0[0], case.out.0[1]);
        let width = case.out.1.bytes_for(1);
        assert!(cover > 1, "{symbol} covers every column: no stand-in");
        for r in 0..rows {
            out[(r * cols + cols / cover) * width..(r + 1) * cols * width].fill(SENTINEL);
        }
        Ok(out)
    }

    fn closure(&self) -> String {
        "cpu-emulation".into()
    }

    fn device(&self) -> String {
        "cpu".into()
    }
}

/// 2026-10-09: One representative point of a family.
struct Point {
    family: &'static str,
    kernel: &'static str,
    op: &'static str,
    point: Values,
    shape: Shape,
    /// 2026-10-09: The contract's wrong symbol and the share of columns its grid covers.
    symbol: Option<(&'static str, usize)>,
}

fn runner(p: &Point, c: &Contract, behaviour: Behaviour) -> Tc {
    let family = families()
        .families
        .into_iter()
        .find(|f| f.id == p.family)
        .unwrap();
    Tc {
        emu: Emu {
            contract: c.clone(),
            family,
            behaviour,
            // 2026-10-09: No entry point is a data-mutation stand-in here.
            wrong: ("none::none".into(), |_| Ok(())),
            shape: p.shape.clone(),
        },
        symbol: p.symbol.map(|(s, f)| (s.to_string(), f)),
    }
}

fn check(p: &Point, c: &Contract, behaviour: Behaviour, input: InputClass) -> Outcome {
    let mut r = runner(p, c, behaviour);
    let family = r.emu.family.clone();
    let job = Job {
        contract: c,
        family: &family,
        kernel: p.kernel,
        point: &p.point,
        shape: &p.shape,
        input,
        seed: SEED,
    };
    run(&job, &mut r)
}

/// 2026-10-09: (a) every input class passes with a margin, (b) every mutation is caught by more
/// than twice the bound, (c) a BF16 accumulator fails the bound, (d) a contract whose K reduction
/// is declared ten million terms deeper lets a mutation pass.
fn prove(p: &Point) {
    let c = committed(p.family, p.kernel, p.op);
    for input in c.inputs.clone() {
        let o = check(p, &c, Behaviour::Conforming, input);
        assert_eq!(
            o.verdict,
            Verdict::Pass,
            "{} {}: {o:#?}",
            p.kernel,
            input.name()
        );
        let (good, floor) = (o.good.clone().unwrap(), o.floor.clone().unwrap());
        println!(
            "{} {}: good {:.3e} floor {:.3e} compared {}",
            p.kernel,
            input.name(),
            good.ratio,
            floor.ratio,
            good.compared
        );
        assert!(
            good.ratio < 0.5,
            "{}: good ratio {}",
            input.name(),
            good.ratio
        );
        assert!(
            good.compared > 100,
            "{}: {} compared",
            input.name(),
            good.compared
        );
        if input == InputClass::Gaussian {
            assert_eq!(o.mutations.len(), c.mutations.len());
            for m in &o.mutations {
                println!("{} mutation {}: {:.3e}", p.kernel, m.name, m.ratio);
                assert!(m.ratio > 2.0, "{} caught only at ratio {}", m.name, m.ratio);
            }
        }
    }
    let broken = check(p, &c, Behaviour::Accumulator(BF16), InputClass::Gaussian);
    assert_eq!(broken.verdict, Verdict::FailBound, "{:#?}", broken.good);
    let mut loose = c.clone();
    loose.reduction.get_mut("k").unwrap().push(Level {
        level: "loose".into(),
        width: "10000000".into(),
        order: "sequential".into(),
    });
    let o = check(p, &loose, Behaviour::Conforming, InputClass::Gaussian);
    assert!(
        matches!(o.verdict, Verdict::FailMutationPassed(_)),
        "{:?}",
        o.verdict
    );
}

#[test]
fn dense_bf16_tc_tc16_proves_both_arms() {
    // 2026-10-09: The 27B drafter's k projection at 9 rows (the tc16 tier's narrowest).
    prove(&Point {
        family: "dense_bf16_tc",
        kernel: "dense_gemv_bf16_tc::dense_gemv_bf16_tc16",
        op: "linear",
        point: Values::new(),
        shape: shape("linear:k", 9, 5120, 1024),
        // 2026-10-09: dense_gemv_bf16_batchm writes 4 columns per CTA of the 16-column grid.
        symbol: Some(("dense_gemv_bf16_batchm::dense_gemv_bf16_batchm", 4)),
    });
}

#[test]
fn w4a16_tc_tc8_proves_both_arms() {
    // 2026-10-09: The 27B's NVFP4 k projection at 4 rows (decode C=1, MTP k=3).
    prove(&Point {
        family: "w4a16_tc",
        kernel: "w4a16_gemv_tc::w4a16_gemv_tc8",
        op: "linear",
        point: Values::new(),
        shape: shape("linear:k", 4, 5120, 1024),
        symbol: None,
    });
}

#[test]
fn tc_rows_nvfp4_head_proves_both_arms() {
    // 2026-10-09: The 35B's declared NVFP4 head (K = 2048), its 248320-row vocabulary scaled down
    // 16x to 15520 rows, which keeps a ragged last 64-column tile.
    prove(&Point {
        family: "tc_rows",
        kernel: "w4a16_tc_rows::w4a16_tc_rows_64",
        op: "lm_head",
        point: Values::from([("weight".to_string(), "nvfp4/g16".to_string())]),
        shape: shape("lm_head", 2, 2048, 248_320 / 16),
        symbol: None,
    });
}

#[test]
fn w8a16_gemm_m32_proves_both_arms() {
    // 2026-10-09: The 35B's FP8 q projection at 4 rows (the skinny body).
    prove(&Point {
        family: "w8a16_gemm",
        kernel: "w8a16_gemm_pipelined_m32::w8a16_gemm_pipelined_m32",
        op: "linear",
        point: Values::new(),
        shape: shape("linear:q", 4, 2048, 8192),
        // 2026-10-09: The 16-row scalar GEMV writes 4 columns per CTA of the skinny 8-column grid.
        symbol: Some(("w8a16_gemv_batch4::w8a16_gemv_batch16_strided", 2)),
    });
}

#[test]
fn the_committed_contracts_fit_the_families_and_cover_every_swept_tc_point() {
    let contracts =
        parse_contracts(&Tree.read("kernels/gb10/common/ACCURACY.toml").unwrap()).unwrap();
    let fams = families();
    assert_eq!(
        metrale_accuracy::jobs::validate(&contracts, &fams),
        Vec::<String>::new()
    );
    let sweep = metrale_accuracy::points::sweep(&Tree, "gb10").unwrap();
    for family in ["dense_bf16_tc", "w4a16_tc", "tc_rows", "w8a16_gemm"] {
        let (planned, cov) = metrale_accuracy::jobs::plan(
            &sweep,
            &contracts,
            &fams,
            metrale_accuracy::jobs::Scope::Quick,
            Some(family),
            None,
        );
        assert!(cov.swept_points > 0, "{family}: nothing swept");
        assert_eq!(
            cov.covered_points, cov.swept_points,
            "{family}: {:?}",
            cov.uncovered
        );
        assert!(cov.unused.is_empty(), "{family}: {:?}", cov.unused);
        assert!(!planned.is_empty());
    }
}
