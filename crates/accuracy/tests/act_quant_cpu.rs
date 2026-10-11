// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The FP8 activation quantizer contract on the CPU: the host model passes every
//! class exactly; a quantizer writing the other scale granularity, or rounding toward zero, fails.

mod common;

use common::{contract, families};
use metrale_accuracy::case::{Case, Tensor};
use metrale_accuracy::check::{Job, Verdict, run};
use metrale_accuracy::elem::{E4M3, F32};
use metrale_accuracy::inputs::InputClass;
use metrale_accuracy::points::Shape;
use metrale_accuracy::refs::act_quant::quantize_group;
use metrale_accuracy::runner::{KernelRunner, RunError};

const ROW: &str = r#"
[[contract]]
family = "w8a8_act_quant"
kernels = ["w8a8_act_quant::w8a8_act_quant_row"]
op = "act_quant"
reference = "act_quant"
class = "derived"
reduction = {}
ftz = false
approx = {}
constants = { scale_floor = 1e-12 }
scale_fold = "none"
inputs = ["gaussian", "outliers", "near_overflow", "all_equal", "zero_rows"]
mutations = ["symbol:w8a8_act_quant::w8a8_act_quant_g128"]
"#;

/// 2026-10-09: A host quantizer: `group` values per scale (the row kernel: k), rounding RNE or
/// toward zero; it writes the decoded values a row-scale consumer would read.
struct Quant {
    truncate: bool,
}

impl KernelRunner for Quant {
    fn run(&mut self, case: &Case) -> Result<Vec<u8>, RunError> {
        let x = case.tensor("x").map_err(RunError::Unavailable)?;
        let (rows, k) = (x.dims[0], x.dims[1]);
        let g = if case.kernel.ends_with("g128") {
            128
        } else {
            k
        };
        let mut out = Vec::with_capacity(rows * k);
        for r in 0..rows {
            // 2026-10-09: The row consumer reads one scale per row: the first scale written.
            let row: Vec<f64> = (0..k).map(|c| x.get(r * k + c)).collect();
            let (s0, _) = quantize_group(&row[..g], 1e-12);
            for c in 0..k {
                let t = F32.round(row[c] / s0).unwrap().clamp(-448.0, 448.0);
                let q = if self.truncate {
                    E4M3.round(t.trunc().max(-448.0))
                        .unwrap_or(0.0)
                        .min(t.abs())
                        .copysign(t)
                } else {
                    E4M3.round(t).unwrap()
                };
                out.push(F32.round(q * s0).unwrap());
            }
        }
        Ok(Tensor::encode(case.out.1, vec![rows * k], &out)
            .map_err(RunError::Unavailable)?
            .bytes
            .to_vec())
    }
    fn closure(&self) -> String {
        "cpu".into()
    }
    fn device(&self) -> String {
        "cpu".into()
    }
}

fn job_on(
    c: &metrale_accuracy::contract::Contract,
    s: &Shape,
    input: InputClass,
    runner: &mut dyn KernelRunner,
) -> Verdict {
    let f = families()
        .families
        .into_iter()
        .find(|f| f.id == c.family)
        .unwrap();
    let point = [("format".to_string(), "fp8/token".to_string())]
        .into_iter()
        .collect();
    run(
        &Job {
            contract: c,
            family: &f,
            kernel: &c.kernels[0],
            point: &point,
            shape: s,
            input,
            seed: 5,
        },
        runner,
    )
    .verdict
}

#[test]
fn the_row_quantizer_is_exact_and_its_mutations_fail() {
    let c = contract(ROW);
    let s = Shape {
        op: "act_quant:fp8/token".into(),
        weight: None,
        activation: None,
        output: None,
        in_dim: 5120,
        out_dim: 5120,
        rows: 4,
        runtime: Default::default(),
    };
    for input in c.inputs.clone() {
        assert_eq!(
            job_on(&c, &s, input, &mut Quant { truncate: false }),
            Verdict::Pass,
            "{}",
            input.name()
        );
    }
    assert_eq!(
        job_on(&c, &s, InputClass::Gaussian, &mut Quant { truncate: true }),
        Verdict::FailBound
    );
}
