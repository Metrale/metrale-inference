// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: The step vocabulary and value spellings: every op has a shape, every value
//! spelling parses back to itself and a malformed one is refused with its step named, and
//! `differences` names each field that differs.
//!
//! Owner: metrale-circuit tests.
//! Invariants: none beyond the types.

use super::vocab::{Shape, parse_step, parse_value, shape_of, step_name, steps_for_base, steps_of};
use super::{NodePipeline, Num, Stated, Step, StepKind, Value, differences};
use crate::format::{Format, Scale};
use crate::ir::{LinearRole, OpKind};
use crate::state::StateDtype;

// 2026-10-02: One value of every shape, spelled canonically: the spelling parses back to the
// value, and the step a mis-shaped spelling is offered to refuses it. Mutation: a `name` arm
// that drops the `->` or `*` breaks the round trip.
#[test]
fn every_value_spelling_parses_back_to_itself() {
    let cases = [
        (
            StepKind::Act,
            Value::Format(Format::Fp8E4m3 {
                scale: Scale::Group(128),
            }),
        ),
        (
            StepKind::Weight,
            Value::Weight {
                stored: Format::Nvfp4 { group: 16 },
                operand: Num::E2m1,
            },
        ),
        (
            StepKind::Mma,
            Value::Mma {
                a: Num::Bf16,
                b: Num::E4m3,
            },
        ),
        (StepKind::Accumulate, Value::Num(Num::F32)),
        (StepKind::Scale, Value::OptNum(None)),
        (StepKind::Combine, Value::OptNum(Some(Num::F16))),
        (StepKind::Cache, Value::Kv("fp8k_turbo3v".into())),
        (StepKind::State, Value::State(StateDtype::F16)),
    ];
    for (k, v) in cases {
        assert_eq!(parse_value(k, &v.name()), Ok(v.clone()), "{}", v.name());
    }
    for (k, bad) in [
        (StepKind::Weight, "nvfp4/g16"),
        (StepKind::Weight, "nvfp4/g16->bf8"),
        (StepKind::Mma, "bf16xbf16"),
        (StepKind::Accumulate, "fp32"),
        (StepKind::Act, "e4m3"),
        (StepKind::Cache, "fp8 kv"),
        (StepKind::State, "e4m3"),
        (StepKind::Scale, "None"),
    ] {
        assert!(parse_value(k, bad).is_err(), "{bad} for {}", step_name(k));
    }
}

#[test]
fn every_step_name_round_trips_and_has_one_shape() {
    for k in [
        StepKind::Gather,
        StepKind::Act,
        StepKind::Weight,
        StepKind::Mma,
        StepKind::Accumulate,
        StepKind::Scale,
        StepKind::Compute,
        StepKind::Move,
        StepKind::Cache,
        StepKind::Scores,
        StepKind::Softmax,
        StepKind::State,
        StepKind::Score,
        StepKind::Scatter,
        StepKind::Combine,
        StepKind::Reduce,
        StepKind::Compare,
    ] {
        assert_eq!(parse_step(step_name(k)), Some(k));
    }
    assert_eq!(shape_of(StepKind::Weight), Shape::Weight);
    assert_eq!(parse_step("epilogue"), None);
}

// 2026-10-02: The op table: projections share one shape, the routed gate/up adds the gather,
// attention its cache and softmax; a manifest's base name finds the same steps as the op.
#[test]
fn the_op_table_gives_each_op_its_steps() {
    let proj = steps_of(&OpKind::Linear(LinearRole::Down));
    assert_eq!(proj, steps_of(&OpKind::Router));
    assert_eq!(proj, steps_of(&OpKind::ExpertDown));
    assert_eq!(steps_of(&OpKind::ExpertGateUp)[0], StepKind::Gather);
    assert_eq!(&steps_of(&OpKind::ExpertGateUp)[1..], proj);
    assert_eq!(
        steps_of(&OpKind::PagedAttention),
        [
            StepKind::Cache,
            StepKind::Scores,
            StepKind::Softmax,
            StepKind::Accumulate
        ]
    );
    assert_eq!(
        steps_of(&OpKind::Blend),
        [StepKind::Scatter, StepKind::Combine]
    );
    assert_eq!(steps_for_base("linear"), Some(proj));
    assert_eq!(
        steps_for_base("act_quant"),
        Some(steps_of(&OpKind::ActQuant(Format::Bf16)))
    );
    assert_eq!(steps_for_base("mamba3_scan"), None);
}

fn pipe(acc: Num, out: Format) -> NodePipeline {
    NodePipeline {
        inputs: vec![Format::Bf16, Format::I32],
        steps: vec![Step {
            kind: StepKind::Accumulate,
            value: Value::Num(acc),
        }],
        outputs: vec![out],
    }
}

#[test]
fn differences_name_each_differing_field_and_the_rendering_marks_stated_parts() {
    let a = pipe(Num::F32, Format::Bf16);
    assert!(differences(&a, &a).is_empty());
    let b = pipe(Num::Bf16, Format::F32);
    assert_eq!(
        differences(&a, &b),
        [
            "accumulate: required f32, declared bf16",
            "out[0]: required bf16, declared f32"
        ]
    );
    let stated = Stated {
        steps: [StepKind::Accumulate].into(),
        inputs: [1].into(),
        outputs: [0].into(),
    };
    assert_eq!(
        a.render(&stated),
        "bf16,i32 (rule) -> [accumulate f32 (rule)] -> bf16 (rule)"
    );
    assert_eq!(a.to_string(), "bf16,i32 -> [accumulate f32] -> bf16");
}
