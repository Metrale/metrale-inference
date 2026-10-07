// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-06: Frozen independent scheduler bytes and staged-arithmetic controls.
#[path = "../src/qwen_image21/scheduler.rs"]
mod scheduler;
use half::bf16;
use scheduler::{Config, Schedule};
use sha2::{Digest, Sha256};

fn fixture() -> serde_json::Value {
    serde_json::from_str(include_str!("fixtures/qwen_image21_scheduler.json")).unwrap()
}
#[test]
fn a_rejects_unsupported_policy_and_degenerate_schedule() {
    let mut bad = fixture()["config"].clone();
    bad["stochastic_sampling"] = true.into();
    assert!(Config::from_value(bad).is_err());
    let config = Config::from_value(fixture()["config"].clone()).unwrap();
    assert!(Schedule::new(&config, 1, 256).is_err());
    assert!(Schedule::new(&config, 40, 0).is_err());
    let mut bad = fixture()["config"].clone();
    bad["unknown_math"] = true.into();
    assert!(Config::from_value(bad).is_err());
}
#[test]
fn matches_pinned_schedule_and_model_timestep_bytes() {
    let data = fixture();
    let config = Config::from_value(data["config"].clone()).unwrap();
    for case in data["schedules"].as_array().unwrap() {
        let schedule = Schedule::new(
            &config,
            case["count"].as_u64().unwrap() as usize,
            case["image_tokens"].as_u64().unwrap() as usize,
        )
        .unwrap();
        let expected: Vec<u32> = case["sigma_bits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        assert_eq!(
            schedule
                .sigmas()
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            expected
        );
        assert!(!schedule.is_empty());
        for i in 0..schedule.len() {
            assert_eq!(
                schedule.delta(i).unwrap(),
                schedule.sigmas()[i + 1] - schedule.sigmas()[i]
            );
            assert_eq!(
                schedule.timestep(i).unwrap().to_bits(),
                case["time_bits"][i].as_u64().unwrap() as u32
            );
            assert_eq!(
                schedule.model_timestep(i).unwrap().to_bits(),
                case["model_time_bits"][i].as_u64().unwrap() as u16
            );
        }
        assert!(schedule.timestep(schedule.len()).is_err());
    }
}
#[test]
fn staged_step_matches_all_finite_bf16_inputs_and_rejects_wrong_math() {
    let data = fixture();
    let prediction: Vec<_> = (0..=u16::MAX)
        .map(bf16::from_bits)
        .filter(|x| x.is_finite())
        .collect();
    let sample: Vec<_> = (0..prediction.len())
        .map(|i| bf16::from_f32([0., 1., -1., 0.5][i % 4]))
        .collect();
    for case in data["steps"].as_array().unwrap() {
        let delta = case["dt"].as_f64().unwrap() as f32;
        let actual = scheduler::step_bf16_host(&sample, &prediction, delta).unwrap();
        let bytes: Vec<_> = actual
            .iter()
            .flat_map(|x| x.to_bits().to_le_bytes())
            .collect();
        assert_eq!(
            Sha256::digest(&bytes)
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>(),
            case["sha256"].as_str().unwrap()
        );
        let wrong = sample
            .iter()
            .zip(&prediction)
            .zip(&actual)
            .filter(|((s, p), a)| {
                bf16::from_f32(s.to_f32() + delta * p.to_f32()).to_bits() != a.to_bits()
            })
            .count();
        assert_eq!(
            wrong,
            case["wrong_fp32_product_differences"].as_u64().unwrap() as usize
        );
    }
    assert!(scheduler::step_bf16_host(&sample[..1], &prediction[..2], -0.1).is_err());
    assert!(scheduler::step_bf16_host(&sample[..1], &[bf16::NAN], -0.1).is_err());
    assert!(scheduler::step_bf16_host(&sample[..1], &prediction[..1], 0.1).is_err());
}
