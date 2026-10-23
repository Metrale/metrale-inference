// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The forced A/B verdict: a known per-step kernel delta is recovered whatever the
//! serve offsets, steps are matched by (request, output index), serves that replayed different
//! records are refused, and acceptance is compared on the same steps.

use super::*;

/// 2026-10-23: A deterministic noise in [-n, n).
fn noise(i: usize, n: f64) -> f64 {
    let x = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 11;
    (x as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) * n
}

/// 2026-10-23: One serve's replay of 200 steps of one request: base step 30000 us + the serve's
/// offset + `delta` on `arm` steps, arms alternating A B B A; own acceptance every `acc_every`.
fn serve(offset: f64, delta: f64, arms: usize, acc_every: usize) -> Vec<Timed> {
    let traj: Vec<TrajStep> = (0..200)
        .map(|i| TrajStep {
            key: 0x42,
            out: 2 + 2 * i,
            kv: 1000 + 2 * i,
            forced_acc: i % 2 == 0,
            own_acc: i % acc_every == 0,
            agree0: true,
        })
        .collect();
    let order = [0usize, 1, 1, 0];
    let steps: Vec<Step> = traj
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let arm = if arms == 1 { 0 } else { order[i % 4] };
            Step {
                step: i as u64,
                arm,
                kv: t.kv,
                us: 30000.0 + offset + noise(i, 300.0) + if arm == 1 { delta } else { 0.0 },
            }
        })
        .collect();
    join(&steps, &traj)
}

fn plan(arm: Option<usize>) -> ForcedPlan {
    ForcedPlan {
        lever_arm: arm,
        warm: 4,
        margin: 0.0025,
        huber_k: 1.345,
        reps: 400,
        seed: 7,
    }
}

#[test]
fn join_keeps_order_and_the_previous_arm() {
    let v = serve(0.0, 0.0, 2, 1);
    assert_eq!(v.len(), 200);
    assert_eq!(v[0].prev_arm, None);
    assert_eq!((v[2].arm, v[2].prev_arm), (1, Some(1)));
    let by = acceptance_by_arm(&v, 2);
    assert_eq!(by[1].0, 50, "only B steps after a B step count for B");
}

/// 2026-10-23: The lever arm is 60 us faster on every step; the pairs carry serve offsets of
/// +-200 us. The pair-level mean recovers -(-60) = +60 within its interval.
#[test]
fn a_known_kernel_delta_is_recovered_through_serve_offsets() {
    let pairs = vec![
        (serve(200.0, 0.0, 1, 3), serve(0.0, -60.0, 2, 3)),
        (serve(-100.0, 0.0, 1, 3), serve(-50.0, -60.0, 2, 3)),
        (serve(0.0, 0.0, 1, 3), serve(120.0, -60.0, 2, 3)),
    ];
    let v = judge_forced(&pairs, plan(Some(1))).unwrap();
    assert_eq!(v.len(), 1);
    let r = &v[0];
    let offsets_mean = (200.0 - 100.0 + 0.0 - (0.0 - 50.0 + 120.0)) / 3.0;
    assert!((r.d_us - (60.0 + offsets_mean)).abs() < 15.0, "{}", r.d_us);
    let (lo, hi) = r.ci.unwrap();
    assert!(lo < r.d_us && r.d_us < hi);
    // 2026-10-23: Each pair alone sees offset + delta; the across-pair interval is wide.
    assert!(
        hi - lo > 100.0,
        "the serve offset is in the interval: {lo}..{hi}"
    );
}

#[test]
fn identical_acceptance_has_no_discordant_steps_and_any_arm_takes_every_step() {
    let pairs = vec![
        (serve(0.0, 0.0, 1, 3), serve(0.0, 0.0, 1, 3)),
        (serve(0.0, 0.0, 1, 3), serve(0.0, 0.0, 1, 3)),
    ];
    let r = &judge_forced(&pairs, plan(None)).unwrap()[0];
    assert_eq!(r.discordant, (0, 0));
    assert_eq!(r.acc.2, 2 * 196);
    assert_eq!(r.mcnemar_p, 1.0);
    // 2026-10-23: The lever accepts on every 2nd step, the control on every 3rd.
    let pairs = vec![(serve(0.0, 0.0, 1, 3), serve(0.0, 0.0, 1, 2))];
    let r = &judge_forced(&pairs, plan(None)).unwrap()[0];
    assert!(
        r.acc.1 > r.acc.0 && r.mcnemar_p < 0.01,
        "{:?} p {}",
        r.acc,
        r.mcnemar_p
    );
}

#[test]
fn serves_that_replayed_different_records_are_refused() {
    let c = serve(0.0, 0.0, 1, 3);
    let mut l = serve(0.0, 0.0, 1, 3);
    l[10].t.kv += 1;
    assert!(judge_forced(&[(c, l)], plan(None)).is_err());
    assert!(judge_forced(&[], plan(None)).is_err());
}

#[test]
fn replay_lines_parse() {
    let t = parse_traj(
        "x PAIRED_TRAJ replay k2 key=00000000000000ff out=3 kv=1001 own_draft=5 v0=5 v1=6 \
         forced_acc=1 own_acc=1 agree0=0 agree1=true\nPAIRED_TRAJ rec k2 key=1 out=1\n",
    )
    .unwrap();
    assert_eq!(
        t,
        vec![TrajStep {
            key: 0xff,
            out: 3,
            kv: 1001,
            forced_acc: true,
            own_acc: true,
            agree0: false
        }]
    );
}
