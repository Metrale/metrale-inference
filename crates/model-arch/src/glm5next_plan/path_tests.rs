// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The all-reduce path as a plan field (`metrale_circuit::Group::path`): the path
//! every plain collective of a GLM rank's plans names is the communicator's own
//! (`metrale_comm::ar_path`) for its payload, at 2, 3, 4 and 8 ranks and on both sides of the
//! one-shot bound; and the boot check (`circuit_exec::collectives::check_paths`) passes the served
//! plans at every admitted N and refuses a plan whose path the communicator does not take.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use metrale_circuit::collective::{ONESHOT_MAX_KB, payload_bytes};
use metrale_circuit::{AvailableKernels, CollectivePath, FusionPlan, Mode, Policy};
use metrale_comm::ar_path::{PathState, ar_path_of};
use metrale_error::ErrorCode;
use metrale_model_layers::circuit_exec::collectives::check_paths;

use super::fixture::{ranks_at, worlds};
use super::install::{DRAFT_GATED, GATED};
use super::world_tests::capped_layout;

/// 2026-10-23: A communicator over `world` ranks with both kernels set and a `kb` KiB one-shot
/// bound: the state the serve's `NcclBackend` reports (its 2-rank buffer bounds no path choice).
fn comm(world: usize, kb: u64) -> PathState {
    PathState {
        world_size: world,
        add_kernel_set: true,
        sum_kernel_set: true,
        oneshot_max_bytes: usize::try_from(kb * 1024).unwrap(),
        recv_capacity: usize::MAX,
    }
}

fn under(policy: &Policy, kb: u64) -> Policy {
    let mut p = policy.clone();
    p.settings.insert(ONESHOT_MAX_KB.into(), kb.to_string());
    p
}

/// 2026-10-23: The kernels each path's rule launches.
fn kernels_of(path: CollectivePath) -> &'static [&'static str] {
    match path {
        CollectivePath::TwoRank => &["bf16_add_inplace"],
        CollectivePath::OneShot => &["bf16_add_rank_sum"],
        CollectivePath::Nccl => &[],
    }
}

/// 2026-10-23: The conformance gate between the rules and the communicator. At 2, 3, 4 and 8
/// ranks, under one-shot bounds of 0 (off), 8 KiB (exactly one 4096-wide BF16 row: the inclusive
/// edge), 16 KiB and the recipe's 1024 KiB, every plain collective group (`tp_all_reduce`) of
/// every rank's decode, verify and (where the vocab splits, 2..=4) draft plans names the path
/// `ar_path` takes for its payload and that path's kernels. The mHC sites' all-reduces are plain
/// here (`glm_ar_fused_post` off). Both sides of the bound and its edge must be exercised.
///
/// The one gap: at two ranks the rules take the 2-rank exchange only (the one-shot rules take
/// three or more, Fable 2026-10-23), so a plan never names the one-shot there. `ar_path` takes it
/// at two ranks only when `bf16_add_inplace` is absent and `bf16_add_rank_sum` present, which
/// cannot happen: both are in module `bf16_add`. Every state here has both.
#[test]
fn every_plain_collective_names_the_communicators_path_for_its_payload() {
    let mut seen = std::collections::BTreeSet::new();
    let mut at_edge = false;
    for world in [2u64, 3, 4, 8] {
        let (c, rules, policy) = capped_layout(world);
        let mut shapes = vec![(Mode::Decode, 1), (Mode::Verify, 2)];
        if world <= 4 {
            shapes.push((Mode::Draft, 1));
        }
        let avail = AvailableKernels::all_named_by(&rules);
        let w = usize::try_from(world).unwrap();
        for kb in [0u64, 8, 16, 1024] {
            let mut policy = under(&policy, kb);
            policy
                .settings
                .insert("glm_ar_fused_post".into(), "off".into());
            for (rank, circuit) in c.iter().enumerate() {
                for &(mode, rows) in &shapes {
                    let at = format!("world {world} kb {kb} rank {rank} {mode:?} {rows}");
                    let p = metrale_circuit::fuse(circuit, &rules, &avail, &policy, mode, rows)
                        .unwrap_or_else(|e| panic!("{at}: {e}"));
                    for g in p.groups.iter().filter(|g| g.emitter == "tp_all_reduce") {
                        let n = &circuit.nodes[g.nodes[0]];
                        let bytes = payload_bytes(circuit, n, rows).unwrap();
                        let want = ar_path_of(&comm(w, kb), usize::try_from(bytes).unwrap());
                        assert_eq!(g.path, Some(want), "{at} {} ({bytes} B): rule `{}`", n.id, g.rule);
                        let names: Vec<&str> = g.kernels.iter().map(|k| k.func.as_str()).collect();
                        assert_eq!(names, kernels_of(want), "{at} {}: rule `{}`", n.id, g.rule);
                        seen.insert((world > 2, want));
                        at_edge |= kb > 0 && bytes == kb * 1024 && want == CollectivePath::OneShot;
                    }
                }
            }
        }
    }
    let want = [
        (false, CollectivePath::TwoRank),
        (true, CollectivePath::OneShot),
        (true, CollectivePath::Nccl),
    ];
    assert_eq!(seen.into_iter().collect::<Vec<_>>(), want);
    assert!(at_edge, "no payload sat exactly on a bound");
}

/// 2026-10-23: The served plans of one rank at the boot's shapes (decode 1, verify 2, draft 1).
fn served_plans(world: usize, rank: usize) -> (metrale_circuit::Circuit, Vec<FusionPlan>) {
    let (c, rules, policy) = ranks_at(world).unwrap();
    let avail = AvailableKernels::all_named_by(&rules);
    let plans = GATED
        .iter()
        .chain(&DRAFT_GATED)
        .map(|&(mode, rows)| {
            metrale_circuit::fuse(&c[rank], &rules, &avail, &policy, mode, rows).unwrap()
        })
        .collect();
    (c[rank].clone(), plans)
}

/// 2026-10-23: At every admitted N (2, 3, 4) every rank's served plans pass the boot check
/// against a communicator at the recipe's bound (positive). Against one with the one-shot off
/// (a serve without `--all-reduce-oneshot-max-kb`) they are refused at the first mHC fused
/// exchange (`glm_ar_fused_post`), at every N (negative): the TP=2 bit-gate's runtime failure,
/// "no one-shot exchange for 8192 bytes", now a boot refusal.
#[test]
fn the_boot_check_passes_the_served_plans_and_refuses_another_bound() {
    for world in worlds() {
        for rank in 0..world {
            let (c, plans) = served_plans(world, rank);
            let n = check_paths(&c, &plans, &comm(world, 1024))
                .unwrap_or_else(|e| panic!("world {world} rank {rank}: {e}"));
            assert!(n > 0, "world {world} rank {rank}: no collective checked");
            let e = check_paths(&c, &plans, &comm(world, 0)).unwrap_err();
            assert_eq!(e.code(), ErrorCode::CollectivePathRefused, "{e}");
            assert!(e.to_string().contains("rule `glm_ar_fused_post`"), "{e}");
        }
    }
}

/// 2026-10-23: The TP=2 bit-gate's case: two ranks, a one-shot bound below one 4096-wide BF16
/// row (8192 B), plans that name the one-shot exchange for the mHC post-mix: refused at boot
/// naming the group, its 8192 bytes and the bound. At 8 KiB the decode row fits but the 2-row
/// verify (16384 B) does not; at 16 KiB every exchange fits.
#[test]
fn a_two_rank_bound_below_the_mhc_exchange_refuses_the_boot() {
    let (c, plans) = served_plans(2, 0);
    let fused = plans
        .iter()
        .flat_map(|p| &p.groups)
        .filter(|g| g.rule == "glm_ar_fused_post")
        .inspect(|g| assert_eq!(g.path, Some(CollectivePath::OneShot)))
        .count();
    assert!(fused > 0, "the served TP=2 plans fuse the mHC exchange");
    let e = check_paths(&c, &plans, &comm(2, 4)).unwrap_err();
    assert_eq!(e.code(), ErrorCode::CollectivePathRefused);
    let text = e.to_string();
    for part in ["the one_shot path cannot run 8192 bytes", "rule `glm_ar_fused_post`", "1..=4096 bytes at 2 ranks"] {
        assert!(text.contains(part), "`{part}` not in {text}");
    }
    let e = check_paths(&c, &plans, &comm(2, 8)).unwrap_err();
    assert!(e.to_string().contains("cannot run 16384 bytes"), "{e}");
    check_paths(&c, &plans, &comm(2, 16)).unwrap();
}

/// 2026-10-23: The plain collectives' drift: plans fused under the recipe's bound with the mHC
/// exchange unfused (`glm_ar_fused_post` off) name the one-shot; a communicator with it off
/// takes NCCL at three or more ranks, so the boot is refused as a mismatch. At two ranks the
/// 2-rank exchange takes every payload whatever the bound.
#[test]
fn plain_collectives_fused_under_another_bound_mismatch_at_three_or_more() {
    for world in worlds() {
        let (c, rules, mut policy) = ranks_at(world).unwrap();
        policy
            .settings
            .insert("glm_ar_fused_post".into(), "off".into());
        let avail = AvailableKernels::all_named_by(&rules);
        let plans: Vec<FusionPlan> = [(Mode::Decode, 1), (Mode::Verify, 2)]
            .iter()
            .map(|&(m, r)| metrale_circuit::fuse(&c[0], &rules, &avail, &policy, m, r).unwrap())
            .collect();
        check_paths(&c[0], &plans, &comm(world, 1024)).unwrap();
        let off = check_paths(&c[0], &plans, &comm(world, 0));
        if world == 2 {
            off.unwrap();
        } else {
            let e = off.unwrap_err();
            assert_eq!(e.code(), ErrorCode::CollectivePathMismatch);
            assert!(
                e.to_string()
                    .contains("the plan names the one_shot path, the communicator takes nccl"),
                "{e}"
            );
        }
    }
}

/// 2026-10-23: A plan whose plain collective names another path than the communicator's is
/// refused naming the group, its payload and both paths, at every admitted N.
#[test]
fn a_corrupted_plan_path_refuses_the_boot() {
    for world in worlds() {
        let (c, mut plans) = served_plans(world, 0);
        let (p, g) = plans
            .iter()
            .enumerate()
            .find_map(|(p, plan)| {
                plan.groups
                    .iter()
                    .position(|g| g.path.is_some() && g.nodes.len() == 1)
                    .map(|g| (p, g))
            })
            .expect("a planned plain collective");
        let planned = plans[p].groups[g].path.unwrap();
        let wrong = if planned == CollectivePath::Nccl {
            CollectivePath::OneShot
        } else {
            CollectivePath::Nccl
        };
        plans[p].groups[g].path = Some(wrong);
        let e = check_paths(&c, &plans, &comm(world, 1024)).unwrap_err();
        assert_eq!(e.code(), ErrorCode::CollectivePathMismatch);
        let text = e.to_string();
        let node = &c.nodes[plans[p].groups[g].nodes[0]];
        let bytes = payload_bytes(&c, node, plans[p].rows).unwrap();
        for part in [
            format!("g{g:04} `{}`", node.id),
            format!("({bytes} bytes)"),
            format!("names the {wrong} path, the communicator takes {planned}"),
        ] {
            assert!(text.contains(&part), "world {world}: `{part}` not in {text}");
        }
    }
}

/// 2026-10-23: A plan that would take a one-shot rule under a policy that states no bound is
/// refused naming the rule and the setting, never fused under a guessed bound (PCND); a value
/// that is not a whole number is refused too.
#[test]
fn an_unstated_or_malformed_bound_refuses_the_plan() {
    let (c, rules, mut policy) = ranks_at(3).unwrap();
    let avail = AvailableKernels::all_named_by(&rules);
    policy.settings.remove(ONESHOT_MAX_KB);
    let e = metrale_circuit::fuse(&c[0], &rules, &avail, &policy, Mode::Draft, 1).unwrap_err();
    let text = e.to_string();
    assert!(
        text.contains(&format!("reads setting `{ONESHOT_MAX_KB}`")) && text.contains("rule `tp_all_"),
        "{text}"
    );
    let e = metrale_circuit::fuse(&c[0], &rules, &avail, &under(&policy, 0), Mode::Draft, 1);
    assert!(e.is_ok(), "a stated 0 (one-shot off) fuses: {e:?}");
    policy.settings.insert(ONESHOT_MAX_KB.into(), "1M".into());
    let e = metrale_circuit::fuse(&c[0], &rules, &avail, &policy, Mode::Draft, 1).unwrap_err();
    assert!(e.to_string().contains("`1M` is not a whole number"), "{e}");
}

/// 2026-10-23: A plan decoded without its paths (a blueprint body stores none; JSON skips the
/// field as postcard does) gets back exactly the fused paths from its rules
/// (`collective::derive_paths`), at every admitted N; a rule set without a group's rule refuses.
#[test]
fn decoded_plans_rederive_the_fused_paths_from_their_rules() {
    for world in worlds() {
        let (_, rules, _) = ranks_at(world).unwrap();
        let (_, plans) = served_plans(world, 0);
        assert!(plans.iter().any(|p| p.groups.iter().any(|g| g.path.is_some())));
        for plan in &plans {
            let mut back: FusionPlan =
                serde_json::from_str(&serde_json::to_string(plan).unwrap()).unwrap();
            assert!(back.groups.iter().all(|g| g.path.is_none()), "the path is not stored");
            metrale_circuit::collective::derive_paths(&mut back, &rules).unwrap();
            assert_eq!(&back, plan, "world {world} {:?} {}", plan.mode, plan.rows);
        }
        let mut back: FusionPlan =
            serde_json::from_str(&serde_json::to_string(&plans[0]).unwrap()).unwrap();
        let e = metrale_circuit::collective::derive_paths(&mut back, &rules[..1]).unwrap_err();
        assert!(e.to_string().contains("is not in the rule set"), "{e}");
    }
}
