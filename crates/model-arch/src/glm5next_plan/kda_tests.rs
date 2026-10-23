// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The CPU equivalence gate of the plan-executed KDA mixers: on the recording device,
//! every KDA layer of every rank run from its plan (pack, gate chain, conv, recurrence and, at one
//! row, the output norm; the W4A16 projections the layer's own) puts the same operations on the
//! same streams as `decode_k` under the served levers, at decode n=1 and at verify n=2 with its
//! snapshot. One changed field of the plan turns it red; a plan at another snapshot count is not
//! taken; installation refuses a lever the rules do not model.
//!
//! The KDA front end and `o` choose their launches from the process-wide dense tier
//! (`--dense-quantization w4a16`, published once per process), which the rest of this test binary
//! runs without. The gates therefore run in a child process of this binary that publishes the
//! served tier first ([`in_w4a16_process`]); nothing on either path is mocked.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use metrale_circuit::Circuit;
use metrale_gpu_runtime::gpu::recording::{Op, RecordingGpu};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_model_layers::layers::{DenseQuantization, DsaQuantization};

use super::*;
use crate::glm5next_kda::plan::{ConvStep, KdaSteps, RecurStep};
use crate::glm5next_kda::{
    Glm5NextKdaConfig, Glm5NextKdaKernels, Glm5NextKdaLayer, Glm5NextKdaWeights,
    Glm5NextKdaWorkspace, KdaSeqState,
};
use crate::glm5next_layer::GlmLevers;
use crate::glm5next_plan::fixture::{RUNS, model_config, plan, ranks, served_levers};
use crate::glm5next_plan::fixture_site::{SMS, STREAM, gate};
use crate::glm5next_plan::{compile_kda, kda};
use metrale_model_layers::weight_map::DenseWeight;

const HIDDEN_IN: DevicePtr = DevicePtr(0x9200_0000);

/// 2026-10-23: Run `body` in a child process of this test binary (only the test `name`) that
/// publishes `--dense-quantization w4a16` before anything reads the tier.
fn in_w4a16_process(name: &str, body: impl FnOnce()) {
    let publish = || {
        let q =
            metrale_model_layers::layers::set_dense_quantization_from_cli(DenseQuantization::W4a16);
        assert_eq!(
            q,
            DenseQuantization::W4a16,
            "the child published the tier first"
        );
    };
    let path = format!("glm5next_plan::kda::tests::{name}");
    crate::glm5next_plan::fixture::in_child_process(&path, publish, body);
}

/// 2026-10-23: The KDA layers of `c`.
fn kda_layers(c: &Circuit) -> Vec<usize> {
    let mut v: Vec<usize> = c
        .nodes
        .iter()
        .filter(|n| n.block == "kda")
        .filter_map(|n| n.layer)
        .collect();
    v.dedup();
    v
}

struct Mixer<'a> {
    gpu: &'a RecordingGpu,
    layer: Glm5NextKdaLayer,
    ws: Glm5NextKdaWorkspace,
    state: KdaSeqState,
    snap: (DevicePtr, DevicePtr),
}

/// 2026-10-23: Rank `rank`'s KDA mixer `idx` with `heads` local heads under `levers`, its
/// projections registered on the served tiers as the loader registers them. The dense registries
/// are process-wide and keyed by address, so every mixer of a process allocates on one device.
fn mixer(
    gpu: &RecordingGpu,
    rank: usize,
    idx: usize,
    heads: usize,
    levers: GlmLevers,
) -> Mixer<'_> {
    let mut cfg = Glm5NextKdaConfig::from_model_config(&model_config(rank));
    cfg.heads = heads;
    let w = || DenseWeight {
        weight: gpu.alloc(64).unwrap(),
    };
    let mut weights = Glm5NextKdaWeights {
        q_proj: w(),
        k_proj: w(),
        v_proj: w(),
        conv: w(),
        f_a: w(),
        f_b: w(),
        dt_bias: gpu.alloc(64).unwrap(),
        a_log: gpu.alloc(64).unwrap(),
        b_proj: w(),
        g_a: w(),
        g_b: w(),
        o_norm: w(),
        o_proj: w(),
    };
    let kernels = Glm5NextKdaKernels::resolve(gpu).unwrap();
    let projs = Glm5NextKdaLayer::new(idx, cfg, weights_view(&weights), kernels, levers)
        .unwrap()
        .dense_projections();
    {
        let x = &mut weights;
        let mut slots = [
            &mut x.q_proj.weight,
            &mut x.k_proj.weight,
            &mut x.v_proj.weight,
            &mut x.f_a.weight,
            &mut x.f_b.weight,
            &mut x.b_proj.weight,
            &mut x.g_a.weight,
            &mut x.g_b.weight,
            &mut x.o_proj.weight,
        ];
        crate::weight_loader::glm5_next_load::dense_tiers::register_projections(
            gpu,
            projs,
            &mut slots,
            DenseQuantization::W4a16,
            DsaQuantization::Declared,
            &levers.dense,
            idx,
        )
        .unwrap();
    }
    let layer = Glm5NextKdaLayer::new(idx, cfg, weights, kernels, levers).unwrap();
    let ws = Glm5NextKdaWorkspace::new(gpu, &cfg, 16).unwrap();
    let state = KdaSeqState {
        conv: gpu.alloc(cfg.conv_state_elems() * 4).unwrap(),
        recurrent: gpu.alloc(cfg.recurrent_state_elems() * 4).unwrap(),
    };
    let snap = (
        gpu.alloc(cfg.recurrent_state_elems() * 4).unwrap(),
        gpu.alloc(cfg.conv_state_elems() * 4).unwrap(),
    );
    gpu.take();
    Mixer {
        gpu,
        layer,
        ws,
        state,
        snap,
    }
}

/// 2026-10-23: A copy of the weight pointers (the layer owns its weights).
fn weights_view(w: &Glm5NextKdaWeights) -> Glm5NextKdaWeights {
    Glm5NextKdaWeights {
        q_proj: w.q_proj,
        k_proj: w.k_proj,
        v_proj: w.v_proj,
        conv: w.conv,
        f_a: w.f_a,
        f_b: w.f_b,
        dt_bias: w.dt_bias,
        a_log: w.a_log,
        b_proj: w.b_proj,
        g_a: w.g_a,
        g_b: w.g_b,
        o_norm: w.o_norm,
        o_proj: w.o_proj,
    }
}

impl Mixer<'_> {
    /// 2026-10-23: `rows` rows (a verify keeps one snapshot per row but the last) from `plan`, or
    /// `decode_k` without one.
    fn run(&self, rows: usize, plan: Option<&KdaSteps<KernelHandle>>) -> anyhow::Result<Vec<Op>> {
        let snaps: Vec<_> = std::iter::repeat_n(self.snap, rows - 1).collect();
        let (l, g) = (&self.layer, self.gpu);
        match plan {
            Some(p) => {
                l.decode_k_plan(g, HIDDEN_IN, rows, &self.state, &self.ws, &snaps, p, STREAM)?
            }
            None => l.decode_k(g, HIDDEN_IN, rows, &self.state, &self.ws, &snaps, STREAM)?,
        }
        Ok(g.take())
    }

    fn bind(&self, s: KdaSteps<metrale_circuit::KernelId>) -> KdaSteps<KernelHandle> {
        s.map(|k| metrale_model_layers::circuit_exec::family::plan_kernel(self.gpu, &k))
            .unwrap()
    }
}

fn kernels(ops: &[Op]) -> Vec<&str> {
    ops.iter().filter_map(|o| o.kernel()).collect()
}

/// 2026-10-23: Every KDA mixer of every rank, at decode n=1 and verify n=2, launches from its
/// plan exactly what `decode_k` launches under the served levers: the same stream graph and the
/// same operations. The plan steps are the served forms (one-row conv and register-resident
/// recurrence at one row, the snapshotting token kernels at two), the projections the declared
/// legacy sites.
#[test]
fn every_kda_mixer_launches_what_decode_k_launches() {
    in_w4a16_process("every_kda_mixer_launches_what_decode_k_launches", || {
        let (c, rules, policy) = ranks();
        let gpu = RecordingGpu::new(SMS);
        let mut mixers = 0;
        for rank in 0..3 {
            let layers = kda_layers(&c[rank]);
            assert!(
                layers.len() > 30,
                "rank {rank}: {} KDA layers",
                layers.len()
            );
            let heads = c[rank].dims["lin_v_heads"] as usize;
            assert_eq!(heads, [22, 22, 20][rank], "the w4a16 head split");
            let built: Vec<Mixer> = layers
                .iter()
                .map(|&l| mixer(&gpu, rank, l, heads, served_levers()))
                .collect();
            for (mode, rows) in RUNS {
                let p = plan(&c, &rules, &policy, rank, mode, rows);
                let rows = rows as usize;
                for (&l, m) in layers.iter().zip(&built) {
                    let site = compile_kda(&c[rank], &p, l)
                        .unwrap_or_else(|e| panic!("rank {rank} {mode:?} layer {l}: {e:#}"));
                    let want_legacy: &[&str] = if rows == 1 {
                        &["kda [q,k,v,f_a,b,g_a]", "kda [o]"]
                    } else {
                        &["kda [q,k,v,f_a,b,g_a]", "kda [out_norm,o]"]
                    };
                    assert_eq!(site.legacy, want_legacy, "rank {rank} n={rows} layer {l}");
                    let steps = m.bind(site.steps.clone());
                    assert!(steps.takes(rows, rows - 1));
                    let legacy = m.run(rows, None).unwrap();
                    let planned = m.run(rows, Some(&steps)).unwrap();
                    if let Some(d) = gate(&legacy, &planned) {
                        panic!("rank {rank} n={rows} layer {l}: {d}");
                    }
                    let ks = kernels(&planned);
                    let own = |e: (&str, &str)| format!("{}::{}", e.0, e.1);
                    use crate::glm5next_kda::entry as e;
                    let want: Vec<String> = if rows == 1 {
                        vec![
                            own(e::PACK),
                            own(e::GATES_ROWS),
                            own(e::CONV_ROW),
                            own(e::RECUR_REG),
                            own(e::O_NORM),
                        ]
                    } else {
                        vec![
                            own(e::PACK),
                            own(e::GATES_ROWS),
                            own(e::CONV_WINDOW_TO),
                            own(e::CONV_TOKENS),
                            own(e::CONV_WINDOW),
                            own(e::RECUR_SNAP),
                            own(e::O_NORM),
                        ]
                    };
                    let mine: Vec<&str> = ks
                        .iter()
                        .copied()
                        .filter(|k| want.iter().any(|w| w == k))
                        .collect();
                    assert_eq!(
                        mine, want,
                        "rank {rank} n={rows}: the plan's own launches, in order"
                    );
                    mixers += 1;
                }
            }
        }
        println!("{mixers} KDA mixer runs gated");
    });
}

/// 2026-10-23: One changed field of a bound plan turns the gate red at that launch: the gate
/// entry, the conv entry, the recurrence entry, the pack entry, the output norm entry, and the
/// snapshot advance (here the window advance named in its place).
#[test]
fn a_changed_plan_field_turns_the_gate_red() {
    in_w4a16_process("a_changed_plan_field_turns_the_gate_red", || {
        let (c, rules, policy) = ranks();
        let l = kda_layers(&c[0])[0];
        let heads = c[0].dims["lin_v_heads"] as usize;
        let gpu = RecordingGpu::new(SMS);
        let m = mixer(&gpu, 0, l, heads, served_levers());
        for (mode, rows) in RUNS {
            let p = plan(&c, &rules, &policy, 0, mode, rows);
            let rows = rows as usize;
            let good = m.bind(compile_kda(&c[0], &p, l).unwrap().steps);
            let legacy = m.run(rows, None).unwrap();
            assert_eq!(gate(&legacy, &m.run(rows, Some(&good)).unwrap()), None);
            let other = m
                .gpu
                .kernel("kda_layer_ops", "kda_sigmoid_bf16_f32")
                .unwrap();
            let mut bad: Vec<(&str, KdaSteps<KernelHandle>)> = vec![
                (
                    "gates",
                    KdaSteps {
                        gates: other,
                        ..good
                    },
                ),
                (
                    "pack",
                    KdaSteps {
                        pack: other,
                        ..good
                    },
                ),
            ];
            match good.conv {
                ConvStep::Row(_) => {
                    bad.push((
                        "conv",
                        KdaSteps {
                            conv: ConvStep::Row(other),
                            ..good
                        },
                    ));
                    bad.push((
                        "recur",
                        KdaSteps {
                            recur: RecurStep::Row(other),
                            ..good
                        },
                    ));
                    bad.push((
                        "out_norm",
                        KdaSteps {
                            out_norm: Some(other),
                            ..good
                        },
                    ));
                }
                ConvStep::TokensSnap { tokens, window, .. } => {
                    let conv = ConvStep::TokensSnap {
                        advance_to: window,
                        tokens,
                        window,
                    };
                    bad.push(("advance_to", KdaSteps { conv, ..good }));
                    bad.push((
                        "recur",
                        KdaSteps {
                            recur: RecurStep::Snap(other),
                            ..good
                        },
                    ));
                }
            }
            for (what, steps) in bad {
                let d = gate(&legacy, &m.run(rows, Some(&steps)).unwrap());
                assert!(d.is_some(), "n={rows}: a changed {what} stayed green");
            }
        }
    });
}

/// 2026-10-23: The layer takes a plan only at its row count and snapshot form (a two-row prefill
/// piece keeps no snapshot: `decode_k`'s token kernels, not the verify's), and the plan path
/// refuses a run it was not compiled for rather than launching it.
#[test]
fn a_plan_runs_only_its_rows_and_snapshots() {
    let (c, rules, policy) = ranks();
    let l = kda_layers(&c[0])[0];
    let one = compile_kda(&c[0], &plan(&c, &rules, &policy, 0, Mode::Decode, 1), l)
        .unwrap()
        .steps;
    let two = compile_kda(&c[0], &plan(&c, &rules, &policy, 0, Mode::Verify, 2), l)
        .unwrap()
        .steps;
    assert!(one.takes(1, 0) && !one.takes(1, 1) && !one.takes(2, 1));
    assert!(two.takes(2, 1) && !two.takes(2, 0) && !two.takes(1, 0) && !two.takes(3, 2));
    assert!(matches!(one.conv, ConvStep::Row(_)) && one.out_norm.is_some());
    assert!(matches!(two.conv, ConvStep::TokensSnap { .. }) && two.out_norm.is_none());
}

/// 2026-10-23: A plan conv naming the window advance where the snapshot advance belongs, or one
/// advance short, is refused at compile, naming the kernel or the count.
#[test]
fn a_verify_conv_of_another_form_is_refused() {
    let (c, rules, policy) = ranks();
    let l = kda_layers(&c[0])[0];
    let mut p = plan(&c, &rules, &policy, 0, Mode::Verify, 2);
    let i = PlanGroup::all(&c[0], &p)
        .position(|g| g.layer() == Some(l) && g.locals() == ["short_conv", "conv_ckpt", "l2"])
        .unwrap();
    let saved = p.groups[i].kernels.clone();
    p.groups[i].kernels[0].func = "causal_conv1d_window_advance".into();
    let e = compile_kda(&c[0], &p, l).err().unwrap();
    assert!(
        format!("{e:#}").contains("causal_conv1d_window_advance_to"),
        "{e:#}"
    );
    p.groups[i].kernels = saved[1..].to_vec();
    let e = compile_kda(&c[0], &p, l).err().unwrap();
    assert!(format!("{e:#}").contains("names 2 kernels"), "{e:#}");
}

/// 2026-10-23: Installation refuses a KDA lever the rules do not model, naming it, and clears on
/// `None`; under the served levers it installs both runs.
#[test]
fn installation_refuses_unmodelled_levers() {
    let mut levers = served_levers();
    assert!(
        kda::unmodelled(&levers).is_empty(),
        "the served levers are modelled"
    );
    levers.kda.gates_fused = true;
    levers.kda.conv_snap_to = false;
    let refused = kda::unmodelled(&levers).join("; ");
    assert!(
        refused.contains("GATES_FUSED") && refused.contains("CONV_SNAP_TO"),
        "{refused}"
    );
    levers = served_levers();
    levers.kda.gates_rows = false;
    assert!(kda::unmodelled(&levers)[0].contains("GATES_ROWS"));
    // 2026-10-23: The one-launch conv the rules do not name (bit-identical, but inert under a plan).
    levers = served_levers();
    levers.kda.conv_one = true;
    assert!(kda::unmodelled(&levers)[0].contains("CONV_ONE"));
}
