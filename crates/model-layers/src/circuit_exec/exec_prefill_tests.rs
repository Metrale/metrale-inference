// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The dense instance's prefill programs, built over the synthetic bindings and a
//! mock buffer arena: every bucket of both prefill modes and both GDN arms compiles, a pass of
//! any row count selects the program whose bucket holds it, and a pass launches exactly the
//! kernels its plan counts (so a bundled `ops::*` call covers the kernels it issues).
//!
//! Owner: model-layers circuit executor tests.
//! Invariants: none beyond the types.

use metrale_circuit::{AvailableKernels, Mode};
use metrale_gpu_runtime::buffers::BufferArena;
use metrale_gpu_runtime::gpu::mock::MockArg;
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, KernelHandle};

use super::bindings::{CircuitLayer, HeadBinding, MixerFacts};
use super::compile::Inputs;
use super::exec_fixture::{RECIPE, dense, fixed, layer_binding, ptr};
use super::exec_fixture_run::STATE_PITCH;
use super::kernels::KernelTable;
use super::prefill::{PrefillBoot, PrefillPrograms};
use super::program::{GdnState, LaunchKind, PrefillStep, SegmentOf, StepEnv};
use super::sources;
use crate::weight_map::DenseWeight;

/// 2026-10-03: The widest pass built: past the 1024-row edge of the attention projections.
const MAX_TOKENS: u64 = 1100;

struct Built {
    gpu: MockGpuBackend,
    programs: PrefillPrograms,
    layers: Vec<CircuitLayer>,
    /// 2026-10-04: The arena's staged token ids.
    token_ids: DevicePtr,
}

/// 2026-10-03: The served config of the dense checkpoint the instance names (its fixture), so
/// the arena and the emitters' sizes are the circuit's dims; the decode fixture's `config()` is
/// a smaller model whose arena rows are too narrow for a 27B prefill.
fn dense_config() -> metrale_config::ModelConfig {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../circuit/tests/fixtures/checkpoints/unsloth--Qwen3.8-27B-NVFP4/config.json");
    let json = std::fs::read_to_string(&path).unwrap();
    metrale_config::parse_config(&json).unwrap()
}

fn build() -> Built {
    let inst = sources::instance(RECIPE).unwrap();
    let loaded = metrale_circuit::load(&inst, sources::sources(&inst).unwrap()).unwrap();
    let gpu = MockGpuBackend::new();
    gpu.set_kernel_n_tile(KernelHandle(0xDEAD), 128);
    crate::layers::ops::w4a4_proj::prepare(&gpu).unwrap();
    let mut attn = 0;
    let layers: Vec<CircuitLayer> = (0..loaded.circuit.layer_kinds.len())
        .map(|i| {
            let l = layer_binding(&loaded.circuit, i, attn);
            attn += usize::from(matches!(l.mixer, MixerFacts::Attention(_)));
            l
        })
        .collect();
    let head = HeadBinding {
        embed: DenseWeight {
            weight: ptr(0x8f00_0000),
        },
        final_norm: DenseWeight {
            weight: ptr(0x9000_0000),
        },
        lm_head: dense(0x9100_0000),
        unmodelled: Vec::new(),
        batchm_max_rows: 8,
    };
    let cfg = dense_config();
    let arena = BufferArena::new(&cfg, MAX_TOKENS as usize, 4096, 16, 4, &gpu).unwrap();
    let available = AvailableKernels::all_named_by(&loaded.rules);
    let table = KernelTable::resolve(&gpu, &available);
    let fixed = fixed(attn);
    let programs = PrefillPrograms::build(
        &PrefillBoot {
            circuit: &loaded.circuit,
            rules: &loaded.rules,
            runtime: &loaded.runtime,
            available: &available,
            policy: &inst.policy,
            arena: &arena,
            max_tokens: MAX_TOKENS,
        },
        &Inputs {
            gpu: &gpu,
            config: &cfg,
            kernels: &table,
            fixed: &fixed,
            layers: &layers,
            head: &head,
            draft: None,
            arena: Some(&arena),
            lane: None,
        },
    )
    .unwrap();
    Built {
        gpu,
        programs,
        layers,
        token_ids: arena.token_ids(),
    }
}

/// 2026-10-03: Every row edge the legacy prefill switches a kernel at (the survey's ladder),
/// both sides of it, and the top.
const ROWS: [u64; 18] = [
    1, 2, 16, 17, 32, 33, 64, 65, 95, 96, 128, 129, 255, 256, 1023, 1024, 1099, 1100,
];

#[test]
fn every_row_count_of_both_modes_and_arms_selects_a_program_holding_it() {
    let b = build();
    for mode in Mode::PREFILL {
        for replay in [false, true] {
            for t in ROWS {
                let p = b.programs.select(mode, t, replay).unwrap();
                assert!(p.bucket.holds(t), "{mode:?} {t}: bucket {:?}", p.bucket);
                assert_eq!((p.mode, p.exact_replay), (mode, replay));
            }
            assert!(b.programs.select(mode, MAX_TOKENS + 1, replay).is_err());
        }
    }
    for p in &b.programs.programs {
        let covered: u64 = p.program.launches.iter().map(|l| l.covers as u64).sum();
        assert_eq!(
            covered,
            p.plan.launches() + p.plan.copies(),
            "{:?}",
            p.bucket
        );
    }
}

/// 2026-10-03: One GDN state per GDN layer.
fn states(layers: &[CircuitLayer]) -> Vec<Vec<GdnState>> {
    layers
        .iter()
        .enumerate()
        .map(|(i, l)| match l.mixer {
            MixerFacts::Gdn(_) => {
                let at = 0x5_0000_0000 + ((i as u64) << 24);
                vec![GdnState {
                    h: ptr(at),
                    conv: ptr(at + STATE_PITCH),
                    h_steps: [ptr(0); 3],
                    conv_steps: [ptr(0); 3],
                }]
            }
            MixerFacts::Attention(_) => Vec::new(),
        })
        .collect()
}

#[test]
fn a_pass_launches_exactly_the_kernels_its_plan_counts() {
    let b = build();
    let gdn = states(&b.layers);
    for mode in Mode::PREFILL {
        for t in [1, 17, 65, 300, 1024] {
            let p = b.programs.select(mode, t, false).unwrap();
            let before = b.gpu.launches_snapshot().len();
            let step = PrefillStep {
                tokens: t as u32,
                start: if mode == Mode::Prefill { 0 } else { 64 },
                kv_write_floor: 0,
                ids_row0: 0,
                meta: Some(fixed(0).meta),
            };
            p.program
                .run_segments(
                    |s| matches!(s, SegmentOf::Layer(_)),
                    &StepEnv {
                        gpu: &b.gpu,
                        stream: 7,
                        gdn: &gdn,
                        max_blocks_per_seq: 9,
                        prefill: Some(step),
                    },
                )
                .unwrap_or_else(|e| panic!("{mode:?} {t}: {e:#}"));
            let launched = b.gpu.launches_snapshot().len() - before;
            let want: usize = p
                .program
                .segments
                .iter()
                .filter(|s| matches!(s.of, SegmentOf::Layer(_)))
                .flat_map(|s| p.program.launches[s.launches.clone()].iter())
                .filter(|l| l.kind == LaunchKind::Kernel)
                .map(|l| l.covers)
                .sum();
            assert_eq!(launched, want, "{mode:?} {t}");
        }
    }
}

// 2026-10-04: The embed segment gathers the pass's rows from the staged ids at `ids_row0` into
// the stream, with the embedding table. Mutation: ignoring `ids_row0` embeds a cached prefix's
// tokens over the tail's rows; another table or output fails the pointer checks.
#[test]
fn the_embed_segment_gathers_the_pass_rows_from_the_staged_ids() {
    let b = build();
    for mode in Mode::PREFILL {
        let p = b.programs.select(mode, 17, false).unwrap();
        let before = b.gpu.launches_snapshot().len();
        let step = PrefillStep {
            tokens: 17,
            start: if mode == Mode::Prefill { 0 } else { 64 },
            kv_write_floor: 0,
            ids_row0: 5,
            meta: Some(fixed(0).meta),
        };
        p.program
            .run_segments(
                |s| s == SegmentOf::Embed,
                &StepEnv {
                    gpu: &b.gpu,
                    stream: 7,
                    gdn: &[],
                    max_blocks_per_seq: 9,
                    prefill: Some(step),
                },
            )
            .unwrap();
        let l = &b.gpu.launches_snapshot()[before..];
        assert_eq!(l.len(), 1, "{mode:?}");
        assert_eq!(l[0].grid, [17, 1, 1]);
        let want = [b.token_ids.offset(5 * 4), ptr(0x8f00_0000), fixed(0).hidden];
        for w in want {
            assert!(l[0].args.contains(&MockArg::Buffer(w)), "{mode:?}: {w:?}");
        }
    }
}
