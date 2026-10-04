// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The dense 27B instance with LoRA adapters on every projection each layer can
//! adapt (`metrale_circuit::lora::adapt`), planned under `lora_active = on` at the shapes LoRA
//! phase 1 serves; and the rendered plans the LoRA goldens under kernels/circuits/plans/lora/
//! pin.
//!
//! Owner: metrale-circuit tests (FEATURES workstream).
//! Invariants: none beyond the types.

use metrale_circuit::lora::{ACTIVE_SETTING, ADAPTABLE, LoraSpec, adapt};
use metrale_circuit::{Instance, Loaded, Mode, OpKind};

/// 2026-10-03: The adapted instance.
pub const RECIPE: &str = "qwen3.8/qwen3.8-27b-nvfp4-unsloth";

/// 2026-10-03: The shapes LoRA phase 1 serves (`circuit_exec::lora::PHASE1_MAX_ROWS`).
pub const SHAPES: [(Mode, u64); 8] = [
    (Mode::Decode, 1),
    (Mode::MultiSeq, 2),
    (Mode::MultiSeq, 3),
    (Mode::MultiSeq, 4),
    (Mode::Verify, 2),
    (Mode::Verify, 3),
    (Mode::Verify, 4),
    (Mode::Draft, 1),
];

/// 2026-10-03: The instance under `lora_active = on`, and its circuit adapted at rank 16 on
/// every adaptable projection of every layer.
pub fn adapted() -> (Instance, Loaded) {
    let mut inst = super::instances()
        .into_iter()
        .find(|i| i.recipe == RECIPE)
        .expect("the dense instance");
    inst.policy
        .settings
        .insert(ACTIVE_SETTING.to_string(), "on".to_string());
    let mut loaded = super::load(&inst);
    let c = &loaded.circuit;
    let spec = LoraSpec {
        rank: 16,
        layers: (0..c.layer_kinds.len())
            .map(|l| {
                let roles = ADAPTABLE
                    .into_iter()
                    .filter(|r| {
                        c.nodes
                            .iter()
                            .any(|n| n.layer == Some(l) && n.op == OpKind::Linear(*r))
                    })
                    .collect();
                (l, roles)
            })
            .collect(),
    };
    loaded.circuit = adapt(&loaded.circuit, &spec).expect("the dense circuit adapts");
    (inst, loaded)
}

/// 2026-10-03: The LoRA goldens: (path under kernels/circuits/plans/, rendered plan).
pub fn plans() -> Vec<(String, String)> {
    let (inst, loaded) = adapted();
    let avail = super::available(&inst, &loaded.rules);
    let fams = super::families(&inst);
    SHAPES
        .into_iter()
        .map(|(mode, rows)| {
            let text = metrale_circuit::render_plan(&inst, &loaded, &avail, mode, rows, &fams)
                .unwrap_or_else(|e| panic!("LoRA {} n={rows}: {e}", mode.name()));
            (format!("lora/{}", inst.plan_file(mode, rows)), text)
        })
        .collect()
}
