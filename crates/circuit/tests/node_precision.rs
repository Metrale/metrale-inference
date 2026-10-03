// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: Node pipelines on the real circuits, manifests and rules.
//!
//! - The two Qwen3.6-35B-A3B checkpoints: the NVFP4 one (W4A16 experts and shared expert) and
//!   the FP8 one (the served recipe's W8A16 experts with the FP32 SiLU product, and the W8A8
//!   prefill rule above the grouped cap) require the pipelines asserted step by step for a MoE
//!   layer.
//! - A kernel whose declared accumulator is not the required one is refused, and a fused edge
//!   whose hand-off the rule stops stating is refused (both mutations of the real files).
//!
//! Owner: metrale-circuit tests.
//! Invariants: none beyond the types.

mod checkpoint_fixtures;
mod common;

use std::collections::BTreeMap;

use metrale_circuit::hardware::model_checkpoint::derive_policy;
use metrale_circuit::pipeline::{Need, PipelineError, required};
use metrale_circuit::{LoadError, Mode};

fn lines(text: &str, prefix: &str) -> Vec<String> {
    text.lines()
        .filter(|l| l.starts_with(prefix))
        .map(str::to_string)
        .collect()
}

/// 2026-10-02: The base requirement of every node of `layer`'s MoE block of a checkpoint, under
/// the policy derived from it, as `<local>: <pipeline>`.
fn checkpoint_moe(name: &str, layer: usize) -> Vec<String> {
    let r = checkpoint_fixtures::ok(name);
    let policy = derive_policy(&r.circuit, r.kv_cache).unwrap();
    let need = Need::base(&r.circuit, &policy.settings, 1);
    let prefix = format!("l{layer}.moe_ffn.");
    r.circuit
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| n.id.starts_with(&prefix))
        .map(|(i, n)| format!("{}: {}", n.local, required(&need, i).unwrap()))
        .collect()
}

const W4A16: &str =
    "act bf16 | weight nvfp4/g16->bf16 | mma bf16*bf16 | accumulate f32 | scale f32";
const BF16: &str = "act bf16 | weight bf16->bf16 | mma bf16*bf16 | accumulate f32 | scale none";

#[test]
fn the_nvfp4_35b_moe_layer_needs_w4a16_experts_and_a_bf16_router() {
    let want = [
        "post_norm: bf16 -> [compute f32] -> bf16".to_string(),
        format!("router: bf16 -> [{BF16}] -> bf16"),
        "top_k: bf16 -> [score f32] -> f32,i32".into(),
        format!("experts_gate_up: bf16,i32 -> [gather bf16 | {W4A16}] -> bf16"),
        "experts_act: bf16 -> [compute f32] -> bf16".into(),
        format!("experts_down: bf16,i32 -> [{W4A16}] -> bf16"),
        format!("shared_gate_up: bf16 -> [{W4A16}] -> bf16"),
        "shared_act: bf16 -> [compute f32] -> bf16".into(),
        format!("shared_down: bf16 -> [{W4A16}] -> bf16"),
        format!("shared_gate: bf16 -> [{BF16}] -> bf16"),
        "blend: bf16,f32,bf16,bf16 -> [scatter f32 | combine f32] -> bf16".into(),
        "add: bf16,bf16 -> [compute f32] -> bf16".into(),
    ];
    assert_eq!(checkpoint_moe("nvidia--Qwen3.6-35B-A3B-NVFP4", 3), want);
}

fn moe_instance() -> metrale_circuit::Instance {
    common::instances()
        .into_iter()
        .find(|i| i.checkpoint == "Qwen/Qwen3.6-35B-A3B-FP8")
        .expect("the MoE instance")
}

fn render(
    inst: &metrale_circuit::Instance,
    texts: &common::Texts,
    fams: &str,
    mode: Mode,
    rows: u64,
) -> Result<String, LoadError> {
    let loaded = common::load_texts(inst, texts)?;
    let avail = common::available(inst, &loaded.rules);
    let fams = metrale_circuit::venn::parse_families(fams).expect("families");
    metrale_circuit::render_plan(inst, &loaded, &avail, mode, rows, &fams)
}

fn manifest() -> String {
    common::read("kernels/gb10/common/KERNEL_FAMILIES.toml")
}

// 2026-10-02: The served FP8 recipe: W8A16 experts and shared expert; the SiLU product handed to
// the down projections in FP32 (the precision table widens `eact`/`sact`); the router and the
// gate BF16, the gate logit held FP32 into the blend's sigmoid as the rule states.
#[test]
fn the_fp8_35b_recipe_runs_w8a16_experts_with_an_fp32_silu_product() {
    let inst = moe_instance();
    let texts = common::Texts::of(&inst);
    let text = render(&inst, &texts, &manifest(), Mode::Decode, 1).unwrap();
    let w8 = "weight fp8/block128x128";
    let want = [
        format!("  router: bf16 -> [{BF16}] -> bf16"),
        "  top_k: bf16 -> [score f32] -> f32,i32".into(),
        format!(
            "  experts_gate_up: bf16,i32 -> [gather bf16 | act bf16 | {w8}->bf16 | mma bf16*bf16 | accumulate f32 | scale f32] -> bf16"
        ),
        format!(
            "  shared_gate_up: bf16 -> [act bf16 | {w8}->bf16 | mma bf16*bf16 | accumulate f32 | scale f32] -> bf16"
        ),
        "  experts_act: bf16 -> [compute f32] -> f32".into(),
        format!(
            "  experts_down: f32,i32 -> [act f32 | {w8}->f32 | mma f32*f32 | accumulate f32 | scale f32] -> bf16"
        ),
        "  shared_act: bf16 -> [compute f32] -> f32".into(),
        format!(
            "  shared_down: f32 -> [act f32 | {w8}->f32 | mma f32*f32 | accumulate f32 | scale f32] -> bf16"
        ),
        format!("  shared_gate: bf16 -> [{BF16}] -> f32 (rule)"),
        "  blend: bf16,f32,bf16,f32 (rule) -> [scatter f32 | combine f32] -> bf16".into(),
    ];
    let body = text.split("== layer 1 ").next().expect("layer 0");
    let got: Vec<String> = body
        .split("== layer 0 ")
        .nth(1)
        .expect("layer 0")
        .lines()
        .filter(|l| {
            l.starts_with("  ")
                && ["router", "top_k", "experts_", "shared_", "blend"]
                    .iter()
                    .any(|p| l[2..].starts_with(p))
        })
        .map(str::to_string)
        .collect();
    assert_eq!(got, want);
    // 2026-10-02: Above the grouped cap the rule states its W8A8 departures, marked `(rule)`, in
    // each of the 40 layers of the primary plan and of the fragmented-slots route's arm.
    let wide = render(&inst, &texts, &manifest(), Mode::MultiSeq, 128).unwrap();
    for want in [
        "  experts_gate_up: bf16,i32 -> [gather fp8/g128 (rule) | act fp8/g128 (rule) | weight fp8/block128x128->e4m3 | mma e4m3*e4m3 | accumulate f32 | scale f32] -> bf16",
        "  experts_act: bf16 -> [compute bf16 (rule)] -> fp8/g128 (rule)",
        "  experts_down: fp8/g128 (rule),i32 -> [act fp8/g128 | weight fp8/block128x128->e4m3 | mma e4m3*e4m3 | accumulate f32 | scale f32] -> bf16",
        "  blend: bf16,f32,bf16,f32 (rule) -> [scatter bf16 (rule) | combine f32] -> bf16",
    ] {
        assert_eq!(lines(&wide, want).len(), 80, "{want}");
    }
}

fn mismatches(e: LoadError) -> Vec<metrale_circuit::pipeline::Mismatch> {
    match e {
        LoadError::Pipeline(PipelineError::Mismatch(m)) => m,
        other => panic!("not a pipeline mismatch: {other}"),
    }
}

// 2026-10-02: Mutation of the real manifest: the one-row FP8 expert kernel declaring a BF16
// accumulator no longer runs the experts the plan requires; the plan is refused, naming every
// such node of the 40 layers and the step.
#[test]
fn a_kernel_declaring_another_accumulator_is_refused() {
    let inst = moe_instance();
    let texts = common::Texts::of(&inst);
    let line = "pipeline.expert_gate_up = { in = [\"bf16\", \"i32\"], gather = \"bf16\", act = \"bf16\", weight = \"fp8/block128x128->bf16\", mma = \"bf16*bf16\", accumulate = \"f32\"";
    let text = manifest();
    let at = text.find("id = \"moe_fp8_1row\"").expect("family");
    let off = text[at..].find(line).expect("its gate_up") + at;
    let mutated = format!(
        "{}{}{}",
        &text[..off],
        line.replace("accumulate = \"f32\"", "accumulate = \"bf16\""),
        &text[off + line.len()..]
    );
    let m = mismatches(render(&inst, &texts, &mutated, Mode::Decode, 1).unwrap_err());
    assert_eq!(m.len(), 40, "{m:?}");
    assert!(
        m.iter()
            .all(|m| m.node.ends_with(".moe_ffn.experts_gate_up")
                && m.rule == "moe_gate_up_shared_fp8"
                && m.family == "moe_fp8_1row"
                && m.diffs == ["accumulate: required f32, declared bf16"])
    );
}

// 2026-10-02: Mutation of the real rules: the blend's rule stops stating that the gate logit
// stays FP32 inside the kernel. The elided `sg` edge would change format silently, so both
// sides of it are refused.
#[test]
fn an_unstated_hand_off_at_a_fused_edge_is_refused() {
    let inst = moe_instance();
    let mut texts = common::Texts::of(&inst);
    let stated = "pattern = [{ op = \"linear\", role = \"shared_gate\", holds = \"f32\" }, { op = \"blend\" }]\nkernels = [{ module = \"moe_expert_gemv\"";
    assert_eq!(texts.rules.matches(stated).count(), 1);
    texts.rules = texts
        .rules
        .replace(stated, &stated.replace(", holds = \"f32\"", ""));
    let m = mismatches(render(&inst, &texts, &manifest(), Mode::Decode, 1).unwrap_err());
    let by_node: BTreeMap<&str, &Vec<String>> =
        m.iter().map(|m| (m.node.as_str(), &m.diffs)).collect();
    assert_eq!(
        by_node["l0.moe_ffn.shared_gate"],
        &vec!["out[0]: required bf16, declared f32".to_string()]
    );
    assert_eq!(
        by_node["l0.moe_ffn.blend"],
        &vec!["in[3]: required bf16, declared f32".to_string()]
    );
}
