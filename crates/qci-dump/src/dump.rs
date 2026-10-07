// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-06: One text receipt per case and topology.
//!
//! Speed and energy stay withheld until bit parity is recorded. Reference and
//! native rows are always both present. Nothing in this text claims a serve.

use std::fmt::Write as _;

use crate::recipe::{self, Recipe};

/// 2026-10-06: Everything the dump renderer is allowed to see. No path, no clock.
#[derive(Debug, Clone, Copy)]
pub struct DumpInput<'a> {
    pub case_id: &'a str,
    pub hardware: &'a str,
    pub weights_present: bool,
    pub weights_bytes: Option<u64>,
    pub vision_present: bool,
    pub vision_bytes: Option<u64>,
    pub runtime_commit: Option<&'a str>,
}

pub fn render(input: &DumpInput<'_>) -> String {
    let Some(recipe) = recipe::for_case(input.case_id) else {
        return unknown(input);
    };
    known(&recipe, input)
}

fn known(recipe: &Recipe, input: &DumpInput<'_>) -> String {
    let mut blockers: Vec<&str> = Vec::new();
    let role = if input.hardware == recipe.supplemental_hardware {
        "supplemental"
    } else if input.hardware == recipe.target_hardware {
        "target"
    } else {
        blockers.push("TOPOLOGY_NOT_IN_ISSUE");
        "outside-campaign"
    };
    if input.hardware == recipe.target_hardware {
        blockers.push("TARGET_NOT_MEASURED");
    }
    let weights = if input.weights_present {
        "present"
    } else {
        blockers.push("WEIGHTS_ABSENT");
        "absent"
    };
    let vision = if recipe.vision_artifact.is_empty() {
        "not-part-of-this-pin"
    } else if input.vision_present {
        "present"
    } else {
        blockers.push("VISION_WEIGHTS_ABSENT");
        "absent"
    };
    let commit = match input.runtime_commit {
        Some(sha) if !sha.is_empty() => sha,
        _ => {
            blockers.push("ENGINE_NOT_BUILT");
            "BLOCKER ENGINE_NOT_BUILT"
        }
    };
    if recipe.license_status.starts_with("BLOCKER") {
        blockers.push("LICENSE_GOVERNING_TEXT_UNDECIDED");
    }
    blockers.extend([
        "PARITY_UNMEASURED",
        "VRAM_UNMEASURED",
        "COHERENCE_UNMEASURED",
        "CONTRACTS_UNFROZEN",
    ]);
    for name in recipe
        .extra_blockers
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        blockers.push(name);
    }
    blockers.sort_unstable();
    blockers.dedup();

    let mut out = String::new();
    line(&mut out, "qci-dump", "1");
    line(&mut out, "case", &recipe.case_id);
    line(&mut out, "issue", &recipe.issue_url);
    line(&mut out, "priority", &recipe.priority);
    line(
        &mut out,
        "blocks_four_model_p0_exit",
        &recipe.blocks_p0_exit,
    );
    line(&mut out, "p0_exit_member", &recipe.p0_exit_member);
    line(&mut out, "live_qci_acceptance", "not-claimed");
    line(&mut out, "support_label", "withheld");
    line(&mut out, "recipe", &recipe.id);
    line(&mut out, "pipeline", &recipe.pipeline);
    line(
        &mut out,
        "pin.original_repository",
        &recipe.original_repository,
    );
    line(&mut out, "pin.repository", &recipe.repository);
    line(&mut out, "pin.revision", &recipe.revision);
    line(&mut out, "pin.artifact", &recipe.artifact);
    line(&mut out, "pin.artifact_bytes", &recipe.artifact_bytes);
    if !recipe.vision_artifact.is_empty() {
        line(&mut out, "pin.vision_artifact", &recipe.vision_artifact);
        line(
            &mut out,
            "pin.vision_artifact_bytes",
            &recipe.vision_artifact_bytes,
        );
    }
    line(&mut out, "pin.quantization", &recipe.quantization);
    line(&mut out, "pin.license_tag", &recipe.license_tag);
    line(&mut out, "pin.license_link", &recipe.license_link);
    line(&mut out, "pin.license_status", &recipe.license_status);
    line(&mut out, "pin.concurrency", &recipe.concurrency);
    line(&mut out, "pin.context", &recipe.context);
    line(&mut out, "pin.mtp", &recipe.mtp);
    if !recipe.distinct_from.is_empty() {
        line(&mut out, "pin.distinct_from", &recipe.distinct_from);
    }
    if !recipe.tokenizer_bos.is_empty() {
        line(&mut out, "pin.tokenizer_bos", &recipe.tokenizer_bos);
        line(&mut out, "pin.tokenizer_eos", &recipe.tokenizer_eos);
        line(
            &mut out,
            "pin.tokenizer_json_sha256",
            &recipe.tokenizer_json_sha256,
        );
        line(
            &mut out,
            "pin.chat_template_blob",
            &recipe.chat_template_blob,
        );
        line(&mut out, "pin.template_status", &recipe.template_status);
    }
    line(&mut out, "pin.runtime_commit", commit);
    line(&mut out, "topology", input.hardware);
    line(&mut out, "topology.role", role);
    line(
        &mut out,
        "topology.limitation",
        &limitation(recipe, input.hardware, role),
    );
    line(&mut out, "weights", weights);
    if let Some(bytes) = input.weights_bytes.filter(|_| input.weights_present) {
        line(&mut out, "weights_file_bytes", &bytes.to_string());
    }
    line(&mut out, "vision_weights", vision);
    if let Some(bytes) = input.vision_bytes.filter(|_| input.vision_present) {
        line(&mut out, "vision_file_bytes", &bytes.to_string());
    }

    row(&mut out, "reference", &recipe.reference_runtime, recipe);
    row(&mut out, "native", "metrale", recipe);
    line(&mut out, "parity", "BLOCKER PARITY_UNMEASURED");
    line(&mut out, "tok_s", "withheld until bit parity");
    line(&mut out, "j_per_tok", "withheld until bit parity");
    for contract in ["41", "42", "46"] {
        let state = if recipe
            .required_contracts
            .split(',')
            .any(|c| c.trim() == contract)
        {
            "BLOCKER UNFROZEN"
        } else {
            "not-a-dependency"
        };
        line(&mut out, &format!("contract.{contract}"), state);
    }
    line(&mut out, "blockers", &blockers.join(","));
    out
}

fn row(out: &mut String, which: &str, runtime: &str, recipe: &Recipe) {
    line(out, &format!("row.{which}.runtime"), runtime);
    line(
        out,
        &format!("row.{which}.memory"),
        "BLOCKER VRAM_UNMEASURED",
    );
    line(
        out,
        &format!("row.{which}.coherence"),
        "BLOCKER COHERENCE_UNMEASURED",
    );
    line(
        out,
        &format!("row.{which}.parameters"),
        &format!(
            "concurrency={} context={} precision={} mtp={}",
            recipe.concurrency, recipe.context, recipe.quantization, recipe.mtp
        ),
    );
    let limits = if which == "native" {
        recipe.native_limitations.as_str()
    } else {
        "reference runtime was not executed on this dump; disk size is not VRAM"
    };
    line(out, &format!("row.{which}.limitations"), limits);
    line(
        out,
        &format!("row.{which}.tok_s"),
        "withheld until bit parity",
    );
    line(
        out,
        &format!("row.{which}.j_per_tok"),
        "withheld until bit parity",
    );
}

fn limitation(recipe: &Recipe, hardware: &str, role: &str) -> String {
    match role {
        "supplemental" => format!(
            "{hardware} results supplement {} target-hardware acceptance and are not that acceptance",
            recipe.target_hardware
        ),
        "target" => format!("{hardware} is the issue target and has no measurement in this dump"),
        _ => format!(
            "{hardware} is outside the issue pair {} / {}",
            recipe.supplemental_hardware, recipe.target_hardware
        ),
    }
}

fn unknown(input: &DumpInput<'_>) -> String {
    let mut out = String::new();
    line(&mut out, "qci-dump", "1");
    line(&mut out, "case", input.case_id);
    line(&mut out, "support_label", "withheld");
    line(&mut out, "live_qci_acceptance", "not-claimed");
    line(&mut out, "topology", input.hardware);
    line(&mut out, "parity", "BLOCKER PARITY_UNMEASURED");
    line(&mut out, "tok_s", "withheld until bit parity");
    line(&mut out, "j_per_tok", "withheld until bit parity");
    line(&mut out, "row.reference.memory", "BLOCKER VRAM_UNMEASURED");
    line(
        &mut out,
        "row.reference.coherence",
        "BLOCKER COHERENCE_UNMEASURED",
    );
    line(&mut out, "row.reference.parameters", "unpinned");
    line(
        &mut out,
        "row.reference.limitations",
        "BLOCKER UNKNOWN_CASE",
    );
    line(&mut out, "row.native.memory", "BLOCKER VRAM_UNMEASURED");
    line(
        &mut out,
        "row.native.coherence",
        "BLOCKER COHERENCE_UNMEASURED",
    );
    line(&mut out, "row.native.parameters", "unpinned");
    line(&mut out, "row.native.limitations", "BLOCKER UNKNOWN_CASE");
    line(
        &mut out,
        "blockers",
        "UNKNOWN_CASE,PARITY_UNMEASURED,VRAM_UNMEASURED,COHERENCE_UNMEASURED",
    );
    out
}

fn line(out: &mut String, key: &str, value: &str) {
    let _ = writeln!(out, "{key}: {value}");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input<'a>(case_id: &'a str, hardware: &'a str) -> DumpInput<'a> {
        DumpInput {
            case_id,
            hardware,
            weights_present: false,
            weights_bytes: None,
            vision_present: false,
            vision_bytes: None,
            runtime_commit: None,
        }
    }

    fn assert_pins(text: &str, repo: &str, revision: &str, artifact: &str) {
        assert!(text.contains(&format!("pin.repository: {repo}")), "{text}");
        assert!(text.contains(&format!("pin.revision: {revision}")));
        assert!(text.contains(&format!("pin.artifact: {artifact}")));
        assert!(text.contains("row.reference.runtime:"));
        assert!(text.contains("row.native.runtime: metrale"));
        assert!(text.contains("row.reference.memory: BLOCKER VRAM_UNMEASURED"));
        assert!(text.contains("row.native.coherence: BLOCKER COHERENCE_UNMEASURED"));
        assert!(text.contains("row.reference.parameters:"));
        assert!(text.contains("row.native.limitations:"));
        assert!(text.contains("support_label: withheld"));
        assert!(!text.to_ascii_lowercase().contains("supported"));
        assert!(text.contains("tok_s: withheld until bit parity"));
        assert!(text.contains("j_per_tok: withheld until bit parity"));
        assert!(text.contains("parity: BLOCKER PARITY_UNMEASURED"));
        assert!(text.contains("COHERENCE_UNMEASURED"));
    }

    #[test]
    fn caller_bytes_and_commit_do_not_clear_unmeasured_blockers() {
        let filled = DumpInput {
            case_id: "gemma-4-26b-a4b",
            hardware: "rtx-3090",
            weights_present: true,
            weights_bytes: Some(16947541728),
            vision_present: true,
            vision_bytes: Some(1193058784),
            runtime_commit: Some("caller-supplied-prose-sample"),
        };
        let text = render(&filled);
        assert!(text.contains("weights_file_bytes: 16947541728"));
        assert!(text.contains("vision_file_bytes: 1193058784"));
        assert!(text.contains("support_label: withheld"));
        assert!(text.contains("PARITY_UNMEASURED"));
        assert!(text.contains("VRAM_UNMEASURED"));
        assert!(text.contains("COHERENCE_UNMEASURED"));
        assert!(text.contains("tok_s: withheld until bit parity"));
        assert!(text.contains("j_per_tok: withheld until bit parity"));
        assert!(text.contains("pin.runtime_commit: caller-supplied-prose-sample"));
        let nemotron = DumpInput {
            case_id: "nemotron-3.5-lightning-30b-a3b",
            hardware: "rtx-3090",
            weights_present: true,
            weights_bytes: Some(18898091584),
            vision_present: true,
            vision_bytes: Some(1),
            runtime_commit: Some("18080 MiB"),
        };
        let other = render(&nemotron);
        assert!(other.contains("weights_file_bytes: 18898091584"));
        assert!(other.contains("PARITY_UNMEASURED"));
        assert!(other.contains("VRAM_UNMEASURED"));
        assert!(other.contains("COHERENCE_UNMEASURED"));
        assert!(other.contains("pin.runtime_commit: 18080 MiB"));
    }

    #[test]
    fn gemma_dump_pins_the_issue_artifact_and_separates_rows() {
        let text = render(&input("gemma-4-26b-a4b", "rtx-3090"));
        assert_eq!(text, render(&input("gemma-4-26b-a4b", "rtx-3090")));
        assert_pins(
            &text,
            "unsloth/gemma-4-26B-A4B-it-GGUF",
            "c099eb48e663fd284577b04978a94ffccb261841",
            "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
        );
        assert!(text.contains("pin.artifact_bytes: 16947541728"));
        assert!(text.contains("pin.vision_artifact: mmproj-F16.gguf"));
        assert!(text.contains("pin.vision_artifact_bytes: 1193058784"));
        assert!(text.contains("topology.role: supplemental"));
        assert!(
            text.contains("supplement rtx-3090") || text.contains("rtx-3090 results supplement")
        );
        assert!(text.contains("weights: absent"));
        assert!(text.contains("WEIGHTS_ABSENT"));
        assert!(text.contains("contract.41: BLOCKER UNFROZEN"));
        assert!(text.contains("contract.42: BLOCKER UNFROZEN"));
        assert!(text.contains("live_qci_acceptance: not-claimed"));
    }

    #[test]
    fn nemotron_dump_pins_tokenizer_and_stays_off_the_p0_exit() {
        let text = render(&input("nemotron-3.5-lightning-30b-a3b", "rtx-3090"));
        assert_eq!(
            text,
            render(&input("nemotron-3.5-lightning-30b-a3b", "rtx-3090"))
        );
        assert_pins(
            &text,
            "ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF",
            "8a08a1c81dadcc75d35dbb96016cfd344b632e67",
            "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf",
        );
        assert!(text.contains("pin.artifact_bytes: 18898091584"));
        assert!(text.contains("pin.tokenizer_bos: <s>"));
        assert!(text.contains("pin.tokenizer_eos: <|im_end|>"));
        assert!(text.contains(
            "pin.tokenizer_json_sha256: 623c34567aebb18582765289fbe23d901c62704d6518d71866e0e58db892b5b7"
        ));
        assert!(text.contains("pin.mtp: off"));
        assert!(text.contains("pin.distinct_from: NVIDIA-Nemotron-3-Nano"));
        assert!(text.contains("blocks_four_model_p0_exit: no"));
        assert!(text.contains("priority: P1"));
        assert!(text.contains("contract.46: BLOCKER UNFROZEN"));
        assert!(text.contains("nvfp4"));
    }

    #[test]
    fn other_hardware_and_missing_paths_stay_deterministic() {
        let strix = render(&input("gemma-4-26b-a4b", "strix-halo"));
        assert!(strix.contains("topology.role: target"));
        assert!(strix.contains("TARGET_NOT_MEASURED"));
        assert!(strix.contains("pin.revision: c099eb48e663fd284577b04978a94ffccb261841"));
        let outside = render(&input("gemma-4-26b-a4b", "h100-sxm"));
        assert!(outside.contains("TOPOLOGY_NOT_IN_ISSUE"));
        assert!(outside.contains("BLOCKER"));
        let mut a = input("nemotron-3.5-lightning-30b-a3b", "rtx-3090");
        let mut b = a;
        a.weights_present = false;
        b.weights_present = false;
        a.weights_bytes = None;
        b.weights_bytes = None;
        assert_eq!(render(&a), render(&b));
        let unknown = render(&input("not-a-campaign", "rtx-3090"));
        assert!(unknown.contains("UNKNOWN_CASE"));
        assert!(!unknown.to_ascii_lowercase().contains("supported"));
    }
}
